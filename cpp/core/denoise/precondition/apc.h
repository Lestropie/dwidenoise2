/* Required Notice: Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;
 * Required Notice: The Florey Institute of Neuroscience and Mental Health.
 *
 * Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     https://polyformproject.org/licenses/noncommercial/1.0.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied.
 * See the License of the specific language
 * governing permissions and limitations under the License.
 */

#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "image.h"
#include "types.h"

namespace MR::Denoise::Precondition {

// Noise-map-driven background-phase re-estimation ("Adaptive Phase Correction", APC),
//   after Pizzolato et al., "Adaptive phase correction of diffusion-weighted images",
//   NeuroImage 206 (2020) 116274.
//
// Context within dwidenoise2 (see dwidenoise2_dephase.md).
//   With -demodulate apc, the preconditioner re-estimates the background phase from the
//   *empirical* complex data at every noise-estimation iteration, with the regularisation
//   strength driven by the current noise level map. This class implements that estimation.
//   The first iteration has no noise map yet: rather than falling back to a fixed Hann-window
//   phase, the estimator self-calibrates a single global noise level from the data (robust
//   MAD of finite differences) and applies uniform spatial weighting, so the same APC solver
//   is used throughout. The phase is estimated per 2-D in-slice plane of each volume
//   (diffusion MRI is a stack of 2-D acquisitions; phase coherence between slices is not
//   assumed).
//
// What is implemented here (and why this specific method).
//   For each slice, a regularised complex image I is obtained as the minimiser of the
//   spatially-varying, noise-weighted total-variation (ROF) energy
//
//       I* = argmin_I  TV(I)  +  (lambda/2) * sum_x w(x) || I(x) - I0(x) ||^2 ,
//
//   where I0 is the empirical complex slice (native intensity units), lambda is Pizzolato's
//   fidelity weight (large lambda => high fidelity => less smoothing), and the per-voxel
//   fidelity weight w(x) = sigmabar^2 / sigma^2(x) (their eq. 13) makes the data term
//   noise-adaptive: voxels with a larger local noise level are trusted less and hence
//   smoothed more. TV(I) is the *vectorial* (coupled) total variation of the two-channel
//   (real, imaginary) image,
//
//       TV(I) = sum_x sqrt( |grad Re I(x)|^2 + |grad Im I(x)|^2 ) ,
//
//   so that both channels are regularised along a common geometry (Sapiro & Ringach 1996;
//   Tschumperle & Deriche 2007) exactly as Pizzolato specify. The estimated phase returned
//   is the unit-magnitude argument of I*, phase(x) = I*(x) / |I*(x)|, matching the existing
//   Image<cfloat> phase-map interface consumed by the preconditioner.
//
//   Solver: a fixed-iteration-budget first-order primal-dual scheme (Chambolle & Pock 2011)
//   with the per-voxel fidelity weight consumed exactly by the (pointwise, closed-form)
//   proximal step of the quadratic data term. This is a deliberate, documented deviation
//   from Pizzolato's own solver (an explicit finite-difference oriented-Laplacian PDE driven
//   to steady state): here the phase is only a means to reduce the PCA rank of the
//   demodulated Casorati matrix, not an end in itself, so phase MSE is not the objective, and
//   the estimate is re-derived every outer iteration. A bounded per-slice cost with graceful
//   (approximate) convergence therefore suits the loop far better than iterating any solver
//   to full convergence. See dwidenoise2_dephase.md sections 2.2 and 2.5.
//
//   Cost across the iteration loop: the phase changes little once the noise map stabilises, so
//   only the first pass is solved from a cold start -- and, because a background phase is smooth
//   (low spatial bandwidth), that first pass is solved on a 2x-downsampled grid and the result
//   upsampled (~1/4 the cost, near-lossless for a smooth field) whenever a later pass will refine
//   it at native resolution (i.e. a multi-iteration schedule). A single-iteration schedule solves
//   its sole, authoritative pass at native resolution instead. Every subsequent pass runs at
//   native resolution but is warm-started from the previous estimate (seeded with |f| * the
//   incoming phase), so a much smaller iteration budget (the Params ..._warm counts) re-settles
//   it. The resolution and warm-start decisions live in operator() / the per-plane functor;
//   solve_plane itself consumes only a warm_start flag (cold-start seed vs warm-start seed, and
//   full vs reduced budget). This is the dominant lever on APC's contribution to run time.
//
//   Regularisation strength lambda: the discrepancy criterion (Morozov), realised as
//   Chambolle's (2004) fixed-point iteration on lambda (their eq. 6) seeded from eq. 7,
//   targeting a weighted residual energy of 2 * N * sigmabar^2 (their eq. 11; the factor 2
//   is the two real+imaginary channels of complex data). This is cheap and, for a *smooth*
//   background phase, its slight bias toward over-smoothing is the safe direction (it will
//   not fit tissue contrast or noise into the phase).
//
// Noise-map convention (verified; see dwidenoise2_dephase.md section 3).
//   dwidenoise2's noise level map holds, per voxel, exactly Pizzolato's per-component
//   (real == imaginary) noise standard deviation sigma(x,y), in the native complex intensity
//   units in which APC operates. The mapping is the identity: no sqrt(2) or 1/2 rescaling.
//   APC must run on the *native* complex data with spatially-varying weighting; it must NOT
//   run on variance-stabilised data (the phase is scale-invariant, so nothing is gained,
//   while dividing by sigma(x,y) would inject spurious sigma-gradients into the TV term).
//
// ---------------------------------------------------------------------------------------
// Other prospective techniques for solving the background phase given a noise-map estimate.
//   Only the primal-dual weighted-TV method above is implemented (by design). The following
//   are recorded for the record: any of them could be substituted behind solve_plane(), and
//   all consume a non-stationary noise map through the same weighted-L2 fidelity term (the
//   crucial requirement; do not whiten the data itself). Cost has two independent axes --
//   the regulariser and the lambda-selection.
//
//   Regulariser alternatives (Pizzolato realise all three through one oriented-Laplacian
//   PDE, differing only in the diffusivities alpha, beta):
//     - Linear Laplacian / heat kernel (alpha = beta = 1): edge-*blind*, but at fixed lambda
//       it is a single linear, symmetric-positive-definite system in the weighted-L2 sense --
//       i.e. a weighted-Laplacian solve amenable to a warm-started Conjugate Gradient (Eigen
//       SparseMatrix + ConjugateGradient) or a few Jacobi/Gauss-Seidel sweeps. Cheapest;
//       sacrifices phase accuracy at tissue/magnitude boundaries. Suitable if rank benchmarks
//       show edge preservation is immaterial for a *background* phase.
//     - Mean-curvature flow (alpha = 1, beta = 0): nonlinear, intermediate.
//     - Total variation (alpha = 1/|grad I|, beta = 0): edge-preserving, best MSE, most
//       expensive -- this is what the primal-dual solver above approximates.
//
//   Solver alternatives for weighted / spatially-varying TV with a known noise map:
//     - Pizzolato's exact scheme: explicit finite-difference oriented-Laplacian anisotropic
//       diffusion driven to steady state (max ~200 sweeps, |grad| regularised as
//       sqrt(|grad|^2 + eps^2)), lambda by the discrepancy criterion, optionally refined by
//       spatially-varying Monte-Carlo SURE (svSURE). Most faithful; slowest -- the paper
//       reports ~98 min for one multi-shell dataset and calls it offline-only, so svSURE per
//       (outer iteration x volume x slice) is not viable here. Keep only as a validation
//       reference; keep svSURE off.
//     - Split Bregman / ADMM: very efficient for TV; spatially-adaptive weighted-TV variants
//       are established (e.g. Dodangeh et al. 2018) and can update the regularisation
//       parameter jointly with the image.
//     - Half-quadratic / IRLS ("lagged diffusivity", Vogel & Oman): reduces Huber- or
//       Charbonnier-TV to a few reweighted SPD solves (Eigen CG); robust; warm-starts well.
//     - O(N) edge-preserving smoothers -- guided filter (He et al. 2013), weighted least
//       squares, domain transform: non-variational, linear-time, no gradient reversal;
//       could smooth the complex image extremely cheaply. They do NOT natively consume a
//       discrepancy criterion, so sigma(x,y) would map to a spatially-varying radius/epsilon
//       rather than a principled residual match -- the fastest-of-all approximation, at the
//       cost of a looser noise-adaptivity guarantee.
//
//   lambda-selection alternatives: the discrepancy criterion (implemented; short, cheap
//   outer loop) vs spatially-varying Monte-Carlo SURE (svSURE; MSE-optimal but the ~98-min
//   bottleneck, deferred).
// ---------------------------------------------------------------------------------------
//
// Data-driven early termination (2026-07-02) and the DWIDENOISE2_APC_PROFILE experiment.
//   Beyond the reduced ..._warm budgets, run_cp() now stops a sweep sequence early once the
//   relative per-sweep change in the primal image falls below Params::cp_tol (a genuinely
//   data-driven cut -- it only shortens a call once *that specific plane, at that specific
//   outer iteration* has actually stopped moving; the ..._warm budgets are a fixed, uniform
//   guess). The comparison is folded into the primal update's existing per-voxel pass, so it
//   costs no extra sweep over the plane. cp_tol defaults tight (1e-6): a safety net against
//   waste, not a substitute for the ..._warm budgets, since the right threshold cannot be
//   chosen from first principles -- it depends on how quickly real (not synthetic) DWI planes
//   actually settle once warm-started, which is an empirical question.
//
//   To answer that question, build with -DDWIDENOISE2_APC_PROFILE to compile in a per-plane
//   DEBUG() line (run under -debug) reporting, for every solve_plane() call: plane size,
//   cold/warm, spatially-varying vs self-calibrated, ndomain fraction, sigmabar, lambda
//   iterations used vs budgeted, total Chambolle-Pock sweeps actually consumed vs the full
//   budget (n_lambda*n_cp + n_polish), and the converged lambda. This is compile-time gated
//   (rather than always emitted under -debug) because computing and formatting it triggers
//   string work on every one of the many thousands of solve_plane() calls in a full run, which
//   would be a needless cost in ordinary -debug diagnostic use; opting in is a deliberate,
//   separate build for this experiment. The sweep- and lambda-iteration counters that feed the
//   line exist *only* to report it -- run_cp()'s early-exit decision itself needs no count, only
//   the per-sweep relative-change comparison -- so those counters (and run_cp()'s ssize_t return)
//   are themselves compiled out under the non-profile build, not merely left unprinted. A
//   non-profile build's timing therefore isolates the effect of cp_tol's early exit from the
//   cost of instrumenting it, which is the point of comparing the two builds' run times.
//
//   Suggested experiment: build a DWIDENOISE2_APC_PROFILE binary and run it (dwidenoise2 or
//   dwi2noise, default -demodulate apc, a representative multi-iteration schedule) over a
//   modest real DWI dataset -- large enough to be representative of real anatomy/coil geometry
//   (not the synthetic data used so far), small enough that the per-plane DEBUG log is
//   tractable to inspect (e.g. a single low-resolution shell, or a cropped FOV). From the log:
//     - Histogram cp_sweeps actually used (post-cp_tol) against the ..._warm budget on warm
//       passes: a distribution clustered well under budget confirms head-room and motivates
//       lowering cp_iter_warm/polish_iter_warm directly (cheaper long-term than relying on
//       cp_tol's per-sweep check); a distribution hugging the budget means the current warm
//       budget is already close to necessary and cp_tol is doing the real work.
//     - Trend of cp_sweeps / lambda_iters across successive outer (noise-estimation)
//       iterations, per plane: Pizzolato's own noise-map update converges over a handful of
//       outer iterations, so later-iteration APC calls are expected to need *less* work than
//       the second call already assumes with a single reduced budget -- if the log shows a
//       further monotonic decline, a schedule-position-dependent budget (not just cold/warm)
//       is worth adding.
//     - Any planes where lambda_iters or cp_sweeps repeatedly hit their budget unconverged
//       (rel change still above tol) flag cp_tol/the ..._warm budgets as too aggressive for
//       that anatomy -- inspect before trusting a global reduction.
//   Validate any resulting parameter change against denoised rank / noise_out on that same
//   dataset (not phase MSE), consistent with the rest of this file's tuning guidance.
// ---------------------------------------------------------------------------------------
class AdaptivePhaseEstimator {
public:
  struct Params {
    // Chambolle-Pock primal-dual iterations run between successive lambda updates. The
    //   solver is warm-started across lambda updates, so this need not fully converge each
    //   time -- it interleaves image and lambda refinement (a standard, efficient scheme).
    ssize_t cp_iter = 25;
    // Discrepancy-criterion (Morozov) fixed-point updates of lambda (their eqs. 6-7, 11).
    ssize_t max_lambda_iter = 10;
    // Extra primal-dual iterations at the converged lambda, to settle the image before the
    //   phase (its argument) is extracted.
    ssize_t polish_iter = 25;
    // Reduced iteration budget for warm-started passes: every noise-estimation iteration after
    //   the first. The first pass solves from a cold start (the data) on a coarse grid; every
    //   later pass is warm-started at native resolution from the previous iteration's phase,
    //   which barely moves once the noise map has stabilised, so far fewer sweeps re-settle it.
    //   Because the solver runs a *fixed* budget (no convergence test drives early exit besides
    //   lambda_tol), this reduced budget -- not the warm start alone -- is what shortens the
    //   later passes; the warm start is what keeps that reduction safe. Tunable; validate on the
    //   denoised rank, not on phase MSE.
    ssize_t cp_iter_warm = 12;
    ssize_t max_lambda_iter_warm = 5;
    ssize_t polish_iter_warm = 12;
    // Relative-change tolerance for early exit of the lambda fixed-point iteration.
    double lambda_tol = 1e-2;
    // Relative-change tolerance for early exit of the *inner* Chambolle-Pock sweeps
    //   themselves (both the lambda-loop sweeps and the polish sweeps), checked every sweep
    //   at negligible extra cost (the comparison piggybacks on the primal update's existing
    //   per-voxel pass, so it does not add another sweep over the plane). Deliberately tight
    //   by default: this is a safety net against the *fixed* per-call budget doing
    //   inconsequential extra work once a warm-started plane has already re-settled -- not a
    //   replacement for the ..._warm budgets, which remain the primary lever validated against
    //   denoised rank. Set to 0 (or a negative value) to disable and always spend the full
    //   budget, matching pre-2026-07 behaviour exactly. See DWIDENOISE2_APC_PROFILE below for
    //   the instrumentation used to choose a data-driven value from empirical runs.
    double cp_tol = 1e-6;
    // The per-voxel fidelity weight w = sigmabar^2 / sigma^2 is clamped to
    //   [1/weight_clamp, weight_clamp] to bound its dynamic range (guards near-zero sigma in
    //   background voxels, where w would otherwise diverge).
    double weight_clamp = 100.0;
    // Threshold on the fraction of a slice's voxels that carry a usable local noise estimate.
    //   At or above it, the slice is calibrated from that (spatially-varying) map; below it
    //   -- including the first iteration, where no map exists at all -- the estimator instead
    //   self-calibrates a single global noise level from the data (uniform weighting). Only a
    //   fully degenerate slice (no estimable noise, e.g. constant/empty) is left untouched, so
    //   its incoming phase stands (edge-slice fallback, dwidenoise2_dephase.md section 7).
    double min_domain_fraction = 0.02;
  };

  AdaptivePhaseEstimator() = default;
  // Two overloads rather than a defaulted `Params` argument: a `= Params()` default argument
  //   for a member function declared inside the class is a complete-class context that would
  //   require Params's default member initialisers before the enclosing class is complete
  //   (rejected by GCC).
  explicit AdaptivePhaseEstimator(std::vector<size_t> inslice_axes) : inslice_axes(std::move(inslice_axes)) {}
  AdaptivePhaseEstimator(std::vector<size_t> inslice_axes, Params params)
      : inslice_axes(std::move(inslice_axes)), params(params) {}

  // Full-image entry point. Re-estimates the background phase for every 2-D in-slice plane
  //   of every volume, multi-threaded over the outer (slice-normal + volume + any serialised
  //   supra-volume) axes via MRtrix3 ThreadedLoop; each plane is an independent problem.
  //   in       : native complex data (empirical input; never variance-stabilised).
  //   sigma    : per-component noise standard-deviation map (the VST scale). May be at a
  //              different resolution to "in"; it is cubic-interpolated at each voxel, exactly
  //              as the preconditioner samples it elsewhere. Voxels with a non-finite or
  //              non-positive interpolated sigma are treated as outside the usable domain. May
  //              also be invalid (empty): on the first iteration no noise map exists yet, and
  //              each slice then self-calibrates a global noise level from the data.
  //   io_phase : unit-magnitude Image<cfloat>, same grid as "in". Overwritten in place with
  //              the new estimate (per-slice: only slices with a usable domain are modified,
  //              so the incoming phase acts as the fallback for the rest). On a warm-started
  //              call it is additionally read first as the solver's initial guess.
  //   warm_start : false on the first pass (cold: seed each slice at the data f, full Params
  //              budget); true on every later pass (seed at |f| * io_phase, reduced Params
  //              ..._warm budget).
  //   downsample : if true, solve each slice on a 2x-downsampled grid and upsample the phase to
  //              native (~1/4 the cost, near-lossless for a smooth phase). Set only on the first
  //              pass of a multi-iteration schedule, where a later pass refines at native
  //              resolution; a single-iteration schedule leaves it false so its sole pass is
  //              solved natively.
  // Only instantiated for complex T (the phase is ill-defined for real data); callers gate
  //   with `if constexpr (is_complex<T>::value)`.
  template <typename T>
  void operator()(Image<T> &in, Image<float> &sigma, Image<cfloat> &io_phase, bool warm_start, bool downsample) const;

  // Per-2-D-plane primitive (the reusable, unit-testable core): solves one slice's weighted
  //   vectorial-TV ROF problem by fixed-budget primal-dual with discrepancy-criterion lambda.
  //   fr, fi   : real / imaginary parts of the empirical complex plane (native units).
  //   sigma    : per-voxel noise standard deviation for the plane; entries <= 0 or non-finite
  //              mark voxels with no usable local estimate. If enough voxels carry one, the
  //              slice is calibrated from them with spatially-varying weighting; otherwise a
  //              single global noise level is self-estimated from the data (uniform weighting).
  //   phase_r, phase_i : on entry (warm_start only) the incoming unit-magnitude phase used to
  //              seed the solver; on exit written with the new unit-magnitude phase iff the
  //              return value is true. Left untouched (so the caller's incoming phase stands)
  //              only when the slice is fully degenerate -- no estimable noise level (false).
  //   warm_start : false -> seed the image at the data f with the full iteration budget (a cold
  //              solve); true -> seed the image at |f| * (phase_r, phase_i) -- the smooth phase
  //              is then already ~converged, only the fast magnitude smoothing remains -- and use
  //              the reduced Params ..._warm budget.
  // Arrays are column-major (Nx rows = first in-slice axis, Ny cols = second).
  bool solve_plane(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                   const Eigen::Ref<const Eigen::ArrayXXd> &fi,
                   const Eigen::Ref<const Eigen::ArrayXXd> &sigma,
                   Eigen::Ref<Eigen::ArrayXXd> phase_r,
                   Eigen::Ref<Eigen::ArrayXXd> phase_i,
                   bool warm_start) const;

  const std::vector<size_t> &axes() const { return inslice_axes; }

private:
  std::vector<size_t> inslice_axes;
  Params params;
};

} // namespace MR::Denoise::Precondition
