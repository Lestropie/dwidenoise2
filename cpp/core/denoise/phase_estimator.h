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

namespace MR::Denoise {

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
class PhaseEstimator {
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
    // Relative-change tolerance for early exit of the lambda fixed-point iteration.
    double lambda_tol = 1e-2;
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

  PhaseEstimator() = default;
  // Two overloads rather than a defaulted `Params` argument: a `= Params()` default argument
  //   for a member function declared inside the class is a complete-class context that would
  //   require Params's default member initialisers before the enclosing class is complete
  //   (rejected by GCC).
  explicit PhaseEstimator(std::vector<size_t> inslice_axes) : inslice_axes(std::move(inslice_axes)) {}
  PhaseEstimator(std::vector<size_t> inslice_axes, Params params)
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
  //              so the incoming phase acts as the fallback for the rest).
  // Only instantiated for complex T (the phase is ill-defined for real data); callers gate
  //   with `if constexpr (is_complex<T>::value)`.
  template <typename T> void operator()(Image<T> &in, Image<float> &sigma, Image<cfloat> &io_phase) const;

  // Per-2-D-plane primitive (the reusable, unit-testable core): solves one slice's weighted
  //   vectorial-TV ROF problem by fixed-budget primal-dual with discrepancy-criterion lambda.
  //   fr, fi   : real / imaginary parts of the empirical complex plane (native units).
  //   sigma    : per-voxel noise standard deviation for the plane; entries <= 0 or non-finite
  //              mark voxels with no usable local estimate. If enough voxels carry one, the
  //              slice is calibrated from them with spatially-varying weighting; otherwise a
  //              single global noise level is self-estimated from the data (uniform weighting).
  //   phase_r, phase_i : outputs, written with the unit-magnitude phase iff the return value
  //              is true. Left untouched (so the caller's incoming phase stands) only when the
  //              slice is fully degenerate -- no estimable noise level at all (returns false).
  // Arrays are column-major (Nx rows = first in-slice axis, Ny cols = second).
  bool solve_plane(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                   const Eigen::Ref<const Eigen::ArrayXXd> &fi,
                   const Eigen::Ref<const Eigen::ArrayXXd> &sigma,
                   Eigen::Ref<Eigen::ArrayXXd> phase_r,
                   Eigen::Ref<Eigen::ArrayXXd> phase_i) const;

  const std::vector<size_t> &axes() const { return inslice_axes; }

private:
  std::vector<size_t> inslice_axes;
  Params params;
};

} // namespace MR::Denoise
