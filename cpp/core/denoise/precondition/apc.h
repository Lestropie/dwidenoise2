/* Required Notice: Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;
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
//   phase, the estimator self-calibrates a single global noise level from the data (a
//   blockwise-robust root mean square of finite differences) and applies uniform spatial
//   weighting, so the same APC solver is used throughout. The phase is estimated per 2-D in-slice plane of each volume
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
//   Cost across the iteration loop: the phase changes little once the noise map stabilises, so a
//   volume is solved from a cold start only on the first pass that estimates it -- and, because a
//   background phase is smooth (low spatial bandwidth), every cold solve begins on a 2x-downsampled
//   grid with the result upsampled (~1/4 the cost, near-lossless for a smooth field). Every pass
//   after a volume's first is native but warm-started from that volume's previous estimate (seeded
//   with |f| * the incoming phase), so a much smaller iteration budget (the Params ..._warm counts)
//   re-settles it. Where a later pass will provide that refinement the cold solve stops at the
//   coarse grid (regime_t::COLD_COARSE); where none will -- the final pass of the schedule, which
//   must leave an authoritative native-resolution estimate -- the same warm native refinement is
//   run immediately after the coarse solve within the one call (COLD_COARSE_THEN_WARM), which is
//   the coarse-then-refine chain of a multi-pass schedule collapsed into a single pass, at roughly
//   half the cost of solving cold at native resolution outright.
//
//   These decisions are made *per volume*, not per pass, because the noise-estimation schedule may
//   temporally sub-sample: an iteration draws a subset of volumes, preconditions only those, and
//   estimates the phase of only those (Preconditioner::update_phase). A volume outside every
//   earlier subset therefore reaches a later pass with no prior estimate at all -- its stored phase
//   is still the unit-phase initialisation. Treating that as a warm start would seed the solver
//   from an uninformative phase *and* grant it only the reduced ..._warm budget, i.e. the worst of
//   both regimes. The caller therefore tracks, per volume, whether an estimate exists from a prior
//   pass, and hands operator() one VolumePlan per volume; a never-estimated volume is planned as a
//   cold solve regardless of how many passes have already run over other volumes. solve_plane
//   itself consumes only a warm_start flag (cold-start seed vs warm-start seed, and full vs
//   reduced budget). This is the dominant lever on APC's contribution to run time -- though not on
//   its result, which the bootstrap grid decides (finding 3 under DWIDENOISE2_APC_PROFILE below).
//
//   Regularisation strength lambda: the discrepancy criterion (Morozov), realised by default as
//   Chambolle's (2004) fixed-point iteration on lambda (their eq. 6) seeded from eq. 7 --
//   Params::lambda_solver selects between that and two bracketing searches for the same
//   condition -- targeting a weighted residual energy of 2 * N * sigmabar^2 (their eq. 11; the factor 2
//   is the two real+imaginary channels of complex data), where the noise level entering that
//   seed and that target is scaled by Params::discrepancy_scale. That factor is empirical and
//   greater than one: the criterion as Pizzolato state it calibrates the *image* estimate u,
//   whereas what is consumed here is arg(u), which is far more noise-sensitive than u itself
//   wherever |u| is small, so a phase that does not chase the noise needs more smoothing than
//   an image-calibrated estimate provides. See the discussion at Params::discrepancy_scale for
//   the measurement that fixes its value.
//
// Noise-map convention (see dwidenoise2_dephase.md section 3).
//   dwidenoise2's noise level map holds, per voxel, exactly Pizzolato's per-component
//   (real == imaginary) noise standard deviation sigma(x,y), in the native complex intensity
//   units in which APC operates. The mapping is the identity: no sqrt(2) or 1/2 rescaling.
//   APC must run on the *native* complex data with spatially-varying weighting; it must NOT
//   run on variance-stabilised data (the phase is scale-invariant, so nothing is gained,
//   while dividing by sigma(x,y) would inject spurious sigma-gradients into the TV term).
//
//   That contract is now guaranteed by construction: the eigenvalues from which the map is derived
//   are divided by the number of channels each Casorati entry carries as soon as they are computed
//   (Denoise::casorati_entry_channels(), Estimate::to_per_channel()). It previously was not. The
//   map handed here for complex data was the aggregate level across the two channels, larger by
//   sqrt(2), which inflated both the eq. 7 seed and the eq. 11 residual target and so over-smoothed
//   the phase. It also put the two calibration branches of solve_plane() on different scales, since
//   the self-calibration has always returned a per-component sigma: the first iteration, which
//   has no map and self-calibrates, regularised more weakly than every iteration after it. Both are
//   fixed, and testing/unit_tests/noise_convention.cpp asserts the two branches now agree.
//
//   That correction removed an accidental over-smoothing margin which, it turns out, this estimator
//   needs: with it gone the phase measurably chased the noise (see Params::discrepancy_scale, whose
//   calibration restores the margin deliberately and on a measured rather than an accidental basis).
//   The two are independent -- the map is now on the convention this file has always documented,
//   and the extra smoothing is now a stated parameter of the lambda selection rather than a units
//   error -- but the second exists because correcting the first exposed the need for it.
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
//   What that experiment found, when finally run on real data (128x128 in-plane, 105 volumes,
//   3T Prisma, MB4 / PF 0.75 / 64-channel head-neck coil; multi-iteration schedule; both the
//   per-plane log and paired denoising runs). Three results, none of which the pre-run
//   expectations above anticipated:
//     1. The lambda fixed point is NOT reached, and at native resolution is not reachable.
//        On the coarse cold pass it needs ~30 lambda updates (budget 10). On the native warm
//        passes lambda dives well below its eq. 7 seed, turns, and then creeps back so slowly
//        that 300 updates x 60 sweeps still has not converged. Chambolle-Pock needs sweeps of
//        the order of the plane's width to propagate information across it, so at 128 wide a
//        budget of tens of sweeps leaves the inner solve far from its own minimiser, and the
//        outer fixed point inherits that lag. The budgets below therefore do NOT merely bound
//        the cost of reaching a lambda the criterion selects: together with
//        discrepancy_scale they *are* what selects it. Raising them raises the achieved
//        smoothing; they cannot be treated as a free speed knob.
//        What rescues this from being a calibration of an arbitrary stopping point is that the
//        *result* settles even though lambda does not: quadrupling the budgets the file shipped
//        with brings the noise-recovery figure R to within a few percent of its value under a
//        further doubling, so there is a well-defined answer to calibrate discrepancy_scale
//        against. That is what the current budgets buy, at roughly 5x the APC cost, and it is
//        why they are what they are. Anyone reducing them for speed is not merely trading
//        accuracy for time -- they are moving the operating point to one where the noise level
//        depends on where the iteration was cut, and must re-derive discrepancy_scale there.
//     2. cp_tol never fires. Over 99% of warm and ~80-93% of cold solves consume their entire
//        sweep budget, and raising cp_tol from 1e-6 to 1e-4 changed neither the run time nor
//        any output. It remains as a safety net for planes that do settle early, but it is not
//        a lever, and loosening it as a speed measure would cut smoothing (see 1) rather than
//        cut waste.
//     3. The dominant lever on the result is not the ..._warm budget at all but the 2x
//        coarse-grid cold bootstrap. Against a reference that solves cold at native resolution
//        on the full budget at *every* pass, replacing only the first pass's grid (native cold
//        bootstrap, then the ordinary reduced warm passes) reproduces it to 1.1% RMS, whereas
//        the shipped coarse bootstrap differs by 4.8% -- and no warm budget from 6/3/6 up to
//        the full 25/10/25 moves that figure by more than 0.2%. The warm start is sticky: at
//        native resolution the solver stays near whatever it is seeded with. That is what makes
//        the reduced warm budget nearly free, and it is also why the bootstrap grid, which sets
//        the phase's spatial bandwidth once and for all, decides the outcome.
//   Validating on rank alone proved insufficient, because the estimated signal rank is a
//   function of the noise level the same run estimates: a configuration whose phase absorbs
//   noise reports both a lower noise level and, through the Marchenko-Pastur threshold that
//   level sets, a lower rank -- so over-fitting reads as an improvement on that metric. The
//   discriminating measurement is the added-noise recovery test described at
//   Params::discrepancy_scale, with rank retained only as a check that the extra smoothing has
//   not cost demodulation quality (it has not: rank rises ~1% while the recovery deficit,
//   14-23% of the added noise variance, is eliminated).
// ---------------------------------------------------------------------------------------
class AdaptivePhaseEstimator {
public:
  // Strategy by which the fidelity weight lambda satisfying the discrepancy criterion is located.
  //   All three target the same condition and differ only in how they search for it; see
  //   Params::lambda_solver.
  enum class lambda_solver_t {
    // Chambolle's (2004) multiplicative fixed point (their eq. 6), seeded from their eq. 7.
    FIXED_POINT,
    // Bracket the crossing by doubling or halving lambda from that same seed, then bisect.
    BISECTION,
    // Reach the crossing by a monotone walk in small geometric steps from an over-fidelity
    //   start, then bisect within the (narrow) interval the walk ends in.
    CONTINUATION
  };

  struct Params {
    // Scale factor applied to the noise level sigmabar *only* where it selects lambda: the
    //   eq. 7 seed and the eq. 11 residual target. The per-voxel fidelity weights
    //   w = sigmabar^2 / sigma^2 are untouched (and are scale-invariant regardless). The
    //   targeted residual energy is therefore discrepancy_scale^2 times the noise energy the
    //   discrepancy principle nominally prescribes, i.e. the phase is deliberately smoothed
    //   beyond the point at which the residual accounts for the noise.
    //
    // Why greater than one. Pizzolato's criterion calibrates the *image* estimate u; APC
    //   consumes arg(u), which is far more noise-sensitive than u wherever |u| is small, so
    //   an image-calibrated u yields a phase that still follows the noise. Demodulating by
    //   such a phase rotates each voxel's noise toward the real axis in a data-dependent way,
    //   depleting the imaginary channel of the Casorati matrix and biasing the estimated noise
    //   level DOWN -- the failure mode this factor exists to prevent.
    //
    // How the value was fixed (2026-08-05, real 3T data, 128x128x105 complex, two separate
    //   slabs). Add complex Gaussian noise of known per-component sigma_add to the data, run
    //   the noise level estimation on both the original and the noisier copy, and form
    //     R = median over the brain of (sigma_noisy^2 - sigma_orig^2) / sigma_add^2 ,
    //   the fraction of a *known* added noise variance the estimator still sees. Provided the
    //   estimator's multiplicative bias is stable across the two noise levels -- which two
    //   values of sigma_add an octave apart confirm here -- R IS that bias: R = 1 is unbiased,
    //   R < 1 is noise absorbed by the phase estimate. Nothing about the phase is simulated;
    //   the anatomy, the background phase and the native noise are all empirical, and only the
    //   increment is synthetic (and white, matching the measured background autocorrelation of
    //   these data: 0.06 along readout, 0.12 along phase-encode, at unit lag).
    //
    // The value cannot be chosen without simultaneously choosing the iteration budgets, because
    //   the lambda fixed point is never reached (see the DWIDENOISE2_APC_PROFILE findings above)
    //   and the two therefore jointly determine the achieved smoothing. Writing M for a common
    //   multiplier on all six budgets below (M = 1 being the counts this file shipped with until
    //   2026-08-05, i.e. 25/10/25 : 12/5/12), R over slab A at sigma_add = 149 behaves thus:
    //         scale \ M      1       2       4       8
    //           1.0        0.85            0.92
    //           1.05                       1.00    0.96
    //           1.1        0.90    0.90    1.02    1.04
    //           1.8        1.00    1.09    1.12    1.14
    //   Two things follow. First, at any fixed scale more iterations means more smoothing, so R
    //   is a function of where the iteration is stopped. Second, R does settle: by M = 4 it moves
    //   by only a few percent for a further doubling. A scale tuned at M = 1 is therefore tuned
    //   to a truncation -- scale 1.8 reads R = 1.00 at M = 1 but drifts to 1.14, the rigid-phase
    //   limit, once the iteration is allowed to run -- whereas a scale tuned at M = 4 is tuned to
    //   the settled answer. The latter is what is configured here, at roughly 5x the APC cost
    //   (2.5-4x end to end). Over slab A at sigma_add = 149, slab B at sigma_add = 149 and slab A
    //   at sigma_add = 75, these values give R = 1.00 / 0.96 / 0.98, and doubling the budgets
    //   again moves the first to 0.96.
    //
    // Anchors for reading R: -demodulate linear -- a three-degree-of-freedom phase that cannot
    //   absorb noise at all -- gives R = 1.13-1.16 on these data, and the magnitude pathway,
    //   which has no phase estimate whatsoever, gives 0.87, so the estimator carries a bias of
    //   order 10% in either direction from sources unrelated to the phase. R = 1 is the target
    //   but agreement to a few percent is the resolution of the measurement, not exactness.
    //
    // WHERE THIS IS CALIBRATED, AND WHAT REMAINS UNCALIBRATED. The table above was measured where
    //   solve_plane() takes its sigmabar from the noise level map. Where no map exists -- the first
    //   iteration of any schedule, and *every* solve under a single-row schedule (-vst_method none,
    //   -fixed_rank, apriori) -- sigmabar is self-calibrated from the plane's own data instead.
    //   Both branches estimate the same functional, the root mean square of the per-voxel noise
    //   level (see estimate_sigma_rms() in apc.cpp), so the two are on one scale. What the
    //   self-calibration cannot do is see past the anatomy where tissue contrast within a
    //   neighbourhood exceeds the noise, which on low b-value images it does: on the reference slab
    //   the self-calibrated level runs about 1.8 times the map's RMS at native resolution and 3.7
    //   times it on the coarse grid a cold solve uses. That errs toward a smoother phase, which is
    //   the harmless direction -- an over-smoothed phase approaches the rigid-phase limit, while an
    //   under-smoothed one absorbs noise into the phase and biases the estimated level down.
    //
    //   End to end on that slab at sigma_add = 149: the single-pass regime (-vst_method none)
    //   reaches R = 1.02 with an estimated noise level of 183, against 166 for -demodulate hann,
    //   181 for the magnitude pathway and 200 for -demodulate linear; the default multi-iteration
    //   schedule, where the self-calibrated solve is only the bootstrap pass and the map-driven
    //   passes that follow correct it, reaches R = 0.96.
    double discrepancy_scale = 1.05;
    // Chambolle-Pock primal-dual iterations run between successive lambda updates. The
    //   solver is warm-started across lambda updates, so this need not fully converge each
    //   time -- it interleaves image and lambda refinement (a standard, efficient scheme).
    // These three counts, and the ..._warm counts below, were quadrupled on 2026-08-05 (from
    //   25/10/25 : 12/5/12) so that the estimate is read where the noise-recovery figure R has
    //   settled rather than mid-transient; see discrepancy_scale, with which they must be
    //   re-calibrated jointly if either is changed.
    ssize_t cp_iter = 100;
    // Discrepancy-criterion (Morozov) fixed-point updates of lambda (their eqs. 6-7, 11).
    ssize_t max_lambda_iter = 40;
    // Extra primal-dual iterations at the converged lambda, to settle the image before the
    //   phase (its argument) is extracted.
    ssize_t polish_iter = 100;
    // Reduced iteration budget for every warm-started solve: each pass over a volume after the one
    //   that first estimated it, and the native-resolution stage that immediately follows a
    //   COLD_COARSE_THEN_WARM volume's own coarse solve. A cold solve starts from the data on the
    //   coarse grid and spends the full budget; a warm-started one starts from an estimate that
    //   (whether upsampled from the coarse grid or carried over from a previous pass) barely moves,
    //   so far fewer sweeps re-settle it. Because the solver runs a *fixed* budget
    //   (no convergence test drives early exit besides lambda_tol), this reduced budget -- not the
    //   warm start alone -- is what shortens the later passes; the warm start is what keeps that
    //   reduction safe, which is precisely why it must not be granted to a volume that has no
    //   previous estimate (see the VolumePlan discussion above).
    //
    // Measured behaviour of this reduction (real data, added-noise recovery test): against a
    //   reference solving cold at native resolution on the full budget at every pass, the
    //   reduced warm budget costs 1.1% RMS in the denoised series, and sweeping the ratio from
    //   a quarter of these counts to parity with the cold budget moves that by under 0.2% --
    //   the warm start really is sticky enough for the reduction to be nearly free. What these
    //   counts are NOT is a free speed knob: since the lambda fixed point is never reached, they
    //   co-determine the achieved regularisation with discrepancy_scale, and moving them moves
    //   the noise-recovery figure R defined there. Validate any change on R first, with the
    //   denoised rank as a secondary check; never on phase MSE.
    ssize_t cp_iter_warm = 48;
    ssize_t max_lambda_iter_warm = 20;
    ssize_t polish_iter_warm = 48;
    // How lambda is chosen to satisfy the discrepancy criterion.
    //
    // The criterion asks for the lambda at which the weighted residual over the calibration
    //   domain, R(lambda) = Rr(lambda) + Ri(lambda), meets its target 2 sqrt(ndomain) sigmabar
    //   (with sigmabar scaled by discrepancy_scale). R is monotone *decreasing* in lambda -- a
    //   larger lambda means higher fidelity and hence a smaller residual -- and runs between two
    //   plateaux: R -> 0 as lambda -> infinity, where u -> f, and R -> its maximum as lambda -> 0,
    //   where u collapses to a constant image. It crosses the target exactly once in between, so
    //   any method that brackets that crossing converges unconditionally, while a fixed-point
    //   iteration converges only at a rate set by the local slope of R.
    //
    // FIXED_POINT is lambda <- lambda R(lambda)/target. Its contraction factor is
    //   1 + d ln R / d ln lambda, which on 3T diffusion planes measures between 0.14 and 0.68 per
    //   update, so with an exactly minimised inner problem it would reach several digits in of
    //   order ten updates. What limits it is that the inner problem is *not* minimised exactly:
    //   a truncated Chambolle-Pock sequence has not yet moved u far from where it started, so it
    //   under-reports R, and the update then drives lambda below the lambda the criterion asks
    //   for -- on 128x128 planes at the budgets above, by a factor of three to four.
    //
    // BISECTION and CONTINUATION instead bracket the crossing and bisect within the bracket, so
    //   the returned lambda is bounded by the bracket however the search reached it. Both work in
    //   log lambda -- geometric steps, geometric midpoints -- because lambda ranges over decades
    //   while R is close to a power law in it. They differ only in how the bracket is found:
    //   BISECTION doubles or halves, which reaches the crossing in few steps but makes each inner
    //   solve start from an image minimising a substantially different energy; CONTINUATION steps
    //   by the smaller lambda_step_factor, so every inner solve is a small perturbation of the one
    //   before and the warm-started image stays near its own minimiser, at the cost of more steps.
    //   Neither removes the truncation bias described above: bracketing locates the root of the
    //   residual *as evaluated*, so the accuracy of the located lambda is set by how accurately
    //   each evaluation minimises the inner problem (see cp_iter), not by the bracket width.
    lambda_solver_t lambda_solver = lambda_solver_t::FIXED_POINT;
    // Geometric ratio by which lambda is stepped while searching for a bracket. Consumed by
    //   CONTINUATION only; BISECTION doubles or halves. Ratios close to one keep each inner solve
    //   a small perturbation of the last but need proportionally more steps to reach the crossing,
    //   which on these data lies a factor of two to sixteen below the eq. 7 seed.
    double lambda_step_factor = 1.5;
    // Multiplier applied to the eq. 7 seed to place the start of the bracket search. One starts at
    //   the seed itself, which on every plane measured here already lies on the high-fidelity side
    //   of the crossing, so the search walks downward from it.
    double lambda_start_scale = 1.0;
    // Maximum number of geometric steps the bracket search may take. Exhausting it means the
    //   criterion is met at no lambda within reach of the seed, which contradicts the shape of R
    //   described above and therefore indicates a degenerate plane rather than a slow search: the
    //   plane is declined outright, so its incoming phase stands, in preference to regularising it
    //   at an arbitrary lambda.
    ssize_t lambda_bracket_cap = 40;
    // Relative half-width of the bracket at which bisection stops, i.e. the relative uncertainty
    //   of the returned lambda. Each bisection evaluation halves the log-width of the bracket, so
    //   tightening this by a factor of two costs one further evaluation. max_lambda_iter (or its
    //   ..._warm counterpart) bounds the number of those refinement evaluations; the bracket
    //   search preceding them is bounded separately by lambda_bracket_cap.
    double lambda_root_tol = 1e-3;
    // Relative-change tolerance for early exit of the lambda fixed-point iteration; consumed by
    //   lambda_solver_t::FIXED_POINT only. Rarely reached in practice: the fixed point converges
    //   within budget only on the coarse cold pass, and even there for a minority of planes.
    double lambda_tol = 1e-2;
    // Relative-change tolerance for early exit of the *inner* Chambolle-Pock sweeps
    //   themselves (both the lambda-loop sweeps and the polish sweeps), checked every sweep
    //   at negligible extra cost (the comparison piggybacks on the primal update's existing
    //   per-voxel pass, so it does not add another sweep over the plane). Deliberately tight
    //   by default: this is a safety net against the *fixed* per-call budget doing
    //   inconsequential extra work once a warm-started plane has already re-settled. Set to 0
    //   (or a negative value) to disable and always spend the full budget, matching pre-2026-07
    //   behaviour exactly.
    //
    // On real data it does not fire: over 99% of warm and ~80-93% of cold solves consume their
    //   whole budget, and raising it to 1e-4 changed neither run time nor any output. Do not
    //   reach for it as a speed measure -- since the budgets co-determine the regularisation
    //   (see discrepancy_scale), a tolerance loose enough to cut sweeps would cut smoothing.
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

  // How one volume's phase is solved in a given pass. Which regime applies follows from two facts
  //   the caller (and only the caller) knows: whether that volume already has an estimate, and
  //   whether a later pass will refine what this one produces.
  enum class regime_t {
    // Warm-started refinement at native resolution, on the reduced Params ..._warm budget: the
    //   volume has an estimate from a prior pass, which barely moves once the noise map settles.
    WARM,
    // Cold solve on the 2x-downsampled grid, its phase upsampled to native. A bootstrap: correct
    //   only where a later pass will refine this volume at native resolution.
    COLD_COARSE,
    // Cold solve on the 2x-downsampled grid, immediately followed by a warm-started
    //   native-resolution refinement of that upsampled result -- the coarse-then-refine chain of a
    //   multi-pass schedule, collapsed into a single call. For a volume with no prior estimate
    //   reached by the final pass of the schedule, where there is no later pass to carry the
    //   refinement, but whose result must still be native-resolution and authoritative. Roughly
    //   half the cost of a full cold native solve (~1/4-cost coarse sweeps plus the reduced warm
    //   budget at native resolution, versus the full budget at native resolution).
    COLD_COARSE_THEN_WARM
  };

  // Per-volume control of a single APC pass: one entry per *serialised* volume index (the
  //   Casorati column index; for 4D data simply the volume index, for >4D the index assigned by
  //   the preconditioner's serialisation image). The caller owns the policy -- which volumes this
  //   iteration needs, which of them have been estimated before, and whether a later pass will
  //   refine the result -- and expresses it here; this class only executes it.
  struct VolumePlan {
    // (Re-)estimate this volume's phase in this pass. False for volumes the current iteration
    //   does not use (outside the temporal subset): their stored phase is left untouched, whether
    //   that is an estimate from an earlier pass or the initial unit phase.
    bool estimate = false;
    // The regime in which to solve it. WARM is valid only for a volume that has been estimated
    //   before, however many passes have already run over other volumes.
    regime_t regime = regime_t::COLD_COARSE_THEN_WARM;
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
  //   of every planned volume, multi-threaded over the outer (slice-normal + volume + any
  //   serialised supra-volume) axes via MRtrix3 ThreadedLoop; each plane is an independent problem.
  //   in       : native complex data (empirical input; never variance-stabilised).
  //   sigma    : per-component noise standard-deviation map (the VST scale). May be at a
  //              different resolution to "in"; it is cubic-interpolated at each voxel, exactly
  //              as the preconditioner samples it elsewhere. Voxels with a non-finite or
  //              non-positive interpolated sigma are treated as outside the usable domain. May
  //              also be invalid (empty): on the first iteration no noise map exists yet, and
  //              each slice then self-calibrates a global noise level from the data.
  //   io_phase : unit-magnitude Image<cfloat>, same grid as "in". Overwritten in place with
  //              the new estimate (per-slice: only slices of planned volumes, and of those only
  //              slices with a usable domain, are modified; the incoming phase acts as the
  //              fallback for the rest). For a warm-started volume it is additionally read first
  //              as the solver's initial guess.
  //   serialise : the preconditioner's serialisation image mapping a >4D volume multi-index to
  //              its Casorati column, used to look up each plane's entry in "plan". Invalid /
  //              empty for 4D data, where the column index is the volume index itself.
  //   plan     : one VolumePlan per serialised volume index (size == the serialised volume
  //              count), selecting which volumes are estimated in this pass and, for each, the
  //              cold/warm and native/downsampled regime. Volumes with estimate == false are
  //              skipped entirely.
  // Only instantiated for complex T (the phase is ill-defined for real data); callers gate
  //   with `if constexpr (is_complex<T>::value)`.
  template <typename T>
  void operator()(Image<T> &in,
                  Image<float> &sigma,
                  Image<cfloat> &io_phase,
                  Image<uint32_t> &serialise,
                  const std::vector<VolumePlan> &plan) const;

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
  //              when the slice is fully degenerate -- no estimable noise level -- and, under a
  //              bracketing lambda solver, when no lambda within Params::lambda_bracket_cap
  //              geometric steps of the seed satisfies the discrepancy criterion (false).
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
