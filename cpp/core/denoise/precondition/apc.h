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
//   the estimate is re-derived every outer iteration. A bounded per-slice cost therefore suits
//   the loop far better than running any solver against a convergence test. The bound is not,
//   however, a truncation: with the step sizes Params::cp_step_ratio selects, the budgets
//   configured below reach the minimiser of the energy above at the weight the discrepancy
//   criterion asks for -- measured per plane against exactly minimised references, to within
//   about 2e-4 of that weight. See dwidenoise2_dephase.md sections 2.2 and 2.5.
//
//   Cost across the iteration loop: the phase changes little once the noise map stabilises, so a
//   volume is solved from a cold start only on the first pass that estimates it -- and, because a
//   background phase is smooth (low spatial bandwidth), every cold solve begins on a downsampled
//   grid with the result upsampled (~1/4 the cost per level of decimation; Params::cascade_depth
//   sets how many such levels there are, two in total by default -- one coarse and the native one,
//   with Params::cascade_lambda deciding how each level calibrates its own regularisation and
//   Params::cascade_prolongation how its result is carried to the level above it. What a coarse
//   level produces is a bootstrap and not an answer: it minimises a differently calibrated energy,
//   which the native level above it repairs. See Params::cascade_depth). Every pass
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
//   reduced budget). This is the dominant lever on APC's contribution to run time.
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
//
//   One alternative that looks promising and is not: splitting the phase into a low-order
//   component removed analytically -- the k-space Hann demodulator this codebase already carries
//   would supply one -- and a TV-regularised remainder, on the expectation that the remainder is a
//   short-wavelength correction the solver settles in few sweeps. It cannot accelerate this solve,
//   and the reason is a property of the functional rather than of the data: TV_vec is built from
//   first differences, so a long-wavelength field carries almost none of its energy. Measured on
//   exactly minimised solutions of the energy above, removing every ideal low-pass component from
//   wavelengths of 64 voxels down to 8 changes TV_vec by between -1.3% and +0.1%, and the Hann
//   component increases it. The fidelity term meanwhile is exactly invariant under demodulation by
//   a unit-modulus phase, so the split poses the same problem in different coordinates: handing the
//   solver the exact long-wavelength phase for free is worth 1.00x in sweeps at every accuracy
//   threshold measured, on both grids and in all four b-value shells. The same argument is why
//   -demodulate hann cannot be recast as a cheap first stage of this estimator.
// ---------------------------------------------------------------------------------------
//
// Data-driven early termination, and the DWIDENOISE2_APC_PROFILE build.
//   Beyond the reduced ..._warm budgets, run_cp() stops a sweep sequence early once the relative
//   per-sweep change in the primal image falls below Params::cp_tol -- a genuinely data-driven
//   cut, in that it only shortens a call once *that specific plane, at that specific outer
//   iteration* has stopped moving, where the ..._warm budgets are a fixed, uniform guess. The
//   comparison is folded into the primal update's existing per-voxel pass, so it costs no extra
//   sweep over the plane. Whether it fires at all is a property of the step sizes rather than of
//   the threshold; Params::cp_tol records what it does at the values configured here, and why it
//   must not be loosened as a speed measure.
//
//   Build with -DDWIDENOISE2_APC_PROFILE to compile in a per-plane DEBUG() line (run under -debug)
//   reporting, for every solve_plane() call: plane size, cold/warm, spatially-varying vs
//   self-calibrated, ndomain fraction, sigmabar, lambda iterations used vs budgeted, total
//   Chambolle-Pock sweeps actually consumed vs the full budget (n_lambda*n_cp + n_polish), and the
//   converged lambda. This is compile-time gated rather than always emitted under -debug because
//   computing and formatting it triggers string work on every one of the many thousands of
//   solve_plane() calls in a full run, which would be a needless cost in ordinary -debug
//   diagnostic use. The sweep- and lambda-iteration counters that feed the line exist *only* to
//   report it -- run_cp()'s early-exit decision itself needs no count, only the per-sweep
//   relative-change comparison -- so those counters (and run_cp()'s ssize_t return) are themselves
//   compiled out under the non-profile build, not merely left unprinted. A non-profile build's
//   timing therefore isolates the effect of cp_tol's early exit from the cost of instrumenting it.
// ---------------------------------------------------------------------------------------
//
// What the defaults below were measured against, and what that measurement does not cover.
//
//   The reference. Every per-plane accuracy figure quoted in this file is a distance to the
//   *exact* minimiser of the energy above at the *exact* root of the discrepancy condition, not to
//   another run of this solver. It is obtained per plane by bracketing and bisecting that condition
//   to 1e-6 relative, each inner problem being solved until its Fenchel duality gap certifies
//   E(u) - min E below 1e-9 of |E|. The dual of that energy is
//
//       D(p) = -<f, div p> - (1/(2 lambda)) sum_x ||div p(x)||^2 / w(x),   ||p(x)||_2 <= 1,
//
//   with div the exact negative adjoint of the forward difference this solver forms, its boundary
//   rows included; a textbook divergence with a symmetric or periodic boundary is a different
//   operator and bounds nothing here. Two certified solves of one plane from different starting
//   points agree to about 1e-5 rad, which is the floor of every per-plane figure quoted below.
//
//   What that reference settled, and it is why several of the defaults are where they are: the
//   iteration budgets were never the binding constraint. With the classical balanced primal-dual
//   steps the lambda search settles near 0.30 of the weight the criterion asks for and stays there
//   at any affordable budget, so the budgets rather than the criterion were selecting the
//   regularisation, a coarse-grid bootstrap rather than the budgets was deciding the answer, and
//   the fixed point looked unreachable at native resolution. None of that survives the step sizes:
//   fixing the dimensionless product tau * lambda instead (Params::cp_step_ratio) locates the
//   criterion's weight to within about 2e-4 on the same budgets, cold and warm alike, and the depth
//   of the cold cascade then moves the phase returned by some 14% rather than by a factor of four.
//
//   The data. One complex acquisition: 3T Siemens Prisma, MB4, PF 0.75, 64-channel head-neck coil,
//   128x128 in plane, 105 volumes over four b-value shells. Two 4-slice slabs of it, one central
//   (high g-factor) and one peripheral; no whole-volume run was validated. In-plane widths 128, 96
//   and 64, obtained by *cropping* that acquisition and never by resampling, which would change the
//   noise correlation. Intensities rescaled over six orders of magnitude. One added-noise increment
//   at which the recovery check closes.
//
//   The metrics, including the two that must not be used. Per plane, the distance to the certified
//   minimiser above says whether the solver solves the problem it is posed. End to end, the
//   added-noise recovery figure R defined at Params::discrepancy_scale says whether that problem is
//   the right one, and it is the criterion any re-tune must be validated on. Do NOT use the
//   retained rank as a primary metric: the rank is thresholded by the noise level the same run
//   estimates, so a phase that absorbs noise reports both a lower level and, through that
//   threshold, a lower rank, and over-fitting reads as an improvement. Rank is a check that added
//   smoothing has not cost demodulation quality, and nothing more. Do NOT tune on phase MSE against
//   another phase estimate either: no reference phase exists on real data, and the phase is a means
//   of reducing the rank of the demodulated Casorati matrix rather than an end in itself.
//
//   What is not covered. No second vendor, field strength, coil or reconstruction, and no second
//   acquisition of any kind. No in-plane matrix size above 128, and none below 64. The dependence
//   of R on the size of the added-noise increment, and the regime in which no noise level map ever
//   exists, are both open and are set out at Params::discrepancy_scale. Denoising quality itself
//   was not assessed: no reference exists on real data against which a denoised series could be
//   scored, so what is established is that the solver reaches the minimiser the criterion selects,
//   and that the noise level estimated from the result is unbiased at the increment measured --
//   not that the phase it returns is the best available background phase.
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

  // Where each level of a cold solve's resolution cascade (Params::cascade_depth) gets the noise
  //   level that calibrates its discrepancy criterion, and hence the fidelity weight it solves at.
  //   See Params::cascade_lambda.
  enum class cascade_lambda_t {
    // Each coarsened level estimates a single global noise level from its own decimated data and
    //   weights every voxel equally.
    SELF_CALIBRATED,
    // Each coarsened level takes the noise level map decimated onto its own grid, and so keeps the
    //   spatially-varying weighting the native problem carries.
    RESTRICTED_MAP,
    // As RESTRICTED_MAP, except that only the coarsest level searches for a fidelity weight; every
    //   finer coarsened level is handed that weight scaled by Params::cascade_lambda_ratio per
    //   level and runs no search of its own.
    COARSEST_ONLY
  };

  // How a per-voxel noise level is carried onto a 2x-decimated grid. The two differ exactly where
  //   sigma varies within a 2x2 block -- at parallel-imaging g-factor gradients, which are steep --
  //   and agree everywhere else. See Params::cascade_restriction.
  enum class cascade_restriction_t {
    // The noise level of the decimated datum itself: 2x2 averaging of four voxels whose noise is
    //   independent gives a variance of one quarter their mean variance.
    SIGMA_SQUARED_MEAN,
    // The mean of the four fidelity weights, i.e. of sigma^-2, placed on the same absolute scale.
    //   Preserves the total weight of the block rather than its noise level.
    WEIGHT_MEAN
  };

  // How the solution of one cascade level is carried to the finer level above it. Neither
  //   interpolates arg(u), which wraps. See Params::cascade_prolongation.
  enum class cascade_prolongation_t {
    // Bilinearly interpolate the unit-magnitude phase (its cos and sin channels), renormalise, and
    //   let the finer level re-take its magnitude from its own data.
    PHASE,
    // Bilinearly interpolate the two channels of the image iterate u itself, carrying the coarse
    //   level's smoothed magnitude up with its phase.
    ITERATE
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
    // How the value was fixed (real 3T data, 128x128x105 complex, two separate slabs). Add complex
    //   Gaussian noise of known per-component sigma_add to the data, run the noise level estimation
    //   on both the original and the noisier copy, and form
    //     R = median over the brain of (sigma_noisy^2 - sigma_orig^2) / sigma_add^2 ,
    //   the fraction of a *known* added noise variance the estimator still sees. Where the
    //   estimator's multiplicative bias does not itself depend on the noise level, R IS that bias:
    //   R = 1 is unbiased, R < 1 is noise absorbed by the phase estimate. Nothing about the phase
    //   is simulated; the anatomy, the background phase and the native noise are all empirical, and
    //   only the increment is synthetic (and white, matching the measured background
    //   autocorrelation of these data: 0.06 along readout, 0.12 along phase-encode, at unit lag).
    //
    // This value and the iteration counts below were derived together, and must be. The counts
    //   decide how nearly the inner solve reaches the fidelity weight the criterion asks for, and
    //   this scale decides which weight that is. At the counts and step ratio configured here the
    //   lambda iteration does reach it -- to within about 2e-4 of the exact root on every plane
    //   measured, on the cold chain and on the steady-state warm pass alike -- so the achieved
    //   smoothing is now a function of this scale alone; but that is a property of those counts and
    //   of cp_step_ratio, and changing either re-couples the two and obliges a re-derivation.
    //
    // Measured over the default multi-iteration schedule at sigma_add = 149, slab A:
    //         scale    1.05   1.15   1.20   1.26   1.35   1.60
    //         R        0.81   0.97   1.01   1.04   1.07   1.12
    //   Slab B reads 1.01 at scale 1.20. Interpolating, R = 1 falls at about 1.19; 1.20 is used
    //   because a third digit would claim a precision the measurement does not have (see the
    //   anchors below). The response is steep at the calibration point -- roughly 0.04 in R per
    //   0.05 in scale -- and saturates above it: by 1.60 it has reached the 1.12 that
    //   -demodulate linear gives on these data, which is the rigid-phase limit and the most any
    //   amount of extra smoothing can buy.
    //
    // Anchors for reading R: -demodulate linear -- a three-degree-of-freedom phase that cannot
    //   absorb noise at all -- gives R = 1.13-1.16 on these data, and the magnitude pathway,
    //   which has no phase estimate whatsoever, gives 0.87, so the estimator carries a bias of
    //   order 10% in either direction from sources unrelated to the phase. R = 1 is the target
    //   but agreement to a few percent is the resolution of the measurement, not exactness.
    //
    // WHAT THIS CALIBRATION DOES NOT COVER. Each of the three is measured, not suspected.
    //   1. R is not independent of sigma_add on these data, so the premise under which R equals the
    //      estimator's bias holds only near the level it was calibrated at. At sigma_add = 75 --
    //      one octave down, where the added variance is 15% of the total and the statistic
    //      magnifies a given error in the estimated level about 2.6-fold -- the same runs read
    //      R = 0.80 (slab A) and 0.85 (slab B). This is a property of the noise level estimation
    //      rather than of these settings: at these same iteration counts with zero step ratio, a
    //      looser lambda tolerance and scale 1.05, the identical measurement reads 0.87 and 0.83.
    //      Read R at a much smaller increment as a consistency check that does not currently
    //      close, not as a bias.
    //   2. Where no noise level map exists -- the first iteration of any schedule, and *every*
    //      solve under a single-row schedule (-vst_method none, -fixed_rank, apriori) -- sigmabar
    //      is self-calibrated from the plane's own data. Both branches estimate the same
    //      functional, the root mean square of the per-voxel noise level (see estimate_sigma_rms()
    //      in apc.cpp), so the two are on one scale. What the self-calibration cannot do is see
    //      past the anatomy where tissue contrast within a neighbourhood exceeds the noise, which
    //      on low b-value images it does: on the reference slab it runs about 1.8 times the map's
    //      RMS. A solve that stops short of its minimiser truncates that inflation away; one that
    //      does not expresses it in full, and the single-row regime consequently lands at R = 1.10
    //      at sigma_add = 149 where the multi-iteration schedule lands at 1.01. The error is toward
    //      a smoother phase, which is the harmless direction -- an over-smoothed phase approaches
    //      the rigid-phase limit, while an under-smoothed one absorbs noise into the phase and
    //      biases the estimated level down -- but one value of this scale cannot place both regimes
    //      at unity, and this one is placed on the multi-iteration schedule.
    //   3. The value was derived on a single acquisition (3T Siemens Prisma, MB4, PF 0.75,
    //      64-channel head-neck coil, 128x128 in plane), on two 4-slice slabs of it. It holds
    //      across in-plane widths 128, 96 and 64 (R = 1.01 / 0.98 / 1.03), all three obtained by
    //      cropping that same acquisition, and it is very nearly invariant to the intensity units
    //      of the data, R agreeing to four significant figures over six orders of magnitude of
    //      rescaling. No other vendor, field strength, coil or reconstruction has been tested; no
    //      in-plane matrix size above 128 or below 64; and no whole-volume run.
    double discrepancy_scale = 1.20;
    // Chambolle-Pock primal-dual iterations run between successive lambda updates. The
    //   solver is warm-started across lambda updates, so this need not fully converge each
    //   time -- it interleaves image and lambda refinement (a standard, efficient scheme).
    // These three counts, and the ..._warm counts below, are placed where the lambda iteration
    //   reaches the root of the discrepancy condition rather than stopping short of it: measured
    //   against exactly minimised references, the weight located here is within about 2e-4 of that
    //   root. What places it there is not the size of these counts but the step sizes
    //   cp_step_ratio selects: at these same counts with the classical balanced steps the search
    //   stops at about 0.3 of the correct weight, and the counts would have to grow by orders of
    //   magnitude to close that gap. They are nevertheless not a free speed knob -- see
    //   discrepancy_scale, with which they must be re-derived jointly if either is changed.
    ssize_t cp_iter = 100;
    // Discrepancy-criterion (Morozov) fixed-point updates of lambda (their eqs. 6-7, 11). This is
    //   an upper bound and not a count: lambda_tol usually stops the search first.
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
    //   counts are NOT is a free speed knob: they decide how nearly the lambda search reaches the
    //   weight the criterion asks for, so they co-determine the achieved regularisation with
    //   discrepancy_scale, and moving them moves the noise-recovery figure R defined there.
    //   Validate any change on R first, with the denoised rank as a secondary check; never on
    //   phase MSE.
    ssize_t cp_iter_warm = 48;
    ssize_t max_lambda_iter_warm = 20;
    ssize_t polish_iter_warm = 48;
    // Number of resolution levels a cold solve passes through, the native grid included: 1 solves
    //   cold at native resolution outright; 2 solves cold on the 2x-downsampled grid and upsamples
    //   that phase to native; 3 places a 4x-downsampled level below that, and so on. Each level
    //   below native carries the data 2x2 box-decimated from the level above it and self-calibrates
    //   its own noise level from those data. Only the coarsest level is solved cold, on the full
    //   iteration budget; every finer level is warm-started from the level beneath it upsampled and
    //   so costs only the reduced ..._warm budget, as does the final native refinement where the
    //   regime asks for one.
    //
    // Why more than one coarsening can pay for itself. Chambolle-Pock propagates information one
    //   voxel per sweep, so a cold solve needs sweeps of the order of its plane's width while each
    //   sweep costs its area: halving the width halves the sweeps needed and quarters the cost of
    //   each. The long-wavelength content those sweeps are chasing is exactly what a background
    //   phase is made of and exactly what survives decimation, so it can be recovered where it is
    //   cheapest, leaving each finer level only the shorter-wavelength correction that a warm start
    //   settles in few sweeps.
    //
    // Why that argument no longer decides how *accurate* the result is, only how cheap. It is an
    //   argument about how far a truncated iterate travels, so it settles the outcome only while
    //   the iterate is truncated. With the step sizes cp_step_ratio selects, a cold solve reaches
    //   its minimiser at native resolution within the budget above whether or not it is bootstrapped
    //   from a coarse grid: measured against exactly minimised references at in-plane widths 64, 96
    //   and 128, over a sixteenfold range of iteration budgets spanning the counts above, the gap
    //   between depths 1 and 2 falls from a factor of 3.9 in the phase error under truncation to
    //   14%, and at the settings configured here the two are indistinguishable in the phase they
    //   return.
    //
    // Why two, given that one is the cheaper route to a given accuracy. Per plane at matched
    //   accuracy, on each width's own cost/accuracy front, one level is cheaper everywhere: any
    //   depth of two or more costs 1.3x to 7.3x more sweeps and 0.7x to 4.0x more voxel-updates to
    //   reach the same phase error, and at an equal budget the single level is the more accurate of
    //   the two in 22 of the 24 (width, budget, step size) cells measured. On that comparison the
    //   cascade is not beneficial and is not retained on it. What retains it is whole-pipeline run
    //   time at the counts above, which is what the defaults were selected on: wall clock tracks
    //   voxel-updates rather than sweeps, a two-level chain spends most of its sweeps on a grid a
    //   quarter the area, and both depths locate the weight the criterion asks for to within about
    //   2e-4 at these counts -- so what separates them per plane leaves no trace in the
    //   noise-recovery figure R, in the retained rank, or in the denoised series, while the run time
    //   does. Reducing the budgets would re-open this, since the argument depends on both depths
    //   converging.
    //
    // No dependence on the plane's width was found, so this is a constant rather than a rule in
    //   Nx, Ny. Fitting the shallowest depth within 5% of the best achievable at each width against
    //   log2(width) gives a slope of -0.014 over 24 cells (residual rms 0.28) where the propagation
    //   argument above predicts +1.000, and within the cascade family the coarsest grid at that knee
    //   tracks the width at W/2 rather than sitting at a fixed size. Measured over in-plane widths
    //   64, 96 and 128 only: no complex acquisition wider than 128 was available, so nothing here
    //   licenses a different value above it.
    //
    // Descent stops at any level whose plane is under 4 voxels along either axis, so this is an
    //   upper bound and not an exact count; a plane too small to decimate at all is solved cold at
    //   native resolution whatever the value. Volumes refined from a prior estimate are solved at
    //   native resolution only and take no part.
    ssize_t cascade_depth = 2;
    // Where each coarsened level of a cold solve gets the noise level its discrepancy criterion is
    //   calibrated against, and hence which energy that level minimises.
    //
    // A cascade level helps the level above it only in so far as it poses a faithful coarsening of
    //   the same problem, and self-calibration does not, for a reason that is a property of the
    //   data rather than of the estimator: 2x2 box averaging halves the noise level but leaves the
    //   anatomy, so the structure-to-noise ratio doubles per level of decimation and the fraction
    //   of anatomical texture that any estimator built from local intensity differences counts as
    //   noise grows with it (see estimate_sigma_rms() in apc.cpp). Measured on 3T diffusion data,
    //   the level a coarsened grid self-calibrates against runs from 0.46 to 4.34 times the native
    //   one where a faithful coarsening would give half per level, and it is not even monotone in
    //   the level; the fidelity weight the criterion then selects there lands between 0.33 and 2.4
    //   times the weight the native problem's own criterion asks for. Each level is consequently
    //   the exact minimiser of a differently regularised energy rather than an approximate
    //   minimiser of the right one.
    //
    // RESTRICTED_MAP removes that by decimating the noise level map onto each level instead
    //   (Params::cascade_restriction chooses how), which also restores the spatially-varying
    //   weighting the native problem carries and which self-calibration replaces with uniform
    //   weights. It is available only where a map exists: the first pass of any schedule has none,
    //   and a level with no usable map self-calibrates whatever this is set to.
    //
    // COARSEST_ONLY additionally stops every level but the coarsest from searching for a weight of
    //   its own, which removes the whole lambda loop from those levels and leaves them only their
    //   polish sweeps. It differs from RESTRICTED_MAP only where more than one coarsened level
    //   exists, i.e. at cascade_depth 3 and above.
    cascade_lambda_t cascade_lambda = cascade_lambda_t::SELF_CALIBRATED;
    // How the noise level map is carried onto each coarsened grid. Consumed by
    //   cascade_lambda_t::RESTRICTED_MAP and cascade_lambda_t::COARSEST_ONLY only.
    cascade_restriction_t cascade_restriction = cascade_restriction_t::SIGMA_SQUARED_MEAN;
    // How a level's solution is carried to the finer level above it: as a unit-magnitude phase, or
    //   as the image iterate itself. Carrying the iterate hands the finer level a warm start for
    //   both the primal variable and its magnitude rather than for the phase alone, which is worth
    //   a clear margin wherever the finer level's solve is truncated -- but with the step sizes
    //   cp_step_ratio selects that level converges either way, and the two choices then differ by
    //   under 1% in cost and not at all in the phase returned.
    cascade_prolongation_t cascade_prolongation = cascade_prolongation_t::PHASE;
    // Factor by which the fidelity weight falls per level of refinement under
    //   cascade_lambda_t::COARSEST_ONLY, i.e. lambda(finer) = lambda(coarser) / this.
    //
    // Discretising the continuum energy at a spacing h gives a total variation of order
    //   h sum|delta u| against a fidelity term of order h^2 sum w (u-f)^2, so the discrete weight
    //   this solver uses carries one factor of h and the same continuum problem should be posed with
    //   a weight that doubles per level of coarsening. It is more than that in practice. Located
    //   exactly by bracketing and bisection on restricted 3T diffusion planes at 128, 64, 32 and 16
    //   voxels across, the realised factor is 2.47 in the median (fitted exponent 1.30 against the
    //   predicted 1.00), and it is ordered by contrast-to-noise: 2.4 on b = 5 planes rising to 4.1
    //   at b = 3000.
    //
    // The reason is that the derivation treats the two grids as discretisations of one continuum
    //   problem, which the data are not: 2x2 averaging resamples smooth structure but *halves the
    //   amplitude* of the noise, so the field the finite differences see is a different field at
    //   each level, and the more of the energy that noise accounts for the further the realised
    //   scaling departs from the smooth-field prediction. Since the departure therefore tracks
    //   contrast-to-noise, which varies by shell and by plane and which the solver cannot see, no
    //   single value of this is right everywhere; the measured spread is 2.2 to 3.1 over planes. In
    //   compensation the result is weakly sensitive to it -- over values from 1 to 4 the phase this
    //   estimator ends up with moves by about 10%.
    double cascade_lambda_ratio = 2.35;
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
    //   for.
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
    // Whether a warm-started solve begins its search for lambda where the previous pass over that
    //   same plane finished, instead of at the eq. 7 seed.
    //
    // The phase is already carried between passes; the fidelity weight is not, so every pass
    //   repeats the entire search from the seed even though the only input that changes between
    //   passes -- the noise level map -- moves the answer very little once it has settled. Carrying
    //   one scalar per plane starts each later pass at the previous answer, leaving only that drift
    //   to be tracked, and so needs far fewer lambda updates (and hence far fewer sweeps) to reach
    //   the same place. It offers nothing to the pass that first estimates a volume, which has
    //   nothing to carry.
    //
    // The state itself is owned by the caller and reaches the solver through the io_lambda argument
    //   of operator() / solve_plane(); it is per (serialised volume, slice), because that pair is
    //   what identifies a plane across passes under a temporally sub-sampling schedule. A plane
    //   with no stored value falls back to the eq. 7 seed, which is what a volume being estimated
    //   for the first time -- at whichever pass -- must do. Only native-resolution solves take part:
    //   the weight a coarse-grid bootstrap converges to is the answer to a different (decimated,
    //   self-calibrated) problem and is not on the native grid's scale.
    //
    // This is safe only because the inner solve is not truncated, and the distinction is not
    //   academic. Carrying makes each pass start from the previous pass's own answer, so a search
    //   that systematically stops short of the weight the criterion asks for compounds that
    //   shortfall from pass to pass instead of merely repeating it, and the phase drifts toward the
    //   truncated iteration's own attractor rather than toward the criterion's root: with the
    //   balanced step sizes and a looser lambda tolerance, that attractor sits near a quarter of
    //   the correct weight and the phase error grows monotonically with pass count. At the step
    //   sizes and tolerance configured here the attractor is the correct weight itself -- the
    //   search lands within 2e-4 of it, carrying or not -- and the answer is constant to four
    //   significant figures from the second pass onward (measured out to six passes) while costing
    //   12% fewer sweeps than repeating the search, for a phase a third further from the exact
    //   minimiser in the mean. Reducing the budgets, loosening lambda_tol or removing cp_step_ratio
    //   re-creates the drift.
    bool carry_lambda = true;
    // Geometric ratio by which lambda is stepped while searching for a bracket. Consumed by
    //   CONTINUATION only; BISECTION doubles or halves. Ratios close to one keep each inner solve
    //   a small perturbation of the last but need proportionally more steps to reach the crossing,
    //   which on these data lies a factor of two to sixteen below the eq. 7 seed.
    double lambda_step_factor = 1.5;
    // Multiplier applied to the starting estimate -- the eq. 7 seed, or a carried weight where one
    //   is available -- to place the start of the bracket search. One starts at that estimate
    //   itself, which for the eq. 7 seed on every plane measured here already lies on the
    //   high-fidelity side of the crossing, so the search walks downward from it.
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
    //   lambda_solver_t::FIXED_POINT only. With the step sizes cp_step_ratio selects this, rather
    //   than max_lambda_iter, is usually what ends the search: the fixed point meets it in a median
    //   of 14 of the 20 updates a warm solve is allowed, at a weight within about 2e-4 of the exact
    //   root, though the upper tail of the distribution does reach that bound. At the looser 1e-2
    //   the search stops a few updates earlier, leaving a phase some sixty times further from the
    //   exact minimiser of the same energy.
    double lambda_tol = 1e-4;
    // Ratio fixing the primal-dual step sizes of the inner Chambolle-Pock sweeps, as the
    //   dimensionless product tau * lambda. Zero (or negative) keeps the classical balanced choice
    //   tau = sigma = 1/||K||.
    //
    // Why the balanced choice is the wrong one here. Chambolle-Pock constrains only the *product*
    //   tau * sigma * ||K||^2 <= 1; their ratio is free, and it is what decides how fast each of the
    //   two variables moves. The primal variable u carries the intensity units of the data, whereas
    //   the dual variable p is confined to the unit ball, so tau and sigma are not dimensionless and
    //   tau = sigma = 1/||K|| tacitly assumes data of order one. Complex MRI data are not: on 3T
    //   diffusion data the plane's own values run into the hundreds. The dimensionless combination
    //   is tau * lambda, and the number of sweeps a solve needs is set by it: at the optimum the
    //   data term balances the regulariser, div p = lambda w (u - f), so the primal iterate closes a
    //   distance |u - f| at a speed tau |div p| = tau lambda w |u - f| per sweep, i.e. in about
    //   1 / (tau lambda w) sweeps regardless of the intensity scale. With tau pinned at 1/sqrt(8)
    //   and lambda of order 1e-3, that is several hundred sweeps of pure travel before the iterate
    //   even reaches its own minimiser -- which is most of what the fixed iteration budgets buy.
    //
    // Fixing tau * lambda instead makes the sweep count scale-free, and adapts tau as the lambda
    //   search moves: a smaller lambda means a solution further from the data, and proportionally
    //   larger primal steps to reach it. Measured against the exactly minimised problem on 3T
    //   diffusion planes, values between 0.01 and 0.05 reach a given phase accuracy in 6 to 29 times
    //   fewer sweeps than the balanced choice, on every plane tested and at every accuracy level;
    //   the optimum is shallow, and drifts from ~0.03 on warm-started 128x128 planes to ~0.015 on
    //   the cold 64x64 bootstrap, whose iterate must additionally propagate information across the
    //   plane and so cannot profit from larger steps beyond that point. Above ~0.1 the dual variable
    //   is starved in proportion and the gain reverses.
    //
    // This is a step-size choice, not a stopping rule: the solve converges to the same minimiser for
    //   any admissible ratio, so unlike the iteration budgets it does not co-determine the achieved
    //   regularisation except through how nearly the budget reaches that minimiser. The value
    //   configured here was chosen by searching {0, 0.01, 0.02, 0.03, 0.05} jointly with those
    //   budgets; over that search it is worth two to three orders of magnitude in the distance
    //   between the phase this estimator returns and the phase that exactly minimises the same
    //   energy, and it is what makes the budgets above sufficient rather than merely large.
    //
    // A side effect worth relying on: with tau fixed by tau * lambda, the whole solve becomes very
    //   nearly invariant to the intensity units of the data rather than merely insensitive to them.
    //   On real data scaled by 1e3 and by 1e-3 the estimated noise level tracks the scaling to
    //   within four parts in 1e5, even though the eq. 7 seed contains a term in 1/sigmabar^2 that is
    //   not dimensionally consistent and reaches 13% of the seed at the smaller scaling: the lambda
    //   search converges away from its seed, so the inconsistency costs iterations rather than
    //   accuracy, and what residual scale dependence remains is a difference in where the search
    //   stops rather than in the minimiser it is approaching.
    double cp_step_ratio = 0.05;
    // Whether the inner sweeps run Chambolle-Pock's Algorithm 2 -- one-sided acceleration, in which
    //   theta, tau and sigma are updated every sweep from the strong convexity modulus of the data
    //   term -- in place of Algorithm 1 at fixed step sizes and theta = 1.
    //
    // The data term (lambda/2) sum_x w(x) ||u - f||^2 is uniformly convex, with Euclidean modulus
    //   gamma = lambda * min_x w(x): the smallest weight anywhere on the plane is what bounds the
    //   curvature from below, and the per-plane minimum is used rather than the worst case
    //   1/weight_clamp the clamp permits, since a larger admissible modulus is strictly better for
    //   the scheme. Algorithm 2 then promises O(1/k^2) in place of O(1/k).
    //
    // It does not deliver here, and the reason is worth recording. The accelerated updates shrink
    //   tau and grow sigma by theta = 1/sqrt(1 + 2 gamma tau) per sweep, so the acceleration only
    //   engages once gamma * tau * k is of order one over the sweep count k. On these data the
    //   weights are bounded below by ~0.1 and lambda is of order 1e-3, making gamma ~ 1e-4: with the
    //   step sizes cp_step_ratio selects, the accelerated scheme reaches a given phase accuracy
    //   within a few percent of Algorithm 1 -- neither reliably better nor worse -- and with any
    //   larger modulus it is progressively worse, because the collapsing primal step costs more
    //   travel than the improved rate returns. It cannot, however, be unstable: theta multiplies tau
    //   and divides sigma, so tau * sigma * ||K||^2 is preserved exactly, and an over-large modulus
    //   stalls the iteration rather than diverging it.
    bool cp_accelerate = false;
    // Relative-change tolerance for early exit of the *inner* Chambolle-Pock sweeps
    //   themselves (both the lambda-loop sweeps and the polish sweeps), checked every sweep
    //   at negligible extra cost (the comparison piggybacks on the primal update's existing
    //   per-voxel pass, so it does not add another sweep over the plane). Zero (or a negative
    //   value) disables it and always spends the full budget.
    //
    // Whether it fires at all depends on the step sizes. With the classical balanced choice the
    //   iterate moves so little per sweep that it never fires -- over 99% of warm and 80-93% of
    //   cold solves consumed their whole budget, and loosening it to 1e-4 changed neither run time
    //   nor any output. With the step sizes cp_step_ratio selects the iterate moves far enough for
    //   the test to fire *within* the budget, and at this value it cuts roughly a third of the cold
    //   sweeps, at a cost of a factor of about 2.4 in the distance between the returned phase and
    //   the exact minimiser of the same energy. That trade is taken deliberately: the distance in
    //   question is four orders of magnitude smaller than the balanced steps reached at any budget,
    //   it leaves no trace in the noise-recovery figure R (see discrepancy_scale), and the saving is
    //   what keeps a converged solve at the cost of the truncated one. Do not loosen it further --
    //   since the budgets co-determine the achieved regularisation, anything that cuts more sweeps
    //   cuts smoothing, and this is not a speed knob.
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
    // Cold solve on the downsampled grid (the coarsest of Params::cascade_depth levels), its phase
    //   carried up to native. A bootstrap: correct only where a later pass will refine this volume
    //   at native resolution.
    COLD_COARSE,
    // Cold solve on the downsampled grid, immediately followed by a warm-started
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
  //   io_lambda : optional per-plane fidelity-weight state carried between passes (see
  //              Params::carry_lambda), or null for none. Must hold
  //              plan.size() * in.size(slice_axis()) entries, indexed
  //              [serialised volume index * number of slices + slice index], with zero meaning
  //              "nothing stored for this plane". Every native-resolution solve writes its
  //              converged weight here, and -- under Params::carry_lambda -- starts from whatever
  //              it finds. Distinct planes occupy distinct entries, so the threaded loop writes it
  //              without locking, exactly as it does the phase map.
  // Only instantiated for complex T (the phase is ill-defined for real data); callers gate
  //   with `if constexpr (is_complex<T>::value)`.
  template <typename T>
  void operator()(Image<T> &in,
                  Image<float> &sigma,
                  Image<cfloat> &io_phase,
                  Image<uint32_t> &serialise,
                  const std::vector<VolumePlan> &plan,
                  std::vector<double> *io_lambda = nullptr) const;

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
  //   io_lambda : optional caller-owned fidelity weight for this plane, or null. A positive
  //              incoming value replaces the eq. 7 seed as the starting point of the lambda search
  //              iff Params::carry_lambda is set; the converged weight is written back whenever
  //              the plane is solved, whether or not the incoming value was consumed. A declined
  //              plane leaves it untouched.
  //   io_magnitude : optional per-voxel magnitude accompanying the phase, of the plane's own size,
  //              or null. On a warm start it replaces |f| as the magnitude the initial image is
  //              built from, so that the caller can seed the solver with an image other than
  //              |f| * (incoming phase); on exit it is written with |u*|, the magnitude of the
  //              solved image whose argument the phase is. A declined plane leaves it untouched.
  //   fixed_lambda : consume *io_lambda as the fidelity weight and run no search for one at all,
  //              so that only the polish sweeps are spent. Requires a positive *io_lambda and is
  //              independent of Params::carry_lambda, which governs where a *search* starts.
  // Arrays are column-major (Nx rows = first in-slice axis, Ny cols = second).
  bool solve_plane(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                   const Eigen::Ref<const Eigen::ArrayXXd> &fi,
                   const Eigen::Ref<const Eigen::ArrayXXd> &sigma,
                   Eigen::Ref<Eigen::ArrayXXd> phase_r,
                   Eigen::Ref<Eigen::ArrayXXd> phase_i,
                   bool warm_start,
                   double *io_lambda = nullptr,
                   Eigen::ArrayXXd *io_magnitude = nullptr,
                   bool fixed_lambda = false) const;

  const std::vector<size_t> &axes() const { return inslice_axes; }

  // The spatial axis normal to the estimated planes: the one axis of 0, 1, 2 that is not an
  //   in-slice axis. With the serialised volume index it identifies a plane, and so indexes the
  //   per-plane state that operator()'s io_lambda argument carries between passes.
  size_t slice_axis() const;

private:
  std::vector<size_t> inslice_axes;
  Params params;
};

} // namespace MR::Denoise::Precondition
