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

#include "denoise/precondition/apc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "algo/loop.h"
#include "algo/threaded_loop.h"
#include "interp/cubic.h"
#include "math/math.h"
#include "mrtrix.h"
#include "transform.h"

namespace MR::Denoise::Precondition {

namespace {
// Global per-component noise standard deviation of a complex plane, estimated from the plane's
//   own finite differences. Used where no usable noise level map is available, to calibrate the
//   discrepancy criterion below.
//
// The functional. With uniform fidelity weights the criterion targets a residual energy of
//   Npix * sigmabar^2, which is the plane's noise energy sum_x sigma^2(x) only if sigmabar^2 is
//   the *mean* of sigma^2. So what has to be estimated is the root mean square of the per-voxel
//   noise level -- exactly what the noise-map branch of solve_plane() computes -- and not a robust
//   centre of it. The two are the same only for stationary noise: parallel-imaging amplification
//   makes sigma(x) strongly right-skewed (on 3T 64-channel diffusion data used to develop this,
//   sigma spans ~6 to ~400 within one slice, with an RMS four times its own median), and reading
//   the criterion off a robust centre then regularises several-fold too weakly.
//
// The estimator. The plane is partitioned into small blocks; a robust scale is estimated within
//   each; the blocks are then combined in quadrature, which is the mean of sigma^2 the noise map
//   branch forms. Within a block the noise level is near-constant, so the classical robust scale
//   applies (Immerkaer 1996; Donoho & Johnstone MAD): the difference of two i.i.d. N(0,sigma^2)
//   samples is N(0, 2*sigma^2), so the median of |differences| -- a robust proxy for the MAD, the
//   noise differences having ~zero median -- divided by 0.6745 = Phi^{-1}(0.75) and by sqrt(2)
//   recovers the per-component sigma, ignoring the sparse large differences of a tissue edge. A
//   plain root mean square of every difference in the plane estimates the same functional but has
//   no resistance to edges whatsoever, and a globally trimmed second moment is worse still: its
//   trimming threshold is set by the bulk of the distribution and discards the high-sigma minority
//   that the RMS is made of.
//
// Block size trades the two error sources against each other: large blocks span a varying noise
//   level, so the block's robust centre falls below its own RMS; small blocks make the median
//   itself noisy. Eight voxels yields 224 differences per block and was measured to be unbiased
//   on synthetic planes whose noise level varies over a realistic range.
//
// Limitation. Where anatomical texture within a block genuinely exceeds the noise -- low b-value
//   diffusion images, in which tissue contrast is strong and every local difference is
//   signal-dominated -- no estimator of this form can separate the two, and the noise level is
//   overestimated. That errs toward a smoother phase, which is the harmless direction: an
//   over-smoothed phase approaches the rigid-phase limit, whereas an under-smoothed one absorbs
//   noise into the phase and biases the noise level estimate down.
//
// Returns -1 if no usable differences are available (degenerate plane).
double estimate_sigma_rms(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                          const Eigen::Ref<const Eigen::ArrayXXd> &fi) {
  const ssize_t Nx = fr.rows();
  const ssize_t Ny = fr.cols();
  constexpr ssize_t block = 8;
  // Whole blocks only; any remainder is absorbed into the last block of each axis, so a block is
  //   between "block" and 2*block-1 voxels wide and a plane smaller than that is one block.
  const ssize_t nbx = std::max<ssize_t>(1, Nx / block);
  const ssize_t nby = std::max<ssize_t>(1, Ny / block);
  std::vector<double> d;
  std::vector<double> sigma_block;
  std::vector<ssize_t> npix_block;
  sigma_block.reserve(size_t(nbx * nby));
  npix_block.reserve(size_t(nbx * nby));
  for (ssize_t bj = 0; bj != nby; ++bj) {
    const ssize_t j0 = bj * block;
    const ssize_t j1 = (bj + 1 == nby) ? Ny : (bj + 1) * block;
    for (ssize_t bi = 0; bi != nbx; ++bi) {
      const ssize_t i0 = bi * block;
      const ssize_t i1 = (bi + 1 == nbx) ? Nx : (bi + 1) * block;
      d.clear();
      // Differences are taken within a block only: one that straddled a block boundary would
      //   couple two different local noise levels.
      for (ssize_t j = j0; j != j1; ++j)
        for (ssize_t i = i0; i != i1; ++i) {
          if (i + 1 != i1) {
            d.push_back(std::abs(fr(i + 1, j) - fr(i, j)));
            d.push_back(std::abs(fi(i + 1, j) - fi(i, j)));
          }
          if (j + 1 != j1) {
            d.push_back(std::abs(fr(i, j + 1) - fr(i, j)));
            d.push_back(std::abs(fi(i, j + 1) - fi(i, j)));
          }
        }
      if (d.empty())
        continue;
      const size_t mid = d.size() / 2;
      std::nth_element(d.begin(), d.begin() + mid, d.end());
      sigma_block.push_back(d[mid] / 0.6744897501960817 / Math::sqrt2);
      npix_block.push_back((i1 - i0) * (j1 - j0));
    }
  }
  if (sigma_block.empty())
    return -1.0;
  // Blocks carrying no measurable noise (a zero-padded or otherwise empty corner of the field of
  //   view) are excluded rather than averaged in as zeros, matching the noise map branch, which
  //   forms its mean over voxels whose level exceeds 1e-3 of the plane's maximum.
  const double floor = 1e-3 * (*std::max_element(sigma_block.begin(), sigma_block.end()));
  double sum_sigma2 = 0.0;
  ssize_t sum_npix = 0;
  for (size_t b = 0; b != sigma_block.size(); ++b) {
    if (!(sigma_block[b] > floor))
      continue;
    sum_sigma2 += double(npix_block[b]) * Math::pow2(sigma_block[b]);
    sum_npix += npix_block[b];
  }
  if (sum_npix == 0)
    return -1.0;
  return std::sqrt(sum_sigma2 / double(sum_npix));
}

// 2x box-downsample of a plane (2x2 average, clamped at the far edge for odd sizes). The first
//   iteration solves the phase on the resulting coarse grid: a background phase is low-bandwidth,
//   so decimation is near-lossless, at ~1/4 the voxel count (hence ~1/4 the per-sweep cost). The
//   coarse-grid noise level after 2x2 averaging is lower than native, but the first pass
//   self-calibrates its noise level from that same coarse data, so the discrepancy criterion stays
//   self-consistent. Exactly so, for the root mean square the self-calibration estimates: averaging
//   four independent voxels gives a coarse variance of one quarter their mean variance, so the RMS
//   over the coarse grid is exactly half the RMS over the native one. Averaging does not, however,
//   attenuate the anatomical structure that the estimate has to see past, so the coarse grid is the
//   harder of the two to self-calibrate on; the coarse solve is only a bootstrap the later native
//   passes refine.
void downsample2x(const Eigen::Ref<const Eigen::ArrayXXd> &src, Eigen::Ref<Eigen::ArrayXXd> dst) {
  const ssize_t Nx = src.rows();
  const ssize_t Ny = src.cols();
  const ssize_t cNx = dst.rows();
  const ssize_t cNy = dst.cols();
  for (ssize_t j = 0; j != cNy; ++j) {
    const ssize_t j0 = 2 * j;
    const ssize_t j1 = std::min(j0 + 1, Ny - 1);
    for (ssize_t i = 0; i != cNx; ++i) {
      const ssize_t i0 = 2 * i;
      const ssize_t i1 = std::min(i0 + 1, Nx - 1);
      dst(i, j) = 0.25 * (src(i0, j0) + src(i1, j0) + src(i0, j1) + src(i1, j1));
    }
  }
}

// Carry a per-voxel noise level onto the 2x-decimated grid the box-averaged data of downsample2x
//   live on, so that a coarsened grid can be solved against the same noise level map as the native
//   one instead of estimating a global level from its own (decimated) data.
//
// A coarse datum is the average of the four native voxels above it, so its own noise variance is
//   one sixteenth of their summed variance -- a quarter of their mean variance, i.e. half their RMS.
//   That factor is what keeps the criterion on one scale down the stack; the choice between the two
//   aggregations is what to do with variation *within* the block:
//     - sigma_squared_mean takes the noise level of the decimated datum itself, exactly as above;
//     - weight_mean takes the mean of the four fidelity weights (w = sigmabar^2/sigma^2, so the
//       mean of sigma^-2) and places it on the same absolute scale, preserving the block's total
//       weight rather than its noise level.
//   They differ only where sigma varies within a 2x2 block, i.e. at parallel-imaging g-factor
//   gradients. A coarse voxel is usable if any of its children is; voxels with no usable estimate
//   are excluded from the aggregate rather than counted as zero noise, and a coarse voxel with no
//   usable child is itself marked unusable.
void restrict_sigma(const Eigen::Ref<const Eigen::ArrayXXd> &src,
                    Eigen::Ref<Eigen::ArrayXXd> dst,
                    const bool weight_mean) {
  const ssize_t Nx = src.rows();
  const ssize_t Ny = src.cols();
  const ssize_t cNx = dst.rows();
  const ssize_t cNy = dst.cols();
  for (ssize_t j = 0; j != cNy; ++j) {
    const ssize_t j0 = 2 * j;
    const ssize_t j1 = std::min(j0 + 1, Ny - 1);
    for (ssize_t i = 0; i != cNx; ++i) {
      const ssize_t i0 = 2 * i;
      const ssize_t i1 = std::min(i0 + 1, Nx - 1);
      const double children[4] = {src(i0, j0), src(i1, j0), src(i0, j1), src(i1, j1)};
      double total = 0.0;
      ssize_t count = 0;
      for (const double s : children) {
        if (!std::isfinite(s) || !(s > 0.0))
          continue;
        total += weight_mean ? 1.0 / Math::pow2(s) : Math::pow2(s);
        ++count;
      }
      if (count == 0) {
        dst(i, j) = -1.0;
        continue;
      }
      const double mean = total / double(count);
      dst(i, j) = weight_mean ? 0.5 / std::sqrt(mean) : 0.5 * std::sqrt(mean);
    }
  }
}

// Bilinearly upsample a coarse unit-magnitude phase back to the native grid and renormalise to
//   unit magnitude. Interpolating the (cos, sin) = (phase_r, phase_i) channels (rather than the
//   wrapped angle) avoids phase-wrap artefacts; renormalisation restores |phase| == 1 as the
//   demodulator requires. Cell-centred mapping is the inverse of downsample2x (coarse voxel I
//   spans native {2I, 2I+1}, centre at native 2I+0.5).
void upsample_phase(const Eigen::Ref<const Eigen::ArrayXXd> &pr_c,
                    const Eigen::Ref<const Eigen::ArrayXXd> &pi_c,
                    Eigen::Ref<Eigen::ArrayXXd> pr,
                    Eigen::Ref<Eigen::ArrayXXd> pi) {
  const ssize_t Nx = pr.rows();
  const ssize_t Ny = pr.cols();
  const ssize_t cNx = pr_c.rows();
  const ssize_t cNy = pr_c.cols();
  for (ssize_t j = 0; j != Ny; ++j) {
    const double cy = std::min(std::max((double(j) - 0.5) * 0.5, 0.0), double(cNy - 1));
    const ssize_t J0 = std::min(ssize_t(std::floor(cy)), cNy - 1);
    const ssize_t J1 = std::min(J0 + 1, cNy - 1);
    const double fy = cy - double(J0);
    for (ssize_t i = 0; i != Nx; ++i) {
      const double cx = std::min(std::max((double(i) - 0.5) * 0.5, 0.0), double(cNx - 1));
      const ssize_t I0 = std::min(ssize_t(std::floor(cx)), cNx - 1);
      const ssize_t I1 = std::min(I0 + 1, cNx - 1);
      const double fx = cx - double(I0);
      const double w00 = (1.0 - fx) * (1.0 - fy);
      const double w10 = fx * (1.0 - fy);
      const double w01 = (1.0 - fx) * fy;
      const double w11 = fx * fy;
      const double r = w00 * pr_c(I0, J0) + w10 * pr_c(I1, J0) + w01 * pr_c(I0, J1) + w11 * pr_c(I1, J1);
      const double im = w00 * pi_c(I0, J0) + w10 * pi_c(I1, J0) + w01 * pi_c(I0, J1) + w11 * pi_c(I1, J1);
      const double mag = std::hypot(r, im);
      if (mag > std::numeric_limits<double>::min() * 1e3) {
        pr(i, j) = r / mag;
        pi(i, j) = im / mag;
      } else {
        pr(i, j) = 1.0;
        pi(i, j) = 0.0;
      }
    }
  }
}

// Bilinearly upsample the two channels of the complex image iterate u = magnitude * (phase_r,
//   phase_i) rather than its phase alone, splitting the interpolated vector back into a magnitude
//   and a unit-magnitude phase for the caller. Interpolation weights and cell-centred mapping are
//   those of upsample_phase.
//
// The difference from upsample_phase is what happens to the magnitude the coarse level settled on:
//   here it is carried up, so the finer level starts from a smoothed image, whereas prolongating
//   the phase alone discards it and the finer level rebuilds its image from its own (unsmoothed)
//   data magnitude. The interpolated vector is also not renormalised before it is split, so where
//   the phase turns quickly within a coarse cell the interpolated magnitude falls, which is what
//   bilinear interpolation of the iterate means and is a further difference from interpolating the
//   phase channels alone.
void upsample_iterate(const Eigen::Ref<const Eigen::ArrayXXd> &pr_c,
                      const Eigen::Ref<const Eigen::ArrayXXd> &pi_c,
                      const Eigen::Ref<const Eigen::ArrayXXd> &mag_c,
                      Eigen::Ref<Eigen::ArrayXXd> pr,
                      Eigen::Ref<Eigen::ArrayXXd> pi,
                      Eigen::Ref<Eigen::ArrayXXd> mag) {
  const ssize_t Nx = pr.rows();
  const ssize_t Ny = pr.cols();
  const ssize_t cNx = pr_c.rows();
  const ssize_t cNy = pr_c.cols();
  for (ssize_t j = 0; j != Ny; ++j) {
    const double cy = std::min(std::max((double(j) - 0.5) * 0.5, 0.0), double(cNy - 1));
    const ssize_t J0 = std::min(ssize_t(std::floor(cy)), cNy - 1);
    const ssize_t J1 = std::min(J0 + 1, cNy - 1);
    const double fy = cy - double(J0);
    for (ssize_t i = 0; i != Nx; ++i) {
      const double cx = std::min(std::max((double(i) - 0.5) * 0.5, 0.0), double(cNx - 1));
      const ssize_t I0 = std::min(ssize_t(std::floor(cx)), cNx - 1);
      const ssize_t I1 = std::min(I0 + 1, cNx - 1);
      const double fx = cx - double(I0);
      const double w00 = (1.0 - fx) * (1.0 - fy);
      const double w10 = fx * (1.0 - fy);
      const double w01 = (1.0 - fx) * fy;
      const double w11 = fx * fy;
      const double r = w00 * mag_c(I0, J0) * pr_c(I0, J0) + w10 * mag_c(I1, J0) * pr_c(I1, J0)    //
                       + w01 * mag_c(I0, J1) * pr_c(I0, J1) + w11 * mag_c(I1, J1) * pr_c(I1, J1); //
      const double im = w00 * mag_c(I0, J0) * pi_c(I0, J0) + w10 * mag_c(I1, J0) * pi_c(I1, J0)    //
                        + w01 * mag_c(I0, J1) * pi_c(I0, J1) + w11 * mag_c(I1, J1) * pi_c(I1, J1); //
      const double length = std::hypot(r, im);
      mag(i, j) = length;
      if (length > std::numeric_limits<double>::min() * 1e3) {
        pr(i, j) = r / length;
        pi(i, j) = im / length;
      } else {
        pr(i, j) = 1.0;
        pi(i, j) = 0.0;
      }
    }
  }
}

#ifdef DWIDENOISE2_APC_PROFILE
const char *lambda_solver_name(const AdaptivePhaseEstimator::lambda_solver_t solver) {
  switch (solver) {
  case AdaptivePhaseEstimator::lambda_solver_t::FIXED_POINT:
    return "fixed_point";
  case AdaptivePhaseEstimator::lambda_solver_t::BISECTION:
    return "bisection";
  case AdaptivePhaseEstimator::lambda_solver_t::CONTINUATION:
    return "continuation";
  }
  return "unknown";
}

const char *regime_name(const AdaptivePhaseEstimator::regime_t regime) {
  switch (regime) {
  case AdaptivePhaseEstimator::regime_t::WARM:
    return "warm";
  case AdaptivePhaseEstimator::regime_t::COLD_COARSE:
    return "cold_coarse";
  case AdaptivePhaseEstimator::regime_t::COLD_COARSE_THEN_WARM:
    return "cold_coarse_then_warm";
  }
  return "unknown";
}
#endif
} // namespace

size_t AdaptivePhaseEstimator::slice_axis() const {
  assert(inslice_axes.size() == 2);
  for (size_t axis = 0; axis != 3; ++axis)
    if (std::find(inslice_axes.begin(), inslice_axes.end(), axis) == inslice_axes.end())
      return axis;
  assert(false);
  return 2;
}

bool AdaptivePhaseEstimator::solve_plane(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                                 const Eigen::Ref<const Eigen::ArrayXXd> &fi,
                                 const Eigen::Ref<const Eigen::ArrayXXd> &sigma,
                                 Eigen::Ref<Eigen::ArrayXXd> phase_r,
                                 Eigen::Ref<Eigen::ArrayXXd> phase_i,
                                 const bool warm_start,
                                 double *const io_lambda,
                                 Eigen::ArrayXXd *const io_magnitude,
                                 const bool fixed_lambda) const {
  const ssize_t Nx = fr.rows();
  const ssize_t Ny = fr.cols();
  const ssize_t Npix = Nx * Ny;
  if (Npix == 0)
    return false;

  // --- Usable domain, per-slice noise statistics, and fidelity weights ------------------
  // Two regimes, both feeding the identical solver below:
  //  (a) spatially-varying: enough voxels carry a usable local noise estimate. Each is
  //      weighted by w = sigmabar^2 / sigma^2 (Pizzolato eq. 13) and the discrepancy criterion
  //      is calibrated over those voxels; voxels without an estimate (background, where the
  //      interpolated sigma collapses toward zero) are still smoothed, with neutral unit
  //      weight, but excluded from the statistics -- so the returned phase is defined
  //      everywhere.
  //  (b) self-calibrated: too few (or, on the first iteration, no) local estimates. A single
  //      global per-component noise level is estimated from the data (blockwise-robust root mean
  //      square of finite differences, on the same functional case (a) forms from the map) and
  //      uniform weighting is applied over the whole slice. This is what lets APC run from the
  //      first iteration without a Hann bootstrap.
  double sigma_max = 0.0;
  for (ssize_t j = 0; j != Ny; ++j)
    for (ssize_t i = 0; i != Nx; ++i) {
      const double s = sigma(i, j);
      if (std::isfinite(s) && s > sigma_max)
        sigma_max = s;
    }
  const double sigma_floor = 1e-3 * sigma_max; // excludes true-zero / background voxels

  ssize_t n_known = 0;
  double sum_sigma2_known = 0.0;
  if (sigma_max > 0.0) {
    for (ssize_t j = 0; j != Ny; ++j)
      for (ssize_t i = 0; i != Nx; ++i) {
        const double s = sigma(i, j);
        if (std::isfinite(s) && s > sigma_floor) {
          ++n_known;
          sum_sigma2_known += Math::pow2(s);
        }
      }
  }
  const bool spatially_varying =
      n_known >= 8 && double(n_known) >= params.min_domain_fraction * double(Npix);

  Eigen::ArrayXXd indom(Nx, Ny); // 1.0 = counted in the discrepancy statistics, 0.0 = not
  Eigen::ArrayXXd w(Nx, Ny);     // per-voxel fidelity weight
  double sigmabar2;
  ssize_t ndomain;
  if (spatially_varying) {
    sigmabar2 = sum_sigma2_known / double(n_known); // global per-component variance
    ndomain = n_known;
    const double w_lo = 1.0 / params.weight_clamp;
    const double w_hi = params.weight_clamp;
    for (ssize_t j = 0; j != Ny; ++j)
      for (ssize_t i = 0; i != Nx; ++i) {
        const double s = sigma(i, j);
        if (std::isfinite(s) && s > sigma_floor) {
          indom(i, j) = 1.0;
          w(i, j) = std::min(w_hi, std::max(w_lo, sigmabar2 / Math::pow2(s)));
        } else {
          indom(i, j) = 0.0; // still smoothed, but not part of the calibration
          w(i, j) = 1.0;
        }
      }
  } else {
    const double sigma_hat = estimate_sigma_rms(fr, fi);
    if (!(sigma_hat > 0.0))
      return false; // fully degenerate slice (constant / empty): keep the incoming phase
    sigmabar2 = Math::pow2(sigma_hat);
    ndomain = Npix;
    indom.setConstant(1.0);
    w.setConstant(1.0); // uniform weighting (no spatial noise information)
  }
  const double sigmabar = std::sqrt(sigmabar2);
  if (!(sigmabar > 0.0))
    return false;

  // --- Iteration budget (reduced once warm-started) -------------------------------------
  const ssize_t n_cp = warm_start ? params.cp_iter_warm : params.cp_iter;
  const ssize_t n_lambda = warm_start ? params.max_lambda_iter_warm : params.max_lambda_iter;
  const ssize_t n_polish = warm_start ? params.polish_iter_warm : params.polish_iter;

  // --- Chambolle-Pock primal-dual state -------------------------------------------------
  // Primal image u = (ur, ui); over-relaxed iterate ubar; dual variable p = (pxr, pyr, pxi, pyi)
  //   = the (per channel, per direction) gradient field. Cold start (first pass): seed u at the
  //   data f. Warm start (later passes): seed u at |f| * (incoming phase) -- same magnitude as f
  //   but carrying the already-smooth previous phase, so the expensive phase smoothing is
  //   ~converged and only the cheap magnitude smoothing remains, matching the reduced budget.
  Eigen::ArrayXXd ur, ui;
  if (warm_start) {
    // The magnitude the warm start's image is built from: the data's own, unless the caller has
    //   supplied one (an image carried up from a coarser grid, whose magnitude is already smoothed).
    const bool magnitude_supplied =
        io_magnitude != nullptr && io_magnitude->rows() == Nx && io_magnitude->cols() == Ny;
    const Eigen::ArrayXXd mag = magnitude_supplied ? *io_magnitude : (fr.square() + fi.square()).sqrt();
    ur = mag * phase_r;
    ui = mag * phase_i;
  } else {
    ur = fr;
    ui = fi;
  }
  Eigen::ArrayXXd ubr = ur, ubi = ui;
  Eigen::ArrayXXd pxr = Eigen::ArrayXXd::Zero(Nx, Ny);
  Eigen::ArrayXXd pyr = Eigen::ArrayXXd::Zero(Nx, Ny);
  Eigen::ArrayXXd pxi = Eigen::ArrayXXd::Zero(Nx, Ny);
  Eigen::ArrayXXd pyi = Eigen::ArrayXXd::Zero(Nx, Ny);

  // For the 2-D forward-difference gradient, ||grad||^2 <= 8; the coupled two-channel operator has
  //   the same norm. Every step-size pair below satisfies tau * sigma_cp * ||K||^2 = 1, the limiting
  //   stable choice, and differs only in how it divides that product between the two variables
  //   (Params::cp_step_ratio).
  const double L2 = 8.0;
  // The strong convexity modulus of the data term, for the accelerated scheme: the smallest weight
  //   anywhere on the plane bounds its curvature from below. Formed here rather than per sweep
  //   because only lambda changes between calls.
  const double w_min = params.cp_accelerate ? w.minCoeff() : 0.0;

  // One primal-dual sweep at fidelity weight "lambda". Warm-starts from the current state, and
  //   stops early once the relative per-sweep change in u falls below params.cp_tol (data-driven
  //   early exit; see the DWIDENOISE2_APC_PROFILE discussion in apc.h). That
  //   sum_diff2/sum_old2 comparison is the termination criterion itself -- it stays unconditional
  //   -- and piggybacks on the primal update's existing per-voxel pass, so it adds no extra sweep
  //   over the plane. The *sweep count itself* (returned only under DWIDENOISE2_APC_PROFILE) is
  //   purely diagnostic and is compiled out otherwise, so that a non-profile build's execution
  //   time reflects only the termination criterion, not the bookkeeping used to report it.
#ifdef DWIDENOISE2_APC_PROFILE
  auto run_cp = [&](const ssize_t niter, const double lambda) -> ssize_t {
    ssize_t used = 0;
#else
  auto run_cp = [&](const ssize_t niter, const double lambda) {
#endif
    // Step sizes for this call. tau carries the intensity units of the primal image while the dual
    //   variable is confined to the unit ball, so the classical balanced pair tau = sigma_cp =
    //   1/||K|| is a choice about the data's scale rather than a neutral one; where
    //   Params::cp_step_ratio asks for it, the dimensionless product tau * lambda is fixed instead,
    //   which both removes that scale dependence and re-balances the pair as the lambda search
    //   moves.
    double tau = 1.0 / std::sqrt(L2);
    double sigma_cp = 1.0 / std::sqrt(L2);
    if (params.cp_step_ratio > 0.0) {
      const double scaled = params.cp_step_ratio / lambda;
      if (std::isfinite(scaled) && scaled > 0.0) {
        tau = scaled;
        sigma_cp = 1.0 / (L2 * tau);
      }
    }
    // Chambolle-Pock Algorithm 2 restarts its step-size sequence for each call, since a call is a
    //   fresh sequence at a fresh lambda, and hence at a fresh modulus.
    const double gamma = params.cp_accelerate ? lambda * w_min : 0.0;
    for (ssize_t iter = 0; iter != niter; ++iter) {
      // Over-relaxation factor. Algorithm 1 holds it at 1; Algorithm 2 derives it from the step
      //   size this sweep uses, then applies it to both step sizes for the next, which leaves their
      //   product -- and so the stability condition -- exactly unchanged.
      const double theta = params.cp_accelerate ? 1.0 / std::sqrt(1.0 + 2.0 * gamma * tau) : 1.0;
#ifdef DWIDENOISE2_APC_PROFILE
      ++used;
#endif
      // Dual update: p <- proj_{||.||<=1} ( p + sigma_cp * grad(ubar) ), with the vectorial
      //   (channel-coupled) projection: divide the whole (r,i)x(x,y) 4-vector at each voxel
      //   by max(1, its joint L2 norm). Forward differences; Neumann boundary (zero gradient
      //   past the far edge), so p on the last row/column stays zero.
      for (ssize_t j = 0; j != Ny; ++j) {
        for (ssize_t i = 0; i != Nx; ++i) {
          const double gxr = (i + 1 < Nx) ? (ubr(i + 1, j) - ubr(i, j)) : 0.0;
          const double gyr = (j + 1 < Ny) ? (ubr(i, j + 1) - ubr(i, j)) : 0.0;
          const double gxi = (i + 1 < Nx) ? (ubi(i + 1, j) - ubi(i, j)) : 0.0;
          const double gyi = (j + 1 < Ny) ? (ubi(i, j + 1) - ubi(i, j)) : 0.0;
          const double qxr = pxr(i, j) + sigma_cp * gxr;
          const double qyr = pyr(i, j) + sigma_cp * gyr;
          const double qxi = pxi(i, j) + sigma_cp * gxi;
          const double qyi = pyi(i, j) + sigma_cp * gyi;
          const double norm = std::sqrt(Math::pow2(qxr) + Math::pow2(qyr) + Math::pow2(qxi) + Math::pow2(qyi));
          const double inv = 1.0 / std::max(1.0, norm);
          pxr(i, j) = qxr * inv;
          pyr(i, j) = qyr * inv;
          pxi(i, j) = qxi * inv;
          pyi(i, j) = qyi * inv;
        }
      }
      // Primal update: u <- prox_{tau G} ( u + tau * div(p) ). div is the negative adjoint of
      //   the forward-difference gradient (backward differences). The proximal step of the
      //   weighted quadratic data term G(u) = (lambda/2) sum_x w(x) ||u - f||^2 is pointwise:
      //   u = ( v + tau*lambda*w*f ) / ( 1 + tau*lambda*w ), v = u + tau*div(p). Over-relaxed
      //   ubar = u_new + theta*(u_new - u_old), which at theta = 1 is written in its exact
      //   2*u_new - u_old form (the two are not bit-identical in floating point). Each voxel is
      //   independent, so the in-place overwrite of u is safe.
      // Relative per-sweep change in u, accumulated alongside the update itself (no extra pass):
      //   the data-driven early-exit test below compares sum_diff2 against sum_old2.
      double sum_diff2 = 0.0;
      double sum_old2 = 0.0;
      for (ssize_t j = 0; j != Ny; ++j) {
        for (ssize_t i = 0; i != Nx; ++i) {
          // divergence, real channel
          const double dxr = (i == 0)          ? pxr(0, j)
                             : (i == Nx - 1)   ? -pxr(Nx - 2, j)
                                               : (pxr(i, j) - pxr(i - 1, j));
          const double dyr = (j == 0)          ? pyr(i, 0)
                             : (j == Ny - 1)   ? -pyr(i, Ny - 2)
                                               : (pyr(i, j) - pyr(i, j - 1));
          // divergence, imaginary channel
          const double dxi = (i == 0)          ? pxi(0, j)
                             : (i == Nx - 1)   ? -pxi(Nx - 2, j)
                                               : (pxi(i, j) - pxi(i - 1, j));
          const double dyi = (j == 0)          ? pyi(i, 0)
                             : (j == Ny - 1)   ? -pyi(i, Ny - 2)
                                               : (pyi(i, j) - pyi(i, j - 1));
          const double wl = lambda * w(i, j);
          const double denom = 1.0 + tau * wl;
          const double vr = ur(i, j) + tau * (dxr + dyr);
          const double vi = ui(i, j) + tau * (dxi + dyi);
          const double unewr = (vr + tau * wl * fr(i, j)) / denom;
          const double unewi = (vi + tau * wl * fi(i, j)) / denom;
          const double oldr = ur(i, j);
          const double oldi = ui(i, j);
          sum_diff2 += Math::pow2(unewr - oldr) + Math::pow2(unewi - oldi);
          sum_old2 += Math::pow2(oldr) + Math::pow2(oldi);
          if (params.cp_accelerate) {
            ubr(i, j) = unewr + theta * (unewr - oldr);
            ubi(i, j) = unewi + theta * (unewi - oldi);
          } else {
            ubr(i, j) = 2.0 * unewr - oldr;
            ubi(i, j) = 2.0 * unewi - oldi;
          }
          ur(i, j) = unewr;
          ui(i, j) = unewi;
        }
      }
      if (params.cp_accelerate) {
        tau *= theta;
        sigma_cp /= theta;
      }
      if (params.cp_tol > 0.0 && sum_old2 > 0.0 && sum_diff2 < Math::pow2(params.cp_tol) * sum_old2)
        break;
    }
#ifdef DWIDENOISE2_APC_PROFILE
    return used;
#endif
  };

  // --- Discrepancy-criterion (Morozov) selection of lambda ------------------------------
  // Seeded by Chambolle (2004) eq. 7 and targeting a weighted residual energy of
  //   Rr^2 + Ri^2 = 2 * ndomain * sigmabar_lambda^2 (spatially-varying form, their eq. 11). Image
  //   and lambda are refined jointly: the solver is warm-started across lambda updates, so each
  //   update's sweeps continue the previous one's rather than restarting. Which search is used --
  //   their eq. 6 multiplicative update, or one of two bracketing searches for the same condition
  //   -- is Params::lambda_solver, documented in apc.h.
  // The noise level driving the seed and the target is scaled by params.discrepancy_scale, so the
  //   residual is targeted above the noise energy the criterion nominally prescribes; the fidelity
  //   weights w computed above deliberately do not see that factor (they are scale-invariant, and
  //   it is the phase, not the image, that the extra smoothing is for). See apc.h.
  const double sigmabar_lambda = params.discrepancy_scale * sigmabar;
  const double sigmabar2_lambda = Math::pow2(sigmabar_lambda);
  const double lambda_seed = 2.1237 / sigmabar_lambda + 0.0547 / sigmabar2_lambda; // eq. 7
  // Where the search for lambda begins. The eq. 7 seed is a function of the noise level alone and
  //   knows nothing of the plane; where the caller has kept what a previous pass over this same
  //   plane converged to, that is a far better starting point, and the search then only has to
  //   track how the criterion's answer has moved as the noise map settled (Params::carry_lambda).
  const bool weight_supplied = io_lambda != nullptr && std::isfinite(*io_lambda) && *io_lambda > 0.0;
  // A caller that already knows the weight this plane is to be solved at -- one level of a
  //   resolution cascade taking the weight of the level below it -- pins it instead, and no search
  //   runs at all: the budget of this call is then its polish sweeps alone.
  const bool pinned = fixed_lambda && weight_supplied;
  const bool carried = params.carry_lambda && weight_supplied && !pinned;
  const double lambda_start = (carried || pinned) ? *io_lambda : lambda_seed;
  double lambda = lambda_start;
  const double target_scale = 2.0 * std::sqrt(double(ndomain)) * sigmabar_lambda;
  // The weighted residual over the calibration domain, summed over the two channels: the quantity
  //   the discrepancy criterion matches against target_scale. Evaluated at whatever image the
  //   sweeps above have reached, so it inherits their truncation.
  auto residual_sum = [&]() {
    const double Rr = std::sqrt((indom * w * (ur - fr).square()).sum());
    const double Ri = std::sqrt((indom * w * (ui - fi).square()).sum());
    return Rr + Ri;
  };
  // Bookkeeping for the DWIDENOISE2_APC_PROFILE DEBUG() call below only -- entirely compiled out
  //   otherwise, so a non-profile build's timing reflects only run_cp's cp_tol early exit, not
  //   the cost of reporting it.
#ifdef DWIDENOISE2_APC_PROFILE
  ssize_t lambda_iters_used = 0;
  ssize_t cp_sweeps_used = 0;
  ssize_t bracket_steps_used = 0;
#endif
  if (pinned) {
    // The weight is the caller's; go straight to the polish.
  } else if (params.lambda_solver == lambda_solver_t::FIXED_POINT) {
    for (ssize_t li = 0; li != n_lambda; ++li) {
#ifdef DWIDENOISE2_APC_PROFILE
      lambda_iters_used = li + 1;
      cp_sweeps_used +=
#endif
          run_cp(n_cp, lambda);
      const double lambda_new = lambda * residual_sum() / target_scale;
      if (!std::isfinite(lambda_new) || lambda_new <= 0.0)
        break;
      const double rel = std::abs(lambda_new - lambda) / lambda;
      lambda = lambda_new;
      if (rel < params.lambda_tol)
        break;
    }
  } else {
    // Bracketing search for the same condition. residual_sum() - target_scale is decreasing in
    //   lambda with a single sign change (see Params::lambda_solver), so a pair of lambda at which
    //   it takes opposite signs traps the crossing, and every subsequent evaluation halves the
    //   interval in log lambda. The primal-dual state is deliberately *not* reset between
    //   evaluations: each inner solve resumes from the image and dual variable the previous lambda
    //   left, which is what keeps the total sweep count of a search comparable to a handful of
    //   solves rather than one per evaluation.
    auto excess_at = [&](const double value) {
#ifdef DWIDENOISE2_APC_PROFILE
      ++lambda_iters_used;
      cp_sweeps_used +=
#endif
          run_cp(n_cp, value);
      return residual_sum() - target_scale;
    };
    // BISECTION halves or doubles; CONTINUATION walks by the configured (smaller) ratio.
    const double step = params.lambda_solver == lambda_solver_t::CONTINUATION
                            ? std::max(1.0 + 1e-9, params.lambda_step_factor)
                            : 2.0;
    lambda = lambda_start * params.lambda_start_scale;
    double excess = excess_at(lambda);
    // Invariant maintained from here on: excess(lambda_lo) >= 0 >= excess(lambda_hi).
    double lambda_lo = lambda;
    double lambda_hi = lambda;
    bool bracketed = (excess == 0.0);
    // A residual below target means too little smoothing has been asked for, so the crossing lies
    //   at a *smaller* lambda; above target, at a larger one.
    const bool downward = (excess < 0.0);
    for (ssize_t k = 0; !bracketed && k != params.lambda_bracket_cap; ++k) {
      const double previous = lambda;
      lambda = downward ? (lambda / step) : (lambda * step);
      if (!std::isfinite(lambda) || !(lambda > 0.0))
        break;
#ifdef DWIDENOISE2_APC_PROFILE
      ++bracket_steps_used;
#endif
      if (downward ? (excess_at(lambda) >= 0.0) : (excess_at(lambda) <= 0.0)) {
        lambda_lo = downward ? lambda : previous;
        lambda_hi = downward ? previous : lambda;
        bracketed = true;
      }
    }
    if (!bracketed) {
      WARN("adaptive phase correction: the discrepancy criterion is met at no fidelity weight "
           "within " +
           str(params.lambda_bracket_cap) + " geometric steps of the initial estimate for a " + str(Nx) + "x" +
           str(Ny) + " plane; that plane's incoming phase is left unchanged");
      return false;
    }
    // Geometric (log-lambda) bisection of the bracket.
    for (ssize_t li = 0; li != n_lambda && 0.5 * std::log(lambda_hi / lambda_lo) > params.lambda_root_tol; ++li) {
      const double middle = std::sqrt(lambda_lo * lambda_hi);
      if (excess_at(middle) >= 0.0)
        lambda_lo = middle;
      else
        lambda_hi = middle;
    }
    lambda = std::sqrt(lambda_lo * lambda_hi);
  }

  // Settle the image at the converged lambda before extracting its argument.
#ifdef DWIDENOISE2_APC_PROFILE
  const ssize_t polish_sweeps_used = run_cp(n_polish, lambda);
  cp_sweeps_used += polish_sweeps_used;
  DEBUG("APC solve_plane: " + str(Nx) + "x" + str(Ny) +                              //
        " warm=" + str(warm_start) +                                                 //
        " spatially_varying=" + str(spatially_varying) +                             //
        " ndomain=" + str(ndomain) + "/" + str(Npix) +                               //
        " sigmabar=" + str(sigmabar) +                                               //
        " lambda_solver=" + lambda_solver_name(params.lambda_solver) +               //
        " lambda_carried=" + str(carried ? 1 : 0) +                                  //
        " lambda_start=" + str(lambda_start) +                                       //
        " bracket_steps=" + str(bracket_steps_used) +                                //
        " lambda_iters=" + str(lambda_iters_used) + "/" + str(n_lambda) +            //
        " cp_sweeps=" + str(cp_sweeps_used) + "/" + str(n_lambda * n_cp + n_polish) + //
        " polish_sweeps=" + str(polish_sweeps_used) + "/" + str(n_polish) +          //
        " final_lambda=" + str(lambda));
#else
  run_cp(n_polish, lambda);
#endif

  // Hand the converged weight back for the next pass over this plane to start from. Written
  //   unconditionally, so that enabling Params::carry_lambda is all that is needed for a later pass
  //   to consume it; the early returns above deliberately leave it alone, a declined plane having
  //   converged to nothing.
  if (io_lambda != nullptr)
    *io_lambda = lambda;

  // --- Extract unit-magnitude phase -----------------------------------------------------
  // The magnitude is handed back too where the caller asked for it, so that the solved image can be
  //   reconstructed rather than only its argument.
  if (io_magnitude != nullptr)
    io_magnitude->resize(Nx, Ny);
  for (ssize_t j = 0; j != Ny; ++j)
    for (ssize_t i = 0; i != Nx; ++i) {
      const double mag = std::hypot(ur(i, j), ui(i, j));
      if (io_magnitude != nullptr)
        (*io_magnitude)(i, j) = mag;
      if (mag > std::numeric_limits<double>::min() * 1e3) {
        phase_r(i, j) = ur(i, j) / mag;
        phase_i(i, j) = ui(i, j) / mag;
      } else {
        // Degenerate (image magnitude vanished): a unit real phase is a harmless no-op
        //   demodulator for that voxel.
        phase_r(i, j) = 1.0;
        phase_i(i, j) = 0.0;
      }
    }
  return true;
}

namespace {

// Per-slice APC functor for ThreadedLoop::run_outer. ThreadedLoop copies the functor per
//   thread, so each thread gets independent Image voxel accessors and its own scratch Eigen
//   planes (deep-copied). Reads only from "in"/"sigma" and writes a disjoint plane of
//   "phase", so no locking is required. See mrdegibbs's Unring2DFunctor for the same idiom.
// The cold/warm and native/downsampled regime is not a property of the pass but of the volume
//   the current plane belongs to (see AdaptivePhaseEstimator's class comment): each outer
//   position's volume is mapped to its serialised index and its VolumePlan looked up.
template <typename T> class APCFunctor {
public:
  APCFunctor(const AdaptivePhaseEstimator &estimator,
             const std::vector<size_t> &outer_axes,
             const std::vector<size_t> &inner_axes,
             Image<T> &in,
             Image<float> &sigma,
             Image<cfloat> &phase,
             Image<uint32_t> &serialise,
             const std::vector<AdaptivePhaseEstimator::VolumePlan> &plan,
             std::vector<double> *lambda_store,
             const AdaptivePhaseEstimator::Params &params)
      : estimator(&estimator),
        params(params),
        outer_axes(outer_axes),
        inner_axes(inner_axes),
        in(in),
        sigma(sigma),
        phase(phase),
        serialise(serialise),
        plan(&plan),
        lambda_store(lambda_store),
        slice_axis(estimator.slice_axis()),
        num_slices(in.size(estimator.slice_axis())),
        have_sigma(sigma.valid()),
        transform(in),
        Nx(in.size(inner_axes[0])),
        Ny(in.size(inner_axes[1])),
        fr(Nx, Ny),
        fi(Nx, Ny),
        sig(Nx, Ny),
        phase_r(Nx, Ny),
        phase_i(Nx, Ny) {
    // Coarsened-grid scratch, allocated once per thread if *any* planned volume is solved cold
    //   (every cold solve begins on a downsampled grid with its phase carried back up: a background
    //   phase is smooth, so decimation is near-lossless at ~1/4 the cost per level). Every level's
    //   arrays are sized here and reused by every plane this thread visits, so a plane's solve
    //   allocates nothing. Volumes with an existing estimate are refined natively and never need
    //   any of it.
    const bool any_coarse =
        std::any_of(plan.begin(), plan.end(), [](const AdaptivePhaseEstimator::VolumePlan &vp) {   //
          return vp.estimate && vp.regime != AdaptivePhaseEstimator::regime_t::WARM;               //
        });
    if (any_coarse) {
      // Descent stops once a plane is too small to halve usefully, so the stack may be shallower
      //   than asked for -- and empty for a plane under 4 voxels across, which is then solved cold
      //   at native resolution instead (no benefit from decimating it, and over-decimation to a
      //   handful of voxels would leave nothing to estimate a noise level from).
      ssize_t lNx = Nx;
      ssize_t lNy = Ny;
      for (ssize_t level = 1; level < std::max<ssize_t>(1, params.cascade_depth) && lNx >= 4 && lNy >= 4; ++level) {
        lNx = (lNx + 1) / 2;
        lNy = (lNy + 1) / 2;
        CascadeLevel entry;
        entry.fr.resize(lNx, lNy);
        entry.fi.resize(lNx, lNy);
        // All-unknown sigma routes solve_plane to its self-calibration branch, which is where a
        //   coarsened level is left unless the noise level map is being restricted onto it (and it
        //   is where it stays regardless on a pass that has no map at all).
        entry.sig = Eigen::ArrayXXd::Constant(lNx, lNy, -1.0);
        entry.pr.resize(lNx, lNy);
        entry.pi.resize(lNx, lNy);
        if (params.cascade_prolongation == AdaptivePhaseEstimator::cascade_prolongation_t::ITERATE)
          entry.mag.resize(lNx, lNy);
        cascade.push_back(std::move(entry));
      }
      if (params.cascade_prolongation == AdaptivePhaseEstimator::cascade_prolongation_t::ITERATE)
        mag.resize(Nx, Ny);
    }
  }

  void operator()(const Iterator &pos) {
    // This plane's volume: skipped entirely (before any data are gathered) unless the current
    //   iteration uses it, and otherwise solved in the regime that volume's own estimation
    //   history dictates.
    const ssize_t volume = volume_index(pos);
    const AdaptivePhaseEstimator::VolumePlan &vp = (*plan)[size_t(volume)];
    if (!vp.estimate)
      return;
    using regime_t = AdaptivePhaseEstimator::regime_t;
    // This plane's slot in the caller's carried fidelity-weight state. (Serialised volume, slice)
    //   is what identifies a plane from one pass to the next: a schedule may draw a different
    //   temporal subset each pass, so neither the order in which planes are visited nor the
    //   position of a volume within the current subset is stable, whereas the serialised index --
    //   the same space the VolumePlan itself is indexed in -- is.
    double *const lambda_slot =
        lambda_store == nullptr ? nullptr
                                : &(*lambda_store)[size_t(volume * num_slices + pos.index(slice_axis))];
#ifdef DWIDENOISE2_APC_PROFILE
    const double lambda_in = lambda_slot == nullptr ? 0.0 : *lambda_slot;
#endif
    // A cold volume starts on the coarsened grids where the plane is large enough to decimate; a
    //   plane too small for that falls back to a cold native solve on the full budget, in either
    //   cold regime.
    const bool use_coarse = vp.regime != regime_t::WARM && !cascade.empty();

    const size_t ax = inner_axes[0];
    const size_t ay = inner_axes[1];
    assign_pos_of(pos, outer_axes).to(in, phase);

    // Gather the empirical complex plane.
    for (auto l = Loop(inner_axes)(in); l; ++l) {
      const ssize_t ix = in.index(ax);
      const ssize_t iy = in.index(ay);
      const T value = in.value();
      fr(ix, iy) = double(value.real());
      fi(ix, iy) = double(value.imag());
    }

    // Pre-load the incoming (previous-iteration, or unit) phase. On a warm-started pass this is
    //   also the solver's initial guess; on any pass it is the fallback if solve_plane declines
    //   the slice (fully degenerate: no estimable noise level), so writing back is a no-op.
    for (auto l = Loop(inner_axes)(phase); l; ++l) {
      const ssize_t ix = phase.index(ax);
      const ssize_t iy = phase.index(ay);
      const cfloat p = phase.value();
      phase_r(ix, iy) = double(p.real());
      phase_i(ix, iy) = double(p.imag());
    }

    if (use_coarse) {
      // Restriction: decimate the data down the stack, each level from the one above it.
      downsample2x(fr, cascade[0].fr);
      downsample2x(fi, cascade[0].fi);
      for (size_t level = 1; level != cascade.size(); ++level) {
        downsample2x(cascade[level - 1].fr, cascade[level].fr);
        downsample2x(cascade[level - 1].fi, cascade[level].fi);
      }
      // Restriction of the noise level, where the policy asks for it and a map exists at all. The
      //   alternative -- and what a pass with no map has no choice but to do -- is for each level to
      //   estimate a global noise level from its own decimated data, which is what the all-unknown
      //   sigma left in place below routes solve_plane to.
      using cascade_lambda_t = AdaptivePhaseEstimator::cascade_lambda_t;
      const bool restrict_map = params.cascade_lambda != cascade_lambda_t::SELF_CALIBRATED && have_sigma;
      bool have_native_sigma = false;
      if (restrict_map) {
        gather_sigma(ax, ay);
        have_native_sigma = true;
        const bool weight_mean =
            params.cascade_restriction == AdaptivePhaseEstimator::cascade_restriction_t::WEIGHT_MEAN;
        restrict_sigma(sig, cascade[0].sig, weight_mean);
        for (size_t level = 1; level != cascade.size(); ++level)
          restrict_sigma(cascade[level - 1].sig, cascade[level].sig, weight_mean);
      }
      // Cold solve on the coarsest grid, then prolongation: each finer level is warm-started from
      //   the level below it upsampled, so it inherits the long-wavelength phase already recovered
      //   there and has only the detail that grid could not represent left to add. If a level is
      //   degenerate, leave the pre-loaded incoming phase untouched and stop (a slice with no
      //   estimable noise level is no more tractable at a finer resolution).
      // None of these solves takes part in the caller's carried state: their fidelity weights answer
      //   decimated problems, whose noise levels -- and hence whose weights -- are on a different
      //   scale from the native grid's. Under cascade_lambda_t::COARSEST_ONLY they do carry a weight
      //   among themselves, one level to the next, scaled by the grid-to-grid factor.
      const bool prolongate_iterate =
          params.cascade_prolongation == AdaptivePhaseEstimator::cascade_prolongation_t::ITERATE;
      const bool pin_lambda = params.cascade_lambda == cascade_lambda_t::COARSEST_ONLY;
      const double lambda_ratio = std::max(1e-12, params.cascade_lambda_ratio);
      double level_lambda = 0.0;
      size_t level = cascade.size() - 1;
      bool solved = estimator->solve_plane(cascade[level].fr,                              //
                                           cascade[level].fi,                              //
                                           cascade[level].sig,                             //
                                           cascade[level].pr,                              //
                                           cascade[level].pi,                              //
                                           false,                                          //
                                           pin_lambda ? &level_lambda : nullptr,           //
                                           prolongate_iterate ? &cascade[level].mag : nullptr);
      while (solved && level != 0) {
        if (prolongate_iterate)
          upsample_iterate(cascade[level].pr,       //
                           cascade[level].pi,       //
                           cascade[level].mag,      //
                           cascade[level - 1].pr,   //
                           cascade[level - 1].pi,   //
                           cascade[level - 1].mag); //
        else
          upsample_phase(cascade[level].pr, cascade[level].pi, cascade[level - 1].pr, cascade[level - 1].pi);
        --level;
        level_lambda /= lambda_ratio;
        solved = estimator->solve_plane(cascade[level].fr,                                //
                                        cascade[level].fi,                                //
                                        cascade[level].sig,                               //
                                        cascade[level].pr,                                //
                                        cascade[level].pi,                                //
                                        true,                                             //
                                        pin_lambda ? &level_lambda : nullptr,             //
                                        prolongate_iterate ? &cascade[level].mag : nullptr, //
                                        pin_lambda);
      }
      if (solved) {
        if (prolongate_iterate)
          upsample_iterate(cascade[0].pr, cascade[0].pi, cascade[0].mag, phase_r, phase_i, mag);
        else
          upsample_phase(cascade[0].pr, cascade[0].pi, phase_r, phase_i);
      }
      if (solved && vp.regime == regime_t::COLD_COARSE_THEN_WARM) {
        // No later pass will refine this volume, so carry out that refinement here: the upsampled
        //   coarsened phase is exactly the warm start a subsequent pass would have used, so the
        //   reduced ..._warm budget re-settles it at native resolution just as it would there.
        if (!have_native_sigma)
          gather_sigma(ax, ay);
        estimator->solve_plane(fr,                                     //
                               fi,                                     //
                               sig,                                    //
                               phase_r,                                //
                               phase_i,                                //
                               true,                                   //
                               lambda_slot,                            //
                               prolongate_iterate ? &mag : nullptr);
      }
    } else {
      // Native resolution: warm-started from the incoming phase, or -- on a plane too small to
      //   decimate -- cold on the full budget.
      gather_sigma(ax, ay);
      estimator->solve_plane(fr, fi, sig, phase_r, phase_i, vp.regime == regime_t::WARM, lambda_slot);
    }

    for (auto l = Loop(inner_axes)(phase); l; ++l) {
      const ssize_t ix = phase.index(ax);
      const ssize_t iy = phase.index(ay);
      phase.value() = cfloat(float(phase_r(ix, iy)), float(phase_i(ix, iy)));
    }
#ifdef DWIDENOISE2_APC_PROFILE
    // Attribution of the per-plane records solve_plane() emits above: which volume and which slice
    //   produced them, in which regime, and the fidelity weight carried into and out of that plane.
    //   solve_plane() is handed an anonymous pair of arrays, so this is the only point at which a
    //   plane's identity is known alongside its result.
    DEBUG("APC plane: volume=" + str(volume) +                                       //
          " slice=" + str(pos.index(slice_axis)) +                                   //
          " regime=" + regime_name(vp.regime) +                                      //
          " lambda_in=" + str(lambda_in) +                                           //
          " lambda_out=" + str(lambda_slot == nullptr ? 0.0 : *lambda_slot));        //
#endif
  }

private:
  // Fill "sig" with the per-voxel noise level for the current native-resolution plane, ready for
  //   spatially-varying weighting: the noise map interpolated at each voxel's scanner position (it
  //   may be lower-resolution than the data). Keyed on the map's availability rather than on the
  //   solve regime: a volume drawn into a temporal subset for the first time at a later iteration
  //   is solved cold, but the noise map its neighbours enjoy is equally valid for it, and there is
  //   no reason to discard it in favour of the self-calibrated global level. Without a map (the
  //   very first pass of a schedule with no -noise_in), all-unknown sigma routes solve_plane to its
  //   self-calibration branch.
  void gather_sigma(const size_t ax, const size_t ay) {
    if (!have_sigma) {
      sig.setConstant(-1.0);
      return;
    }
    Interp::Cubic<Image<float>> sigma_interp(sigma);
    for (auto l = Loop(inner_axes)(in); l; ++l) {
      const ssize_t ix = in.index(ax);
      const ssize_t iy = in.index(ay);
      sigma_interp.scanner(transform.voxel2scanner * Eigen::Vector3d({double(in.index(0)),   //
                                                                      double(in.index(1)),   //
                                                                      double(in.index(2))})); //
      const double s = double(sigma_interp.value());
      sig(ix, iy) = std::isfinite(s) ? s : -1.0;
    }
  }

  // Serialised (Casorati column) index of the volume this outer position belongs to: the axis-3
  //   index for 4D data, otherwise the index the preconditioner's serialisation image assigns to
  //   the supra-spatial multi-index. This is the space in which VolumePlan entries (and the
  //   preconditioner's temporal subset) are indexed.
  ssize_t volume_index(const Iterator &pos) {
    if (!serialise.valid())
      return pos.index(3);
    for (size_t axis = 3; axis != pos.ndim(); ++axis)
      serialise.index(axis - 3) = pos.index(axis);
    return ssize_t(serialise.value());
  }

  const AdaptivePhaseEstimator *estimator;
  AdaptivePhaseEstimator::Params params;
  std::vector<size_t> outer_axes;
  std::vector<size_t> inner_axes;
  Image<T> in;
  Image<float> sigma;
  Image<cfloat> phase;
  Image<uint32_t> serialise;
  const std::vector<AdaptivePhaseEstimator::VolumePlan> *plan;
  // Caller-owned fidelity weight per (serialised volume, slice), or null where no state is carried
  //   between passes. Each plane owns one entry and no two planes share one, so the threaded loop
  //   writes it without locking, exactly as it does the phase map.
  std::vector<double> *lambda_store;
  size_t slice_axis;
  ssize_t num_slices;
  bool have_sigma;
  Transform transform;
  ssize_t Nx, Ny;
  Eigen::ArrayXXd fr, fi, sig, phase_r, phase_i;
  // Magnitude of the image the finest coarsened level solved, carried up with its phase; used only
  //   where the prolongation carries the iterate rather than the phase alone.
  Eigen::ArrayXXd mag;
  // One coarsened level of a cold solve: the decimated data, the noise level it is solved against
  //   (all-unknown where that level self-calibrates), and the phase and image magnitude solved
  //   there.
  struct CascadeLevel {
    Eigen::ArrayXXd fr, fi, sig, pr, pi, mag;
  };
  // The coarsened levels, finest first, so the cold solve starts at the back and works forward.
  //   The native grid is not among them; it is the members above. Empty where no planned volume
  //   takes a downsampled cold-solve path, or where the plane is too small to decimate.
  std::vector<CascadeLevel> cascade;
};

} // namespace

template <typename T>
void AdaptivePhaseEstimator::operator()(Image<T> &in,
                                        Image<float> &sigma,
                                        Image<cfloat> &io_phase,
                                        Image<uint32_t> &serialise,
                                        const std::vector<VolumePlan> &plan,
                                        std::vector<double> *const io_lambda) const {
  assert(inslice_axes.size() == 2);
  assert(io_lambda == nullptr || io_lambda->size() == plan.size() * size_t(in.size(slice_axis())));
  // Outer axes = every axis of the data except the two in-slice (demodulation) axes: the
  //   slice-normal spatial axis, the volume axis, and any serialised supra-volume axes. Each
  //   outer position is one independent 2-D phase-estimation problem.
  std::vector<size_t> outer_axes;
  for (size_t a = 0; a != in.ndim(); ++a)
    if (std::find(inslice_axes.begin(), inslice_axes.end(), a) == inslice_axes.end())
      outer_axes.push_back(a);

  // Describe the pass from the plan: "estimating" if it contains any cold solve, "updating" if
  //   every planned volume is being refined from a prior estimate, and how many of the image's
  //   volumes it touches at all (fewer than all under temporal sub-sampling). Volumes with
  //   estimate == false still cost one no-op functor invocation per plane, so they remain in the
  //   progress total; that dispatch is negligible against a solve.
  ssize_t num_warm = 0;
  ssize_t num_bootstrap = 0;
  ssize_t num_cold_complete = 0;
  for (const VolumePlan &vp : plan) {
    if (!vp.estimate)
      continue;
    switch (vp.regime) {
    case regime_t::WARM:
      ++num_warm;
      break;
    case regime_t::COLD_COARSE:
      ++num_bootstrap;
      break;
    case regime_t::COLD_COARSE_THEN_WARM:
      ++num_cold_complete;
      break;
    }
  }
  const ssize_t num_estimated = num_warm + num_bootstrap + num_cold_complete;
  if (num_estimated == 0)
    return;
  std::string message =
      std::string(num_warm == num_estimated ? "updating" : "estimating") + " background phase (adaptive phase correction)";
  if (num_estimated < ssize_t(plan.size()))
    message += " for " + str(num_estimated) + " of " + str(plan.size()) + " volumes";
  INFO("Adaptive phase correction: " + str(num_bootstrap) + " volume(s) bootstrapped cold on a coarse grid, " +
       str(num_cold_complete) + " solved cold and refined at native resolution, " + str(num_warm) +
       " refined from a prior estimate, " + str(ssize_t(plan.size()) - num_estimated) +
       " skipped (not used by this iteration)");

  ThreadedLoop(message, in, outer_axes, inslice_axes)
      .run_outer(
          APCFunctor<T>(*this, outer_axes, inslice_axes, in, sigma, io_phase, serialise, plan, io_lambda, params));
}

// Explicitly instantiated for complex data only; the phase is ill-defined for real input, and
//   callers gate instantiation with `if constexpr (is_complex<T>::value)`.
template void AdaptivePhaseEstimator::operator()(Image<cfloat> &,
                                                 Image<float> &,
                                                 Image<cfloat> &,
                                                 Image<uint32_t> &,
                                                 const std::vector<VolumePlan> &,
                                                 std::vector<double> *) const;
template void AdaptivePhaseEstimator::operator()(Image<cdouble> &,
                                                 Image<float> &,
                                                 Image<cfloat> &,
                                                 Image<uint32_t> &,
                                                 const std::vector<VolumePlan> &,
                                                 std::vector<double> *) const;

} // namespace MR::Denoise::Precondition
