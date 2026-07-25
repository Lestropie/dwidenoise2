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

#define DWIDENOISE2_APC_PROFILE

namespace MR::Denoise::Precondition {

namespace {
// Robust global per-component noise standard deviation from the finite differences of a
//   complex plane (Immerkaer 1996; Donoho & Johnstone MAD). The difference of two i.i.d.
//   N(0,sigma^2) samples is N(0, 2*sigma^2); the median of |differences| (a robust proxy for
//   the MAD, as the noise differences are ~zero-median) divided by 0.6745 estimates that
//   difference's std, and dividing by sqrt(2) recovers the per-component sigma. Sparse tissue
//   edges are large-value outliers the median ignores. Used only on the first iteration (or a
//   slice with no usable noise map) to calibrate the discrepancy criterion in the absence of a
//   noise level estimate. Returns -1 if no differences are available (degenerate slice).
double estimate_sigma_mad(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                          const Eigen::Ref<const Eigen::ArrayXXd> &fi) {
  const ssize_t Nx = fr.rows();
  const ssize_t Ny = fr.cols();
  std::vector<double> d;
  d.reserve(size_t(std::max<ssize_t>(0, 2 * (Nx - 1) * Ny) + std::max<ssize_t>(0, 2 * Nx * (Ny - 1))));
  for (ssize_t j = 0; j != Ny; ++j)
    for (ssize_t i = 0; i != Nx; ++i) {
      if (i + 1 < Nx) {
        d.push_back(std::abs(fr(i + 1, j) - fr(i, j)));
        d.push_back(std::abs(fi(i + 1, j) - fi(i, j)));
      }
      if (j + 1 < Ny) {
        d.push_back(std::abs(fr(i, j + 1) - fr(i, j)));
        d.push_back(std::abs(fi(i, j + 1) - fi(i, j)));
      }
    }
  if (d.empty())
    return -1.0;
  const size_t mid = d.size() / 2;
  std::nth_element(d.begin(), d.begin() + mid, d.end());
  const double median_abs_diff = d[mid];
  return median_abs_diff / 0.6744897501960817 / Math::sqrt2; // 0.6745 = Phi^{-1}(0.75)
}

// 2x box-downsample of a plane (2x2 average, clamped at the far edge for odd sizes). The first
//   iteration solves the phase on the resulting coarse grid: a background phase is low-bandwidth,
//   so decimation is near-lossless, at ~1/4 the voxel count (hence ~1/4 the per-sweep cost). The
//   coarse-grid noise level after 2x2 averaging is lower than native, but the first pass
//   self-calibrates its noise level from that same coarse data (robust MAD), so the discrepancy
//   criterion stays self-consistent; the coarse solve is only a bootstrap the later native passes
//   refine.
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
} // namespace

bool AdaptivePhaseEstimator::solve_plane(const Eigen::Ref<const Eigen::ArrayXXd> &fr,
                                 const Eigen::Ref<const Eigen::ArrayXXd> &fi,
                                 const Eigen::Ref<const Eigen::ArrayXXd> &sigma,
                                 Eigen::Ref<Eigen::ArrayXXd> phase_r,
                                 Eigen::Ref<Eigen::ArrayXXd> phase_i,
                                 const bool warm_start) const {
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
  //      global per-component noise level is estimated from the data (robust MAD of finite
  //      differences) and uniform weighting is applied over the whole slice. This is what lets
  //      APC run from the first iteration without a Hann bootstrap.
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
    const double sigma_hat = estimate_sigma_mad(fr, fi);
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
    const Eigen::ArrayXXd mag = (fr.square() + fi.square()).sqrt();
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

  // Step sizes: for the 2-D forward-difference gradient, ||grad||^2 <= 8; the coupled
  //   two-channel operator has the same norm. tau * sigma_cp * ||K||^2 = 1 (the limiting
  //   stable choice for Chambolle-Pock Algorithm 1, theta = 1).
  const double L2 = 8.0;
  const double tau = 1.0 / std::sqrt(L2);
  const double sigma_cp = 1.0 / std::sqrt(L2);

  // One primal-dual sweep at fidelity weight "lambda". Warm-starts from the current state, and
  //   stops early once the relative per-sweep change in u falls below params.cp_tol (data-driven
  //   early exit; see the DWIDENOISE2_APC_PROFILE discussion in phase_estimator.h). That
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
    for (ssize_t iter = 0; iter != niter; ++iter) {
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
      //   ubar = 2*u_new - u_old (theta = 1). Each voxel is independent, so the in-place
      //   overwrite of u is safe.
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
          ubr(i, j) = 2.0 * unewr - oldr;
          ubi(i, j) = 2.0 * unewi - oldi;
          ur(i, j) = unewr;
          ui(i, j) = unewi;
        }
      }
      if (params.cp_tol > 0.0 && sum_old2 > 0.0 && sum_diff2 < Math::pow2(params.cp_tol) * sum_old2)
        break;
    }
#ifdef DWIDENOISE2_APC_PROFILE
    return used;
#endif
  };

  // --- Discrepancy-criterion (Morozov) fixed point on lambda ----------------------------
  // Chambolle (2004) update, eq. 6, seeded by eq. 7, targeting a weighted residual energy of
  //   Rr^2 + Ri^2 = 2 * ndomain * sigmabar^2 (spatially-varying form, their eq. 11). Image
  //   and lambda are refined jointly (the solver is warm-started across updates).
  double lambda = 2.1237 / sigmabar + 0.0547 / sigmabar2; // eq. 7
  const double target_scale = 2.0 * std::sqrt(double(ndomain)) * sigmabar;
  // Bookkeeping for the DWIDENOISE2_APC_PROFILE DEBUG() call below only -- entirely compiled out
  //   otherwise, so a non-profile build's timing reflects only run_cp's cp_tol early exit, not
  //   the cost of reporting it.
#ifdef DWIDENOISE2_APC_PROFILE
  ssize_t lambda_iters_used = 0;
  ssize_t cp_sweeps_used = 0;
#endif
  for (ssize_t li = 0; li != n_lambda; ++li) {
#ifdef DWIDENOISE2_APC_PROFILE
    lambda_iters_used = li + 1;
    cp_sweeps_used +=
#endif
        run_cp(n_cp, lambda);
    const double Rr = std::sqrt((indom * w * (ur - fr).square()).sum());
    const double Ri = std::sqrt((indom * w * (ui - fi).square()).sum());
    const double lambda_new = lambda * (Rr + Ri) / target_scale;
    if (!std::isfinite(lambda_new) || lambda_new <= 0.0)
      break;
    const double rel = std::abs(lambda_new - lambda) / lambda;
    lambda = lambda_new;
    if (rel < params.lambda_tol)
      break;
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
        " lambda_iters=" + str(lambda_iters_used) + "/" + str(n_lambda) +            //
        " cp_sweeps=" + str(cp_sweeps_used) + "/" + str(n_lambda * n_cp + n_polish) + //
        " polish_sweeps=" + str(polish_sweeps_used) + "/" + str(n_polish) +          //
        " final_lambda=" + str(lambda));
#else
  run_cp(n_polish, lambda);
#endif

  // --- Extract unit-magnitude phase -----------------------------------------------------
  for (ssize_t j = 0; j != Ny; ++j)
    for (ssize_t i = 0; i != Nx; ++i) {
      const double mag = std::hypot(ur(i, j), ui(i, j));
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
             const std::vector<AdaptivePhaseEstimator::VolumePlan> &plan)
      : estimator(&estimator),
        outer_axes(outer_axes),
        inner_axes(inner_axes),
        in(in),
        sigma(sigma),
        phase(phase),
        serialise(serialise),
        plan(&plan),
        have_sigma(sigma.valid()),
        transform(in),
        Nx(in.size(inner_axes[0])),
        Ny(in.size(inner_axes[1])),
        fr(Nx, Ny),
        fi(Nx, Ny),
        sig(Nx, Ny),
        phase_r(Nx, Ny),
        phase_i(Nx, Ny) {
    // Coarse-grid scratch, allocated once per thread if *any* planned volume requests a
    //   downsampled solve (a 2x-downsampled solve with the phase upsampled: a background phase is
    //   smooth, so decimation is near-lossless at ~1/4 the cost). Disabled for tiny planes (no
    //   benefit; avoids over-decimation). Warm-started volumes never request it.
    const bool any_coarse =
        std::any_of(plan.begin(), plan.end(), [](const AdaptivePhaseEstimator::VolumePlan &vp) { //
          return vp.estimate && vp.downsample;                                                   //
        });
    coarse_available = any_coarse && Nx >= 4 && Ny >= 4;
    if (coarse_available) {
      const ssize_t cNx = (Nx + 1) / 2;
      const ssize_t cNy = (Ny + 1) / 2;
      fr_c.resize(cNx, cNy);
      fi_c.resize(cNx, cNy);
      // A cold solve self-calibrates its noise level from the (coarse) data, so no map is consumed
      //   here; all-unknown sigma routes solve_plane to its MAD self-calibration branch.
      sig_c = Eigen::ArrayXXd::Constant(cNx, cNy, -1.0);
      pr_c.resize(cNx, cNy);
      pi_c.resize(cNx, cNy);
    }
  }

  void operator()(const Iterator &pos) {
    // This plane's volume: skipped entirely (before any data are gathered) unless the current
    //   iteration uses it, and otherwise solved in the regime that volume's own estimation
    //   history dictates.
    const AdaptivePhaseEstimator::VolumePlan &vp = (*plan)[size_t(volume_index(pos))];
    if (!vp.estimate)
      return;
    const bool warm_start = vp.warm_start;
    const bool use_coarse = vp.downsample && coarse_available;

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
      // Cold solve that a later pass will refine: solve on the 2x-downsampled grid, then upsample.
      //   If the coarse slice is degenerate, leave the pre-loaded incoming phase untouched.
      //   Deliberately self-calibrated even where a noise map exists (a volume first estimated at
      //   iteration 2+ of a schedule that sub-samples): sigma would have to be decimated onto the
      //   coarse grid to be consistent with the decimated data, and this solve is only a bootstrap
      //   that the volume's next (native, spatially-weighted) pass refines.
      downsample2x(fr, fr_c);
      downsample2x(fi, fi_c);
      if (estimator->solve_plane(fr_c, fi_c, sig_c, pr_c, pi_c, false))
        upsample_phase(pr_c, pi_c, phase_r, phase_i);
    } else {
      // Native resolution: warm-started from the incoming phase, or cold (the volume's first
      //   estimate, with no later pass to refine it -- or a plane too small to decimate). Where a
      //   noise map exists, gather the per-voxel noise level (interpolated at each voxel's scanner
      //   position; the map may be lower-resolution than the data) for spatially-varying weighting.
      //   This is keyed on the map's availability rather than on warm_start: a volume drawn into a
      //   temporal subset for the first time at a later iteration is solved cold, but the noise map
      //   its neighbours enjoy is equally valid for it, and there is no reason to discard it in
      //   favour of the self-calibrated global level. Without a map (the very first pass of a
      //   schedule with no -noise_in), all-unknown sigma routes solve_plane to self-calibration.
      if (have_sigma) {
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
      } else {
        sig.setConstant(-1.0);
      }
      estimator->solve_plane(fr, fi, sig, phase_r, phase_i, warm_start);
    }

    for (auto l = Loop(inner_axes)(phase); l; ++l) {
      const ssize_t ix = phase.index(ax);
      const ssize_t iy = phase.index(ay);
      phase.value() = cfloat(float(phase_r(ix, iy)), float(phase_i(ix, iy)));
    }
  }

private:
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
  std::vector<size_t> outer_axes;
  std::vector<size_t> inner_axes;
  Image<T> in;
  Image<float> sigma;
  Image<cfloat> phase;
  Image<uint32_t> serialise;
  const std::vector<AdaptivePhaseEstimator::VolumePlan> *plan;
  bool have_sigma;
  // Whether coarse-grid scratch was allocated for this thread: false if no planned volume requests
  //   a downsampled solve, or if the planes are too small to decimate.
  bool coarse_available = false;
  Transform transform;
  ssize_t Nx, Ny;
  Eigen::ArrayXXd fr, fi, sig, phase_r, phase_i;
  // Coarse-grid scratch, allocated only where some planned volume takes the 2x-downsampled
  //   cold-solve path.
  Eigen::ArrayXXd fr_c, fi_c, sig_c, pr_c, pi_c;
};

} // namespace

template <typename T>
void AdaptivePhaseEstimator::operator()(Image<T> &in,
                                        Image<float> &sigma,
                                        Image<cfloat> &io_phase,
                                        Image<uint32_t> &serialise,
                                        const std::vector<VolumePlan> &plan) const {
  assert(inslice_axes.size() == 2);
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
  ssize_t num_cold = 0;
  ssize_t num_warm = 0;
  for (const VolumePlan &vp : plan) {
    if (!vp.estimate)
      continue;
    ++(vp.warm_start ? num_warm : num_cold);
  }
  if (num_cold + num_warm == 0)
    return;
  std::string message = std::string(num_cold > 0 ? "estimating" : "updating") +
                        " background phase (adaptive phase correction)";
  if (num_cold + num_warm < ssize_t(plan.size()))
    message += " for " + str(num_cold + num_warm) + " of " + str(plan.size()) + " volumes";
  INFO("Adaptive phase correction: " + str(num_cold) + " volume(s) estimated from a cold start, " +
       str(num_warm) + " refined from a prior estimate, " + str(ssize_t(plan.size()) - num_cold - num_warm) +
       " skipped (not used by this iteration)");

  ThreadedLoop(message, in, outer_axes, inslice_axes)
      .run_outer(APCFunctor<T>(*this, outer_axes, inslice_axes, in, sigma, io_phase, serialise, plan));
}

// Explicitly instantiated for complex data only; the phase is ill-defined for real input, and
//   callers gate instantiation with `if constexpr (is_complex<T>::value)`.
template void AdaptivePhaseEstimator::operator()(
    Image<cfloat> &, Image<float> &, Image<cfloat> &, Image<uint32_t> &, const std::vector<VolumePlan> &) const;
template void AdaptivePhaseEstimator::operator()(
    Image<cdouble> &, Image<float> &, Image<cfloat> &, Image<uint32_t> &, const std::vector<VolumePlan> &) const;

} // namespace MR::Denoise::Precondition
