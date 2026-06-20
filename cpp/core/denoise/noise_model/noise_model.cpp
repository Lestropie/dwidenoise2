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

#include "denoise/noise_model/noise_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "exception.h"

namespace MR::Denoise::NoiseModel {

namespace {

// --- Tabulation grid parameters -------------------------------------------
// All transforms are tabulated as functions of the normalised intensity
//   theta = m / sigma (equivalently the normalised SNR a = nu / sigma).
// The grids cover the low-to-moderate SNR regime in which the transforms are
//   appreciably nonlinear. Real data routinely exceed these bounds (DWI b=0 SNR
//   reaches several hundred), so values beyond the tabulated maximum are NOT
//   clamped: the forward / algebraic-inverse Luts already extrapolate linearly
//   (slope -> 1 at high SNR), and the unbiased-inverse construction does the same
//   via interp_increasing_extrap(). The transforms are asymptotically linear
//   there (the magnitude bias vanishes, a -> theta_m), so the first-order
//   extrapolation is accurate; A_MAX / X_MAX need only bound the nonlinear range.
constexpr default_type A_MAX = 32.0; // maximum tabulated normalised SNR (nu / sigma)
constexpr default_type X_MAX = 64.0; // maximum tabulated normalised intensity (m / sigma)
constexpr ssize_t N_A = 2048;        // samples on the normalised-SNR grid
constexpr ssize_t N_X = 8192;        // samples on the normalised-intensity grid
constexpr ssize_t N_U = 4096;        // samples on the stabilised-domain grid

// 1F1(alpha; gamma; w) (confluent hypergeometric / Kummer) for w >= 0,
//   evaluated as a positive-term series (no catastrophic cancellation).
default_type oneF1_pos(const default_type alpha, const default_type gamma, const default_type w) {
  default_type term = 1.0;
  default_type sum = 1.0;
  for (ssize_t k = 0; k != 100000; ++k) {
    term *= (alpha + default_type(k)) / ((gamma + default_type(k)) * default_type(k + 1)) * w;
    sum += term;
    if (term <= 1.0e-15 * sum)
      break;
  }
  return sum;
}

// E[m | nu] / sigma as a function of a = nu / sigma for a non-central chi
//   distribution with 2N degrees of freedom:
//     mean = sqrt(2) * Gamma(N+1/2)/Gamma(N) * 1F1(-1/2; N; -a^2/2)
// The Kummer transform 1F1(-1/2; N; -w) = e^{-w} 1F1(N+1/2; N; w), w = a^2/2,
//   is used to avoid cancellation; for large w the asymptotic sqrt(a^2 + 2N - 1)
//   (which yields the correct leading mean correction and Var -> 1) is used.
default_type mean_normalised(const default_type a, const ssize_t N) {
  const default_type w = 0.5 * a * a;
  if (w <= 350.0) {
    const default_type gamma_ratio = std::exp(std::lgamma(default_type(N) + 0.5) - std::lgamma(default_type(N)));
    return std::sqrt(2.0) * gamma_ratio * std::exp(-w) * oneF1_pos(default_type(N) + 0.5, default_type(N), w);
  }
  return std::sqrt(a * a + default_type(2 * N - 1));
}

// Var[m | nu] / sigma^2 = (a^2 + 2N) - mean_normalised(a)^2.
default_type variance_normalised(const default_type a, const ssize_t N) {
  const default_type mu = mean_normalised(a, N);
  const default_type v = (a * a + default_type(2 * N)) - mu * mu;
  return std::max(v, 1.0e-12);
}

// Linear interpolation of ytab(xtab) with monotonically-increasing xtab,
//   clamping to the end values outside the tabulated range.
// Used to invert the (non-uniform) tabulated transforms.
default_type interp_increasing(const std::vector<default_type> &xtab, //
                               const std::vector<default_type> &ytab, //
                               const default_type x) {                 //
  const ssize_t n = ssize_t(xtab.size());
  if (x <= xtab.front())
    return ytab.front();
  if (x >= xtab.back())
    return ytab.back();
  ssize_t lo = 0;
  ssize_t hi = n - 1;
  while (hi - lo > 1) {
    const ssize_t mid = (lo + hi) / 2;
    if (xtab[mid] <= x)
      lo = mid;
    else
      hi = mid;
  }
  const default_type frac = (x - xtab[lo]) / (xtab[lo + 1] - xtab[lo]);
  return ytab[lo] * (1.0 - frac) + ytab[lo + 1] * frac;
}

// As interp_increasing(), but above the tabulated range linearly extrapolates
//   using the slope of the final tabulated segment rather than clamping to the
//   last value (the behaviour below the range is unchanged: clamp to the first).
// Used when inverting the stabilising functions (eta, mu) to recover the
//   normalised SNR a = nu / sigma. Both are asymptotically linear with unit slope
//   at high SNR (the magnitude bias vanishes, so a -> theta_m), and beyond the
//   maximum tabulated SNR continuing that trend is far more accurate than
//   saturating at the table edge. Saturating would instead return a constant
//   a = A_MAX and, via the derived Jacobian, a vanishing inverse-transform gain,
//   collapsing every high-SNR voxel onto a noise-level-scaled pedestal with no
//   residual detail (see vst_plan.md; the failure mode this replaces).
default_type interp_increasing_extrap(const std::vector<default_type> &xtab, //
                                      const std::vector<default_type> &ytab, //
                                      const default_type x) {                //
  const ssize_t n = ssize_t(xtab.size());
  if (x <= xtab.front())
    return ytab.front();
  if (x >= xtab.back()) {
    const default_type slope = (ytab[n - 1] - ytab[n - 2]) / (xtab[n - 1] - xtab[n - 2]);
    return ytab[n - 1] + (x - xtab[n - 1]) * slope;
  }
  ssize_t lo = 0;
  ssize_t hi = n - 1;
  while (hi - lo > 1) {
    const ssize_t mid = (lo + hi) / 2;
    if (xtab[mid] <= x)
      lo = mid;
    else
      hi = mid;
  }
  const default_type frac = (x - xtab[lo]) / (xtab[lo + 1] - xtab[lo]);
  return ytab[lo] * (1.0 - frac) + ytab[lo + 1] * frac;
}

std::vector<default_type> uniform_grid(const default_type max_value, const ssize_t n) {
  std::vector<default_type> result(n);
  const default_type spacing = max_value / default_type(n - 1);
  for (ssize_t i = 0; i != n; ++i)
    result[i] = default_type(i) * spacing;
  return result;
}

// Central differences of y(x) on a uniform grid of spacing dx.
std::vector<default_type> derivative(const std::vector<default_type> &y, const default_type dx) {
  const ssize_t n = ssize_t(y.size());
  std::vector<default_type> d(n);
  for (ssize_t i = 1; i != n - 1; ++i)
    d[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx);
  d[0] = (y[1] - y[0]) / dx;
  d[n - 1] = (y[n - 1] - y[n - 2]) / dx;
  return d;
}

} // namespace

NonCentralChi::NonCentralChi(const ssize_t num_channels, const vst_method_t vst_method)
    : N(num_channels), vst_method(vst_method) {
  if (N < 1)
    throw Exception("Non-central chi noise model requires at least one receive channel");

  floor_mean = mean_normalised(0.0, N);
  const default_type floor_sd = std::sqrt(variance_normalised(0.0, N));

  // Normalised-SNR grid and the corresponding distribution mean (monotone in a).
  const std::vector<default_type> a_grid = uniform_grid(A_MAX, N_A);
  std::vector<default_type> mu(N_A);
  for (ssize_t i = 0; i != N_A; ++i)
    mu[i] = mean_normalised(a_grid[i], N);

  // --- Forward variance-stabilising transform f(theta) -----------------------
  // f'(theta) = 1 / sqrt(Var) evaluated at the SNR whose mean equals theta;
  //   below the floor mean the constant slope 1 / floor_sd is used.
  // Anchored f(0) = 0; integrated by the trapezoidal rule.
  const std::vector<default_type> x_grid = uniform_grid(X_MAX, N_X);
  const default_type dx = x_grid[1] - x_grid[0];
  std::vector<default_type> f(N_X);
  std::vector<default_type> fprime(N_X);
  for (ssize_t j = 0; j != N_X; ++j) {
    if (x_grid[j] < floor_mean) {
      fprime[j] = 1.0 / floor_sd;
    } else {
      const default_type a_at = interp_increasing(mu, a_grid, x_grid[j]);
      fprime[j] = 1.0 / std::sqrt(variance_normalised(a_at, N));
    }
  }
  f[0] = 0.0;
  for (ssize_t j = 1; j != N_X; ++j)
    f[j] = f[j - 1] + 0.5 * (fprime[j - 1] + fprime[j]) * dx;
  const default_type u_max = f.back();
  forward = Lut(0.0, dx, std::vector<default_type>(f));

  // --- Algebraic inverse f^{-1}(u) ------------------------------------------
  const std::vector<default_type> u_grid = uniform_grid(u_max, N_U);
  const default_type du = u_grid[1] - u_grid[0];
  {
    std::vector<default_type> inv(N_U);
    for (ssize_t k = 0; k != N_U; ++k)
      inv[k] = interp_increasing(f, x_grid, u_grid[k]);
    inverse_algebraic_lut = Lut(0.0, du, std::move(inv));
  }

  // --- Exact-/moment- inverse Psi(u) (the debias step) -----------------------
  std::vector<default_type> psi(N_U);
  switch (vst_method) {
  case vst_method_t::FOI: {
    // eta(a) = E[ f(m/sigma) | a ] computed via the non-central chi-square
    //   Poisson mixture:  ncchi^2_{2N}(a^2) = chi^2_{2(N+J)}, J ~ Poisson(a^2/2).
    //   E_j = E[ f(sqrt(chi^2_{2(N+j)})) ] is independent of a (precomputed once),
    //   and eta(a) = sum_j Poisson(j; a^2/2) E_j.
    // Psi = eta^{-1}, the exact-unbiased inverse: smooth and well-defined at a=0.
    const default_type lambda_max = 0.5 * A_MAX * A_MAX;
    const ssize_t Jmax = ssize_t(std::ceil(lambda_max + 8.0 * std::sqrt(lambda_max))) + 5;

    // Central chi pdf (2N DOF) on the intensity grid; advanced across orders by
    //   the recurrence p(x; 2(N+j)) = p(x; 2(N+j-1)) * x^2 / (2(N+j-1)).
    std::vector<default_type> p(N_X, 0.0);
    const default_type log_norm = (default_type(N) - 1.0) * std::log(2.0) + std::lgamma(default_type(N));
    for (ssize_t i = 1; i != N_X; ++i) {
      const default_type x = x_grid[i];
      p[i] = std::exp(default_type(2 * N - 1) * std::log(x) - 0.5 * x * x - log_norm);
    }
    std::vector<default_type> E(Jmax + 1);
    for (ssize_t j = 0; j <= Jmax; ++j) {
      default_type integral = 0.0;
      for (ssize_t i = 0; i != N_X - 1; ++i)
        integral += 0.5 * (f[i] * p[i] + f[i + 1] * p[i + 1]) * dx;
      E[j] = integral;
      const default_type scale = 1.0 / default_type(2 * (N + j));
      for (ssize_t i = 0; i != N_X; ++i)
        p[i] *= x_grid[i] * x_grid[i] * scale;
    }

    // eta(a) on the SNR grid via numerically-stable (max-subtracted) Poisson sum.
    std::vector<default_type> lgam(Jmax + 1);
    for (ssize_t j = 0; j <= Jmax; ++j)
      lgam[j] = std::lgamma(default_type(j + 1));
    std::vector<default_type> eta(N_A);
    eta[0] = E[0]; // a = 0: pure central chi, eta(0) = E_0
    for (ssize_t i = 1; i != N_A; ++i) {
      const default_type lambda = 0.5 * a_grid[i] * a_grid[i];
      const default_type log_lambda = std::log(lambda);
      default_type log_max = -std::numeric_limits<default_type>::infinity();
      for (ssize_t j = 0; j <= Jmax; ++j) {
        const default_type log_w = -lambda + default_type(j) * log_lambda - lgam[j];
        log_max = std::max(log_max, log_w);
      }
      default_type num = 0.0;
      default_type den = 0.0;
      for (ssize_t j = 0; j <= Jmax; ++j) {
        const default_type w = std::exp(-lambda + default_type(j) * log_lambda - lgam[j] - log_max);
        num += w * E[j];
        den += w;
      }
      eta[i] = num / den;
    }

    for (ssize_t k = 0; k != N_U; ++k)
      psi[k] = interp_increasing_extrap(eta, a_grid, u_grid[k]);
  } break;
  case vst_method_t::KOAY:
  case vst_method_t::MOM: {
    // First-moment inverse: Psi(u) = mu^{-1}( f^{-1}(u) ).
    // mu^{-1} clamps to a = 0 below the floor mean (mu is flat near a = 0,
    //   so this inverse is ill-conditioned there).
    // KOAY enforces the same hard floor clamp explicitly; in the present
    //   known-sigma / group-mean setting the Koay-Basser fixed point coincides
    //   with the first-moment inverse (its distinctive sub-SNR-1.913 behaviour
    //   requires a per-group variance estimate, which is not propagated here).
    for (ssize_t k = 0; k != N_U; ++k) {
      const default_type theta_m = interp_increasing(f, x_grid, u_grid[k]);
      if (vst_method == vst_method_t::KOAY && theta_m <= floor_mean)
        psi[k] = 0.0;
      else
        psi[k] = interp_increasing_extrap(mu, a_grid, theta_m);
    }
  } break;
  }
  inverse_unbiased_lut = Lut(0.0, du, std::vector<default_type>(psi));

  // --- Local gain J = d(theta_nu)/d(u) --------------------------------------
  jacobian_lut = Lut(0.0, du, derivative(psi, du));
}

default_type NonCentralChi::stabilise(const default_type m, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return default_type(0);
  return forward(m / sigma);
}

default_type NonCentralChi::inverse_algebraic(const default_type u, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return default_type(0);
  return sigma * inverse_algebraic_lut(u);
}

default_type NonCentralChi::inverse_unbiased(const default_type u, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return default_type(0);
  return sigma * inverse_unbiased_lut(u);
}

default_type NonCentralChi::jacobian(const default_type u, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return default_type(0);
  return sigma * jacobian_lut(u);
}

default_type NonCentralChi::mean(const default_type nu, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return nu;
  return sigma * mean_normalised(nu / sigma, N);
}

default_type NonCentralChi::variance(const default_type nu, const default_type sigma) const {
  if (!(sigma > default_type(0)))
    return default_type(0);
  return sigma * sigma * variance_normalised(nu / sigma, N);
}

std::string NonCentralChi::description() const {
  return "non-central chi (" + std::to_string(N) + " channels, " + std::to_string(2 * N) +
         " DOF; VST: " + vst_methods[size_t(vst_method)] + ")";
}

std::string Rician::description() const { return "Rician (2 DOF; VST: " + vst_methods[size_t(vst_method)] + ")"; }

std::shared_ptr<Base> make(const distribution_t distribution, //
                           const ssize_t num_channels,        //
                           const vst_method_t vst_method) {   //
  // -vst_method none / linear select the transform directly, independent of the
  //   underlying noise distribution:
  //   - none   : identity (no variance stabilisation);
  //   - linear : the simple linear scaling u = m / sigma (the Gaussian model),
  //              exact for additive Gaussian (complex / phase-demodulated) noise.
  switch (vst_method) {
  case vst_method_t::NONE:
    return std::make_shared<Identity>();
  case vst_method_t::LINEAR:
    return std::make_shared<Gaussian>();
  default:
    break;
  }
  switch (distribution) {
  case distribution_t::GAUSSIAN:
    return std::make_shared<Gaussian>();
  case distribution_t::RICIAN:
    return std::make_shared<Rician>(vst_method);
  case distribution_t::NONCENTRALCHI:
    return std::make_shared<NonCentralChi>(num_channels, vst_method);
  }
  assert(false);
  return nullptr;
}

} // namespace MR::Denoise::NoiseModel
