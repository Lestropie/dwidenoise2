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

#include "denoise/noise_model/noncentralchi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "denoise/noise_model/noise_model.h"
#include "exception.h"

namespace MR::Denoise::NoiseModel {

namespace {

// --- Tabulation grid parameters -------------------------------------------
// All transforms are tabulated as functions of the normalised intensity
//   theta = m / sigma (equivalently the normalised SNR a = nu / sigma).
//
// The grids cover SNR up to A_MAX, the regime in which the magnitude bias is
//   non-negligible (it falls as ~(2N-1)/(2a) for 2N degrees of freedom). Beyond
//   the tabulated range the bias-corrected inverse is NOT extrapolated: the bias
//   has effectively vanished, so the unbiased inverse reverts to the algebraic
//   inverse (nu = m, unit local gain). The residual bias frozen at the revert
//   point is ~(2N-1)/(2*A_MAX); to keep this ~constant for any receive-channel
//   count N, A_MAX is scaled with N (A_MAX = A_POISSON*(2N-1)) rather than being a
//   hard-coded constant, so high-channel sum-of-squares data is handled correctly.
//   The grid STEP SIZES are instead fixed (independent of N); the sample counts
//   N_A / N_X / N_U are derived per-model from the ranges (see the constructor).
//
// The exact-unbiased (FOI) eta integral is an O(A_MAX^2) Poisson mixture, so it is
//   evaluated only up to A_POISSON (a fixed SNR); beyond that the Jensen correction
//   is negligible and eta(a) -> f(mean_normalised(a)) (the moment-inverse limit,
//   where FOI and MOM coincide). This keeps the build cost independent of N.
constexpr default_type GRID_STEP_THETA = 128.0 / 16383.0;   // normalised-intensity step (m/sigma), ~0.0078
constexpr default_type GRID_STEP_A = 64.0 / 4095.0;         // normalised-SNR step (nu/sigma), ~0.0156
constexpr default_type GRID_STEP_U = 2.0 * GRID_STEP_THETA; // stabilised-domain step
constexpr default_type A_POISSON = 64.0;                    // SNR cap for the exact (Poisson-mixture) eta

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

  // Tabulated SNR / intensity ranges, scaled with the channel count so the
  //   residual bias frozen at the revert point (~(2N-1)/(2*A_MAX)) is ~constant;
  //   the step sizes are fixed, so the sample counts grow with N (and DOF).
  const default_type A_MAX = A_POISSON * default_type(2 * N - 1);
  const default_type X_MAX = A_MAX + A_POISSON; // headroom for the noise distribution's upper tail
  const ssize_t N_A = ssize_t(std::lround(A_MAX / GRID_STEP_A)) + 1;
  const ssize_t N_X = ssize_t(std::lround(X_MAX / GRID_STEP_THETA)) + 1;

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
  forward = LUT(0.0, dx, std::vector<default_type>(f));

  // --- Algebraic inverse f^{-1}(u) ------------------------------------------
  const ssize_t N_U = ssize_t(std::lround(u_max / GRID_STEP_U)) + 1;
  const std::vector<default_type> u_grid = uniform_grid(u_max, N_U);
  const default_type du = u_grid[1] - u_grid[0];
  {
    std::vector<default_type> inv(N_U);
    for (ssize_t k = 0; k != N_U; ++k)
      inv[k] = interp_increasing(f, x_grid, u_grid[k]);
    inverse_algebraic_lut = LUT(0.0, du, std::move(inv));
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
    // Evaluated only up to A_POISSON; eta(a > A_POISSON) uses the high-SNR
    //   asymptote below (the Poisson mixture is O(A^2), this keeps it N-independent).
    const default_type lambda_max = 0.5 * A_POISSON * A_POISSON;
    const ssize_t Jmax = ssize_t(std::ceil(lambda_max + 8.0 * std::sqrt(lambda_max))) + 5;

    // Central chi pdf (2N DOF) on the intensity grid, advanced across orders by
    //   the recurrence p(x; 2(N+j)) = p(x; 2(N+j-1)) * x^2 / (2(N+j-1)), carried in
    //   log space: log p(x; 2(N+j+1)) = log p(x; 2(N+j)) + 2 log(x) - log(2(N+j)).
    // The advance is purely additive, so the pdf stays accurate at its (high-x)
    //   peak for every order. Advancing the linear-space pdf instead would zero the
    //   high-x entries irrecoverably (the j=0 initialiser exp(-x^2/2) underflows in
    //   double precision for x > ~38.6). exp() is taken only to evaluate the
    //   integrand (clamped to zero far below the peak, where it is negligible).
    // The integral is truncated to the pdf's support: the highest order's chi peak
    //   is at x = sqrt(2(N+Jmax)-1) (set by A_POISSON, ~70), so x up to A_POISSON+16
    //   suffices; this bounds the per-order cost independently of N / X_MAX.
    const ssize_t n_int = std::min(N_X, ssize_t(std::lround((A_POISSON + 16.0) / GRID_STEP_THETA)) + 1);
    std::vector<default_type> log_x(n_int, -std::numeric_limits<default_type>::infinity());
    for (ssize_t i = 1; i != n_int; ++i)
      log_x[i] = std::log(x_grid[i]);
    std::vector<default_type> log_p(n_int);
    const default_type log_norm = (default_type(N) - 1.0) * std::log(2.0) + std::lgamma(default_type(N));
    for (ssize_t i = 0; i != n_int; ++i)
      log_p[i] = default_type(2 * N - 1) * log_x[i] - 0.5 * x_grid[i] * x_grid[i] - log_norm;
    std::vector<default_type> p(n_int);
    std::vector<default_type> E(Jmax + 1);
    for (ssize_t j = 0; j <= Jmax; ++j) {
      for (ssize_t i = 0; i != n_int; ++i)
        p[i] = log_p[i] > -700.0 ? std::exp(log_p[i]) : 0.0;
      default_type integral = 0.0;
      for (ssize_t i = 0; i != n_int - 1; ++i)
        integral += 0.5 * (f[i] * p[i] + f[i + 1] * p[i + 1]) * dx;
      E[j] = integral;
      const default_type log_scale = -std::log(default_type(2 * (N + j)));
      for (ssize_t i = 0; i != n_int; ++i)
        log_p[i] += 2.0 * log_x[i] + log_scale;
    }

    // eta(a) on the SNR grid via numerically-stable (max-subtracted) Poisson sum.
    std::vector<default_type> lgam(Jmax + 1);
    for (ssize_t j = 0; j <= Jmax; ++j)
      lgam[j] = std::lgamma(default_type(j + 1));
    std::vector<default_type> eta(N_A);
    eta[0] = E[0]; // a = 0: pure central chi, eta(0) = E_0
    for (ssize_t i = 1; i != N_A; ++i) {
      if (a_grid[i] > A_POISSON) {
        // High-SNR asymptote: the Jensen correction is negligible, so
        //   eta(a) = E[f(m/sigma)] -> f(E[m/sigma]) = f(mean_normalised(a)).
        //   (Here the exact-unbiased inverse coincides with the moment inverse.)
        eta[i] = interp_increasing(x_grid, f, mean_normalised(a_grid[i], N));
        continue;
      }
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

    // Within the tabulated range Psi = eta^{-1}. Beyond it (u >= eta(A_MAX)) the
    //   magnitude bias has effectively vanished, so the unbiased inverse reverts to
    //   the algebraic inverse nu = m (= theta_m = f^{-1}(u)), the high-SNR limit
    //   with unit local gain. This avoids the drift / overshoot of extrapolating
    //   eta^{-1}'s terminal slope. The (tiny) residual bias at the join is held
    //   constant ("bias_join") so that Psi -- and hence the derived local gain --
    //   is continuous across the join (no spurious gain spike); the resulting
    //   constant offset from nu = m is bounded by bias(A_MAX) ~ 1/(2*A_MAX).
    const default_type eta_max = eta.back();
    const default_type bias_join = interp_increasing(f, x_grid, eta_max) - A_MAX;
    for (ssize_t k = 0; k != N_U; ++k) {
      const default_type theta_m = interp_increasing(f, x_grid, u_grid[k]);
      psi[k] = (u_grid[k] >= eta_max) ? (theta_m - bias_join) : interp_increasing(eta, a_grid, u_grid[k]);
    }
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
    // As for FOI, beyond the tabulated SNR range (theta_m >= mu(A_MAX)) the bias
    //   has vanished and the inverse reverts to nu = m (theta_m), with the residual
    //   bias held constant at the join (bias_join) for gain continuity.
    const default_type mu_max = mu.back();
    const default_type bias_join = mu_max - A_MAX;
    for (ssize_t k = 0; k != N_U; ++k) {
      const default_type theta_m = interp_increasing(f, x_grid, u_grid[k]);
      if (vst_method == vst_method_t::KOAY && theta_m <= floor_mean)
        psi[k] = 0.0;
      else if (theta_m >= mu_max)
        psi[k] = theta_m - bias_join;
      else
        psi[k] = interp_increasing(mu, a_grid, theta_m);
    }
  } break;
  }
  inverse_unbiased_lut = LUT(0.0, du, std::vector<default_type>(psi));

  // --- Local gain J = d(theta_nu)/d(u) --------------------------------------
  jacobian_lut = LUT(0.0, du, derivative(psi, du));
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

} // namespace MR::Denoise::NoiseModel
