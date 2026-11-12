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

#include "denoise/precondition/noise_model/gamma_stats.h"

#include <cassert>
#include <cmath>
#include <limits>

#ifndef MRTRIX_HAVE_LGAMMA_R
#include <mutex>
#endif

#include "math/betainc.h"
#include "math/math.h"
#include "math/erfinv.h"

#ifdef MRTRIX_HAVE_LGAMMA_R
extern "C" double lgamma_r(double, int *);
#endif

namespace MR::Denoise::Precondition::NoiseModel::GammaStats {

namespace {

// log Gamma(x), thread-safely. std::lgamma() writes the sign of the result to a global (signgam),
//   so calling it concurrently is undefined; MRtrix3 core resolves this once, preferring the
//   re-entrant lgamma_r() where the platform has it and otherwise serialising on a mutex that it
//   owns. That mutex is reused here rather than a second one being introduced, so that all users
//   of std::lgamma() within the process contend on the same lock.
default_type log_gamma(const default_type x) {
#ifdef MRTRIX_HAVE_LGAMMA_R
  int sign = 0;
  return lgamma_r(x, &sign);
#else
  const std::lock_guard<std::mutex> lock(MR::Math::mutex_lgamma);
  return std::lgamma(x);
#endif
}

// Argument above which the asymptotic Bernoulli series for psi and psi' is used directly. At 20
//   the first omitted term is below 1e-14 relative for both functions, so the recurrence that
//   raises a smaller argument to this threshold costs at most twenty additions and buys full
//   double precision across the whole domain.
constexpr default_type ASYMPTOTIC_THRESHOLD = 20.0;

// Relative precision to which the incomplete-gamma series / continued fraction are summed, and
//   the iteration cap that protects against a non-terminating sum. The cap is generous because
//   the number of terms the series needs grows as sqrt(alpha), and alpha here is K * dof / 2 over
//   the whole volume series -- of order thousands for a long acquisition.
constexpr default_type SERIES_EPS = 1.0e-15;
constexpr int SERIES_ITMAX = 100000;

// Standard normal quantile, used only to seed the Wilson-Hilferty estimate of the Gamma quantile.
//   Formed from the complementary inverse error function on whichever tail p lies in, so that no
//   precision is lost to the cancellation 1 - p for p close to either bound.
default_type normal_quantile(const default_type p) {
  static const default_type sqrt2 = std::sqrt(default_type(2));
  return p <= 0.5 ? -sqrt2 * MR::Math::erfcinv(2.0 * p) : sqrt2 * MR::Math::erfcinv(2.0 * (1.0 - p));
}

// log( x^alpha e^-x / Gamma(alpha) ): the prefactor shared by the series, the continued fraction
//   and the density that drives the quantile iteration.
//
// Written this way rather than as the literal alpha*log(x) - x - log_gamma(alpha) because that
//   expression cancels catastrophically where it matters most. At alpha = 5000 and x near the
//   distribution's centre its three terms are each of magnitude ~3.8e4 and their sum is of order
//   unity, so the double-precision rounding of the terms alone leaves ~4e-12 of relative error in
//   the result -- which is precisely the accuracy the quantile is required to deliver. Regrouped
//   as below, no term is larger than the answer: alpha * (log1p(u) - u) with u = x/alpha - 1 is
//   the part that vanishes at the mode, and the Stirling excess is O(log alpha).
default_type log_gamma_density(const default_type alpha, const default_type x) {
  // alpha*log(alpha) - alpha - log Gamma(alpha), by its asymptotic series where that is accurate
  //   to machine precision, and directly below the threshold where the direct difference is of
  //   modest magnitude and so loses nothing.
  const default_type stirling_excess = [](const default_type a) -> default_type {
    if (a < 20.0)
      return a * std::log(a) - a - log_gamma(a);
    static const default_type half_log_two_pi = 0.5 * std::log(2.0 * Math::pi);
    const default_type inv = 1.0 / a;
    const default_type inv2 = inv * inv;
    // log Gamma(a) = (a-1/2) log a - a + log(2 pi)/2 + 1/(12a) - 1/(360a^3) + 1/(1260a^5)
    //                  - 1/(1680a^7) + 1/(1188a^9) - ...
    const default_type correction =
        inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 * (1.0 / 1680.0 - inv2 / 1188.0))));
    return 0.5 * std::log(a) - half_log_two_pi - correction;
  }(alpha);
  // alpha * (log(x/alpha) - x/alpha + 1), which vanishes at the mode x == alpha. Written with
  //   log1p near the mode, where log(x/alpha) and (x/alpha - 1) cancel to leading order, and with
  //   the plain logarithm away from it, where they do not and where forming x/alpha - 1 would
  //   instead be the lossy step (for x much smaller than alpha the subtraction of one leaves only
  //   the rounding of the ratio behind).
  const default_type t = x / alpha;
  const default_type shape =
      (t > 0.5 && t < 2.0) ? alpha * (std::log1p(t - 1.0) - (t - 1.0)) : alpha * (std::log(t) - t + 1.0);
  return shape + stirling_excess;
}

// P(alpha, x) by its power series, for x < alpha + 1:
//   P(a,x) = x^a e^-x / Gamma(a+1) * sum_{n>=0} x^n / ((a+1)(a+2)...(a+n))
default_type gamma_p_series(const default_type alpha, const default_type x) {
  default_type term = 1.0 / alpha;
  default_type sum = term;
  default_type denominator = alpha;
  for (int n = 0; n != SERIES_ITMAX; ++n) {
    denominator += 1.0;
    term *= x / denominator;
    sum += term;
    if (std::abs(term) < std::abs(sum) * SERIES_EPS)
      break;
  }
  return sum * std::exp(log_gamma_density(alpha, x));
}

// Q(alpha, x) = 1 - P(alpha, x) by its continued fraction, for x >= alpha + 1, evaluated with the
//   modified Lentz algorithm (the same machinery MRtrix3's regularised incomplete beta function
//   uses for the analogous job).
default_type gamma_q_continued_fraction(const default_type alpha, const default_type x) {
  const default_type tiny = std::numeric_limits<default_type>::min() / SERIES_EPS;
  default_type b = x + 1.0 - alpha;
  default_type c = 1.0 / tiny;
  default_type d = 1.0 / b;
  default_type h = d;
  for (int i = 1; i != SERIES_ITMAX; ++i) {
    const default_type an = -default_type(i) * (default_type(i) - alpha);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < tiny)
      d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny)
      c = tiny;
    d = 1.0 / d;
    const default_type delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < SERIES_EPS)
      break;
  }
  return std::exp(log_gamma_density(alpha, x)) * h;
}

} // namespace

default_type digamma(const default_type x_in) {
  assert(x_in > 0.0);
  if (!(x_in > 0.0))
    return std::numeric_limits<default_type>::quiet_NaN();
  // psi(x) = psi(x+1) - 1/x, applied until the argument reaches the asymptotic regime.
  default_type x = x_in;
  default_type result = 0.0;
  while (x < ASYMPTOTIC_THRESHOLD) {
    result -= 1.0 / x;
    x += 1.0;
  }
  // psi(x) ~ log x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)
  const default_type inv = 1.0 / x;
  const default_type inv2 = inv * inv;
  result += std::log(x) - 0.5 * inv -
            inv2 * (1.0 / 12.0 -
                    inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))));
  return result;
}

default_type trigamma(const default_type x_in) {
  assert(x_in > 0.0);
  if (!(x_in > 0.0))
    return std::numeric_limits<default_type>::quiet_NaN();
  // psi'(x) = psi'(x+1) + 1/x^2.
  default_type x = x_in;
  default_type result = 0.0;
  while (x < ASYMPTOTIC_THRESHOLD) {
    result += 1.0 / (x * x);
    x += 1.0;
  }
  // psi'(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9)
  const default_type inv = 1.0 / x;
  const default_type inv2 = inv * inv;
  result += inv * (1.0 + 0.5 * inv +
                   inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 * (1.0 / 30.0)))));
  return result;
}

default_type inverse_digamma(const default_type y) {
  // Minka (2002), "Estimating a Dirichlet distribution", Eq. (149): the two regimes of psi, its
  //   exponential behaviour for large argument and its -1/x pole at the origin, each inverted to
  //   leading order. The switch at psi(x) = -2.22 is where the two are of comparable accuracy.
  static const default_type euler_mascheroni = -digamma(1.0);
  default_type x = y >= -2.22 ? std::exp(y) + 0.5 : -1.0 / (y + euler_mascheroni);
  // Newton on psi(x) - y. psi is increasing and concave on x > 0, so the iteration is stable; the
  //   step is nonetheless halved as often as necessary to keep the iterate inside the domain,
  //   which the leading-order start can otherwise leave for y near the switch.
  for (int iteration = 0; iteration != 100; ++iteration) {
    const default_type f = digamma(x) - y;
    const default_type fprime = trigamma(x);
    if (!std::isfinite(f) || !std::isfinite(fprime) || fprime <= 0.0)
      break;
    default_type step = f / fprime;
    default_type next = x - step;
    for (int backtrack = 0; !(next > 0.0) && backtrack != 100; ++backtrack) {
      step *= 0.5;
      next = x - step;
    }
    if (!(next > 0.0))
      break;
    const default_type change = std::abs(next - x);
    x = next;
    if (change <= 1.0e-15 * x)
      break;
  }
  return x;
}

default_type gamma_p(const default_type alpha, const default_type x) {
  assert(alpha > 0.0);
  if (!(alpha > 0.0))
    return std::numeric_limits<default_type>::quiet_NaN();
  if (!(x > 0.0))
    return 0.0;
  return x < alpha + 1.0 ? gamma_p_series(alpha, x) : 1.0 - gamma_q_continued_fraction(alpha, x);
}

default_type gamma_p_inv(const default_type alpha, const default_type p) {
  assert(alpha > 0.0);
  assert(p >= 0.0 && p <= 1.0);
  if (!(alpha > 0.0) || !(p >= 0.0) || !(p <= 1.0))
    return std::numeric_limits<default_type>::quiet_NaN();
  if (p == 0.0)
    return 0.0;
  if (p == 1.0)
    return std::numeric_limits<default_type>::infinity();

  // Initial estimate.
  // - alpha >= 1: Wilson-Hilferty, which is the statement that (X/alpha)^(1/3) is approximately
  //     normal with mean 1 - 2/(9 alpha) and variance 2/(9 alpha). It is asymptotically exact as
  //     alpha grows, which is the regime the estimator spends most of its time in (alpha is
  //     K * dof / 2 summed over every volume of the series).
  // - alpha < 1: the density is unbounded at the origin and the cube-root approximation fails;
  //     the leading term of P(alpha, x) = x^alpha / Gamma(alpha+1) + O(x^(alpha+1)) is inverted
  //     instead, which is exact in the limit of small x and merely a starting point elsewhere.
  default_type x = 0.0;
  if (alpha >= 1.0) {
    const default_type z = normal_quantile(p);
    const default_type factor = 1.0 - 2.0 / (9.0 * alpha) + z * std::sqrt(2.0 / (9.0 * alpha));
    x = factor > 0.0 ? alpha * factor * factor * factor : 0.0;
  }
  if (!(x > 0.0))
    x = std::exp((std::log(p) + log_gamma(alpha + 1.0)) / alpha);
  if (!std::isfinite(x) || !(x > 0.0))
    x = alpha;

  // Bracketed Halley iteration on P(alpha, x) - p. Halley converges cubically and reaches machine
  //   precision from either start within a handful of steps; the bracket is maintained purely as a
  //   guarantee, catching the case where a step from a poor start would leave the region in which
  //   the local cubic model is valid, and substituting a bisection (or a doubling, while the upper
  //   bound is still unknown) for it.
  default_type lower = 0.0;
  default_type upper = std::numeric_limits<default_type>::infinity();
  for (int iteration = 0; iteration != 200; ++iteration) {
    const default_type error = gamma_p(alpha, x) - p;
    if (error > 0.0)
      upper = x;
    else
      lower = x;
    // The density of Gamma(alpha, 1), which is P'(x) and is what turns the error into a step.
    const default_type density = std::exp(log_gamma_density(alpha, x)) / x;
    default_type next;
    if (density > 0.0 && std::isfinite(density)) {
      const default_type newton_step = error / density;
      // P''/P' = (alpha-1)/x - 1: the second-order term that distinguishes Halley from Newton.
      const default_type curvature = (alpha - 1.0) / x - 1.0;
      const default_type denominator = 1.0 - 0.5 * newton_step * curvature;
      const default_type step = denominator != 0.0 ? newton_step / denominator : newton_step;
      next = x - step;
    } else {
      next = std::numeric_limits<default_type>::quiet_NaN();
    }
    // A proposed step of less than one unit in the last place is the iteration having converged
    //   as far as the representation allows, and must be recognised as such BEFORE the bracket is
    //   consulted: the current iterate has just become one of the bracket's own endpoints, so an
    //   unmoved "next" would otherwise be read as having escaped the bracket and be replaced by a
    //   bisection (or, with no upper bound yet established, by a doubling that discards a
    //   converged answer outright).
    if (std::isfinite(next) && next == x)
      break;
    if (!std::isfinite(next) || next <= lower || next >= upper)
      next = std::isfinite(upper) ? 0.5 * (lower + upper) : 2.0 * x;
    const default_type change = std::abs(next - x);
    x = next;
    if (change <= 1.0e-15 * x)
      break;
  }
  return x;
}

} // namespace MR::Denoise::Precondition::NoiseModel::GammaStats
