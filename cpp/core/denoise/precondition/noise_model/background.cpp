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

#include "denoise/precondition/noise_model/background.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>

#include "algo/histogram.h"
#include "algo/loop.h"
#include "algo/threaded_loop.h"
#include "exception.h"
#include "image.h"
#include "math/erfinv.h"
#include "math/gamma.h"
#include "math/math.h"
#include "math/median.h"
#include "mrtrix.h"

namespace MR::Denoise::Precondition::NoiseModel {

using Math::digamma;
using Math::gamma_p;
using Math::gamma_p_inv;
using Math::gamma_q;
using Math::gamma_q_inv;
using Math::inverse_digamma;
using Math::log_gamma;
using Math::trigamma;

Accumulation::Accumulation(const std::array<ssize_t, 3> &size, const size_t slice_axis)
    : voxels(size_t(size[0]) * size_t(size[1]) * size_t(size[2])), size(size), slice_axis(slice_axis) {
  assert(slice_axis < 3);
}

default_type Accumulation::zeroed_fraction() const {
  if (voxels.empty())
    return 0.0;
  size_t zeroed = 0;
  for (const VoxelStats &stats : voxels) {
    if (stats.count == 0)
      ++zeroed;
  }
  return default_type(zeroed) / default_type(voxels.size());
}

default_type Accumulation::censoring_threshold(const default_type min_positive) const {
  if (!std::isfinite(min_positive))
    return 0.0;
  const default_type threshold = min_positive - 0.5 * quantum;
  return threshold > 0.0 ? threshold : 0.0;
}

namespace {

// How many of the smallest distinct positive magnitudes are retained for detecting the storage
//   quantum, and the tolerance on their being integer multiples of it, relative to the quantum.
//   Float storage of k q is exact to a relative 6e-8, so for any k up to the bound below the
//   rounding error is far inside the tolerance; whereas for continuous data the chance of seven
//   successive spacings all falling within it of integer multiples of the smallest is of order
//   (2e-3)^6.
constexpr size_t num_smallest_retained = 8;
constexpr default_type quantum_tolerance = 1.0e-3;
constexpr default_type quantum_max_multiple = 1.0e3;

// Insert a value into a sorted list of at most num_smallest_retained distinct values, keeping the
//   smallest.
void retain_smallest(std::vector<default_type> &smallest, const default_type value) {
  if (smallest.size() == num_smallest_retained && !(value < smallest.back()))
    return;
  const auto position = std::lower_bound(smallest.begin(), smallest.end(), value);
  if (position != smallest.end() && *position == value)
    return;
  smallest.insert(position, value);
  if (smallest.size() > num_smallest_retained)
    smallest.pop_back();
}

// The constant of which the smallest distinct magnitudes are all integer multiples, if they are
//   evidently so: the smallest spacing between them, provided every one of them is a multiple of it
//   to within the tolerance. Zero otherwise, including where too few distinct values exist to judge.
//   Only the bottom of the distribution is examined, which is where the quantum matters (the
//   censoring threshold is set from it) and where background magnitudes are densest.
default_type detect_quantum(const std::vector<default_type> &smallest) {
  if (smallest.size() < 3)
    return 0.0;
  default_type spacing = std::numeric_limits<default_type>::infinity();
  for (size_t i = 1; i != smallest.size(); ++i)
    spacing = std::min(spacing, smallest[i] - smallest[i - 1]);
  if (!(spacing > 0.0))
    return 0.0;
  for (const default_type value : smallest) {
    const default_type multiple = value / spacing;
    if (!(multiple <= quantum_max_multiple) || std::fabs(multiple - std::round(multiple)) > quantum_tolerance)
      return 0.0;
  }
  return spacing;
}

// What every thread contributes to the image-wide statistics.
class SharedExtremes {
public:
  std::mutex mutex;
  std::vector<default_type> smallest;
};

// One spatial voxel per invocation: walk that voxel's full column of volumes and write its
//   sufficient statistics. Every voxel is visited by exactly one thread and writes only its own
//   entry, so the shared array needs no locking. The smallest distinct magnitudes seen are a
//   reduction: each thread retains its own and merges them into the shared set in its destructor,
//   which runs only after all threads have rejoined.
template <class ImageType> class AccumulateFunctor {
public:
  AccumulateFunctor(Accumulation &accumulation, std::shared_ptr<SharedExtremes> shared)
      : accumulation(accumulation), shared(std::move(shared)) {}
  AccumulateFunctor(const AccumulateFunctor &that) : accumulation(that.accumulation), shared(that.shared) {}
  ~AccumulateFunctor() {
    std::lock_guard<std::mutex> lock(shared->mutex);
    for (const default_type value : smallest)
      retain_smallest(shared->smallest, value);
  }
  void operator()(ImageType &input) {
    VoxelStats stats;
    for (auto l = Loop(input, 3)(input); l; ++l) {
      const default_type m = default_type(input.value());
      // Non-finite samples carry no information and are dropped. A zero, by contrast, is a
      //   statement that the magnitude was too small to be represented -- rounded to zero by integer
      //   storage, or clipped by the reconstruction -- and it is counted, so that the fit can treat
      //   it as a sample censored below the threshold estimated from the smallest positive magnitude
      //   (see estimate_background()). Excluding such zeros instead, as the paper's reference
      //   implementation does, deletes the smallest samples of the distribution, which are the
      //   most informative about its shape, and overestimates the degrees of freedom.
      if (!std::isfinite(m))
        continue;
      if (m == 0.0) {
        ++stats.num_zero;
        continue;
      }
      if (!(m > 0.0))
        continue;
      const default_type m2 = Math::pow2(m);
      ++stats.count;
      stats.sum_sq += m2;
      stats.sum_quad += Math::pow2(m2);
      stats.sum_log_sq += std::log(m2);
      stats.min_positive = std::min(stats.min_positive, float(m));
      retain_smallest(smallest, m);
    }
    accumulation.voxels[accumulation.index(input.index(0), input.index(1), input.index(2))] = stats;
  }

private:
  Accumulation &accumulation;
  std::shared_ptr<SharedExtremes> shared;
  std::vector<default_type> smallest;
};

} // namespace

template <class ImageType> Accumulation accumulate_background(ImageType &input, const size_t slice_axis) {
  Accumulation result({input.size(0), input.size(1), input.size(2)}, slice_axis);
  size_t num_volumes = 1;
  for (size_t axis = 3; axis < input.ndim(); ++axis)
    num_volumes *= size_t(input.size(axis));
  result.num_volumes = uint32_t(num_volumes);
  auto shared = std::make_shared<SharedExtremes>();
  ThreadedLoop("Accumulating magnitude statistics for noise distribution estimation", input, 0, 3)
      .run(AccumulateFunctor<ImageType>(result, shared), input);
  result.quantum = detect_quantum(shared->smallest);
  return result;
}

template Accumulation accumulate_background<Image<float>>(Image<float> &, const size_t);
template Accumulation accumulate_background<Image<double>>(Image<double> &, const size_t);

namespace {

// ZERO-VALUED SAMPLES AS CENSORED OBSERVATIONS.
//
// A background sample has x = m^2 / (2 sigma^2) ~ Gamma(N, 1), N = dof/2. A stored zero states only
//   that the true magnitude lay in [0, tau), tau being half the storage quantum for rounded integer
//   data or the clip level for clipped data (Accumulation::censoring_threshold()). Its exact
//   contribution to the likelihood is therefore log P(N, tau^2 / (2 sigma^2)), the probability of
//   having fallen below tau, and the log-likelihood of a selected set is
//     L(N, beta) = (N-1) sum log m^2 - sum m^2 / beta - n (N log beta + log Gamma(N))
//                  + n0 log P(N, tau^2 / beta),                          beta = 2 sigma^2,
//   over the n positive samples and n0 zeros. It depends on the data only through the sums and n0.
//
//   Excluding the zeros instead treats the positive samples as an unconditioned draw, although they
//   were selected for being at least tau. The samples deleted are the smallest, which are those
//   that carry the most information about the shape through the mean of log m^2, so N is
//   overestimated: by 21% for a half-Gaussian background stored as integers with a noise level of
//   ten quanta, which leaves 4% of samples zero. Imputing a fixed value for each zero is no
//   substitute: matching the score of the censored likelihood requires the value tau exp(-1/(2N)),
//   which depends on the N being estimated.

// log P(a, z) for z > 0, accurate where P itself would underflow, together with -- where z < a + 1
//   -- the sum S of the series P(a, z) = z^a e^{-z} / Gamma(a+1) S, which also furnishes the
//   logarithmic derivative below in closed form. On the other side P exceeds about one half, and is
//   formed from the upper tail so as not to lose the precision with which Q is known.
class LowerTail {
public:
  default_type log_p{0.0};
  default_type series{0.0};
};

LowerTail lower_tail(const default_type a, const default_type z) {
  LowerTail result;
  if (z < a + 1.0) {
    default_type sum = 1.0;
    default_type term = 1.0;
    for (int k = 1; k != 100000; ++k) {
      term *= z / (a + default_type(k));
      sum += term;
      if (term <= sum * std::numeric_limits<default_type>::epsilon())
        break;
    }
    result.series = sum;
    result.log_p = a * std::log(z) - z - log_gamma(a + 1.0) + std::log(sum);
  } else {
    result.log_p = std::log1p(-gamma_q(a, z));
  }
  return result;
}

// d log P(a, z) / d log z = z p(z) / P(z), with p the Gamma(a, 1) density. Lies in (0, a], and
//   decreases in z: the density of log x is proportional to exp(a y - e^y), which is log-concave
//   for every a, so its cumulative distribution is log-concave in y = log z too.
default_type log_tail_slope(const default_type a, const default_type z, const LowerTail &tail) {
  if (tail.series > 0.0)
    return a / tail.series;
  return std::exp(a * std::log(z) - z - log_gamma(a) - tail.log_p);
}

// log Q(a, z) = log (1 - P(a, z)), and z p(z) / Q(z) = -d log Q(a, z) / d log z, for z > 0.
default_type log_upper_tail(const default_type a, const default_type z, const LowerTail &tail) {
  return tail.series > 0.0 ? std::log1p(-std::exp(tail.log_p)) : std::log(gamma_q(a, z));
}

// Which of two likelihoods of the positive samples is maximised by fit_profiled_ml().
// - CENSORED: together with the zeros, each censored below the threshold; the estimator.
// - TRUNCATED: the positive samples alone, as a draw from the distribution truncated below the
//     threshold, with no reference to the zeros. Consistent whatever the zeros are, and used to
//     predict how many zeros there should be (see zeros_consistent()).
enum class profile_t { CENSORED, TRUNCATED };

// Censored (or truncated) maximum likelihood.
//
// The likelihood is profiled over beta at fixed N. Its derivative in beta, scaled by beta, is
//     g(beta) = sum m^2 / beta - n N - n0 h(N, tau^2 / beta),     h = d log P / d log z,
//   for the censored likelihood. This is strictly decreasing in beta (each term is, h being
//   decreasing in z and z in beta), and since 0 < h <= N it is non-negative at
//   beta = sum m^2 / ((n + n0) N) and non-positive at beta = sum m^2 / (n N): the root is unique and
//   bracketed exactly, and the Illinois variant of regula falsi locates it. For the truncated
//   likelihood, -n log Q(N, tau^2 / beta) replaces n0 log P, the last term of g becomes
//   -n z p(z) / Q(z), and the same upper end of the bracket holds; the lower end is found by
//   expansion. The profile is then maximised over log N by golden-section search within a bracket
//   expanded from the maximum-likelihood estimate of the positive samples alone.
bool fit_profiled_ml(const SampleSums &sums, const default_type start_shape, const profile_t mode, Estimate &result) {
  const bool censored = mode == profile_t::CENSORED;
  const default_type n = default_type(sums.count);
  const default_type n0 = default_type(sums.num_zero);
  const default_type sum_sq = sums.sum_sq;
  const default_type sum_log_sq = sums.sum_log_sq;
  const default_type threshold_sq = sums.threshold_sq;

  // The term of the log-likelihood contributed by the threshold, and its derivative in log beta.
  const auto tail_term = [&](const default_type shape, const default_type z) {
    const LowerTail tail = lower_tail(shape, z);
    if (censored)
      return n0 * tail.log_p;
    return -n * log_upper_tail(shape, z, tail);
  };
  const auto tail_slope = [&](const default_type shape, const default_type z) {
    const LowerTail tail = lower_tail(shape, z);
    if (censored)
      return -n0 * log_tail_slope(shape, z, tail);
    return -n * std::exp(shape * std::log(z) - z - log_gamma(shape) - log_upper_tail(shape, z, tail));
  };

  const auto scale_at = [&](const default_type shape) {
    const auto g = [&](const default_type log_scale) {
      const default_type scale = std::exp(log_scale);
      return sum_sq / scale - n * shape + tail_slope(shape, threshold_sq / scale);
    };
    default_type b = std::log(sum_sq / (n * shape));
    default_type a = censored ? std::log(sum_sq / ((n + n0) * shape)) : b - 1.0;
    default_type fa = g(a);
    if (!censored) {
      for (int i = 0; i != BackgroundParameters::maxiter_rootbracket && !(fa > 0.0); ++i) {
        a -= std::ldexp(1.0, i);
        fa = g(a);
      }
    }
    default_type fb = g(b);
    if (!(fa > 0.0))
      return std::exp(a);
    if (!(fb < 0.0))
      return std::exp(b);
    for (int iteration = 0; iteration != BackgroundParameters::maxiter_illinoisrootfind; ++iteration) {
      const default_type c = b - fb * (b - a) / (fb - fa);
      const default_type fc = g(c);
      if (fc * fb < 0.0) {
        a = b;
        fa = fb;
      } else {
        fa *= 0.5;
      }
      b = c;
      fb = fc;
      if (fb == 0.0 || std::fabs(b - a) <= BackgroundParameters::rootfind_tolerance * (1.0 + std::fabs(b)))
        break;
    }
    return std::exp(b);
  };

  const auto profile = [&](const default_type log_shape) {
    const default_type shape = std::exp(log_shape);
    const default_type scale = scale_at(shape);
    const default_type value = (shape - 1.0) * sum_log_sq - sum_sq / scale -
                               n * (shape * std::log(scale) + log_gamma(shape)) +
                               tail_term(shape, threshold_sq / scale);
    return std::isfinite(value) ? value : -std::numeric_limits<default_type>::infinity();
  };

  // Admissible range of log N, beyond which the likelihood is taken to have no maximum.
  const default_type limit = std::log(1.0e4);
  default_type step = 0.25;
  default_type a = std::log(start_shape);
  default_type fa = profile(a);
  default_type b = a + step;
  default_type fb = profile(b);
  if (!std::isfinite(fa) && !std::isfinite(fb)) {
    result.failure = "the censored likelihood could not be evaluated";
    return false;
  }
  if (fb < fa) {
    std::swap(a, b);
    std::swap(fa, fb);
    step = -step;
  }
  default_type c = b + step;
  default_type fc = profile(c);
  while (fc >= fb) {
    if (std::abs(c) > limit) {
      result.failure = "the censored likelihood has no maximum within the admissible range of degrees of freedom";
      return false;
    }
    step *= 2.0;
    a = b;
    fa = fb;
    b = c;
    fb = fc;
    c = b + step;
    fc = profile(c);
  }
  default_type lower = std::min(a, c);
  default_type upper = std::max(a, c);
  const default_type ratio = 0.5 * (std::sqrt(5.0) - 1.0);
  default_type x1 = upper - ratio * (upper - lower);
  default_type x2 = lower + ratio * (upper - lower);
  default_type f1 = profile(x1);
  default_type f2 = profile(x2);
  for (int iteration = 0;
       iteration != BackgroundParameters::maxiter_goldensectionmlfind && upper - lower > BackgroundParameters::peakfind_tolerance;
       ++iteration) {
    if (f1 >= f2) {
      upper = x2;
      x2 = x1;
      f2 = f1;
      x1 = upper - ratio * (upper - lower);
      f1 = profile(x1);
    } else {
      lower = x1;
      x1 = x2;
      f1 = f2;
      x2 = lower + ratio * (upper - lower);
      f2 = profile(x2);
    }
  }
  const default_type shape = std::exp(0.5 * (lower + upper));
  result.sigma = std::sqrt(0.5 * scale_at(shape));
  result.dof = 2.0 * shape;
  return true;
}

// Censored method of moments. Each zero's contributions to sum m^2 and sum m^4 are replaced by
//   their expectations conditional on m < tau under the current estimate, and the closed-form
//   moment equations re-solved, until the two agree. Under Gamma(N, 1),
//     E[x | x < z] = N P(N+1, z) / P(N, z),     E[x^2 | x < z] = N (N+1) P(N+2, z) / P(N, z).
//   Entering the zeros as literal zeros, the starting point of the iteration, would already be
//   adequate for rounded data, where tau is half a quantum; it is not for clipped data, where
//   tau is the clip level.
bool fit_censored_moments(const SampleSums &sums, Estimate &result) {
  const default_type n0 = default_type(sums.num_zero);
  const default_type num_samples = default_type(sums.count) + n0;
  default_type sum_sq = sums.sum_sq;
  default_type sum_quad = sums.sum_quad;
  for (int iteration = 0; iteration != 1000; ++iteration) {
    const default_type bracket = sum_quad / sum_sq - sum_sq / num_samples;
    if (!(bracket > 0.0)) {
      result.failure = "the method-of-moments variance estimate is non-positive"
                       " (the selected sample is too small, or is not background)";
      return false;
    }
    const default_type variance = 0.5 * bracket;
    const default_type shape = sum_sq / (2.0 * num_samples * variance);
    const default_type scale = 2.0 * variance;
    result.sigma = std::sqrt(variance);
    result.dof = 2.0 * shape;
    const default_type z = sums.threshold_sq / scale;
    const default_type log_p = lower_tail(shape, z).log_p;
    const default_type expected_sq = scale * shape * std::exp(lower_tail(shape + 1.0, z).log_p - log_p);
    const default_type expected_quad =
        Math::pow2(scale) * shape * (shape + 1.0) * std::exp(lower_tail(shape + 2.0, z).log_p - log_p);
    const default_type next_sq = sums.sum_sq + n0 * expected_sq;
    const default_type next_quad = sums.sum_quad + n0 * expected_quad;
    if (!std::isfinite(next_sq) || !std::isfinite(next_quad)) {
      result.failure = "the censored method-of-moments iteration produced a non-finite value";
      return false;
    }
    if (std::abs(next_sq - sum_sq) <= 1.0e-12 * next_sq && std::abs(next_quad - sum_quad) <= 1.0e-12 * next_quad)
      return true;
    sum_sq = next_sq;
    sum_quad = next_quad;
  }
  result.failure = "the censored method-of-moments iteration did not converge";
  return false;
}

} // namespace

Estimate estimate_from_sums(const SampleSums &sums, const background_method_t method) {
  Estimate result;
  if (method == background_method_t::NONE)
    return result;
  if (sums.count < 2) {
    result.failure = "fewer than two samples";
    return result;
  }
  const default_type num_samples = default_type(sums.count);
  if (sums.num_zero != 0 && !(sums.threshold_sq > 0.0)) {
    result.failure = "zero-valued samples cannot be treated as censored without a censoring threshold";
    return result;
  }

  switch (method) {

  // Method of moments (the paper's Eqs. 8 and 9). Writing N = dof/2, a background sample has
  //   E[m^2] = 2 sigma^2 N and E[m^4] = 4 sigma^4 N (N+1), so
  //     sum(m^4)/sum(m^2) - sum(m^2)/V  ->  2 sigma^2 (N+1) - 2 sigma^2 N  =  2 sigma^2,
  //   which isolates sigma with no iteration and no special functions, and the first moment then
  //   gives N. The bracket is a difference of two positive quantities and a small or contaminated
  //   sample can drive it non-positive, which is a refusal rather than a number.
  case background_method_t::MOMENTS: {
    if (sums.num_zero != 0) {
      if (!fit_censored_moments(sums, result))
        return result;
      break;
    }
    if (!(sums.sum_sq > 0.0)) {
      result.failure = "no positive magnitudes among the selected samples";
      return result;
    }
    const default_type bracket = sums.sum_quad / sums.sum_sq - sums.sum_sq / num_samples;
    if (!(bracket > 0.0)) {
      result.failure = "the method-of-moments variance estimate is non-positive"
                       " (the selected sample is too small, or is not background)";
      return result;
    }
    const default_type variance = 0.5 * bracket;
    result.sigma = std::sqrt(variance);
    result.dof = sums.sum_sq / (num_samples * variance);
    break;
  }

  // Maximum likelihood (the paper's Eqs. 11 and 12, and its Appendix A).
  //
  // With A = sum(m^2) / (2V) and B = sum(log m^2) / V, the two likelihood equations reduce to the
  //   single scalar equation in sigma
  //     f(sigma)  = psi(A/sigma^2) - B + log(2 sigma^2)                                = 0,
  //     f'(sigma) = -(2A/sigma^3) psi'(A/sigma^2) + 2/sigma,
  //   and the shape is then recovered by inverting the digamma function,
  //     dof = 2 psi^{-1}(B - log(2 sigma^2)).
  //
  // f is strictly decreasing on sigma > 0: f'(sigma) = (2/sigma) (1 - x psi'(x)) with x = A/sigma^2,
  //   and x psi'(x) > 1 for every x > 0 since psi'(x) > 1/x. It tends to log(2A) - B as sigma -> 0
  //   and to -infinity as sigma grows, so a root exists exactly when B < log(2A) -- which Jensen's
  //   inequality guarantees for any sample that is not degenerate, B being the mean of log(m^2) and
  //   log(2A) the log of their mean. The root is therefore unique and bracketable, and the Newton
  //   iteration below is run inside a bracket obtained by doubling / halving from the starting
  //   value, so that it cannot be defeated by a poor start.
  case background_method_t::ML: {
    if (sums.num_zero != 0) {
      // Started from the fit to the positive samples alone, which the censored maximum lies below.
      SampleSums positive = sums;
      positive.num_zero = 0;
      const Estimate uncensored = estimate_from_sums(positive, method);
      if (!fit_profiled_ml(sums, uncensored ? 0.5 * uncensored.dof : 1.0, profile_t::CENSORED, result))
        return result;
      break;
    }
    const default_type a_term = sums.sum_sq / (2.0 * num_samples);
    const default_type b_term = sums.sum_log_sq / num_samples;
    if (!(a_term > 0.0) || !std::isfinite(b_term)) {
      result.failure = "the selected samples do not admit a likelihood (non-positive or non-finite)";
      return result;
    }
    if (!(b_term < std::log(2.0 * a_term))) {
      result.failure = "the selected magnitudes are degenerate (all of one value), so the likelihood has no maximum";
      return result;
    }
    const auto f = [a_term, b_term](const default_type sigma) {
      return digamma(a_term / Math::pow2(sigma)) - b_term + std::log(2.0 * Math::pow2(sigma));
    };
    // The correctly-scaled root mean square of the selected magnitudes. This is the value sigma
    //   would take were there exactly one effective channel, and is a better-conditioned start
    //   than the sample standard deviation the paper names (which would in any case require a
    //   fifth accumulator, the first moment, that nothing else needs).
    default_type sigma = std::sqrt(a_term);
    // Bracket the root by expansion in the direction the sign of f indicates.
    default_type lower = sigma;
    default_type upper = sigma;
    if (f(sigma) > 0.0) {
      for (int i = 0; i != BackgroundParameters::maxiter_rootbracket && f(upper) > 0.0; ++i) {
        lower = upper;
        upper *= 2.0;
      }
    } else {
      for (int i = 0; i != BackgroundParameters::maxiter_rootbracket && f(lower) < 0.0; ++i) {
        upper = lower;
        lower *= 0.5;
      }
    }
    if (!(f(lower) >= 0.0) || !(f(upper) <= 0.0)) {
      result.failure = "the likelihood equation for the noise level could not be bracketed";
      return result;
    }
    bool converged = false;
    for (int iteration = 0; iteration != BackgroundParameters::maxiter_newtonrootfind; ++iteration) {
      const default_type x = a_term / Math::pow2(sigma);
      const default_type value = digamma(x) - b_term + std::log(2.0 * Math::pow2(sigma));
      if (value > 0.0)
        lower = sigma;
      else
        upper = sigma;
      const default_type derivative = -(2.0 * a_term / Math::pow3(sigma)) * trigamma(x) + 2.0 / sigma;
      default_type next = derivative < 0.0 ? sigma - value / derivative
                                           : std::numeric_limits<default_type>::quiet_NaN();
      if (std::isfinite(next) && next == sigma) {
        converged = true;
        break;
      }
      if (!std::isfinite(next) || next <= lower || next >= upper)
        next = 0.5 * (lower + upper);
      const default_type change = std::abs(next - sigma);
      sigma = next;
      if (change <= BackgroundParameters::rootfind_tolerance * sigma) {
        converged = true;
        break;
      }
    }
    if (!converged || !std::isfinite(sigma) || !(sigma > 0.0)) {
      result.failure = "the maximum-likelihood iteration for the noise level did not converge";
      return result;
    }
    const default_type num_channels = inverse_digamma(b_term - std::log(2.0 * Math::pow2(sigma)));
    if (!std::isfinite(num_channels) || !(num_channels > 0.0)) {
      result.failure = "the maximum-likelihood iteration for the degrees of freedom did not converge";
      return result;
    }
    result.sigma = sigma;
    result.dof = 2.0 * num_channels;
    break;
  }

  case background_method_t::NONE:
    assert(false);
    return result;
  }

  if (!std::isfinite(result.sigma) || !(result.sigma > 0.0) || !std::isfinite(result.dof) || !(result.dof > 0.0)) {
    result.failure = "the fit produced a non-finite or non-positive value";
    result.valid = false;
    return result;
  }
  result.valid = true;
  return result;
}

namespace {

// The voxels of one slice sharing a common number of usable volumes, ordered by their summed
//   squared magnitude, with prefix sums taken along that order.
//
// THIS IS WHERE ALGORITHM 1'S COST GOES. As written in the manuscript the algorithm is, per slice,
//   up to max_iter iterations x l candidate noise levels x n voxels of interval tests, followed by
//   a re-scan of the accepted voxels to form the moment sums -- of order 1e8 voxel visits per
//   slice, which would dominate the command's run time (and is presumably why the reference
//   implementation parallelises over slices). The work is entirely avoidable, EXACTLY, because the
//   acceptance test has been rearranged from "lambda_lo <= S2v / (2 sigma_c^2) <= lambda_hi" to
//   "2 sigma_c^2 lambda_lo <= S2v <= 2 sigma_c^2 lambda_hi": the candidate noise level then appears
//   only in the two bounds, so within a group of constant Kv the accepted set is a CONTIGUOUS
//   INTERVAL in S2v, located by two binary searches, and each of the four sums the estimators need
//   is one subtraction of prefix sums. The per-slice cost collapses from O(max_iter x l x n) to
//   O(n log n) for the sort plus O(max_iter x l x log n) for the searches.
//
// This is a reformulation and not an approximation: it computes the identical selected set and the
//   identical sums, which is what lets the implementation be faithful to the manuscript's iteration
//   counts rather than truncating them.
//
// Where zeros are censored, the group is that of a common number of FINITE volumes, zeros included:
//   a zero enters the summed squared magnitude of the acceptance test as zero (an error of at most
//   tau^2 per zero), and the number of positive samples, which then varies within the group, is
//   carried as a prefix sum of its own. Where they are excluded, the group is that of a common
//   number of POSITIVE volumes, as in the paper.
class Group {
public:
  Group(const uint32_t count, const bool include_zeros) : count(count), include_zeros(include_zeros) {}
  void sort_and_accumulate(const std::vector<VoxelStats> &all) {
    std::sort(voxels.begin(), voxels.end(), [&all](const size_t a, const size_t b) {
      return all[a].sum_sq < all[b].sum_sq;
    });
    const size_t n = voxels.size();
    sum_sq.resize(n);
    prefix_sq.assign(n + 1, 0.0);
    prefix_quad.assign(n + 1, 0.0);
    prefix_log_sq.assign(n + 1, 0.0);
    prefix_count.assign(n + 1, 0);
    prefix_zero.assign(n + 1, 0);
    for (size_t i = 0; i != n; ++i) {
      const VoxelStats &stats = all[voxels[i]];
      sum_sq[i] = stats.sum_sq;
      prefix_count[i + 1] = prefix_count[i] + stats.count;
      prefix_zero[i + 1] = prefix_zero[i] + (include_zeros ? stats.num_zero : 0);
      prefix_sq[i + 1] = prefix_sq[i] + stats.sum_sq;
      prefix_quad[i + 1] = prefix_quad[i] + stats.sum_quad;
      prefix_log_sq[i + 1] = prefix_log_sq[i] + stats.sum_log_sq;
    }
  }
  // Half-open index range [first, last) of the voxels whose summed squared magnitude lies within
  //   the acceptance interval implied by the candidate noise level and the current quantiles.
  std::pair<size_t, size_t> accepted(const default_type lower_bound, const default_type upper_bound) const {
    const auto first = std::lower_bound(sum_sq.begin(), sum_sq.end(), lower_bound);
    const auto last = std::upper_bound(first, sum_sq.end(), upper_bound);
    return {size_t(first - sum_sq.begin()), size_t(last - sum_sq.begin())};
  }
  uint32_t count;
  bool include_zeros;
  std::vector<size_t> voxels;
  std::vector<default_type> sum_sq;
  std::vector<default_type> prefix_sq;
  std::vector<default_type> prefix_quad;
  std::vector<default_type> prefix_log_sq;
  std::vector<size_t> prefix_count;
  std::vector<size_t> prefix_zero;
};

// The state of one candidate selection: the accepted index range within each group, and the sums
//   over all of them.
class Selection {
public:
  size_t num_voxels{0};
  SampleSums sums;
  std::vector<std::pair<size_t, size_t>> ranges;
};

Selection select(const std::vector<Group> &groups,
                 const std::vector<default_type> &lambda_lo,
                 const std::vector<default_type> &lambda_hi,
                 const default_type sigma_candidate,
                 const default_type threshold_sq) {
  Selection result;
  result.ranges.resize(groups.size());
  result.sums.threshold_sq = threshold_sq;
  const default_type scale = 2.0 * Math::pow2(sigma_candidate);
  for (size_t g = 0; g != groups.size(); ++g) {
    const Group &group = groups[g];
    const auto range = group.accepted(scale * lambda_lo[g], scale * lambda_hi[g]);
    result.ranges[g] = range;
    const size_t n = range.second - range.first;
    result.num_voxels += n;
    result.sums.count += group.prefix_count[range.second] - group.prefix_count[range.first];
    result.sums.num_zero += group.prefix_zero[range.second] - group.prefix_zero[range.first];
    result.sums.sum_sq += group.prefix_sq[range.second] - group.prefix_sq[range.first];
    result.sums.sum_quad += group.prefix_quad[range.second] - group.prefix_quad[range.first];
    result.sums.sum_log_sq += group.prefix_log_sq[range.second] - group.prefix_log_sq[range.first];
  }
  return result;
}

// One iteration's state, retained so that a limit cycle can be recognised.
//
// WHY THIS IS NECESSARY. Algorithm 1's iterate is not really the pair (sigma, dof) but the
//   DISCRETE selected set that produces it, and a discrete map can enter a cycle that no tolerance
//   on the parameters will ever certify. It does so in practice: on ordinary synthetic background
//   the iteration is regularly observed to alternate indefinitely between two selections differing
//   by a handful of voxels out of a few thousand, whose degrees of freedom differ by 0.04% and
//   whose noise levels differ by 0.2% -- comfortably converged by any standard except the paper's
//   own, which compares consecutive iterates only and therefore reads the alternation as failure.
//   Left unhandled, such slices exhaust the iteration cap and decline, discarding a perfectly good
//   estimate. Recognising the recurrence of a selection is the exact statement that the iteration
//   has reached its terminal behaviour, whatever the period of that behaviour.
class IterationState {
public:
  std::vector<std::pair<size_t, size_t>> ranges;
  size_t num_voxels{0};
  size_t num_samples{0};
  size_t num_zero{0};
  SampleSums sums;
  default_type dof{0.0};
  default_type sigma{0.0};
};

// Whether two successive iterations agree to within the tolerance, on either an absolute or a
//   relative reading of it (the paper's stopping rule is the disjunction of the two).
bool has_converged(const default_type dof,
                   const default_type sigma,
                   const default_type previous_dof,
                   const default_type previous_sigma,
                   const default_type eps) {
  const default_type d_dof = std::fabs(dof - previous_dof);
  const default_type d_sigma = std::fabs(sigma - previous_sigma);
  if (d_dof < eps && d_sigma < eps)
    return true;
  return d_dof < eps * dof && d_sigma < eps * sigma;
}

// The test of a converged selection's consistency with a single Gamma distribution (see
//   BackgroundParameters::gamma_ratio_floor). Sets SliceEstimate::gamma_ratio and gamma_ratio_z, and
//   returns why the selection fails the test, or an empty string where it passes. Both estimators
//   are evaluated on the same sums whichever of them the user chose, with the zeros censored where
//   the sums carry them, so the test is of the model actually fitted.
std::string test_gamma_consistency(SliceEstimate &slice, const BackgroundParameters &parameters) {
  const Estimate moments = estimate_from_sums(slice.sums, background_method_t::MOMENTS);
  const Estimate ml = estimate_from_sums(slice.sums, background_method_t::ML);
  if (!moments || !ml) {
    return "the consistency of the voxels selected with a single Gamma distribution could not be tested (" +
           (moments ? ml.failure : moments.failure) + ")";
  }
  const default_type shape = 0.5 * ml.dof;
  const default_type num_samples = default_type(slice.sums.count + slice.sums.num_zero);
  const default_type variance =
      std::max(0.0, 2.0 * (shape + 1.0) / shape - 1.0 / (Math::pow2(shape) * (trigamma(shape) - 1.0 / shape)));
  const default_type log_ratio = std::log(moments.dof / ml.dof);
  slice.gamma_ratio = moments.dof / ml.dof;
  slice.gamma_ratio_z = variance > 0.0 ? log_ratio / std::sqrt(variance / num_samples)
                                       : -std::numeric_limits<default_type>::infinity();
  if (slice.gamma_ratio < parameters.gamma_ratio_floor && slice.gamma_ratio_z < -parameters.gamma_ratio_z) {
    return "the voxels selected are not consistent with a single Gamma distribution (the method-of-moments and"
           " maximum-likelihood estimates of the degrees of freedom from them, " +
           str(moments.dof, 4) + " and " + str(ml.dof, 4) + ", differ by a factor of " +
           str(slice.gamma_ratio, 3) + ", " + str(-slice.gamma_ratio_z, 3) +
           " standard errors); they are likely to contain tissue signal, or a noise level that differs between"
           " volumes";
  }
  return std::string();
}

// Algorithm 1 for one slice: alternate between selecting the voxels whose summed statistic falls
//   inside the central 1-p interval of the Gamma distribution implied by the current parameters,
//   and re-estimating those parameters from the voxels selected. Zeros enter the fit as censored
//   at threshold_sq where the groups include them. A selection on which the iteration has settled
//   is then tested for consistency with the Gamma model; this is not done at every iteration,
//   since the selections on the way there are legitimately poor fits.
SliceEstimate estimate_slice(std::vector<Group> &groups,
                             const default_type initial_sigma,
                             const background_method_t method,
                             const BackgroundParameters &parameters,
                             const default_type threshold_sq) {
  SliceEstimate result;
  assert(!groups.empty());

  // On the first outer iteration the acceptance interval is that of the broadest admissible
  //   background distribution, both of its ends being evaluated at dof_min, with the noise level
  //   that centres it on the slice's lowest mode (see estimate_background()). Thereafter both ends
  //   are evaluated at the current estimate.
  default_type dof_lo = parameters.dof_min;
  default_type dof_hi = parameters.dof_min;
  std::vector<default_type> candidates(1, initial_sigma);
  std::vector<default_type> lambda_lo(groups.size());
  std::vector<default_type> lambda_hi(groups.size());
  default_type previous_dof = std::numeric_limits<default_type>::quiet_NaN();
  default_type previous_sigma = std::numeric_limits<default_type>::quiet_NaN();
  std::vector<IterationState> history;

  // Both ways of terminating successfully end here, with result already holding the selection
  //   reported.
  const auto conclude = [&result, &parameters]() {
    result.decline = test_gamma_consistency(result, parameters);
    result.outcome = result.decline.empty() ? slice_outcome_t::ESTIMATED : slice_outcome_t::FAILED;
    return result;
  };

  for (ssize_t iteration = 0; iteration != parameters.max_iter; ++iteration) {
    result.num_iterations = iteration + 1;
    // The quantiles depend on the voxel only through its number of usable volumes, so they are
    //   evaluated once per distinct value of that count rather than once per voxel.
    for (size_t g = 0; g != groups.size(); ++g) {
      const default_type volumes = default_type(groups[g].count);
      lambda_lo[g] = gamma_p_inv(volumes * dof_lo / 2.0, 0.5 * parameters.p);
      lambda_hi[g] = gamma_q_inv(volumes * dof_hi / 2.0, 0.5 * parameters.p);
    }
    // The candidate noise level is the one that accepts the most voxels: a level too small or too
    //   large places the interval away from the bulk of the background and admits few voxels,
    //   whereas the true level centres it on them.
    Selection best;
    for (const default_type candidate : candidates) {
      const Selection trial = select(groups, lambda_lo, lambda_hi, candidate, threshold_sq);
      if (trial.num_voxels > best.num_voxels)
        best = trial;
    }
    result.num_selected = best.num_voxels;
    if (best.num_voxels < parameters.min_voxels_slice) {
      result.decline = "only " + str(best.num_voxels) + " voxels were identified as containing noise alone" +
                       " (at least " + str(parameters.min_voxels_slice) + " are required)";
      return result;
    }
    const Estimate estimate = estimate_from_sums(best.sums, method);
    if (!estimate) {
      result.decline = estimate.failure;
      return result;
    }
    const default_type ceiling = parameters.dof_ceiling_factor * parameters.dof_max;
    if (estimate.dof > ceiling) {
      result.decline = "the estimate of " + str(estimate.dof, 4) + " degrees of freedom exceeds the plausibility" +
                       " ceiling of " + str(ceiling, 4) + ", indicating that what was selected is not background";
      return result;
    }
    result.dof = estimate.dof;
    result.sigma = estimate.sigma;
    result.num_samples = best.sums.count + best.sums.num_zero;
    result.num_zero = best.sums.num_zero;
    result.sums = best.sums;
    // Both ends of the interval now come from the current estimate, and the search for the noise
    //   level is narrowed to its neighbourhood.
    dof_lo = dof_hi = estimate.dof;
    candidates.resize(11);
    for (size_t i = 0; i != candidates.size(); ++i)
      candidates[i] = (0.95 + 0.01 * default_type(i)) * estimate.sigma;
    if (iteration > 0 && has_converged(estimate.dof, estimate.sigma, previous_dof, previous_sigma, parameters.eps))
      return conclude();
    previous_dof = estimate.dof;
    previous_sigma = estimate.sigma;

    // Has this exact selection been visited before? If so the iteration has closed a cycle and
    //   will now repeat forever. The states of that cycle are all equally the algorithm's answer;
    //   the one built from the most voxels is taken, that being the selection the maximisation
    //   step would itself prefer, and it is a deterministic choice rather than an arbitrary one.
    const auto recurrence = std::find_if(history.begin(), history.end(), [&best](const IterationState &state) {
      return state.ranges == best.ranges;
    });
    if (recurrence != history.end()) {
      const auto preferred = std::max_element(
          recurrence, history.end(), [](const IterationState &a, const IterationState &b) {
            return a.num_voxels < b.num_voxels;
          });
      if (preferred->num_voxels > best.num_voxels) {
        result.dof = preferred->dof;
        result.sigma = preferred->sigma;
        result.num_selected = preferred->num_voxels;
        result.num_samples = preferred->num_samples;
        result.num_zero = preferred->num_zero;
        result.sums = preferred->sums;
      }
      return conclude();
    }
    IterationState state;
    state.ranges = best.ranges;
    state.num_voxels = best.num_voxels;
    state.num_samples = result.num_samples;
    state.num_zero = result.num_zero;
    state.sums = best.sums;
    state.dof = estimate.dof;
    state.sigma = estimate.sigma;
    history.push_back(std::move(state));
  }

  result.decline = "the estimate did not converge within " + str(parameters.max_iter) + " iterations";
  return result;
}

// Recover the voxels the converged parameters accept, for the diagnostic mask. Re-derived after
//   the fact rather than carried through the iteration so that what is exported is the selection
//   implied by the values actually reported.
void mark_selected(const std::vector<Group> &groups,
                   const SliceEstimate &slice,
                   const BackgroundParameters &parameters,
                   std::vector<uint8_t> &selected) {
  if (!slice.estimated())
    return;
  for (const Group &group : groups) {
    const default_type alpha = default_type(group.count) * slice.dof / 2.0;
    const default_type scale = 2.0 * Math::pow2(slice.sigma);
    const auto range = group.accepted(scale * gamma_p_inv(alpha, 0.5 * parameters.p),
                                      scale * gamma_q_inv(alpha, 0.5 * parameters.p));
    for (size_t i = range.first; i != range.second; ++i)
      selected[group.voxels[i]] = 1;
  }
}

// The lowest mode of a slice's distribution of voxel-wise mean squared magnitudes, as the natural
//   logarithm of that mean square, or NaN with the reason where there is none.
class SliceMode {
public:
  default_type log_mean_square{std::numeric_limits<default_type>::quiet_NaN()};
  std::string failure;
};

// WHY A MODE, AND WHY THE LOWEST. Background voxels are the darkest voxels of a slice that occur in
//   number: whatever the proportions of tissue and air, they form a cluster at the bottom of the
//   distribution of voxel-wise mean squared magnitude, beneath every tissue. The lowest mode that
//   holds at least min_voxels_slice voxels -- the number below which the slice would decline anyway
//   -- therefore identifies it without any threshold set from intensities, and in particular without
//   any global statistic, which can lie within the tissue where background is scarce or below the
//   background where it is abundant.
//
// WHY ON A LOGARITHMIC SCALE. For a background voxel whose K samples have per-sample mean square
//   sigma^2 dof, the voxel-wise mean square is a Gamma variate whose relative spread depends on K
//   and dof but not on sigma, so on a logarithmic scale the background cluster has the same width
//   at every noise level and every slice. And if X ~ Gamma(alpha, theta), the density of log X is
//   proportional to exp(alpha y - e^y / theta), which is maximal where e^y = alpha theta, the mean of
//   X: the mode on this scale locates the cluster's mean exactly, not approximately.
//
// THE HISTOGRAM is MRtrix3's, with its bin width set by the Freedman-Diaconis rule (twice the
//   interquartile range times the cube root of the reciprocal of the number of voxels), which is
//   the width minimising the integrated squared error of the density estimate. Both the number of
//   voxels in the slice, set by the spatial resolution, and the spread of the slice's intensities,
//   set by the proportions of tissue and air, therefore determine the binning, and neither is
//   embedded as a constant.
//
// WHICH LOCAL MAXIMA ARE MODES. A histogram of finite counts has local maxima that are sampling
//   fluctuation, and at fine bin widths many of them. Each is tested against the higher of the two
//   minima flanking it, the difference of two Poisson counts having variance equal to their sum; the
//   least significant maximum that fails is merged into its neighbour (removed together with that
//   minimum), and the test is repeated until every maximum that remains passes. The critical value
//   is corrected for the number of local maxima initially present, so that the probability of any
//   fluctuation being accepted is mode_significance whatever the number of bins. Without that
//   correction, the far tail of the background cluster in a finely-sampled slice (beyond three
//   standard deviations of 147 456 voxels at 384 x 384, still more than min_voxels_slice voxels)
//   offers a fluctuation that passes the test somewhere. A mode's voxels are those of the bins
//   between the minima that flank it.
SliceMode lowest_mode(std::vector<default_type> &log_mean_squares, const BackgroundParameters &parameters) {
  SliceMode result;
  const size_t n = log_mean_squares.size();
  if (n < parameters.min_voxels_slice) {
    result.failure = "the slice holds only " + str(n) + " voxels with usable magnitudes";
    return result;
  }

  // Histogram::Calibrator divides by the interquartile range unguarded, and with a range far
  //   exceeding it would allocate more bins than there are voxels. Both conditions are checked here,
  //   with the quantile indices it uses, the lower quartile it reads being no greater than the one
  //   computed here.
  {
    const default_type upper = Math::quantile(log_mean_squares, 0.75);
    const default_type lower = Math::quantile(log_mean_squares, 0.25);
    const auto extremes = std::minmax_element(log_mean_squares.begin(), log_mean_squares.end());
    const default_type range = *extremes.second - *extremes.first;
    if (!(upper > lower) || !(range <= 2.0 * (upper - lower) * std::pow(default_type(n), 2.0 / 3.0))) {
      result.failure = "the voxel intensities of the slice are too uniform to form a histogram";
      return result;
    }
  }

  Algo::Histogram::Calibrator calibrator;
  for (const default_type value : log_mean_squares)
    calibrator(value);
  calibrator.finalize(1, false);
  if (calibrator.get_num_bins() == 0) {
    result.failure = "the voxel intensities of the slice are too few to form a histogram";
    return result;
  }
  Algo::Histogram::Data histogram(calibrator);
  for (const default_type value : log_mean_squares)
    histogram(value);
  const ssize_t num_bins = ssize_t(histogram.size());

  // The alternating sequence of minima and maxima of the counts. Runs of equal counts are treated as
  //   one bin, and the sequence is bounded at either end by a virtual minimum of count zero, so that a
  //   maximum at the edge of the histogram is tested against the absence of voxels beyond it.
  class Extremum {
  public:
    ssize_t first;
    ssize_t last;
    size_t count;
  };
  std::vector<Extremum> runs;
  for (ssize_t bin = 0; bin != num_bins; ++bin) {
    if (!runs.empty() && runs.back().count == histogram[size_t(bin)])
      runs.back().last = bin;
    else
      runs.push_back({bin, bin, histogram[size_t(bin)]});
  }
  std::vector<Extremum> extrema(1, {-1, -1, 0});
  for (size_t r = 0; r != runs.size(); ++r) {
    const size_t left = r == 0 ? 0 : runs[r - 1].count;
    const size_t right = r + 1 == runs.size() ? 0 : runs[r + 1].count;
    if ((runs[r].count > left && runs[r].count > right) || (runs[r].count < left && runs[r].count < right))
      extrema.push_back(runs[r]);
  }
  extrema.push_back({num_bins, num_bins, 0});
  // Maxima occupy the odd positions, minima the even ones.
  assert(extrema.size() % 2 == 1);

  const size_t num_maxima = extrema.size() / 2;
  const default_type alpha = parameters.mode_significance / default_type(num_maxima);
  const default_type critical = std::sqrt(2.0) * Math::erfcinv(2.0 * alpha);
  while (extrema.size() > 3) {
    size_t weakest = 0;
    default_type weakest_score = std::numeric_limits<default_type>::infinity();
    for (size_t k = 1; k < extrema.size(); k += 2) {
      const default_type peak = default_type(extrema[k].count);
      const default_type saddle = default_type(std::max(extrema[k - 1].count, extrema[k + 1].count));
      const default_type score = (peak - saddle) / std::sqrt(peak + saddle);
      if (score < weakest_score) {
        weakest_score = score;
        weakest = k;
      }
    }
    if (weakest_score > critical)
      break;
    // Remove the maximum together with the higher of its flanking minima, joining it to the
    //   neighbour on that side.
    const size_t saddle = extrema[weakest - 1].count >= extrema[weakest + 1].count ? weakest - 1 : weakest + 1;
    extrema.erase(extrema.begin() + ssize_t(std::min(weakest, saddle)),
                  extrema.begin() + ssize_t(std::max(weakest, saddle)) + 1);
  }

  std::vector<size_t> cumulative(size_t(num_bins) + 1, 0);
  for (ssize_t bin = 0; bin != num_bins; ++bin)
    cumulative[size_t(bin) + 1] = cumulative[size_t(bin)] + histogram[size_t(bin)];
  for (size_t k = 1; k < extrema.size(); k += 2) {
    const ssize_t first = extrema[k - 1].last + 1;
    const ssize_t last = extrema[k + 1].first - 1;
    const size_t voxels = cumulative[size_t(last) + 1] - cumulative[size_t(first)];
    if (voxels >= parameters.min_voxels_slice) {
      const default_type position = 0.5 * default_type(extrema[k].first + extrema[k].last) + 0.5;
      result.log_mean_square = calibrator.get_min() + position * calibrator.get_bin_width();
      return result;
    }
  }
  result.failure = "no mode of the distribution of voxel intensities in the slice comprises at least " +
                   str(parameters.min_voxels_slice) + " voxels, so too few can be identified as containing noise alone";
  return result;
}

// Group a slice's voxels by their number of usable volumes (see Group), sort each group and form
//   its prefix sums. The voxels are given in the order in which the slice was walked.
std::vector<Group>
make_groups(const Accumulation &accumulation, const std::vector<size_t> &indices, const bool include_zeros) {
  std::map<uint32_t, Group> by_count;
  for (const size_t index : indices) {
    const VoxelStats &stats = accumulation.voxels[index];
    const uint32_t count = stats.count + (include_zeros ? stats.num_zero : 0);
    auto it = by_count.find(count);
    if (it == by_count.end())
      it = by_count.emplace(count, Group(count, include_zeros)).first;
    it->second.voxels.push_back(index);
  }
  std::vector<Group> groups;
  groups.reserve(by_count.size());
  for (auto &entry : by_count)
    groups.push_back(std::move(entry.second));
  for (Group &group : groups)
    group.sort_and_accumulate(accumulation.voxels);
  return groups;
}

// Whether the zeros among a slice's selected samples are consistent with being censored noise.
//
// WHY THIS IS ESSENTIAL. Censoring is the exact treatment of zeros that ARE noise rounded or clipped
//   below the threshold, and a badly wrong one of zeros that are not: a reconstruction or resampling
//   that zero-fills part of the field of view, or a real-valued reconstruction that clips negative
//   values, produces zeros far in excess of what the distribution predicts below the threshold, and
//   the censored fit drives the degrees of freedom down to explain them (to 6.4 from a truth of 8
//   with only 0.5% of samples so zeroed). Excluding the zeros is unaffected by them.
//
// WHY THE PREDICTION COMES FROM THE TRUNCATED FIT. The censored fit itself cannot arbitrate, because
//   it is fitted to the zero count: where the threshold is a small fraction of the noise level,
//   P(N, z) approaches one as N approaches zero, so a fit with a tiny N reproduces any proportion of
//   zeros whatever, and duly predicts the count it was fitted to (a real-valued reconstruction with
//   negative values clipped, half its samples zero, gives 0.14 degrees of freedom from a truth of 2,
//   self-consistently). The positive samples alone, fitted as a draw from the distribution
//   truncated below the threshold, give a consistent estimate whatever the zeros are, and the
//   zeros are judged against the proportion that fit predicts, n P(N, z) among the n selected
//   samples: an observed count significantly greater than zero_excess_factor times that rejects
//   censoring.
//
// The significance is assessed by the Chernoff bound on the binomial upper tail,
//   P(X >= k) <= exp(-n KL(k/n || p)), where KL is the Kullback-Leibler divergence of two Bernoulli
//   distributions. Being a bound on the tail, it can only make the test reject less readily than an
//   exact tail would.
bool zeros_consistent(const SliceEstimate &estimate,
                      const default_type threshold,
                      const BackgroundParameters &parameters) {
  if (estimate.num_zero == 0 || estimate.num_samples == 0)
    return true;
  SampleSums positive = estimate.sums;
  positive.num_zero = 0;
  const Estimate uncensored = estimate_from_sums(positive, background_method_t::ML);
  Estimate truncated;
  if (!fit_profiled_ml(estimate.sums, uncensored ? 0.5 * uncensored.dof : 1.0, profile_t::TRUNCATED, truncated))
    return false;
  const default_type shape = 0.5 * truncated.dof;
  const default_type z = threshold * threshold / (2.0 * Math::pow2(truncated.sigma));
  const default_type rate = std::min(1.0, parameters.zero_excess_factor * std::exp(lower_tail(shape, z).log_p));
  const default_type n = default_type(estimate.num_samples);
  const default_type observed = default_type(estimate.num_zero) / n;
  if (!(observed > rate))
    return true;
  if (!(rate > 0.0))
    return false;
  default_type divergence = observed * std::log(observed / rate);
  if (observed < 1.0)
    divergence += (1.0 - observed) * std::log((1.0 - observed) / (1.0 - rate));
  return !(-n * divergence < std::log(parameters.zero_consistency_p));
}

} // namespace

default_type BackgroundResult::dof_at_slice(const ssize_t slice) const {
  assert(estimated);
  if (slice < 0 || size_t(slice) >= slices.size())
    return dof;
  return slices[slice].estimated() ? slices[slice].dof : dof;
}

size_t BackgroundResult::count(const slice_outcome_t outcome) const {
  return size_t(std::count_if(
      slices.begin(), slices.end(), [outcome](const SliceEstimate &slice) { return slice.outcome == outcome; }));
}

BackgroundResult estimate_background(const Accumulation &accumulation,
                                     const background_method_t method,
                                     const BackgroundParameters &parameters) {
  BackgroundResult result;
  result.slice_axis = accumulation.slice_axis;
  result.size = accumulation.size;
  result.zeroed_fraction = accumulation.zeroed_fraction();
  result.num_volumes = accumulation.num_volumes;
  if (method == background_method_t::NONE) {
    result.decline = "no estimation was requested";
    return result;
  }

  // A field of view whose background the reconstruction has already zeroed carries no information
  //   for this estimator, and what remains adjacent to the mask boundary is not background either.
  //   This deserves its own message because it is a property of the data that the user can do
  //   nothing about and that no amount of parameter tuning will fix.
  if (result.zeroed_fraction > parameters.max_zeroed_fraction) {
    result.decline = str(std::round(100.0 * result.zeroed_fraction)) +
                     "% of the field of view is identically zero in every volume,"
                     " which is the signature of a reconstruction that has already masked the background;"
                     " there are no background voxels from which to estimate the noise distribution";
    return result;
  }

  const size_t slice_axis = accumulation.slice_axis;
  const std::array<size_t, 2> plane_axes = slice_axis == 0   ? std::array<size_t, 2>{{1, 2}}
                                           : slice_axis == 1 ? std::array<size_t, 2>{{0, 2}}
                                                             : std::array<size_t, 2>{{0, 1}};
  const ssize_t num_slices = accumulation.size[slice_axis];
  result.slices.resize(size_t(num_slices));
  result.selected.assign(accumulation.voxels.size(), 0);

  // ZEROS. A voxel with no positive sample in any volume is excluded outright, being either outside
  //   a masked reconstruction or without information. Within the remaining voxels, zero-valued
  //   samples are treated as censored below the threshold tau where one can be determined (see
  //   Accumulation::censoring_threshold() and the discussion preceding fit_censored_ml()), from the
  //   smallest positive magnitude of the slice and the storage quantum of the image, and the
  //   censored fit of each slice is then checked for consistency with the number of zeros it
  //   selected (see zeros_consistent()). Where the check fails, or the censored fit declines, the
  //   slice is fitted afresh with its zeros excluded, as the paper's reference implementation does
  //   throughout. A slice holding no zeros is fitted exactly as it would be were zeros not
  //   considered at all.
  result.quantum = accumulation.quantum;
  size_t total_samples = 0;
  size_t usable_samples = 0;
  size_t censored_samples = 0;

  for (ssize_t slice = 0; slice != num_slices; ++slice) {
    // Voxels are grouped by their number of usable volumes, because the Gamma distribution the
    //   acceptance test is applied to has shape K dof/2 and so differs between them. Ordinarily
    //   every voxel is in one group.
    std::vector<size_t> indices;
    bool has_zeros = false;
    default_type min_positive = std::numeric_limits<default_type>::infinity();
    std::array<ssize_t, 3> position{{0, 0, 0}};
    position[slice_axis] = slice;
    for (ssize_t a = 0; a != accumulation.size[plane_axes[0]]; ++a) {
      position[plane_axes[0]] = a;
      for (ssize_t b = 0; b != accumulation.size[plane_axes[1]]; ++b) {
        position[plane_axes[1]] = b;
        const size_t index = accumulation.index(position[0], position[1], position[2]);
        const VoxelStats &stats = accumulation.voxels[index];
        if (stats.count == 0)
          continue;
        indices.push_back(index);
        has_zeros = has_zeros || stats.num_zero != 0;
        min_positive = std::min(min_positive, default_type(stats.min_positive));
      }
    }

    SliceEstimate &estimate = result.slices[size_t(slice)];
    estimate.index = slice;
    if (indices.empty()) {
      estimate.outcome = slice_outcome_t::NO_SIGNAL;
      estimate.decline = "the slice holds no data: every sample of every voxel is zero or non-finite";
      continue;
    }

    const default_type threshold = has_zeros ? accumulation.censoring_threshold(min_positive) : 0.0;
    const auto attempt = [&](const bool censor, std::vector<Group> &groups) {
      groups = make_groups(accumulation, indices, censor);
      SliceEstimate outcome;
      std::vector<default_type> log_mean_squares;
      for (const Group &group : groups) {
        for (const size_t index : group.voxels)
          log_mean_squares.push_back(std::log(accumulation.voxels[index].sum_sq / default_type(group.count)));
      }
      const SliceMode mode = lowest_mode(log_mean_squares, parameters);
      if (!std::isfinite(mode.log_mean_square)) {
        outcome.decline = mode.failure;
        return outcome;
      }
      // A background voxel's per-sample mean square is sigma^2 dof; the noise level at which the
      //   first window, evaluated at dof_min, has its mean at the mode is therefore:
      const default_type initial_sigma = std::sqrt(std::exp(mode.log_mean_square) / parameters.dof_min);
      return estimate_slice(groups, initial_sigma, method, parameters, censor ? Math::pow2(threshold) : 0.0);
    };

    std::vector<Group> groups;
    const bool censor = has_zeros && threshold > 0.0;
    estimate = attempt(censor, groups);
    if (censor && !(estimate.estimated() && zeros_consistent(estimate, threshold, parameters))) {
      estimate = attempt(false, groups);
      estimate.zeros_excluded = true;
    } else if (has_zeros && !censor) {
      estimate.zeros_excluded = true;
    }
    estimate.index = slice;
    estimate.censoring_threshold = threshold;
    if (estimate.estimated()) {
      result.num_selected_total += estimate.num_selected;
      if (estimate.zeros_excluded)
        ++result.num_slices_zeros_excluded;
      mark_selected(groups, estimate, parameters, result.selected);
      if (accumulation.num_volumes > 0) {
        for (const size_t index : indices) {
          if (result.selected[index] == 0)
            continue;
          const VoxelStats &stats = accumulation.voxels[index];
          total_samples += accumulation.num_volumes;
          usable_samples += stats.count;
          if (!estimate.zeros_excluded) {
            usable_samples += stats.num_zero;
            censored_samples += stats.num_zero;
          }
        }
      }
    }
  }

  if (total_samples > 0) {
    result.excluded_fraction = default_type(total_samples - usable_samples) / default_type(total_samples);
    result.censored_fraction = default_type(censored_samples) / default_type(total_samples);
  }

  std::vector<default_type> dofs;
  std::vector<default_type> sigmas;
  for (const SliceEstimate &slice : result.slices) {
    if (!slice.estimated())
      continue;
    dofs.push_back(slice.dof);
    sigmas.push_back(slice.sigma);
  }
  if (dofs.size() < parameters.min_slices) {
    const size_t num_no_signal = result.count(slice_outcome_t::NO_SIGNAL);
    result.decline =
        "only " + str(dofs.size()) + " of " + str(num_slices) +
        " slices yielded an estimate of the noise distribution (at least " + str(parameters.min_slices) +
        " are required)" +
        (num_no_signal > 0 ? "; " + str(num_no_signal) + " of the slices hold no data at all" : std::string());
    return result;
  }
  if (result.num_selected_total < parameters.min_voxels_total) {
    result.decline = "only " + str(result.num_selected_total) +
                     " voxels in total were identified as containing noise alone (at least " +
                     str(parameters.min_voxels_total) + " are required)";
    return result;
  }

  result.dof = Math::median(dofs);
  result.sigma = Math::median(sigmas);
  result.dof_iqr = Math::quantile(dofs, 0.75) - Math::quantile(dofs, 0.25);
  result.estimated = true;
  return result;
}

namespace {
// Thresholds of estimate_caveats(). They were set against synthetic phantoms (non-central chi noise
//   of 1 to 16 degrees of freedom; 4 to 64 volumes; ample, scarce and absent background; ghosting;
//   an in-plane gradient of noise level; radially varying degrees of freedom; clipped and
//   integer-rounded magnitudes), and have not been calibrated against real data.
//
// Relative interquartile range of the per-slice degrees of freedom, as a floor: healthy phantoms
//   of 64 x 64 voxels per slice gave at most 0.021 (and smaller ones with 4 to 6 volumes up to 0.048,
//   which the sampling term below accommodates), and those in which tissue was selected in some
//   slices at least 0.071.
constexpr default_type max_relative_dof_iqr = 0.05;
// The same, as a multiple of the spread expected from sampling variation alone, which governs where
//   slices yield few samples. The relative standard deviation of a Gamma shape estimate from n samples
//   is at most sqrt(2/n) by maximum likelihood (its large-shape limit) and about sqrt(4/n) by the
//   method of moments at the smallest admissible shape; the latter is taken, and converted to an
//   interquartile range by the factor 1.349 of the normal distribution.
constexpr default_type max_dof_iqr_sampling_ratio = 3.0;
// Proportion of excluded samples (non-finite, or zeros found inconsistent with censoring) among
//   those of the selected voxels. Exactly zero for every unclipped floating-point phantom. Before
//   zeros were treated as censored, 0.03 to 0.39 of samples excluded as zero had inflated the
//   estimate by 18% to 200%; where zeros are excluded now, that is the risk if they are nonetheless
//   low magnitudes.
constexpr default_type max_excluded_fraction = 0.01;
} // namespace

std::vector<std::string> estimate_caveats(const BackgroundResult &result) {
  std::vector<std::string> caveats;
  if (!result || !(result.dof > 0.0))
    return caveats;

  std::vector<default_type> samples;
  for (const SliceEstimate &slice : result.slices) {
    if (slice.estimated())
      samples.push_back(default_type(slice.num_selected) * default_type(result.num_volumes));
  }
  default_type threshold = max_relative_dof_iqr;
  if (!samples.empty() && result.num_volumes > 0) {
    const default_type typical = Math::median(samples);
    if (typical > 0.0)
      threshold = std::max(threshold, max_dof_iqr_sampling_ratio * 1.349 * std::sqrt(4.0 / typical));
  }
  const default_type relative_iqr = result.dof_iqr / result.dof;
  if (relative_iqr > threshold) {
    caveats.push_back("The per-slice estimates of the degrees of freedom disagree by more than sampling"
                      " variation explains (interquartile range " +
                      str(result.dof_iqr, 3) + " about a median of " + str(result.dof, 4) +
                      "). This can reflect a noise distribution that genuinely varies across slices, but"
                      " is also what results where the field of view contains too little air and the"
                      " voxels selected in some slices lie within the head, in which case the degrees of"
                      " freedom are typically underestimated; inspect the voxels selected"
                      " (-noise_voxels_out) and the per-slice estimates (-noise_dof_out)");
  }

  if (result.excluded_fraction > max_excluded_fraction) {
    caveats.push_back(str(100.0 * result.excluded_fraction, 3) +
                      "% of the samples of the voxels identified as containing noise alone are zero (or"
                      " non-finite), and were excluded from the fit" +
                      (result.num_slices_zeros_excluded > 0
                           ? " in " + str(result.num_slices_zeros_excluded) +
                                 " slices, their number being inconsistent with their being noise magnitudes"
                                 " rounded or clipped to zero"
                           : std::string()) +
                      ". Where these zeros are regions that the reconstruction or resampling filled with zeros,"
                      " excluding them is correct and the estimate is unaffected. Where they are nonetheless"
                      " low noise magnitudes (clipped, for instance, at a level that varies across the image),"
                      " the fit sees a truncated distribution, and the degrees of freedom are overestimated");
  }
  return caveats;
}

} // namespace MR::Denoise::Precondition::NoiseModel
