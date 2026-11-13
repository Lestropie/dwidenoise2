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

// Validation of the estimation of the magnitude noise model's degrees of freedom from background
// voxels (denoise/precondition/noise_model/background.{h,cpp} and its supporting special functions
// in math/gamma.{h,cpp}), following St-Jean et al. (2020).
//
// Fitting the degrees of freedom to the residual of a filtered PCA reconstruction was found not to
// work at ordinary signal ranks. The estimator tested here sees raw magnitudes instead, with no PCA,
// no filter and no variance-stabilising transform between it and the data, and it passes.
//
// Four groups of checks, in increasing order of what they exercise:
//
// 1. THE SAMPLER in ncchi_sampler.h, from which every synthetic magnitude below is drawn: its
//    fractional-dof path must reproduce the integer-channel definition wherever both apply, and
//    both must match the closed-form moments of the non-central chi.
//
// 2. THE SPECIAL FUNCTIONS. Deterministic identities to fixed tolerances, the decisive one being
//    that P(alpha, P^{-1}(alpha, p)) reproduces p to 1e-12 across alpha in [0.5, 5000] and p in
//    [1e-6, 1 - 1e-6]. The Gamma quantile is the one genuinely new routine and the one the
//    algorithm depends on most directly; it is measured here at a worst case of ~3e-15.
//
// 3. THE TWO ESTIMATORS ON PURE BACKGROUND, drawn from the sampler validated in group 1. Both are
//    unbiased to well under one percent at every value of the degrees of freedom tested, and the
//    maximum-likelihood equations have the lower scatter at every one of them -- the paper's claim,
//    checked rather than assumed.
//
// 4. ALGORITHM 1 END TO END on synthetic 4D phantoms whose background distribution is known: a
//    signal-bearing ellipsoid over background whose noise level varies twofold across slices.
//    Measured on the clean phantom: 8.07 degrees of freedom recovered against a truth of 8, no
//    voxel of the signal region selected, and 95.0% of the background selected -- exactly the
//    nominal acceptance rate. The residual ~1% positive bias is the truncation the acceptance
//    interval imposes and is consistent across every value tested. The phantom is also used to
//    check the paper's principal robustness claim (a ghost band and a spike are rejected, and the
//    estimate is unmoved), the identifiable results it reports (a Rician truth recovers one
//    effective channel, a half-Gaussian truth recovers a half), that degenerate inputs decline
//    rather than answer, the expectation that interpolation of the complex data before the
//    magnitude is formed leaves the degrees of freedom unchanged while altering only the noise
//    level (which, if it holds, makes this estimator valid on data for which Marchenko-Pastur
//    estimation is not), and that neither scarce or abundant background nor in-plane matrices from
//    24 x 24 to 256 x 256 disturb the per-slice initialisation.
//
// 5. ZERO-VALUED SAMPLES. Zeros produced by integer rounding or by clipping are treated as samples
//    censored below a threshold, and the censored fits recover the truth where excluding the zeros
//    overestimates it; the storage quantum from which the threshold is set is detected; and zeros
//    that are not noise (scattered at random, or whole regions zero-filled in some volumes) are
//    recognised as inconsistent with censoring and excluded instead, with the caveat raised.
//
// 6. CONSISTENCY WITH A SINGLE GAMMA DISTRIBUTION. Pure background passes the test of the converged
//    selection at every number of volumes and degree of freedom; a slice in which the iteration has
//    converged on tissue, for want of background, declines by it; and so does background whose
//    noise level differs between volumes. Each of the latter two is shown to give a wrong estimate
//    with the test disabled.
//
// Modes:
//   (no arguments)     regression test; non-zero exit status on failure.
//   --evidence         print the full tables, including the larger sample sizes.
//   --sweep <file.tsv> the evidence as a machine-readable table as well.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "algo/loop.h"
#include "denoise/precondition/noise_model/background.h"
#include "header.h"
#include "image.h"
#include "math/gamma.h"
#include "math/noncentral_chi.h"
#include "ncchi_sampler.h"

using MR::Denoise::Precondition::NoiseModel::accumulate_background;
using MR::Denoise::Precondition::NoiseModel::Accumulation;
using MR::Denoise::Precondition::NoiseModel::background_method_t;
using MR::Denoise::Precondition::NoiseModel::BackgroundParameters;
using MR::Denoise::Precondition::NoiseModel::BackgroundResult;
using MR::Denoise::Precondition::NoiseModel::Estimate;
using MR::Denoise::Precondition::NoiseModel::estimate_background;
using MR::Denoise::Precondition::NoiseModel::estimate_caveats;
using MR::Denoise::Precondition::NoiseModel::estimate_from_sums;
using MR::Denoise::Precondition::NoiseModel::SampleSums;
using MR::Denoise::Precondition::NoiseModel::SliceEstimate;
using MR::Denoise::Precondition::NoiseModel::VoxelStats;
using MR::Math::digamma;
using MR::Math::gamma_p;
using MR::Math::gamma_p_inv;
using MR::Math::gamma_q;
using MR::Math::gamma_q_inv;
using MR::Math::inverse_digamma;
using MR::Math::trigamma;
using MR::Math::NonCentralChi::normalised_mean;
using MR::Math::NonCentralChi::normalised_variance;
using MR::Testing::NCChi::sample_fractional;
using MR::Testing::NCChi::sample_integer;
using MR::Testing::NCChi::sample_magnitude;

namespace {

using RNG = std::mt19937_64;

int failures = 0;

void check(const bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

void check_near(const double value, const double target, const double tolerance, const std::string &message) {
  if (!(std::fabs(value - target) <= tolerance)) {
    std::fprintf(stderr,
                 "FAIL: %s (got %.12g, expected %.12g +/- %.3g)\n",
                 message.c_str(),
                 value,
                 target,
                 tolerance);
    ++failures;
  }
}

// Agreement to within a fraction of the target, which is what almost every assertion below wants:
//   the quantities span four orders of magnitude and an absolute tolerance would mean nothing.
void check_relative(const double value, const double target, const double fraction, const std::string &message) {
  check_near(value, target, fraction * std::fabs(target), message);
}

// Agreement of a quantity formed by DIFFERENCING two nearly equal values, whose tolerance cannot be
//   keyed to the difference alone: at x = 1e4, psi(x+1) - psi(x) is 1e-4 formed from two numbers of
//   order 9.2, so the rounding of psi's own output is 1e-12 of the answer however accurately psi is
//   computed. "scale" is the magnitude of the values being differenced, and what the check then
//   asserts is that psi is accurate to 1e-14 of itself, which is the real claim.
void check_difference(const double value, const double target, const double scale, const std::string &message) {
  check_near(value, target, 1.0e-12 * std::fabs(target) + 1.0e-14 * std::fabs(scale), message);
}

double median_of(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double quantile_of(std::vector<double> values, const double q) {
  std::sort(values.begin(), values.end());
  const double position = q * double(values.size() - 1);
  const size_t below = size_t(std::floor(position));
  const size_t above = std::min(below + 1, values.size() - 1);
  const double weight = position - double(below);
  return (1.0 - weight) * values[below] + weight * values[above];
}

double mean_of(const std::vector<double> &values) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

double variance_of(const std::vector<double> &values) {
  if (values.size() < 2)
    return std::numeric_limits<double>::quiet_NaN();
  const double mu = mean_of(values);
  double acc = 0.0;
  for (const double value : values)
    acc += (value - mu) * (value - mu);
  return acc / double(values.size() - 1);
}

// ---------------------------------------------------------------------------
// 1. The sampler
// ---------------------------------------------------------------------------

// The fractional-degrees-of-freedom sampler must agree with the definition wherever the definition
//   applies, and both must agree with the closed-form moments the noise model is built on. Without
//   this, everything below would be validating the estimator against a distribution it invented.
void test_samplers() {
  RNG rng(12345);
  constexpr int num_samples = 400000;
  for (const int channels : {1, 2, 4}) {
    const double dof = 2.0 * double(channels);
    for (const double a : {0.0, 0.5, 1.0, 2.0, 4.0}) {
      std::vector<double> direct(num_samples);
      std::vector<double> mixture(num_samples);
      for (int i = 0; i != num_samples; ++i) {
        direct[size_t(i)] = sample_integer(a, channels, 1.0, rng);
        mixture[size_t(i)] = sample_fractional(a, dof, 1.0, rng);
      }
      const double mean_expected = normalised_mean(a, dof);
      const double variance_expected = normalised_variance(a, dof);
      const double tolerance_mean = 4.0 * std::sqrt(variance_expected / double(num_samples));
      const double tolerance_variance = 6.0 * variance_expected * std::sqrt(2.0 / double(num_samples));
      const std::string context = " (dof " + std::to_string(dof) + ", a " + std::to_string(a) + ")";
      check(std::fabs(mean_of(direct) - mean_expected) < tolerance_mean, "integer sampler mean" + context);
      check(std::fabs(mean_of(mixture) - mean_expected) < tolerance_mean, "Poisson-mixture sampler mean" + context);
      check(std::fabs(variance_of(direct) - variance_expected) < tolerance_variance,
            "integer sampler variance" + context);
      check(std::fabs(variance_of(mixture) - variance_expected) < tolerance_variance,
            "Poisson-mixture sampler variance" + context);
      // Two-sample Kolmogorov-Smirnov: the two must agree in distribution, not merely in their
      //   first two moments.
      std::sort(direct.begin(), direct.end());
      std::sort(mixture.begin(), mixture.end());
      double ks = 0.0;
      size_t i = 0;
      size_t j = 0;
      while (i < direct.size() && j < mixture.size()) {
        if (direct[i] <= mixture[j])
          ++i;
        else
          ++j;
        ks = std::max(ks, std::fabs(double(i) / double(direct.size()) - double(j) / double(mixture.size())));
      }
      check(ks < 1.95 * std::sqrt(2.0 / double(num_samples)), "the two samplers agree in distribution" + context);
    }
  }
  for (const double dof : {1.0, 3.7, 7.52}) {
    for (const double a : {0.0, 1.0, 3.0}) {
      std::vector<double> sample(num_samples);
      for (int i = 0; i != num_samples; ++i)
        sample[size_t(i)] = sample_fractional(a, dof, 1.0, rng);
      const double variance_expected = normalised_variance(a, dof);
      const std::string context = " (dof " + std::to_string(dof) + ", a " + std::to_string(a) + ")";
      check(std::fabs(mean_of(sample) - normalised_mean(a, dof)) <
                4.0 * std::sqrt(variance_expected / double(num_samples)),
            "fractional sampler mean" + context);
      check(std::fabs(variance_of(sample) - variance_expected) <
                6.0 * variance_expected * std::sqrt(2.0 / double(num_samples)),
            "fractional sampler variance" + context);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. The special functions
// ---------------------------------------------------------------------------

void test_special_functions(std::FILE *out) {
  constexpr double euler_mascheroni = 0.57721566490153286061;
  constexpr double pi = 3.14159265358979323846;

  check_near(digamma(1.0), -euler_mascheroni, 1.0e-14, "psi(1) = -gamma");
  check_near(digamma(0.5), -euler_mascheroni - 2.0 * std::log(2.0), 1.0e-14, "psi(1/2) = -gamma - 2 log 2");
  check_near(trigamma(1.0), pi * pi / 6.0, 1.0e-14, "psi'(1) = pi^2/6");
  check_near(trigamma(0.5), pi * pi / 2.0, 1.0e-14, "psi'(1/2) = pi^2/2");

  // The recurrences, which pin both functions across the whole operating range independently of
  //   the asymptotic series each is ultimately evaluated from.
  for (double exponent = -3.0; exponent <= 4.0001; exponent += 0.25) {
    const double x = std::pow(10.0, exponent);
    check_difference(digamma(x + 1.0) - digamma(x), 1.0 / x, digamma(x), "psi(x+1) - psi(x) = 1/x");
    check_difference(
        trigamma(x) - trigamma(x + 1.0), 1.0 / (x * x), trigamma(x), "psi'(x) - psi'(x+1) = 1/x^2");
    check_relative(inverse_digamma(digamma(x)), x, 1.0e-12, "inverse_digamma(psi(x)) = x");
  }

  // P(alpha, x) against the two closed forms it has.
  for (double x = 0.01; x < 20.0; x *= 1.3) {
    check_near(gamma_p(1.0, x), 1.0 - std::exp(-x), 1.0e-13, "P(1, x) = 1 - exp(-x)");
    check_near(gamma_p(0.5, x), std::erf(std::sqrt(x)), 1.0e-13, "P(1/2, x) = erf(sqrt x)");
  }
  check_near(gamma_p(3.0, 0.0), 0.0, 0.0, "P(alpha, 0) = 0");
  check(gamma_p(3.0, 1.0e6) == 1.0, "P(alpha, x) -> 1");
  {
    double previous = -1.0;
    bool monotone = true;
    for (double x = 0.01; x < 60.0; x *= 1.05) {
      const double value = gamma_p(4.0, x);
      monotone = monotone && value >= previous;
      previous = value;
    }
    check(monotone, "P(alpha, x) is monotone in x");
  }

  // The assertion that matters: the quantile is the one genuinely new routine, and the one the
  //   algorithm depends on most directly.
  double worst = 0.0;
  double worst_alpha = 0.0;
  double worst_p = 0.0;
  const double log_alpha_min = std::log(0.5);
  const double log_alpha_max = std::log(5000.0);
  for (int i = 0; i != 41; ++i) {
    const double alpha = std::exp(log_alpha_min + (log_alpha_max - log_alpha_min) * double(i) / 40.0);
    for (double exponent = -6.0; exponent <= -0.24; exponent += 0.25) {
      const double tail = std::pow(10.0, exponent);
      for (const double p : {tail, 1.0 - tail}) {
        if (!(p > 0.0) || !(p < 1.0))
          continue;
        const double error = std::fabs(gamma_p(alpha, gamma_p_inv(alpha, p)) - p);
        if (error > worst) {
          worst = error;
          worst_alpha = alpha;
          worst_p = p;
        }
      }
    }
  }
  check(worst < 1.0e-12, "P(alpha, P^{-1}(alpha, p)) = p to 1e-12");
  std::fprintf(out,
               "Gamma quantile round trip: worst |P(alpha, P^-1(alpha, p)) - p| = %.3g"
               " (at alpha = %.4g, p = %.6g)\n",
               worst,
               worst_alpha,
               worst_p);

  // Q(alpha, x): complementary to P, and accurate RELATIVE to its own value deep in the upper tail,
  //   where 1 - P(alpha, x) has no significant digits left.
  for (const double alpha : {0.5, 1.0, 4.0, 60.0, 3000.0}) {
    for (double x = 0.01 * alpha; x < 3.0 * alpha + 10.0; x *= 1.2)
      check_near(gamma_p(alpha, x) + gamma_q(alpha, x), 1.0, 1.0e-14, "P(alpha, x) + Q(alpha, x) = 1");
  }
  check(gamma_q(3.0, 0.0) == 1.0, "Q(alpha, 0) = 1");
  for (const double x : {40.0, 200.0, 700.0})
    check_relative(gamma_q(1.0, x), std::exp(-x), 1.0e-13, "Q(1, x) = exp(-x)");
  for (const double x : {10.0, 30.0})
    check_relative(gamma_q(0.5, x), std::erfc(std::sqrt(x)), 1.0e-12, "Q(1/2, x) = erfc(sqrt x)");

  // The upper-tail quantile, assessed relative to the tail probability itself; this includes tail
  //   probabilities far below the resolution of 1 - q, which gamma_p_inv() could not even be asked for.
  worst = 0.0;
  worst_alpha = 0.0;
  double worst_q = 0.0;
  for (int i = 0; i != 41; ++i) {
    const double alpha = std::exp(log_alpha_min + (log_alpha_max - log_alpha_min) * double(i) / 40.0);
    for (double exponent = -200.0; exponent <= -0.31; exponent += exponent < -10.0 ? 10.0 : 0.25) {
      const double q = std::pow(10.0, exponent);
      const double error = std::fabs(gamma_q(alpha, gamma_q_inv(alpha, q)) - q) / q;
      if (error > worst) {
        worst = error;
        worst_alpha = alpha;
        worst_q = q;
      }
    }
  }
  check(worst < 1.0e-10, "Q(alpha, Q^{-1}(alpha, q)) = q to 1e-10 relative");
  std::fprintf(out,
               "Gamma upper quantile round trip: worst |Q(alpha, Q^-1(alpha, q)) - q| / q = %.3g"
               " (at alpha = %.4g, q = %.3g)\n",
               worst,
               worst_alpha,
               worst_q);
  // (Shapes below one are omitted only because their quantile at these probabilities underflows.)
  for (const double alpha : {1.5, 4.0, 300.0, 5000.0}) {
    for (const double p : {1.0e-30, 1.0e-100, 1.0e-200})
      check_relative(gamma_p(alpha, gamma_p_inv(alpha, p)), p, 1.0e-10, "P(alpha, P^{-1}(alpha, p)) = p deep in the lower tail");
  }
  for (const double alpha : {0.7, 3.0, 250.0}) {
    for (const double q : {0.3, 0.025, 1.0e-4})
      check_relative(gamma_q_inv(alpha, q), gamma_p_inv(alpha, 1.0 - q), 1.0e-12, "Q^{-1}(alpha, q) = P^{-1}(alpha, 1 - q)");
  }
}

// ---------------------------------------------------------------------------
// 3. The two estimators on pure background
// ---------------------------------------------------------------------------

class CellResult {
public:
  double bias_dof{0.0};
  double iqr_dof{0.0};
  double bias_sigma{0.0};
  double within_ten_percent{0.0};
  size_t refusals{0};
};

CellResult evaluate_cell(const double dof_true,
                         const double sigma_true,
                         const int num_samples,
                         const int num_repetitions,
                         const background_method_t method) {
  // Seeded from the cell's own parameters so that every cell is reproducible in isolation and no
  //   two cells share a stream.
  RNG rng(20260908u + unsigned(num_samples) + 977u * unsigned(std::lround(100.0 * dof_true)) +
          7919u * unsigned(method == background_method_t::ML ? 1 : 0));
  std::vector<double> relative_dof;
  std::vector<double> relative_sigma;
  CellResult result;
  int within = 0;
  for (int repetition = 0; repetition != num_repetitions; ++repetition) {
    SampleSums sums;
    for (int sample = 0; sample != num_samples; ++sample) {
      const double m = sample_magnitude(0.0, dof_true, sigma_true, rng);
      const double m2 = m * m;
      ++sums.count;
      sums.sum_sq += m2;
      sums.sum_quad += m2 * m2;
      sums.sum_log_sq += std::log(m2);
    }
    const Estimate estimate = estimate_from_sums(sums, method);
    if (!estimate) {
      ++result.refusals;
      continue;
    }
    relative_dof.push_back(estimate.dof / dof_true - 1.0);
    relative_sigma.push_back(estimate.sigma / sigma_true - 1.0);
    if (std::fabs(estimate.dof / dof_true - 1.0) <= 0.10)
      ++within;
  }
  if (relative_dof.empty())
    return result;
  result.bias_dof = median_of(relative_dof);
  result.iqr_dof = quantile_of(relative_dof, 0.75) - quantile_of(relative_dof, 0.25);
  result.bias_sigma = median_of(relative_sigma);
  result.within_ten_percent = double(within) / double(num_repetitions);
  return result;
}

void test_estimators(std::FILE *out, std::FILE *tsv, const bool evidence) {
  // Spanning the half-Gaussian that partial-Fourier homodyne reconstruction produces, the Rician
  //   case, the one measured effective channel count in existence (3.76 channels), and beyond the
  //   upper bound of the rejection window.
  const std::vector<double> dof_values{1.0, 2.0, 4.0, 7.52, 8.0, 16.0, 24.0};
  // One noise level suffices for the grid: both estimators are exactly equivariant under a change
  //   of scale, which is asserted separately and much more sharply below.
  const std::vector<double> sigma_values{1.0};
  struct Cell {
    int num_samples;
    int num_repetitions;
  };
  std::vector<Cell> cells{{1000, 200}, {10000, 60}};
  if (evidence)
    cells.push_back({100000, 20});

  std::fprintf(out,
               "\n%-8s %-7s %-7s %-8s %9s %8s %10s %9s %6s\n",
               "method",
               "dof",
               "sigma",
               "n",
               "bias(dof)",
               "IQR",
               "bias(sigma)",
               "within10%",
               "refuse");
  for (const double dof_true : dof_values) {
    for (const double sigma_true : sigma_values) {
      for (const Cell &cell : cells) {
        double moments_iqr = 0.0;
        double ml_iqr = 0.0;
        for (const background_method_t method : {background_method_t::MOMENTS, background_method_t::ML}) {
          const char *const name = method == background_method_t::ML ? "ml" : "moments";
          const CellResult cell_result =
              evaluate_cell(dof_true, sigma_true, cell.num_samples, cell.num_repetitions, method);
          std::fprintf(out,
                       "%-8s %-7.4g %-7.4g %-8d %+9.4f %8.4f %+10.5f %9.3f %6zu\n",
                       name,
                       dof_true,
                       sigma_true,
                       cell.num_samples,
                       cell_result.bias_dof,
                       cell_result.iqr_dof,
                       cell_result.bias_sigma,
                       cell_result.within_ten_percent,
                       cell_result.refusals);
          if (tsv != nullptr) {
            const char *const keys[4] = {"bias_dof", "iqr_dof", "bias_sigma", "within_ten_percent"};
            const double values[4] = {cell_result.bias_dof,
                                      cell_result.iqr_dof,
                                      cell_result.bias_sigma,
                                      cell_result.within_ten_percent};
            for (int k = 0; k != 4; ++k)
              std::fprintf(tsv,
                           "estimators\t%s\t%g\t%g\t%d\t%s\t%.8g\n",
                           name,
                           dof_true,
                           sigma_true,
                           cell.num_samples,
                           keys[k],
                           values[k]);
          }
          const std::string label = std::string(name) + " dof=" + std::to_string(dof_true) +
                                    " sigma=" + std::to_string(sigma_true) +
                                    " n=" + std::to_string(cell.num_samples);
          check(cell_result.refusals == 0, label + ": every repetition yields an estimate");
          // Assert on the BIAS, which is a property of the estimator, rather than on the scatter,
          //   which is a property of the sample size. Both estimators measure well inside one
          //   percent at n = 1000 and inside a quarter of a percent at n = 10000.
          const double bias_tolerance = cell.num_samples >= 10000 ? 0.015 : 0.03;
          check(std::fabs(cell_result.bias_dof) <= bias_tolerance,
                label + ": median relative bias of dof is " + std::to_string(cell_result.bias_dof));
          check(std::fabs(cell_result.bias_sigma) <= 0.02,
                label + ": median relative bias of sigma is " + std::to_string(cell_result.bias_sigma));
          const double within_floor = cell.num_samples >= 10000 ? 0.98 : 0.75;
          check(cell_result.within_ten_percent >= within_floor,
                label + ": fraction of repetitions within 10% of the truth is " +
                    std::to_string(cell_result.within_ten_percent));
          (method == background_method_t::ML ? ml_iqr : moments_iqr) = cell_result.iqr_dof;
        }
        // The paper's claim that the maximum-likelihood equations have the lower variance, checked
        //   rather than assumed. Asserted only where the repetition count is enough for an
        //   interquartile range to be worth comparing, and with a margin, because the two converge
        //   as the degrees of freedom grow: the measured ratio runs from 0.44 at dof = 1 to 0.98 at
        //   dof = 16.
        if (cell.num_repetitions >= 200)
          check(ml_iqr <= 1.15 * moments_iqr,
                "maximum likelihood has no more scatter than the method of moments at dof=" +
                    std::to_string(dof_true) + " n=" + std::to_string(cell.num_samples));
      }
    }
  }
}

// Both estimators are exactly equivariant under a change of scale: multiplying every magnitude by
//   a constant multiplies sigma by that constant and leaves the degrees of freedom untouched. It is
//   worth asserting sharply rather than sampling loosely, because a scale dependence would be the
//   signature of an accumulator overflowing or losing precision on data stored at large intensities
//   -- the failure mode the paper guards against by normalising by the dataset median, and which
//   accumulating in double precision is relied upon here to make impossible.
void test_scale_equivariance(std::FILE *out) {
  for (const background_method_t method : {background_method_t::MOMENTS, background_method_t::ML}) {
    const char *const name = method == background_method_t::ML ? "ml" : "moments";
    const CellResult small = evaluate_cell(7.52, 1.0e-3, 10000, 20, method);
    const CellResult large = evaluate_cell(7.52, 1.0e+3, 10000, 20, method);
    std::fprintf(out,
                 "scale equivariance (%-7s): bias(dof) %+0.8f vs %+0.8f, bias(sigma) %+0.8f vs %+0.8f\n",
                 name,
                 small.bias_dof,
                 large.bias_dof,
                 small.bias_sigma,
                 large.bias_sigma);
    check(std::fabs(small.bias_dof - large.bias_dof) <= 1.0e-9,
          std::string(name) + ": the degrees of freedom are invariant under a 1e6-fold change of scale");
    check(std::fabs(small.bias_sigma - large.bias_sigma) <= 1.0e-9,
          std::string(name) + ": the noise level scales exactly with the data");
  }
}

// ---------------------------------------------------------------------------
// 4. Algorithm 1 end to end
// ---------------------------------------------------------------------------

class PhantomSpec {
public:
  ssize_t nx{64};
  ssize_t ny{64};
  ssize_t nz{8};
  ssize_t nvol{20};
  double dof{8.0};
  // The noise level at the first slice, and the factor by which it rises across the slices. The
  //   whole reason the paper's algorithm works per slice is that sigma genuinely varies between
  //   them, so a phantom in which it does not would not test the thing.
  double sigma{10.0};
  double sigma_ratio{2.0};
  // Signal-to-noise ratio within the ellipsoid, and its semi-axes as fractions of the field of view.
  double snr{20.0};
  double semi_axis_x{0.45};
  double semi_axis_y{0.30};
  // A band of elevated background intensity across two slices (ghosting), and a bright patch in
  //   two volumes of one slice (a spike). Both are placed OUTSIDE the ellipsoid, which is where
  //   they have to be for the rejection step to be what is under test.
  bool ghost{false};
  bool spike{false};
  // Zero the background outright, as a reconstruction that masks it would.
  bool zero_background{false};
};

bool in_signal(const PhantomSpec &spec, const ssize_t x, const ssize_t y) {
  const double dx = (double(x) - 0.5 * double(spec.nx - 1)) / (spec.semi_axis_x * double(spec.nx));
  const double dy = (double(y) - 0.5 * double(spec.ny - 1)) / (spec.semi_axis_y * double(spec.ny));
  return dx * dx + dy * dy <= 1.0;
}

bool in_ghost(const PhantomSpec &spec, const ssize_t x, const ssize_t y, const ssize_t z) {
  return spec.ghost && (z == 1 || z == 2) && y >= 4 && y < 8 && !in_signal(spec, x, y);
}

bool in_spike(const PhantomSpec &spec, const ssize_t x, const ssize_t y, const ssize_t z, const ssize_t volume) {
  return spec.spike && z == 4 && (volume == 3 || volume == 7) && x >= 4 && x < 12 && y >= 4 && y < 12 &&
         !in_signal(spec, x, y);
}

double slice_sigma(const PhantomSpec &spec, const ssize_t z) {
  return spec.sigma * (1.0 + (spec.sigma_ratio - 1.0) * double(z) / double(spec.nz - 1));
}

MR::Header phantom_header(const PhantomSpec &spec) {
  MR::Header H;
  H.ndim() = 4;
  H.size(0) = spec.nx;
  H.size(1) = spec.ny;
  H.size(2) = spec.nz;
  H.size(3) = spec.nvol;
  for (size_t axis = 0; axis != 4; ++axis) {
    H.spacing(axis) = 1.0;
    H.stride(axis) = ssize_t(axis) + 1;
  }
  H.transform().setIdentity();
  H.datatype() = MR::DataType::from<float>();
  return H;
}

MR::Image<float> make_phantom(const PhantomSpec &spec, const unsigned seed) {
  MR::Image<float> image = MR::Image<float>::scratch(phantom_header(spec), "background estimation phantom");
  RNG rng(seed);
  for (ssize_t z = 0; z != spec.nz; ++z) {
    image.index(2) = z;
    const double sigma = slice_sigma(spec, z);
    for (ssize_t y = 0; y != spec.ny; ++y) {
      image.index(1) = y;
      for (ssize_t x = 0; x != spec.nx; ++x) {
        image.index(0) = x;
        const bool signal = in_signal(spec, x, y);
        const bool ghost = in_ghost(spec, x, y, z);
        for (ssize_t volume = 0; volume != spec.nvol; ++volume) {
          image.index(3) = volume;
          const double a = signal                                ? spec.snr
                           : in_spike(spec, x, y, z, volume)     ? 10.0
                           : ghost                               ? 6.0
                                                                 : 0.0;
          const double m = sample_magnitude(a, spec.dof, sigma, rng);
          image.value() = float(spec.zero_background && !signal ? 0.0 : m);
        }
      }
    }
  }
  image.index(3) = 0;
  return image;
}

// The same phantom, but formed by half-voxel-shifting the individual complex channels along the
//   first axis before combining them by sum of squares -- the order in which a scanner applies
//   zero-fill interpolation or distortion correction.
//
// Every output component is the mean of two independent Gaussians, so its standard deviation is
//   sigma/sqrt(2) UNIFORMLY over the field of view (a position-dependent interpolation weight would
//   instead make the noise level vary within the slice, which is a different question), and the
//   magnitude remains non-central chi with the SAME degrees of freedom. Adjacent output voxels
//   share an input, so the noise is spatially correlated -- which is the property that defeats
//   Marchenko-Pastur-based estimation and which this estimator should be indifferent to.
//   Restricted to integral channel counts, the only case in which channels can be drawn explicitly.
MR::Image<float> make_resampled_phantom(const PhantomSpec &spec, const unsigned seed) {
  MR::Image<float> image = MR::Image<float>::scratch(phantom_header(spec), "resampled phantom");
  RNG rng(seed);
  const int channels = int(std::lround(0.5 * spec.dof));
  std::normal_distribution<double> normal(0.0, 1.0);
  const size_t plane = size_t(spec.nx) * size_t(spec.ny);
  std::vector<double> real(plane);
  std::vector<double> imaginary(plane);
  std::vector<double> sum_of_squares(plane);
  for (ssize_t z = 0; z != spec.nz; ++z) {
    image.index(2) = z;
    const double sigma = slice_sigma(spec, z);
    for (ssize_t volume = 0; volume != spec.nvol; ++volume) {
      image.index(3) = volume;
      std::fill(sum_of_squares.begin(), sum_of_squares.end(), 0.0);
      for (int channel = 0; channel != channels; ++channel) {
        for (ssize_t y = 0; y != spec.ny; ++y) {
          for (ssize_t x = 0; x != spec.nx; ++x) {
            const size_t i = size_t(y) * size_t(spec.nx) + size_t(x);
            real[i] = sigma * normal(rng) +
                      (channel == 0 && in_signal(spec, x, y) ? spec.snr * sigma : 0.0);
            imaginary[i] = sigma * normal(rng);
          }
        }
        for (ssize_t y = 0; y != spec.ny; ++y) {
          for (ssize_t x = 0; x != spec.nx; ++x) {
            const size_t i = size_t(y) * size_t(spec.nx) + size_t(x);
            const size_t j = size_t(y) * size_t(spec.nx) + size_t((x + 1) % spec.nx);
            const double re = 0.5 * (real[i] + real[j]);
            const double im = 0.5 * (imaginary[i] + imaginary[j]);
            sum_of_squares[i] += re * re + im * im;
          }
        }
      }
      for (ssize_t y = 0; y != spec.ny; ++y) {
        image.index(1) = y;
        for (ssize_t x = 0; x != spec.nx; ++x) {
          image.index(0) = x;
          image.value() = float(std::sqrt(sum_of_squares[size_t(y) * size_t(spec.nx) + size_t(x)]));
        }
      }
    }
  }
  image.index(3) = 0;
  return image;
}

BackgroundResult run_phantom(MR::Image<float> &image,
                             const background_method_t method,
                             const BackgroundParameters &parameters = BackgroundParameters()) {
  const Accumulation accumulation = accumulate_background(image, 2);
  return estimate_background(accumulation, method, parameters);
}

void report(std::FILE *out, const std::string &name, const PhantomSpec &spec, const BackgroundResult &result) {
  std::fprintf(out, "%-26s dof_true=%-6g ", name.c_str(), spec.dof);
  if (!result) {
    std::fprintf(out, "DECLINED: %s\n", result.decline.c_str());
    for (const auto &slice : result.slices) {
      if (!slice.decline.empty()) {
        std::fprintf(out, "%-26s   slice %2zd: %s\n", "", ssize_t(slice.index), slice.decline.c_str());
        break;
      }
    }
    return;
  }
  std::fprintf(out,
               "dof=%.4f (N=%.4f) IQR=%.4f sigma=%.4f selected=%zu\n",
               result.dof,
               0.5 * result.dof,
               result.dof_iqr,
               result.sigma,
               result.num_selected_total);
}

// Counts of what the acceptance step chose, partitioned by what the voxel actually is.
class MaskCounts {
public:
  size_t signal_selected{0};
  size_t signal_total{0};
  size_t background_selected{0};
  size_t background_total{0};
  size_t artefact_selected{0};
  size_t artefact_total{0};
};

MaskCounts count_mask(const PhantomSpec &spec, const BackgroundResult &result) {
  MaskCounts counts;
  const Accumulation geometry({spec.nx, spec.ny, spec.nz}, 2);
  for (ssize_t z = 0; z != spec.nz; ++z) {
    for (ssize_t y = 0; y != spec.ny; ++y) {
      for (ssize_t x = 0; x != spec.nx; ++x) {
        const size_t index = geometry.index(x, y, z);
        const bool selected = result.selected[index] != 0;
        // A voxel that any volume corrupts is an artefact voxel: the acceptance test pools over
        //   volumes, so a spike in two volumes of twenty is exactly what it is meant to catch.
        bool artefact = in_ghost(spec, x, y, z);
        for (ssize_t volume = 0; volume != spec.nvol && !artefact; ++volume)
          artefact = in_spike(spec, x, y, z, volume);
        if (in_signal(spec, x, y)) {
          ++counts.signal_total;
          counts.signal_selected += selected ? 1 : 0;
        } else if (artefact) {
          ++counts.artefact_total;
          counts.artefact_selected += selected ? 1 : 0;
        } else {
          ++counts.background_total;
          counts.background_selected += selected ? 1 : 0;
        }
      }
    }
  }
  return counts;
}

void test_algorithm(std::FILE *out, std::FILE *tsv) {
  std::fprintf(out, "\n");

  // (a) The clean phantom, both methods. Recovery of the degrees of freedom per slice and in the
  //     aggregate, of the per-slice noise level (which varies twofold across the volume), and of
  //     the selection itself.
  const PhantomSpec clean;
  BackgroundResult clean_ml;
  for (const background_method_t method : {background_method_t::ML, background_method_t::MOMENTS}) {
    const char *const name = method == background_method_t::ML ? "clean (ml)" : "clean (moments)";
    MR::Image<float> image = make_phantom(clean, 424242u);
    const BackgroundResult result = run_phantom(image, method);
    report(out, name, clean, result);
    check(bool(result), std::string(name) + ": an estimate is produced");
    if (!result)
      continue;
    if (method == background_method_t::ML)
      clean_ml = result;
    check_relative(result.dof, clean.dof, 0.05, std::string(name) + ": global degrees of freedom");
    for (const auto &slice : result.slices) {
      check(slice.estimated,
            std::string(name) + ": slice " + std::to_string(slice.index) + " yields an estimate (" + slice.decline + ")");
      if (!slice.estimated)
        continue;
      check_relative(slice.dof, clean.dof, 0.05, std::string(name) + ": slice degrees of freedom");
      check_relative(slice.sigma,
                     slice_sigma(clean, slice.index),
                     0.05,
                     std::string(name) + ": slice " + std::to_string(slice.index) + " noise level");
    }
    const MaskCounts counts = count_mask(clean, result);
    std::fprintf(out,
                 "%-26s   selected: signal %zu/%zu, background %zu/%zu (%.3f)\n",
                 "",
                 counts.signal_selected,
                 counts.signal_total,
                 counts.background_selected,
                 counts.background_total,
                 double(counts.background_selected) / double(counts.background_total));
    check(counts.signal_selected == 0, std::string(name) + ": no signal-bearing voxel is selected");
    check(double(counts.background_selected) / double(counts.background_total) >= 0.85,
          std::string(name) + ": the great majority of background voxels are selected");
    if (tsv != nullptr)
      std::fprintf(tsv, "phantom\t%s\t%g\t\t\tdof\t%.8g\n", name, clean.dof, result.dof);
  }

  // (b) Artefact rejection, the paper's principal robustness claim (its Figs. 9 and 10D): a ghost
  //     band across two slices and a spike in two volumes of a third. Both the exclusion of the
  //     affected voxels and the invariance of the estimate are asserted, since either alone would
  //     be satisfiable by an estimator that had simply stopped responding to the data.
  {
    PhantomSpec spec = clean;
    spec.ghost = true;
    spec.spike = true;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "artefacts (ml)", spec, result);
    check(bool(result), "artefacts: an estimate is produced");
    if (result) {
      const MaskCounts counts = count_mask(spec, result);
      std::fprintf(out,
                   "%-26s   selected: artefact %zu/%zu, background %zu/%zu\n",
                   "",
                   counts.artefact_selected,
                   counts.artefact_total,
                   counts.background_selected,
                   counts.background_total);
      check(counts.artefact_total > 0, "artefacts: the phantom does contain artefact voxels");
      check(counts.artefact_selected == 0, "artefacts: no corrupted voxel is selected");
      check(counts.signal_selected == 0, "artefacts: no signal-bearing voxel is selected");
      if (clean_ml)
        check_relative(result.dof, clean_ml.dof, 0.02, "artefacts: the estimate is unmoved by the corruption");
    }
  }

  // (c) The manuscript's identifiable results, and a fractional truth. dof = 2 is the Rician case
  //     that any phase-sensitive coil combination produces; dof = 1 is the half-Gaussian of
  //     partial-Fourier homodyne reconstruction, where the paper's GE result (0.45 - 0.49 effective
  //     channels) lands; 7.52 is the one measured effective channel count in existence.
  for (const double dof_true : {2.0, 1.0, 7.52}) {
    PhantomSpec spec = clean;
    spec.dof = dof_true;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "dof_true = " + std::to_string(dof_true), spec, result);
    check(bool(result), "dof=" + std::to_string(dof_true) + ": an estimate is produced");
    if (result)
      check_relative(result.dof, dof_true, 0.05, "dof=" + std::to_string(dof_true) + ": recovered");
    if (tsv != nullptr && result)
      std::fprintf(tsv, "phantom\tdof_sweep\t%g\t\t\tdof\t%.8g\n", dof_true, result.dof);
  }

  // (d) Degenerate inputs must decline rather than answer. Silent failure here is worse than no
  //     estimator at all: the value would go on to parameterise the noise floor of everything.
  {
    PhantomSpec spec = clean;
    spec.zero_background = true;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "zeroed background", spec, result);
    check(!result, "a masked background declines");
    check(result.decline.find("identically zero") != std::string::npos,
          "a masked background is reported as such, and not as some other failure");
  }
  {
    // A field of view too small to furnish the per-slice floor of background voxels: 144 voxels per
    //   slice of which about half are signal.
    PhantomSpec spec = clean;
    spec.nx = 12;
    spec.ny = 12;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "too few background voxels", spec, result);
    check(!result, "too few background voxels declines");
    check(!result.slices.empty() && result.slices[0].decline.find("noise alone") != std::string::npos,
          "the per-slice voxel floor is what is reported");
  }
  {
    // An all-signal field of view. What is selected is then a set of voxels whose spread is far
    //   narrower than any background, so the fitted degrees of freedom run away and the
    //   plausibility ceiling is what catches it.
    PhantomSpec spec = clean;
    spec.semi_axis_x = 10.0;
    spec.semi_axis_y = 10.0;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "all signal", spec, result);
    check(!result, "an all-signal field of view declines");
    check(!result.slices.empty() && result.slices[0].decline.find("ceiling") != std::string::npos,
          "the plausibility ceiling is what reports the all-signal case");
  }

  // (e) Interpolated data. The expectation is that interpolation applied to the complex data before
  //     the magnitude is formed leaves the degrees of freedom unchanged and alters only the noise
  //     level; here that level is sigma/sqrt(2) by construction. It is an expectation and not a
  //     result, which is why it is measured here rather than assumed.
  for (const double dof_true : {8.0, 2.0}) {
    PhantomSpec spec = clean;
    spec.dof = dof_true;
    MR::Image<float> image = make_resampled_phantom(spec, 99991u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report(out, "resampled, dof_true = " + std::to_string(dof_true), spec, result);
    check(bool(result), "resampled dof=" + std::to_string(dof_true) + ": an estimate is produced");
    if (!result)
      continue;
    check_relative(result.dof, dof_true, 0.05, "resampled dof=" + std::to_string(dof_true) + ": dof unchanged");
    check(!result.slices.empty() && result.slices[0].estimated, "resampled: the first slice yields an estimate");
    if (!result.slices.empty() && result.slices[0].estimated) {
      check_relative(result.slices[0].sigma,
                     slice_sigma(spec, 0) / std::sqrt(2.0),
                     0.05,
                     "resampled: the noise level is that of the interpolated data");
    }
  }

  // (f) Independence of the proportion of each slice that is background, and of the number of voxels
  //     per slice. Each slice is initialised at the lowest mode of a histogram whose binning follows
  //     from the slice's own voxels; the proportions and in-plane matrices below span those at which
  //     a single global intensity threshold either falls within the tissue (scarce background) or
  //     below the background (abundant background), and at which a fixed binning would be too coarse
  //     for the smallest slices or too noisy for the largest.
  class GeometryCase {
  public:
    const char *name;
    ssize_t matrix;
    double semi_axis_x;
    double semi_axis_y;
  };
  for (const GeometryCase &geometry : {GeometryCase{"scarce background, 64^2", 64, 0.55, 0.55},
                                       GeometryCase{"abundant background, 64^2", 64, 0.10, 0.10},
                                       GeometryCase{"coarse matrix, 24^2", 24, 0.45, 0.30},
                                       GeometryCase{"fine matrix, 256^2", 256, 0.45, 0.30},
                                       GeometryCase{"fine, scarce background", 256, 0.55, 0.55},
                                       GeometryCase{"fine, abundant background", 256, 0.10, 0.10}}) {
    PhantomSpec spec = clean;
    spec.nx = spec.ny = geometry.matrix;
    spec.semi_axis_x = geometry.semi_axis_x;
    spec.semi_axis_y = geometry.semi_axis_y;
    MR::Image<float> image = make_phantom(spec, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    const std::string name(geometry.name);
    report(out, name, spec, result);
    check(bool(result), name + ": an estimate is produced");
    if (!result)
      continue;
    const MaskCounts counts = count_mask(spec, result);
    std::fprintf(out,
                 "%-26s   selected: signal %zu/%zu, background %zu/%zu\n",
                 "",
                 counts.signal_selected,
                 counts.signal_total,
                 counts.background_selected,
                 counts.background_total);
    check_relative(result.dof, spec.dof, 0.05, name + ": global degrees of freedom");
    check(counts.signal_selected == 0, name + ": no signal-bearing voxel is selected");
    for (const auto &slice : result.slices)
      check(slice.estimated,
            name + ": slice " + std::to_string(slice.index) + " yields an estimate (" + slice.decline + ")");
  }
}

// Whether any of the caveats contains a phrase, which is how the two kinds are told apart.
bool has_caveat(const std::vector<std::string> &caveats, const std::string &phrase) {
  return std::any_of(caveats.begin(), caveats.end(), [&phrase](const std::string &caveat) {
    return caveat.find(phrase) != std::string::npos;
  });
}

// The warnings that flag, without correcting, a successful estimate that is likely to be wrong.
//   Each is asserted to fire on the failure it is for and to stay silent on healthy data, since a
//   warning that fires on clean data teaches users to ignore it.
void test_caveats(std::FILE *out) {
  std::fprintf(out, "\n");
  const PhantomSpec clean;
  const std::string spread_phrase = "disagree by more than sampling";
  const std::string zero_phrase = "excluded from the fit";

  // Clean data: nothing excluded, nothing flagged.
  {
    MR::Image<float> image = make_phantom(clean, 424242u);
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    check(bool(result), "caveats (clean): an estimate is produced");
    check(result.num_volumes == uint32_t(clean.nvol), "caveats (clean): the volume count is recorded");
    check_near(result.excluded_fraction, 0.0, 0.0, "caveats (clean): no sample of the selected voxels is excluded");
    const std::vector<std::string> caveats = estimate_caveats(result);
    check(caveats.empty(), "caveats (clean): no caveat is raised");
    std::fprintf(out, "%-26s excluded=%.4f caveats=%zu\n", "caveats: clean", result.excluded_fraction, caveats.size());
  }

  // Zeros that are not noise, scattered at random through 1% of the samples: they are excluded
  //   from the fit, and the zero-sample caveat reports it (see test_zero_phantoms() for the
  //   estimate itself).
  {
    PhantomSpec spec = clean;
    MR::Image<float> image = make_phantom(spec, 424242u);
    RNG rng(4242u);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (auto l = MR::Loop(image)(image); l; ++l) {
      if (uniform(rng) < 0.01)
        image.value() = 0.0f;
    }
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    check(bool(result), "caveats (zeroed): an estimate is produced");
    const std::vector<std::string> caveats = estimate_caveats(result);
    std::fprintf(out,
                 "%-26s dof=%.4f excluded=%.4f caveats=%zu\n",
                 "caveats: zeroed",
                 result.dof,
                 result.excluded_fraction,
                 caveats.size());
    check(result.excluded_fraction > 0.005, "caveats (zeroed): the excluded zeros are counted");
    check(has_caveat(caveats, zero_phrase), "caveats (zeroed): the zero-sample caveat is raised");
    check(!has_caveat(caveats, spread_phrase), "caveats (zeroed): the spread caveat is not raised");
  }

  // The spread caveat, on constructed results, so that its threshold is tested directly: per-slice
  //   estimates spread as they are where tissue is selected in some slices (relative interquartile
  //   range 0.13) are flagged where the slices are well sampled, a spread typical of clean data
  //   (0.005) is not, and the former is NOT flagged where so few samples were available per slice
  //   that sampling variation alone would explain it.
  const auto constructed = [](const double dof, const double iqr, const size_t voxels, const uint32_t volumes) {
    BackgroundResult result;
    result.estimated = true;
    result.dof = dof;
    result.dof_iqr = iqr;
    result.num_volumes = volumes;
    result.slices.resize(8);
    for (SliceEstimate &slice : result.slices) {
      slice.estimated = true;
      slice.dof = dof;
      slice.num_selected = voxels;
    }
    return result;
  };
  check(has_caveat(estimate_caveats(constructed(2.5, 0.33, 2000, 64)), spread_phrase),
        "caveats: a wide spread of well-sampled per-slice estimates is flagged");
  check(estimate_caveats(constructed(8.0, 0.04, 2000, 64)).empty(),
        "caveats: the spread of clean per-slice estimates is not flagged");
  check(estimate_caveats(constructed(2.5, 0.33, 100, 4)).empty(),
        "caveats: a spread that sampling variation explains is not flagged");
  BackgroundResult declined = constructed(2.5, 0.33, 2000, 64);
  declined.estimated = false;
  declined.excluded_fraction = 0.5;
  check(estimate_caveats(declined).empty(), "caveats: nothing is raised about a declined estimate");
}

// ---------------------------------------------------------------------------
// 5. Zero-valued samples
// ---------------------------------------------------------------------------

// How the zeros of a set of samples arise.
enum class zeros_t { ROUNDED, CLIPPED, ROUNDED_PLUS_EXTRA };

// Sums of a large sample from one background distribution, with zeros produced as named. For
//   ROUNDED and ROUNDED_PLUS_EXTRA the magnitudes are rounded to integers and the censoring
//   threshold is half the quantum; for CLIPPED, magnitudes below "level" are set to zero and the
//   threshold is that level. ROUNDED_PLUS_EXTRA additionally zeroes a proportion "level" of samples
//   at random. Also returns the sums with the zeros excluded, as was formerly done.
std::pair<SampleSums, SampleSums> zero_sums(const zeros_t mode,
                                            const double dof,
                                            const double sigma,
                                            const double level,
                                            const int num_samples,
                                            const unsigned seed) {
  RNG rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  SampleSums sums;
  sums.threshold_sq = mode == zeros_t::CLIPPED ? level * level : 0.25;
  for (int sample = 0; sample != num_samples; ++sample) {
    double m = sample_magnitude(0.0, dof, sigma, rng);
    if (mode == zeros_t::CLIPPED) {
      if (m < level)
        m = 0.0;
    } else {
      m = std::round(m);
      if (mode == zeros_t::ROUNDED_PLUS_EXTRA && uniform(rng) < level)
        m = 0.0;
    }
    if (m == 0.0) {
      ++sums.num_zero;
      continue;
    }
    const double m2 = m * m;
    ++sums.count;
    sums.sum_sq += m2;
    sums.sum_quad += m2 * m2;
    sums.sum_log_sq += std::log(m2);
  }
  SampleSums excluded = sums;
  excluded.num_zero = 0;
  return {sums, excluded};
}

// The censored fits on large samples, at parameter values where excluding the zeros is badly
//   biased: a half-Gaussian background (dof 1, as reported for GE data) stored as integers leaves 4%
//   of samples zero even at a noise level of ten quanta, and exclusion overestimates its degrees of
//   freedom by about 20%. Tolerances are 2% for maximum likelihood (whose residual error at a noise
//   level of a few quanta is the rounding of the non-zero magnitudes, which censoring the zeros
//   does not address) and 3% for the method of moments.
void test_censored_estimators(std::FILE *out) {
  class Case {
  public:
    const char *name;
    zeros_t mode;
    double dof;
    double sigma;
    double level;
  };
  const std::vector<Case> cases{{"rounded", zeros_t::ROUNDED, 1.0, 10.0, 0.0},
                                {"rounded", zeros_t::ROUNDED, 1.0, 5.0, 0.0},
                                {"rounded", zeros_t::ROUNDED, 2.0, 2.0, 0.0},
                                {"clipped", zeros_t::CLIPPED, 2.0, 10.0, 5.0},
                                {"clipped", zeros_t::CLIPPED, 2.0, 10.0, 10.0}};
  std::fprintf(out,
               "\n%-9s %-5s %-6s %-6s %8s %9s %9s %9s %9s\n",
               "zeros",
               "dof",
               "sigma",
               "level",
               "zero%",
               "excluded",
               "ml",
               "moments",
               "sigma(ml)");
  unsigned seed = 31337u;
  for (const Case &c : cases) {
    const auto sums = zero_sums(c.mode, c.dof, c.sigma, c.level, 400000, seed++);
    const Estimate excluded = estimate_from_sums(sums.second, background_method_t::ML);
    const Estimate ml = estimate_from_sums(sums.first, background_method_t::ML);
    const Estimate moments = estimate_from_sums(sums.first, background_method_t::MOMENTS);
    const double zero_fraction = double(sums.first.num_zero) / double(sums.first.num_zero + sums.first.count);
    std::fprintf(out,
                 "%-9s %-5g %-6g %-6g %8.2f %9.4f %9.4f %9.4f %9.4f\n",
                 c.name,
                 c.dof,
                 c.sigma,
                 c.level,
                 100.0 * zero_fraction,
                 excluded.dof,
                 ml.dof,
                 moments.dof,
                 ml.sigma);
    const std::string label = std::string("censored (") + c.name + ", dof " + std::to_string(c.dof) + ", sigma " +
                              std::to_string(c.sigma) + ", level " + std::to_string(c.level) + ")";
    check(bool(ml) && bool(moments), label + ": both censored fits succeed");
    check_relative(ml.dof, c.dof, 0.02, label + ": maximum likelihood recovers the degrees of freedom");
    check_relative(ml.sigma, c.sigma, 0.02, label + ": maximum likelihood recovers the noise level");
    check_relative(moments.dof, c.dof, 0.03, label + ": the method of moments recovers the degrees of freedom");
    check(excluded.dof > 1.05 * c.dof, label + ": excluding the zeros overestimates (the bias being corrected)");
  }

  // Censoring at threshold T is the same model at every scale: multiplying every magnitude and the
  //   threshold by a common factor leaves the degrees of freedom unchanged.
  {
    const auto sums = zero_sums(zeros_t::CLIPPED, 2.0, 10.0, 5.0, 20000, 99u);
    SampleSums scaled = sums.first;
    const double factor = 1.0e3;
    scaled.sum_sq *= factor * factor;
    scaled.sum_quad *= factor * factor * factor * factor;
    scaled.sum_log_sq += double(scaled.count) * std::log(factor * factor);
    scaled.threshold_sq *= factor * factor;
    for (const background_method_t method : {background_method_t::ML, background_method_t::MOMENTS}) {
      const Estimate original = estimate_from_sums(sums.first, method);
      const Estimate rescaled = estimate_from_sums(scaled, method);
      check_relative(rescaled.dof, original.dof, 1.0e-6, "censored: the degrees of freedom are scale-invariant");
      check_relative(rescaled.sigma, factor * original.sigma, 1.0e-6, "censored: the noise level scales with the data");
    }
  }

  // Without zeros, the threshold is never consulted: the fit is that of the positive samples.
  {
    const auto sums = zero_sums(zeros_t::CLIPPED, 8.0, 10.0, 0.0, 5000, 7u);
    SampleSums with_threshold = sums.first;
    with_threshold.threshold_sq = 123.0;
    for (const background_method_t method : {background_method_t::ML, background_method_t::MOMENTS}) {
      const Estimate a = estimate_from_sums(sums.first, method);
      const Estimate b = estimate_from_sums(with_threshold, method);
      check(a.dof == b.dof && a.sigma == b.sigma, "censored: a threshold without zeros changes nothing");
    }
  }

  // Zeros without a threshold are refused rather than fitted.
  {
    SampleSums sums = zero_sums(zeros_t::ROUNDED, 1.0, 5.0, 0.0, 5000, 8u).first;
    sums.threshold_sq = 0.0;
    check(!estimate_from_sums(sums, background_method_t::ML), "censored: zeros without a threshold are refused");
  }
}

// Detection of the storage quantum from the smallest distinct magnitudes, and the threshold derived
//   from it: integer data, integers scaled by a non-integral factor, and continuous data.
void test_quantum_detection(std::FILE *out) {
  PhantomSpec spec;
  spec.dof = 2.0;
  spec.sigma = 3.0;
  spec.sigma_ratio = 1.0;
  for (const double scale : {1.0, 0.37, 0.0}) {
    MR::Image<float> image = make_phantom(spec, 777u);
    if (scale > 0.0) {
      for (auto l = MR::Loop(image)(image); l; ++l)
        image.value() = float(scale * std::round(double(image.value())));
    }
    const Accumulation accumulation = accumulate_background(image, 2);
    std::fprintf(out, "quantum detection: scale %-5g detected %.8g\n", scale, accumulation.quantum);
    if (scale > 0.0) {
      check_relative(
          accumulation.quantum, scale, 1.0e-6, "quantum: detected for integers scaled by " + std::to_string(scale));
      check_relative(
          accumulation.censoring_threshold(scale), 0.5 * scale, 1.0e-6, "quantum: threshold is half the quantum");
    } else {
      check(accumulation.quantum == 0.0, "quantum: none is detected in continuous data");
      check(accumulation.censoring_threshold(4.0) == 4.0,
            "quantum: threshold of continuous data is the smallest magnitude");
    }
    // Rounding at a noise level of three quanta leaves zeros, which must be counted per voxel.
    size_t zeros = 0;
    for (const VoxelStats &stats : accumulation.voxels)
      zeros += stats.num_zero;
    check((scale > 0.0) == (zeros > 0), "quantum: zeros are counted where, and only where, rounding produced them");
  }
}

// Algorithm 1 end to end on phantoms whose zeros are, and are not, censored noise.
void test_zero_phantoms(std::FILE *out) {
  std::fprintf(out, "\n");
  const auto report_zeros = [out](const char *name, const BackgroundResult &result) {
    std::fprintf(out,
                 "%-26s dof=%.4f censored=%.4f excluded=%.4f slices_excluding_zeros=%zu caveats=%zu\n",
                 name,
                 result.dof,
                 result.censored_fraction,
                 result.excluded_fraction,
                 result.num_slices_zeros_excluded,
                 estimate_caveats(result).size());
  };
  const auto round_image = [](MR::Image<float> &image) {
    for (auto l = MR::Loop(image)(image); l; ++l)
      image.value() = std::round(image.value());
  };

  // Half-Gaussian background stored as integers at a noise level of 5 to 10 quanta: 4% to 8% of
  //   background samples round to zero. Every slice censors them, and the truth is recovered.
  for (const background_method_t method : {background_method_t::ML, background_method_t::MOMENTS}) {
    PhantomSpec spec;
    spec.dof = 1.0;
    spec.sigma = 5.0;
    MR::Image<float> image = make_phantom(spec, 515151u);
    round_image(image);
    const BackgroundResult result = run_phantom(image, method);
    const std::string name =
        std::string("rounded half-Gaussian (") + (method == background_method_t::ML ? "ml" : "moments") + ")";
    report_zeros(name.c_str(), result);
    check(bool(result), name + ": an estimate is produced");
    check_relative(result.dof, spec.dof, 0.05, name + ": degrees of freedom");
    check(result.num_slices_zeros_excluded == 0, name + ": the rounding zeros are censored in every slice");
    check(result.censored_fraction > 0.02, name + ": the censored zeros are counted");
    check(estimate_caveats(result).empty(), name + ": no caveat is raised");
  }

  // Rician background clipped below one noise level, as a reconstruction that discards low
  //   magnitudes would: 39% of background samples become zero, and the clip level rises across the
  //   slices with the noise level. Formerly excluded, the zeros inflated the estimate by half.
  {
    PhantomSpec spec;
    spec.dof = 2.0;
    MR::Image<float> image = make_phantom(spec, 424242u);
    for (auto l = MR::Loop(image)(image); l; ++l) {
      if (image.value() < float(slice_sigma(spec, image.index(2))))
        image.value() = 0.0f;
    }
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report_zeros("clipped Rician", result);
    check(bool(result), "clipped: an estimate is produced");
    check_relative(result.dof, spec.dof, 0.05, "clipped: degrees of freedom");
    check(result.num_slices_zeros_excluded == 0, "clipped: the clipped zeros are censored in every slice");
    check(result.censored_fraction > 0.3, "clipped: the censored zeros are counted");
    check(estimate_caveats(result).empty(), "clipped: no caveat is raised");
  }

  // Integer data with no rounding zeros to speak of, plus zeros that are not noise: 0.5% and 2% of
  //   samples zeroed at random, and an edge slab of the background zeroed in half the volumes (as
  //   resampling at the edge of the field of view can do). Censoring these would drive the degrees
  //   of freedom far down. Scattered zeros are recognised by the consistency check, and excluded
  //   in every slice. The slab's voxels, whose summed squared magnitude is halved, are mostly
  //   rejected by the acceptance test itself, so their zeros seldom reach a fit; where they do, the
  //   consistency check excludes them, unless they are too few to contradict censoring, in which
  //   case they are too few to matter.
  for (const double fraction : {0.005, 0.02, -1.0}) {
    PhantomSpec spec;
    MR::Image<float> image = make_phantom(spec, 616161u);
    round_image(image);
    RNG rng(6161u);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (auto l = MR::Loop(image)(image); l; ++l) {
      const bool zero = fraction > 0.0 ? uniform(rng) < fraction : image.index(0) < 6 && image.index(3) % 2 == 0;
      if (zero)
        image.value() = 0.0f;
    }
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    const std::string name =
        fraction > 0.0 ? "random zeros " + std::to_string(fraction) : std::string("zeroed edge slab");
    report_zeros(name.c_str(), result);
    check(bool(result), name + ": an estimate is produced");
    check_relative(result.dof, spec.dof, 0.05, name + ": degrees of freedom");
    if (fraction > 0.0)
      check(result.num_slices_zeros_excluded == size_t(spec.nz), name + ": the zeros are excluded in every slice");
    check(result.censored_fraction < 1.0e-3, name + ": no more than a negligible number of zeros is censored");
  }

  // A real-valued reconstruction whose negative values are clipped to zero: half of all background
  //   samples are zero, and the positive half is distributed as the magnitude is. The threshold of
  //   such floating-point data is minute, so the censored fit can explain any proportion of zeros by
  //   driving the degrees of freedom towards zero, and is consistent with its own prediction of the
  //   zero count; the consistency check must therefore predict that count from the positive
  //   samples alone, which it does, and the zeros are excluded.
  {
    PhantomSpec spec;
    spec.dof = 2.0;
    MR::Image<float> image = make_phantom(spec, 717171u);
    RNG rng(7171u);
    std::bernoulli_distribution negative(0.5);
    for (auto l = MR::Loop(image)(image); l; ++l) {
      if (negative(rng))
        image.value() = 0.0f;
    }
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report_zeros("real part, clipped", result);
    check(bool(result), "real part clipped: an estimate is produced");
    check_relative(result.dof, spec.dof, 0.05, "real part clipped: degrees of freedom");
    check(result.num_slices_zeros_excluded == size_t(spec.nz),
          "real part clipped: the zeros are excluded in every slice");
    check(result.censored_fraction == 0.0, "real part clipped: no zero is censored");
  }
}

// ---------------------------------------------------------------------------
// 6. Consistency of the selection with a single Gamma distribution
// ---------------------------------------------------------------------------

// The parameters with the consistency test disabled, against which each test below demonstrates
//   the failure that the test is there to prevent.
BackgroundParameters without_gamma_test() {
  BackgroundParameters parameters;
  parameters.gamma_ratio_floor = 0.0;
  return parameters;
}

// A diffusion-weighted head that overfills the in-plane field of view in every slice from
//   first_overfilled onwards, leaving no background there at all, and that occupies the middle of
//   it in the slices before. The tissue model is that of an anisotropic diffusion tensor whose
//   principal direction rotates over the slice, sampled on b = 1000 and 2000 shells with a b = 0
//   volume every sixteen: with a b = 0 signal-to-noise ratio of twenty, the most strongly attenuated
//   samples are near the noise floor. Their voxels then form the lowest mode of an overfilled slice,
//   which is where the iteration begins and converges.
class TissuePhantom {
public:
  ssize_t n{64};
  ssize_t nz{8};
  ssize_t nvol{64};
  ssize_t first_overfilled{3};
  double dof{8.0};
  double sigma{10.0};
};

bool in_tissue(const TissuePhantom &spec, const ssize_t x, const ssize_t y, const ssize_t z) {
  const double scale = z < spec.first_overfilled ? 1.0 : 2.0;
  const double u = (double(x) - 0.5 * double(spec.n - 1)) / (0.5 * double(spec.n) * 0.7 * scale);
  const double v = (double(y) - 0.5 * double(spec.n - 1)) / (0.5 * double(spec.n) * 0.8 * scale);
  return u * u + v * v < 1.0;
}

MR::Image<float> make_tissue_phantom(const TissuePhantom &spec, const unsigned seed) {
  PhantomSpec geometry;
  geometry.nx = geometry.ny = spec.n;
  geometry.nz = spec.nz;
  geometry.nvol = spec.nvol;
  MR::Image<float> image = MR::Image<float>::scratch(phantom_header(geometry), "tissue phantom");
  RNG rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> bvalue(size_t(spec.nvol), 0.0);
  std::vector<std::array<double, 3>> direction(size_t(spec.nvol), {{0.0, 0.0, 0.0}});
  for (ssize_t volume = 0; volume != spec.nvol; ++volume) {
    if (volume % 16 == 0)
      continue;
    bvalue[size_t(volume)] = volume % 2 ? 1000.0 : 2000.0;
    std::array<double, 3> g{{normal(rng), normal(rng), normal(rng)}};
    const double norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    for (double &component : g)
      component /= norm;
    direction[size_t(volume)] = g;
  }
  const double md = 0.8e-3;
  const double anisotropy = 0.55;
  const double l1 = md * (1.0 + 2.0 * anisotropy);
  const double l2 = md * (1.0 - anisotropy);
  for (ssize_t z = 0; z != spec.nz; ++z) {
    image.index(2) = z;
    for (ssize_t y = 0; y != spec.n; ++y) {
      image.index(1) = y;
      const double v = (double(y) - 0.5 * double(spec.n - 1)) / (0.5 * double(spec.n));
      for (ssize_t x = 0; x != spec.n; ++x) {
        image.index(0) = x;
        const double u = (double(x) - 0.5 * double(spec.n - 1)) / (0.5 * double(spec.n));
        const bool tissue = in_tissue(spec, x, y, z);
        const double s0 = 20.0 * spec.sigma * (0.7 + 0.3 * std::cos(3.0 * u) * std::cos(2.5 * v));
        const double theta = M_PI * (u + 0.5 * v);
        const std::array<double, 3> e1{
            {std::cos(theta) * std::sin(1.0), std::sin(theta) * std::sin(1.0), std::cos(1.0)}};
        for (ssize_t volume = 0; volume != spec.nvol; ++volume) {
          image.index(3) = volume;
          double amplitude = 0.0;
          if (tissue) {
            const std::array<double, 3> &g = direction[size_t(volume)];
            const double cosine = e1[0] * g[0] + e1[1] * g[1] + e1[2] * g[2];
            amplitude = s0 * std::exp(-bvalue[size_t(volume)] * (l2 + (l1 - l2) * cosine * cosine));
          }
          image.value() = float(sample_magnitude(amplitude / spec.sigma, spec.dof, spec.sigma, rng));
        }
      }
    }
  }
  image.index(3) = 0;
  return image;
}

// Of the voxels selected in slice z, the proportion that are tissue.
double tissue_fraction(const TissuePhantom &spec, const BackgroundResult &result, const ssize_t z) {
  const Accumulation geometry({spec.n, spec.n, spec.nz}, 2);
  size_t selected = 0;
  size_t tissue = 0;
  for (ssize_t y = 0; y != spec.n; ++y) {
    for (ssize_t x = 0; x != spec.n; ++x) {
      if (result.selected[geometry.index(x, y, z)] == 0)
        continue;
      ++selected;
      tissue += in_tissue(spec, x, y, z) ? 1 : 0;
    }
  }
  return selected == 0 ? 0.0 : double(tissue) / double(selected);
}

void report_gamma(std::FILE *out, const std::string &name, const BackgroundResult &result) {
  std::fprintf(out, "%-26s %s", name.c_str(), result ? "" : "DECLINED; ");
  if (result)
    std::fprintf(out, "dof=%.4f; ", result.dof);
  std::fprintf(out, "per-slice ratio:");
  for (const SliceEstimate &slice : result.slices)
    std::fprintf(out, " %.3f%s", slice.gamma_ratio, slice.estimated ? "" : "x");
  std::fprintf(out, "\n");
}

void test_gamma_consistency(std::FILE *out) {
  std::fprintf(out, "\n");
  const std::string gamma_phrase = "single Gamma distribution";

  // Pure background must pass at every number of volumes and every degree of freedom, both where
  //   the window's truncation of the per-voxel sums distorts the selected samples most (two
  //   volumes) and where the number of samples, and so the power of the test, is largest. No
  //   slice may decline, and none may come near the floor.
  class NullCase {
  public:
    ssize_t nvol;
    double dof;
    ssize_t nz;
  };
  for (const NullCase &null_case : {NullCase{2, 2.0, 8},
                                    NullCase{2, 8.0, 8},
                                    NullCase{4, 1.0, 8},
                                    NullCase{4, 8.0, 8},
                                    NullCase{20, 1.0, 8},
                                    NullCase{20, 16.0, 8},
                                    NullCase{200, 2.0, 4},
                                    NullCase{200, 7.52, 4}}) {
    PhantomSpec spec;
    spec.nvol = null_case.nvol;
    spec.dof = null_case.dof;
    spec.nz = null_case.nz;
    MR::Image<float> image = make_phantom(spec, 515151u);
    for (const background_method_t method : {background_method_t::ML, background_method_t::MOMENTS}) {
      const BackgroundResult result = run_phantom(image, method);
      const std::string name = "Gamma null, K=" + std::to_string(null_case.nvol) +
                               " dof=" + std::to_string(null_case.dof) +
                               (method == background_method_t::ML ? " (ml)" : " (moments)");
      report_gamma(out, name, result);
      check(bool(result), name + ": an estimate is produced");
      for (const SliceEstimate &slice : result.slices) {
        check(slice.estimated,
              name + ": slice " + std::to_string(slice.index) + " yields an estimate (" + slice.decline + ")");
        check(std::isfinite(slice.gamma_ratio) && slice.gamma_ratio > 0.97,
              name + ": slice " + std::to_string(slice.index) + " is well clear of the floor");
      }
    }
  }

  // Tissue selected for want of background. With the test, the overfilled slices decline for that
  //   reason, and the estimate comes from the slices that do hold background. Without it, every
  //   overfilled slice is estimated from tissue, at too few degrees of freedom, and the median
  //   across slices goes with them.
  for (const double dof : {8.0, 4.0}) {
    TissuePhantom spec;
    spec.dof = dof;
    MR::Image<float> image = make_tissue_phantom(spec, 616161u);
    const std::string name = "overfilled head, dof=" + std::to_string(dof);

    const BackgroundResult ungated = run_phantom(image, background_method_t::ML, without_gamma_test());
    report_gamma(out, name + " (untested)", ungated);
    size_t tissue_estimated = 0;
    for (ssize_t z = spec.first_overfilled; z != spec.nz; ++z) {
      if (ungated.slices[size_t(z)].estimated && tissue_fraction(spec, ungated, z) > 0.5)
        ++tissue_estimated;
    }
    check(tissue_estimated == size_t(spec.nz - spec.first_overfilled),
          name + ", untested: every overfilled slice is estimated from tissue");
    check(!ungated || std::fabs(ungated.dof / dof - 1.0) > 0.10, name + ", untested: the estimate is wrong");

    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report_gamma(out, name, result);
    for (ssize_t z = 0; z != spec.nz; ++z) {
      const SliceEstimate &slice = result.slices[size_t(z)];
      if (z < spec.first_overfilled) {
        check(slice.estimated, name + ": slice " + std::to_string(z) + ", which holds background, yields an estimate");
        check(tissue_fraction(spec, result, z) == 0.0, name + ": no tissue is selected where background exists");
      } else {
        check(!slice.estimated, name + ": overfilled slice " + std::to_string(z) + " declines");
        check(slice.decline.find(gamma_phrase) != std::string::npos,
              name + ": overfilled slice " + std::to_string(z) + " declines by the Gamma consistency test (" +
                  slice.decline + ")");
      }
    }
    check(bool(result), name + ": an estimate is produced from the slices holding background");
    if (result)
      check_relative(result.dof, dof, 0.05, name + ": degrees of freedom");
  }

  // A noise level that differs between volumes, here doubled in two volumes of twenty. This is what
  //   the preprocessed data of MRtrix3's script_test_data show: the voxels selected there carry many
  //   times the mean power in b = 0 volumes as in diffusion-weighted ones, because they are not
  //   noise alone. Every voxel is equally affected, so the per-voxel sums remain a compact cluster
  //   that the iteration converges on happily; but the samples are a mixture of two Gamma
  //   distributions of different scale, and no degrees of freedom fitted to them is that of either.
  {
    const PhantomSpec spec;
    MR::Image<float> image = make_phantom(spec, 424242u);
    for (auto l = MR::Loop(image)(image); l; ++l) {
      if (image.index(3) % 10 == 0)
        image.value() *= 2.0f;
    }
    const std::string name = "per-volume noise level";
    const BackgroundResult ungated = run_phantom(image, background_method_t::ML, without_gamma_test());
    report_gamma(out, name + " (untested)", ungated);
    check(bool(ungated), name + ", untested: an estimate is produced");
    if (ungated)
      check(std::fabs(ungated.dof / spec.dof - 1.0) > 0.10, name + ", untested: the estimate is wrong");
    const BackgroundResult result = run_phantom(image, background_method_t::ML);
    report_gamma(out, name, result);
    check(!result, name + ": no estimate is produced");
    for (const SliceEstimate &slice : result.slices) {
      check(slice.decline.find(gamma_phrase) != std::string::npos,
            name + ": slice " + std::to_string(slice.index) + " declines by the Gamma consistency test (" +
                slice.decline + ")");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  bool evidence = false;
  const char *tsv_path = nullptr;
  for (int i = 1; i != argc; ++i) {
    if (std::strcmp(argv[i], "--evidence") == 0) {
      evidence = true;
    } else if (std::strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
      tsv_path = argv[++i];
      evidence = true;
    } else {
      std::fprintf(stderr, "usage: %s [--evidence] [--sweep <file.tsv>]\n", argv[0]);
      return 2;
    }
  }

  std::FILE *tsv = nullptr;
  if (tsv_path != nullptr) {
    tsv = std::fopen(tsv_path, "w");
    if (tsv == nullptr) {
      std::fprintf(stderr, "unable to open \"%s\" for writing\n", tsv_path);
      return 2;
    }
    std::fprintf(tsv, "experiment\tmethod\tdof_true\tsigma_true\tn\tkey\tvalue\n");
  }

  std::FILE *const out = evidence ? stdout : stderr;
  test_samplers();
  test_special_functions(out);
  test_estimators(out, tsv, evidence);
  test_scale_equivariance(out);
  test_algorithm(out, tsv);
  test_caveats(out);
  test_censored_estimators(out);
  test_quantum_detection(out);
  test_zero_phantoms(out);
  test_gamma_consistency(out);

  if (tsv != nullptr)
    std::fclose(tsv);
  if (failures == 0)
    std::printf("background_dof: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
