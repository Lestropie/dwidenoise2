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
// in gamma_stats.{h,cpp}), following St-Jean et al. (2020).
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
//    rather than answer, and the expectation that interpolation of the complex data before the
//    magnitude is formed leaves the degrees of freedom unchanged while altering only the noise
//    level (which, if it holds, makes this estimator valid on data for which Marchenko-Pastur
//    estimation is not).
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

#include "denoise/precondition/noise_model/background.h"
#include "denoise/precondition/noise_model/gamma_stats.h"
#include "header.h"
#include "image.h"
#include "math/noncentral_chi.h"
#include "ncchi_sampler.h"

using MR::Denoise::Precondition::NoiseModel::Accumulation;
using MR::Denoise::Precondition::NoiseModel::accumulate_background;
using MR::Denoise::Precondition::NoiseModel::background_method_t;
using MR::Denoise::Precondition::NoiseModel::BackgroundParameters;
using MR::Denoise::Precondition::NoiseModel::BackgroundResult;
using MR::Denoise::Precondition::NoiseModel::Estimate;
using MR::Denoise::Precondition::NoiseModel::estimate_background;
using MR::Denoise::Precondition::NoiseModel::estimate_from_sums;
using MR::Denoise::Precondition::NoiseModel::SampleSums;
using MR::Denoise::Precondition::NoiseModel::GammaStats::digamma;
using MR::Denoise::Precondition::NoiseModel::GammaStats::gamma_p;
using MR::Denoise::Precondition::NoiseModel::GammaStats::gamma_p_inv;
using MR::Denoise::Precondition::NoiseModel::GammaStats::inverse_digamma;
using MR::Denoise::Precondition::NoiseModel::GammaStats::trigamma;
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

BackgroundResult run_phantom(MR::Image<float> &image, const background_method_t method) {
  const Accumulation accumulation = accumulate_background(image, 2);
  return estimate_background(accumulation, method);
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

  if (tsv != nullptr)
    std::fclose(tsv);
  if (failures == 0)
    std::printf("background_dof: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
