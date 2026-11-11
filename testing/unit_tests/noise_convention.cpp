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

// Self-contained unit test for the PER-CHANNEL noise level convention.
//
// Every noise level in this software -- exported, imported, propagated between iterations, or
//   handed to another component -- is the standard deviation of a single real-valued channel, for
//   complex input data exactly as for magnitude. That is achieved by dividing the eigenvalues of
//   each Casorati matrix by the number of real noise components its entries carry, as soon as they
//   are computed (Denoise::casorati_entry_channels(), Estimate::to_per_channel()); it is where this
//   software departs from MRtrix3 dwidenoise, whose complex-data noise level is the aggregate
//   across channels and hence larger by sqrt(2).
//
// A plain executable whose non-zero exit status indicates failure, following
//   testing/unit_tests/{noise_model,dof_estimation}.cpp. It links the real Spectrum, estimators,
//   NoiseModel and AdaptivePhaseEstimator; nothing is re-implemented here, because a
//   re-implementation would validate the re-implementation.
//
// What is asserted, and why each matters to the convention:
//
//   1. Every data-driven estimator, and the fixed-rank estimator, is homogeneous of degree 1 in
//      sigma: scaling the whole eigenspectrum by c^2 scales sigma2 and lamplus by c^2 and leaves
//      the signal / noise classification untouched. This is what makes rescaling the SPECTRUM
//      equivalent to rescaling those estimators' OUTPUT, and hence what makes the correction safe
//      to apply once, upstream, rather than estimator by estimator.
//   2. The imposed estimators (Unity / Constant / Import) are, correctly, NOT homogeneous: their
//      sigma is external to the spectrum. This is why the correction must NOT be applied to an
//      estimator's output -- doing so would rescale a level that is already per-channel a second
//      time -- and the test exists so that "fixing" them for consistency fails loudly.
//   3. A complex Casorati entry carries twice the noise energy of a real one, so the uncorrected
//      complex estimate is exactly sqrt(2) times the per-channel level. The factor is pinned
//      explicitly, so that applying it twice, or not at all, cannot pass.
//   4. Across SNR: the complex estimate tracks the truth everywhere; the magnitude estimate
//      converges to it at high SNR and diverges near the noise floor; on a patch spanning a
//      realistic range of operating points the non-linear variance-stabilising transform recovers
//      most of that divergence.
//   5. Cross-application in both directions -- a level estimated from magnitude data correctly
//      stabilises the complex data, and one estimated from complex data correctly stabilises the
//      magnitude data. Both would fail, by a measured factor, under the aggregate convention.
//   6. The Unity estimator remains exact for complex data stabilised by a per-channel map, which
//      is what the final (reconstruction-only) row of a schedule relies on.
//   7. Adaptive phase correction's two calibration branches now agree: the noise level it is
//      handed and the one it self-calibrates from the data are on the same scale, and on the same
//      functional. The second half matters only where the noise is non-stationary, so the case is
//      made twice: on a stationary phantom, where every reasonable functional coincides, and on a
//      phantom whose noise level spans the range a parallel-imaging acquisition produces, where a
//      root mean square and a robust centre differ by six-fold and only one of them is what the
//      criterion needs.
//
// Tolerances are set against Monte Carlo error at the fixed seeds below; the SNR sweep tables are
//   printed in full rather than asserted term by term.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/datadriven/exp.h"
#include "denoise/estimator/datadriven/med.h"
#include "denoise/estimator/datadriven/mrm2023.h"
#include "denoise/estimator/datadriven/tbme2022.h"
#include "denoise/estimator/imposed/constant.h"
#include "denoise/estimator/imposed/rank.h"
#include "denoise/estimator/imposed/unity.h"
#include "denoise/estimator/result.h"
#include "denoise/precondition/apc.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "denoise/spectrum.h"
#include "math/math.h"

using MR::cdouble;
using MR::default_type;
using MR::Denoise::casorati_entry_channels;
using MR::Denoise::eigenvalues_type;
using MR::Denoise::Spectrum;
using MR::Denoise::Estimator::Constant;
using MR::Denoise::Estimator::Exp;
using MR::Denoise::Estimator::Med;
using MR::Denoise::Estimator::MRM2023;
using MR::Denoise::Estimator::Rank;
using MR::Denoise::Estimator::Result;
using MR::Denoise::Estimator::TBME2022;
using MR::Denoise::Estimator::Unity;
using MR::Denoise::Precondition::AdaptivePhaseEstimator;
using MR::Denoise::Precondition::NoiseModel::Base;
using MR::Denoise::Precondition::NoiseModel::distribution_t;
using MR::Denoise::Precondition::NoiseModel::make;
using MR::Denoise::Precondition::NoiseModel::vst_method_t;

namespace {

using RNG = std::mt19937_64;
using Matrix = Eigen::MatrixXd;
using CMatrix = Eigen::MatrixXcd;
using Array2D = Eigen::ArrayXXd;

int failures = 0;

void check(const bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

void check_near(const double value, const double target, const double tol, const std::string &message) {
  if (!(std::fabs(value - target) <= tol)) {
    std::fprintf(stderr, "FAIL: %s (got %.8g, expected %.8g +/- %.3g)\n", message.c_str(), value, target, tol);
    ++failures;
  }
}

void check_relative(const double value, const double target, const double tol, const std::string &message) {
  const double denom = std::fabs(target) > 0.0 ? std::fabs(target) : 1.0;
  if (!(std::fabs(value - target) / denom <= tol)) {
    std::fprintf(stderr,
                 "FAIL: %s (got %.10g, expected %.10g, relative error %.3g > %.3g)\n",
                 message.c_str(),
                 value,
                 target,
                 std::fabs(value - target) / denom,
                 tol);
    ++failures;
  }
}

double median_of(std::vector<double> values) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t half = values.size() / 2;
  return (values.size() & 1) ? values[half] : 0.5 * (values[half - 1] + values[half]);
}

double mean_of(const std::vector<double> &values) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

// ---------------------------------------------------------------------------
// Synthetic patches
// ---------------------------------------------------------------------------

// Non-negative rank-r signal of m volumes by n voxels, scaled so that its mean entry is
//   "mean_level". Rank 0 gives a signal constant across volumes, which the demeaning below removes
//   entirely, leaving the decomposition pure noise.
Matrix make_signal(const ssize_t m, const ssize_t n, const ssize_t rank, const double mean_level, RNG &rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  Matrix result(m, n);
  if (rank == 0) {
    result.setConstant(mean_level);
    return result;
  }
  Matrix a(m, rank);
  Matrix b(n, rank);
  for (ssize_t i = 0; i != m; ++i)
    for (ssize_t k = 0; k != rank; ++k)
      a(i, k) = std::fabs(nd(rng));
  for (ssize_t j = 0; j != n; ++j)
    for (ssize_t k = 0; k != rank; ++k)
      b(j, k) = std::fabs(nd(rng));
  result = a * b.transpose();
  const double mean = result.mean();
  result *= (mean > 0.0 ? mean_level / mean : 0.0);
  return result;
}

// As above, but with the per-voxel level spanning [lo, hi] logarithmically across the columns: the
//   realistic case, in which one patch straddles the noise floor and the high-SNR regime at once.
Matrix
make_signal_graded(const ssize_t m, const ssize_t n, const ssize_t rank, const double lo, const double hi, RNG &rng) {
  Matrix result = make_signal(m, n, rank, 1.0, rng);
  for (ssize_t j = 0; j != n; ++j) {
    const double target = lo * std::pow(hi / lo, double(j) / double(n - 1));
    const double mean = result.col(j).mean();
    result.col(j) *= (mean > 0.0 ? target / mean : 0.0);
  }
  return result;
}

// Complex data: real signal (zero phase, so the magnitude is exactly non-central chi with 2 degrees
//   of freedom) plus circularly-symmetric noise of PER-CHANNEL standard deviation sigma.
CMatrix draw_complex(const Matrix &signal, const double sigma, RNG &rng) {
  std::normal_distribution<double> nd(0.0, sigma);
  CMatrix result(signal.rows(), signal.cols());
  for (ssize_t i = 0; i != signal.rows(); ++i)
    for (ssize_t j = 0; j != signal.cols(); ++j)
      result(i, j) = cdouble(signal(i, j) + nd(rng), nd(rng));
  return result;
}

Matrix draw_real(const Matrix &signal, const double sigma, RNG &rng) {
  std::normal_distribution<double> nd(0.0, sigma);
  Matrix result(signal.rows(), signal.cols());
  for (ssize_t i = 0; i != signal.rows(); ++i)
    for (ssize_t j = 0; j != signal.cols(); ++j)
      result(i, j) = signal(i, j) + nd(rng);
  return result;
}

// Per-voxel demeaning across volumes: the single demeaning group, regressing one component from
//   every column, which is the preconditioner rank of 1 declared to each Spectrum below.
template <typename M> M demean_columns(const M &x) { return x.rowwise() - x.colwise().mean(); }

// The eigenspectrum of a patch, exactly as Denoise::Estimate forms it: the self-adjoint product on
//   the shorter dimension, negatives clamped, ascending -- and then, when "per_channel", divided by
//   the number of real noise components each entry carries. That division is the whole subject of
//   this test file; passing false reproduces the aggregate convention it replaces.
template <typename M> eigenvalues_type spectrum_of(const M &x, const bool per_channel) {
  Eigen::SelfAdjointEigenSolver<M> eig;
  if (x.rows() <= x.cols())
    eig.compute(x * x.adjoint(), Eigen::EigenvaluesOnly);
  else
    eig.compute(x.adjoint() * x, Eigen::EigenvaluesOnly);
  eigenvalues_type s = eig.eigenvalues().cwiseMax(0.0);
  if (per_channel)
    s /= casorati_entry_channels<typename M::Scalar>();
  return s;
}

// ---------------------------------------------------------------------------
// 1 and 2. Behaviour of the estimators under a rescaled eigenspectrum
// ---------------------------------------------------------------------------

// The estimators, held by base pointer so that the sweep can be written once. Import is
//   deliberately absent: it requires an Image, and it derives from the same ImposedSigma base as
//   Unity and Constant, differing only in where get_sigma_sq() reads its value from -- so its
//   behaviour under a rescaled spectrum is that of Constant, which is covered.
struct NamedEstimator {
  std::string name;
  std::shared_ptr<MR::Denoise::Estimator::Base> estimator;
  bool homogeneous; // sigma is derived from the spectrum, so it follows any rescaling of it
  bool cut_in_band; // the level it imposes falls within this spectrum's bulk (imposed only)
};

std::vector<NamedEstimator> all_estimators(const ssize_t imposed_rank, const double imposed_sigma) {
  return {{"exp1", std::make_shared<Exp<1>>(), true, false},
          {"exp2", std::make_shared<Exp<2>>(), true, false},
          {"med", std::make_shared<Med>(), true, false},
          {"mrm2023", std::make_shared<MRM2023>(), true, false},
          {"tbme2022", std::make_shared<TBME2022>(), true, false},
          {"rank", std::make_shared<Rank>(imposed_rank), true, false},
          {"unity", std::make_shared<Unity>(), false, false},
          {"constant", std::make_shared<Constant>(imposed_sigma), false, true}};
}

void test_homogeneity() {
  std::printf("\n== 1/2. Estimator behaviour under a rescaled eigenspectrum ==\n");
  RNG rng(20260804);
  const double sigma_true = 7.0;
  // c^2, the factor applied to every eigenvalue. 0.5 is the factor this software applies to complex
  //   data, and is included for that reason as much as for coverage.
  const std::vector<double> scales = {0.25, 0.5, 1.0, 2.0, 4.0, 100.0};

  struct Shape {
    ssize_t m, n, rank;
  };
  const std::vector<Shape> shapes = {{16, 32, 0}, {24, 64, 4}, {40, 41, 8}, {48, 24, 3}};

  for (const Shape &shape : shapes) {
    const Matrix signal = make_signal(shape.m, shape.n, shape.rank, 3.0 * sigma_true, rng);
    const Matrix x = demean_columns(draw_real(signal, sigma_true, rng));
    const eigenvalues_type s = spectrum_of(x, false);
    const ssize_t r = std::min(shape.m, shape.n);
    const ssize_t imposed_rank = std::max<ssize_t>(1, std::min(r - 1, shape.rank + 1));

    // A reference noise level for the imposed estimators, on the same scale as the unscaled
    //   spectrum, so that the cut it implies falls inside the bulk rather than beyond either end.
    const Spectrum reference(s, shape.m, shape.n, 1);
    const Result fitted = MRM2023()(reference, Eigen::Vector3d::Zero());
    check(bool(fitted),
          "reference noise level available for shape " + std::to_string(shape.m) + "x" + std::to_string(shape.n));
    const double imposed_sigma = bool(fitted) ? std::sqrt(fitted.sigma2) : sigma_true;

    for (const NamedEstimator &named : all_estimators(imposed_rank, imposed_sigma)) {
      const Result base = (*named.estimator)(reference, Eigen::Vector3d::Zero());
      const std::string shape_context =
          " [" + named.name + " @ " + std::to_string(shape.m) + "x" + std::to_string(shape.n) + "]";
      check(bool(base), "baseline threshold usable" + shape_context);
      if (!base)
        continue;
      // Ascending in "scales", for the monotonicity check on the imposed estimators below.
      std::vector<ssize_t> cutoff_by_scale;
      for (const double scale : scales) {
        eigenvalues_type scaled = s * scale;
        const Spectrum spectrum(scaled, shape.m, shape.n, 1);
        const Result result = (*named.estimator)(spectrum, Eigen::Vector3d::Zero());
        const std::string context = shape_context + " scale " + std::to_string(scale);
        cutoff_by_scale.push_back(result.cutoff_p);
        if (named.homogeneous) {
          check_relative(result.sigma2, base.sigma2 * scale, 1.0e-9, "sigma2 scales exactly" + context);
          check_relative(result.lamplus, base.lamplus * scale, 1.0e-9, "lamplus scales exactly" + context);
          check(result.cutoff_p == base.cutoff_p, "signal/noise classification is invariant" + context);
        } else {
          // The imposed level is external to the spectrum, so it does NOT follow the rescaling --
          //   which is exactly why the per-channel correction has to be applied to the spectrum
          //   itself rather than to what an estimator returns.
          check_relative(result.sigma2, base.sigma2, 1.0e-12, "imposed sigma2 is unaffected" + context);
        }
      }
      if (!named.homogeneous && named.cut_in_band) {
        // And what that costs: the cut an imposed level denotes tracks the spectrum instead of
        //   staying put. Inflating the eigenvalues against a fixed level makes every component look
        //   more significant, so FEWER are classified as noise, monotonically. Applying the
        //   per-channel correction to an estimator's output rather than to the spectrum would move
        //   this cut for complex data by exactly this mechanism.
        for (size_t i = 1; i != cutoff_by_scale.size(); ++i)
          check(cutoff_by_scale[i] <= cutoff_by_scale[i - 1],
                "an imposed cut is monotonically non-increasing in the scale of the spectrum" + shape_context);
        check(cutoff_by_scale.back() < cutoff_by_scale.front(),
              "and moves materially across the range of scales swept" + shape_context);
      }
    }
  }

  // The partitioned overload, with deliberately unequal partitions: every estimator must implement
  //   it, and the pooling arithmetic is entirely separate code from the single-PCA path.
  const std::vector<ssize_t> partition_sizes = {14, 10, 20};
  const ssize_t n = 56;
  const double scale = 0.5;
  std::vector<eigenvalues_type> storage_base;
  std::vector<eigenvalues_type> storage_scaled;
  storage_base.reserve(partition_sizes.size());
  storage_scaled.reserve(partition_sizes.size());
  for (const ssize_t m_p : partition_sizes) {
    const Matrix signal = make_signal(m_p, n, 3, 3.0 * sigma_true, rng);
    const Matrix x = demean_columns(draw_real(signal, sigma_true, rng));
    storage_base.push_back(spectrum_of(x, false));
    storage_scaled.push_back(eigenvalues_type(storage_base.back() * scale));
  }
  std::vector<Spectrum> spectra_base;
  std::vector<Spectrum> spectra_scaled;
  spectra_base.reserve(partition_sizes.size());
  spectra_scaled.reserve(partition_sizes.size());
  ssize_t total_rnz = 0;
  for (size_t p = 0; p != partition_sizes.size(); ++p) {
    spectra_base.emplace_back(storage_base[p], partition_sizes[p], n, 1);
    spectra_scaled.emplace_back(storage_scaled[p], partition_sizes[p], n, 1);
    total_rnz += spectra_base.back().rnz();
  }
  for (const NamedEstimator &named : all_estimators(total_rnz / 2, 1.0)) {
    if (!named.homogeneous)
      continue;
    const Result base = (*named.estimator)(spectra_base, Eigen::Vector3d::Zero());
    const Result scaled = (*named.estimator)(spectra_scaled, Eigen::Vector3d::Zero());
    const std::string context = " [" + named.name + ", partitioned]";
    check(bool(base) && bool(scaled), "partitioned threshold usable" + context);
    if (!base || !scaled)
      continue;
    check_relative(scaled.sigma2, base.sigma2 * scale, 1.0e-9, "sigma2 scales exactly" + context);
    check_relative(scaled.lamplus, base.lamplus * scale, 1.0e-9, "lamplus scales exactly" + context);
    check(scaled.cutoff_p == base.cutoff_p, "pooled classification is invariant" + context);
    check(scaled.cutoff_p_partition == base.cutoff_p_partition, "per-partition classification is invariant" + context);
  }
  std::printf("   swept: exp1 exp2 med mrm2023 tbme2022 rank (homogeneous); unity constant (imposed);\n"
              "          four single-PCA shapes and one three-way unequal partitioning\n");
}

// ---------------------------------------------------------------------------
// 3. The factor itself
// ---------------------------------------------------------------------------

void test_entry_variance() {
  std::printf("\n== 3. Casorati entry variance, and the sqrt(2) factor ==\n");
  RNG rng(20260805);
  const double sigma_true = 5.0;
  const ssize_t m = 32;
  const ssize_t n = 96;
  const ssize_t draws = 64;

  check_relative(casorati_entry_channels<double>(), 1.0, 0.0, "casorati_entry_channels is 1 for real data");
  check_relative(casorati_entry_channels<cdouble>(), 2.0, 0.0, "casorati_entry_channels is 2 for complex data");

  // The total eigenvalue energy is the squared Frobenius norm: sigma^2 m n for real entries, twice
  //   that for complex. This is the entire origin of the convention discrepancy.
  std::vector<double> energy_real;
  std::vector<double> energy_complex;
  const Matrix zero = Matrix::Zero(m, n);
  for (ssize_t d = 0; d != draws; ++d) {
    const CMatrix xc = draw_complex(zero, sigma_true, rng);
    const Matrix xr = xc.real();
    energy_complex.push_back(spectrum_of(xc, false).sum() / (double(m) * double(n)));
    energy_real.push_back(spectrum_of(xr, false).sum() / (double(m) * double(n)));
  }
  check_relative(mean_of(energy_real), MR::Math::pow2(sigma_true), 0.02, "a real entry carries variance sigma^2");
  check_relative(
      mean_of(energy_complex), 2.0 * MR::Math::pow2(sigma_true), 0.02, "a complex entry carries variance 2 sigma^2");

  // And what that means for an estimate: uncorrected, the complex estimate is sqrt(2) times the
  //   per-channel level; corrected, it is the per-channel level.
  const MRM2023 estimator{};
  std::vector<double> ratio;
  std::vector<double> corrected;
  for (ssize_t d = 0; d != draws; ++d) {
    const Matrix signal = make_signal(m, n, 3, 4.0 * sigma_true, rng);
    const CMatrix x = demean_columns(draw_complex(signal, sigma_true, rng));
    eigenvalues_type aggregate = spectrum_of(x, false);
    eigenvalues_type per_channel = spectrum_of(x, true);
    const Result ra = estimator(Spectrum(aggregate, m, n, 1), Eigen::Vector3d::Zero());
    const Result rp = estimator(Spectrum(per_channel, m, n, 1), Eigen::Vector3d::Zero());
    if (!ra || !rp)
      continue;
    ratio.push_back(std::sqrt(ra.sigma2 / rp.sigma2));
    corrected.push_back(std::sqrt(rp.sigma2) / sigma_true);
    check(ra.cutoff_p == rp.cutoff_p, "the correction does not alter the signal/noise classification");
  }
  std::printf("   uncorrected / corrected = %.12f (sqrt(2) = %.12f); corrected / truth = %.4f\n",
              median_of(ratio),
              MR::Math::sqrt2,
              median_of(corrected));
  check_relative(median_of(ratio), MR::Math::sqrt2, 1.0e-9, "the uncorrected complex estimate is sqrt(2) larger");
  check_relative(median_of(corrected), 1.0, 0.03, "the corrected complex estimate recovers the per-channel level");
}

// ---------------------------------------------------------------------------
// 4. Complex versus magnitude across SNR
// ---------------------------------------------------------------------------

// One estimate of the noise level from magnitude data, following the software's own iteration:
//   stabilise at the current level, demean, decompose, estimate, and multiply the (unit-less)
//   result back by the level that produced them (Iterative::estimate). The seed is the estimate
//   obtained with no transform at all, which is what the first row of a schedule has to work with.
double estimate_magnitude(const Matrix &magnitude,
                          const Base &model,
                          const MRM2023 &estimator,
                          const double sigma_seed,
                          const ssize_t iterations) {
  const ssize_t m = magnitude.rows();
  const ssize_t n = magnitude.cols();
  double sigma = sigma_seed;
  for (ssize_t it = 0; it != iterations; ++it) {
    Matrix u(m, n);
    for (ssize_t i = 0; i != m; ++i)
      for (ssize_t j = 0; j != n; ++j)
        u(i, j) = model.stabilise(magnitude(i, j), sigma);
    const eigenvalues_type s = spectrum_of(Matrix(demean_columns(u)), true);
    const Result result = estimator(Spectrum(s, m, n, 1), Eigen::Vector3d::Zero());
    if (!result)
      return std::numeric_limits<double>::quiet_NaN();
    sigma *= std::sqrt(result.sigma2);
  }
  return sigma;
}

// The noise level a complex patch yields, on the per-channel scale.
double estimate_complex(const CMatrix &data, const MRM2023 &estimator) {
  const eigenvalues_type s = spectrum_of(CMatrix(demean_columns(data)), true);
  const Result result = estimator(Spectrum(s, data.rows(), data.cols(), 1), Eigen::Vector3d::Zero());
  return bool(result) ? std::sqrt(result.sigma2) : std::numeric_limits<double>::quiet_NaN();
}

void test_snr_sweep() {
  std::printf("\n== 4. Noise level from complex versus magnitude data, across SNR ==\n");
  RNG rng(20260806);
  const double sigma_true = 10.0;
  const ssize_t m = 24;
  const ssize_t n = 64;
  const ssize_t rank = 4;
  const ssize_t patches = 96;

  const MRM2023 estimator{};
  const std::shared_ptr<Base> identity = make(distribution_t::RICIAN, 2.0, vst_method_t::NONE);
  const std::shared_ptr<Base> rician = make(distribution_t::RICIAN, 2.0, vst_method_t::FOI);

  std::printf("       a   complex/truth   magnitude, no VST   magnitude, non-linear VST\n");
  for (const double a : {0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0}) {
    std::vector<double> rel_complex, rel_novst, rel_vst;
    for (ssize_t p = 0; p != patches; ++p) {
      const Matrix signal = make_signal(m, n, rank, a * sigma_true, rng);
      const CMatrix xc = draw_complex(signal, sigma_true, rng);
      const Matrix magnitude = xc.cwiseAbs();

      const double complex_estimate = estimate_complex(xc, estimator);
      if (std::isfinite(complex_estimate))
        rel_complex.push_back(complex_estimate / sigma_true);
      const double novst = estimate_magnitude(magnitude, *identity, estimator, 1.0, 1);
      if (!std::isfinite(novst))
        continue;
      rel_novst.push_back(novst / sigma_true);
      const double vst = estimate_magnitude(magnitude, *rician, estimator, novst, 3);
      if (std::isfinite(vst))
        rel_vst.push_back(vst / sigma_true);
    }
    const double mc = median_of(rel_complex);
    const double mn = median_of(rel_novst);
    const double mv = median_of(rel_vst);
    std::printf("   %5.2f          %.4f              %.4f                      %.4f\n", a, mc, mn, mv);

    // The complex estimate is unaffected by the operating point: its noise is Gaussian at every SNR.
    check_near(mc, 1.0, 0.05, "the complex estimate recovers the per-channel level at a = " + std::to_string(a));
    if (a >= 8.0) {
      // High SNR: the magnitude data are effectively Gaussian, and the two representations agree.
      //   This is the property the convention exists to provide.
      check_near(mn, mc, 0.05, "magnitude and complex estimates converge at a = " + std::to_string(a));
      check_near(mv, mc, 0.05, "the transform does not disturb that agreement at a = " + std::to_string(a));
    }
    if (a <= 0.25) {
      // Near the floor they must NOT agree without a transform: the Rician noise variance falls to
      //   (2 - pi/2) sigma^2 there. Asserting the divergence keeps the agreement above from being
      //   vacuous, and bounds it, so that a change in the factor cannot hide here either.
      check(mn < 0.85 * mc, "the magnitude estimate diverges near the noise floor at a = " + std::to_string(a));
      check(mn > 0.55 * mc, "and does so by the amount the Rician distribution predicts, no more");
    }
  }

  // Patches spanning a realistic range of operating points, which is what the software's own
  //   estimation patches contain: it is here, rather than on a uniformly noise-floor patch, that
  //   the non-linear transform is expected to recover the divergence above.
  std::vector<double> graded_novst, graded_vst;
  for (ssize_t p = 0; p != patches; ++p) {
    const Matrix signal = make_signal_graded(m, n, rank, 0.2 * sigma_true, 8.0 * sigma_true, rng);
    const Matrix magnitude = draw_complex(signal, sigma_true, rng).cwiseAbs();
    const double novst = estimate_magnitude(magnitude, *identity, estimator, 1.0, 1);
    if (!std::isfinite(novst))
      continue;
    graded_novst.push_back(novst / sigma_true);
    const double vst = estimate_magnitude(magnitude, *rician, estimator, novst, 3);
    if (std::isfinite(vst))
      graded_vst.push_back(vst / sigma_true);
  }
  const double gn = median_of(graded_novst);
  const double gv = median_of(graded_vst);
  std::printf("   graded patch (a from 0.2 to 8): no VST %.4f, non-linear VST %.4f (truth 1.0)\n", gn, gv);
  check(std::fabs(gv - 1.0) < std::fabs(gn - 1.0),
        "the non-linear transform brings the magnitude estimate closer to the per-channel truth");
  check_near(gv, 1.0, 0.10, "and to within 10% of it");
}

// ---------------------------------------------------------------------------
// 5. Cross-application, in both directions
// ---------------------------------------------------------------------------

void test_cross_application() {
  std::printf("\n== 5. Applying a level estimated from one representation to the other ==\n");
  RNG rng(20260807);
  const double sigma_true = 10.0;
  const ssize_t m = 24;
  const ssize_t n = 64;
  const ssize_t patches = 96;
  const MRM2023 estimator{};
  const std::shared_ptr<Base> gaussian = make(distribution_t::GAUSSIAN, 2.0, vst_method_t::LINEAR);
  const std::shared_ptr<Base> identity = make(distribution_t::RICIAN, 2.0, vst_method_t::NONE);
  const std::shared_ptr<Base> rician = make(distribution_t::RICIAN, 2.0, vst_method_t::FOI);

  // (a) magnitude -> complex. Estimate the level from the magnitude data with the software's own
  //     non-linear transform, then use it to stabilise the complex data, as -noise_in would.
  std::vector<double> ratio_mag, var_real, var_imag, residual_sigma2;
  for (ssize_t p = 0; p != patches; ++p) {
    const Matrix signal = make_signal_graded(m, n, 4, 0.5 * sigma_true, 8.0 * sigma_true, rng);
    const CMatrix xc = draw_complex(signal, sigma_true, rng);
    const Matrix magnitude = xc.cwiseAbs();
    const double seed = estimate_magnitude(magnitude, *identity, estimator, 1.0, 1);
    const double sigma_m = std::isfinite(seed) ? estimate_magnitude(magnitude, *rician, estimator, seed, 3)
                                               : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(sigma_m) || !(sigma_m > 0.0))
      continue;
    ratio_mag.push_back(sigma_m / sigma_true);

    // Stabilise the complex data with that level, exactly as the preconditioner does: the Gaussian
    //   model's transform, one scalar applied independently to each channel. Measured on the noise
    //   alone, so that what is reported is the accuracy of the level and nothing else.
    double sr = 0.0;
    double si = 0.0;
    CMatrix u(m, n);
    for (ssize_t i = 0; i != m; ++i) {
      for (ssize_t j = 0; j != n; ++j) {
        sr += MR::Math::pow2(gaussian->stabilise(xc(i, j).real() - signal(i, j), sigma_m));
        si += MR::Math::pow2(gaussian->stabilise(xc(i, j).imag(), sigma_m));
        u(i, j) = cdouble(gaussian->stabilise(xc(i, j).real(), sigma_m), gaussian->stabilise(xc(i, j).imag(), sigma_m));
      }
    }
    var_real.push_back(sr / (double(m) * double(n)));
    var_imag.push_back(si / (double(m) * double(n)));

    // And what the Unity estimator would then see: the per-channel-scaled spectrum of the
    //   stabilised complex data, whose noise level must be 1.
    const double residual = estimate_complex(u, estimator);
    if (std::isfinite(residual))
      residual_sigma2.push_back(MR::Math::pow2(residual));
  }
  const double vr = median_of(var_real);
  const double vi = median_of(var_imag);
  std::printf("   magnitude -> complex: sigma/truth = %.4f; stabilised channel variance "
              "re = %.4f, im = %.4f; residual sigma^2 = %.4f\n",
              median_of(ratio_mag),
              vr,
              vi,
              median_of(residual_sigma2));
  check_near(vr, 1.0, 0.12, "a level estimated from magnitude data stabilises the complex real channel");
  check_near(vi, 1.0, 0.12, "a level estimated from magnitude data stabilises the complex imaginary channel");
  check_near(median_of(residual_sigma2), 1.0, 0.12, "the Unity estimator's premise holds on that stabilised data");
  // Under the aggregate convention the level supplied here would be sqrt(2) larger, halving both.
  check(std::fabs(0.5 * vr - 1.0) > 0.3, "the aggregate convention would fail that check");

  // (b) complex -> magnitude. Estimate the level from the complex data, then use it to
  //     variance-stabilise magnitude data drawn at the same true level. This is the check that the
  //     transform is being parameterised in the units it expects.
  const Matrix signal_ref = make_signal_graded(m, n, 4, 0.5 * sigma_true, 8.0 * sigma_true, rng);
  std::vector<double> sigma_c_samples;
  for (ssize_t p = 0; p != patches; ++p) {
    const double estimate = estimate_complex(draw_complex(signal_ref, sigma_true, rng), estimator);
    if (std::isfinite(estimate))
      sigma_c_samples.push_back(estimate);
  }
  const double sigma_c = median_of(sigma_c_samples);
  check_near(sigma_c / sigma_true, 1.0, 0.05, "a level estimated from complex data recovers the truth");

  std::printf("       a   Var[Phi(m)] with the complex level   ... with its aggregate (sqrt(2) x)\n");
  const ssize_t samples = 200000;
  std::normal_distribution<double> nd(0.0, sigma_true);
  for (const double a : {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}) {
    const double nu = a * sigma_true;
    double s1 = 0.0, s2 = 0.0, t1 = 0.0, t2 = 0.0;
    for (ssize_t k = 0; k != samples; ++k) {
      const double magnitude = std::hypot(nu + nd(rng), nd(rng));
      const double u = rician->stabilise(magnitude, sigma_c);
      const double v = rician->stabilise(magnitude, MR::Math::sqrt2 * sigma_c);
      s1 += u;
      s2 += u * u;
      t1 += v;
      t2 += v * v;
    }
    const double var_ok = s2 / double(samples) - MR::Math::pow2(s1 / double(samples));
    const double var_aggregate = t2 / double(samples) - MR::Math::pow2(t1 / double(samples));
    std::printf("   %5.2f                    %.4f                            %.4f\n", a, var_ok, var_aggregate);
    if (a >= 4.0) {
      check_near(var_ok, 1.0, 0.06, "the complex-derived level stabilises magnitude data at a = " + std::to_string(a));
      check(var_aggregate < 0.65, "the aggregate convention would under-stabilise at a = " + std::to_string(a));
    } else {
      // Below the point at which the transform is exact, the band the noise model's own unit test
      //   uses: stabilisation is approximate near the floor, but the level itself is the right one.
      check(var_ok > 0.6 && var_ok < 1.3,
            "the complex-derived level keeps the stabilised variance near unity at a = " + std::to_string(a));
    }
  }
}

// ---------------------------------------------------------------------------
// 6. The Unity estimator after a linear transform
// ---------------------------------------------------------------------------

void test_unity_fixed_point() {
  std::printf("\n== 6. Unity on complex data stabilised by a per-channel level ==\n");
  RNG rng(20260808);
  const double sigma_true = 10.0;
  const ssize_t m = 24;
  const ssize_t n = 64;
  const ssize_t patches = 96;
  const MRM2023 estimator{};
  const Unity unity{};
  std::vector<double> fitted;
  ssize_t cutoff_agreements = 0;
  ssize_t comparisons = 0;
  double lamplus_error = 0.0;
  for (ssize_t p = 0; p != patches; ++p) {
    const Matrix signal = make_signal_graded(m, n, 4, 0.5 * sigma_true, 8.0 * sigma_true, rng);
    // Stabilised by the TRUE per-channel level, which is what the final (reconstruction-only) row
    //   of a schedule is handed, and on which the Unity estimator's premise rests. Drawing at unit
    //   noise from a signal already divided by that level is exactly the linear transform applied.
    const CMatrix xc = draw_complex(signal / sigma_true, 1.0, rng);
    const eigenvalues_type s = spectrum_of(CMatrix(demean_columns(xc)), true);
    const Spectrum spectrum(s, m, n, 1);
    const Result fit = estimator(spectrum, Eigen::Vector3d::Zero());
    const Result imposed = unity(spectrum, Eigen::Vector3d::Zero());
    check(bool(imposed), "Unity yields a usable threshold");
    if (!fit || !imposed)
      continue;
    fitted.push_back(fit.sigma2);
    ++comparisons;
    if (std::abs(fit.cutoff_p - imposed.cutoff_p) <= 1)
      ++cutoff_agreements;
    lamplus_error += std::fabs(fit.lamplus - imposed.lamplus) / imposed.lamplus;
  }
  const double median_fit = median_of(fitted);
  const double mean_lamplus_error = lamplus_error / double(std::max<ssize_t>(1, comparisons));
  std::printf("   fitted sigma^2 on the stabilised data = %.4f (Unity imposes 1.0); "
              "cut agrees within one component in %ld/%ld patches; mean lamplus error %.4f\n",
              median_fit,
              long(cutoff_agreements),
              long(comparisons),
              mean_lamplus_error);
  check_near(median_fit, 1.0, 0.08, "a fitted level on per-channel-stabilised complex data is unity");
  check(cutoff_agreements >= (9 * comparisons) / 10,
        "Unity reproduces the fitted classification on per-channel-stabilised complex data");
  check(mean_lamplus_error < 0.08, "Unity reproduces the fitted signal/noise boundary");
}

// ---------------------------------------------------------------------------
// 7. Adaptive phase correction
// ---------------------------------------------------------------------------

// Mean absolute angular difference between two unit-magnitude phase fields, in radians.
double phase_discrepancy(const Array2D &ar, const Array2D &ai, const Array2D &br, const Array2D &bi) {
  double acc = 0.0;
  for (ssize_t j = 0; j != ar.cols(); ++j) {
    for (ssize_t i = 0; i != ar.rows(); ++i) {
      const double dot = ar(i, j) * br(i, j) + ai(i, j) * bi(i, j);
      const double cross = ar(i, j) * bi(i, j) - ai(i, j) * br(i, j);
      acc += std::fabs(std::atan2(cross, dot));
    }
  }
  return acc / double(ar.rows() * ar.cols());
}

// The noise level solve_plane() self-calibrated for a plane, recovered through the public
//   interface rather than re-implemented. The self-calibration branch and a CONSTANT noise map
//   differ in exactly one quantity: with a constant sigma the fidelity weights sigmabar^2/sigma^2
//   are identically one and the calibration domain is the whole plane, which is precisely what the
//   self-calibrated branch imposes unconditionally. So the two solves differ only in the noise
//   level driving the discrepancy criterion, and the constant map that reproduces the
//   self-calibrated solve is the level that solve_plane calibrated for itself.
// The budget is cut to one lambda evaluation with no polish, so that the phase depends on the noise
//   level through the eq. 7 seed alone. That makes the bisected statistic -- how far the solved
//   phase has moved from the raw data phase, which grows monotonically with the level, since a
//   larger level means a smaller lambda and more smoothing -- monotone, and each evaluation cheap.
double measure_self_calibrated_sigma(const Array2D &fr, const Array2D &fi, const double lo, const double hi) {
  const ssize_t Nx = fr.rows();
  const ssize_t Ny = fr.cols();
  AdaptivePhaseEstimator::Params params;
  params.cp_iter = 16;
  params.max_lambda_iter = 1;
  params.polish_iter = 0;
  const AdaptivePhaseEstimator apc(std::vector<size_t>{0, 1}, params);
  auto departure = [&](const Array2D &sigma) {
    Array2D pr = Array2D::Ones(Nx, Ny);
    Array2D pi = Array2D::Zero(Nx, Ny);
    if (!apc.solve_plane(fr, fi, sigma, pr, pi, false))
      return std::numeric_limits<double>::quiet_NaN();
    double acc = 0.0;
    for (ssize_t j = 0; j != Ny; ++j) {
      for (ssize_t i = 0; i != Nx; ++i) {
        const double dot = pr(i, j) * fr(i, j) + pi(i, j) * fi(i, j);
        const double cross = pr(i, j) * fi(i, j) - pi(i, j) * fr(i, j);
        acc += std::fabs(std::atan2(cross, dot));
      }
    }
    return acc / double(Nx * Ny);
  };
  const double target = departure(Array2D::Constant(Nx, Ny, -1.0)); // -1 routes to self-calibration
  double a = lo;
  double b = hi;
  for (int iteration = 0; iteration != 24; ++iteration) { // geometric bisection: the level spans decades
    const double m = std::sqrt(a * b);
    if (departure(Array2D::Constant(Nx, Ny, m)) < target)
      a = m;
    else
      b = m;
  }
  return std::sqrt(a * b);
}

void test_apc_scale() {
  std::printf("\n== 7. Adaptive phase correction: supplied level versus self-calibration ==\n");
  RNG rng(20260809);
  const ssize_t Nx = 48;
  const ssize_t Ny = 48;
  const double sigma_true = 6.0;

  // A smooth magnitude field and a smooth background phase: what APC is designed for, and what
  //   keeps its self-calibration from the data an honest measure of the noise level (tissue
  //   structure would inflate it, confounding the comparison being made here).
  Array2D fr(Nx, Ny), fi(Nx, Ny);
  std::normal_distribution<double> nd(0.0, sigma_true);
  for (ssize_t j = 0; j != Ny; ++j) {
    for (ssize_t i = 0; i != Nx; ++i) {
      const double x = (double(i) - 0.5 * double(Nx - 1)) / double(Nx);
      const double y = (double(j) - 0.5 * double(Ny - 1)) / double(Ny);
      const double magnitude = 120.0 * std::exp(-4.0 * (x * x + y * y));
      const double phase =
          1.4 * x + 0.9 * y + 0.8 * std::sin(2.0 * MR::Math::pi * x) * std::cos(2.0 * MR::Math::pi * y);
      fr(i, j) = magnitude * std::cos(phase) + nd(rng);
      fi(i, j) = magnitude * std::sin(phase) + nd(rng);
    }
  }

  const AdaptivePhaseEstimator apc{};
  const Array2D sigma_per_channel = Array2D::Constant(Nx, Ny, sigma_true);
  const Array2D sigma_aggregate = Array2D::Constant(Nx, Ny, MR::Math::sqrt2 * sigma_true);
  const Array2D sigma_absent = Array2D::Constant(Nx, Ny, -1.0); // routes to MAD self-calibration

  Array2D pr_self(Nx, Ny), pi_self(Nx, Ny), pr_ok(Nx, Ny), pi_ok(Nx, Ny), pr_agg(Nx, Ny), pi_agg(Nx, Ny);
  auto solve = [&](const Array2D &sigma, Array2D &pr, Array2D &pi) {
    pr = Array2D::Ones(Nx, Ny);
    pi = Array2D::Zero(Nx, Ny);
    return apc.solve_plane(fr, fi, sigma, pr, pi, false);
  };
  check(solve(sigma_absent, pr_self, pi_self), "the self-calibrated APC solve succeeds");
  check(solve(sigma_per_channel, pr_ok, pi_ok), "the APC solve with a per-channel noise map succeeds");
  check(solve(sigma_aggregate, pr_agg, pi_agg), "the APC solve with an aggregate noise map succeeds");

  const double d_ok = phase_discrepancy(pr_ok, pi_ok, pr_self, pi_self);
  const double d_aggregate = phase_discrepancy(pr_agg, pi_agg, pr_self, pi_self);
  std::printf("   mean |phase difference| from the self-calibrated solve: "
              "per-channel map %.5f rad, aggregate map %.5f rad\n",
              d_ok,
              d_aggregate);
  // The map-driven and self-calibrated branches are calibrated from the same quantity and must
  //   therefore agree. Under the aggregate convention the map handed to APC was sqrt(2) larger,
  //   which inflated its regularisation and put the first row of a schedule -- which has no map and
  //   self-calibrates -- on a different scale to every row after it.
  check(d_ok < d_aggregate, "a per-channel map agrees with self-calibration more closely than an aggregate one");

  // And the same statement quantitatively: what the self-calibration actually estimated. The
  //   residual few per cent is the phantom's own smooth magnitude and phase gradient, which any
  //   estimator built from finite differences counts as noise; it is present for every functional
  //   and is not what this test is about.
  const double sigma_self = measure_self_calibrated_sigma(fr, fi, 0.05 * sigma_true, 20.0 * sigma_true);
  std::printf("   stationary phantom: self-calibrated sigma = %.4f, supplied %.4f (ratio %.4f)\n",
              sigma_self,
              sigma_true,
              sigma_self / sigma_true);
  check_relative(sigma_self, sigma_true, 0.15, "self-calibration recovers the supplied level on a stationary plane");
}

// A noise level that varies smoothly across the plane by the two orders of magnitude that parallel
//   imaging produces in vivo. Here the two branches' functionals part company: the map branch forms
//   sqrt(mean(sigma^2)), which is what the discrepancy criterion needs -- with unit weights it
//   targets a residual energy of Npix * sigmabar^2, and that equals the plane's noise energy
//   sum_x sigma^2(x) only for the mean of sigma^2 -- while a robust centre of the same field lands
//   near its median, six-fold lower. A stationary phantom cannot see the difference.
void test_apc_nonstationary() {
  std::printf("\n== 7b. Adaptive phase correction on a non-stationary noise field ==\n");
  RNG rng(20260810);
  const ssize_t Nx = 96;
  const ssize_t Ny = 96;
  const double sigma_lo = 7.0;   // periphery
  const double sigma_hi = 400.0; // centre, where parallel-imaging amplification peaks

  Array2D fr(Nx, Ny), fi(Nx, Ny), sigma(Nx, Ny);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (ssize_t j = 0; j != Ny; ++j) {
    for (ssize_t i = 0; i != Nx; ++i) {
      const double x = (double(i) - 0.5 * double(Nx - 1)) / double(Nx);
      const double y = (double(j) - 0.5 * double(Ny - 1)) / double(Ny);
      const double r2 = x * x + y * y;
      const double s = sigma_lo * std::pow(sigma_hi / sigma_lo, std::exp(-12.0 * r2));
      const double magnitude = 1400.0 * std::exp(-4.0 * r2);
      const double phase =
          1.4 * x + 0.9 * y + 0.8 * std::sin(2.0 * MR::Math::pi * x) * std::cos(2.0 * MR::Math::pi * y);
      sigma(i, j) = s;
      fr(i, j) = magnitude * std::cos(phase) + s * nd(rng);
      fi(i, j) = magnitude * std::sin(phase) + s * nd(rng);
    }
  }

  // What the map branch would form from this field, by its own definition.
  const double sigmabar_map = std::sqrt(sigma.square().mean());
  std::vector<double> flat(sigma.data(), sigma.data() + sigma.size());
  const double sigma_median = median_of(flat);
  // Without this the test would be vacuous: on a field whose RMS and median coincide, a robust
  //   centre passes the check below just as well as the functional the criterion requires.
  check(sigmabar_map > 2.5 * sigma_median,
        "the phantom's noise field separates its root mean square from its median");

  const double sigma_self = measure_self_calibrated_sigma(fr, fi, 0.02 * sigmabar_map, 20.0 * sigmabar_map);
  std::printf("   sigma spans %.1f to %.1f; map branch sqrt(mean(sigma^2)) = %.2f, its median = %.2f;\n"
              "   self-calibrated sigma = %.2f (ratio to the map branch %.4f)\n",
              sigma.minCoeff(),
              sigma.maxCoeff(),
              sigmabar_map,
              sigma_median,
              sigma_self,
              sigma_self / sigmabar_map);
  check_relative(sigma_self,
                 sigmabar_map,
                 0.10,
                 "self-calibration recovers the map branch's level on a non-stationary plane");

  // Note what is deliberately NOT asserted: that the self-calibrated phase resembles the phase the
  //   non-stationary map itself produces. Those two solves differ in their fidelity weights as well
  //   as in their noise level -- the map branch weights each voxel by sigmabar^2/sigma^2, over two
  //   orders of magnitude here -- so their phases differ for reasons that have nothing to do with
  //   the quantity under test. The stationary phantom above, where those weights are identically
  //   one, is where a phase comparison between the branches means anything.
}

} // namespace

int main() {
  test_homogeneity();
  test_entry_variance();
  test_snr_sweep();
  test_cross_application();
  test_unity_fixed_point();
  test_apc_scale();
  test_apc_nonstationary();

  if (failures == 0)
    std::fprintf(stderr, "\nnoise_convention: all checks passed\n");
  else
    std::fprintf(stderr, "\nnoise_convention: %d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
