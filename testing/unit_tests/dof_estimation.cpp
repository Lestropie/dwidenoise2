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

// Monte Carlo evaluation of data-driven estimation of the magnitude noise model's degrees of
// freedom from the residual of the filtered PCA reconstruction (dof_estimation_plan.md sections 1
// and 8). This is the stage-3 decision gate of that plan's staging, and it is the artifact that
// decided it.
//
// ---------------------------------------------------------------------------------------------
// VERDICT: the scheme does not work on data carrying signal of ordinary rank. Do not build on it.
// ---------------------------------------------------------------------------------------------
//
// Two findings, in order of discovery. Both are reproduced by the experiments below, and the
// numbers quoted are those the executable prints.
//
// 1. The plan's equation (1) is a first-order approximation whose error exceeds the effect it is
//    meant to isolate. The transform satisfies Phi'(theta) = 1 / sqrt(V(a; nu_hat)), which
//    stabilises the variance only under the delta method; in fact, for correctly specified Rician
//    data, Var[Phi(m) | a] measures 0.786 at a = 0, 1.129 at a = 2 and 1.000 by a = 8. That 44%
//    excursion has the same monotone shape as, and is larger than, the entire signature that
//    distinguishes one value of the degrees of freedom from another (V(a;8)/V(a;2) spans 1.13 to
//    0.78 over the same range). Modelled as the ratio V(a;nu)/V(a;nu_hat), the prediction is flat
//    where the data are strongly curved, the objective is monotone, and the fit runs to a bound at
//    every iteration whatever the truth: nu* is not a fixed point.
//
//    This is repairable, and is repaired here: StabilisedVariance computes the exact quantity
//    W(a; nu) = Var[ Phi(m) | a, nu ] by the same Poisson-mixture quadrature the noise model
//    already uses to build its exact-unbiased inverse. At nu = nu* the model then reproduces the
//    observed profile including its curvature, so the truth becomes a fixed point by construction.
//
// 2. With that repair, and under conditions more favourable than the software could ever provide
//    -- the noise level held at the truth, the transform built at the truth, the operating points
//    taken from the exact-unbiased inverse, the residual operator (I - S)P modelled exactly rather
//    than approximated, and the fit made sample-wise so that no binning error enters -- the
//    estimator recovers the truth to within 9% for patches whose signal is of rank 0 or 1
//    (2 -> 1.96/1.84, 4 -> 3.90/3.93, 7.52 -> 7.34/6.96, 16 -> 16.7/16.9). At rank 4 it
//    overestimates by a factor of about three (2 -> 6.55, 4 -> 14.8, 7.52 -> 24.4, 16 -> 46.4),
//    and that bias is unchanged by quadrupling the patch voxel count, by any choice of
//    eigenspectrum filter, by restricting the fit to low-SNR samples, and by making the signal
//    exactly low-rank in the stabilised domain rather than in intensity.
//
//    The cause is measured directly (experiment_contamination): the ratio of the real residual's
//    power to that of a residual made of filtered noise alone is flat to within ~5% at rank 0 and
//    1, but at rank 4 it rises from ~0.90 at low SNR to ~1.46 (nu = 7.52) or ~2.08 (nu = 16) at
//    high SNR. That 60-110% tilt is three to five times the full model tilt spanning a fourfold
//    range of the degrees of freedom. What the filter leaves behind beyond filtered noise is
//    therefore not a second-order correction to the signature, as sections 1.6.1 and 1.6.2 of the
//    plan assume; at ordinary signal ranks it is several times larger than the signature, and it
//    has the same shape.
//
// The machinery itself is sound, and experiment_machinery asserts as much: fed a residual that
// obeys the model by construction, the same objective returns the same answer whatever the patch's
// signal rank (a spread of ~15% across ranks 0, 1 and 4, against the threefold spread the real
// residuals produce). The failure is therefore in what the real residual contains, not in how it is
// fitted -- which is why no refinement of the fit addresses it. That check carries a residual
// upward offset of ~20% of its own, because the noise it draws is scaled by the TRUE operating
// points while the fit reads back the ESTIMATED ones; that mismatch is a real, if much smaller,
// error source in its own right, and it is why the check's tolerance is loose.
//
// Modes:
//   (no arguments)     regression test: sampler cross-validation, plus the machinery checks that
//                      must hold for the evidence above to mean anything. Non-zero exit on failure.
//   --evidence         print all four experiments, ie. reproduce the numbers quoted above.
//   --sweep <file.tsv> the evidence as a machine-readable table as well.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "denoise/denoise.h"
#include "denoise/estimator/datadriven/mrm2023.h"
#include "denoise/estimator/result.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "denoise/spectrum.h"
#include "denoise/spectrum_filter.h"
#include "math/noncentral_chi.h"

using MR::Denoise::eigenvalues_type;
using MR::Denoise::filter_type;
using MR::Denoise::Moments;
using MR::Denoise::Spectrum;
using MR::Denoise::SpectrumFilter;
using MR::Denoise::vector_type;
using MR::Denoise::Estimator::MRM2023;
using MR::Denoise::Precondition::NoiseModel::Base;
using MR::Denoise::Precondition::NoiseModel::distribution_t;
using MR::Denoise::Precondition::NoiseModel::make;
using MR::Denoise::Precondition::NoiseModel::vst_method_t;
using MR::Math::NonCentralChi::normalised_mean;
using MR::Math::NonCentralChi::normalised_variance;

namespace {

using RNG = std::mt19937_64;
using Matrix = Eigen::MatrixXd;

constexpr double MIN_DOF = 1.0;  // NonCentralChi::min_dof
constexpr double MAX_DOF = 64.0; // upper bound of the search, ie. 32 effective channels
constexpr ssize_t NUM_GRID = 65; // candidate values, logarithmically spaced over [MIN_DOF, MAX_DOF]

int failures = 0;

void check(const bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
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

std::vector<double> candidate_grid() {
  std::vector<double> grid(NUM_GRID);
  for (ssize_t g = 0; g != NUM_GRID; ++g)
    grid[g] = MIN_DOF * std::pow(MAX_DOF / MIN_DOF, double(g) / double(NUM_GRID - 1));
  return grid;
}

// ---------------------------------------------------------------------------
// 1. Sampling the non-central chi
// ---------------------------------------------------------------------------

// Integer channel count: the definition. N complex Gaussians of per-component standard deviation
//   sigma, the first offset by the underlying level nu = a * sigma, combined by sum of squares.
double sample_integer(const double a, const int channels, const double sigma, RNG &rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  double acc = 0.0;
  for (int c = 0; c != channels; ++c) {
    const double re = nd(rng) + (c == 0 ? a : 0.0);
    const double im = nd(rng);
    acc += re * re + im * im;
  }
  return sigma * std::sqrt(acc);
}

// Real (possibly fractional) degrees of freedom, via the Poisson mixture representation of the
//   non-central chi-square that the noise model's own exact-unbiased inverse is built on:
//     ncchi^2_{dof}(lambda) = chi^2_{dof + 2J},  J ~ Poisson(lambda / 2),  lambda = a^2.
//   Exact for any real dof > 0, and reduces to the integer sampler above at dof = 2N. That
//   equivalence is checked rather than assumed: see test_samplers().
double sample_fractional(const double a, const double dof, const double sigma, RNG &rng) {
  std::poisson_distribution<int> poisson(0.5 * a * a);
  const int j = poisson(rng);
  // chi^2_k == Gamma(shape = k/2, scale = 2)
  std::gamma_distribution<double> gamma(0.5 * dof + double(j), 2.0);
  return sigma * std::sqrt(gamma(rng));
}

double sample_magnitude(const double a, const double dof, const double sigma, RNG &rng) {
  const int channels = int(std::lround(0.5 * dof));
  const bool integral = std::fabs(dof - 2.0 * double(channels)) < 1.0e-12;
  return integral ? sample_integer(a, channels, sigma, rng) : sample_fractional(a, dof, sigma, rng);
}

// ---------------------------------------------------------------------------
// 2. What the stabilised data's variance actually is
// ---------------------------------------------------------------------------

// The variance-stabilising transform is stabilising only to FIRST order: its defining property
//   Phi'(theta) = 1 / sqrt(V(a; nu_hat)) makes Var[Phi(m) | a] constant under the delta method
//   alone. See finding 1 in the header for how large that discrepancy is relative to the effect
//   being measured.
//
// This class supplies the exact quantity instead:
//
//     W(a; nu)  =  Var[ Phi(m) | a, degrees of freedom nu ]
//
//   -- the variance the data would actually have, after the transform in force, if their degrees of
//   freedom were nu. At nu = nu*, W reproduces the observed profile including its curvature, so the
//   truth is a fixed point by construction, and the residual dependence on nu is what identifies it.
//
// A non-central chi with nu degrees of freedom and normalised signal a is a Poisson(a^2/2) mixture
//   of central chi distributions of order nu + 2j, so
//     E[Phi^k | a] = sum_j Poisson(j; a^2/2) E_j^(k),   E_j^(k) = integral Phi(x)^k p(x; nu + 2j) dx
//   with the inner integrals independent of a and hence formed once per candidate nu. The central
//   chi densities are advanced across orders by the recurrence p(x; k+2) = p(x; k) x^2 / k carried
//   in log space, so the density stays accurate at its peak for every order; the quadrature is
//   confined to the window in which each order has support (a central chi has standard deviation
//   ~1/sqrt(2) whatever its order).
class StabilisedVariance {
public:
  // - model : the noise model whose forward transform is in force, ie. the one built at nu_hat
  // - a_max : the largest operating point that will be asked for
  // - dof_max : the largest candidate value of the degrees of freedom that will be asked for
  StabilisedVariance(const Base &model, const double a_max, const double dof_max)
      : jmax(ssize_t(std::ceil(0.5 * a_max * a_max + 8.0 * std::sqrt(0.5 * a_max * a_max + 1.0))) + 5), dx(0.02) {
    const double x_max = std::sqrt(dof_max + 2.0 * double(jmax)) + 10.0;
    const ssize_t n = ssize_t(std::lround(x_max / dx)) + 1;
    x.resize(n);
    log_x.resize(n);
    phi.resize(n);
    phi2.resize(n);
    for (ssize_t i = 0; i != n; ++i) {
      x[i] = double(i) * dx;
      log_x[i] = i == 0 ? -std::numeric_limits<double>::infinity() : std::log(x[i]);
      phi[i] = model.stabilise(x[i], 1.0);
      phi2[i] = phi[i] * phi[i];
    }
    lgamma_j.resize(jmax + 1);
    for (ssize_t j = 0; j <= jmax; ++j)
      lgamma_j[j] = std::lgamma(double(j + 1));
  }

  void evaluate(const std::vector<double> &a_values, const double dof, std::vector<double> &variance) const {
    std::vector<double> ignored;
    evaluate(a_values, dof, variance, ignored);
  }

  // W(a; dof) -- and, since the quadrature yields it anyway, E[Phi(m) | a, dof] -- for every a in
  //   "a_values", for one candidate dof.
  void evaluate(const std::vector<double> &a_values,
                const double dof,
                std::vector<double> &variance,
                std::vector<double> &mean) const {
    const ssize_t n = ssize_t(x.size());
    std::vector<double> log_p(n);
    const double log_norm = (0.5 * dof - 1.0) * std::log(2.0) + std::lgamma(0.5 * dof);
    for (ssize_t i = 0; i != n; ++i)
      log_p[i] = (dof - 1.0) * log_x[i] - 0.5 * x[i] * x[i] - log_norm;
    std::vector<double> e1(jmax + 1, 0.0);
    std::vector<double> e2(jmax + 1, 0.0);
    for (ssize_t j = 0; j <= jmax; ++j) {
      const double k = dof + 2.0 * double(j);
      const double centre = std::sqrt(std::max(k - 1.0, 0.0));
      const ssize_t lo = std::max<ssize_t>(0, ssize_t((centre - 9.0) / dx));
      const ssize_t hi = std::min<ssize_t>(n - 1, ssize_t((centre + 9.0) / dx) + 1);
      double mass = 0.0;
      double m1 = 0.0;
      double m2 = 0.0;
      double previous_p = 0.0;
      for (ssize_t i = lo; i <= hi; ++i) {
        // The chi density ~ x^(dof-1) is unbounded at the origin for dof < 1; every integrand of
        //   interest vanishes there (Phi(0) = 0 and Phi ~ x), so the origin contributes nothing and
        //   is not formed as an indeterminate product.
        const double p = (i == 0 || log_p[i] < -700.0) ? 0.0 : std::exp(log_p[i]);
        if (i != lo) {
          mass += 0.5 * (previous_p + p);
          m1 += 0.5 * (phi[i - 1] * previous_p + phi[i] * p);
          m2 += 0.5 * (phi2[i - 1] * previous_p + phi2[i] * p);
        }
        previous_p = p;
      }
      // Normalising by the quadrature's own mass, rather than by dx, makes the result insensitive
      //   to truncation of the window and to the grid step.
      e1[j] = mass > 0.0 ? m1 / mass : 0.0;
      e2[j] = mass > 0.0 ? m2 / mass : 0.0;
      const double log_scale = -std::log(k);
      for (ssize_t i = 0; i != n; ++i)
        log_p[i] += 2.0 * log_x[i] + log_scale;
    }

    variance.assign(a_values.size(), std::numeric_limits<double>::quiet_NaN());
    mean.assign(a_values.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t index = 0; index != a_values.size(); ++index) {
      const double a = a_values[index];
      if (!(a >= 0.0))
        continue;
      const double lambda = 0.5 * a * a;
      double first = e1[0];
      double second = e2[0];
      if (lambda > 0.0) {
        const double log_lambda = std::log(lambda);
        double log_max = -std::numeric_limits<double>::infinity();
        for (ssize_t j = 0; j <= jmax; ++j)
          log_max = std::max(log_max, -lambda + double(j) * log_lambda - lgamma_j[j]);
        double total = 0.0;
        first = 0.0;
        second = 0.0;
        for (ssize_t j = 0; j <= jmax; ++j) {
          const double w = std::exp(-lambda + double(j) * log_lambda - lgamma_j[j] - log_max);
          total += w;
          first += w * e1[j];
          second += w * e2[j];
        }
        first /= total;
        second /= total;
      }
      variance[index] = std::max(second - first * first, 1.0e-9);
      mean[index] = first;
    }
  }

private:
  const ssize_t jmax;
  const double dx;
  std::vector<double> x;
  std::vector<double> log_x;
  std::vector<double> phi; // the transform in force, tabulated once and shared by every candidate
  std::vector<double> phi2;
  std::vector<double> lgamma_j;
};

// Which model of the observable the fit uses. EXACT is the one above; DELTA is the first-order
//   ratio V(a; nu) / V(a; nu_hat) that the plan proposes, retained so that the difference between
//   them can be demonstrated rather than merely asserted.
enum class model_t { EXACT, DELTA };

const char *name_of(const model_t model) { return model == model_t::EXACT ? "exact" : "delta"; }

// ---------------------------------------------------------------------------
// 3. Synthetic patches, and what one PCA pass yields from them
// ---------------------------------------------------------------------------

class PatchSpec {
public:
  ssize_t m = 16;    // volumes
  ssize_t n = 32;    // voxels
  ssize_t rank = 4;  // rank of the underlying signal (0 = none beyond a constant across volumes)
  double a_lo = 0.2; // lowest per-voxel mean operating point
  double a_hi = 6.0; // highest
  ssize_t patches = 96;
};

// Underlying operating points a_ij = nu_ij / sigma of one patch: a rank-r non-negative matrix whose
//   columns are rescaled to means spanning [a_lo, a_hi] logarithmically. Column scaling preserves
//   the rank, so the signal remains exactly r-dimensional in volume space while the range of SNR is
//   controlled independently of it.
// Rank 0 denotes a patch whose signal is constant across volumes, and hence entirely absorbed by
//   the demeaning: the decomposition then sees pure noise, while the samples still span a range of
//   operating points. It is the most favourable case the estimator can be given.
Matrix make_operating_points(const PatchSpec &spec, RNG &rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  Matrix result(spec.m, spec.n);
  if (spec.rank == 0) {
    for (ssize_t j = 0; j != spec.n; ++j)
      result.col(j).setConstant(spec.a_lo * std::pow(spec.a_hi / spec.a_lo, double(j) / double(spec.n - 1)));
    return result;
  }
  Matrix a(spec.m, spec.rank);
  Matrix b(spec.n, spec.rank);
  for (ssize_t i = 0; i != spec.m; ++i)
    for (ssize_t k = 0; k != spec.rank; ++k)
      a(i, k) = std::fabs(nd(rng));
  for (ssize_t j = 0; j != spec.n; ++j)
    for (ssize_t k = 0; k != spec.rank; ++k)
      b(j, k) = std::fabs(nd(rng));
  result = a * b.transpose();
  for (ssize_t j = 0; j != spec.n; ++j) {
    const double target = spec.a_lo * std::pow(spec.a_hi / spec.a_lo, double(j) / double(spec.n - 1));
    const double mean = result.col(j).mean();
    result.col(j) *= (mean > 0.0 ? target / mean : 0.0);
  }
  return result;
}

// Everything one patch contributes to the fit.
class PatchPass {
public:
  bool ok = false;
  double sigma = std::numeric_limits<double>::quiet_NaN();
  Matrix residual_sq;     // r_ij^2, the observable
  Matrix operating_point; // a_ij, recovered from the reconstruction
  Matrix mixing;          // squared entries of the composite residual operator (see below)
  Matrix truth;           // the operating points actually used to draw the data (diagnosis only)
};

// Stabilise at (dof_hat, sigma), demean, decompose, estimate the noise level, filter, project, and
//   report the residual together with the operating point of every sample. This is the sequence the
//   software performs, in the order it performs it; only the plumbing differs.
//
// The composite residual operator is reported alongside, because it is not (I - S). The residual is
//   (I - S) applied to the DEMEANED data, and the demeaning is itself a projector applied to the
//   independent stabilised samples: r = (I - S) P u with P = I - 11^T/m. It is that composite whose
//   squared entries weight the per-sample variances,
//     E[r_ij^2] = sum_k [(I - S) P]_ik^2 W(a_kj; nu),
//   so a sample's residual is NOT a measurement of its own operating point alone -- it draws on
//   every volume at the same voxel. Modelling that mixing exactly is what section 1.3 of the plan
//   omits (its cancellation argument holds in the voxel direction but not in the volume direction),
//   and it is necessary before the estimator has any accuracy at all.
PatchPass process_patch(const Matrix &magnitude,
                        const Base &model,
                        const double sigma_in,
                        const SpectrumFilter &filter,
                        const MRM2023 &estimator) {
  PatchPass result;
  const ssize_t m = magnitude.rows();
  const ssize_t n = magnitude.cols();
  const ssize_t r = std::min(m, n);

  Matrix u(m, n);
  for (ssize_t i = 0; i != m; ++i)
    for (ssize_t j = 0; j != n; ++j)
      u(i, j) = model.stabilise(magnitude(i, j), sigma_in);

  // Per-voxel demeaning over the single group of volumes; one component of every column is thereby
  //   forced to zero, which is what the preconditioner rank of 1 below states.
  const Eigen::RowVectorXd means = u.colwise().mean();
  const Matrix x = u.rowwise() - means;

  Eigen::SelfAdjointEigenSolver<Matrix> eig;
  eig.compute(x * x.transpose(), Eigen::ComputeEigenvectors);
  if (eig.info() != Eigen::Success)
    return result;
  eigenvalues_type s = eig.eigenvalues().cwiseMax(0.0);
  const Spectrum spectrum(s, m, n, 1);
  const MR::Denoise::Estimator::Result threshold = estimator(spectrum, Eigen::Vector3d::Zero());
  if (!threshold)
    return result;
  // First-order refinement of the noise level: on stabilised data the estimate is a unit-less
  //   correction factor to the level that produced them.
  result.sigma = sigma_in * std::sqrt(threshold.sigma2);

  vector_type w(r);
  Moments moments;
  if (!filter.weights(spectrum, threshold, -1, w, moments))
    return result;
  const Matrix contraction = eig.eigenvectors() * w.head(r).matrix().asDiagonal() * eig.eigenvectors().transpose();
  const Matrix demean = Matrix::Identity(m, m) - Matrix::Constant(m, m, 1.0 / double(m));
  const Matrix composite = (Matrix::Identity(m, m) - contraction) * demean;
  const Matrix reconstructed = contraction * x;
  result.residual_sq = (x - reconstructed).array().square().matrix();
  result.mixing = composite.array().square().matrix();

  // The operating point of each sample, from the reconstruction with the demeaning offset added
  //   back. inverse_unbiased() returns sigma * psi(u); at sigma == 1 it is the normalised level
  //   a = nu / sigma directly, which is what the model is expressed in.
  const Matrix restored = reconstructed.rowwise() + means;
  result.operating_point.resize(m, n);
  for (ssize_t i = 0; i != m; ++i)
    for (ssize_t j = 0; j != n; ++j)
      result.operating_point(i, j) = std::max(0.0, model.inverse_unbiased(restored(i, j), 1.0));
  result.ok = true;
  return result;
}

// ---------------------------------------------------------------------------
// 4. The fit
// ---------------------------------------------------------------------------

class Corpus {
public:
  std::vector<PatchPass> patches;
  double a_max = 8.0;
};

// Draw a corpus of patches and take each through one PCA pass at the stated degrees of freedom.
// The noise level is held at the truth throughout: this is a test of the degrees of freedom, and
//   letting the noise level be wrong as well would only confound the two.
Corpus make_corpus(const PatchSpec &spec,
                   const double dof_true,
                   const double dof_hat,
                   const double sigma_true,
                   const filter_type ftype,
                   RNG &rng) {
  Corpus corpus;
  corpus.a_max = std::min(12.0, std::max(5.0, 2.0 * std::sqrt(dof_hat)));
  auto model = make(distribution_t::NONCENTRALCHI, dof_hat, vst_method_t::FOI);
  const SpectrumFilter filter(ftype);
  const MRM2023 estimator;
  for (ssize_t p = 0; p != spec.patches; ++p) {
    const Matrix a = make_operating_points(spec, rng);
    Matrix magnitude(spec.m, spec.n);
    for (ssize_t i = 0; i != spec.m; ++i)
      for (ssize_t j = 0; j != spec.n; ++j)
        magnitude(i, j) = sample_magnitude(a(i, j), dof_true, sigma_true, rng);
    PatchPass pass = process_patch(magnitude, *model, sigma_true, filter, estimator);
    if (!pass.ok)
      continue;
    pass.truth = a;
    corpus.patches.push_back(std::move(pass));
  }
  return corpus;
}

// Replace every patch's residual by one that obeys the model exactly: independent stabilised noise
//   of precisely the per-sample variance the model claims, combined with the weights of the same
//   composite residual operator. Nothing then contaminates it -- no signal the filter failed to
//   capture, and no dependence between the filter and the noise it is filtering.
void impose_model_conforming_residuals(Corpus &corpus, const Base &model, const double dof_true, RNG &rng) {
  const StabilisedVariance variance(model, corpus.a_max, MAX_DOF);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (PatchPass &patch : corpus.patches) {
    const ssize_t m = patch.truth.rows();
    const ssize_t n = patch.truth.cols();
    std::vector<double> a_values(size_t(m * n));
    for (ssize_t i = 0; i != m; ++i)
      for (ssize_t j = 0; j != n; ++j)
        a_values[size_t(i * n + j)] = std::min(std::max(patch.truth(i, j), 0.0), corpus.a_max);
    std::vector<double> w;
    variance.evaluate(a_values, dof_true, w);
    Matrix noise(m, n);
    for (ssize_t i = 0; i != m; ++i)
      for (ssize_t j = 0; j != n; ++j)
        noise(i, j) = std::sqrt(w[size_t(i * n + j)]) * nd(rng);
    // Only the squared entries of the composite operator enter the variance, and it is those that
    //   "mixing" holds; the residual is therefore formed directly from that decomposition.
    Matrix residual_sq(m, n);
    for (ssize_t i = 0; i != m; ++i)
      for (ssize_t j = 0; j != n; ++j) {
        double accumulated = 0.0;
        for (ssize_t k = 0; k != m; ++k)
          accumulated += patch.mixing(i, k) * noise(k, j) * noise(k, j);
        residual_sq(i, j) = accumulated;
      }
    patch.residual_sq = residual_sq;
  }
}

class FitResult {
public:
  double dof = std::numeric_limits<double>::quiet_NaN();
  bool at_bound = false;
  std::vector<double> objective;
};

// Sample-wise least squares: for each candidate, predict every sample's residual power from the
//   model and the composite residual operator, normalise both prediction and observation by their
//   own means over the patch (which is what eliminates the unknown per-patch scale c_p), and sum
//   the squared discrepancy. Binning, as the plan proposes, is a coarsening of this; doing it
//   sample-wise removes binning error from consideration entirely.
FitResult fit_corpus(const Corpus &corpus,
                     const Base &model,
                     const double dof_hat,
                     const model_t model_choice,
                     const bool model_mixing = true,
                     const double a_ceiling = std::numeric_limits<double>::infinity()) {
  FitResult result;
  const std::vector<double> grid = candidate_grid();
  // The model tabulated on a fine grid of operating points for each candidate, then looked up per
  //   sample. One tabulation of the transform in force is shared by every candidate.
  constexpr ssize_t NUM_A = 240;
  std::vector<double> a_grid(NUM_A);
  for (ssize_t i = 0; i != NUM_A; ++i)
    a_grid[size_t(i)] = corpus.a_max * double(i) / double(NUM_A - 1);
  std::vector<std::vector<double>> wtab(NUM_GRID);
  switch (model_choice) {
  case model_t::EXACT: {
    const StabilisedVariance variance(model, corpus.a_max, MAX_DOF);
    for (ssize_t g = 0; g != NUM_GRID; ++g)
      variance.evaluate(a_grid, grid[size_t(g)], wtab[size_t(g)]);
  } break;
  case model_t::DELTA:
    for (ssize_t g = 0; g != NUM_GRID; ++g) {
      wtab[size_t(g)].resize(NUM_A);
      for (ssize_t i = 0; i != NUM_A; ++i)
        wtab[size_t(g)][size_t(i)] =
            normalised_variance(a_grid[size_t(i)], grid[size_t(g)]) / normalised_variance(a_grid[size_t(i)], dof_hat);
    }
    break;
  }

  result.objective.assign(NUM_GRID, 0.0);
  ssize_t best = 0;
  for (ssize_t g = 0; g != NUM_GRID; ++g) {
    double total = 0.0;
    for (const PatchPass &patch : corpus.patches) {
      const ssize_t m = patch.residual_sq.rows();
      const ssize_t n = patch.residual_sq.cols();
      Matrix w(m, n);
      for (ssize_t i = 0; i != m; ++i)
        for (ssize_t j = 0; j != n; ++j) {
          const double t = std::min(patch.operating_point(i, j), corpus.a_max) / corpus.a_max * double(NUM_A - 1);
          const ssize_t k = std::min<ssize_t>(NUM_A - 2, ssize_t(t));
          const double frac = t - double(k);
          w(i, j) = wtab[size_t(g)][size_t(k)] * (1.0 - frac) + wtab[size_t(g)][size_t(k + 1)] * frac;
        }
      const Matrix prediction = model_mixing ? (patch.mixing * w).eval() : w;
      double observed_sum = 0.0;
      double predicted_sum = 0.0;
      double count = 0.0;
      for (ssize_t i = 0; i != m; ++i)
        for (ssize_t j = 0; j != n; ++j)
          if (patch.operating_point(i, j) < a_ceiling) {
            observed_sum += patch.residual_sq(i, j);
            predicted_sum += prediction(i, j);
            count += 1.0;
          }
      if (count < 8.0 || !(observed_sum > 0.0) || !(predicted_sum > 0.0))
        continue;
      const double observed_mean = observed_sum / count;
      const double predicted_mean = predicted_sum / count;
      for (ssize_t i = 0; i != m; ++i)
        for (ssize_t j = 0; j != n; ++j)
          if (patch.operating_point(i, j) < a_ceiling) {
            const double d = patch.residual_sq(i, j) / observed_mean - prediction(i, j) / predicted_mean;
            total += d * d;
          }
    }
    result.objective[size_t(g)] = total;
    if (total < result.objective[size_t(best)])
      best = g;
  }
  result.at_bound = (best == 0 || best == NUM_GRID - 1);
  if (result.at_bound) {
    result.dof = grid[size_t(best)];
  } else {
    // Parabolic refinement in log(dof), the variable the grid is uniform in.
    const double y0 = result.objective[size_t(best - 1)];
    const double y1 = result.objective[size_t(best)];
    const double y2 = result.objective[size_t(best + 1)];
    const double denominator = y0 - 2.0 * y1 + y2;
    const double delta = denominator > 0.0 ? 0.5 * (y0 - y2) / denominator : 0.0;
    const double log_step = std::log(MAX_DOF / MIN_DOF) / double(NUM_GRID - 1);
    result.dof = std::exp(std::log(grid[size_t(best)]) + std::max(-1.0, std::min(1.0, delta)) * log_step);
  }
  return result;
}

// ---------------------------------------------------------------------------
// 5. Experiments
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

// The transform is not exactly variance-stabilising, and the size of that discrepancy relative to
//   the effect being measured is finding 1.
void experiment_transform(std::FILE *out) {
  std::fprintf(out, "\n[1] Is the variance-stabilising transform variance-stabilising?\n");
  std::fprintf(out, "    %8s %12s %12s %10s %10s %10s\n", "a", "Var[Phi_2]", "Var[Phi_8]", "V(a;2)", "V(a;8)", "V8/V2");
  auto model2 = make(distribution_t::NONCENTRALCHI, 2.0, vst_method_t::FOI);
  auto model8 = make(distribution_t::NONCENTRALCHI, 8.0, vst_method_t::FOI);
  const std::vector<double> a_values{0.0, 0.5, 1.0, 2.0, 4.0, 8.0};
  std::vector<double> w2;
  std::vector<double> w8;
  StabilisedVariance(*model2, 8.0, 8.0).evaluate(a_values, 2.0, w2);
  StabilisedVariance(*model8, 8.0, 8.0).evaluate(a_values, 8.0, w8);
  for (size_t i = 0; i != a_values.size(); ++i)
    std::fprintf(out,
                 "    %8.3f %12.5f %12.5f %10.5f %10.5f %10.5f\n",
                 a_values[i],
                 w2[i],
                 w8[i],
                 normalised_variance(a_values[i], 2.0),
                 normalised_variance(a_values[i], 8.0),
                 normalised_variance(a_values[i], 8.0) / normalised_variance(a_values[i], 2.0));
  std::fprintf(out,
               "    For correctly specified data the stabilised variance is NOT constant: it swings by ~44%%\n"
               "    across this range, more than the ~30%% by which one candidate value of the degrees of\n"
               "    freedom differs from another, and in the same monotone direction.\n");
}

// Fed a residual that obeys the model by construction, the objective must recover the truth, and
//   must do so independently of the patch's signal rank. This is the check that separates an error
//   in the machinery from a confound in the data; it is asserted, because everything else rests on
//   it. The tolerance is loose because the corpus is small: the point is the absence of the
//   threefold, rank-dependent bias that the real residuals produce, not the precision.
void experiment_machinery(std::FILE *out) {
  std::fprintf(out, "\n[2] Does the fit recover the truth from data that obey the model?\n");
  std::fprintf(out, "    %10s %6s %12s\n", "dof_true", "rank", "estimate");
  for (const double dof_true : {2.0, 7.52, 16.0}) {
    for (const ssize_t rank : {0, 1, 4}) {
      PatchSpec spec;
      spec.rank = rank;
      spec.patches = 96;
      RNG rng(20260731);
      Corpus corpus = make_corpus(spec, dof_true, dof_true, 20.0, filter_type::OPTSHRINK, rng);
      auto model = make(distribution_t::NONCENTRALCHI, dof_true, vst_method_t::FOI);
      impose_model_conforming_residuals(corpus, *model, dof_true, rng);
      const FitResult fitted = fit_corpus(corpus, *model, dof_true, model_t::EXACT);
      std::fprintf(out, "    %10.3f %6lld %12.3f\n", dof_true, (long long)rank, fitted.dof);
      check(std::fabs(fitted.dof / dof_true - 1.0) < 0.6,
            "model-conforming residuals recover the truth (dof " + std::to_string(dof_true) + ", rank " +
                std::to_string(rank) + ")");
    }
  }
}

// The estimator on real residuals, started AT the truth, so that what it reports is precisely
//   whether the truth is a fixed point. Findings 1 and 2.
void experiment_estimator(std::FILE *out, std::FILE *tsv) {
  std::fprintf(out, "\n[3] The estimator on real residuals, started AT the truth\n");
  std::fprintf(out, "    %10s %6s %8s %10s %12s %8s\n", "dof_true", "rank", "model", "estimate", "rel. error", "bound");
  for (const double dof_true : {2.0, 4.0, 7.52, 16.0}) {
    for (const ssize_t rank : {0, 1, 4}) {
      for (const model_t model_choice : {model_t::DELTA, model_t::EXACT}) {
        PatchSpec spec;
        spec.rank = rank;
        spec.patches = 96;
        RNG rng(20260731);
        const Corpus corpus = make_corpus(spec, dof_true, dof_true, 20.0, filter_type::OPTSHRINK, rng);
        auto model = make(distribution_t::NONCENTRALCHI, dof_true, vst_method_t::FOI);
        const FitResult fitted = fit_corpus(corpus, *model, dof_true, model_choice);
        std::fprintf(out,
                     "    %10.3f %6lld %8s %10.3f %11.1f%% %8s\n",
                     dof_true,
                     (long long)rank,
                     name_of(model_choice),
                     fitted.dof,
                     100.0 * (fitted.dof / dof_true - 1.0),
                     fitted.at_bound ? "yes" : "no");
        if (tsv != nullptr)
          std::fprintf(tsv,
                       "estimator\t%.4g\t%lld\t%s\t%.5g\t%.5g\n",
                       dof_true,
                       (long long)rank,
                       name_of(model_choice),
                       fitted.dof,
                       fitted.dof / dof_true - 1.0);
        if (model_choice == model_t::EXACT && rank <= 1)
          check(std::fabs(fitted.dof / dof_true - 1.0) < 0.30,
                "with the exact model and no signal subspace, the truth is a fixed point (dof " +
                    std::to_string(dof_true) + ", rank " + std::to_string(rank) + ")");
      }
    }
  }
}

// Why the estimator fails at ordinary signal rank: the real residual's power, relative to that of a
//   residual made of filtered noise alone, as a function of the operating point. Finding 2.
void experiment_contamination(std::FILE *out, std::FILE *tsv) {
  std::fprintf(out, "\n[4] Contamination of the residual: real power / filtered-noise-only power\n");
  constexpr ssize_t NUM_BINS = 8;
  for (const double dof_true : {7.52, 16.0}) {
    for (const ssize_t rank : {0, 1, 4}) {
      PatchSpec spec;
      spec.rank = rank;
      spec.patches = 96;
      RNG rng(20260731);
      const Corpus real = make_corpus(spec, dof_true, dof_true, 20.0, filter_type::OPTSHRINK, rng);
      Corpus conforming = real;
      auto model = make(distribution_t::NONCENTRALCHI, dof_true, vst_method_t::FOI);
      RNG rng2(918273);
      impose_model_conforming_residuals(conforming, *model, dof_true, rng2);
      std::vector<double> real_sum(NUM_BINS, 0.0);
      std::vector<double> conforming_sum(NUM_BINS, 0.0);
      std::vector<double> counts(NUM_BINS, 0.0);
      for (size_t p = 0; p != real.patches.size(); ++p) {
        const double real_mean = real.patches[p].residual_sq.mean();
        const double conforming_mean = conforming.patches[p].residual_sq.mean();
        if (!(real_mean > 0.0) || !(conforming_mean > 0.0))
          continue;
        for (ssize_t i = 0; i != real.patches[p].residual_sq.rows(); ++i)
          for (ssize_t j = 0; j != real.patches[p].residual_sq.cols(); ++j) {
            const double a = std::min(real.patches[p].operating_point(i, j), real.a_max);
            const ssize_t b = std::min(NUM_BINS - 1, ssize_t(a / real.a_max * double(NUM_BINS)));
            real_sum[size_t(b)] += real.patches[p].residual_sq(i, j) / real_mean;
            conforming_sum[size_t(b)] += conforming.patches[p].residual_sq(i, j) / conforming_mean;
            counts[size_t(b)] += 1.0;
          }
      }
      std::fprintf(out, "    dof %6.3f rank %lld:", dof_true, (long long)rank);
      for (ssize_t b = 0; b != NUM_BINS; ++b) {
        // An under-populated bin reports a ratio of two nearly-empty averages, which is noise
        //   rather than evidence; the highest bin of a low-dof cell is routinely such a bin.
        if (counts[size_t(b)] < 200.0)
          continue;
        const double ratio =
            (real_sum[size_t(b)] / counts[size_t(b)]) / (conforming_sum[size_t(b)] / counts[size_t(b)]);
        std::fprintf(out, " a<%.1f:%.3f", real.a_max * double(b + 1) / double(NUM_BINS), ratio);
        if (tsv != nullptr)
          std::fprintf(tsv,
                       "contamination\t%.4g\t%lld\ta<%.4g\t%.5g\n",
                       dof_true,
                       (long long)rank,
                       real.a_max * double(b + 1) / double(NUM_BINS),
                       ratio);
      }
      std::fprintf(out, "\n");
    }
  }
  std::fprintf(out,
               "    Flat to within ~5%% where there is no signal subspace; strongly sloped at rank 4, by\n"
               "    three to five times the tilt that separates candidate values of the degrees of freedom.\n");
}

void print_verdict(std::FILE *out) {
  std::fprintf(out,
               "\nVERDICT (dof_estimation_plan.md section 9, stage 3): the residual-based estimator does not\n"
               "have a stable fixed point at the truth for patches carrying signal of ordinary rank, and the\n"
               "obstruction is not in the fit but in what the residual contains. Stages 4 to 6 of the plan\n"
               "are not worth building on it. See the header of this file for the full statement.\n");
}

} // namespace

int main(int argc, char **argv) {
  bool evidence = false;
  const char *tsv_path = nullptr;
  for (int i = 1; i != argc; ++i) {
    if (std::strcmp(argv[i], "--evidence") == 0)
      evidence = true;
    else if (std::strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
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
    std::fprintf(tsv, "experiment\tdof_true\trank\tkey\tvalue\n");
  }

  test_samplers();
  experiment_machinery(evidence ? stdout : stderr);
  if (evidence) {
    experiment_transform(stdout);
    experiment_estimator(stdout, tsv);
    experiment_contamination(stdout, tsv);
    print_verdict(stdout);
  } else {
    experiment_estimator(stderr, nullptr);
  }

  if (tsv != nullptr)
    std::fclose(tsv);
  if (failures == 0)
    std::printf("dof_estimation: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
