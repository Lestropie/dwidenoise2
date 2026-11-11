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

#include "denoise/estimator/datadriven/tbme2022.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "denoise/estimator/pooling.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

Result TBME2022::operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const {
  const auto &s = spectrum.eigenvalues();
  const ssize_t rz = spectrum.rz();
  const ssize_t qnz = spectrum.qnz();
  Result result;
  // Need at least two nonzero noise candidates to define a bulk
  if (spectrum.rnz() < 2)
    return result;

  const double beta = spectrum.beta();
  const double sqrtbeta = spectrum.sqrt_beta();

  // Moment constants of the generalized quarter-circle law (sigma == 1), C[k] for k = 0..K;
  //   C[0] == 1 by construction, and C[2] == 1 reproduces the MPPCA mean-eigenvalue estimate.
  std::vector<double> C;
  quarter_circle_moments(beta, K, C);

  // Theoretical bulk-edge denominators of Eq. (10): (1 + sqrt(beta))^k - (1 - sqrt(beta))^k
  std::vector<double> edge_denom(K + 1, 0.0);
  for (ssize_t k = 1; k <= K; ++k)
    edge_denom[k] = std::pow(1.0 + sqrtbeta, double(k)) - std::pow(1.0 - sqrtbeta, double(k));

  // Singular values (paper convention s_i = sqrt(lambda_i)); s is ascending, so the smallest
  //   nonzero singular value (lower bulk edge s_-) is at index rz
  const double sv_min = std::sqrt(s[rz] / double(qnz));
  std::vector<double> svmin_pow(K + 1, 1.0);
  for (ssize_t k = 1; k <= K; ++k)
    svmin_pow[k] = svmin_pow[k - 1] * sv_min;

  // First pass: for each moment order k, find p_k = largest boundary p at which the candidate
  //   noise set {rz..p} still satisfies the noise criterion sigma1(k) >= sigma2(k). Because
  //   sigma2(k) is inflated once a signal component contaminates the set, the turnover p_k marks
  //   that criterion's signal/noise boundary. Running power sums P[k] = sum of (sv)^k over the set.
  std::vector<ssize_t> cutoff_k(K + 1, -1);
  std::vector<double> P(K + 1, 0.0);

  for (ssize_t p = rz; p != s.size(); ++p) {
    const double sv_max = std::sqrt(s[p] / double(qnz));
    double svk = 1.0;
    for (ssize_t k = 0; k <= K; ++k) {
      P[k] += svk;
      svk *= sv_max;
    }
    const ssize_t N = p + 1 - rz;
    double svmaxk = sv_max; // (sv_max)^k, starting at k == 1
    for (ssize_t k = 1; k <= K; ++k) {
      // sigma1(k)^k vs sigma2(k)^k; the >= test is invariant to the common 1/k root
      const double sigma1pow = P[k] / (C[k] * double(N));
      const double sigma2pow = (svmaxk - svmin_pow[k]) / edge_denom[k];
      if (sigma1pow >= sigma2pow)
        cutoff_k[k] = p + 1;
      svmaxk *= sv_max;
    }
  }

  // Effective rank R = max_k r(k): take the criterion that detects signal earliest (the most
  //   sensitive), i.e. the fewest noise components / smallest cutoff_p. This is the safeguard
  //   against the rank under-estimation that a single (k == 2) criterion can suffer.
  ssize_t cutoff_p = std::numeric_limits<ssize_t>::max();
  for (ssize_t k = 1; k <= K; ++k) {
    if (cutoff_k[k] >= 0)
      cutoff_p = std::min(cutoff_p, cutoff_k[k]);
  }
  if (cutoff_p == std::numeric_limits<ssize_t>::max())
    return result;

  // Noise level sigma = max_k sigma1(k), with every sigma1(k) evaluated over the *same* selected
  //   noise set {rz..cutoff_p-1}. Evaluating on a common pure-noise set (rather than each
  //   criterion's own turnover) keeps the estimate near the true sigma; the max guards against
  //   the downward bias of the higher-order empirical moments under finite sampling.
  const ssize_t Nnoise = cutoff_p - rz;
  std::vector<double> Pn(K + 1, 0.0);
  for (ssize_t j = rz; j != cutoff_p; ++j) {
    const double sv = std::sqrt(s[j] / double(qnz));
    double svk = 1.0;
    for (ssize_t k = 0; k <= K; ++k) {
      Pn[k] += svk;
      svk *= sv;
    }
  }
  double sigma = 0.0;
  for (ssize_t k = 1; k <= K; ++k)
    sigma = std::max(sigma, std::pow(Pn[k] / (C[k] * double(Nnoise)), 1.0 / double(k)));

  result.cutoff_p = cutoff_p;
  result.sigma2 = Math::pow2(sigma);
  result.lamplus = Math::pow2(1.0 + sqrtbeta) * result.sigma2;
  return result;
}

Result TBME2022::operator()(const std::vector<Spectrum> &spectra, //
                            const Eigen::Vector3d & /*unused*/) const {
  Result result;
  const std::vector<PooledEigenvalue> pooled = pool_standardized(spectra);
  const ssize_t Ntot = ssize_t(pooled.size());
  // Need at least two nonzero noise candidates to define a bulk
  if (Ntot < 2)
    return result;

  // Quarter-circle moment constants of each partition's own law.
  std::vector<std::vector<double>> C(spectra.size());
  for (size_t p = 0; p != spectra.size(); ++p)
    quarter_circle_moments(spectra[p].beta(), K, C[p]);

  // Bulk-edge denominators of Eq. (10) at the widest aspect ratio among the partitions
  //   represented in the candidate set so far; beta_running rises monotonically as partitions
  //   are drawn in, so these need rebuilding only when it does.
  double beta_running = 0.0;
  std::vector<double> edge_denom(K + 1, 0.0);

  std::vector<ssize_t> k(spectra.size(), 0); // candidates taken from each partition
  std::vector<double> Csum(K + 1, 0.0);      // sum_p k_p C_k(beta_p)
  std::vector<double> Pk(K + 1, 0.0);        // empirical power sums over the candidate set
  std::vector<double> svmin_pow(K + 1, 1.0);
  std::vector<ssize_t> cutoff_k(K + 1, -1);
  double sv_min = std::numeric_limits<double>::max();
  double sv_max = 0.0;

  for (ssize_t i = 0; i != Ntot; ++i) {
    const PooledEigenvalue &entry = pooled[i];
    const Spectrum &spectrum = spectra[entry.partition];
    ++k[entry.partition];
    const double sv = std::sqrt(entry.lambda);
    if (sv < sv_min) {
      sv_min = sv;
      double svk = 1.0;
      for (ssize_t j = 0; j <= K; ++j) {
        svmin_pow[j] = svk;
        svk *= sv_min;
      }
    }
    if (spectrum.beta() > beta_running) {
      beta_running = spectrum.beta();
      const double sqrtbeta = std::sqrt(beta_running);
      for (ssize_t j = 1; j <= K; ++j)
        edge_denom[j] = std::pow(1.0 + sqrtbeta, double(j)) - std::pow(1.0 - sqrtbeta, double(j));
    }
    double svk = 1.0;
    for (ssize_t j = 0; j <= K; ++j) {
      Pk[j] += svk;
      Csum[j] += C[entry.partition][j];
      svk *= sv;
    }
    // The candidate set's largest singular value: standardised order is not singular-value
    //   order once the bulks differ, so this is a running maximum rather than the current entry.
    sv_max = std::max(sv_max, sv);
    double svmaxk = sv_max; // (sv_max)^k, starting at k == 1
    for (ssize_t j = 1; j <= K; ++j) {
      // sigma1(k)^k vs sigma2(k)^k; the >= test is invariant to the common 1/k root
      const double sigma1pow = Pk[j] / Csum[j];
      const double sigma2pow = (svmaxk - svmin_pow[j]) / edge_denom[j];
      if (sigma1pow >= sigma2pow)
        cutoff_k[j] = i + 1;
      svmaxk *= sv_max;
    }
  }

  ssize_t cutoff_pooled = std::numeric_limits<ssize_t>::max();
  for (ssize_t j = 1; j <= K; ++j) {
    if (cutoff_k[j] >= 0)
      cutoff_pooled = std::min(cutoff_pooled, cutoff_k[j]);
  }
  if (cutoff_pooled == std::numeric_limits<ssize_t>::max())
    return result;

  // Noise level from the selected noise set, with the same per-partition moment constants.
  std::vector<double> Pn(K + 1, 0.0);
  std::vector<double> Cn(K + 1, 0.0);
  for (ssize_t j = 0; j != cutoff_pooled; ++j) {
    const double sv = std::sqrt(pooled[j].lambda);
    double svk = 1.0;
    for (ssize_t q = 0; q <= K; ++q) {
      Pn[q] += svk;
      Cn[q] += C[pooled[j].partition][q];
      svk *= sv;
    }
  }
  double sigma = 0.0;
  for (ssize_t j = 1; j <= K; ++j)
    sigma = std::max(sigma, std::pow(Pn[j] / Cn[j], 1.0 / double(j)));

  apply_threshold_standardized(spectra, pooled[cutoff_pooled - 1].z, result);
  result.sigma2 = Math::pow2(sigma);
  result.lamplus = result.sigma2 * mean_edge(spectra);
  return result;
}

double TBME2022::predicted_rmse(const ssize_t m, const ssize_t n, const double r) const {
  const double lm = std::log(double(m));
  const double ln = std::log(double(n));
  const double lr = std::log(std::max(r, 1.0));
  return std::exp(1.77080 - 0.77121 * lm - 0.96847 * ln + 0.70667 * lr + 0.068829 * lm * ln);
}

void TBME2022::quarter_circle_moments(const double beta, const ssize_t K, std::vector<double> &C) {
  const double sqrtbeta = std::sqrt(beta);
  const double sminus = 1.0 - sqrtbeta;
  const double splus = 1.0 + sqrtbeta;
  const double sminus2 = Math::pow2(sminus);
  const double splus2 = Math::pow2(splus);
  constexpr ssize_t Nq = 1024;
  const double ds = (splus - sminus) / double(Nq);
  std::vector<double> M(K + 1, 0.0);
  for (ssize_t i = 0; i != Nq; ++i) {
    const double sval = sminus + (double(i) + 0.5) * ds;
    const double rad = (Math::pow2(sval) - sminus2) * (splus2 - Math::pow2(sval));
    if (rad <= 0.0)
      continue;
    const double kernel = std::sqrt(rad) / sval; // f(s) up to the constant prefactor
    double sk = 1.0;
    for (ssize_t k = 0; k <= K; ++k) {
      M[k] += sk * kernel;
      sk *= sval;
    }
  }
  C.assign(K + 1, 0.0);
  for (ssize_t k = 0; k <= K; ++k)
    C[k] = M[k] / M[0];
}

} // namespace MR::Denoise::Estimator
