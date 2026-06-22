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

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

// Multiple-criteria rank / noise-level estimator of Zhu et al. (2022),
//   "Denoise Functional Magnetic Resonance Imaging With Random Matrix Theory Based
//    Principal Component Analysis", IEEE Trans. Biomed. Eng. 69(11):3377-3388.
//
// Distinction from the Veraart / Cordero-Grande estimators (-estimator exp1/exp2):
//   The Marchenko-Pastur (MP) law describes the limiting distribution of the *eigenvalues*
//   of the noise sample covariance matrix; the "generalized quarter-circle" law is precisely
//   that same law expressed for the *singular values* s = sqrt(lambda) (it degenerates to a
//   literal quarter-circle when the matrix is square, beta == 1). The two are therefore the
//   same random-matrix phenomenon viewed in different coordinates, not competing models.
//   The methodological difference lies entirely in how many summary statistics of that bulk
//   are exploited to locate the signal/noise boundary:
//     - Exp1/Exp2 (MPPCA) use a *single* criterion, equivalent to matching the k == 2 moment
//       (the mean eigenvalue, sigma^2) against the bulk width; this can under-estimate rank.
//     - This estimator exploits the fact that *every* moment of the quarter-circle law is a
//       known multiple of sigma^k (Eq. 8): for each moment order k it forms two independent
//       noise estimates and uses their crossover as a separate rank criterion, then combines
//       the family of criteria. Using moments up to k == 10 makes the boundary more robust.
//   With K == 1 (single criterion, k == 2) this estimator reduces structurally to MPPCA.
//
// For a candidate noise set comprising the smallest N nonzero singular values:
//   sigma1(k) : from the empirical k-th moment of those singular values, normalised by the
//               quarter-circle moment constant C_k (Eq. 9);
//   sigma2(k) : from the bulk edges s_+ , s_- approximated by the largest / smallest noise
//               singular value (Eq. 10).
// The set is consistent with being pure noise iff sigma1(k) >= sigma2(k) (Eq. 11), because
//   sigma2(k) is the more sensitive estimator and is inflated when residual signal contaminates
//   the noise set. For each k the turnover (last boundary at which the inequality holds) yields
//   a candidate rank r(k); the effective rank is R = max_k r(k) (retain the most signal, i.e.
//   the criterion that first detects signal governs the cut), and sigma = max_k sigma1(k).
class TBME2022 : public Base {
public:
  TBME2022(const ssize_t max_moment_order = 10) : K(max_moment_order) { assert(K >= 1); }

  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s, //
                    const ssize_t m,                              //
                    const ssize_t n,                              //
                    const ssize_t rp,                             //
                    const Eigen::Vector3d & /*unused*/) const final {
    assert(s.size() == std::min(m, n));
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t rnz = rank_nonzero(m, n, rp);
    const ssize_t qnz = dimlong_nonzero(m, n, rp);
    Result result;
    // Need at least two nonzero noise candidates to define a bulk
    if (rnz < 2)
      return result;

    const double beta = double(rnz) / double(qnz);
    const double sqrtbeta = std::sqrt(beta);

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

  // Partitioned form: the multi-moment scan runs on the pooled (normalised) singular values of
  //   all partitions, using the rnz-weighted mean aspect ratio for the quarter-circle constants
  //   (valid because the partitions are sized to share a common beta). The pooled signal/noise
  //   boundary is then applied to each partition to obtain its own rank.
  Result operator()(const std::vector<eigenvalues_type> &s, //
                    const std::vector<ssize_t> &m,           //
                    const std::vector<ssize_t> &n,           //
                    const std::vector<ssize_t> &rp,          //
                    const Eigen::Vector3d & /*unused*/) const final {
    const std::vector<PartitionDims> d = partition_dims(m, n, rp);
    Result result;
    // Pooled, sorted-ascending normalised singular values sv = sqrt(s_p[i]/qnz_p).
    std::vector<double> sv;
    {
      const std::vector<double> pooled_lam = pool_normalized(s, d); // normalised eigenvalues
      sv.reserve(pooled_lam.size());
      for (const double lam : pooled_lam)
        sv.push_back(std::sqrt(lam));
    }
    const ssize_t Ntot = ssize_t(sv.size());
    if (Ntot < 2)
      return result;

    const double beta = mean_beta(d);
    const double sqrtbeta = std::sqrt(beta);
    std::vector<double> C;
    quarter_circle_moments(beta, K, C);
    std::vector<double> edge_denom(K + 1, 0.0);
    for (ssize_t k = 1; k <= K; ++k)
      edge_denom[k] = std::pow(1.0 + sqrtbeta, double(k)) - std::pow(1.0 - sqrtbeta, double(k));

    const double sv_min = sv.front();
    std::vector<double> svmin_pow(K + 1, 1.0);
    for (ssize_t k = 1; k <= K; ++k)
      svmin_pow[k] = svmin_pow[k - 1] * sv_min;

    std::vector<ssize_t> cutoff_k(K + 1, -1);
    std::vector<double> P(K + 1, 0.0);
    for (ssize_t i = 0; i != Ntot; ++i) {
      const double sv_max = sv[i];
      double svk = 1.0;
      for (ssize_t k = 0; k <= K; ++k) {
        P[k] += svk;
        svk *= sv_max;
      }
      const ssize_t N = i + 1;
      double svmaxk = sv_max;
      for (ssize_t k = 1; k <= K; ++k) {
        const double sigma1pow = P[k] / (C[k] * double(N));
        const double sigma2pow = (svmaxk - svmin_pow[k]) / edge_denom[k];
        if (sigma1pow >= sigma2pow)
          cutoff_k[k] = i + 1;
        svmaxk *= sv_max;
      }
    }
    ssize_t cutoff_pooled = std::numeric_limits<ssize_t>::max();
    for (ssize_t k = 1; k <= K; ++k) {
      if (cutoff_k[k] >= 0)
        cutoff_pooled = std::min(cutoff_pooled, cutoff_k[k]);
    }
    if (cutoff_pooled == std::numeric_limits<ssize_t>::max())
      return result;

    const ssize_t Nnoise = cutoff_pooled;
    std::vector<double> Pn(K + 1, 0.0);
    for (ssize_t j = 0; j != cutoff_pooled; ++j) {
      const double svj = sv[j];
      double svk = 1.0;
      for (ssize_t k = 0; k <= K; ++k) {
        Pn[k] += svk;
        svk *= svj;
      }
    }
    double sigma = 0.0;
    for (ssize_t k = 1; k <= K; ++k)
      sigma = std::max(sigma, std::pow(Pn[k] / (C[k] * double(Nnoise)), 1.0 / double(k)));

    // Boundary in normalised eigenvalue units = (largest noise singular value)^2; classify each
    //   partition against it to reproduce the pooled rank with monotone per-partition cutoffs.
    const double tstar = Math::pow2(sv[cutoff_pooled - 1]);
    double noise_sum = 0.0;
    ssize_t noise_count = 0;
    apply_threshold(s, d, tstar, result, noise_sum, noise_count);
    result.sigma2 = Math::pow2(sigma);
    result.lamplus = tstar;
    return result;
  }

  bool supports_partitioning() const final { return true; }

protected:
  const ssize_t K;

  // Normalised moments C[k] = E[s^k] (for k = 0..K) of the generalized quarter-circle density
  //   f(s) proportional to sqrt((s^2 - s_-^2)(s_+^2 - s^2)) / s  on  [s_-, s_+] = [1-sqrt(beta), 1+sqrt(beta)]
  //   (Eq. 2 with sigma == 1). Any constant prefactor cancels in the normalisation C[k] = M[k]/M[0],
  //   so the ambiguous prefactor of Eq. (2) is irrelevant. Evaluated by midpoint quadrature; the
  //   integrand vanishes at both edges, so the midpoint rule avoids the boundary singularities.
  static void quarter_circle_moments(const double beta, const ssize_t K, std::vector<double> &C) {
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
};

} // namespace MR::Denoise::Estimator
