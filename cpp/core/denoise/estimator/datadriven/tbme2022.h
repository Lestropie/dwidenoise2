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

#pragma once

#include <cassert>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

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

  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final;

  // Partitioned form: the multi-moment scan run over the concatenated spectrum, in standardised
  //   order so that candidates enter the noise set by significance within their own bulk.
  //
  // The two criteria of the method depend on the aspect ratio in fundamentally different ways, and
  //   unequal partitions therefore demand two different treatments of beta:
  //
  // - sigma1(k) is built from a *moment* of the candidate set. Moments of a mixture average
  //   linearly over its components, so the exact pooled constant is the weight-average
  //   sum_p (k_p/N) C_k(beta_p) of the per-partition quarter-circle constants. (C_k is nonlinear
  //   in beta, so evaluating it once at an average beta is not the same thing, and errs by an
  //   amount that grows with the spread of the partition sizes.) Accumulating the running sums
  //   sum_p k_p C_k(beta_p) makes the count N cancel entirely.
  //
  // - sigma2(k) is built from the *support* of the candidate set. The support of a mixture is
  //   that of its widest component alone -- (1+sqrt(beta))^k - (1-sqrt(beta))^k is increasing in
  //   beta -- so the observed singular-value range must be divided by the edge difference of the
  //   largest beta present, not of an average one.
  Result operator()(const std::vector<Spectrum> &spectra, //
                    const Eigen::Vector3d & /*unused*/) const final;

  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n) over m in [30,2000], n >= m
  //   and rank density a in [0.25,4] (sigma = 1; conservative theta = 1.5 fit). R^2 = 0.94.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final;

protected:
  const ssize_t K;

  // Normalised moments C[k] = E[s^k] (for k = 0..K) of the generalized quarter-circle density
  //   f(s) proportional to sqrt((s^2 - s_-^2)(s_+^2 - s^2)) / s  on  [s_-, s_+] = [1-sqrt(beta), 1+sqrt(beta)]
  //   (Eq. 2 with sigma == 1). Any constant prefactor cancels in the normalisation C[k] = M[k]/M[0],
  //   so the ambiguous prefactor of Eq. (2) is irrelevant. Evaluated by midpoint quadrature; the
  //   integrand vanishes at both edges, so the midpoint rule avoids the boundary singularities.
  static void quarter_circle_moments(const double beta, const ssize_t K, std::vector<double> &C);
};

} // namespace MR::Denoise::Estimator
