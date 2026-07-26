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

#include "denoise/estimator/datadriven/exp.h"

#include <cmath>
#include <limits>
#include <vector>

// Necessary for precompiler flags
#include "denoise/denoise.h"
#include "denoise/estimator/pooling.h"

namespace MR::Denoise::Estimator {

template <ssize_t version>
Result Exp<version>::operator()(const Eigen::VectorBlock<eigenvalues_type> s, //
                                const ssize_t m,                              //
                                const ssize_t n,                              //
                                const ssize_t rp,                             //
                                const Eigen::Vector3d & /*unused*/) const {   //
  assert(s.size() == std::min(m, n));
  const ssize_t qnz = dimlong_nonzero(m, n, rp);
  const ssize_t rz = rank_zero(m, n, rp);
  const double lam_r = s[rz] / qnz;
  double clam = 0.0;
  Result result;
  // Note that the paper utilised symbol "p" to refer to the number of signal components;
  //   here "p" is instead the index of the last noise component;
  //   therefore the number of noise compoments is (p + 1 - z)
  for (ssize_t p = rz; p < s.size(); ++p) {
    const double lam = s[p] / qnz;
    clam += lam;
    double denominator = std::numeric_limits<double>::signaling_NaN();
    switch (version) {
    case 1:
      denominator = qnz;
      break;
    case 2:
      denominator = qnz - (s.size() - p - 1);
      break;
    default:
      assert(false);
    }
    const double gam = double(p + 1 - rz) / denominator;
    const double sigsq1 = clam / double(p + 1 - rz);
    const double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
    // sigsq2 > sigsq1 if signal else noise
    if (sigsq2 < sigsq1) {
      result.sigma2 = sigsq1;
      result.cutoff_p = p + 1;
      result.lamplus = lam;
    }
  }
  return result;
}

template <ssize_t version>
Result Exp<version>::operator()(const std::vector<eigenvalues_type> &s, //
                                const std::vector<ssize_t> &m,          //
                                const std::vector<ssize_t> &n,          //
                                const std::vector<ssize_t> &rp,         //
                                const Eigen::Vector3d & /*unused*/) const {
  const std::vector<PartitionDims> d = partition_dims(m, n, rp);
  // Pooled, sorted-ascending normalised noise eigenvalues across partitions.
  const std::vector<double> pooled = pool_normalized(s, d);
  Result result;
  if (pooled.empty())
    return result;
  const double lam_r = pooled.front(); // smallest normalised nonzero eigenvalue
  // Version 1 normalises gamma by the aggregate long dimension sum_p qnz_p; version 2 applies
  //   each partition's finite-size correction, which (summed) is sum_p (qnz_p - rnz_p) plus the
  //   running noise count -- the exact generalisation of the single-PCA "qnz - (size-p-1)".
  const double Q = double(total_qnz(d));
  ssize_t denom_const = 0;
  for (const auto &x : d)
    denom_const += (x.qnz - x.rnz);
  double clam = 0.0;
  double best_sigma2 = std::numeric_limits<double>::signaling_NaN();
  double tstar = std::numeric_limits<double>::signaling_NaN();
  bool found = false;
  for (ssize_t i = 0; i != ssize_t(pooled.size()); ++i) {
    const double lam = pooled[i];
    clam += lam;
    const ssize_t count = i + 1; // number of pooled noise components considered
    double denominator = std::numeric_limits<double>::signaling_NaN();
    switch (version) {
    case 1:
      denominator = Q;
      break;
    case 2:
      denominator = double(denom_const + count);
      break;
    default:
      assert(false);
    }
    const double gam = double(count) / denominator;
    const double sigsq1 = clam / double(count);
    const double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
    if (sigsq2 < sigsq1) {
      best_sigma2 = sigsq1;
      tstar = lam;
      found = true;
    }
  }
  if (!found)
    return result;
  double noise_sum = 0.0;
  ssize_t noise_count = 0;
  apply_threshold(s, d, tstar, result, noise_sum, noise_count);
  result.sigma2 = best_sigma2;
  result.lamplus = tstar;
  return result;
}

template class Exp<1>;
template class Exp<2>;

} // namespace MR::Denoise::Estimator
