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

class MRM2023 : public Base {
public:
  MRM2023() = default;
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s,     //
                    const ssize_t m,                                  //
                    const ssize_t n,                                  //
                    const ssize_t rp,                                 //
                    const Eigen::Vector3d & /*unused*/) const final { //
    // MRM2023 article suggests that mprime should subtract z
    //   since it refers to "non-zero singular values";
    //   possible that this is the case for all estimators
    assert(s.size() == std::min(m, n));
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t mprime = rank_nonzero(m, n, rp);
    const ssize_t nprime = dimlong_nonzero(m, n, rp);
    const double sigmasq_to_lamplus = Math::pow2(std::sqrt(nprime) + std::sqrt(mprime));
    double clam = s.segment(rz, mprime).sum();
    // Unlike Exp# code,
    //   MRM2023 article uses p to index number of signal components,
    //   and here doing a direct translation of the manuscript content to code

    Result result;
    for (ssize_t p = 0; p < mprime; ++p) {
      const ssize_t i = s.size() - 1 - p;
      const double sigmasq = clam / (static_cast<double>(mprime - p) * static_cast<double>(nprime - p));
      const double lamplus = sigmasq * sigmasq_to_lamplus;
      if (s[i] < lamplus) {
        result.cutoff_p = i + 1;
        result.sigma2 = sigmasq;
        result.lamplus = lamplus / nprime;
        return result;
      }
      clam -= s[i];
    }
    result.cutoff_p = 0;
    result.sigma2 = 0.0;
    result.lamplus = 0.0;
    return result;
  }

  // Partitioned form: the peel-off scan runs on the pooled (normalised) eigenvalues of all
  //   partitions, using the rnz-weighted mean aspect ratio for the Marchenko-Pastur edge and the
  //   aggregate long dimension (sum_p qnz_p) for the finite-size correction (valid because the
  //   partitions are sized to share a common beta). The pooled signal/noise boundary is then
  //   applied to each partition to obtain its own rank. Mirrors the single-PCA recursion: at each
  //   step the largest-remaining candidate is tested against lambda+ = sigma^2 (1 + sqrt(beta))^2,
  //   with sigma^2 = (mean of the remaining normalised eigenvalues) * Neff / (Neff - p) (the
  //   pooled analogue of clam / ((m'-p)(n'-p))); the first step at which it falls below the edge
  //   fixes the boundary.
  Result operator()(const std::vector<eigenvalues_type> &s, //
                    const std::vector<ssize_t> &m,           //
                    const std::vector<ssize_t> &n,           //
                    const std::vector<ssize_t> &rp,          //
                    const Eigen::Vector3d & /*unused*/) const final {
    const std::vector<PartitionDims> d = partition_dims(m, n, rp);
    // Pooled, sorted-ascending normalised nonzero eigenvalues across partitions.
    const std::vector<double> pooled = pool_normalized(s, d);
    Result result;
    const ssize_t mtot = ssize_t(pooled.size()); // total nonzero components ("mprime")
    if (mtot == 0)
      return result;
    const double neff = double(total_qnz(d)); // aggregate long dimension ("nprime")
    const double beta = mean_beta(d);
    const double edge = Math::pow2(1.0 + std::sqrt(beta));
    double clam = 0.0;
    for (const double lam : pooled)
      clam += lam;

    double sigma2_found = std::numeric_limits<double>::signaling_NaN();
    double lamplus_found = std::numeric_limits<double>::signaling_NaN();
    double tstar = std::numeric_limits<double>::signaling_NaN();
    bool found = false;
    for (ssize_t p = 0; p < mtot; ++p) {
      const ssize_t k = mtot - p; // remaining noise candidates (the k smallest)
      const double sigmasq = clam * neff / (double(k) * (neff - double(p)));
      const double lamplus = sigmasq * edge;
      const double lam_largest = pooled[k - 1]; // largest of the remaining candidates
      if (lam_largest < lamplus) {
        sigma2_found = sigmasq;
        lamplus_found = lamplus;
        tstar = lam_largest; // classify lambda <= largest noise eigenvalue as noise
        found = true;
        break;
      }
      clam -= lam_largest; // peel this (it is signal) and continue
    }
    if (!found) {
      // No noise boundary located: every component is treated as signal (mirrors the single-PCA
      //   fall-through, in which cutoff_p / sigma2 / lamplus are all zero).
      result.cutoff_p = 0;
      result.sigma2 = 0.0;
      result.lamplus = 0.0;
      result.cutoff_p_partition.assign(d.size(), 0);
      result.lamplus_partition.assign(d.size(), 0.0);
      return result;
    }
    double noise_sum = 0.0;
    ssize_t noise_count = 0;
    apply_threshold(s, d, tstar, result, noise_sum, noise_count);
    result.sigma2 = sigma2_found;
    result.lamplus = lamplus_found; // Marchenko-Pastur edge (normalised), as in the single-PCA path
    return result;
  }

  bool supports_partitioning() const final { return true; }

  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.93.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final {
    const double lm = std::log(double(m));
    const double ln = std::log(double(n));
    const double lr = std::log(std::max(r, 1.0));
    return std::exp(3.07484 - 1.01543 * lm - 1.16065 * ln + 0.69571 * lr + 0.088552 * lm * ln);
  }
};

} // namespace MR::Denoise::Estimator
