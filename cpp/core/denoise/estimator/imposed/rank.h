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
#include <limits>
#include <vector>

#include "denoise/estimator/base.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

class Rank : public Base {
public:
  Rank(const ssize_t r) : rank(r) {}
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s,     //
                    const ssize_t m,                                  //
                    const ssize_t n,                                  //
                    const ssize_t rp,                                 //
                    const Eigen::Vector3d & /*unused*/) const final { //
    assert(s.size() == std::min(m, n));
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t rnz = rank_nonzero(m, n, rp);
    const ssize_t qnz = dimlong_nonzero(m, n, rp);
    Result result;
    // Bear in mind that any assumed-zero singular values "rz" due to preconditioning "rp"
    //   must be assumed to contribute to the rank
    if (rnz == rank) {
      // All components contribute (even the assumed-zero ones)
      result.cutoff_p = 0;
      result.lamplus = 0.0;
      result.sigma2 = 0.0;
    } else if (rnz > rank) {
      result.cutoff_p = s.size() - (rank - rz);
      result.sigma2 = s.segment(rz, result.cutoff_p - rz).sum() / (qnz * (result.cutoff_p + 1 - rz));
      result.lamplus = s[result.cutoff_p - 1] / qnz;
    } // If requested rank is greater than available rank, leave "result" completely uninitialised
    return result;
  }

  // Partitioned form: the imposed (total) signal rank is honoured across the whole patch by
  //   choosing the corresponding ordered element of the pooled (normalised) eigenspectrum as the
  //   signal/noise boundary; the per-partition rank then follows from each partition's own
  //   eigenvalues (a partition may receive rank 0). Mirrors the single-PCA convention in which
  //   the imposed rank counts the assumed-zero (preconditioner) components.
  Result operator()(const std::vector<eigenvalues_type> &s, //
                    const std::vector<ssize_t> &m,           //
                    const std::vector<ssize_t> &n,           //
                    const std::vector<ssize_t> &rp,          //
                    const Eigen::Vector3d & /*unused*/) const final {
    const std::vector<PartitionDims> d = partition_dims(m, n, rp);
    ssize_t total_rz = 0;
    for (const auto &x : d)
      total_rz += x.rz;
    const ssize_t total_nonzero = total_rnz(d);
    Result result;
    // Number of genuine (nonzero) signal components to retain across the patch.
    const ssize_t signal_nonzero = rank - total_rz;
    if (signal_nonzero < 0 || signal_nonzero > total_nonzero || total_nonzero == 0)
      return result; // requested rank unattainable: leave invalid
    if (signal_nonzero == total_nonzero) {
      // Keep all nonzero components as signal.
      result.cutoff_p = 0;
      result.sigma2 = 0.0;
      result.lamplus = 0.0;
      result.cutoff_p_partition.assign(d.size(), 0);
      result.lamplus_partition.assign(d.size(), 0.0);
      return result;
    }
    const std::vector<double> pooled = pool_normalized(s, d); // ascending, length total_nonzero
    const ssize_t noise_nonzero = total_nonzero - signal_nonzero; // > 0 here
    // Largest retained-as-noise normalised eigenvalue defines the boundary.
    const double tstar = pooled[noise_nonzero - 1];
    double noise_sum = 0.0;
    ssize_t noise_count = 0;
    apply_threshold(s, d, tstar, result, noise_sum, noise_count);
    result.sigma2 = noise_sum / double(noise_count + 1);
    result.lamplus = tstar;
    return result;
  }

  bool supports_partitioning() const final { return true; }

protected:
  const ssize_t rank;
};

} // namespace MR::Denoise::Estimator
