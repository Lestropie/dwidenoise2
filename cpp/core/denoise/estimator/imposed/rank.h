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
#include <cassert>
#include <limits>
#include <vector>

#include "denoise/estimator/base.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"
#include "exception.h"
#include "mrtrix.h"

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
    // The imposed rank counts the assumed-zero singular values "rz" that preconditioning ("rp"
    //   regressed means) forces to zero, as the arithmetic below (rank - rz) requires. The
    //   maximal attainable rank is therefore rz + rnz == min(m, n) == s.size(), NOT rnz: a patch
    //   whose demeaning removed rz directions still has min(m, n) components in total, rz of
    //   which are identically zero. Comparing against rnz alone understates the maximum by rz,
    //   which both mis-classified attainable ranks as no-ops and let unattainable ones fall
    //   through to a default-constructed Result.
    const ssize_t max_rank = rz + rnz;
    assert(max_rank == s.size());
    if (rank > max_rank)
      throw Exception("Imposed signal rank of " + str(rank) + " cannot be attained: this patch has " + str(m) +
                      " volumes and " + str(n) + " voxels, giving at most " + str(max_rank) + " components (of which " +
                      str(rz) +
                      " are forced to zero by the demeaning of the preconditioner). "
                      "Reduce -fixed_rank to " +
                      str(max_rank) + " or fewer");
    if (rank < rz)
      throw Exception("Imposed signal rank of " + str(rank) + " is below the " + str(rz) +
                      " component(s) that the preconditioner's demeaning has already forced to zero "
                      "in this patch, and which necessarily count towards the signal rank. "
                      "Increase -fixed_rank to at least " +
                      str(rz));
    Result result;
    if (rank == max_rank) {
      // Every component is signal, so there is no noise boundary to report. sigma2 == 0 is the
      //   established encoding of "no usable threshold" (see Result::operator bool), on which
      //   Estimate and Recon take their documented conservative path of copying the patch through
      //   unchanged -- which is exactly the no-op this case calls for. The Result is explicitly
      //   populated rather than left default-constructed.
      result.cutoff_p = 0;
      result.lamplus = 0.0;
      result.sigma2 = 0.0;
      return result;
    }
    result.cutoff_p = s.size() - (rank - rz);
    result.sigma2 = s.segment(rz, result.cutoff_p - rz).sum() / (qnz * (result.cutoff_p + 1 - rz));
    result.lamplus = s[result.cutoff_p - 1] / qnz;
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
    // As in the single-PCA form: an unattainable request is a fault to report, not a Result to
    //   leave default-constructed and pass on to callers that would read it as "no threshold".
    if (total_nonzero == 0)
      throw Exception("Imposed signal rank of " + str(rank) +
                      " cannot be attained: preconditioning has forced every component of this "
                      "partitioned patch to zero, leaving no components to classify");
    if (signal_nonzero < 0)
      throw Exception("Imposed signal rank of " + str(rank) + " is below the " + str(total_rz) +
                      " component(s) that the preconditioner's demeaning has already forced to zero "
                      "across this patch's partitions, and which necessarily count towards the "
                      "signal rank. Increase -fixed_rank to at least " +
                      str(total_rz));
    if (signal_nonzero > total_nonzero)
      throw Exception("Imposed signal rank of " + str(rank) +
                      " cannot be attained: this patch's partitions "
                      "provide at most " +
                      str(total_rz + total_nonzero) + " components (of which " + str(total_rz) +
                      " are forced to zero by the demeaning of the preconditioner). "
                      "Reduce -fixed_rank to " +
                      str(total_rz + total_nonzero) + " or fewer");
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

protected:
  const ssize_t rank;
};

} // namespace MR::Denoise::Estimator
