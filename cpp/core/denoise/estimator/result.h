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

#include <limits>
#include <vector>

namespace MR::Denoise::Estimator {

class Result {
public:
  Result()
      : cutoff_p(-1),
        sigma2(std::numeric_limits<double>::signaling_NaN()),
        lamplus(std::numeric_limits<double>::signaling_NaN()) {}
  // A noise level of zero is not a usable threshold: lambda+ = (1 + sqrt(beta))^2 * sigma^2 is then
  //   zero, so every component is classified as signal, and the optimal-shrinkage weights divide by
  //   sigma2 (yielding NaN weights, and hence NaN output data in a build without assertions).
  //   Estimators signal "no boundary could be found" by returning sigma2 == 0 (MRM2023's
  //   fall-through when the peel-off scan classifies every component as signal;
  //   Estimator::Rank when the imposed rank equals the available rank), so a strictly positive
  //   sigma2 is required here. Recon and Estimate then take their documented conservative path
  //   for an unusable threshold: copy the input data through unchanged.
  operator bool() const { return cutoff_p >= 0 && std::isfinite(sigma2) && sigma2 > 0.0 && std::isfinite(lamplus); }
  bool operator!() const { return !bool(*this); }
  // From dwidenoise code / estimator::Exp :
  //   cutoff_p is the *number of noise components considered to be part of the MP distribution*.
  // For the partitioned path cutoff_p carries the pooled total noise count across partitions;
  //   sigma2 is the single noise level shared by the patch; and lamplus is a representative
  //   (normalised) signal/noise boundary.
  ssize_t cutoff_p;
  double sigma2;
  double lamplus;
  // Partitioned results (empty for the single-PCA path; otherwise length P). With the patch
  //   split into P partitions, a single sigma2 applies but the signal rank may differ per
  //   partition: cutoff_p_partition[p] is the number of noise components in partition p (its
  //   smallest cutoff_p_partition[p] eigenvalues, ascending, are noise; a partition may be all
  //   noise, i.e. rank 0), and lamplus_partition[p] is that partition's (normalised) boundary.
  std::vector<ssize_t> cutoff_p_partition;
  std::vector<double> lamplus_partition;
};

} // namespace MR::Denoise::Estimator
