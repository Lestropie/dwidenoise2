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

#include "denoise/estimator/result.h"

#include <cassert>
#include <limits>

namespace MR::Denoise::Estimator {

Result::Result()
    : cutoff_p(-1),
      sigma2(std::numeric_limits<double>::signaling_NaN()),
      lamplus(std::numeric_limits<double>::signaling_NaN()) {}

ssize_t Result::cutoff(const ssize_t partition) const {
  assert(partition < 0 || partition < ssize_t(cutoff_p_partition.size()));
  return partition < 0 ? cutoff_p : cutoff_p_partition[partition];
}

} // namespace MR::Denoise::Estimator
