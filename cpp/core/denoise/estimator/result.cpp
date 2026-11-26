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
#include <cmath>
#include <limits>

namespace MR::Denoise::Estimator {

Result::Result()
    : cutoff_p(-1),
      sigma2(std::numeric_limits<double>::signaling_NaN()),
      edge(std::numeric_limits<double>::signaling_NaN()),
      lamplus(std::numeric_limits<double>::signaling_NaN()),
      lammax_noise(std::numeric_limits<double>::signaling_NaN()) {}

ssize_t Result::cutoff(const ssize_t partition) const {
  assert(partition < 0 || partition < ssize_t(cutoff_p_partition.size()));
  return partition < 0 ? cutoff_p : cutoff_p_partition[partition];
}

void Result::set_boundaries(const double mp_edge, const double z_max_noise) {
  // An estimator that reaches here without a noise level has mis-ordered its own assignments:
  //   lamplus is formed from sigma2, so a NaN level would silently produce a NaN boundary that
  //   only Result::operator bool() further downstream would catch.
  assert(std::isfinite(sigma2) && sigma2 >= 0.0);
  // The edge is a property of the Casorati shape alone, (1 + sqrt(beta))^2 >= 1, and is available
  //   whether or not a boundary was found; it is never legitimately zero or negative.
  assert(std::isfinite(mp_edge) && mp_edge > 0.0);
  assert(std::isfinite(z_max_noise) && z_max_noise >= 0.0);
  edge = mp_edge;
  lamplus = sigma2 * mp_edge;
  lammax_noise = z_max_noise * mp_edge;
}

double Result::lammax_noise_standardized() const {
  assert(std::isfinite(edge) && edge > 0.0);
  assert(std::isfinite(lammax_noise));
  return lammax_noise / edge;
}

} // namespace MR::Denoise::Estimator
