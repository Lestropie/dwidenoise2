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

#include <limits>

namespace MR::Denoise::Estimator {

class Result {
public:
  Result()
      : cutoff_p(-1),
        sigma2(std::numeric_limits<double>::signaling_NaN()),
        lamplus(std::numeric_limits<double>::signaling_NaN()) {}
  operator bool() const { return cutoff_p >= 0 && std::isfinite(sigma2) && std::isfinite(lamplus); }
  bool operator!() const { return !bool(*this); }
  // From dwidenoise code / estimator::Exp :
  //   cutoff_p is the *number of noise components considered to be part of the MP distribution*
  ssize_t cutoff_p;
  double sigma2;
  double lamplus;
};

} // namespace MR::Denoise::Estimator
