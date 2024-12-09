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

#include <atomic>

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

template <ssize_t version> class Exp : public Base {
public:
  Exp() : failure_count(0) {}
  ~Exp() {
    const ssize_t total = failure_count.load();
    if (total > 0) {
      WARN("Noise level estimator failed to converge for " + str(total) + " patches");
    }
  }
  Result operator()(const eigenvalues_type &s,
                    const ssize_t m,
                    const ssize_t n,
                    const Eigen::Vector3d & /*unused*/) const final;

protected:
  mutable std::atomic<ssize_t> failure_count;
};

} // namespace MR::Denoise::Estimator
