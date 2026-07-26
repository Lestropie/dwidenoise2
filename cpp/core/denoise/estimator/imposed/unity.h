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

#include "denoise/denoise.h"
#include "denoise/estimator/imposed/imposed.h"

namespace MR::Denoise::Estimator {

// This class assumes that in a prior iteration,
//   a noise level image has been computed,
//   and that image is being used for both variance-stabilising transform
//   and as a noise level estimate
// Where this occurs,
//   the levels for the a priori noise level estimate and the VST are always identical,
//   and so sigma^2 == 1.0 always
class Unity : public ImposedSigma {
public:
  Unity() = default;

protected:
  bool get_sigma_sq(const Eigen::Vector3d & /*pos*/, double &sigma_sq) const override {
    sigma_sq = 1.0;
    return true;
  }
};

} // namespace MR::Denoise::Estimator
