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

#include <string>

#include "denoise/noise_model/noise_model.h"
#include "denoise/noise_model/noncentralchi.h"
#include "enum.h"

namespace MR::Denoise::NoiseModel {

// Rician noise model: the single-channel (N = 1) special case of the
//   non-central chi distribution.
class Rician : public NonCentralChi {
public:
  explicit Rician(const vst_method_t vst_method) : NonCentralChi(1, vst_method) {}
  std::string description() const final {
    return "Rician (2 DOF; VST: " + Enum::lowercase_name(vst_method) + ")";
  }
};

} // namespace MR::Denoise::NoiseModel
