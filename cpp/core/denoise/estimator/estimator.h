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

#include <memory>
#include <string>
#include <vector>

#include "app.h"
#include "image.h"

namespace MR::Denoise::Estimator {

class Base;

extern const App::Option estimator_option;
extern const App::OptionGroup estimator_denoise_options;
enum class estimator_type { EXP1, EXP2, MED, MRM2023, TBME2022 };
// Default data-driven estimator, used both to populate the -estimator help text
//   and as the fallback when the option is not specified. MRM2023 is preferred as the
//   default because it reaches an acceptable noise-level estimate precision with fewer
//   kernel voxels than the alternatives (its predicted-RMSE model decays faster in the
//   voxel count n), so the rank-adaptive / RMSE-tolerance kernels can stay smaller.
constexpr estimator_type default_estimator = estimator_type::MRM2023;
std::shared_ptr<Base> make_estimator(Image<float> &vst_noise_in, const bool permit_bypass);

} // namespace MR::Denoise::Estimator
