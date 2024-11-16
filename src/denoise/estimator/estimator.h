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

#include <memory>
#include <string>
#include <vector>

#include "app.h"

namespace MR::Denoise::Estimator {

class Base;

extern const App::Option option;
const std::vector<std::string> estimators = {"exp1", "exp2", "mrm2022"};
enum class estimator_type { EXP1, EXP2, MRM2022 };
std::shared_ptr<Base> make_estimator();

} // namespace MR::Denoise::Estimator
