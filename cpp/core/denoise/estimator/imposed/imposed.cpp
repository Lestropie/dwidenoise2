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

#include "denoise/estimator/imposed/imposed.h"

#include <memory>
#include <string>

#include "denoise/estimator/imposed/rank.h"

#include "app.h"
#include "exception.h"

namespace MR::Denoise::Estimator {

using namespace App;

// Only -fixed_rank bypasses data-driven estimation (imposing a fixed signal rank).
//   -noise_in no longer bypasses estimation: it merely seeds the variance-stabilising
//   transform (the first iteration's noise level), after which the schedule's data-driven
//   estimator refines the estimate. The "apply a known noise map without re-estimating"
//   behaviour is instead obtained via a single-row schedule with update_noise = false.
std::shared_ptr<Base> make_imposed(Image<float> & /*vst_noise_in*/) {
  auto fixed_rank = get_options("fixed_rank");
  if (fixed_rank.empty())
    return nullptr;
  if (!get_options("estimator").empty())
    throw Exception("Cannot both provide an input signal rank and specify a noise level estimator");
  return std::make_shared<Rank>(fixed_rank[0][0]);
}

} // namespace MR::Denoise::Estimator
