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

#include "denoise/estimator/imposed/fixed.h"
#include "denoise/estimator/imposed/import.h"
#include "denoise/estimator/imposed/rank.h"

#include "app.h"
#include "exception.h"

namespace MR::Denoise::Estimator {

using namespace App;

std::shared_ptr<Base> make_imposed(Image<float> &vst_noise_in) {
  auto noise_in = get_options("noise_in");
  auto fixed_rank = get_options("fixed_rank");
  // Neither bypass option supplied: defer to data-driven estimation.
  if (noise_in.empty() && fixed_rank.empty())
    return nullptr;
  auto opt = get_options("estimator");
  if (!noise_in.empty()) {
    if (!opt.empty())
      throw Exception("Cannot both provide an input noise level image and specify a noise level estimator");
    if (!fixed_rank.empty())
      throw Exception("Cannot both provide an input noise level image and request a fixed signal rank");
    // -noise_in may be either a scalar value (Fixed) or an image path (Import);
    //   attempt the scalar interpretation first and fall back to opening an image.
    try {
      return std::make_shared<Fixed>(default_type(noise_in[0][0]), vst_noise_in);
    } catch (Exception &) {
      return std::make_shared<Import>(std::string(noise_in[0][0]), vst_noise_in);
    }
  }
  // -fixed_rank only
  if (!opt.empty())
    throw Exception("Cannot both provide an input signal rank and specify a noise level estimator");
  return std::make_shared<Rank>(fixed_rank[0][0]);
}

} // namespace MR::Denoise::Estimator
