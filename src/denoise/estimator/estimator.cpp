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

#include "denoise/estimator/estimator.h"

#include "denoise/estimator/base.h"
#include "denoise/estimator/exp.h"
#include "denoise/estimator/import.h"
#include "denoise/estimator/med.h"
#include "denoise/estimator/mrm2022.h"
#include "denoise/estimator/rank.h"

namespace MR::Denoise::Estimator {

using namespace App;

// clang-format off
const Option estimator_option =
    Option("estimator",
           "Select the noise level estimator"
           " (default = Exp2),"
           " either: \n"
           "* Exp1: the original estimator used in Veraart et al. (2016); \n"
           "* Exp2: the improved estimator introduced in Cordero-Grande et al. (2019); \n"
           "* Med: estimate based on the median eigenvalue as in Gavish and Donohue (2014); \n"
           "* MRM2022: the alternative estimator introduced in Olesen et al. (2022). \n"
           "Operation will be bypassed if -noise_in or -fixed_rank are specified")
      + Argument("algorithm").type_choice(estimators);

const OptionGroup estimator_denoise_options =
    OptionGroup("Options relating to signal / noise level estimation for denoising")

    + estimator_option

    + Option("noise_in",
             "import a pre-estimated noise level map for denoising rather than estimating this level from data")
      + Argument("image").type_image_in()

    + Option("fixed_rank",
             "set a fixed input signal rank rather than estimating the noise level from the data")
      + Argument("value").type_integer(1);

std::shared_ptr<Base> make_estimator(Image<float> &vst_noise_in, const bool permit_bypass) {
  auto opt = get_options("estimator");
  if (permit_bypass) {
    auto noise_in = get_options("noise_in");
    auto fixed_rank = get_options("fixed_rank");
    if (!noise_in.empty()) {
      if (!opt.empty())
        throw Exception("Cannot both provide an input noise level image and specify a noise level estimator");
      if (!fixed_rank.empty())
        throw Exception("Cannot both provide an input noise level image and request a fixed signal rank");
      return std::make_shared<Import>(noise_in[0][0], vst_noise_in);
    }
    if (!fixed_rank.empty()) {
      if (!opt.empty())
        throw Exception("Cannot both provide an input signal rank and specify a noise level estimator");
      return std::make_shared<Rank>(fixed_rank[0][0]);
    }
  }
  const estimator_type est = opt.empty() ? estimator_type::EXP2 : estimator_type((int)(opt[0][0]));
  switch (est) {
  case estimator_type::EXP1:
    return std::make_shared<Exp<1>>();
  case estimator_type::EXP2:
    return std::make_shared<Exp<2>>();
  case estimator_type::MED:
    return std::make_shared<Med>();
  case estimator_type::MRM2022:
    return std::make_shared<MRM2022>();
  default:
    assert(false);
  }
  return nullptr;
}

} // namespace MR::Denoise::Estimator
