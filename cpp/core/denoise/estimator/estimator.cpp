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
#include "denoise/estimator/datadriven/exp.h"
#include "denoise/estimator/datadriven/med.h"
#include "denoise/estimator/datadriven/mrm2023.h"
#include "denoise/estimator/datadriven/tbme2022.h"
#include "denoise/estimator/imposed/imposed.h"

namespace MR::Denoise::Estimator {

using namespace App;

// clang-format off
const Option estimator_option =
    Option("estimator",
           "Select the noise level estimator (default = " + Enum::lowercase_name(default_estimator) +
               "), either: \n"
               "* Exp1: the original estimator used in Veraart et al. (2016); \n"
               "* Exp2: the improved estimator introduced in Cordero-Grande et al. (2019); \n"
               "* Med: estimate based on the median eigenvalue as in Gavish and Donohue (2014); \n"
               "* MRM2023: the alternative estimator introduced in Olesen et al. (2023); \n"
               "* TBME2022: the multiple-moment generalised-quarter-circle estimator of Zhu et al. (2022).")
      + Argument("algorithm").type_choice<estimator_type>();

const OptionGroup estimator_denoise_options =
    OptionGroup("Options relating to signal / noise level estimation for denoising")

    + estimator_option

    + Option("noise_in",
             "import a pre-estimated noise level, either as a scalar value or as a 3D image, "
             "to be used directly rather than estimated from the data; "
             "this bypasses noise level estimation and also parameterises the variance-stabilising transform")
      + Argument("value/image").type_float(0.0).type_image_in()

    + Option("fixed_rank",
             "impose a fixed input signal rank rather than estimating the noise level from the data. "
             "This selects the fixed-rank estimator (mutually exclusive with -estimator) and performs "
             "a single denoising pass with a spherical kernel sized to n = m + r voxels "
             "(the bundled \"fixedrank\" schedule); a fixed-rank noise level is not robust enough to "
             "drive an iterative variance-stabilising transform, so no multi-resolution refinement is "
             "performed")
      + Argument("value").type_integer(1);

std::shared_ptr<Base> make_estimator(Image<float> &vst_noise_in, const bool permit_bypass) {
  // The -noise_in / -fixed_rank options bypass data-driven estimation, imposing an
  //   externally-determined noise level or signal rank instead (see imposed/imposed.h).
  if (permit_bypass) {
    auto imposed = make_imposed(vst_noise_in);
    if (imposed)
      return imposed;
  }
  const estimator_type est = get_option_choice("estimator", default_estimator);
  switch (est) {
  case estimator_type::EXP1:
    return std::make_shared<Exp<1>>();
  case estimator_type::EXP2:
    return std::make_shared<Exp<2>>();
  case estimator_type::MED:
    return std::make_shared<Med>();
  case estimator_type::MRM2023:
    return std::make_shared<MRM2023>();
  case estimator_type::TBME2022:
    return std::make_shared<TBME2022>();
  default:
    assert(false);
  }
  return nullptr;
}

} // namespace MR::Denoise::Estimator
