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

namespace MR::Denoise::Estimator {

using namespace App;

const Option option = Option("estimator",
                             "Select the noise level estimator"
                             " (default = Exp2),"
                             " either: \n"
                             "* Exp1: the original estimator used in Veraart et al. (2016); \n"
                             "* Exp2: the improved estimator introduced in Cordero-Grande et al. (2019); \n"
                             "* Import: import from a pre-estimated noise level map; \n"
                             "* Med: estimate based on the median eigenvalue as in Gavish and Donohue (2014); \n"
                             "* MRM2022: the alternative estimator introduced in Olesen et al. (2022).") +
                      Argument("algorithm").type_choice(estimators);

std::shared_ptr<Base> make_estimator() {
  auto opt = App::get_options("estimator");
  const estimator_type est = opt.empty() ? estimator_type::EXP2 : estimator_type((int)(opt[0][0]));
  auto noise_in = get_options("noise_in");
  // TODO Remove once -noise_in is used in other ways
  if (!noise_in.empty() && est != estimator_type::IMPORT) {
    WARN("-noise_in option has no effect unless -estimator import is specified");
  }
  switch (est) {
  case estimator_type::EXP1:
    return std::make_shared<Exp<1>>();
  case estimator_type::EXP2:
    return std::make_shared<Exp<2>>();
  case estimator_type::IMPORT:
    if (noise_in.empty())
      throw Exception("-estimator import requires a pre-estimated noise level image "
                      "to be provided via the -noise_in option");
    return std::make_shared<Import>(noise_in[0][0]);
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
