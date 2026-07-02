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

#include "denoise/precondition/noise_model/noise_model.h"

#include <cassert>

#include "denoise/precondition/noise_model/gaussian.h"
#include "denoise/precondition/noise_model/identity.h"
#include "denoise/precondition/noise_model/noncentralchi.h"
#include "denoise/precondition/noise_model/rician.h"

namespace MR::Denoise::Precondition::NoiseModel {

std::shared_ptr<Base> make(const distribution_t distribution, //
                           const ssize_t num_channels,        //
                           const vst_method_t vst_method) {   //
  // -vst_method none / linear select the transform directly, independent of the
  //   underlying noise distribution:
  //   - none   : identity (no variance stabilisation);
  //   - linear : the simple linear scaling u = m / sigma (the Gaussian model),
  //              exact for additive Gaussian (complex / phase-demodulated) noise.
  switch (vst_method) {
  case vst_method_t::NONE:
    return std::make_shared<Identity>();
  case vst_method_t::LINEAR:
    return std::make_shared<Gaussian>();
  default:
    break;
  }
  switch (distribution) {
  case distribution_t::GAUSSIAN:
    return std::make_shared<Gaussian>();
  case distribution_t::RICIAN:
    return std::make_shared<Rician>(vst_method);
  case distribution_t::NONCENTRALCHI:
    return std::make_shared<NonCentralChi>(num_channels, vst_method);
  }
  assert(false);
  return nullptr;
}

} // namespace MR::Denoise::Precondition::NoiseModel
