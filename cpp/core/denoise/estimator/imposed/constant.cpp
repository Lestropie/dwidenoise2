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

#include "denoise/estimator/imposed/constant.h"

#include "interp/cubic.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

Constant::Constant(const default_type value, Image<float> &vst_noise_in) //
    : sigma2(Math::pow2(value)),                                         //
      vst_image(vst_noise_in),                                           //
      rescale_by_vst(true) {}                                            //

Constant::Constant(const default_type value) //
    : sigma2(Math::pow2(value)),             //
      rescale_by_vst(false) {}               //

bool Constant::get_sigma_sq(const Eigen::Vector3d &pos, double &sigma_sq) const {
  // If the data have been preconditioned at input based on a pre-estimated noise level,
  //   then we need to rescale the threshold that we load from this image
  //   based on knowledge of that rescaling
  if (vst_image.valid()) {
    Interp::Cubic<Image<float>> vst_interp(vst_image);
    if (!vst_interp.scanner(pos))
      return false;
    sigma_sq = sigma2 / Math::pow2(vst_interp.value());
  } else {
    sigma_sq = sigma2;
  }
  return true;
}

} // namespace MR::Denoise::Estimator
