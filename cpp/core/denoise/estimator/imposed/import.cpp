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

#include "denoise/estimator/imposed/import.h"

#include <string>

#include "denoise/denoise.h"
#include "interp/cubic.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

Import::Import(const std::string &path, Image<float> &vst_image)                            //
    : noise_image(Image<float>::open(path)),                                               //
      noise_log_image(log_noise_map(noise_image)),                                         //
      vst_image(vst_image),                                                                //
      vst_log_image(vst_image.valid() ? log_noise_map(vst_image) : Image<float>()),        //
      rescale_by_vst(true) {}                                                              //

Import::Import(const Image<float> &conditioned_noise_image)                                //
    : noise_image(conditioned_noise_image),                                                //
      noise_log_image(log_noise_map(noise_image)),                                         //
      rescale_by_vst(false) {}                                                             //

double Import::get_sigma_sq(const Eigen::Vector3d &pos) const {
  // Construct on each call to preserve const-ness & thread-safety.
  // Both maps are sampled through their log-domain twins: cubic interpolation of sigma undershoots
  //   at the edge of the object, and the squaring below would launder a negative interpolated value
  //   into a plausible-looking positive sigma^2.
  LogNoiseInterp interp(noise_log_image);
  // TODO This will cause issues at the edge of the image FoV
  // Addressing this may require integration of the mrfilter changes
  //   that provide wrappers for various handling of FoV edges
  // For now, just expect that denoising won't do anything
  //   where the patch centre is too close to the image edge for cubic interpolation
  if (!interp.scanner(pos))
    return std::numeric_limits<double>::quiet_NaN();
  // If the data have been preconditioned at input based on a pre-estimated noise level,
  //   then we need to rescale the threshold that we load from this image
  //   based on knowledge of that rescaling
  if (vst_image.valid()) {
    LogNoiseInterp vst_interp(vst_log_image);
    if (!vst_interp.scanner(pos))
      return std::numeric_limits<double>::quiet_NaN();
    return Math::pow2(interp.value() / vst_interp.value());
  }
  return Math::pow2(interp.value());
}

} // namespace MR::Denoise::Estimator
