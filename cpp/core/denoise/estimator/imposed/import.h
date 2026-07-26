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

#include <string>

#include "denoise/denoise.h"
#include "denoise/estimator/imposed/imposed.h"

#include "image.h"
#include "interp/cubic.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

// Impose a noise level imported from a pre-estimated 3D noise level image.
class Import : public ImposedSigma {
public:
  Import(const std::string &path, Image<float> &vst_image) //
      : noise_image(Image<float>::open(path)),             //
        vst_image(vst_image) {}                            //
  void update_vst_image(Image<float> &new_vst_image) override { vst_image = new_vst_image; }

protected:
  bool get_sigma_sq(const Eigen::Vector3d &pos, double &sigma_sq) const override {
    // Construct on each call to preserve const-ness & thread-safety
    Interp::Cubic<Image<float>> interp(noise_image);
    // TODO This will cause issues at the edge of the image FoV
    // Addressing this may require integration of the mrfilter changes
    //   that provide wrappers for various handling of FoV edges
    // For now, just expect that denoising won't do anything
    //   where the patch centre is too close to the image edge for cubic interpolation
    if (!interp.scanner(pos))
      return false;
    // If the data have been preconditioned at input based on a pre-estimated noise level,
    //   then we need to rescale the threshold that we load from this image
    //   based on knowledge of that rescaling
    if (vst_image.valid()) {
      Interp::Cubic<Image<float>> vst_interp(vst_image);
      if (!vst_interp.scanner(pos))
        return false;
      sigma_sq = Math::pow2(interp.value() / vst_interp.value());
    } else {
      sigma_sq = Math::pow2(interp.value());
    }
    return true;
  }

private:
  Image<float> noise_image;
  Image<float> vst_image;
};

} // namespace MR::Denoise::Estimator
