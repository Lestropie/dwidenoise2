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

namespace MR::Denoise::Estimator {

// Impose a noise level imported from a pre-estimated 3D noise level image.
class Import : public ImposedSigma {
public:
  Import(const std::string &path, Image<float> &vst_image);
  // Impose an already-conditioned noise level map directly on unmodified data. Used when
  //   -vst_method none leaves the data in their native scale: there is then no stabilising
  //   transform whose parameterisation could absorb the supplied noise level, so the level must
  //   be applied to the Casorati matrix as-is. Because no rescaling is in play, this form also
  //   ignores any subsequently-supplied VST image; dividing the noise map by itself would
  //   collapse sigma to unity.
  explicit Import(const Image<float> &conditioned_noise_image);
  void update_vst_image(Image<float> &new_vst_image) override {
    if (rescale_by_vst)
      vst_image = new_vst_image;
  }

protected:
  bool get_sigma_sq(const Eigen::Vector3d &pos, double &sigma_sq) const override;

private:
  Image<float> noise_image;
  Image<float> vst_image;
  bool rescale_by_vst;
};

} // namespace MR::Denoise::Estimator
