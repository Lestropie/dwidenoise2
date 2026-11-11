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

#include "denoise/denoise.h"
#include "denoise/estimator/imposed/imposed.h"

#include "image.h"

namespace MR::Denoise::Estimator {

// Impose a spatially-constant noise level provided as a scalar value. The counterpart of
//   Import, which imposes a spatially-varying level read from a map; Unity is the special
//   case of this with sigma == 1.
class Constant : public ImposedSigma {
public:
  Constant(const default_type value, Image<float> &vst_noise_in);
  // Impose the level directly on unmodified data, for -vst_method none: there is then no
  //   stabilising transform whose parameterisation could have absorbed it, and no rescaling to
  //   undo. Mirrors the equivalent Import constructor.
  explicit Constant(const default_type value);
  void update_vst_image(Image<float> &new_vst_image) override {
    if (rescale_by_vst)
      vst_image = new_vst_image;
  }

protected:
  bool get_sigma_sq(const Eigen::Vector3d &pos, double &sigma_sq) const override;

private:
  const default_type sigma2;
  Image<float> vst_image;
  bool rescale_by_vst;
};

} // namespace MR::Denoise::Estimator
