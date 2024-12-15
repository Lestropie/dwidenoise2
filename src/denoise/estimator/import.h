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

#pragma once

#include <string>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "image.h"
#include "interp/cubic.h"

namespace MR::Denoise::Estimator {

class Import : public Base {
public:
  Import(const std::string &path, Image<float> &vst_noise_in) //
      : noise_image(Image<float>::open(path)),                //
        vst_noise_image(vst_noise_in) {}                      //
  Result operator()(const eigenvalues_type &s,                //
                    const ssize_t m,                          //
                    const ssize_t n,                          //
                    const Eigen::Vector3d &pos) const final { //
    Result result;
    const ssize_t r = std::min(m, n);
    const ssize_t q = std::max(m, n);
    {
      // Construct on each call to preserve const-ness & thread-safety
      Interp::Cubic<Image<float>> interp(noise_image);
      // TODO This will cause issues at the edge of the image FoV
      // Addressing this may require integration of the mrfilter changes
      //   that provide wrappers for various handling of FoV edges
      // For now, just expect that denoising won't do anything
      //   where the patch centre is too close to the image edge for cubic interpolation
      if (!interp.scanner(pos))
        return result;
      // If the data have been preconditioned at input based on a pre-estimated noise level,
      //   then we need to rescale the threshold that we load from this image
      //   based on knowledge of that rescaling
      if (vst_noise_image.valid()) {
        Interp::Cubic<Image<float>> vst_interp(vst_noise_image);
        if (!vst_interp.scanner(pos))
          return result;
        result.sigma2 = Math::pow2(interp.value() / vst_interp.value());
      } else {
        result.sigma2 = Math::pow2(interp.value());
      }
    }
    // From this noise level,
    //   estimate the upper bound of the MP distribution and rank of signal
    //   given the ordered list of eigenvalues
    double cumulative_lambda = 0.0;
    for (ssize_t p = 0; p != r; ++p) {
      const double lambda = std::max(s[p], 0.0) / q;
      cumulative_lambda += lambda;
      const double sigma_sq = cumulative_lambda / (p + 1);
      if (sigma_sq < result.sigma2) {
        result.cutoff_p = p;
        result.lamplus = lambda;
      }
    }
    // TODO It would be nice if the upper bound, lambda_plus,
    //   could be yielded at a higher precision than the discrete eigenvalues,
    //   as optimal shrinkage / optimal thresholding could make use of this precision if available
    return result;
  }

private:
  Image<float> noise_image;
  Image<float> vst_noise_image;
};

} // namespace MR::Denoise::Estimator
