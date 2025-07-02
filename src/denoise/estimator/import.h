/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
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
  Import(const std::string &path) : noise_image(Image<float>::open(path)) {}
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
      result.sigma2 = Math::pow2(interp.value());
    }
    // From this noise level,
    //   estimate the upper bound of the MP distribution and rank of signal
    //   given the ordered list of eigenvalues
    double cumulative_lambda = 0.0;
    double recalc_sigmasq = 0.0;
    for (ssize_t p = 0; p != r; ++p) {
      const double lambda = std::max(s[p], 0.0) / q;
      cumulative_lambda += lambda;
      const double sigma_sq = cumulative_lambda / (p + 1);
      if (sigma_sq < result.sigma2) {
        result.cutoff_p = p;
        result.lamplus = lambda;
        recalc_sigmasq = sigma_sq;
      }
    }
    // TODO It would be nice if the upper bound, lambda_plus,
    //   could be yielded at a higher precision than the discrete eigenvalues,
    //   as optimal shrinkage / optimal thresholding could make use of this precision if available
    return result;
  }

private:
  Image<float> noise_image;
};

} // namespace MR::Denoise::Estimator
