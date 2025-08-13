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

#include <array>
#include <vector>

#include "algo/threaded_loop.h"
#include "denoise/estimate.h"
#include "denoise/estimator/estimator.h"
#include "denoise/exports.h"
#include "denoise/kernel/kernel.h"
#include "denoise/precondition.h"
#include "denoise/subsample.h"
#include "filter/smooth.h"
#include "image.h"
#include "interp/linear.h"
#include "types.h"

namespace MR::Denoise::Iterative {

struct Iteration {
  std::array<ssize_t, 3> subsample_ratios;
  default_type kernel_size_multiplier;
  bool smooth_noiseout;
};

// Internal function covering as much as possible for iterative implementation
template <typename T>
void estimate(Image<T> &input,
              Image<T> &input_preconditioned,
              Image<bool> &mask,
              Image<float> &vst_image,
              Image<float> &rank_per_mm_image,
              const Iteration &config,
              const ssize_t iter,
              std::shared_ptr<Subsample> subsample,
              std::shared_ptr<Estimator::Base> estimator,
              const Precondition<T> &preconditioner,
              Exports &exports) {
  auto kernel = Kernel::make_kernel(input,
                                    subsample->get_factors(),
                                    config.kernel_size_multiplier,
                                    rank_per_mm_image);
  kernel->set_mask(mask);
  if (preconditioner.noop())
    input_preconditioned = input;
  else
    preconditioner(input, input_preconditioned, false);
  Estimate<T> func(input_preconditioned, subsample, kernel, estimator, exports, preconditioner.null_rank(), false);
  ThreadedLoop("MPPCA noise level estimation", input_preconditioned, 0, 3).run(func, input_preconditioned);
  // If a VST was applied to the input data for this iteration,
  //   need to remove its effect from the estimated noise map
  if (vst_image.valid()) {
    Interp::Linear<Image<float>> vst_interp(vst_image);
    const Transform transform(subsample->header());
    for (auto l = Loop(exports.noise_out)(exports.noise_out); l; ++l) {
      vst_interp.scanner(transform.voxel2scanner * Eigen::Vector3d({double(exports.noise_out.index(0)),
                                                                    double(exports.noise_out.index(1)),
                                                                    double(exports.noise_out.index(2))}));
      exports.noise_out.value() *= vst_interp.value();
    }
  }
  // TODO Ideally include more complex processing on this image;
  //   eg. outlier detection & infilling
  if (config.smooth_noiseout) {
    assert(exports.noise_out.valid());
    Filter::Smooth smooth_filter(exports.noise_out);
    smooth_filter(exports.noise_out);
  }
}

} // namespace MR::Denoise::Iterative
