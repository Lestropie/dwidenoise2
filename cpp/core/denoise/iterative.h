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
#include "denoise/loop.h"
#include "denoise/precondition.h"
#include "denoise/subsample.h"
#include "filter/smooth.h"
#include "image.h"
#include "interp/cubic.h"
#include "thread_queue.h"
#include "types.h"

namespace MR::Denoise::Iterative {

struct Iteration {
  std::array<ssize_t, 3> subsample_ratios;
  default_type kernel_size_multiplier;
  noise_smooth_type smooth_noiseout;
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
              const decomp_type decomposition,
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
  {
    Denoise::Sender sender(input_preconditioned, subsample);
    Estimate<T> func(input_preconditioned, kernel, decomposition, estimator, preconditioner.null_rank(), false);
    Denoise::ReceiverEstimate receiver(subsample, exports);
    Thread::run_queue(sender,
                      Kernel::Voxel::index_type(),
                      Thread::multi(func),
                      EstimatedPatch(),
                      receiver);
  }
  // If a VST was applied to the input data for this iteration,
  //   need to remove its effect from the estimated noise map
  if (vst_image.valid()) {
    Interp::Cubic<Image<float>> vst_interp(vst_image);
    const Transform transform(subsample->header());
    for (auto l = Loop(exports.noise_out)(exports.noise_out); l; ++l) {
      vst_interp.scanner(transform.voxel2scanner * Eigen::Vector3d({double(exports.noise_out.index(0)),
                                                                    double(exports.noise_out.index(1)),
                                                                    double(exports.noise_out.index(2))}));
      exports.noise_out.value() *= vst_interp.value();
    }
  }
}

} // namespace MR::Denoise::Iterative
