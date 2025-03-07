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

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

class Base {
public:
  Base() = default;
  Base(const Base &) = delete;
  virtual void update_vst_image(Image<float> &) {}
  // m = Number of image volumes;
  // n = Number of voxels in patch;
  // rp = Preconditioner rank = number of means regressed from the data;
  // pos = realspace position of the centre of the patch
  virtual Result operator()(const Eigen::VectorBlock<eigenvalues_type> eigenvalues, //
                            const ssize_t m,                                        //
                            const ssize_t n,                                        //
                            const ssize_t rp,                                       //
                            const Eigen::Vector3d &pos) const = 0;                  //
};

} // namespace MR::Denoise::Estimator
