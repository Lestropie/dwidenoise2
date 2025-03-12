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

#include "denoise/kernel/data.h"
#include "denoise/kernel/sphere_base.h"
#include "header.h"
#include "types.h"

namespace MR::Denoise::Kernel {

class SphereFixedRadius : public SphereBase {
public:
  SphereFixedRadius(const Header &voxel_grid,                         //
                    const std::array<ssize_t, 3> &subsample_factors,  //
                    const default_type radius)                        //
      : SphereBase(voxel_grid, subsample_factors, radius),            //
        maximum_size(std::distance(shared->begin(), shared->end())) { //
    INFO("Maximum number of voxels in " + str(radius) + "mm fixed-radius kernel is " + str(maximum_size));
  }
  SphereFixedRadius(const SphereFixedRadius &) = default;
  ~SphereFixedRadius() override = default;
  Data operator()(const Voxel::index_type &pos) const override;
  ssize_t estimated_size() const override { return maximum_size; }

private:
  const ssize_t maximum_size;
};

} // namespace MR::Denoise::Kernel
