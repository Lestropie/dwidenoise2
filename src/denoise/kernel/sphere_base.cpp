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

#include "denoise/kernel/sphere_base.h"

#include "math/math.h"

namespace MR::Denoise::Kernel {

SphereBase::Shared::Shared(const Header &voxel_grid, const default_type max_radius) {
  const default_type max_radius_sq = Math::pow2(max_radius);
  const Voxel::index_type half_extents({ssize_t(std::ceil(max_radius / voxel_grid.spacing(0))),   //
                                        ssize_t(std::ceil(max_radius / voxel_grid.spacing(1))),   //
                                        ssize_t(std::ceil(max_radius / voxel_grid.spacing(2)))}); //
  // Build the searchlight
  data.reserve(size_t(2 * half_extents[0] + 1) * size_t(2 * half_extents[1] + 1) * size_t(2 * half_extents[2] + 1));
  Offset::index_type offset({0, 0, 0});
  for (offset[2] = -half_extents[2]; offset[2] <= half_extents[2]; ++offset[2]) {
    for (offset[1] = -half_extents[1]; offset[1] <= half_extents[1]; ++offset[1]) {
      for (offset[0] = -half_extents[0]; offset[0] <= half_extents[0]; ++offset[0]) {
        const default_type squared_distance = Math::pow2(offset[0] * voxel_grid.spacing(0))    //
                                              + Math::pow2(offset[1] * voxel_grid.spacing(1))  //
                                              + Math::pow2(offset[2] * voxel_grid.spacing(2)); //
        if (squared_distance <= max_radius_sq)
          data.emplace_back(Offset(offset, squared_distance));
      }
    }
  }
  std::sort(data.begin(), data.end());
}

} // namespace MR::Denoise::Kernel
