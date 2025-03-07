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

#include "denoise/kernel/sphere_ratio.h"

namespace MR::Denoise::Kernel {

Data SphereRatio::operator()(const Voxel::index_type &pos) const {
  assert(mask_image.valid());
  // For thread-safety
  Image<bool> mask(mask_image);
  Data result(voxel2real(pos), centre_index);
  auto table_it = shared->begin();
  while (table_it != shared->end()) {
    // If there's a tie in distances, want to include all such offsets in the kernel,
    //   even if the size of the utilised kernel extends beyond the minimum size
    if (result.voxels.size() >= min_size && table_it->sq_distance != result.max_distance)
      break;
    const Voxel::index_type voxel({pos[0] + table_it->index[0],   //
                                   pos[1] + table_it->index[1],   //
                                   pos[2] + table_it->index[2]}); //
    if (!is_out_of_bounds(H, voxel, 0, 3)) {
      assign_pos_of(voxel).to(mask);
      if (mask.value()) {
        result.voxels.push_back(Voxel(voxel, table_it->sq_distance));
        result.max_distance = table_it->sq_distance;
      }
    }
    ++table_it;
  }
  result.max_distance = std::sqrt(result.max_distance);
  return result;
}

default_type SphereRatio::compute_max_radius(const Header &voxel_grid, const default_type min_ratio) const {
  const size_t num_volumes = voxel_grid.size(3);
  const default_type voxel_volume = voxel_grid.spacing(0) * voxel_grid.spacing(1) * voxel_grid.spacing(2);
  const default_type sphere_volume = 8.0 * num_volumes * min_ratio * voxel_volume;
  const default_type approx_radius = std::sqrt(sphere_volume * 0.75 / Math::pi);
  const Voxel::index_type half_extents({ssize_t(std::ceil(approx_radius / voxel_grid.spacing(0))),   //
                                        ssize_t(std::ceil(approx_radius / voxel_grid.spacing(1))),   //
                                        ssize_t(std::ceil(approx_radius / voxel_grid.spacing(2)))}); //
  return std::max({half_extents[0] * voxel_grid.spacing(0),
                   half_extents[1] * voxel_grid.spacing(1),
                   half_extents[2] * voxel_grid.spacing(2)});
}

} // namespace MR::Denoise::Kernel
