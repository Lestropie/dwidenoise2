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

#include "denoise/kernel/sphere_minvoxels.h"

namespace MR::Denoise::Kernel {

SphereMinVoxels::~SphereMinVoxels() {
  auto guard = min_truncated_size.lock();
  if (*guard != 0) {
    WARN("Some PCA kernels may have been smaller than minimum size due to incomplete kernel initialisation"
         " (minimum specified size = " + str(min_size) + "; minimum actual size = " + str(*guard) + ")");
  }
}

Data SphereMinVoxels::operator()(const Voxel::index_type &pos) const {
  assert(mask_image.valid());
  // For thread-safety
  Image<bool> mask(mask_image);
  Data result(pos, voxel2real(pos), centre_index);
  auto table_it = shared->begin();
  while (table_it != shared->end()) {
    // If there's a tie in distances, want to include all such offsets in the kernel,
    //   even if the size of the utilised kernel extends beyond the minimum size
    if (result.voxels.size() >= min_size && table_it->sq_distance != result.max_distance) {
      result.max_distance = std::sqrt(result.max_distance);
      return result;
    }
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
  auto guard = min_truncated_size.lock();
  if (*guard == 0)
    *guard = result.voxels.size();
  else if (result.voxels.size() < *guard)
    *guard = result.voxels.size();
  result.max_distance = std::sqrt(result.max_distance);
  return result;
}

} // namespace MR::Denoise::Kernel
