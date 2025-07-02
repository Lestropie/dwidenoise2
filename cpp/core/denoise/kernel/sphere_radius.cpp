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

#include "denoise/kernel/sphere_radius.h"

namespace MR::Denoise::Kernel {

Data SphereFixedRadius::operator()(const Voxel::index_type &pos) const {
  assert(mask_image.valid());
  // For thread-safety
  Image<bool> mask(mask_image);
  Data result(voxel2real(pos), centre_index);
  result.voxels.reserve(maximum_size);
  for (auto map_it = shared->begin(); map_it != shared->end(); ++map_it) {
    const Voxel::index_type voxel({pos[0] + map_it->index[0],   //
                                   pos[1] + map_it->index[1],   //
                                   pos[2] + map_it->index[2]}); //
    if (!is_out_of_bounds(H, voxel, 0, 3)) {
      assign_pos_of(voxel).to(mask);
      if (mask.value()) {
        result.voxels.push_back(Voxel(voxel, map_it->sq_distance));
        result.max_distance = map_it->sq_distance;
      }
    }
  }
  result.max_distance = std::sqrt(result.max_distance);
  return result;
}

} // namespace MR::Denoise::Kernel
