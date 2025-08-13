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

#include "denoise/kernel/sphere_rank.h"

#include "interp/linear.h"

namespace MR::Denoise::Kernel {

Data SphereRank::operator()(const Voxel::index_type &pos) const {
  assert(mask_image.valid());
  const Eigen::Vector3d realpos(voxel2real(pos));
  // For thread-safety
  default_type local_rank_per_mm (default_type(0));
  {
    Interp::Linear<Image<float>> interp(rank_per_mm);
    interp.scanner(realpos);
    if (!interp)
      throw Exception("Linear interpolation of rank from prior iteration failed");
    local_rank_per_mm = interp.value();
  }
  // For thread-safety
  Image<bool> mask(mask_image);
  Data result(realpos, centre_index);
  auto table_it = shared->begin();
  // Here it's best to keep track of both the squared radius and the radius;
  //   they are used for different purposes
  default_type max_sq_distance (default_type(0));
  while (table_it != shared->end()) {
    // Defining feature for this kernel:
    // Set the kernel size in such a way that it should be large enough
    //   that if one were to remove the signal components,
    //   the remaining noise section would be square
    if (std::isfinite(result.max_distance)
        && result.voxels.size() >= (num_volumes + (local_rank_per_mm * result.max_distance))
        && table_it->sq_distance != max_sq_distance)
      break;
    const Voxel::index_type voxel({pos[0] + table_it->index[0],   //
                                   pos[1] + table_it->index[1],   //
                                   pos[2] + table_it->index[2]}); //
    if (!is_out_of_bounds(H, voxel, 0, 3)) {
      assign_pos_of(voxel).to(mask);
      if (mask.value()) {
        result.voxels.push_back(Voxel(voxel, table_it->sq_distance));
        max_sq_distance = table_it->sq_distance;
        result.max_distance = std::sqrt(max_sq_distance);
      }
    }
    ++table_it;
  }
  return result;
}

} // namespace MR::Denoise::Kernel
