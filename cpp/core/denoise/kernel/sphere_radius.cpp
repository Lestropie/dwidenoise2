/* Required Notice: Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;
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

#include <algorithm>
#include <iterator>

#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise::Kernel {

SphereFixedRadius::SphereFixedRadius(const Header &voxel_grid,                        //
                                     const std::array<ssize_t, 3> &subsample_factors, //
                                     const default_type radius)                       //
    : SphereBase(voxel_grid, subsample_factors, radius),                              //
      maximum_size(std::distance(shared->begin(), shared->end())) {                    //
  // A fixed-radius kernel does not expand at the image edges, so a radius exceeding the image
  //   is not merely wasteful: every patch becomes the whole image. Report that explicitly,
  //   since the request itself is silently capped at the image bounding radius.
  const default_type effective_radius = std::min(radius, fov_bounding_radius(voxel_grid));
  if (effective_radius < radius)
    WARN("Fixed-radius kernel of " + str(radius) +
         "mm exceeds the image field of view; "
         "every patch will contain the whole image");
  INFO("Maximum number of voxels in " + str(effective_radius) + "mm fixed-radius kernel is " + str(maximum_size));
}

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
