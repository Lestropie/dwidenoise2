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

#include "denoise/kernel/sphere_rmse.h"

#include "interp/linear.h"

namespace MR::Denoise::Kernel {

Data SphereRMSE::operator()(const Voxel::index_type &pos) const {
  assert(mask_image.valid());
  const Eigen::Vector3d realpos(voxel2real(pos));
  // Signal-rank density at this location (ranks per mm of kernel radius), from a prior iteration.
  default_type local_rank_per_mm(default_type(0));
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
  default_type max_sq_distance(default_type(0));
  const ssize_t cap_voxels = ssize_t(std::ceil(cap_multiplier * num_volumes));
  while (table_it != shared->end()) {
    // Stopping test, evaluated only at shell boundaries (so tied distances are kept together):
    //   grow at least to the square noise block (n >= m + r), then stop once the predicted noise
    //   estimate is precise enough, or the hard voxel cap is reached.
    if (std::isfinite(result.max_distance) && table_it->sq_distance != max_sq_distance) {
      const default_type r = local_rank_per_mm * result.max_distance;
      const ssize_t n = ssize_t(result.voxels.size());
      const bool at_least_square = (default_type(n) >= num_volumes + r);
      if (at_least_square &&
          (n >= cap_voxels || predicted_rmse(num_volumes, n, r) <= rmse_tolerance))
        break;
    }
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
