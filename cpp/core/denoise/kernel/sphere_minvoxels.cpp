/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
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
  Data result(voxel2real(pos), centre_index);
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
