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

#include "denoise/kernel/sphere_ratio.h"

namespace MR::Denoise::Kernel {

Data SphereRatio::operator()(const Voxel::index_type &pos) const {
  Data result(0);
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
      result.voxels.push_back(Voxel(voxel, table_it->sq_distance));
      result.max_distance = table_it->sq_distance;
    }
    ++table_it;
  }
  if (table_it == shared->end()) {
    throw Exception(                                                                   //
        std::string("Inadequate spherical kernel initialisation ")                     //
        + "(lookup table " + str(std::distance(shared->begin(), shared->end())) + "; " //
        + "min size " + str(min_size) + "; "                                           //
        + "read size " + str(result.voxels.size()) + ")");                             //
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
