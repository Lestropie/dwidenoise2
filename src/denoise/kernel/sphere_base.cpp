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
