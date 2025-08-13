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

#include "denoise/denoise.h"
#include "math/math.h"

namespace MR::Denoise::Kernel {

SphereBase::Shared::Shared(const Header &voxel_grid,
                           const std::array<ssize_t, 3> &subsample_factors,
                           const std::array<default_type, 3> &halfvoxel_offsets,
                           const default_type max_radius) {
  const default_type max_radius_sq = Math::pow2(max_radius);
  Eigen::Array<int, 3, 2> bounding_box;
  for (ssize_t axis = 0; axis != 3; ++axis) {
    if (subsample_factors[axis] % 2) {
      bounding_box(axis, 1) = int(std::ceil(max_radius / voxel_grid.spacing(axis)));
      bounding_box(axis, 0) = -bounding_box(axis, 1);
    } else {
      bounding_box(axis, 0) = -int(std::ceil((max_radius / voxel_grid.spacing(axis)) - 0.5));
      bounding_box(axis, 1) = int(std::ceil((max_radius / voxel_grid.spacing(axis)) + 0.5));
    }
  }
  // Build the searchlight
  data.reserve(size_t(bounding_box(0, 1) + 1 - bounding_box(0, 0)) * //
               size_t(bounding_box(1, 1) + 1 - bounding_box(1, 0)) * //
               size_t(bounding_box(2, 1) + 1 - bounding_box(2, 0))); //
  Offset::index_type offset({0, 0, 0});
  for (offset[2] = bounding_box(2, 0); offset[2] <= bounding_box(2, 1); ++offset[2]) {
    for (offset[1] = bounding_box(1, 0); offset[1] <= bounding_box(1, 1); ++offset[1]) {
      for (offset[0] = bounding_box(0, 0); offset[0] <= bounding_box(0, 1); ++offset[0]) {
        const default_type squared_distance =
            Math::pow2((offset[0] - halfvoxel_offsets[0]) * voxel_grid.spacing(0))    //
            + Math::pow2((offset[1] - halfvoxel_offsets[1]) * voxel_grid.spacing(1))  //
            + Math::pow2((offset[2] - halfvoxel_offsets[2]) * voxel_grid.spacing(2)); //
        if (squared_distance <= max_radius_sq)
          data.emplace_back(Offset(offset, squared_distance));
      }
    }
  }
  std::sort(data.begin(), data.end());
  DEBUG("Spherical searchlight construction:");
  DEBUG("  Voxel spacing: ["                 //
        + str(voxel_grid.spacing(0)) + ","   //
        + str(voxel_grid.spacing(1)) + ","   //
        + str(voxel_grid.spacing(2)) + "]"); //
  DEBUG("  Maximum nominated radius: " + str(max_radius));
  DEBUG("  Halfvoxel offsets: ["            //
        + str(halfvoxel_offsets[0]) + ","   //
        + str(halfvoxel_offsets[1]) + ","   //
        + str(halfvoxel_offsets[2]) + "]"); //
  DEBUG("  Bounding box for search: ["      //
        "[" +
        str(bounding_box(0, 0)) + " " + str(bounding_box(0, 1)) + "] " +       //
        "[" + str(bounding_box(1, 0)) + " " + str(bounding_box(1, 1)) + "] " + //
        "[" + str(bounding_box(2, 0)) + " " + str(bounding_box(2, 1)) + "]]"); //
  DEBUG("  First element: " + str(data.front().index.transpose()) + " @ " + str(data.front().distance()));
  DEBUG("  Last element: " + str(data.back().index.transpose()) + " @ " + str(data.back().distance()));
  DEBUG("  Number of elements: " + str(data.size()));
  /*
    // Warning is omitted:
    //   just because the search table itself is exceptionally large,
    //   does not mean that that number of voxels may be going into the PCA;
    //   computation of the maximal required radius for constructing the lookup table is intentionally pessimistic,
    //   and many of these offsets will be outside of the FoV.
    const size_t voxel_count = voxel_grid.size(0) * voxel_grid.size(1) * voxel_grid.size(2);
    if (data.size() > voxel_count) {
      WARN(std::string("Spherical sliding window larger than input image ")                          //
           + "(" + str(data.size()) + " > "                                                          //
           + str(voxel_grid.size(0)) + "x" + str(voxel_grid.size(1)) + "x" + str(voxel_grid.size(2)) //
           + "=" + str(voxel_count) + "); "                                                          //
           + "operation likely to be equivalent to running PCA on entire image");                    //
    }
  */
}

default_type SphereBase::compute_max_radius(const Header &voxel_grid, const default_type min_ratio) const {
  const size_t values_per_voxel = Denoise::num_volumes(voxel_grid);
  const default_type voxel_volume = voxel_grid.spacing(0) * voxel_grid.spacing(1) * voxel_grid.spacing(2);
  // Consider the worst case scenario, where the corner of the FoV is being processed;
  //   we do not want to run out of elements in our lookup table before reaching our desired # voxels
  // Define a sphere for which the volume is eight times that of what would be required
  //   for processing a voxel in the middle of the FoV;
  //   only one of the eight octants will have valid image data
  const default_type sphere_volume = 8.0 * values_per_voxel * min_ratio * voxel_volume;
  const default_type approx_radius = std::sqrt(sphere_volume * 0.75 / Math::pi);
  const Voxel::index_type half_extents({ssize_t(std::ceil(approx_radius / voxel_grid.spacing(0))),   //
                                        ssize_t(std::ceil(approx_radius / voxel_grid.spacing(1))),   //
                                        ssize_t(std::ceil(approx_radius / voxel_grid.spacing(2)))}); //
  default_type max_radius = std::max({half_extents[0] * voxel_grid.spacing(0),                       //
                                      half_extents[1] * voxel_grid.spacing(1),                       //
                                      half_extents[2] * voxel_grid.spacing(2)});                     //
  DEBUG("Calibrating maximal radius for building spherical denoising kernel:");
  std::string dim_string = str(voxel_grid.size(0));
  for (size_t axis = 1; axis != voxel_grid.ndim(); ++axis)
    dim_string += "x" + str(voxel_grid.size(axis));
  DEBUG("  Image dimensions: " + dim_string);
  DEBUG("  Values per voxel: " + str(values_per_voxel));
  std::string spacing_string = str(voxel_grid.spacing(0));
  for (size_t axis = 1; axis != 3; ++axis)
    spacing_string += "x" + str(voxel_grid.spacing(axis));
  DEBUG("  Voxel spacing: " + spacing_string + "mm");
  DEBUG("  Voxel volume: " + str(voxel_volume) + "mm^3");
  DEBUG("  Maximal sphere volume: " + str(sphere_volume));
  DEBUG("  Approximate radius: " + str(approx_radius));
  std::string halfextent_string = str(half_extents[0]);
  for (size_t axis = 1; axis != 3; ++axis)
    halfextent_string += "," + str(half_extents[axis]);
  DEBUG("  Half-extents: [" + halfextent_string + "]");
  std::string maxradius_string = str(half_extents[0] * voxel_grid.spacing(0));
  for (size_t axis = 1; axis != 3; ++axis)
    maxradius_string += "," + str(half_extents[axis] * voxel_grid.spacing(axis));
  DEBUG("  Maximum radius = max(" + maxradius_string + ") = " + str(max_radius));
  return max_radius;
}

} // namespace MR::Denoise::Kernel
