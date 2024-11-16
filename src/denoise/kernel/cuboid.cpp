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

#include "denoise/kernel/cuboid.h"

namespace MR::Denoise::Kernel {

Cuboid::Cuboid(const Header &header, const std::vector<uint32_t> &extent)
    : Base(header),
      half_extent({ssize_t(extent[0] / 2), ssize_t(extent[1] / 2), ssize_t(extent[2] / 2)}),
      size(ssize_t(extent[0]) * ssize_t(extent[1]) * ssize_t(extent[2])),
      centre_index(size / 2) {
  for (auto e : extent) {
    if (!(e % 2))
      throw Exception("Size of cubic kernel must be an odd integer");
  }
}

namespace {
// patch handling at image edges
inline ssize_t wrapindex(int p, int r, int e, int max) {
  int rr = p + r;
  if (rr < 0)
    rr = e - r;
  if (rr >= max)
    rr = (max - 1) - e - r;
  return rr;
}
} // namespace

Data Cuboid::operator()(const Voxel::index_type &pos) const {
  Data result(centre_index);
  Voxel::index_type voxel;
  Offset::index_type offset;
  for (offset[2] = -half_extent[2]; offset[2] <= half_extent[2]; ++offset[2]) {
    voxel[2] = wrapindex(pos[2], offset[2], half_extent[2], H.size(2));
    for (offset[1] = -half_extent[1]; offset[1] <= half_extent[1]; ++offset[1]) {
      voxel[1] = wrapindex(pos[1], offset[1], half_extent[1], H.size(1));
      for (offset[0] = -half_extent[0]; offset[0] <= half_extent[0]; ++offset[0]) {
        voxel[0] = wrapindex(pos[0], offset[0], half_extent[0], H.size(0));
        // Both "pos" and "voxel" are unsigned, so beware of integer overflow
        const default_type sq_distance = Math::pow2(std::min(pos[0] - voxel[0], voxel[0] - pos[0]) * H.spacing(0)) +
                                         Math::pow2(std::min(pos[1] - voxel[1], voxel[1] - pos[1]) * H.spacing(1)) +
                                         Math::pow2(std::min(pos[2] - voxel[2], voxel[2] - pos[2]) * H.spacing(2));
        result.voxels.push_back(Voxel(voxel, sq_distance));
        result.max_distance = std::max(result.max_distance, sq_distance);
      }
    }
  }
  result.max_distance = std::sqrt(result.max_distance);
  return result;
}

} // namespace MR::Denoise::Kernel
