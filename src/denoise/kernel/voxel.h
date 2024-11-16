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

#pragma once

#include <Eigen/Dense>

#include "types.h"

namespace MR::Denoise::Kernel {

template <class T> class VoxelBase {
public:
  using index_type = Eigen::Array<T, 3, 1>;
  VoxelBase(const index_type &index, const default_type sq_distance) : index(index), sq_distance(sq_distance) {}
  VoxelBase(const VoxelBase &) = default;
  VoxelBase(VoxelBase &&) = default;
  ~VoxelBase() {}
  VoxelBase &operator=(const VoxelBase &that) {
    index = that.index;
    sq_distance = that.sq_distance;
    return *this;
  }
  VoxelBase &operator=(VoxelBase &&that) noexcept {
    index = that.index;
    sq_distance = that.sq_distance;
    return *this;
  }
  bool operator<(const VoxelBase &that) const { return sq_distance < that.sq_distance; }
  default_type distance() const { return std::sqrt(sq_distance); }

  index_type index;
  default_type sq_distance;
};

// Need signed integer to represent offsets from the centre of the kernel;
//   however absolute voxel indices should be unsigned
using Voxel = VoxelBase<ssize_t>;
using Offset = VoxelBase<int>;

} // namespace MR::Denoise::Kernel
