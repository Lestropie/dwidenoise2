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
// For nonstationarity correction,
//   "Voxel" also needs a noise level estimate per voxel in the patch,
//   as the scaling on insertion of data into the matrix
//   needs to be reversed when reconstructing the denoised signal from the eigenvectors
class Voxel : public VoxelBase<ssize_t> {
public:
  using index_type = VoxelBase<ssize_t>::index_type;
  Voxel(const index_type &index, const default_type sq_distance, const default_type noise_level)
      : VoxelBase<ssize_t>(index, sq_distance), noise_level(noise_level) {}
  Voxel(const index_type &index, const default_type sq_distance)
      : VoxelBase<ssize_t>(index, sq_distance), noise_level(std::numeric_limits<default_type>::signaling_NaN()) {}
  default_type noise_level;
};

using Offset = VoxelBase<int>;

} // namespace MR::Denoise::Kernel
