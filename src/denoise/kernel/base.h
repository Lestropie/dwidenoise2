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

#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "header.h"
#include "transform.h"

namespace MR::Denoise::Kernel {

class Base {
public:
  Base(const Header &H, const std::array<ssize_t, 3> &subsample_factors)
      : H(H),
        transform(H),
        halfvoxel_offsets({subsample_factors[0] & 1 ? 0.0 : 0.5,
                           subsample_factors[1] & 1 ? 0.0 : 0.5,
                           subsample_factors[2] & 1 ? 0.0 : 0.5}) {}
  Base(const Base &) = default;
  virtual ~Base() = default;
  // This is just for pre-allocating matrices
  virtual ssize_t estimated_size() const = 0;
  // This is the interface that kernels must provide
  virtual Data operator()(const Voxel::index_type &) const = 0;

protected:
  const Header H;
  const Transform transform;
  std::array<default_type, 3> halfvoxel_offsets;

  // For translating the index of a processed voxel
  //   into a realspace position corresponding to the centre of the patch,
  //   accounting for the fact that subsampling may be introducing an offset
  //   such that the actual centre of the patch is not at the centre of this voxel
  Eigen::Vector3d voxel2real(const Kernel::Voxel::index_type &pos) const {
    return (                                               //
        transform.voxel2scanner *                          //
        Eigen::Vector3d({pos[0] + halfvoxel_offsets[0],    //
                         pos[1] + halfvoxel_offsets[1],    //
                         pos[2] + halfvoxel_offsets[2]})); //
  }
};

} // namespace MR::Denoise::Kernel
