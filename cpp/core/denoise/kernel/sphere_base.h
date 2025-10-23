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

#include <array>
#include <memory>
#include <vector>

#include "denoise/kernel/base.h"
#include "denoise/kernel/kernel.h"
#include "denoise/kernel/voxel.h"
#include "header.h"

namespace MR::Denoise::Kernel {

class SphereBase : public Base {

public:
  SphereBase(const Header &voxel_grid, const std::array<ssize_t, 3> &subsample_factors, const default_type max_radius)
      : Base(voxel_grid, subsample_factors),
        shared(new Shared(voxel_grid, subsample_factors, halfvoxel_offsets, max_radius)),
        centre_index(subsample_factors == std::array<ssize_t, 3>({1, 1, 1}) ? 0 : -1) {}

  SphereBase(const SphereBase &) = default;

  virtual ~SphereBase() override {}

protected:
  class Shared {
  public:
    using TableType = std::vector<Offset>;
    Shared(const Header &voxel_grid,
           const std::array<ssize_t, 3> &subsample_factors,
           const std::array<default_type, 3> &halfvoxel_offsets,
           const default_type max_radius);
    TableType::const_iterator begin() const { return data.begin(); }
    TableType::const_iterator end() const { return data.end(); }

  private:
    TableType data;
  };

  std::shared_ptr<Shared> shared;
  const ssize_t centre_index;

  // Determine an appropriate bounding box from which to generate the search table
  // Find the radius for which 7/8 of the sphere will contain the minimum number of voxels, then round up
  // This is only for setting the maximal radius for generation of the lookup table
  default_type compute_max_radius(const Header &voxel_grid, const ssize_t min_size) const;
};

} // namespace MR::Denoise::Kernel
