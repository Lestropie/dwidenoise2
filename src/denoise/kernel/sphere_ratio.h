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
#include "denoise/kernel/sphere_base.h"
#include "header.h"

namespace MR::Denoise::Kernel {

constexpr default_type sphere_multiplier_default = 1.0 / 0.85;

class SphereRatio : public SphereBase {

public:
  SphereRatio(const Header &voxel_grid, const std::array<ssize_t, 3> &subsample_factors, const default_type min_ratio)
      : SphereBase(voxel_grid, subsample_factors, compute_max_radius(voxel_grid, min_ratio)),
        min_size(std::ceil(voxel_grid.size(3) * min_ratio)) {}

  SphereRatio(const SphereRatio &) = default;

  ~SphereRatio() override = default;

  Data operator()(const Voxel::index_type &pos) const override;

  ssize_t estimated_size() const override { return min_size; }

private:
  ssize_t min_size;

  // Determine an appropriate bounding box from which to generate the search table
  // Find the radius for which 7/8 of the sphere will contain the minimum number of voxels, then round up
  // This is only for setting the maximal radius for generation of the lookup table
  default_type compute_max_radius(const Header &voxel_grid, const default_type min_ratio) const;
};

} // namespace MR::Denoise::Kernel
