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

#include "denoise/denoise.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/sphere_base.h"
#include "header.h"
#include "mutexprotected.h"

namespace MR::Denoise::Kernel {

constexpr default_type default_aspect_ratio = 1.0 / 0.85;

class SphereMinVoxels : public SphereBase {

public:
  SphereMinVoxels(const Header &voxel_grid, const std::array<ssize_t, 3> &subsample_factors, const ssize_t min_voxels)
      : SphereBase(voxel_grid, subsample_factors, SphereBase::compute_max_radius(voxel_grid, min_voxels)),
        min_size(min_voxels),
        min_truncated_size(0) {}

  SphereMinVoxels(const SphereMinVoxels &) = default;

  ~SphereMinVoxels() override;

  Data operator()(const Voxel::index_type &pos) const override;

  ssize_t estimated_size() const override { return min_size; }

private:
  ssize_t min_size;

  mutable MutexProtected<ssize_t> min_truncated_size;

};

} // namespace MR::Denoise::Kernel
