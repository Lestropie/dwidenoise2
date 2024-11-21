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
  SphereBase(const Header &voxel_grid, const default_type max_radius, const std::array<ssize_t, 3> &subsample_factors)
      : Base(voxel_grid), shared(new Shared(voxel_grid, max_radius, subsample_factors)) {}

  SphereBase(const SphereBase &) = default;

  virtual ~SphereBase() override {}

protected:
  class Shared {
  public:
    using TableType = std::vector<Offset>;
    Shared(const Header &voxel_grid, const default_type max_radius, const std::array<ssize_t, 3> &subsample_factors);
    TableType::const_iterator begin() const { return data.begin(); }
    TableType::const_iterator end() const { return data.end(); }

  private:
    TableType data;
  };

  std::shared_ptr<Shared> shared;
};

} // namespace MR::Denoise::Kernel
