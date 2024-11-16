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

#include <vector>

#include "denoise/kernel/base.h"
#include "denoise/kernel/data.h"
#include "header.h"

namespace MR::Denoise::Kernel {

class Cuboid : public Base {

public:
  Cuboid(const Header &header, const std::vector<uint32_t> &extent);
  Cuboid(const Cuboid &) = default;
  ~Cuboid() final = default;
  Data operator()(const Voxel::index_type &pos) const override;
  ssize_t estimated_size() const override { return size; }

private:
  const Voxel::index_type half_extent;
  const ssize_t size;
  const ssize_t centre_index;
};

} // namespace MR::Denoise::Kernel
