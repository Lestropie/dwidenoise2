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

namespace MR::Denoise::Kernel {

class Base {
public:
  Base(const Header &H) : H(H) {}
  Base(const Base &) = default;
  virtual ~Base() = default;
  // This is just for pre-allocating matrices
  virtual ssize_t estimated_size() const = 0;
  // This is the interface that kernels must provide
  virtual Data operator()(const Voxel::index_type &) const = 0;

protected:
  const Header H;
};

} // namespace MR::Denoise::Kernel
