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

#include "app.h"
#include "denoise/denoise.h"
#include "denoise/kernel/voxel.h"
#include "header.h"

namespace MR::Denoise {

extern const App::Option subsample_option;

class Subsample {
public:
  Subsample(const Header &in, const std::array<ssize_t, 3> &factors);

  const Header &header() const { return H_ss; }

  // TODO May want to move definition of Kernel::Voxel out of Kernel namespace
  bool process(const Kernel::Voxel::index_type &pos) const;
  std::array<ssize_t, 3> in2ss(const Kernel::Voxel::index_type &pos) const;
  std::array<ssize_t, 3> ss2in(const Kernel::Voxel::index_type &pos) const;
  const std::array<ssize_t, 3> &get_factors() const { return factors; }

  static std::shared_ptr<Subsample> make(const Header &in, const ssize_t default_factor);
  static std::shared_ptr<Subsample> make(const Header &in, std::array<ssize_t, 3> default_factors);

protected:
  const Header H_in;
  const std::array<ssize_t, 3> factors;
  const std::array<ssize_t, 3> size;
  const std::array<ssize_t, 3> origin;
  const Header H_ss;

  Header make_input_header(const Header &) const;
  Header make_subsample_header() const;
};

} // namespace MR::Denoise
