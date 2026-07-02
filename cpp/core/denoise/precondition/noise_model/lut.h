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

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include "types.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Reusable lookup table on a uniform grid, with linear interpolation in the
//   interior and linear extrapolation beyond either end.
// Used to tabulate the (otherwise expensive) forward / inverse transforms
//   as one-dimensional functions of the normalised intensity theta = m / sigma.
class LUT {
public:
  LUT() : grid_origin(0.0), grid_spacing(1.0) {}
  LUT(const default_type origin, const default_type spacing, std::vector<default_type> &&values)
      : grid_origin(origin), grid_spacing(spacing), y(std::move(values)) {
    assert(grid_spacing > default_type(0));
  }
  bool empty() const { return y.empty(); }
  default_type operator()(const default_type x) const {
    assert(!y.empty());
    if (y.size() == 1)
      return y[0];
    const default_type t = (x - grid_origin) / grid_spacing;
    if (t <= default_type(0))
      return y[0] + t * (y[1] - y[0]);
    const ssize_t n = ssize_t(y.size());
    const ssize_t i = ssize_t(std::floor(t));
    if (i >= n - 1)
      return y[n - 1] + (t - default_type(n - 1)) * (y[n - 1] - y[n - 2]);
    const default_type frac = t - default_type(i);
    return y[i] * (default_type(1) - frac) + y[i + 1] * frac;
  }
  default_type origin() const { return grid_origin; }
  default_type spacing() const { return grid_spacing; }
  const std::vector<default_type> &values() const { return y; }

private:
  default_type grid_origin;
  default_type grid_spacing;
  std::vector<default_type> y;
};

} // namespace MR::Denoise::Precondition::NoiseModel
