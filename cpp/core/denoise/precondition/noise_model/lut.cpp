/* Required Notice: Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;
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

#include "denoise/precondition/noise_model/lut.h"

#include <cassert>
#include <cmath>

namespace MR::Denoise::Precondition::NoiseModel {

default_type LUT::operator()(const default_type x) const {
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

} // namespace MR::Denoise::Precondition::NoiseModel
