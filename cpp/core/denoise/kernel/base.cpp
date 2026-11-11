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

#include "denoise/kernel/base.h"

namespace MR::Denoise::Kernel {

Base::Base(const Header &H, const std::array<ssize_t, 3> &subsample_factors)
    : H(H),
      transform(H),
      halfvoxel_offsets({subsample_factors[0] & 1 ? 0.0 : 0.5,
                         subsample_factors[1] & 1 ? 0.0 : 0.5,
                         subsample_factors[2] & 1 ? 0.0 : 0.5}) {}

} // namespace MR::Denoise::Kernel
