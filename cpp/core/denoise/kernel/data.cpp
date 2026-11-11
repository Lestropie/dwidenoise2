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

#include "denoise/kernel/data.h"

#include <limits>

namespace MR::Denoise::Kernel {

Data::Data(const ssize_t i)
    : centre_realspace(Eigen::Vector3d::Constant(std::numeric_limits<double>::signaling_NaN())),
      centre_index(i),
      max_distance(-std::numeric_limits<default_type>::infinity()) {}

Data::Data()
    : centre_realspace(Eigen::Vector3d::Constant(std::numeric_limits<double>::signaling_NaN())),
      centre_index(-1),
      max_distance(-std::numeric_limits<default_type>::infinity()) {}

} // namespace MR::Denoise::Kernel
