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
#include <string>
#include <vector>

#include "app.h"
#include "header.h"
#include "image.h"
#include "types.h"

namespace MR::Denoise::Kernel {

class Base;

extern const char *const shape_description;
extern const char *const default_size_description;
extern const char *const cuboid_size_description;

const std::vector<std::string> shapes = {"cuboid", "sphere"};
enum class shape_type { CUBOID, SPHERE };
extern const App::OptionGroup options;

// H is the header of the data to be decomposed; num_volumes(H) gives the number of volumes
//   in the Casorati matrix for this iteration (m', which is reduced under temporal
//   sub-sampling). full_num_volumes is the number of volumes in the absence of temporal
//   sub-sampling; it is used only to decide whether the Casorati shape warning applies.
std::shared_ptr<Base> make_kernel(const Header &H,                                 //
                                  const std::array<ssize_t, 3> &subsample_factors, //
                                  const default_type size_multiplier,              //
                                  const Image<float> &rank_per_mm_image,           //
                                  const ssize_t full_num_volumes);                 //

} // namespace MR::Denoise::Kernel
