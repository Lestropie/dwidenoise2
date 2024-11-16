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

#include <string>

#include "header.h"
#include "image.h"

namespace MR::Denoise {

class Exports {
public:
  Exports(const Header &in) : H(in) {
    H.ndim() = 3;
    H.reset_intensity_scaling();
  }
  void set_noise_out(const std::string &path) { noise_out = Image<float>::create(path, H); }
  void set_rank_input(const std::string &path) { rank_input = Image<uint16_t>::create(path, H); }
  void set_rank_output(const std::string &path) { rank_output = Image<float>::create(path, H); }
  void set_sum_optshrink(const std::string &path) { sum_optshrink = Image<float>::create(path, H); }
  void set_max_dist(const std::string &path) { max_dist = Image<float>::create(path, H); }
  void set_voxelcount(const std::string &path) { voxelcount = Image<uint16_t>::create(path, H); }
  void set_sum_aggregation(const std::string &path) {
    if (path.empty())
      sum_aggregation = Image<float>::scratch(H, "Scratch image for patch aggregation sums");
    else
      sum_aggregation = Image<float>::create(path, H);
  }

  Image<float> noise_out;
  Image<uint16_t> rank_input;
  Image<float> rank_output;
  Image<float> sum_optshrink;
  Image<float> max_dist;
  Image<uint16_t> voxelcount;
  Image<float> sum_aggregation;

protected:
  Header H;
};

} // namespace MR::Denoise
