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

#include "denoise/exports.h"

#include "dwi/gradient.h"

namespace MR::Denoise {

Exports::Exports(const Header &in, const Header &ss) //
    : H_in(std::make_shared<Header>(in)),            //
      H_ss(std::make_shared<Header>(ss)) {           //
  DWI::clear_DW_scheme(*H_in);
  H_in->ndim() = 3;
  H_in->reset_intensity_scaling();
  H_in->datatype() = DataType::Float32;
  H_in->datatype().set_byte_order_native();
}

void Exports::set_noise_out(const std::string &path) { noise_out = Image<float>::create(path, *H_ss); }

void Exports::set_rank_input(const std::string &path) {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  rank_input = Image<uint16_t>::create(path, H);
}

void Exports::set_rank_output(const std::string &path) { rank_output = Image<float>::create(path, *H_in); }

void Exports::set_sum_optshrink(const std::string &path) { sum_optshrink = Image<float>::create(path, *H_ss); }

void Exports::set_max_dist(const std::string &path) { max_dist = Image<float>::create(path, *H_ss); }

void Exports::set_voxelcount(const std::string &path) {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  voxelcount = Image<uint16_t>::create(path, H);
}

void Exports::set_patchcount(const std::string &path) {
  Header H(*H_in);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  patchcount = Image<uint16_t>::create(path, H);
}

void Exports::set_sum_aggregation(const std::string &path) {
  if (path.empty())
    sum_aggregation = Image<float>::scratch(*H_in, "Scratch image for patch aggregation sums");
  else
    sum_aggregation = Image<float>::create(path, *H_in);
}

} // namespace MR::Denoise
