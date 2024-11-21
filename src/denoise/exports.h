/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#pragma once

#include <string>

#include "header.h"
#include "image.h"

namespace MR::Denoise {

class Exports {
public:
  Exports(const Header &in, const Header &ss) : H_in(in), H_ss(ss) {
    H_in.ndim() = 3;
    H_in.reset_intensity_scaling();
    H_in.datatype() = DataType::Float32;
    H_in.datatype().set_byte_order_native();
  }
  void set_noise_out(const std::string &path) { noise_out = Image<float>::create(path, H_ss); }
  void set_rank_input(const std::string &path) {
    Header H(H_ss);
    H.datatype() = DataType::UInt16;
    H.datatype().set_byte_order_native();
    rank_input = Image<uint16_t>::create(path, H);
  }
  void set_rank_output(const std::string &path) { rank_output = Image<float>::create(path, H_in); }
  void set_sum_optshrink(const std::string &path) { sum_optshrink = Image<float>::create(path, H_ss); }
  void set_max_dist(const std::string &path) { max_dist = Image<float>::create(path, H_ss); }
  void set_voxelcount(const std::string &path) {
    Header H(H_ss);
    H.datatype() = DataType::UInt16;
    H.datatype().set_byte_order_native();
    voxelcount = Image<uint16_t>::create(path, H);
  }
  void set_patchcount(const std::string &path) {
    Header H(H_in);
    H.datatype() = DataType::UInt16;
    H.datatype().set_byte_order_native();
    patchcount = Image<uint16_t>::create(path, H);
  }
  void set_sum_aggregation(const std::string &path) {
    if (path.empty())
      sum_aggregation = Image<float>::scratch(H_in, "Scratch image for patch aggregation sums");
    else
      sum_aggregation = Image<float>::create(path, H_in);
  }

  Image<float> noise_out;
  Image<uint16_t> rank_input;
  Image<float> rank_output;
  Image<float> sum_optshrink;
  Image<float> max_dist;
  Image<uint16_t> voxelcount;
  Image<uint16_t> patchcount;
  Image<float> sum_aggregation;

protected:
  Header H_in;
  Header H_ss;
};

} // namespace MR::Denoise
