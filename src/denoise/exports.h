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
  Exports(const Header &in, const Header &ss);
  Exports(const Exports &that) = default;

  void set_noise_out(const std::string &path);
  void set_rank_input(const std::string &path);
  void set_rank_output(const std::string &path);
  void set_sum_optshrink(const std::string &path);
  void set_max_dist(const std::string &path);
  void set_voxelcount(const std::string &path);
  void set_patchcount(const std::string &path);
  void set_sum_aggregation(const std::string &path);

  Image<float> noise_out;
  Image<uint16_t> rank_input;
  Image<float> rank_output;
  Image<float> sum_optshrink;
  Image<float> max_dist;
  Image<uint16_t> voxelcount;
  Image<uint16_t> patchcount;
  Image<float> sum_aggregation;

protected:
  std::shared_ptr<Header> H_in;
  std::shared_ptr<Header> H_ss;
};

} // namespace MR::Denoise
