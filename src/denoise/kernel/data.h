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

#include <vector>

#include "denoise/kernel/voxel.h"
#include "types.h"

namespace MR::Denoise::Kernel {

class Data {
public:
  Data() : centre_index(-1), max_distance(-std::numeric_limits<default_type>::infinity()) {}
  Data(const ssize_t i) : centre_index(i), max_distance(-std::numeric_limits<default_type>::infinity()) {}
  std::vector<Voxel> voxels;
  ssize_t centre_index;
  default_type max_distance;
};

} // namespace MR::Denoise::Kernel
