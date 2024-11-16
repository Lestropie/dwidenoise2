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

#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "header.h"

namespace MR::Denoise::Kernel {

class Base {
public:
  Base(const Header &H) : H(H) {}
  Base(const Base &) = default;
  virtual ~Base() = default;
  // This is just for pre-allocating matrices
  virtual ssize_t estimated_size() const = 0;
  // This is the interface that kernels must provide
  virtual Data operator()(const Voxel::index_type &) const = 0;

protected:
  const Header H;
};

} // namespace MR::Denoise::Kernel
