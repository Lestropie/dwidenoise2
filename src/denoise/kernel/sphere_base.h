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

#include <array>
#include <memory>
#include <vector>

#include "denoise/kernel/base.h"
#include "denoise/kernel/kernel.h"
#include "denoise/kernel/voxel.h"
#include "header.h"

namespace MR::Denoise::Kernel {

class SphereBase : public Base {

public:
  SphereBase(const Header &voxel_grid, const default_type max_radius, const std::array<ssize_t, 3> &subsample_factors)
      : Base(voxel_grid), shared(new Shared(voxel_grid, max_radius, subsample_factors)) {}

  SphereBase(const SphereBase &) = default;

  virtual ~SphereBase() override {}

protected:
  class Shared {
  public:
    using TableType = std::vector<Offset>;
    Shared(const Header &voxel_grid, const default_type max_radius, const std::array<ssize_t, 3> &subsample_factors);
    TableType::const_iterator begin() const { return data.begin(); }
    TableType::const_iterator end() const { return data.end(); }

  private:
    TableType data;
  };

  std::shared_ptr<Shared> shared;
};

} // namespace MR::Denoise::Kernel
