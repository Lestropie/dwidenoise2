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

#include "denoise/denoise.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/sphere_base.h"
#include "header.h"
#include "mutexprotected.h"

namespace MR::Denoise::Kernel {

constexpr default_type default_aspect_ratio = 1.0 / 0.85;

class SphereMinVoxels : public SphereBase {

public:
  SphereMinVoxels(const Header &voxel_grid, const std::array<ssize_t, 3> &subsample_factors, const ssize_t min_voxels)
      : SphereBase(voxel_grid, subsample_factors, SphereBase::compute_max_radius(voxel_grid, min_voxels)),
        min_size(min_voxels),
        min_truncated_size(0) {}

  SphereMinVoxels(const SphereMinVoxels &) = default;

  ~SphereMinVoxels() override;

  Data operator()(const Voxel::index_type &pos) const override;

  ssize_t estimated_size() const override { return min_size; }

private:
  ssize_t min_size;

  mutable MutexProtected<ssize_t> min_truncated_size;

};

} // namespace MR::Denoise::Kernel
