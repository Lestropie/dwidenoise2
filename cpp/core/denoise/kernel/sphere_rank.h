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
#include "image.h"

namespace MR::Denoise::Kernel {

class SphereRank : public SphereBase {

// TODO Get rank image estimate from prior iteration into here
// From experimentation, rank approximately scales *linearly with respect to radius*
//   (with an intercept of zero);
//   rank image from early iterations will require rescaling
//   if the kernel size changes between iterations

public:
  SphereRank(const Header &voxel_grid,
             const std::array<ssize_t, 3> &subsample_factors,
             const Image<float> &rank_per_mm)
      : SphereBase(voxel_grid, subsample_factors, SphereBase::compute_max_radius(voxel_grid, 2.0)),
        rank_per_mm (rank_per_mm),
        num_volumes (Denoise::num_volumes(voxel_grid)) {}

  SphereRank(const SphereRank &) = default;

  ~SphereRank() override = default;

  Data operator()(const Voxel::index_type &pos) const override;

  // Only used for preallocation of matrices; imprecision not consequential
  ssize_t estimated_size() const override { return num_volumes; }

private:
  // In testing, the signal rank appears to scale proportionately
  //   with the radius of the kernel.
  // Since the kernel size might differ between iterations,
  //   encapsulate this image information as a rank per millimetre kernel radius
  Image<float> rank_per_mm;
  ssize_t num_volumes;

};

} // namespace MR::Denoise::Kernel
