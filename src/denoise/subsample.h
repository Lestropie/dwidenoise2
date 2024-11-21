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

#include "app.h"
#include "denoise/kernel/voxel.h"
#include "header.h"

namespace MR::Denoise {

extern const App::Option subsample_option;

class Subsample {
public:
  Subsample(const Header &in, const std::array<ssize_t, 3> &factors);

  const Header &header() const { return H_ss; }

  // TODO What other functionalities does this class need?
  // - Decide whether a given 3-vector voxel position should be sampled vs. not
  // - Convert input image 3-vector voxel position to output subsampled header voxel position
  // - Convert subsampled header voxel position to centre voxel in input image
  // TODO May want to move definition of Kernel::Voxel out of Kernel namespace
  bool process(const Kernel::Voxel::index_type &pos) const;
  // TODO Rename these
  std::array<ssize_t, 3> in2ss(const Kernel::Voxel::index_type &pos) const;
  std::array<ssize_t, 3> ss2in(const Kernel::Voxel::index_type &pos) const;
  const std::array<ssize_t, 3> &get_factors() const { return factors; }

  static std::shared_ptr<Subsample> make(const Header &in);

protected:
  const Header H_in;
  const std::array<ssize_t, 3> factors;
  const std::array<ssize_t, 3> size;
  const std::array<ssize_t, 3> origin;
  const Header H_ss;

  Header make_input_header(const Header &) const;
  Header make_subsample_header() const;
};

} // namespace MR::Denoise
