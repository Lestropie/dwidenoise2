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

#include <memory>

#include <Eigen/Dense>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

template <typename F> class Estimate {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;

  Estimate(const Header &header,
           Image<bool> &mask,
           std::shared_ptr<Kernel::Base> kernel,
           std::shared_ptr<Estimator::Base> estimator,
           Exports &exports);

  void operator()(Image<F> &dwi);

protected:
  const ssize_t m;

  // Denoising configuration
  Image<bool> mask;
  std::shared_ptr<Kernel::Base> kernel;
  std::shared_ptr<Estimator::Base> estimator;

  // Reusable memory
  Kernel::Data neighbourhood;
  MatrixType X;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  eigenvalues_type s;
  Estimator::Result threshold;
  vector_type clam;

  // Export images
  Exports exports;

  void load_data(Image<F> &image, const std::vector<Kernel::Voxel> &voxels);
};

template class Estimate<float>;
template class Estimate<cfloat>;
template class Estimate<double>;
template class Estimate<cdouble>;

} // namespace MR::Denoise
