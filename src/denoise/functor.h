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

#include <limits>
#include <memory>
#include <mutex>

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

template <typename F> class Functor {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;

  Functor(const Header &header,
          Image<bool> &mask,
          std::shared_ptr<Kernel::Base> kernel,
          std::shared_ptr<Estimator::Base> estimator,
          filter_type filter,
          aggregator_type aggregator,
          Exports &exports);

  void operator()(Image<F> &dwi, Image<F> &out);

private:
  // Denoising configuration
  const ssize_t m;
  Image<bool> mask;
  std::shared_ptr<Kernel::Base> kernel;
  std::shared_ptr<Estimator::Base> estimator;
  filter_type filter;
  aggregator_type aggregator;
  double gaussian_multiplier;

  // Reusable memory
  MatrixType X;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  eigenvalues_type s;
  vector_type clam;
  vector_type w;

  // Export images
  Exports exports;

  // Data that can only be written in a thread-safe manner
  // Note that this applies not just to this scratch buffer, but also the output image
  //   (while it would be thread-safe to create a full copy of the output image for each thread
  //   and combine them only at destruction time,
  //   this runs the risk of becoming prohibitively large)
  // Not placing this within a MutexProtexted<> as the image type is still templated
  static std::mutex mutex_aggregator;

  void load_data(Image<F> &image, const std::vector<Kernel::Voxel> &voxels);
};

template <typename F> std::mutex Functor<F>::mutex_aggregator;

template class Functor<float>;
template class Functor<cfloat>;
template class Functor<double>;
template class Functor<cdouble>;

} // namespace MR::Denoise
