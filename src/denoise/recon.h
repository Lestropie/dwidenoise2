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

#include "denoise/estimate.h"
#include "denoise/estimator/base.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

template <typename F> class Recon : public Estimate<F> {

public:
  Recon(const Header &header,
        Image<bool> &mask,
        std::shared_ptr<Kernel::Base> kernel,
        std::shared_ptr<Estimator::Base> estimator,
        filter_type filter,
        aggregator_type aggregator,
        Exports &exports);

  void operator()(Image<F> &dwi, Image<F> &out);

protected:
  // Denoising configuration
  filter_type filter;
  aggregator_type aggregator;
  double gaussian_multiplier;

  // Reusable memory
  vector_type w;
  typename Estimate<F>::MatrixType Xr;

  // Some data can only be written in a thread-safe manner
  static std::mutex mutex_aggregator;
};

template <typename F> std::mutex Recon<F>::mutex_aggregator;

template class Recon<float>;
template class Recon<cfloat>;
template class Recon<double>;
template class Recon<cdouble>;

} // namespace MR::Denoise
