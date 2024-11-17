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

#include "denoise/estimate.h"

#include <limits>

#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Estimate<F>::Estimate(const Header &header,
                      Image<bool> &mask,
                      std::shared_ptr<Kernel::Base> kernel,
                      std::shared_ptr<Estimator::Base> estimator,
                      Exports &exports)
    : m(header.size(3)),
      mask(mask),
      kernel(kernel),
      estimator(estimator),
      X(m, kernel->estimated_size()),
      XtX(std::min(m, kernel->estimated_size()), std::min(m, kernel->estimated_size())),
      eig(std::min(m, kernel->estimated_size())),
      s(std::min(m, kernel->estimated_size())),
      exports(exports) {}

template <typename F> void Estimate<F>::operator()(Image<F> &dwi) {

  // Process voxels in mask only
  if (mask.valid()) {
    assign_pos_of(dwi, 0, 3).to(mask);
    if (!mask.value())
      return;
  }

  // Load list of voxels from which to load data
  neighbourhood = (*kernel)({dwi.index(0), dwi.index(1), dwi.index(2)});
  const ssize_t n = neighbourhood.voxels.size();
  const ssize_t r = std::min(m, n);
  const ssize_t q = std::max(m, n);

  // Expand local storage if necessary
  if (n > X.cols()) {
    DEBUG("Expanding data matrix storage from " + str(m) + "x" + str(X.cols()) + " to " + str(m) + "x" + str(n));
    X.resize(m, n);
  }
  if (r > XtX.cols()) {
    DEBUG("Expanding decomposition matrix storage from " + str(X.rows()) + " to " + str(r));
    XtX.resize(r, r);
    s.resize(r);
  }

  // Fill matrices with NaN when in debug mode;
  //   make sure results from one voxel are not creeping into another
  //   due to use of block oberations to prevent memory re-allocation
  //   in the presence of variation in kernel sizes
#ifndef NDEBUG
  X.fill(std::numeric_limits<F>::signaling_NaN());
  XtX.fill(std::numeric_limits<F>::signaling_NaN());
  s.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  load_data(dwi, neighbourhood.voxels);

  // Compute Eigendecomposition:
  if (m <= n)
    XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n) * X.leftCols(n).adjoint();
  else
    XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n).adjoint() * X.leftCols(n);
  eig.compute(XtX.topLeftCorner(r, r));
  // eigenvalues sorted in increasing order:
  s.head(r) = eig.eigenvalues().template cast<double>();

  // Marchenko-Pastur optimal threshold determination
  threshold = (*estimator)(s, m, n);
  const ssize_t in_rank = r - threshold.cutoff_p;

  // Store additional output maps if requested
  if (exports.noise_out.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.noise_out);
    exports.noise_out.value() = float(std::sqrt(threshold.sigma2));
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.rank_input);
    exports.rank_input.value() = in_rank;
  }
  if (exports.max_dist.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.max_dist);
    exports.max_dist.value() = neighbourhood.max_distance;
  }
  if (exports.voxelcount.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.voxelcount);
    exports.voxelcount.value() = n;
  }
}

template <typename F> void Estimate<F>::load_data(Image<F> &image, const std::vector<Kernel::Voxel> &voxels) {
  const Kernel::Voxel::index_type pos({image.index(0), image.index(1), image.index(2)});
  for (ssize_t i = 0; i != voxels.size(); ++i) {
    assign_pos_of(voxels[i].index, 0, 3).to(image);
    X.col(i) = image.row(3);
  }
  assign_pos_of(pos, 0, 3).to(image);
}

} // namespace MR::Denoise
