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

#include "denoise/functor.h"

#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Functor<F>::Functor(const Header &header,
                    Image<bool> &mask,
                    std::shared_ptr<Kernel::Base> kernel,
                    std::shared_ptr<Estimator::Base> estimator,
                    filter_type filter,
                    aggregator_type aggregator,
                    Exports &exports)
    : m(header.size(3)),
      mask(mask),
      kernel(kernel),
      estimator(estimator),
      filter(filter),
      aggregator(aggregator),
      // FWHM = 2 x cube root of voxel spacings
      gaussian_multiplier(-std::log(2.0) /
                          Math::pow2(std::cbrt(header.spacing(0) * header.spacing(1) * header.spacing(2)))),
      X(m, kernel->estimated_size()),
      XtX(std::min(m, kernel->estimated_size()), std::min(m, kernel->estimated_size())),
      eig(std::min(m, kernel->estimated_size())),
      s(std::min(m, kernel->estimated_size())),
      clam(std::min(m, kernel->estimated_size())),
      w(std::min(m, kernel->estimated_size())),
      exports(exports) {}

template <typename F> void Functor<F>::operator()(Image<F> &dwi, Image<F> &out) {
  // Process voxels in mask only
  if (mask.valid()) {
    assign_pos_of(dwi, 0, 3).to(mask);
    if (!mask.value())
      return;
  }

  // Load list of voxels from which to load data
  const Kernel::Data neighbourhood = (*kernel)({dwi.index(0), dwi.index(1), dwi.index(2)});
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
    clam.resize(r);
    w.resize(r);
  }

  // Fill matrices with NaN when in debug mode;
  //   make sure results from one voxel are not creeping into another
  //   due to use of block oberations to prevent memory re-allocation
  //   in the presence of variation in kernel sizes
#ifndef NDEBUG
  X.fill(std::numeric_limits<F>::signaling_NaN());
  XtX.fill(std::numeric_limits<F>::signaling_NaN());
  s.fill(std::numeric_limits<default_type>::signaling_NaN());
  clam.fill(std::numeric_limits<default_type>::signaling_NaN());
  w.fill(std::numeric_limits<default_type>::signaling_NaN());
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
  const Estimator::Result threshold = (*estimator)(s, m, n);

  // Generate weights vector
  double sum_weights = 0.0;
  ssize_t out_rank = 0;
  switch (filter) {
  case filter_type::TRUNCATE:
    out_rank = r - threshold.cutoff_p;
    w.head(threshold.cutoff_p).setZero();
    w.segment(threshold.cutoff_p, r - threshold.cutoff_p).setOnes();
    sum_weights = double(out_rank);
    break;
  case filter_type::FROBENIUS: {
    const double beta = r / q;
    const double transition = 1.0 + std::sqrt(beta);
    double clam = 0.0;
    for (ssize_t i = 0; i != r; ++i) {
      const double lam = std::max(s[i], 0.0) / q;
      clam += lam;
      const double y = clam / (threshold.sigma2 * (i + 1));
      double nu = 0.0;
      if (y > transition) {
        nu = std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y;
        ++out_rank;
      }
      w[i] = nu / y;
      sum_weights += w[i];
    }
  } break;
  default:
    assert(false);
  }

  // recombine data using only eigenvectors above threshold
  // If only the data computed when this voxel was the centre of the patch
  //   is to be used for synthesis of the output image,
  //   then only that individual column needs to be reconstructed;
  //   if however the result from this patch is to contribute to the synthesized image
  //   for all voxels that were utilised within this patch,
  //   then we need to instead compute the full projection
  switch (aggregator) {
  case aggregator_type::EXCLUSIVE:
    if (m <= n)
      X.col(neighbourhood.centre_index) =
          eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() *
                                (eig.eigenvectors().adjoint() * X.col(neighbourhood.centre_index)));
    else
      X.col(neighbourhood.centre_index) =
          X.leftCols(n) * (eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() *
                                                 eig.eigenvectors().adjoint().col(neighbourhood.centre_index)));
    assign_pos_of(dwi).to(out);
    out.row(3) = X.col(neighbourhood.centre_index);
    if (exports.sum_aggregation.valid()) {
      assign_pos_of(dwi, 0, 3).to(exports.sum_aggregation);
      exports.sum_aggregation.value() = 1.0;
    }
    if (exports.rank_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(exports.rank_output);
      exports.rank_output.value() = out_rank;
    }
    break;
  default: {
    if (m <= n)
      X = eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() * (eig.eigenvectors().adjoint() * X));
    else
      X.leftCols(n) = X.leftCols(n) *
                      (eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() * eig.eigenvectors().adjoint()));
    std::lock_guard<std::mutex> lock(mutex_aggregator);
    for (size_t voxel_index = 0; voxel_index != neighbourhood.voxels.size(); ++voxel_index) {
      assign_pos_of(neighbourhood.voxels[voxel_index].index, 0, 3).to(out);
      assign_pos_of(neighbourhood.voxels[voxel_index].index).to(exports.sum_aggregation);
      double weight = std::numeric_limits<double>::signaling_NaN();
      switch (aggregator) {
      case aggregator_type::EXCLUSIVE:
        assert(false);
        break;
      case aggregator_type::GAUSSIAN:
        weight = std::exp(gaussian_multiplier * neighbourhood.voxels[voxel_index].sq_distance);
        break;
      case aggregator_type::INVL0:
        weight = 1.0 / (1 + out_rank);
        break;
      case aggregator_type::RANK:
        weight = out_rank;
        break;
      case aggregator_type::UNIFORM:
        weight = 1.0;
        break;
      }
      out.row(3) += weight * X.col(voxel_index);
      exports.sum_aggregation.value() += weight;
      if (exports.rank_output.valid()) {
        assign_pos_of(neighbourhood.voxels[voxel_index].index, 0, 3).to(exports.rank_output);
        exports.rank_output.value() += weight * out_rank;
      }
    }
  } break;
  }

  // Store additional output maps if requested
  if (exports.noise_out.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.noise_out);
    exports.noise_out.value() = float(std::sqrt(threshold.sigma2));
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.rank_input);
    exports.rank_input.value() = out_rank;
  }
  if (exports.sum_optshrink.valid()) {
    assign_pos_of(dwi, 0, 3).to(exports.sum_optshrink);
    exports.sum_optshrink.value() = sum_weights;
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

template <typename F> void Functor<F>::load_data(Image<F> &image, const std::vector<Kernel::Voxel> &voxels) {
  const Kernel::Voxel::index_type pos({image.index(0), image.index(1), image.index(2)});
  for (ssize_t i = 0; i != voxels.size(); ++i) {
    assign_pos_of(voxels[i].index, 0, 3).to(image);
    X.col(i) = image.row(3);
  }
  assign_pos_of(pos, 0, 3).to(image);
}

} // namespace MR::Denoise
