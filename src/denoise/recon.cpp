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

#include "denoise/recon.h"

#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Recon<F>::Recon(const Header &header,
                Image<bool> &mask,
                std::shared_ptr<Kernel::Base> kernel,
                std::shared_ptr<Estimator::Base> estimator,
                filter_type filter,
                aggregator_type aggregator,
                Exports &exports)
    : Estimate<F>(header, mask, kernel, estimator, exports),
      filter(filter),
      aggregator(aggregator),
      // FWHM = 2 x cube root of voxel spacings
      gaussian_multiplier(-std::log(2.0) /
                          Math::pow2(std::cbrt(header.spacing(0) * header.spacing(1) * header.spacing(2)))),
      w(std::min(Estimate<F>::m, kernel->estimated_size())) {}

template <typename F> void Recon<F>::operator()(Image<F> &dwi, Image<F> &out) {

  Estimate<F> (*this)(dwi);

  const ssize_t n = Estimate<F>::neighbourhood.voxels.size();
  const ssize_t r = std::min(Estimate<F>::m, n);
  const ssize_t q = std::max(Estimate<F>::m, n);
  const ssize_t in_rank = r - Estimate<F>::threshold.cutoff_p;

  if (r > w.size())
    w.resize(r);
#ifndef NDEBUG
  w.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  // Generate weights vector
  double sum_weights = 0.0;
  ssize_t out_rank = 0;
  switch (filter) {
  case filter_type::TRUNCATE:
    out_rank = in_rank;
    w.head(Estimate<F>::threshold.cutoff_p).setZero();
    w.segment(Estimate<F>::threshold.cutoff_p, in_rank).setOnes();
    sum_weights = double(out_rank);
    break;
  case filter_type::FROBENIUS: {
    const double beta = r / q;
    const double transition = 1.0 + std::sqrt(beta);
    double clam = 0.0;
    for (ssize_t i = 0; i != r; ++i) {
      const double lam = std::max(Estimate<F>::s[i], 0.0) / q;
      clam += lam;
      const double y = clam / (Estimate<F>::threshold.sigma2 * (i + 1));
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
  // TODO Use a new data member local to Recon<>
  switch (aggregator) {
  case aggregator_type::EXCLUSIVE:
    if (Estimate<F>::m <= n)
      Estimate<F>::X.col(Estimate<F>::neighbourhood.centre_index) =
          Estimate<F>::eig.eigenvectors() *
          (w.head(r).cast<F>().matrix().asDiagonal() *
           (Estimate<F>::eig.eigenvectors().adjoint() * Estimate<F>::X.col(Estimate<F>::neighbourhood.centre_index)));
    else
      Estimate<F>::X.col(Estimate<F>::neighbourhood.centre_index) =
          Estimate<F>::X.leftCols(n) *
          (Estimate<F>::eig.eigenvectors() *
           (w.head(r).cast<F>().matrix().asDiagonal() *
            Estimate<F>::eig.eigenvectors().adjoint().col(Estimate<F>::neighbourhood.centre_index)));
    assign_pos_of(dwi).to(out);
    out.row(3) = Estimate<F>::X.col(Estimate<F>::neighbourhood.centre_index);
    if (Estimate<F>::exports.sum_aggregation.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.sum_aggregation);
      Estimate<F>::exports.sum_aggregation.value() = 1.0;
    }
    if (Estimate<F>::exports.rank_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.rank_output);
      Estimate<F>::exports.rank_output.value() = out_rank;
    }
    break;
  default: {
    if (Estimate<F>::m <= n)
      Estimate<F>::X = Estimate<F>::eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() *
                                                          (Estimate<F>::eig.eigenvectors().adjoint() * Estimate<F>::X));
    else
      Estimate<F>::X.leftCols(n) =
          Estimate<F>::X.leftCols(n) * (Estimate<F>::eig.eigenvectors() * (w.head(r).cast<F>().matrix().asDiagonal() *
                                                                           Estimate<F>::eig.eigenvectors().adjoint()));
    std::lock_guard<std::mutex> lock(mutex_aggregator);
    for (size_t voxel_index = 0; voxel_index != Estimate<F>::neighbourhood.voxels.size(); ++voxel_index) {
      assign_pos_of(Estimate<F>::neighbourhood.voxels[voxel_index].index, 0, 3).to(out);
      assign_pos_of(Estimate<F>::neighbourhood.voxels[voxel_index].index).to(Estimate<F>::exports.sum_aggregation);
      double weight = std::numeric_limits<double>::signaling_NaN();
      switch (aggregator) {
      case aggregator_type::EXCLUSIVE:
        assert(false);
        break;
      case aggregator_type::GAUSSIAN:
        weight = std::exp(gaussian_multiplier * Estimate<F>::neighbourhood.voxels[voxel_index].sq_distance);
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
      out.row(3) += weight * Estimate<F>::X.col(voxel_index);
      Estimate<F>::exports.sum_aggregation.value() += weight;
      if (Estimate<F>::exports.rank_output.valid()) {
        assign_pos_of(Estimate<F>::neighbourhood.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.rank_output);
        Estimate<F>::exports.rank_output.value() += weight * out_rank;
      }
    }
  } break;
  }

  if (Estimate<F>::exports.sum_optshrink.valid()) {
    assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.sum_optshrink);
    Estimate<F>::exports.sum_optshrink.value() = sum_weights;
  }
}

} // namespace MR::Denoise
