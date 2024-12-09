/* Required Notice: Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;
 * Required Notice: The Florey Institute of Neuroscience and Mental Health.
 *
 * Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     https://polyformproject.org/licenses/noncommercial/1.0.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied.
 * See the License of the specific language
 * governing permissions and limitations under the License.
 */

#include "denoise/recon.h"

#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Recon<F>::Recon(const Header &header,
                std::shared_ptr<Subsample> subsample,
                std::shared_ptr<Kernel::Base> kernel,
                Image<float> &nonstationarity_image,
                std::shared_ptr<Estimator::Base> estimator,
                filter_type filter,
                aggregator_type aggregator,
                Exports &exports)
    : Estimate<F>(header, subsample, kernel, nonstationarity_image, estimator, exports),
      filter(filter),
      aggregator(aggregator),
      // FWHM = 2 x cube root of spacings between kernels
      gaussian_multiplier(-std::log(2.0) /                                                           //
                          Math::pow2(std::cbrt(subsample->get_factors()[0] * header.spacing(0)       //
                                               * subsample->get_factors()[1] * header.spacing(1)     //
                                               * subsample->get_factors()[2] * header.spacing(2)))), //
      w(std::min(Estimate<F>::m, kernel->estimated_size())),
      Xr(Estimate<F>::m, aggregator == aggregator_type::EXCLUSIVE ? 1 : kernel->estimated_size()) {}

template <typename F> void Recon<F>::operator()(Image<F> &dwi, Image<F> &out) {

  if (!Estimate<F>::subsample->process({dwi.index(0), dwi.index(1), dwi.index(2)}))
    return;

  Estimate<F>::operator()(dwi);

  const ssize_t n = Estimate<F>::patch.voxels.size();
  const ssize_t r = std::min(Estimate<F>::m, n);
  const ssize_t q = std::max(Estimate<F>::m, n);
  const double beta = double(r) / double(q);
  const ssize_t in_rank = r - Estimate<F>::threshold.cutoff_p;

  if (r > w.size())
    w.resize(r);
  if (aggregator != aggregator_type::EXCLUSIVE && n > Xr.cols())
    Xr.resize(Estimate<F>::m, n);
#ifndef NDEBUG
  w.fill(std::numeric_limits<default_type>::signaling_NaN());
  Xr.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  // Generate weights vector
  double sum_weights = 0.0;
  ssize_t out_rank = 0;
  if (Estimate<F>::threshold.sigma2 == 0.0 || !std::isfinite(Estimate<F>::threshold.sigma2)) {
    w.head(r).setOnes();
    out_rank = r;
    sum_weights = double(r);
  } else {
    switch (filter) {
    case filter_type::OPTSHRINK: {
      const double transition = 1.0 + std::sqrt(beta);
      for (ssize_t i = 0; i != r; ++i) {
        const double lam = std::max(Estimate<F>::s[i], 0.0) / q;
        // TODO Should this be based on the noise level,
        //   or on the estimated upper bound of the MP distribution?
        // If based on upper bound,
        //   there will be an issue with importing this information from a pre-estimated noise map
        // TODO Unexpected absence of sqrt() here
        // const double y = lam / std::sqrt(Estimate<F>::threshold.sigma2);
        const double y = lam / Estimate<F>::threshold.sigma2;
        double nu = 0.0;
        if (y > transition) {
          nu = std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y;
          ++out_rank;
        }
        w[i] = lam > 0.0 ? (nu / y) : 0.0;
        assert(w[i] >= 0.0 && w[i] <= 1.0);
        sum_weights += w[i];
      }
    } break;
    case filter_type::OPTTHRESH: {
      const std::map<double, double>::const_iterator it = beta2lambdastar.find(beta);
      double lambda_star = 0.0;
      if (it == beta2lambdastar.end()) {
        lambda_star =
            sqrt(2.0 * (beta + 1.0) + ((8.0 * beta) / (beta + 1.0 + std::sqrt(Math::pow2(beta) + 14.0 * beta + 1.0))));
        beta2lambdastar[beta] = lambda_star;
      } else {
        lambda_star = it->second;
      }
      const double tau_star = lambda_star * std::sqrt(q) * std::sqrt(Estimate<F>::threshold.sigma2);
      // TODO Unexpected requisite square applied to q here
      const double threshold = tau_star * Math::pow2(q);
      for (ssize_t i = 0; i != r; ++i) {
        if (Estimate<F>::s[i] >= threshold) {
          w[i] = 1.0;
          ++out_rank;
        } else {
          w[i] = 0.0;
        }
      }
      sum_weights = out_rank;
    } break;
    case filter_type::TRUNCATE:
      out_rank = in_rank;
      w.head(Estimate<F>::threshold.cutoff_p).setZero();
      w.segment(Estimate<F>::threshold.cutoff_p, in_rank).setOnes();
      sum_weights = double(out_rank);
      break;
    default:
      assert(false);
    }
  }
  assert(w.head(r).allFinite());
  assert(std::isfinite(sum_weights));

  // recombine data using only eigenvectors above threshold
  // If only the data computed when this voxel was the centre of the patch
  //   is to be used for synthesis of the output image,
  //   then only that individual column needs to be reconstructed;
  //   if however the result from this patch is to contribute to the synthesized image
  //   for all voxels that were utilised within this patch,
  //   then we need to instead compute the full projection
  switch (aggregator) {
  case aggregator_type::EXCLUSIVE:
    assert(Estimate<F>::patch.centre_index >= 0);
    if (Estimate<F>::m <= n)
      Xr.noalias() =                                               //
          Estimate<F>::eig.eigenvectors() *                        //
          (w.head(r).cast<F>().matrix().asDiagonal() *             //
           (Estimate<F>::eig.eigenvectors().adjoint() *            //
            Estimate<F>::X.col(Estimate<F>::patch.centre_index))); //
    else
      Xr.noalias() =                                                                          //
          Estimate<F>::X.leftCols(n) *                                                        //
          (Estimate<F>::eig.eigenvectors() *                                                  //
           (w.head(r).cast<F>().matrix().asDiagonal() *                                       //
            Estimate<F>::eig.eigenvectors().adjoint().col(Estimate<F>::patch.centre_index))); //
    assert(Xr.allFinite());
    assign_pos_of(dwi).to(out);
    out.row(3) = Xr.col(0);
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
    if (in_rank == r) {
      Xr.leftCols(n).noalias() = Estimate<F>::X.leftCols(n);
    } else if (Estimate<F>::m <= n) {
      Xr.leftCols(n).noalias() =                        //
          Estimate<F>::eig.eigenvectors() *             //
          (w.head(r).cast<F>().matrix().asDiagonal() *  //
           (Estimate<F>::eig.eigenvectors().adjoint() * //
            Estimate<F>::X.leftCols(n)));               //
    } else {
      Xr.leftCols(n).noalias() =                         //
          Estimate<F>::X.leftCols(n) *                   //
          (Estimate<F>::eig.eigenvectors() *             //
           (w.head(r).cast<F>().matrix().asDiagonal() *  //
            Estimate<F>::eig.eigenvectors().adjoint())); //
    }
    assert(Xr.leftCols(n).allFinite());
    // Undo prior within-patch non-stationarity correction
    if (std::isfinite(Estimate<F>::patch.centre_noise)) {
      for (ssize_t i = 0; i != n; ++i)
        if (Estimate<F>::patch.voxels[i].noise_level > 0.0)
          Xr.col(i) *= Estimate<F>::patch.voxels[i].noise_level / Estimate<F>::patch.centre_noise;
    }
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    for (size_t voxel_index = 0; voxel_index != Estimate<F>::patch.voxels.size(); ++voxel_index) {
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(out);
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index).to(Estimate<F>::exports.sum_aggregation);
      double weight = std::numeric_limits<double>::signaling_NaN();
      switch (aggregator) {
      case aggregator_type::EXCLUSIVE:
        assert(false);
        break;
      case aggregator_type::GAUSSIAN:
        weight = std::exp(gaussian_multiplier * Estimate<F>::patch.voxels[voxel_index].sq_distance);
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
      out.row(3) += weight * Xr.col(voxel_index);
      Estimate<F>::exports.sum_aggregation.value() += weight;
      if (Estimate<F>::exports.rank_output.valid()) {
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.rank_output);
        Estimate<F>::exports.rank_output.value() += weight * out_rank;
      }
    }
  } break;
  }

  auto ss_index = Estimate<F>::subsample->in2ss({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (Estimate<F>::exports.sum_optshrink.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.sum_optshrink);
    Estimate<F>::exports.sum_optshrink.value() = sum_weights;
  }
}

} // namespace MR::Denoise
