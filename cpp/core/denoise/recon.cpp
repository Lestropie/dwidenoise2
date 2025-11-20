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

#include "denoise/denoise.h"
#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Recon<F>::Recon(const Image<F> &image,
                std::shared_ptr<Kernel::Base> kernel,
                std::shared_ptr<Subsample> subsample,
                const decomp_type decomposition,
                std::shared_ptr<Estimator::Base> estimator,
                filter_type filter,
                aggregator_type aggregator,
                const ssize_t preconditioner_rank)
    : Estimate<F>(image, kernel, decomposition, estimator, preconditioner_rank, true),
      filter(filter),
      aggregator(aggregator),
      // FWHM = 2 x cube root of spacings between kernels
      gaussian_multiplier(-std::log(2.0) /                                                          //
                          Math::pow2(std::cbrt(subsample->get_factors()[0] * image.spacing(0)       //
                                               * subsample->get_factors()[1] * image.spacing(1)     //
                                               * subsample->get_factors()[2] * image.spacing(2)))), //
      shrinkage_weights(std::min(Estimate<F>::values_per_voxel, kernel->estimated_size())) { }


template <typename F> bool Recon<F>::operator()(const Kernel::Voxel::index_type &pos, ReconstructedPatch<F> &out) {

  Estimate<F>::operator()(pos, out);

  const ssize_t rz = rank_zero(Estimate<F>::values_per_voxel,
                               out.num_voxels(),
                               Estimate<F>::preconditioner_rank);
  const ssize_t rnz = rank_nonzero(Estimate<F>::values_per_voxel,
                                   out.num_voxels(),
                                   Estimate<F>::preconditioner_rank);
  const ssize_t qnz = dimlong_nonzero(Estimate<F>::values_per_voxel,
                                      out.num_voxels(),
                                      Estimate<F>::preconditioner_rank);
  const double beta = double(rnz) / double(qnz);

  if (out.rank_pca > shrinkage_weights.size()) {
    DEBUG(std::string("Expanding local storage of eigenvalue weights") + //
          " from " + str(shrinkage_weights.size()) +                     //
          " to " + str(out.rank_pca));                                   //
    shrinkage_weights.resize(out.rank_pca);
  }
  if (aggregator != aggregator_type::EXCLUSIVE && out.Xr.cols() < out.num_voxels()) {
    DEBUG(std::string("Expanding local storage of denoised patch") +                  //
          " from " + str(out.Xr.rows()) + "x" + str(out.Xr.cols()) +                  //
          " to " + str(Estimate<F>::values_per_voxel) + "x" + str(out.num_voxels())); //
    out.Xr.resize(Estimate<F>::values_per_voxel, out.num_voxels());
  }
  const ssize_t num_aggregation_weights = aggregator == aggregator_type::EXCLUSIVE ? 1 : out.num_voxels();
  if (out.aggregation_weights.size() < num_aggregation_weights) {

    out.aggregation_weights.resize(num_aggregation_weights);
  }

#ifndef NDEBUG
  shrinkage_weights.fill(std::numeric_limits<default_type>::signaling_NaN());
  out.Xr.fill(std::numeric_limits<default_type>::signaling_NaN());
  out.aggregation_weights.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  // Generate weights vector
  out.sum_shrinkage_weights = 0.0;
  out.rank_recon = 0;
  double sum_variance = 0.0;
  if (static_cast<bool>(out.threshold)) {
    switch (filter) {
    case filter_type::OPTSHRINK: {
      shrinkage_weights.head(rz).setZero();
      const double transition = 1.0 + std::sqrt(beta);
      for (ssize_t i = rz; i != out.rank_pca; ++i) {
        // TODO For non-binary determination of weights for optimal shrinkage,
        //   should the expression be identical between BDCSVD and SelfAdjointEigenSolver?
        //   Or eg. is one equivalent to scaling singular values whereas the other is equivalent to scaling eigenvalues?
        const double lam = out.eigenspectrum[i] / qnz;
        const double y = std::sqrt(lam / out.threshold.sigma2);
        double nu = 0.0;
        if (y > transition) {
          // Occasionally floating-point precision will drive this calculation to fractionally greater than y,
          //   which will erroneously yield a weight fractionally greater than 1.0
          nu = std::min(y, std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y);
          ++out.rank_recon;
        }
        shrinkage_weights[i] = lam > 0.0 ? (nu / y) : 0.0;
        assert(shrinkage_weights[i] >= 0.0 && shrinkage_weights[i] <= 1.0);
        out.sum_shrinkage_weights += shrinkage_weights[i];
        sum_variance += shrinkage_weights[i] * out.eigenspectrum[i];
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
      const double tau_star = lambda_star * std::sqrt(qnz) * std::sqrt(out.threshold.sigma2);
      // TODO Unexpected requisite square applied to qnz here
      const double threshold = tau_star * Math::pow2(qnz);
      shrinkage_weights.head(rz).setZero();
      for (ssize_t i = rz; i != out.rank_pca; ++i) {
        if (out.eigenspectrum[i] >= threshold) {
          shrinkage_weights[i] = 1.0;
          ++out.rank_recon;
          sum_variance += out.eigenspectrum[i];
        } else {
          shrinkage_weights[i] = 0.0;
        }
      }
      out.sum_shrinkage_weights = static_cast<double>(out.rank_recon);
    } break;
    case filter_type::TRUNCATE:
      out.rank_recon = out.rank_pca - out.threshold.cutoff_p;
      shrinkage_weights.head(out.threshold.cutoff_p).setZero();
      shrinkage_weights.segment(out.threshold.cutoff_p, out.rank_recon).setOnes();
      out.sum_shrinkage_weights = static_cast<double>(out.rank_recon);
      sum_variance += shrinkage_weights.head(out.rank_pca).matrix().dot(out.eigenspectrum.head(out.rank_pca).matrix());
      break;
    default:
      assert(false);
    }
    assert(std::isfinite(out.sum_shrinkage_weights));
  } else { // Threshold for this patch is invalid
    // Erring on the conservative side:
    //   If the decomposition fails, or a threshold can't be found,
    //   copy the input data to the output data as-is,
    //   regardless of whether performing overcomplete local PCA
    shrinkage_weights.head(out.rank_pca).setOnes();
    out.rank_recon = out.rank_pca;
    out.sum_shrinkage_weights = out.rank_pca;
    sum_variance = out.eigenspectrum.head(out.rank_pca).sum();
  }
  assert(shrinkage_weights.head(out.rank_pca).allFinite());
  out.variance_removed = 1.0 - sum_variance / out.eigenspectrum.head(out.rank_pca).sum();

  // Recombine data using eigenvalue weights
  // If only the data computed when this voxel was the centre of the patch
  //   is to be used for synthesis of the output image,
  //   then only that individual column needs to be reconstructed;
  //   if however the result from this patch is to contribute to the synthesized image
  //   for all voxels that were utilised within this patch,
  //   then we need to instead compute the full projection
  switch (aggregator) {
  case aggregator_type::EXCLUSIVE: {
    // If doing exclusive aggregation,
    //   cannot be using a kernel that isn't exactly centred at the voxel being denoised
    assert(out.patch.centre_index != -1);
    if (static_cast<bool>(out.threshold)) {
      switch (Estimate<F>::decomp) {
      case decomp_type::BDCSVD: {
        assert(Estimate<F>::SVD.matrixU().allFinite());
        assert(Estimate<F>::SVD.matrixV().allFinite());
        assert(shrinkage_weights.head(out.rank_pca).allFinite());
        assert(Estimate<F>::SVD.singularValues().allFinite());
        // TODO Re-try reconstruction without use of V:
        //   https://github.com/MRtrix3/mrtrix3/pull/2906/commits/eb34f3c57dd460d2b3bd86b9653066be15e916c6
        // It might be that in the case of anything other than EXCLUSIVE,
        //   computing V is no more expensive than doing the full patch reconstruction in its absence,
        //   whereas for EXCLUSIVE since only a small portion of V is used it's worthwhile
        out.Xr.noalias() =                                                               //
            Estimate<F>::SVD.matrixU() *                                                 //
            (shrinkage_weights.head(out.rank_pca).reverse().template cast<F>().array() * //
             Estimate<F>::SVD.singularValues().array()).matrix().asDiagonal() *          //
            Estimate<F>::SVD.matrixV().row(out.patch.centre_index).adjoint();            //
      } break;
      case decomp_type::SELFADJOINT: {
        if (Estimate<F>::values_per_voxel <= out.num_voxels())
          out.Xr.noalias() =                                                                   //
              Estimate<F>::eig.eigenvectors() *                                                //
              (shrinkage_weights.head(out.rank_pca).template cast<F>().matrix().asDiagonal() * //
               (Estimate<F>::eig.eigenvectors().adjoint() *                                    //
                Estimate<F>::X.col(out.patch.centre_index)));                                  //
        else
          out.Xr.noalias() =                                                                    //
              Estimate<F>::X.leftCols(out.num_voxels()) *                                       //
              (Estimate<F>::eig.eigenvectors() *                                                //
               (shrinkage_weights.head(out.rank_pca).template cast<F>().matrix().asDiagonal() * //
                Estimate<F>::eig.eigenvectors().adjoint().col(out.patch.centre_index)));        //
      } break;
      }
      assert(out.Xr.allFinite());
    } else {
      // In the case of -aggregator exclusive,
      //   where a decomposition fails or we can't find a threshold,
      //   we simply copy the input data into the output image
      out.Xr.noalias() = Estimate<F>::X.col(out.patch.centre_index);
    }
  } break;
  default: { // All aggregators other than EXCLUSIVE
    if (static_cast<bool>(out.threshold)) {
      switch (Estimate<F>::decomp) {
      case decomp_type::BDCSVD:
        out.Xr.leftCols(out.num_voxels()).noalias() =                                    //
            Estimate<F>::SVD.matrixU() *                                                 //
            (shrinkage_weights.head(out.rank_pca).reverse().template cast<F>().array() * //
             Estimate<F>::SVD.singularValues().array()).matrix().asDiagonal() *          //
            Estimate<F>::SVD.matrixV().adjoint();                                        //
        break;
      case decomp_type::SELFADJOINT:
        if (Estimate<F>::values_per_voxel <= out.num_voxels()) {
          out.Xr.leftCols(out.num_voxels()).noalias() =                                        //
              Estimate<F>::eig.eigenvectors() *                                                //
              (shrinkage_weights.head(out.rank_pca).template cast<F>().matrix().asDiagonal() * //
               (Estimate<F>::eig.eigenvectors().adjoint() *                                    //
                Estimate<F>::X.leftCols(out.num_voxels())));                                   //
        } else {
          out.Xr.leftCols(out.num_voxels()).noalias() =                                         //
              Estimate<F>::X.leftCols(out.num_voxels()) *                                       //
              (Estimate<F>::eig.eigenvectors() *                                                //
               (shrinkage_weights.head(out.rank_pca).template cast<F>().matrix().asDiagonal() * //
                Estimate<F>::eig.eigenvectors().adjoint()));                                    //
        }
        break;
      }
    } else {
      out.Xr.leftCols(out.num_voxels()).noalias() = Estimate<F>::X.leftCols(out.num_voxels());
    }
    assert(out.Xr.leftCols(out.num_voxels()).allFinite());
    // Undo prior within-patch variance-stabilising transform
    if (std::isfinite(out.patch.centre_noise)) {
      for (ssize_t i = 0; i != out.num_voxels(); ++i)
        if (out.patch.voxels[i].noise_level > 0.0)
          out.Xr.col(i) *= out.patch.voxels[i].noise_level / out.patch.centre_noise;
    }
  } break;
  }

  switch (aggregator) {
  case aggregator_type::EXCLUSIVE:
    out.aggregation_weights = vector_type::Ones(1);
    break;
  case aggregator_type::GAUSSIAN:
    for (ssize_t voxel_index = 0; voxel_index != out.num_voxels(); ++voxel_index)
      out.aggregation_weights[voxel_index] = std::exp(gaussian_multiplier * out.patch.voxels[voxel_index].sq_distance);
    break;
  case aggregator_type::INVL0:
    out.aggregation_weights.head(out.num_voxels()).setConstant(1.0 / (1 + out.rank_recon));
    break;
  case aggregator_type::RANK:
    out.aggregation_weights.head(out.num_voxels()).setConstant(out.rank_recon);
    break;
  case aggregator_type::UNIFORM:
    out.aggregation_weights.head(out.num_voxels()).setOnes();
    break;
  }

  return true;
}

template class Recon<float>;
template class Recon<cfloat>;
template class Recon<double>;
template class Recon<cdouble>;

} // namespace MR::Denoise
