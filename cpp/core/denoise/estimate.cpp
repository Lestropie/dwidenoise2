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

#include "denoise/estimate.h"

#include <limits>

#include "interp/cubic.h"
#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Estimate<F>::Estimate(const Image<F> &image,
                      std::shared_ptr<Kernel::Base> kernel,
                      decomp_type decomp,
                      std::shared_ptr<Estimator::Base> estimator,
                      const ssize_t preconditioner_rank,
                      const bool enable_recon)
    : image(image),
      values_per_voxel(Denoise::num_volumes(image)),
      kernel(kernel),
      decomp(decomp),
      estimator(estimator),
      preconditioner_rank(preconditioner_rank),
      enable_recon(enable_recon),
      X(values_per_voxel, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? values_per_voxel : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0) {
  // If input image is > 4D, should have been preconditioned into 4D
  assert(image.ndim() == 4);
}

template <typename F>
Estimate<F>::Estimate(const Estimate<F> &that)
    : image(that.image),
      values_per_voxel(that.values_per_voxel),
      kernel(that.kernel),
      decomp(that.decomp),
      estimator(that.estimator),
      preconditioner_rank(that.preconditioner_rank),
      enable_recon(that.enable_recon),
      X(values_per_voxel, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? values_per_voxel : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(values_per_voxel, kernel->estimated_size()) : 0) {
}

template <typename F> bool Estimate<F>::operator()(const Kernel::Voxel::index_type &pos,
                                                   EstimatedPatch &out) {

  out.patch = (*kernel)(pos);
  out.rank_pca = std::min(values_per_voxel, out.num_voxels());

  // Expand local storage if necessary
  if (out.num_voxels() > X.cols()) {
    DEBUG(std::string("Expanding data matrix storage") +                 //
          " from " + str(values_per_voxel) + "x" + str(X.cols()) +       //
          " to " + str(values_per_voxel) + "x" + str(out.num_voxels())); //
    X.resize(values_per_voxel, out.num_voxels());
  }
  if (decomp == decomp_type::SELFADJOINT && out.rank_pca > XtX.cols()) {
    DEBUG("Expanding decomposition matrix storage from " + str(X.rows()) + " to " + str(out.rank_pca));
    XtX.resize(out.rank_pca, out.rank_pca);
  }
  if (out.rank_pca > out.eigenspectrum.size()) {
    DEBUG("Expanding eigenspectrum storage from " + str(out.eigenspectrum.size()) + " to " + str(out.rank_pca));
    out.eigenspectrum.resize(out.rank_pca);
  }

  // Fill matrices with NaN when in debug mode;
  //   make sure results from one voxel are not creeping into another
  //   due to use of block oberations to prevent memory re-allocation
  //   in the presence of variation in kernel sizes
#ifndef NDEBUG
  X.fill(std::numeric_limits<F>::signaling_NaN());
  XtX.fill(std::numeric_limits<F>::signaling_NaN());
  out.eigenspectrum.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  load_data(out.patch);
  // TODO Investigate possible persistance of non-finite values in sample data
  assert(X.leftCols(out.num_voxels()).allFinite());

  // Compute Eigendecomposition
  out.valid = false;
  switch (decomp) {
  case decomp_type::BDCSVD: {
    SVD.compute(X.leftCols(out.num_voxels()),                                                         //
                enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly); //
    if ((out.valid = SVD.info() == Eigen::Success))
      out.eigenspectrum.head(out.rank_pca) = SVD.singularValues().array().reverse().square().template cast<double>();
  } break;
  case decomp_type::SELFADJOINT: {
    if (values_per_voxel <= out.num_voxels())
      XtX.topLeftCorner(out.rank_pca, out.rank_pca).template triangularView<Eigen::Lower>() =
        X.leftCols(out.num_voxels()) * X.leftCols(out.num_voxels()).adjoint();
    else
      XtX.topLeftCorner(out.rank_pca, out.rank_pca).template triangularView<Eigen::Lower>() =
        X.leftCols(out.num_voxels()).adjoint() * X.leftCols(out.num_voxels());
    eig.compute(XtX.topLeftCorner(out.rank_pca, out.rank_pca),                       //
                enable_recon ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly); //
    if ((out.valid = eig.info() == Eigen::Success))
      out.eigenspectrum.head(out.rank_pca) = eig.eigenvalues().template cast<double>().cwiseMax(0.0);
  } break;
  }

  if (out.valid) {
    // Threshold determination, possibly via Marchenko-Pastur
    out.threshold = (*estimator)(out.eigenspectrum.head(out.rank_pca),
                                 values_per_voxel,
                                 out.num_voxels(),
                                 preconditioner_rank,
                                 out.patch.centre_realspace);
    out.rank_pcanonzero = Denoise::rank_nonzero(values_per_voxel, out.num_voxels(), preconditioner_rank);
  } else {
    out.eigenspectrum.head(out.rank_pca).fill(std::numeric_limits<double>::signaling_NaN());
    out.threshold = Estimator::Result();
    out.rank_pcanonzero = 0;
  }

  // No exports here; that will be dealt with by the Receiver class
  return true;
}

template <typename F> void Estimate<F>::load_data(const Kernel::Data &patch) {
  for (ssize_t i = 0; i != patch.voxels.size(); ++i) {
    assign_pos_of(patch.voxels[i].index, 0, 3).to(image);
    X.col(i) = image.row(3);
  }
}

template class Estimate<float>;
template class Estimate<cfloat>;
template class Estimate<double>;
template class Estimate<cdouble>;

} // namespace MR::Denoise
