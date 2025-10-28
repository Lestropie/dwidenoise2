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
                      std::shared_ptr<Subsample> subsample,
                      std::shared_ptr<Kernel::Base> kernel,
                      decomp_type decomp,
                      std::shared_ptr<Estimator::Base> estimator,
                      Exports &exports,
                      const ssize_t preconditioner_rank,
                      const bool enable_recon)
    : m(Denoise::num_volumes(image)),
      subsample(subsample),
      kernel(kernel),
      decomp(decomp),
      estimator(estimator),
      preconditioner_rank(preconditioner_rank),
      enable_recon(enable_recon),
      X(m, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? m : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      s(std::min(m, kernel->estimated_size())),
      exports(exports) {
  // If input image is > 4D, should have been preconditioned into 4D
  assert(image.ndim() == 4);
}

template <typename F>
Estimate<F>::Estimate(const Estimate<F> &that)
    : m(that.m),
      subsample(that.subsample),
      kernel(that.kernel),
      decomp(that.decomp),
      estimator(that.estimator),
      preconditioner_rank(that.preconditioner_rank),
      enable_recon(that.enable_recon),
      X(m, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? m : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      s(std::min(m, kernel->estimated_size())),
      exports(that.exports) {
}

template <typename F> void Estimate<F>::operator()(Image<F> &dwi) {

  // There are two options here for looping in the presence of subsampling:
  // 1. Loop over the input image
  //    Skip voxels that don't lie at the centre of a patch
  //    Have to transform input image voxel indices to subsampled image voxel indices for some optional outputs
  // 2. Loop over the subsampled image
  //    In some use cases there may not be any image created that conforms to this voxel grid
  //    Have to transform the subsampled voxel index into an input image voxel index for the centre of the patch
  // Going to go with 1. for now, as for 2. may not have a suitable image over which to loop
  Kernel::Voxel::index_type voxel({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (!subsample->process(voxel))
    return;

  // Load list of voxels from which to import data
  patch = (*kernel)(voxel);
  const ssize_t n = patch.voxels.size();
  const ssize_t r = std::min(m, n);

  // Expand local storage if necessary
  if (n > X.cols()) {
    DEBUG("Expanding data matrix storage from " + str(m) + "x" + str(X.cols()) + " to " + str(m) + "x" + str(n));
    X.resize(m, n);
  }
  if (decomp == decomp_type::SELFADJOINT && r > XtX.cols()) {
    DEBUG("Expanding decomposition matrix storage from " + str(X.rows()) + " to " + str(r));
    XtX.resize(r, r);
  }
  if (r > s.size()) {
    DEBUG("Expanding eigenvalue storage from " + str(s.size()) + " to " + str(r));
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

  load_data(dwi);
  assert(X.leftCols(n).allFinite());

  // Compute Eigendecomposition
  bool successful_decomposition = false;
  switch (decomp) {
  case decomp_type::BDCSVD: {
    SVD.compute(X.leftCols(n), enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly);
    successful_decomposition = SVD.info() == Eigen::Success;
    if (successful_decomposition) {
      // eigenvalues sorted in increasing order:
      s.head(r) = SVD.singularValues().array().reverse().square().template cast<double>();
    }
  } break;
  case decomp_type::SELFADJOINT: {
    if (m <= n)
      XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n) * X.leftCols(n).adjoint();
    else
      XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n).adjoint() * X.leftCols(n);
    eig.compute(XtX.topLeftCorner(r, r), enable_recon ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    successful_decomposition = eig.info() == Eigen::Success;
    if (successful_decomposition) {
      // eigenvalues sorted in increasing order,
      //   additionally clamping any negtive values to zero:
      s.head(r) = eig.eigenvalues().template cast<double>().cwiseMax(0.0);
    }
  } break;
  }

  if (successful_decomposition) {
    // Threshold determination, possibly via Marchenko-Pastur
    threshold = (*estimator)(s.head(r), m, n, preconditioner_rank, patch.centre_realspace);
  } else {
    s.head(r).fill(std::numeric_limits<double>::signaling_NaN());
    threshold = Estimator::Result();
  }

  // Store additional output maps if requested
  auto ss_index = subsample->in2ss(voxel);
  if (exports.noise_out.valid()) {
    assign_pos_of(ss_index).to(exports.noise_out);
    exports.noise_out.value() = bool(threshold)                                //
                                    ? float(std::sqrt(threshold.sigma2))       //
                                    : std::numeric_limits<float>::quiet_NaN(); //
  }
  if (exports.lamplus.valid()) {
    assign_pos_of(ss_index).to(exports.lamplus);
    exports.lamplus.value() = threshold.lamplus;
  }
  if (exports.rank_pcanonzero.valid()) {
    assign_pos_of(ss_index).to(exports.rank_pcanonzero);
    exports.rank_pcanonzero.value() = rank_nonzero(m, n, preconditioner_rank);
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(ss_index).to(exports.rank_input);
    if (!successful_decomposition)
      exports.rank_input.value() = 0;
    else if (bool(threshold))
      exports.rank_input.value() = r - threshold.cutoff_p;
    else
      exports.rank_input.value() = r;
  }
  if (exports.max_dist.valid()) {
    assign_pos_of(ss_index).to(exports.max_dist);
    exports.max_dist.value() = patch.max_distance;
  }
  if (exports.voxelcount.valid()) {
    assign_pos_of(ss_index).to(exports.voxelcount);
    exports.voxelcount.value() = n;
  }
  if (exports.patchcount.valid() || exports.saving_eigenspectra()) {
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    if (exports.patchcount.valid()) {
      for (const auto &v : patch.voxels) {
        assign_pos_of(v.index).to(exports.patchcount);
        exports.patchcount.value() = exports.patchcount.value() + 1;
      }
    }
    if (exports.saving_eigenspectra())
      exports.add_eigenspectrum(s);
  }
}

template <typename F> void Estimate<F>::load_data(Image<F> &image) {
  const Kernel::Voxel::index_type pos({image.index(0), image.index(1), image.index(2)});
  for (ssize_t i = 0; i != patch.voxels.size(); ++i) {
    assign_pos_of(patch.voxels[i].index, 0, 3).to(image);
    X.col(i) = image.row(3);
  }
  assign_pos_of(pos, 0, 3).to(image);
}

template class Estimate<float>;
template class Estimate<cfloat>;
template class Estimate<double>;
template class Estimate<cdouble>;

} // namespace MR::Denoise
