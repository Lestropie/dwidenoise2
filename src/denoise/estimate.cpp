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

#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Estimate<F>::Estimate(const Header &header,
                      Image<bool> &mask,
                      std::shared_ptr<Subsample> subsample,
                      std::shared_ptr<Kernel::Base> kernel,
                      std::shared_ptr<Estimator::Base> estimator,
                      Exports &exports)
    : m(header.size(3)),
      mask(mask),
      subsample(subsample),
      kernel(kernel),
      estimator(estimator),
      X(m, kernel->estimated_size()),
      XtX(std::min(m, kernel->estimated_size()), std::min(m, kernel->estimated_size())),
      eig(std::min(m, kernel->estimated_size())),
      s(std::min(m, kernel->estimated_size())),
      exports(exports) {}

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

  // Process voxels in mask only
  if (mask.valid()) {
    assign_pos_of(voxel).to(mask);
    if (!mask.value())
      return;
  }

  // Load list of voxels from which to load data
  neighbourhood = (*kernel)(voxel);
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

  // Centre of patch in realspace
  //   (might be used by estimator)
  const Eigen::Vector3d pos(subsample->patch_centre(voxel));

  // Marchenko-Pastur optimal threshold determination
  threshold = (*estimator)(s, m, n, pos);

  // Store additional output maps if requested
  auto ss_index = subsample->in2ss(voxel);
  if (exports.noise_out.valid()) {
    assign_pos_of(ss_index).to(exports.noise_out);
    exports.noise_out.value() = float(std::sqrt(threshold.sigma2));
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(ss_index).to(exports.rank_input);
    exports.rank_input.value() = r - threshold.cutoff_p;
  }
  if (exports.max_dist.valid()) {
    assign_pos_of(ss_index).to(exports.max_dist);
    exports.max_dist.value() = neighbourhood.max_distance;
  }
  if (exports.voxelcount.valid()) {
    assign_pos_of(ss_index).to(exports.voxelcount);
    exports.voxelcount.value() = n;
  }
  if (exports.patchcount.valid()) {
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    for (const auto &v : neighbourhood.voxels) {
      assign_pos_of(v.index).to(exports.patchcount);
      exports.patchcount.value() = exports.patchcount.value() + 1;
    }
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
