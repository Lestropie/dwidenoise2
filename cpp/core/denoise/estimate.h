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

#pragma once

// Need to import this first to get relevant precompiler definitions
#include "denoise/denoise.h"

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "denoise/subsample.h"
#include "header.h"
#include "image.h"
#include "transform.h"

namespace MR::Denoise {

// TODO Define a class that encapsulates all results of the decomposition,
//   excluding the denoised version of the input data
class EstimatedPatch {
public:
  EstimatedPatch() : rank_pca(-1), rank_pcanonzero(-1), valid(false) {}
  Kernel::Data patch;
  eigenvalues_type eigenspectrum;
  ssize_t rank_pca;
  ssize_t rank_pcanonzero;
  bool valid;
  Estimator::Result threshold;

  const ssize_t num_voxels() const { return patch.num_voxels(); }
};

template <typename F> class Estimate {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;

  Estimate(const Image<F> &image,
           std::shared_ptr<Kernel::Base> kernel,
           decomp_type decomp,
           std::shared_ptr<Estimator::Base> estimator,
           const ssize_t preconditioner_rank = 0,
           const bool enable_recon = false);

  Estimate(const Estimate &);

  bool operator()(const Kernel::Voxel::index_type &pos, EstimatedPatch &out);

protected:
  Image<F> image;
  const ssize_t values_per_voxel;

  // Denoising configuration
  std::shared_ptr<Kernel::Base> kernel;
  decomp_type decomp;
  std::shared_ptr<Estimator::Base> estimator;
  ssize_t preconditioner_rank;
  bool enable_recon;

  // Reusable memory
  MatrixType X;
  // TODO For both BDCSVD and SelfAdjointEigenSolver,
  //   the template type is MatrixType,
  //   and it doesn't seem to be possible to define an Eigen::Block as this template type;
  //   as such, most likely in both circumstances it is actually constructing a MatrixType from Eigen::Block
  //   in order to construct the decomposition
  // What could conceivably be done instead,
  //   given that these matrices are relatively small
  //   and the number of unique patch sizes is small (though not necessarily one),
  //   would be to construct a std::map<> from patch size to PCA memory;
  //   each processing thread would allocate new memory for new patch sizes not yet encountered by it,
  //   but the total memory consumption should still be relatively small;
  //   note that "X" would be subsumed within such a mechanism also
  Eigen::BDCSVD<MatrixType> SVD;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;

  void load_data(const Kernel::Data &patch);
};

} // namespace MR::Denoise
