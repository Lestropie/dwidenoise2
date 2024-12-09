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

#include <memory>
#include <mutex>

#include <Eigen/Dense>

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

template <typename F> class Estimate {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;

  Estimate(const Header &header,
           std::shared_ptr<Subsample> subsample,
           std::shared_ptr<Kernel::Base> kernel,
           Image<float> &nonstationarity_image,
           std::shared_ptr<Estimator::Base> estimator,
           Exports &exports);

  void operator()(Image<F> &dwi);

protected:
  const ssize_t m;

  // Denoising configuration
  std::shared_ptr<Subsample> subsample;
  std::shared_ptr<Kernel::Base> kernel;
  std::shared_ptr<Estimator::Base> estimator;

  // Necessary for transform from input voxel locations to nonstationarity image
  std::shared_ptr<Transform> transform;

  // Reusable memory
  Kernel::Data patch;
  Image<float> nonstationarity_image;
  MatrixType X;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  eigenvalues_type s;
  Estimator::Result threshold;

  // Export images
  // Note: One instance created per thread,
  //   so that when possible output image data can be written without mutex-locking
  Exports exports;

  // Some data can only be written in a thread-safe manner
  static std::mutex mutex;

  void load_data(Image<F> &image);
};

template <typename F> std::mutex Estimate<F>::mutex;

template class Estimate<float>;
template class Estimate<cfloat>;
template class Estimate<double>;
template class Estimate<cdouble>;

} // namespace MR::Denoise
