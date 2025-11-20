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

#include <limits>
#include <map>
#include <memory>

#include <Eigen/Dense>

#include "denoise/estimate.h"
#include "denoise/estimator/base.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

// TODO Class that can be passed from this functor to a Receiver class
//   in order to avoid locking across threads
// TODO Because this has to store the denoised data,
//   it needs to be templated
template <typename F>
class ReconstructedPatch : public EstimatedPatch {
public:
  ReconstructedPatch() = default;

  typename Estimate<F>::MatrixType Xr;
  ssize_t rank_recon;
  double sum_shrinkage_weights;
  double variance_removed;
  vector_type aggregation_weights;
};

template <typename F> class Recon : public Estimate<F> {

public:
  Recon(const Image<F> &image,
        std::shared_ptr<Kernel::Base> kernel,
        std::shared_ptr<Subsample> subsample,
        const decomp_type decomposition,
        std::shared_ptr<Estimator::Base> estimator,
        filter_type filter,
        aggregator_type aggregator,
        const ssize_t null_rank);

  bool operator()(const Kernel::Voxel::index_type &pos, ReconstructedPatch<F> &out);

protected:
  // Denoising configuration
  filter_type filter;
  aggregator_type aggregator;
  double gaussian_multiplier;

  // Reusable memory
  std::map<double, double> beta2lambdastar;
  vector_type shrinkage_weights;
};

} // namespace MR::Denoise
