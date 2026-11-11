/* Required Notice: Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;
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
#include <memory>

#include <Eigen/Dense>

#include "denoise/estimate.h"
#include "denoise/estimator/base.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "denoise/spectrum_filter.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

template <typename F> class Recon : public Estimate<F> {

public:
  Recon(const Image<F> &image,
        std::shared_ptr<SpatialSubsample> subsample,
        std::shared_ptr<Kernel::Base> kernel,
        const decomp_type decomposition,
        std::shared_ptr<Estimator::Base> estimator,
        filter_type filter,
        aggregator_type aggregator,
        const default_type aggregator_fwhm,
        Exports &exports,
        const ssize_t null_rank,
        std::shared_ptr<const Partitioning> level_partitioning = nullptr,
        std::vector<ssize_t> volume_group = {},
        const demean_site_t demean_site = demean_site_t::NONE,
        const ssize_t kernel_num_partitions = 1);

  void operator()(Image<F> &dwi, Image<F> &out);

protected:
  // Denoising configuration. Everything specific to the eigenspectrum filter -- the weights, the
  //   spectral moments they imply, and the optimal-hard-threshold cache -- belongs to the filter
  //   rather than to this class, which is left responsible for spatial aggregation alone.
  SpectrumFilter filter;
  aggregator_type aggregator;
  double gaussian_multiplier;

  // Reusable memory
  vector_type w;
  typename Estimate<F>::MatrixType Xr;
};

} // namespace MR::Denoise
