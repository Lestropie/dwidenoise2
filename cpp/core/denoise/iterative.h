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

#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <vector>

#include "algo/threaded_copy.h"
#include "algo/threaded_loop.h"
#include "denoise/estimate.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/estimator.h"
#include "denoise/exports.h"
#include "denoise/kernel/kernel.h"
#include "denoise/partition.h"
#include "denoise/precondition/preconditioner.h"
#include "denoise/spatial_subsample.h"
#include "filter/smooth.h"
#include "image.h"
#include "interp/cubic.h"
#include "types.h"

namespace MR::Denoise::Iterative {

struct Iteration {
  std::array<ssize_t, 3> spatial_subsample_ratios;
  // Per-iteration kernel sizing (type + free parameter); replaces the former kernel_size_multiplier.
  Kernel::KernelSpec kernel;
  noise_smooth_type smooth_noiseout;
  // Fraction of volumes (along the supra-spatial axes) used for noise level estimation
  //   in this iteration; 1.0 uses all volumes. Sub-sampling is stratified by demeaning
  //   group and is owned entirely by the preconditioner (Preconditioner::set_temporal_subsample).
  default_type temporal_subsample = 1.0;
  // Whether the noise level estimate is (re)computed within this iteration.
  //   Must be true for any non-final iteration. For the final iteration the default is
  //   false (the dummy Estimator::Unity is used, carrying the prior estimate through);
  //   a custom schedule may set it true to re-estimate in the final iteration.
  //   Unset here; resolved to a concrete value per command after the schedule is loaded.
  std::optional<bool> update_noise;
  // Volume partitioning for this iteration (large-series PCA acceleration). Each PCA patch's
  //   volumes are split into P partitions; an independent decomposition is performed per
  //   partition and the eigenspectra are pooled, reducing PCA cost ~P^2. At most one of the
  //   following may be set to an "active" value on a given schedule row (mutually exclusive):
  //   - num_partitions > 1            : an explicit partition count P;
  //   - max_partition_size.has_value(): P = ceil(m'/max_partition_size), derived once the
  //                                       effective volume count m' for the iteration is known.
  //   num_partitions == 1 with max_partition_size unset ⇒ no partitioning (P=1; the default,
  //   reproducing the non-partitioned behaviour exactly).
  ssize_t num_partitions = 1;
  std::optional<ssize_t> max_partition_size;
};

// Resolve the number of partitions P for an iteration given the effective volume count
//   m_effective (the preconditioned/sub-sampled volume count actually decomposed). P is
//   clamped to [1, m_effective]; with neither partition control active it is 1.
inline ssize_t resolve_num_partitions(const Iteration &iteration, const ssize_t m_effective) {
  ssize_t p = iteration.num_partitions;
  if (iteration.max_partition_size.has_value()) {
    const ssize_t maxsize = iteration.max_partition_size.value();
    assert(maxsize >= 1);
    p = (m_effective + maxsize - 1) / maxsize;
  }
  return std::max<ssize_t>(1, std::min(p, m_effective));
}

// Internal function covering as much as possible for iterative implementation.
//   num_partitions / partitioning describe the volume partitioning for this iteration (P == 1
//   and a null assignment ⇒ no partitioning). volume_group carries the per-volume demeaning-group
//   labels used for the per-partition demeaning performed within Estimate (empty ⇒ no demeaning).
template <typename T>
void estimate(Image<T> &input,
              Image<T> &input_preconditioned,
              Image<bool> &mask,
              Image<float> &vst_image,
              Image<float> &rank_per_mm_image,
              const Iteration &config,
              const ssize_t iter,
              std::shared_ptr<SpatialSubsample> subsample,
              const decomp_type decomposition,
              std::shared_ptr<Estimator::Base> estimator,
              const Precondition::Preconditioner<T> &preconditioner,
              Exports &exports,
              const ssize_t num_partitions = 1,
              std::shared_ptr<const Partitioning> partitioning = nullptr,
              const std::vector<ssize_t> &volume_group = {}) {
  // Size the kernel from the preconditioned data, whose volume count is m' (reduced under
  //   temporal sub-sampling); this keeps the Casorati matrix aspect ratio consistent with the
  //   number of volumes actually decomposed. With partitioning the kernel is sized from the
  //   smallest partition (m'/P). The full volume count is passed for the shape warning only.
  //   make_kernel reads only the header, so it is valid to call before the preconditioned data
  //   are filled below.
  // The per-iteration kernel type and free parameter come from this schedule row (config.kernel):
  //   typically an aspect-ratio kernel for the first iteration (no rank map yet) and an RMSE-
  //   tolerance kernel thereafter. An RMSE kernel is supplied the active estimator's predicted-RMSE
  //   model so it can grow the patch until the noise level is estimated precisely enough.
  Kernel::predicted_rmse_func prmse;
  if (config.kernel.type == Kernel::kernel_spec_type::RMSE)
    prmse = [estimator](ssize_t mm, ssize_t nn, double rr) { return estimator->predicted_rmse(mm, nn, rr); };
  auto kernel = Kernel::make_kernel(input_preconditioned,
                                    subsample->get_factors(),
                                    config.kernel,
                                    rank_per_mm_image,
                                    Denoise::num_volumes(input),
                                    num_partitions,
                                    prmse);
  kernel->set_mask(mask);
  if (preconditioner.noop())
    threaded_copy(input, input_preconditioned);
  else
    preconditioner(input, input_preconditioned, false);
  {
    Estimate<T> func(input_preconditioned, subsample, kernel, decomposition, estimator, exports,
                     preconditioner.null_rank(), false, partitioning, volume_group, num_partitions);
    ThreadedLoop("MPPCA noise level estimation", input_preconditioned, 0, 3).run(func, input_preconditioned);
    func.report_warnings();
  }
  // If a VST was applied to the input data for this iteration,
  //   need to remove its effect from the estimated noise map.
  // Under the nonlinear variance-stabilising transform the stabilised data are
  //   homoscedastic with variance ~ 1 by construction, so the estimate on the
  //   stabilised data is a unit-less correction factor; multiplying by the noise
  //   level used for stabilisation (sigma_{k-1}) yields the first-order refinement
  //   sigma_k = sigma_{k-1} * (post-VST sigma) (vst_plan.md section 5).
  if (vst_image.valid()) {
    Interp::Cubic<Image<float>> vst_interp(vst_image);
    const Transform transform(subsample->header());
    for (auto l = Loop(exports.noise_out)(exports.noise_out); l; ++l) {
      vst_interp.scanner(transform.voxel2scanner * Eigen::Vector3d({double(exports.noise_out.index(0)),
                                                                    double(exports.noise_out.index(1)),
                                                                    double(exports.noise_out.index(2))}));
      exports.noise_out.value() *= vst_interp.value();
    }
  }
}

} // namespace MR::Denoise::Iterative
