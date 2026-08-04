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

#include "denoise/iterative.h"

#include <algorithm>
#include <cassert>

#include "algo/threaded_copy.h"
#include "algo/threaded_loop.h"
#include "denoise/estimate.h"
#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise::Iterative {

ssize_t resolve_num_partitions(const Iteration &iteration, const ssize_t m_effective, const std::string &context) {
  ssize_t p = iteration.num_partitions;
  if (iteration.max_partition_size.has_value()) {
    const ssize_t maxsize = iteration.max_partition_size.value();
    assert(maxsize >= 1);
    p = (m_effective + maxsize - 1) / maxsize;
  }
  const ssize_t capacity = std::max<ssize_t>(1, m_effective / 2);
  if (p > capacity)
    throw Exception(
        context + " requests " + str(p) + " volume partitions, but the " + str(m_effective) +
        " volumes available to that iteration support at most " + str(capacity) +
        " (each partition must hold at least two volumes). " +
        (iteration.max_partition_size.has_value()
             ? "Raise \"max_partition_size\" (currently " + str(iteration.max_partition_size.value()) + ") or remove it"
             : "Reduce \"partitions\" to " + str(capacity) + " or fewer") +
        " in the schedule");
  return std::max<ssize_t>(1, p);
}

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
              const ssize_t num_partitions,
              std::shared_ptr<const Partitioning> partitioning,
              const std::vector<ssize_t> &volume_group,
              const demean_site_t demean_site) {
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
                                    prmse,
                                    preconditioner.partition_null_rank());
  kernel->set_mask(mask);
  if (preconditioner.noop())
    threaded_copy(input, input_preconditioned);
  else
    preconditioner(input, input_preconditioned, false);
  {
    Estimate<T> func(input_preconditioned,
                     subsample,
                     kernel,
                     decomposition,
                     estimator,
                     exports,
                     preconditioner.partition_null_rank(),
                     false,
                     partitioning,
                     volume_group,
                     demean_site,
                     num_partitions);
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
  // Any exported lambda+ needs the same treatment, but by the SQUARE of the noise level: it is an
  //   eigenvalue (sigma^2 * edge), not a noise level. Done here rather than only in the commands so
  //   that dwi2noise's map, whose final iteration is this one, is on the native scale as
  //   dwidenoise2's is; dwidenoise2's own reconstruction pass does not run through this function
  //   and rescales its exports itself. A no-op for the iterations that export no lambda+ map.
  if (vst_image.valid()) {
    ThreadedLoop(exports.noise_out)
        .run(NoiseMapVSTRescaleFunctor(vst_image, subsample->header(), 1), exports.noise_out);
    if (exports.lamplus.valid()) {
      ThreadedLoop(exports.lamplus).run(NoiseMapVSTRescaleFunctor(vst_image, subsample->header(), 2), exports.lamplus);
    }
  }
}

template void estimate(Image<float> &,
                       Image<float> &,
                       Image<bool> &,
                       Image<float> &,
                       Image<float> &,
                       const Iteration &,
                       const ssize_t,
                       std::shared_ptr<SpatialSubsample>,
                       const decomp_type,
                       std::shared_ptr<Estimator::Base>,
                       const Precondition::Preconditioner<float> &,
                       Exports &,
                       const ssize_t,
                       std::shared_ptr<const Partitioning>,
                       const std::vector<ssize_t> &,
                       const demean_site_t);

template void estimate(Image<double> &,
                       Image<double> &,
                       Image<bool> &,
                       Image<float> &,
                       Image<float> &,
                       const Iteration &,
                       const ssize_t,
                       std::shared_ptr<SpatialSubsample>,
                       const decomp_type,
                       std::shared_ptr<Estimator::Base>,
                       const Precondition::Preconditioner<double> &,
                       Exports &,
                       const ssize_t,
                       std::shared_ptr<const Partitioning>,
                       const std::vector<ssize_t> &,
                       const demean_site_t);

template void estimate(Image<cfloat> &,
                       Image<cfloat> &,
                       Image<bool> &,
                       Image<float> &,
                       Image<float> &,
                       const Iteration &,
                       const ssize_t,
                       std::shared_ptr<SpatialSubsample>,
                       const decomp_type,
                       std::shared_ptr<Estimator::Base>,
                       const Precondition::Preconditioner<cfloat> &,
                       Exports &,
                       const ssize_t,
                       std::shared_ptr<const Partitioning>,
                       const std::vector<ssize_t> &,
                       const demean_site_t);

template void estimate(Image<cdouble> &,
                       Image<cdouble> &,
                       Image<bool> &,
                       Image<float> &,
                       Image<float> &,
                       const Iteration &,
                       const ssize_t,
                       std::shared_ptr<SpatialSubsample>,
                       const decomp_type,
                       std::shared_ptr<Estimator::Base>,
                       const Precondition::Preconditioner<cdouble> &,
                       Exports &,
                       const ssize_t,
                       std::shared_ptr<const Partitioning>,
                       const std::vector<ssize_t> &,
                       const demean_site_t);

} // namespace MR::Denoise::Iterative
