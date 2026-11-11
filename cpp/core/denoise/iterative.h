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

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/exports.h"
#include "denoise/kernel/kernel.h"
#include "denoise/partition.h"
#include "denoise/precondition/preconditioner.h"
#include "denoise/spatial_subsample.h"
#include "image.h"
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
//   m_effective (the preconditioned/sub-sampled volume count actually decomposed). With neither
//   partition control active P is 1.
// A partition must hold at least two volumes to have a Casorati matrix worth decomposing, so
//   the realisable partition count is bounded by m_effective/2. A request beyond that bound is
//   refused rather than quietly reduced: silently substituting a different partition count would
//   change the decomposition actually performed (and hence the noise estimate and the cost model
//   the user was reaching for) while reporting success, so the run terminates instead.
//   "context" names the schedule row responsible.
// The max_partition_size assertion is an implementation invariant: the schedule parser rejects
//   any value below 1, so it cannot be reached from the command line.
ssize_t resolve_num_partitions(const Iteration &iteration, const ssize_t m_effective, const std::string &context);

// Internal function covering as much as possible for iterative implementation.
//   num_partitions / partitioning describe the volume partitioning for this iteration (P == 1
//   and a null assignment ⇒ no partitioning). volume_group carries the per-volume demeaning-group
//   labels, which stratify the partitioning and name the means to subtract; whether Estimate is the
//   component that subtracts them is stated by demean_site (the iteration's resolved site, the same
//   value the caller imposed on the preconditioner), not inferred from either of those two.
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
              const std::vector<ssize_t> &volume_group = {},
              const demean_site_t demean_site = demean_site_t::NONE);

} // namespace MR::Denoise::Iterative
