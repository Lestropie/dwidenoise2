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

#include "denoise/denoise.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/sphere_base.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise::Kernel {

class SphereRankAdaptive : public SphereBase {

  // TODO Get rank image estimate from prior iteration into here
  // From experimentation, rank approximately scales *linearly with respect to radius*
  //   (with an intercept of zero);
  //   rank image from early iterations will require rescaling
  //   if the kernel size changes between iterations

public:
  // effective_num_volumes is the per-partition volume count (m'/P) from which the kernel is
  //   sized; with no partitioning it equals the full volume count Denoise::num_volumes(voxel_grid).
  //   Sizing to the smallest partition leaves that partition's noise block ~square.
  // num_nulls is the number of volume-space directions that preconditioning's group-mean
  //   demeaning forces to zero within a partition (Preconditioner::demean_rank()). They carry
  //   neither signal nor noise, so the square-noise-block target is formed from
  //   (effective_num_volumes - num_nulls); see make_kernel().
  SphereRankAdaptive(const Header &voxel_grid,
                     const std::array<ssize_t, 3> &subsample_factors,
                     const Image<float> &rank_per_mm,
                     const ssize_t effective_num_volumes,
                     const ssize_t num_nulls = 0)
      : SphereBase(
            voxel_grid, subsample_factors, SphereBase::compute_max_radius(voxel_grid, 2 * effective_num_volumes)),
        rank_per_mm(rank_per_mm),
        num_volumes(effective_num_volumes),
        num_nondegenerate(std::max<ssize_t>(1, effective_num_volumes - num_nulls)) {}

  SphereRankAdaptive(const SphereRankAdaptive &) = default;

  ~SphereRankAdaptive() override = default;

  Data operator()(const Voxel::index_type &pos) const override;

  // Only used for preallocation of matrices; imprecision not consequential
  ssize_t estimated_size() const override { return num_volumes; }

private:
  // In testing, the signal rank appears to scale proportionately
  //   with the radius of the kernel.
  // Since the kernel size might differ between iterations,
  //   encapsulate this image information as a rank per millimetre kernel radius
  Image<float> rank_per_mm;
  ssize_t num_volumes;
  // Volume-space dimensions carrying data once the regressed group means are removed; the
  //   quantity the square noise block is targeted against.
  ssize_t num_nondegenerate;
};

} // namespace MR::Denoise::Kernel
