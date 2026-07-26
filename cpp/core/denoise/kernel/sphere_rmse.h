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

#include <cmath>

#include "denoise/denoise.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/kernel.h"
#include "denoise/kernel/sphere_base.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise::Kernel {

// Spherical kernel grown until the noise-level estimate is predicted to be precise enough.
//
// Unlike SphereRank (which stops at the square noise block n = m + r), this kernel keeps growing
//   past the square block, while the noise block is too small / the Marchenko-Pastur bulk too wide
//   to estimate sigma accurately, until the active estimator's predicted relative RMSE of the
//   noise level (predicted_rmse(m, n, r), with r = rank_per_mm * radius) falls to the requested
//   tolerance. Growth is floored at the square block (n >= m + r, so it is never smaller than
//   SphereRank) and capped at cap_multiplier * m (the tolerance may be unattainable for very small
//   m / high rank density). For large m the square block already meets the tolerance, so the kernel
//   stops at n = m + r and this reduces to SphereRank behaviour.
class SphereRMSE : public SphereBase {

public:
  // effective_num_volumes is the per-partition volume count (m'/P) from which the kernel is sized.
  SphereRMSE(const Header &voxel_grid,
             const std::array<ssize_t, 3> &subsample_factors,
             const Image<float> &rank_per_mm,
             const ssize_t effective_num_volumes,
             const default_type rmse_tolerance,
             const predicted_rmse_func &predicted_rmse,
             const default_type cap_multiplier = 12.0)
      : SphereBase(voxel_grid,
                   subsample_factors,
                   SphereBase::compute_max_radius(
                       voxel_grid, ssize_t(std::ceil(cap_multiplier * effective_num_volumes)))),
        rank_per_mm(rank_per_mm),
        num_volumes(effective_num_volumes),
        rmse_tolerance(rmse_tolerance),
        cap_multiplier(cap_multiplier),
        predicted_rmse(predicted_rmse) {}

  SphereRMSE(const SphereRMSE &) = default;

  ~SphereRMSE() override = default;

  Data operator()(const Voxel::index_type &pos) const override;

  // Hint for matrix preallocation only; Estimate resizes as needed when a patch is larger.
  ssize_t estimated_size() const override { return num_volumes; }

private:
  Image<float> rank_per_mm;
  ssize_t num_volumes;
  default_type rmse_tolerance;
  default_type cap_multiplier;
  predicted_rmse_func predicted_rmse;
};

} // namespace MR::Denoise::Kernel
