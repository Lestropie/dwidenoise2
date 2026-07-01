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

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "app.h"
#include "header.h"
#include "image.h"
#include "types.h"

namespace MR::Denoise::Kernel {

class Base;

extern const char *const shape_description;
extern const char *const default_size_description;
extern const char *const cuboid_size_description;

// Per-iteration kernel sizing specification (set by each schedule row; see denoise/schedule.cpp).
//   Every attribute of the sliding-window kernel (shape and size) is now expressed through the
//   schedule's per-row "kernel" column; there are no command-line kernel options. The desired
//   patch is selected through the kernel type and its free parameter(s):
//   - ASPECT_RATIO : spherical kernel sized to n ~ param * m voxels (param = Casorati aspect
//                    ratio n/m). Naive to signal rank; valid on the first iteration (param = 2).
//   - RMSE         : spherical kernel grown until the estimator's predicted sigma-RMSE falls to
//                    param (the tolerance), floored at the square noise block n >= m + r and
//                    capped; used for intermediate noise-estimation iterations. Requires a
//                    signal-rank density from a prior iteration (not valid on the first row).
//   - RANK         : spherical kernel grown until n >= m + r (square noise block); param unused.
//                    Requires a prior rank density; reserved for the final reconstruction pass.
//   - RADIUS       : fixed-radius spherical kernel of param mm (rank-naive; replaces the former
//                    -radius command-line option).
//   - VOXELS       : spherical kernel enlarged until it contains at least param voxels, an
//                    absolute count independent of the volume count (replaces -minvoxels).
//   - CUBOID       : cuboid kernel. extent gives the per-axis voxel counts; {0,0,0} requests
//                    automatic aspect-ratio sizing, growing the near-isotropic patch until it
//                    holds at least param * m voxels (param defaults to 2.0). Replaces the former
//                    -shape cuboid / -extent options.
//   - RANK_FIXED   : spherical kernel of exactly n = m + r voxels, where r is the fixed signal
//                    rank imposed via dwidenoise2 -fixed_rank (the "rank_fixed" schedule keyword).
//                    r is read from the -fixed_rank option at kernel construction, so this type is
//                    meaningful only in conjunction with -fixed_rank.
enum class kernel_spec_type { ASPECT_RATIO, RMSE, RANK, RADIUS, VOXELS, CUBOID, RANK_FIXED };
struct KernelSpec {
  kernel_spec_type type{kernel_spec_type::ASPECT_RATIO};
  // Scalar free parameter: aspect ratio (ASPECT_RATIO), RMSE tolerance (RMSE), radius in mm
  //   (RADIUS), absolute voxel count (VOXELS), or the auto-size aspect ratio of an
  //   extent-less CUBOID. Unused by RANK / RANK_FIXED and by a CUBOID with an explicit extent.
  default_type param{2.0};
  // Explicit cuboid extent (per spatial axis) for CUBOID; {0,0,0} requests automatic aspect-ratio
  //   sizing (using param as the aspect ratio).
  std::array<ssize_t, 3> extent{0, 0, 0};
};

// predicted_rmse(m, n, r) -> expected relative RMSE of the noise level estimate for the selected
//   estimator at a Casorati matrix of m volumes, n voxels and signal rank r. Supplied (built from
//   the active estimator) only when a row uses an RMSE kernel; an empty function otherwise.
using predicted_rmse_func = std::function<double(ssize_t, ssize_t, double)>;

// H is the header of the data to be decomposed; num_volumes(H) gives the number of volumes
//   in the Casorati matrix for this iteration (m', which is reduced under temporal
//   sub-sampling). full_num_volumes is the number of volumes in the absence of temporal
//   sub-sampling; it is used only to decide whether the Casorati shape warning applies.
// num_partitions is the number of volume partitions for this iteration (>= 1). When > 1 the
//   volume-derived kernel size is based on the smallest partition's volume count (m'/P), so
//   that each partition's Casorati sub-matrix preserves the target aspect ratio; fixed-geometry
//   kernels (RADIUS / VOXELS / CUBOID) are unaffected (their size is pinned explicitly by the
//   schedule row).
// kernel_spec selects the per-iteration kernel shape and size (see KernelSpec above); it is the
//   sole source of kernel configuration (there are no command-line kernel options).
//   rank_per_mm_image is required by the RMSE and RANK kernels (a rank density from a prior
//   iteration); predicted_rmse is required by the RMSE kernel (built from the active estimator).
//   Both may be empty/invalid for the rank-naive kernels (ASPECT_RATIO / RADIUS / VOXELS / CUBOID /
//   RANK_FIXED).
std::shared_ptr<Base> make_kernel(const Header &H,                                 //
                                  const std::array<ssize_t, 3> &subsample_factors, //
                                  const KernelSpec &kernel_spec,                   //
                                  const Image<float> &rank_per_mm_image,           //
                                  const ssize_t full_num_volumes,                  //
                                  const ssize_t num_partitions = 1,                //
                                  const predicted_rmse_func &predicted_rmse = {}); //

} // namespace MR::Denoise::Kernel
