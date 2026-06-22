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

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "app.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

constexpr ssize_t default_spatial_subsample_ratio = 2;

using eigenvalues_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using vector_type = Eigen::Array<double, Eigen::Dynamic, 1>;

extern const char *patent_description;
extern const char *first_step_description;
extern const char *non_gaussian_noise_description;
extern const char *decomposition_description;
extern const char *filter_description;
extern const char *aggregation_description;

// Eigenvalue-decomposition working precision selected by -datatype
//   (the enumerator order maps to the float32/float64 processing path index).
enum class dtype_t { FLOAT32, FLOAT64 };
extern const App::Option datatype_option;

enum class decomp_type { BDCSVD, SELFADJOINT };
extern const App::Option decomposition_option;
constexpr decomp_type default_decomposition = decomp_type::BDCSVD;

enum class filter_type { OPTSHRINK, OPTTHRESH, TRUNCATE };

enum class aggregator_type { EXCLUSIVE, GAUSSIAN, INVL0, RANK, UNIFORM };

// These functions resolve dimensions of the matrix decomposition
//   in the presence of precoditioning that make the data rank-deficient
// - m = number of volumes
// - n = number of voxels in patch
// - rp = rank of preconditioner
ssize_t dimlong_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp);
ssize_t rank_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp);
ssize_t rank_zero(const ssize_t m, const ssize_t n, const ssize_t rp);

// Convenience function for determining the total number of volumes
//   whether the input is 4D or higher
size_t num_volumes(const Header&);

// Function for performing filtering operations on a noise map:
// - May need to replace NaNs with zeros
//   (want to persist with putting NaNs in the noise map as indication of PCA failure rather than rank estimation failure,
//   but this is deleterious in an iterative environment)
// - May need to perform padding in order to facilitate interpolation at the next resolution
//   (while it would be preferable for this to be managed using EdgeHandlers in MRtrix3/mrtrix3#2278,
//    for the sake of progress we will here just do manual explicit padding)
// - May be instructed to perform explicit smoothing of the noise map prior to the next iteration
enum class noise_impute_type{ NAN_TO_ZERO, NONE };
enum class noise_pad_type{ PAD, NONE };
enum class noise_smooth_type{ SMOOTH, NONE };
Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type impute = noise_impute_type::NAN_TO_ZERO,
                                 const noise_pad_type pad = noise_pad_type::PAD,
                                 const noise_smooth_type smooth = noise_smooth_type::NONE);

// Construct the externally-provided noise level map (the variance-stabilising-transform scale)
//   from a -noise_in command-line argument, which may be either:
// - a scalar value, yielding a spatially-constant map over the spatial grid of H_spatial; or
// - the filesystem path to a pre-estimated 3D noise level image.
// The scalar-vs-image dispatch mirrors that of Estimator::make_imposed().
// The returned map is conditioned (padded) for safe cubic interpolation.
Image<float> import_vst_noise_map(const App::ParsedArgument &arg, const Header &H_spatial);

} // namespace MR::Denoise
