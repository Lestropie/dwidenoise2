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

constexpr ssize_t default_subsample_ratio = 2;

using eigenvalues_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using vector_type = Eigen::Array<double, Eigen::Dynamic, 1>;

extern const char *first_step_description;
extern const char *non_gaussian_noise_description;
extern const char *decomposition_description;
extern const char *filter_description;
extern const char *aggregation_description;

const std::vector<std::string> dtypes = {"float32", "float64"};
extern const App::Option datatype_option;

const std::vector<std::string> decompositions = {"bdcsvd", "selfadjoint"};
enum class decomp_type { BDCSVD, SELFADJOINT };
extern const App::Option decomposition_option;
constexpr decomp_type default_decomposition = decomp_type::BDCSVD;

const std::vector<std::string> filters = {"optshrink", "optthresh", "truncate"};
enum class filter_type { OPTSHRINK, OPTTHRESH, TRUNCATE };

const std::vector<std::string> aggregators = {"exclusive", "gaussian", "invl0", "rank", "uniform"};
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
Image<float> condition_noise_map(Image<float> &in,
                                 const bool nan_to_zero = true,
                                 const bool pad = true,
                                 const bool smooth = false);

} // namespace MR::Denoise
