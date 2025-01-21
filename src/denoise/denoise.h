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

namespace MR::Denoise {

using eigenvalues_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using vector_type = Eigen::Array<double, Eigen::Dynamic, 1>;

extern const char *first_step_description;
extern const char *non_gaussian_noise_description;
extern const char *filter_description;
extern const char *aggregation_description;

const std::vector<std::string> dtypes = {"float32", "float64"};
extern const App::Option datatype_option;

const std::vector<std::string> filters = {"optshrink", "optthresh", "truncate"};
enum class filter_type { OPTSHRINK, OPTTHRESH, TRUNCATE };

const std::vector<std::string> aggregators = {"exclusive", "gaussian", "invl0", "rank", "uniform"};
enum class aggregator_type { EXCLUSIVE, GAUSSIAN, INVL0, RANK, UNIFORM };

} // namespace MR::Denoise
