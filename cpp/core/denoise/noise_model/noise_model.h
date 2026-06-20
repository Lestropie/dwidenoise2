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

#include <memory>
#include <string>
#include <vector>

#include "types.h"

#include "denoise/noise_model/base.h"

namespace MR::Denoise::NoiseModel {

// Statistical distribution governing the raw image intensities prior to PCA:
// - GAUSSIAN: complex (or phase-demodulated) data; zero-mean Gaussian per channel.
// - RICIAN: magnitude data from a single receive channel (non-central chi, 2 DOF).
// - NONCENTRALCHI: magnitude data combined across N channels by sum-of-squares (2N DOF).
const std::vector<std::string> distributions = {"gaussian", "rician", "noncentralchi"};
enum class distribution_t { GAUSSIAN, RICIAN, NONCENTRALCHI };

// Variance-stabilising transform (VST) applied to the raw data prior to PCA.
// Two of the options select the transform directly, independent of the noise
//   distribution:
// - NONE:   identity; no transform is applied (the data reach PCA unmodified
//             save for any demeaning). As nothing is divided by the noise level,
//             refining that level across iterations has no effect on subsequent
//             processing, so the calling commands fall back to a single pass.
// - LINEAR: the simple linear transform u = m / sigma (the Gaussian model);
//             exact for additive Gaussian (complex / phase-demodulated) noise,
//             but only a scale normalisation for magnitude data (which remains
//             heteroscedastic near the floor) and with no noise-floor debiasing.
// The remaining options build the nonlinear transform for magnitude data and,
//   in particular, the exact-unbiased inverse that debiases the noise floor:
// - FOI:  Foi (2011)-style exact-unbiased inverse via numerical integration;
//           smooth and well-defined across the entire SNR range including nu=0.
// - KOAY: Koay-Basser (2006) first-moment inverse with a hard floor clamp
//           (no unique solution below SNR ~1.913).
// - MOM:  method-of-moments (closed-form) first-moment inverse.
// For these three the forward stabilising transform itself is shared;
//   they differ only in how the stabilised domain is mapped back to a
//   bias-free underlying level (the inverse / debias step).
const std::vector<std::string> vst_methods = {"none", "linear", "foi", "koay", "mom"};
enum class vst_method_t { NONE, LINEAR, FOI, KOAY, MOM };

// Construct a noise model for the requested distribution.
// - num_channels is used only for NONCENTRALCHI (ignored for GAUSSIAN / RICIAN).
// - vst_method is used only for magnitude distributions (ignored for GAUSSIAN).
std::shared_ptr<Base> make(const distribution_t distribution, //
                           const ssize_t num_channels,        //
                           const vst_method_t vst_method);    //

} // namespace MR::Denoise::NoiseModel
