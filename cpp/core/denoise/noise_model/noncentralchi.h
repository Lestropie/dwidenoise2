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

#include <string>

#include "types.h"

#include "denoise/noise_model/base.h"
#include "denoise/noise_model/lut.h"
#include "denoise/noise_model/noise_model.h"

namespace MR::Denoise::NoiseModel {

// Magnitude noise model: non-central chi with 2N degrees of freedom
//   (N receive channels combined by sum-of-squares); N = 1 is the Rician case.
//
// At construction the forward stabilising transform and the chosen inverse /
//   debias mapping are tabulated as one-dimensional functions of the normalised
//   intensity theta = m / sigma (a one-time build, per DOF).
class NonCentralChi : public Base {
public:
  NonCentralChi(const ssize_t num_channels, const vst_method_t vst_method);
  default_type stabilise(const default_type m, const default_type sigma) const final;
  default_type inverse_algebraic(const default_type u, const default_type sigma) const final;
  default_type inverse_unbiased(const default_type u, const default_type sigma) const final;
  default_type jacobian(const default_type u, const default_type sigma) const final;
  default_type mean(const default_type nu, const default_type sigma) const final;
  default_type variance(const default_type nu, const default_type sigma) const final;
  ssize_t num_channels() const final { return N; }
  ssize_t dof() const final { return 2 * N; }
  std::string description() const override;

protected:
  const ssize_t N;
  const vst_method_t vst_method;
  // Normalised forward transform f: theta_m -> u (stabilise without sigma scaling).
  LUT forward;
  // Normalised algebraic inverse f^{-1}: u -> theta_m.
  LUT inverse_algebraic_lut;
  // Normalised exact-/moment- inverse: u -> theta_nu (= nu / sigma).
  LUT inverse_unbiased_lut;
  // Normalised local gain d(theta_nu)/d(u).
  LUT jacobian_lut;
  // Normalised floor mean f-domain argument mean_normalised(0) (= sqrt(pi/2) for N=1).
  default_type floor_mean;
};

} // namespace MR::Denoise::NoiseModel
