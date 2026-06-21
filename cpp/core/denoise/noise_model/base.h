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

namespace MR::Denoise::NoiseModel {

// Abstract noise model.
//
// All member functions operate on real-valued scalars; for complex (Gaussian)
//   input data the same transform is applied independently to the real and
//   imaginary channels by the caller.
//
// Notation:
//   m     raw image datum (e.g. magnitude intensity);
//   sigma per-channel noise level (the VST scale / dispersion parameter);
//   u     stabilised-domain value (approximately unit-variance, Gaussian);
//   nu    underlying (bias-free) signal level.
class Base {
public:
  Base() = default;
  Base(const Base &) = delete;
  virtual ~Base() = default;

  // Forward variance-stabilising transform Phi: raw datum -> stabilised value.
  // Designed so that Var[Phi(m) | nu] ~ 1 for all nu >= 0.
  virtual default_type stabilise(const default_type m, const default_type sigma) const = 0;

  // Algebraic inverse Phi^{-1}: stabilised value -> raw (still-biased) datum.
  // Recovers the conventional biased-magnitude intensity; used for the
  //   -preserve_noise_bias output mode.
  virtual default_type inverse_algebraic(const default_type u, const default_type sigma) const = 0;

  // Exact-unbiased inverse Psi: stabilised-domain mean -> bias-free level nu.
  // Applied only to the per-group DC (mean) term so that the noise-floor bias
  //   is not re-introduced into the denoised output.
  virtual default_type inverse_unbiased(const default_type u, const default_type sigma) const = 0;

  // Local gain J = d(nu)/d(u) at the operating point u; the linear factor by
  //   which a stabilised-domain residual is mapped back to the intensity scale
  //   (this linearity is what preserves the Gaussian character of the residual).
  virtual default_type jacobian(const default_type u, const default_type sigma) const = 0;

  // Distribution moments of the raw datum given the underlying level nu.
  virtual default_type mean(const default_type nu, const default_type sigma) const = 0;
  virtual default_type variance(const default_type nu, const default_type sigma) const = 0;

  // Number of receive channels combined by sum-of-squares (1 = Rician / Gaussian).
  virtual ssize_t num_channels() const = 0;
  // Statistical degrees of freedom (2 * num_channels for magnitude data).
  virtual ssize_t dof() const = 0;

  virtual std::string description() const = 0;

  // Whether the variance-stabilising transform is linear: the stabilised data are mapped
  //   back to the intensity scale by a pure (sigma-dependent) gain, with no noise-floor bias
  //   (inverse_unbiased coincides with inverse_algebraic). True for the Gaussian and identity
  //   models (complex data, or -vst_method none / linear); false for the magnitude
  //   (Rician / non-central chi) models, whose inverse debiases the noise floor. When true the
  //   -preserve_noise_bias distinction is a no-op: there is no estimated noise bias to
  //   preserve or remove.
  virtual bool is_linear() const = 0;
};

} // namespace MR::Denoise::NoiseModel
