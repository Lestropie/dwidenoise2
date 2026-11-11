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

#include <string>

#include "types.h"

#include "denoise/precondition/noise_model/base.h"
#include "denoise/precondition/noise_model/lut.h"
#include "denoise/precondition/noise_model/noise_model.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Magnitude noise model: non-central chi with "dof" degrees of freedom, arising from dof/2
//   receive channels combined by sum-of-squares; dof == 2 is the Rician case.
//
// The parameter is real-valued, not integral. What governs the distribution is the EFFECTIVE
//   channel count: correlation of the noise between channels, and spatial overlap of their
//   sensitivities, mean that L physical channels combined by sum of squares behave as fewer than L
//   independent ones, and the equivalent count is in general fractional (Sakaie & Lowe 2017
//   measured 3.76 +/- 0.07 effective channels, i.e. 7.52 degrees of freedom, for a 12-element
//   coil). Writing N = dof/2, every expression below is analytically valid for real N > 0: the
//   mean sqrt(2) Gamma(N+1/2)/Gamma(N) 1F1(-1/2; N; -a^2/2), the variance a^2 + 2N - mean^2, the
//   high-SNR asymptote sqrt(a^2 + 2N - 1), and the Poisson-mixture representation of the
//   non-central chi-square used to build the exact-unbiased inverse.
//
// At construction the forward stabilising transform and the chosen inverse /
//   debias mapping are tabulated as one-dimensional functions of the normalised
//   intensity theta = m / sigma (a one-time build, per DOF).
class NonCentralChi : public Base {
public:
  // Smallest admissible degrees of freedom, short of which the model ceases to be interpretable
  //   rather than merely unusual: one degree of freedom is the half-Gaussian, a single real
  //   Gaussian component, and no sum of squares of Gaussian variates can yield fewer.
  //   (It is also the point below which the tabulation grid, whose extent is set by dof - 1, and
  //   the central-chi quadrature, whose integrand diverges at the origin once dof < 1, would both
  //   need special handling for no physical gain.) Callers clamp to this bound and report having
  //   done so; the constructor rejects anything below it outright.
  static constexpr default_type min_dof = 1.0;

  NonCentralChi(const default_type dof, const vst_method_t vst_method);
  default_type stabilise(const default_type m, const default_type sigma) const final;
  default_type inverse_algebraic(const default_type u, const default_type sigma) const final;
  default_type inverse_unbiased(const default_type u, const default_type sigma) const final;
  default_type jacobian(const default_type u, const default_type sigma) const final;
  default_type mean(const default_type signal, const default_type sigma) const final;
  default_type variance(const default_type signal, const default_type sigma) const final;
  default_type dof() const final { return ndof; }
  std::string description() const override;
  // The magnitude (non-central chi / Rician) model is non-linear: its inverse debiases
  //   the noise floor, so -preserve_noise_bias is meaningful.
  bool is_linear() const final { return false; }

protected:
  // The degrees of freedom: the model's sole distributional parameter, and the currency in which
  //   it is constructed, reported and compared. The half of it that the standard expressions are
  //   written in terms of is formed where needed rather than stored, so the two cannot disagree.
  //   Not named "nu", which throughout this library denotes the underlying signal level (base.h).
  const default_type ndof;
  const vst_method_t vst_method;
  // Normalised forward transform f: theta_m -> u (stabilise without sigma scaling).
  LUT forward;
  // Normalised algebraic inverse f^{-1}: u -> theta_m.
  LUT inverse_algebraic_lut;
  // Normalised exact-/moment- inverse: u -> theta_nu (= nu / sigma).
  LUT inverse_unbiased_lut;
  // Normalised local gain d(theta_nu)/d(u).
  LUT jacobian_lut;
  // Normalised floor mean f-domain argument normalised_mean(0) (= sqrt(pi/2) for dof=2).
  default_type floor_mean;
};

} // namespace MR::Denoise::Precondition::NoiseModel
