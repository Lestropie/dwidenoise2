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

#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "types.h"

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

// Reusable lookup table on a uniform grid, with linear interpolation in the
//   interior and linear extrapolation beyond either end.
// Used to tabulate the (otherwise expensive) forward / inverse transforms
//   as one-dimensional functions of the normalised intensity theta = m / sigma.
class Lut {
public:
  Lut() : grid_origin(0.0), grid_spacing(1.0) {}
  Lut(const default_type origin, const default_type spacing, std::vector<default_type> &&values)
      : grid_origin(origin), grid_spacing(spacing), y(std::move(values)) {
    assert(grid_spacing > default_type(0));
  }
  bool empty() const { return y.empty(); }
  default_type operator()(const default_type x) const {
    assert(!y.empty());
    if (y.size() == 1)
      return y[0];
    const default_type t = (x - grid_origin) / grid_spacing;
    if (t <= default_type(0))
      return y[0] + t * (y[1] - y[0]);
    const ssize_t n = ssize_t(y.size());
    const ssize_t i = ssize_t(std::floor(t));
    if (i >= n - 1)
      return y[n - 1] + (t - default_type(n - 1)) * (y[n - 1] - y[n - 2]);
    const default_type frac = t - default_type(i);
    return y[i] * (default_type(1) - frac) + y[i + 1] * frac;
  }
  default_type origin() const { return grid_origin; }
  default_type spacing() const { return grid_spacing; }
  const std::vector<default_type> &values() const { return y; }

private:
  default_type grid_origin;
  default_type grid_spacing;
  std::vector<default_type> y;
};

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
};

// Identity ("none") noise model: no variance-stabilising transform.
//
// Every mapping is the identity with unit Jacobian, so the data reach PCA
//   unmodified (save for any demeaning) and the inverse simply re-adds the
//   subtracted mean. None of the mappings involve the noise level sigma, so the
//   stabilised-domain group means carry no sigma dependence and the post-PCA
//   noise estimate is already in absolute units (no sigma rescaling applies);
//   iterating to refine sigma therefore cannot influence subsequent processing.
class Identity : public Base {
public:
  Identity() = default;
  default_type stabilise(const default_type m, const default_type /*sigma*/) const final { return m; }
  default_type inverse_algebraic(const default_type u, const default_type /*sigma*/) const final { return u; }
  default_type inverse_unbiased(const default_type u, const default_type /*sigma*/) const final { return u; }
  default_type jacobian(const default_type /*u*/, const default_type /*sigma*/) const final { return default_type(1); }
  default_type mean(const default_type nu, const default_type /*sigma*/) const final { return nu; }
  default_type variance(const default_type /*nu*/, const default_type sigma) const final { return sigma * sigma; }
  ssize_t num_channels() const final { return 1; }
  ssize_t dof() const final { return 2; }
  std::string description() const final { return "none (no variance-stabilising transform)"; }
};

// Gaussian (complex / phase-demodulated) noise model.
//
// The variance-stabilising transform is the trivial linear scaling u = m / sigma,
//   so that the stabilised data are zero-mean, unit-variance Gaussian; there is
//   no distribution bias to remove, and both output modes coincide.
// This reproduces exactly the behaviour of the prior linear VST.
class Gaussian : public Base {
public:
  Gaussian() = default;
  default_type stabilise(const default_type m, const default_type sigma) const final {
    return sigma > default_type(0) ? m / sigma : default_type(0);
  }
  default_type inverse_algebraic(const default_type u, const default_type sigma) const final { return u * sigma; }
  default_type inverse_unbiased(const default_type u, const default_type sigma) const final { return u * sigma; }
  default_type jacobian(const default_type /*u*/, const default_type sigma) const final { return sigma; }
  default_type mean(const default_type nu, const default_type /*sigma*/) const final { return nu; }
  default_type variance(const default_type /*nu*/, const default_type sigma) const final { return sigma * sigma; }
  ssize_t num_channels() const final { return 1; }
  ssize_t dof() const final { return 2; }
  std::string description() const final { return "Gaussian"; }
};

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
  Lut forward;
  // Normalised algebraic inverse f^{-1}: u -> theta_m.
  Lut inverse_algebraic_lut;
  // Normalised exact-/moment- inverse: u -> theta_nu (= nu / sigma).
  Lut inverse_unbiased_lut;
  // Normalised local gain d(theta_nu)/d(u).
  Lut jacobian_lut;
  // Normalised floor mean f-domain argument mean_normalised(0) (= sqrt(pi/2) for N=1).
  default_type floor_mean;
};

// Rician noise model: the single-channel (N = 1) special case of the
//   non-central chi distribution.
class Rician : public NonCentralChi {
public:
  explicit Rician(const vst_method_t vst_method) : NonCentralChi(1, vst_method) {}
  std::string description() const final;
};

// Construct a noise model for the requested distribution.
// - num_channels is used only for NONCENTRALCHI (ignored for GAUSSIAN / RICIAN).
// - vst_method is used only for magnitude distributions (ignored for GAUSSIAN).
std::shared_ptr<Base> make(const distribution_t distribution, //
                           const ssize_t num_channels,        //
                           const vst_method_t vst_method);    //

} // namespace MR::Denoise::NoiseModel
