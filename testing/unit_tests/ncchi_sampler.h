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
#include <random>

// Sampling of the non-central chi magnitude distribution, shared by the unit tests that need
//   synthetic magnitude data with a known distribution.
//
// It lives in a header of its own because the tests are separate executables: each is compiled
//   from one source file with its own main(), so there is no other way for two of them to draw
//   from the same, cross-validated, sampler. The cross-validation itself -- that the fractional-dof
//   sampler reproduces the integer-channel definition wherever both apply -- is asserted by
//   test_samplers() in background_dof.cpp, and applies to every user of this header.

namespace MR::Testing::NCChi {

// Integer channel count: the definition. N complex Gaussians of per-component standard deviation
//   sigma, the first offset by the underlying level nu = a * sigma, combined by sum of squares.
template <class Generator> double sample_integer(const double a, const int channels, const double sigma, Generator &rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  double acc = 0.0;
  for (int c = 0; c != channels; ++c) {
    const double re = nd(rng) + (c == 0 ? a : 0.0);
    const double im = nd(rng);
    acc += re * re + im * im;
  }
  return sigma * std::sqrt(acc);
}

// Real (possibly fractional) degrees of freedom, via the Poisson mixture representation of the
//   non-central chi-square that the noise model's own exact-unbiased inverse is built on:
//     ncchi^2_{dof}(lambda) = chi^2_{dof + 2J},  J ~ Poisson(lambda / 2),  lambda = a^2.
//   Exact for any real dof > 0, and reduces to the integer sampler above at dof = 2N. That
//   equivalence is checked rather than assumed: see test_samplers() in background_dof.cpp.
template <class Generator>
double sample_fractional(const double a, const double dof, const double sigma, Generator &rng) {
  std::poisson_distribution<int> poisson(0.5 * a * a);
  const int j = poisson(rng);
  // chi^2_k == Gamma(shape = k/2, scale = 2)
  std::gamma_distribution<double> gamma(0.5 * dof + double(j), 2.0);
  return sigma * std::sqrt(gamma(rng));
}

template <class Generator> double sample_magnitude(const double a, const double dof, const double sigma, Generator &rng) {
  const int channels = int(std::lround(0.5 * dof));
  const bool integral = std::fabs(dof - 2.0 * double(channels)) < 1.0e-12;
  return integral ? sample_integer(a, channels, sigma, rng) : sample_fractional(a, dof, sigma, rng);
}

} // namespace MR::Testing::NCChi
