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

namespace MR::Denoise::Precondition::NoiseModel {

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
  bool is_linear() const final { return true; }
};

} // namespace MR::Denoise::Precondition::NoiseModel
