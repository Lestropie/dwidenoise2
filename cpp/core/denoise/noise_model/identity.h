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

namespace MR::Denoise::NoiseModel {

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

} // namespace MR::Denoise::NoiseModel
