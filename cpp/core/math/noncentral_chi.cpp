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

#include "math/noncentral_chi.h"

#include <algorithm>
#include <cmath>

namespace MR::Math::NonCentralChi {

namespace {

// 1F1(alpha; gamma; w) (confluent hypergeometric / Kummer) for w >= 0 and alpha, gamma > 0,
//   evaluated as a positive-term series (no catastrophic cancellation).
default_type oneF1_pos(const default_type alpha, const default_type gamma, const default_type w) {
  default_type term = 1.0;
  default_type sum = 1.0;
  for (ssize_t k = 0; k != 100000; ++k) {
    term *= (alpha + default_type(k)) / ((gamma + default_type(k)) * default_type(k + 1)) * w;
    sum += term;
    if (term <= 1.0e-15 * sum)
      break;
  }
  return sum;
}

} // namespace

// The Kummer transform 1F1(-1/2; N; -w) = e^{-w} 1F1(N+1/2; N; w), w = a^2/2, turns the
//   alternating series into a positive one.
default_type normalised_mean(const default_type a, const default_type dof) {
  const default_type N = 0.5 * dof;
  const default_type w = 0.5 * a * a;
  if (w <= 350.0) {
    const default_type gamma_ratio = std::exp(std::lgamma(N + 0.5) - std::lgamma(N));
    return std::sqrt(2.0) * gamma_ratio * std::exp(-w) * oneF1_pos(N + 0.5, N, w);
  }
  return std::sqrt(a * a + (dof - 1.0));
}

default_type normalised_variance(const default_type a, const default_type dof) {
  const default_type mu = normalised_mean(a, dof);
  const default_type v = (a * a + dof) - mu * mu;
  return std::max(v, 1.0e-12);
}

} // namespace MR::Math::NonCentralChi
