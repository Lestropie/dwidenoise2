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

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

// TODO Move to .cpp
template <ssize_t version> class Exp : public Base {
public:
  Exp() = default;
  Result operator()(const eigenvalues_type &s, const ssize_t m, const ssize_t n) const final {
    Result result;
    const ssize_t r = std::min(m, n);
    const ssize_t q = std::max(m, n);
    const double lam_r = std::max(s[0], 0.0) / q;
    double clam = 0.0;
    for (ssize_t p = 0; p < r; ++p) // p+1 is the number of noise components
    {                               // (as opposed to the paper where p is defined as the number of signal components)
      const double lam = std::max(s[p], 0.0) / q;
      clam += lam;
      double denominator = std::numeric_limits<double>::signaling_NaN();
      switch (version) {
      case 1:
        denominator = q;
        break;
      case 2:
        denominator = q - (r - p - 1);
        break;
      default:
        assert(false);
      }
      const double gam = double(p + 1) / denominator;
      const double sigsq1 = clam / double(p + 1);
      const double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
      // sigsq2 > sigsq1 if signal else noise
      if (sigsq2 < sigsq1) {
        result.sigma2 = sigsq1;
        result.cutoff_p = p + 1;
        result.lamplus = lam;
      }
    }
    return result;
  }
};

} // namespace MR::Denoise::Estimator
