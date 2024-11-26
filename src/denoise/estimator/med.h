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

#include <limits>

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "math/math.h"
#include "math/median.h"

namespace MR::Denoise::Estimator {

class Med : public Base {
public:
  Med() = default;
  Result operator()(const eigenvalues_type &s, const ssize_t m, const ssize_t n) const final {
    Result result;
    const ssize_t beta = double(std::min(m, n)) / double(std::max(m, n));
    // Eigenvalues should already be sorted;
    //   no need to execute a sort for median calculation
    const double ymed = s.size() & 1 ? s[s.size() / 2] : (0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]));
    result.sigma2 = Math::pow2(ymed) / (std::max(m, n) * mu(beta));
    return result;
  }

protected:
  // Coefficients as provided in Gavish and Donohue 2014
  // double omega(const double beta) const {
  //   const double betasq = Math::pow2(beta);
  //   return (0.56*beta*betasq - 0.95*betasq + 1.82*beta + 1.43);
  // }
  // Median of Marcenko-Pastur distribution
  // Third-order polynomial fit to data generated using Matlab code supplementary to Gavish and Donohue 2014
  double mu(const double beta) const {
    const double betasq = Math::pow2(beta);
    return ((-0.005882794526340723 * betasq * beta) - (0.007508551496715836 * betasq) - (0.3338169644754149 * beta) +
            1.0);
  }
};

} // namespace MR::Denoise::Estimator
