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

namespace MR::Denoise::Estimator {

class MRM2022 : public Base {
public:
  MRM2022() = default;
  Result operator()(const eigenvalues_type &s,
                    const ssize_t m,
                    const ssize_t n,
                    const Eigen::Vector3d & /*unused*/) const final {
    Result result;
    const ssize_t mprime = std::min(m, n);
    const ssize_t nprime = std::max(m, n);
    const double sigmasq_to_lamplus = Math::pow2(std::sqrt(nprime) + std::sqrt(mprime));
    double clam = 0.0;
    for (ssize_t i = 0; i != mprime; ++i)
      clam += std::max(s[i], 0.0);
    clam /= nprime;
    // Unlike Exp# code,
    //   MRM2022 article uses p to index number of signal components,
    //   and here doing a direct translation of the manuscript content to code
    double lamplusprev = -std::numeric_limits<double>::infinity();
    for (ssize_t p = 0; p < mprime; ++p) {
      const ssize_t i = mprime - 1 - p;
      const double lam = std::max(s[i], 0.0) / nprime;
      if (lam < lamplusprev)
        return result;
      clam -= lam;
      const double sigmasq = clam / ((mprime - p) * (nprime - p));
      lamplusprev = sigmasq * sigmasq_to_lamplus;
      result.cutoff_p = i;
      result.sigma2 = sigmasq;
      result.lamplus = lamplusprev;
    }
    return result;
  }
};

} // namespace MR::Denoise::Estimator
