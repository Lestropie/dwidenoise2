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

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

class MRM2023 : public Base {
public:
  MRM2023() = default;
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s,     //
                    const ssize_t m,                                  //
                    const ssize_t n,                                  //
                    const ssize_t rp,                                 //
                    const Eigen::Vector3d & /*unused*/) const final { //
    // MRM2023 article suggests that mprime should subtract z
    //   since it refers to "non-zero singular values";
    //   possible that this is the case for all estimators
    assert(s.size() == std::min(m, n));
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t mprime = rank_nonzero(m, n, rp);
    const ssize_t nprime = dimlong_nonzero(m, n, rp);
    const double sigmasq_to_lamplus = Math::pow2(std::sqrt(nprime) + std::sqrt(mprime));
    double clam = s.segment(rz, mprime).sum();
    // Unlike Exp# code,
    //   MRM2023 article uses p to index number of signal components,
    //   and here doing a direct translation of the manuscript content to code

    Result result;
    for (ssize_t p = 0; p < mprime; ++p) {
      const ssize_t i = s.size() - 1 - p;
      const double sigmasq = clam / (static_cast<double>(mprime - p) * static_cast<double>(nprime - p));
      const double lamplus = sigmasq * sigmasq_to_lamplus;
      if (s[i] < lamplus) {
        result.cutoff_p = i + 1;
        result.sigma2 = sigmasq;
        result.lamplus = lamplus / nprime;
        return result;
      }
      clam -= s[i];
    }
    result.cutoff_p = 0;
    result.sigma2 = 0.0;
    result.lamplus = 0.0;
    return result;
  }
};

} // namespace MR::Denoise::Estimator
