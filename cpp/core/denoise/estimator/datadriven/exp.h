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

#include <algorithm>
#include <cmath>
#include <vector>

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

template <ssize_t version> class Exp : public Base {
public:
  Exp() {}
  ~Exp() {}
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s,    //
                    const ssize_t m,                                 //
                    const ssize_t n,                                 //
                    const ssize_t rp,                                //
                    const Eigen::Vector3d & /*unused*/) const final; //
  Result operator()(const std::vector<eigenvalues_type> &s,          //
                    const std::vector<ssize_t> &m,                   //
                    const std::vector<ssize_t> &n,                   //
                    const std::vector<ssize_t> &rp,                  //
                    const Eigen::Vector3d & /*unused*/) const final; //
  bool supports_partitioning() const final { return true; }

  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n) over m in [30,2000],
  //   n >= m and rank density a in [0.25,4] (sigma = 1; conservative theta = 1.5 fit).
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final {
    const double lm = std::log(double(m));
    const double ln = std::log(double(n));
    const double lr = std::log(std::max(r, 1.0));
    if (version == 1) // Exp1 (Veraart 2016): R^2 = 0.96
      return std::exp(0.02654 - 0.42175 * lm - 0.70154 * ln + 0.73385 * lr + 0.019899 * lm * ln);
    // Exp2 (Cordero-Grande 2019): R^2 = 0.94
    return std::exp(0.48522 - 0.60721 * lm - 0.72413 * ln + 0.68613 * lr + 0.039857 * lm * ln);
  }
};

} // namespace MR::Denoise::Estimator
