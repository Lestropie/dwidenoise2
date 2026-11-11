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

#include <vector>

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise::Estimator {

template <ssize_t version> class Exp : public Base {
public:
  Exp() {}
  ~Exp() {}
  Result operator()(const Spectrum &spectrum,                        //
                    const Eigen::Vector3d & /*unused*/) const final; //
  Result operator()(const std::vector<Spectrum> &spectra,            //
                    const Eigen::Vector3d & /*unused*/) const final; //
  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n) over m in [30,2000],
  //   n >= m and rank density a in [0.25,4] (sigma = 1; conservative theta = 1.5 fit).
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final;
};

} // namespace MR::Denoise::Estimator
