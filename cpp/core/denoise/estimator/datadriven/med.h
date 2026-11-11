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

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise::Estimator {

class Med : public Base {
public:
  Med() = default;
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final;

  // Partitioned form. The median is the one Marchenko-Pastur statistic that can be pooled across
  //   unequal partitions *exactly*, and it is worth stating why, because the obvious approach is
  //   wrong: the median of the pooled eigenvalues is not the median of MP(beta) for any beta.
  //   Pooling mixes P laws of differing shape, and a mixture's median is not the median of any of
  //   its components -- so evaluating mu() at an average beta leaves a bias that grows with the
  //   spread of the partition sizes.
  // Dividing each partition's normalised eigenvalues by the median mu(beta_p) of its *own* law
  //   removes that bias outright. Every partition then contributes values whose median is exactly
  //   sigma^2, and a mixture whose components all share a median has that same median exactly
  //   (F_mix(sigma^2) = sum_p w_p/2 = 1/2), for any mixing weights whatsoever. The pooled median
  //   is therefore an estimate of sigma^2 with no residual dependence on how the volumes were
  //   distributed between partitions, and it costs one division per eigenvalue.
  // The rescaling doubles as the ordering key here, in place of the standardised value used by the
  //   other estimators: the statistic of interest is the centre of each bulk rather than its edge.
  Result operator()(const std::vector<Spectrum> &spectra, //
                    const Eigen::Vector3d & /*unused*/) const final;

  // Predicted relative RMSE of the (corrected) median noise level estimate; calibrated by numerical
  //   simulation of log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.87.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final;

protected:
  // Coefficients as provided in Gavish and Donohue 2014
  // double omega(const double beta) const {
  //   const double betasq = Math::pow2(beta);
  //   return (0.56*beta*betasq - 0.95*betasq + 1.82*beta + 1.43);
  // }
  // Median of Marcenko-Pastur distribution
  // Third-order polynomial fit to data generated using Matlab code supplementary to Gavish and Donohue 2014
  double mu(const double beta) const;
};

} // namespace MR::Denoise::Estimator
