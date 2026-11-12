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
  //   simulation of log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.92.
  // Refitted for the deflated degrees of freedom used above. The protocol, which the preceding fit
  //   did not record and which any future change to this estimator must repeat, is:
  //   Y = W L + G, with W (m x r) orthonormal, L (r x n) iid N(0, theta) and G iid N(0, 1), so the
  //   population spike is theta and the Baik-Ben Arous-Peche threshold is theta > sqrt(beta);
  //   m in {30, 60, 120, 250, 500, 1000, 2000}; n/m in {1.1, 1.5, 2, 3, 4}, and additionally
  //   {6, 8} for m <= 120; rank r = round(a n^(1/3)) for a in {0.25, 0.5, 1, 2, 4}; theta in
  //   {1.5, 4}; 200 realisations per cell, falling to 20 at m = 2000; ordinary least squares on
  //   the conservative subset theta == 1.5, n >= m, r < m/2 (199 cells). Fitting the undeflated
  //   arithmetic under this protocol reproduces the coefficients it replaces to within 1.6% of
  //   predicted RMSE everywhere, so the change is the deflation and not sampling drift.
  // Deflating lowers the true error across most of that grid -- the median per-cell ratio is 0.50,
  //   reaching 0.34 for m >= 500, and the bias falls from -0.6% to -0.01% where the estimate is
  //   precise enough to be used -- so the predicted values fall with it, and the RMSE-tolerance
  //   kernel stops growing the patch sooner than it did.
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
