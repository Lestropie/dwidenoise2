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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

class Med : public Base {
public:
  Med() = default;
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final {
    const auto &s = spectrum.eigenvalues();
    const ssize_t rz = spectrum.rz();
    const ssize_t rnz = spectrum.rnz();
    // Eigenvalues should already be sorted;
    //   no need to execute a sort for median calculation
    // Do however need to skip any components assumed to be zero based on preconditioning
    const double ymed = rnz & 1                                                //
                            ? s[rz + rnz / 2]                                  //
                            : (0.5 * (s[rz + rnz / 2 - 1] + s[rz + rnz / 2])); //
    Result result;
    // Gavish & Donoho (2014) median estimator of the noise *variance*: sigma^2 = ymed/(qnz*mu(beta)).
    //   This is robust to the presence of signal components (the median is unaffected by the few
    //   large signal eigenvalues). Note that this is the noise level, NOT the bulk edge: the signal/
    //   noise boundary used for rank determination is the MP upper edge lambda+ = sigma^2*edge.
    double sigma2 = ymed / (double(spectrum.qnz()) * mu(spectrum.beta()));
    // Count noise components: those at or below the bulk edge (plus the rz assumed-zero ones).
    //   On the standardised scale the bulk edge is sigma^2 itself, so the noise level doubles as
    //   the cut, and lambda+ need only be formed for export.
    result.cutoff_p = spectrum.noise_count(sigma2);
    // The median gives a robust signal/noise threshold; once the noise eigenvalues are isolated,
    //   the minimum-variance estimator of sigma^2 is their energy divided by their degrees of
    //   freedom (the Marchenko-Pastur / MPPCA estimate, consistent with the other estimators).
    //   Recompute sigma^2 that way, then recompute the signal/noise cutoff from the refined level.
    if (result.cutoff_p > rz) {
      const ssize_t k = result.cutoff_p - rz;
      sigma2 = spectrum.raw_sum(rz, k) / spectrum.dof(k);
      result.cutoff_p = spectrum.noise_count(sigma2);
    }
    result.sigma2 = sigma2;
    result.lamplus = sigma2 * spectrum.edge();
    return result;
  }

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
                    const Eigen::Vector3d & /*unused*/) const final {
    Result result;
    std::vector<double> mp_median(spectra.size());
    for (size_t p = 0; p != spectra.size(); ++p)
      mp_median[p] = mu(spectra[p].beta());
    const std::vector<double> pooled = pool_rescaled(spectra, mp_median);
    if (pooled.empty())
      return result;
    const size_t np = pooled.size();
    double sigma2 = (np & 1) ? pooled[np / 2] : (0.5 * (pooled[np / 2 - 1] + pooled[np / 2]));
    // The signal/noise boundary is each partition's own MP upper edge sigma^2 (1+sqrt(beta_p))^2,
    //   which is the standardised cut z == sigma^2.
    NoiseStats stats = apply_threshold_standardized(spectra, sigma2, result);
    // Refine sigma^2 as the pooled noise energy per degree of freedom, then reclassify.
    if (!stats.empty()) {
      sigma2 = stats.sigma2();
      stats = apply_threshold_standardized(spectra, sigma2, result);
    }
    result.sigma2 = sigma2;
    result.lamplus = sigma2 * mean_edge(spectra); // representative edge (mean beta)
    return result;
  }

  bool supports_partitioning() const final { return true; }

  // Predicted relative RMSE of the (corrected) median noise level estimate; calibrated by numerical
  //   simulation of log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.87.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final {
    const double lm = std::log(double(m));
    const double ln = std::log(double(n));
    const double lr = std::log(std::max(r, 1.0));
    return std::exp(2.26745 - 1.08083 * lm - 0.92260 * ln + 0.56761 * lr + 0.097574 * lm * ln);
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
    return ((-0.005882794526340723 * betasq * beta) //
            - (0.007508551496715836 * betasq)       //
            - (0.3338169644754149 * beta)           //
            + 1.0);                                 //
  }
};

} // namespace MR::Denoise::Estimator
