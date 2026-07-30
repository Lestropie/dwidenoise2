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
#include "math/median.h"

namespace MR::Denoise::Estimator {

class Med : public Base {
public:
  Med() = default;
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final {
    const auto &s = spectrum.eigenvalues;
    const ssize_t qnz = spectrum.qnz;
    const ssize_t rz = spectrum.rz;
    const ssize_t rnz = spectrum.rnz;
    // Eigenvalues should already be sorted;
    //   no need to execute a sort for median calculation
    // Do however need to skip any components assumed to be zero based on preconditioning
    const double ymed = rnz & 1                                                //
                            ? s[rz + rnz / 2]                                  //
                            : (0.5 * (s[rz + rnz / 2 - 1] + s[rz + rnz / 2])); //
    const double beta = spectrum.beta;
    // Ratio of the Marchenko-Pastur upper bulk edge lambda+ to the noise level sigma^2.
    const double edge = Math::pow2(1.0 + std::sqrt(beta));
    Result result;
    // Gavish & Donoho (2014) median estimator of the noise *variance*: sigma^2 = ymed/(qnz*mu(beta)).
    //   This is robust to the presence of signal components (the median is unaffected by the few
    //   large signal eigenvalues). Note that this is the noise level, NOT the bulk edge: the signal/
    //   noise boundary used for rank determination is the MP upper edge lambda+ = sigma^2*edge.
    double sigma2 = ymed / (qnz * mu(beta));
    double lamplus = sigma2 * edge;
    // Count noise components: nonzero eigenvalues at/below the bulk edge (plus the rz assumed-zero
    //   components), ascending.
    const auto count_noise = [&s, rz, qnz](const double thresh) {
      ssize_t c = rz;
      for (ssize_t p = rz; p != s.size(); ++p) {
        if (s[p] / qnz > thresh)
          break;
        c = p + 1;
      }
      return c;
    };
    result.cutoff_p = count_noise(lamplus);
    // The median gives a robust signal/noise threshold; once the noise eigenvalues are isolated,
    //   the minimum-variance estimator of sigma^2 is their mean (the Marchenko-Pastur / MPPCA
    //   estimate, consistent with the other estimators). Recompute sigma^2 as that mean, then
    //   recompute the bulk edge (lamplus) and the signal/noise cutoff from the refined noise level.
    if (result.cutoff_p > rz) {
      sigma2 = s.segment(rz, result.cutoff_p - rz).sum() / (qnz * double(result.cutoff_p - rz));
      lamplus = sigma2 * edge;
      result.cutoff_p = count_noise(lamplus);
    }
    result.sigma2 = sigma2;
    result.lamplus = lamplus;
    return result;
  }

  // Partitioned form: the median is taken across the pooled (normalised) noise eigenvalues,
  //   giving a single, lower-variance noise level; the resulting threshold is applied to each
  //   partition to obtain its own signal rank.
  Result operator()(const std::vector<Spectrum> &spectra, //
                    const Eigen::Vector3d & /*unused*/) const final {
    const std::vector<double> pooled = pool_normalized(spectra); // ascending, normalised
    Result result;
    if (pooled.empty())
      return result;
    const size_t np = pooled.size();
    const double ymed = (np & 1) ? pooled[np / 2] : (0.5 * (pooled[np / 2 - 1] + pooled[np / 2]));
    const double beta = mean_beta(spectra);
    // Gavish & Donoho (2014) median noise *variance* (pooled values are already normalised, so no
    //   qnz factor). The signal/noise boundary is the MP upper edge sigma^2*(1+sqrt(beta_p))^2,
    //   applied per partition because beta_p differs between partitions.
    double sigma2 = ymed / mu(beta);
    std::vector<double> t(spectra.size());
    const auto set_edges = [&t, &spectra](const double s2) {
      for (size_t p = 0; p != spectra.size(); ++p)
        t[p] = s2 * Math::pow2(1.0 + std::sqrt(spectra[p].beta));
    };
    set_edges(sigma2);
    double noise_sum = 0.0;
    ssize_t noise_count = 0;
    apply_threshold_per_partition(spectra, t, result, noise_sum, noise_count);
    // Refine sigma^2 as the mean of the pooled (normalised) noise eigenvalues, then recompute the
    //   per-partition edges and signal/noise cutoffs from the refined noise level.
    if (noise_count > 0) {
      sigma2 = noise_sum / double(noise_count);
      set_edges(sigma2);
      apply_threshold_per_partition(spectra, t, result, noise_sum, noise_count);
    }
    result.sigma2 = sigma2;
    result.lamplus = sigma2 * Math::pow2(1.0 + std::sqrt(beta)); // representative edge (mean beta)
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
