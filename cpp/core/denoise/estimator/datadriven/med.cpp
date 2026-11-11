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

#include "denoise/estimator/datadriven/med.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "denoise/estimator/pooling.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

Result Med::operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const {
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

Result Med::operator()(const std::vector<Spectrum> &spectra, //
                       const Eigen::Vector3d & /*unused*/) const {
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

double Med::predicted_rmse(const ssize_t m, const ssize_t n, const double r) const {
  const double lm = std::log(double(m));
  const double ln = std::log(double(n));
  const double lr = std::log(std::max(r, 1.0));
  return std::exp(2.26745 - 1.08083 * lm - 0.92260 * ln + 0.56761 * lr + 0.097574 * lm * ln);
}

double Med::mu(const double beta) const {
  const double betasq = Math::pow2(beta);
  return ((-0.005882794526340723 * betasq * beta) //
          - (0.007508551496715836 * betasq)       //
          - (0.3338169644754149 * beta)           //
          + 1.0);                                 //
}

} // namespace MR::Denoise::Estimator
