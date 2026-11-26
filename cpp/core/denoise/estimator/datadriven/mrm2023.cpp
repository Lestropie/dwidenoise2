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

#include "denoise/estimator/datadriven/mrm2023.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "denoise/estimator/pooling.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

Result MRM2023::operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const {
  // MRM2023 article suggests that mprime should subtract z
  //   since it refers to "non-zero singular values";
  //   possible that this is the case for all estimators
  const auto &s = spectrum.eigenvalues();
  const ssize_t rz = spectrum.rz();
  const ssize_t mprime = spectrum.rnz();
  const ssize_t nprime = spectrum.qnz();
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
      // lamplus above is on the raw eigenvalue scale (it is compared against s[i]); the boundaries
      //   recorded on the Result are normalised, and sigma2 * edge() reproduces lamplus / nprime
      //   identically, edge() being (sqrt(nprime) + sqrt(mprime))^2 / nprime.
      result.set_boundaries(spectrum.edge(), spectrum.max_noise_standardized(result.cutoff_p));
      return result;
    }
    clam -= s[i];
  }
  result.cutoff_p = 0;
  result.sigma2 = 0.0;
  result.set_boundaries(spectrum.edge(), 0.0);
  return result;
}

Result MRM2023::operator()(const std::vector<Spectrum> &spectra, //
                           const Eigen::Vector3d & /*unused*/) const {
  Result result;
  const std::vector<PooledEigenvalue> pooled = pool_standardized(spectra);
  const ssize_t mtot = ssize_t(pooled.size());
  if (mtot == 0)
    return result;

  // Every component starts as a noise candidate; peeling proceeds from the top.
  std::vector<ssize_t> k(spectra.size(), 0);      // components of each partition still candidates
  std::vector<ssize_t> peeled(spectra.size(), 0); // components of each partition already removed
  double clam = 0.0;                              // raw energy of the candidates
  for (const PooledEigenvalue &entry : pooled) {
    ++k[entry.partition];
    clam += spectra[entry.partition].eigenvalue(entry.index);
  }

  double sigma2_found = std::numeric_limits<double>::signaling_NaN();
  double zstar = std::numeric_limits<double>::signaling_NaN();
  bool found = false;
  for (ssize_t p = 0; p < mtot; ++p) {
    double dof = 0.0;
    for (size_t q = 0; q != spectra.size(); ++q)
      dof += spectra[q].dof_deflated(k[q], peeled[q]);
    const double sigmasq = clam / dof;
    const PooledEigenvalue &largest = pooled[mtot - p - 1]; // most significant remaining candidate
    if (largest.z < sigmasq) {
      sigma2_found = sigmasq;
      zstar = largest.z; // classify components at or below the largest remaining as noise
      found = true;
      break;
    }
    // Peel this component: it is signal, and its partition loses one direction of both its
    //   candidate count and its long dimension.
    clam -= spectra[largest.partition].eigenvalue(largest.index);
    --k[largest.partition];
    ++peeled[largest.partition];
  }
  if (!found) {
    // No noise boundary located: every component is treated as signal (mirrors the single-PCA
    //   fall-through, in which cutoff_p / sigma2 / lamplus are all zero).
    result.cutoff_p = 0;
    result.sigma2 = 0.0;
    result.set_boundaries(mean_edge(spectra), 0.0);
    result.cutoff_p_partition.assign(spectra.size(), 0);
    result.lamplus_partition.assign(spectra.size(), 0.0);
    return result;
  }
  const NoiseStats stats = apply_threshold_standardized(spectra, zstar, result);
  result.sigma2 = sigma2_found;
  // Marchenko-Pastur edge (normalised) at the representative aspect ratio, as in the single-PCA
  //   path; the per-partition boundaries actually applied are in Result::lamplus_partition.
  result.set_boundaries(mean_edge(spectra), stats.z_max);
  return result;
}

double MRM2023::predicted_rmse(const ssize_t m, const ssize_t n, const double r) const {
  const double lm = std::log(double(m));
  const double ln = std::log(double(n));
  const double lr = std::log(std::max(r, 1.0));
  return std::exp(3.07484 - 1.01543 * lm - 1.16065 * ln + 0.69571 * lr + 0.088552 * lm * ln);
}

} // namespace MR::Denoise::Estimator
