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

#include "denoise/estimator/datadriven/exp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Necessary for precompiler flags
#include "denoise/denoise.h"
#include "denoise/estimator/pooling.h"

namespace MR::Denoise::Estimator {

template <ssize_t version>
Result Exp<version>::operator()(const Spectrum &spectrum,                   //
                                const Eigen::Vector3d & /*unused*/) const { //
  const ssize_t rz = spectrum.rz();
  const double lam_r = spectrum.normalized(rz);
  double clam = 0.0;
  Result result;
  // Note that the paper utilised symbol "p" to refer to the number of signal components;
  //   here "p" is instead the index of the last noise component;
  //   therefore the number of noise compoments is (p + 1 - z)
  for (ssize_t p = rz; p < spectrum.r(); ++p) {
    const double lam = spectrum.normalized(p);
    clam += lam;
    double denominator = std::numeric_limits<double>::signaling_NaN();
    switch (version) {
    case 1:
      denominator = spectrum.qnz();
      break;
    case 2:
      denominator = spectrum.qnz() - (spectrum.r() - p - 1);
      break;
    default:
      assert(false);
    }
    const double gam = double(p + 1 - rz) / denominator;
    const double sigsq1 = clam / double(p + 1 - rz);
    const double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
    // sigsq2 > sigsq1 if signal else noise
    if (sigsq2 < sigsq1) {
      result.sigma2 = sigsq1;
      result.cutoff_p = p + 1;
      result.lamplus = lam;
    }
  }
  return result;
}

// Partitioned form. The scan walks the concatenated spectrum in standardised order, so that the
//   next candidate noise component is the one least remarkable within its own partition's bulk
//   rather than the one with the smallest normalised eigenvalue.
//
// Both of the criterion's two noise estimates need partition-aware treatment:
//
// - sigsq1, the mean of the candidate noise eigenvalues, becomes the pooled energy per degree of
//   freedom. Averaging normalised eigenvalues weights each partition by its component count;
//   weighting by degrees of freedom (k_p qnz_p) is the minimum-variance combination, and the two
//   differ as soon as the partitions have unequal long dimensions.
//
// - sigsq2 estimates sigma^2 from the width of the candidate bulk, using the Marchenko-Pastur
//   relation width = 4 sigma^2 sqrt(gamma). Here gamma is the aspect ratio of the *pooled*
//   candidate set, and that is where an unequal split bites: the pool is a mixture of P bulks of
//   differing width, and the support of a mixture is the support of its widest component alone.
//   Both extremes of the pooled sample therefore come from the single partition of largest
//   gamma_p, so the width must be divided by sqrt(max_p gamma_p). The aggregate ratio
//   sum_p k_p / sum_p qnz_p that this replaces is a weighted mean of the gamma_p, which
//   under-states the width, inflates sigsq2, and so trips the criterion early -- under-counting
//   noise components and over-stating the signal rank.
//   gamma_p increases monotonically with k_p under either version, so the running maximum needs
//   no rescan.
//
// A per-partition sigsq2, averaged over partitions, was considered instead: it would use every
//   partition's edges rather than only the widest, but it also changes the criterion's finite-size
//   operating point for equal partitions (a range over k_p samples in place of one over sum_p k_p),
//   which is a separate change of behaviour from the imbalance correction being made here.
template <ssize_t version>
Result Exp<version>::operator()(const std::vector<Spectrum> &spectra, //
                                const Eigen::Vector3d & /*unused*/) const {
  Result result;
  const std::vector<PooledEigenvalue> pooled = pool_standardized(spectra);
  if (pooled.empty())
    return result;

  std::vector<ssize_t> k(spectra.size(), 0); // candidate noise components taken from each partition
  double noise_sum = 0.0;                    // raw energy of the candidate set
  double noise_dof = 0.0;                    // degrees of freedom of the candidate set
  double lam_min = std::numeric_limits<double>::max();
  double lam_max = std::numeric_limits<double>::lowest();
  double gam_max = 0.0;
  double best_sigma2 = std::numeric_limits<double>::signaling_NaN();
  double zstar = std::numeric_limits<double>::signaling_NaN();
  bool found = false;

  for (const PooledEigenvalue &entry : pooled) {
    const Spectrum &spectrum = spectra[entry.partition];
    const ssize_t kp = ++k[entry.partition];
    noise_sum += spectrum.eigenvalue(entry.index);
    noise_dof += spectrum.dof(1);
    lam_min = std::min(lam_min, entry.lambda);
    lam_max = std::max(lam_max, entry.lambda);
    // Aspect ratio of this partition's own candidate sub-block; version 2 applies the
    //   Cordero-Grande finite-size deflation of its long dimension by the components of that same
    //   partition that lie above the candidate set.
    double denominator = std::numeric_limits<double>::signaling_NaN();
    switch (version) {
    case 1:
      denominator = double(spectrum.qnz());
      break;
    case 2:
      denominator = double(spectrum.qnz() - (spectrum.rnz() - kp));
      break;
    default:
      assert(false);
    }
    gam_max = std::max(gam_max, double(kp) / denominator);
    const double sigsq1 = noise_sum / noise_dof;
    const double sigsq2 = (lam_max - lam_min) / (4.0 * std::sqrt(gam_max));
    // sigsq2 > sigsq1 if signal else noise
    if (sigsq2 < sigsq1) {
      best_sigma2 = sigsq1;
      zstar = entry.z;
      found = true;
    }
  }
  if (!found)
    return result;
  apply_threshold_standardized(spectra, zstar, result);
  result.sigma2 = best_sigma2;
  result.lamplus = zstar * mean_edge(spectra);
  return result;
}

template <ssize_t version>
double Exp<version>::predicted_rmse(const ssize_t m, const ssize_t n, const double r) const {
  const double lm = std::log(double(m));
  const double ln = std::log(double(n));
  const double lr = std::log(std::max(r, 1.0));
  if (version == 1) // Exp1 (Veraart 2016): R^2 = 0.96
    return std::exp(0.02654 - 0.42175 * lm - 0.70154 * ln + 0.73385 * lr + 0.019899 * lm * ln);
  // Exp2 (Cordero-Grande 2019): R^2 = 0.94
  return std::exp(0.48522 - 0.60721 * lm - 0.72413 * ln + 0.68613 * lr + 0.039857 * lm * ln);
}

template class Exp<1>;
template class Exp<2>;

} // namespace MR::Denoise::Estimator
