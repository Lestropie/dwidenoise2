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

class MRM2023 : public Base {
public:
  MRM2023() = default;
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final {
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

  // Partitioned form: the peel-off recursion of the single-PCA path, run over the concatenated
  //   spectrum of all partitions. Two of its three ingredients are per-partition quantities once
  //   the partitions are not of equal size.
  //
  // - The order of peeling. Components are removed most-significant-first, and significance means
  //   position within one's own bulk, not raw magnitude; the scan therefore descends the
  //   standardised value z rather than the normalised eigenvalue.
  //
  // - The stopping test. The manuscript's test is s_max < sigma^2 (sqrt(n') + sqrt(m'))^2, which
  //   in normalised units is lambda < sigma^2 (1 + sqrt(beta))^2 and hence, on the standardised
  //   scale, simply z < sigma^2 -- whatever beta may be. Testing the largest remaining z against
  //   sigma^2 thus compares every partition against its own Marchenko-Pastur edge simultaneously,
  //   where a single pooled edge built from an average beta would cut the wide-bulk partitions too
  //   low and the narrow-bulk ones too high.
  //
  // - The noise level. sigma^2 = clam / ((m'-p)(n'-p)) is an energy divided by the degrees of
  //   freedom that remain after p components have been peeled off, each removal costing one
  //   direction of the long dimension. Generalised across partitions this is
  //       sigma^2 = sum_p raw_sum_p / sum_p k_p (qnz_p - p_p)
  //   with each partition deflated by the number of components peeled from *it*. Deflating one
  //   notional aggregate dimension sum_p qnz_p by the total peel count instead is exact only while
  //   the peeling happens to be distributed evenly between partitions.
  Result operator()(const std::vector<Spectrum> &spectra, //
                    const Eigen::Vector3d & /*unused*/) const final {
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
      result.lamplus = 0.0;
      result.cutoff_p_partition.assign(spectra.size(), 0);
      result.lamplus_partition.assign(spectra.size(), 0.0);
      return result;
    }
    apply_threshold_standardized(spectra, zstar, result);
    result.sigma2 = sigma2_found;
    // Marchenko-Pastur edge (normalised) at the representative aspect ratio, as in the single-PCA
    //   path; the per-partition boundaries actually applied are in Result::lamplus_partition.
    result.lamplus = sigma2_found * mean_edge(spectra);
    return result;
  }

  bool supports_partitioning() const final { return true; }

  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.93.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final {
    const double lm = std::log(double(m));
    const double ln = std::log(double(n));
    const double lr = std::log(std::max(r, 1.0));
    return std::exp(3.07484 - 1.01543 * lm - 1.16065 * ln + 0.69571 * lr + 0.088552 * lm * ln);
  }
};

} // namespace MR::Denoise::Estimator
