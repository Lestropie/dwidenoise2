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

#include "denoise/estimator/pooling.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "math/math.h"

namespace MR::Denoise::Estimator {

double mean_beta(const std::vector<Spectrum> &spectra) {
  double num = 0.0;
  double den = 0.0;
  for (const auto &spectrum : spectra) {
    num += double(spectrum.rnz()) * spectrum.beta();
    den += double(spectrum.rnz());
  }
  return den > 0.0 ? num / den : 0.0;
}

double mean_edge(const std::vector<Spectrum> &spectra) { return Math::pow2(1.0 + std::sqrt(mean_beta(spectra))); }

double max_beta(const std::vector<Spectrum> &spectra) {
  double result = 0.0;
  for (const auto &spectrum : spectra)
    result = std::max(result, spectrum.beta());
  return result;
}

ssize_t total_rnz(const std::vector<Spectrum> &spectra) {
  ssize_t s = 0;
  for (const auto &spectrum : spectra)
    s += spectrum.rnz();
  return s;
}

ssize_t total_qnz(const std::vector<Spectrum> &spectra) {
  ssize_t s = 0;
  for (const auto &spectrum : spectra)
    s += spectrum.qnz();
  return s;
}

std::vector<PooledEigenvalue> pool_standardized(const std::vector<Spectrum> &spectra) {
  std::vector<PooledEigenvalue> pooled;
  pooled.reserve(size_t(std::max<ssize_t>(0, total_rnz(spectra))));
  for (size_t p = 0; p != spectra.size(); ++p) {
    const Spectrum &spectrum = spectra[p];
    for (ssize_t i = spectrum.rz(); i < spectrum.r(); ++i)
      pooled.push_back({spectrum.standardized(i), spectrum.normalized(i), ssize_t(p), i});
  }
  std::sort(
      pooled.begin(), pooled.end(), [](const PooledEigenvalue &a, const PooledEigenvalue &b) { return a.z < b.z; });
  return pooled;
}

std::vector<double> pool_rescaled(const std::vector<Spectrum> &spectra, const std::vector<double> &divisor) {
  assert(divisor.size() == spectra.size());
  std::vector<double> pooled;
  pooled.reserve(size_t(std::max<ssize_t>(0, total_rnz(spectra))));
  for (size_t p = 0; p != spectra.size(); ++p) {
    const Spectrum &spectrum = spectra[p];
    for (ssize_t i = spectrum.rz(); i < spectrum.r(); ++i)
      pooled.push_back(spectrum.normalized(i) / divisor[p]);
  }
  std::sort(pooled.begin(), pooled.end());
  return pooled;
}

NoiseStats apply_threshold_standardized(const std::vector<Spectrum> &spectra, const double z, Result &result) {
  const size_t P = spectra.size();
  result.cutoff_p_partition.assign(P, 0);
  result.lamplus_partition.assign(P, 0.0);
  NoiseStats stats;
  ssize_t total_cutoff = 0;
  for (size_t p = 0; p != P; ++p) {
    const Spectrum &spectrum = spectra[p];
    const ssize_t c = spectrum.noise_count(z);
    const ssize_t nonzero = c - spectrum.rz();
    stats.sum += spectrum.raw_sum(spectrum.rz(), nonzero);
    stats.dof += spectrum.dof(nonzero);
    stats.count += nonzero;
    stats.z_max = std::max(stats.z_max, spectrum.max_noise_standardized(c));
    result.cutoff_p_partition[p] = c;
    result.lamplus_partition[p] = spectrum.lambda_of(z);
    total_cutoff += c;
  }
  result.cutoff_p = total_cutoff;
  return stats;
}

} // namespace MR::Denoise::Estimator
