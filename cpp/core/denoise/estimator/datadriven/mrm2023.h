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

class MRM2023 : public Base {
public:
  MRM2023() = default;
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final;

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
                    const Eigen::Vector3d & /*unused*/) const final;

  // Predicted relative RMSE of the noise level estimate; calibrated by numerical simulation of
  //   log(RMSE) = c + p*log m + q*log n + s*log r + g*(log m)(log n). R^2 = 0.93.
  double predicted_rmse(const ssize_t m, const ssize_t n, const double r) const final;
};

} // namespace MR::Denoise::Estimator
