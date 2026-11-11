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

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise::Estimator {

class Rank : public Base {
public:
  Rank(const ssize_t r) : rank(r) {}
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const final;

  // Partitioned form: the imposed (total) signal rank is honoured across the whole patch by
  //   choosing the corresponding ordered element of the concatenated eigenspectrum as the
  //   signal/noise boundary; the per-partition rank then follows from each partition's own
  //   eigenvalues (a partition may receive rank 0). Mirrors the single-PCA convention in which
  //   the imposed rank counts the assumed-zero (preconditioner) components.
  //
  // The ordering is by standardised value rather than by normalised eigenvalue, and here that
  //   matters more than anywhere else: an imposed rank is a fixed budget, so whatever one
  //   partition gains another must lose. Ranking by raw magnitude would place a wide-bulk
  //   partition's ordinary noise above a narrow-bulk partition's genuine weak signal and hand the
  //   budget to the former; ranking by position within each partition's own bulk spends it on the
  //   components that are actually the most significant.
  Result operator()(const std::vector<Spectrum> &spectra, //
                    const Eigen::Vector3d & /*unused*/) const final;

protected:
  const ssize_t rank;
};

} // namespace MR::Denoise::Estimator
