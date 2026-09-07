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

#include "denoise/estimator/imposed/imposed.h"

#include <memory>
#include <string>
#include <vector>

#include "denoise/estimator/imposed/rank.h"
#include "denoise/estimator/pooling.h"

#include "app.h"
#include "exception.h"

namespace MR::Denoise::Estimator {

using namespace App;

Result ImposedSigma::operator()(const Spectrum &spectrum, const Eigen::Vector3d &pos) const {
  Result result;
  double sigma_sq;
  if (!get_sigma_sq(pos, sigma_sq))
    return result;
  result.sigma2 = sigma_sq;
  // From this noise level,
  //   get the upper bound of the MP distribution and rank of signal
  //   given the ordered list of eigenvalues.
  // On the standardised scale the Marchenko-Pastur boundary is the noise level itself, so the
  //   known sigma^2 doubles as the cut and lambda+ need only be formed for export.
  result.cutoff_p = spectrum.noise_count(sigma_sq);
  result.set_boundaries(spectrum.edge(), spectrum.max_noise_standardized(result.cutoff_p));
  return result;
}

Result ImposedSigma::operator()(const std::vector<Spectrum> &spectra, const Eigen::Vector3d &pos) const {
  Result result;
  double sigma_sq;
  if (!get_sigma_sq(pos, sigma_sq))
    return result;
  result.sigma2 = sigma_sq;
  const NoiseStats stats = apply_threshold_standardized(spectra, sigma_sq, result);
  // Representative boundary (normalised units) for export, using the pooled mean aspect ratio.
  result.set_boundaries(mean_edge(spectra), stats.z_max);
  return result;
}

// Only -fixed_rank bypasses data-driven estimation (imposing a fixed signal rank).
//   -noise_in no longer bypasses estimation: it merely seeds the variance-stabilising
//   transform (the first iteration's noise level), after which the schedule's data-driven
//   estimator refines the estimate. The "apply a known noise map without re-estimating"
//   behaviour is instead obtained via a single-row schedule with update_noise = false.
std::shared_ptr<Base> make_imposed(Image<float> & /*vst_noise_in*/) {
  auto fixed_rank = get_options("fixed_rank");
  if (fixed_rank.empty())
    return nullptr;
  if (!get_options("estimator").empty())
    throw Exception("Cannot both provide an input signal rank and specify a noise level estimator");
  return std::make_shared<Rank>(fixed_rank[0][0]);
}

} // namespace MR::Denoise::Estimator
