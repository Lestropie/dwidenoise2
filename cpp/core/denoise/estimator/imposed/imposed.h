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

#include <memory>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"
#include "image.h"

namespace MR::Denoise::Estimator {

// Intermediate base for the "imposed" (non-data-driven) noise-level classes.
//
// Unlike the data-driven estimators (-estimator exp1/exp2/med/mrm2023/tbme2022),
//   which infer the noise level from the patch eigenvalue spectrum, these classes
//   obtain a *known* noise level sigma^2 from an external source -- a fixed scalar
//   (Fixed), an imported image (Import), or the unit level reached after iterative
//   variance-stabilising-transform refinement (Unity) -- and then apply the
//   standard Marchenko-Pastur threshold: the upper bulk edge
//   lambda+ = (1 + sqrt(rnz/qnz))^2 * sigma^2 sets the signal/noise boundary, and
//   the signal rank is the count of sub-threshold eigenvalues.
//
// This shared threshold logic lives here so that the subclasses need only supply
//   the per-patch sigma^2 via get_sigma_sq().
class ImposedSigma : public Base {
public:
  Result operator()(const Spectrum &spectrum, const Eigen::Vector3d &pos) const final;

  // Partitioned form: a single externally-provided sigma^2 applies to the whole patch, but the
  //   signal/noise boundary it implies is each partition's own Marchenko-Pastur edge
  //   sigma^2 (1+sqrt(beta_p))^2, since partitions of unequal volume count have unequal beta_p.
  //   That family of boundaries is exactly the standardised cut z == sigma^2.
  Result operator()(const std::vector<Spectrum> &spectra, const Eigen::Vector3d &pos) const final;

protected:
  // Provide sigma^2 at the patch centre `pos`; return false if it cannot be
  //   determined there (e.g. cubic interpolation outside the image FoV), in which
  //   case the patch is left unprocessed (an invalid Result is returned).
  virtual bool get_sigma_sq(const Eigen::Vector3d &pos, double &sigma_sq) const = 0;
};

// Construct an imposed (non-data-driven) estimator from the -fixed_rank command-line
//   option, or nullptr if it is not supplied (in which case data-driven estimation via
//   make_estimator() is used). Note: -noise_in does NOT yield an imposed estimator; it
//   only seeds the variance-stabilising transform and the schedule then refines the
//   estimate (the vst_noise_in argument is retained for interface symmetry but unused).
std::shared_ptr<Base> make_imposed(Image<float> &vst_noise_in);

} // namespace MR::Denoise::Estimator
