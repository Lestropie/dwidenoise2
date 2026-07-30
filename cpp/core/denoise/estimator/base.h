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

#include <cassert>
#include <limits>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise::Estimator {

class Base {
public:
  Base() = default;
  Base(const Base &) = delete;
  virtual ~Base() = default;
  virtual void update_vst_image(Image<float> &) {}
  // Single-PCA (non-partitioned) noise level estimation from the eigenspectrum of the whole
  //   patch, ie. the single partition spanning every volume of the series.
  // pos = realspace position of the centre of the patch
  virtual Result operator()(const Spectrum &spectrum,              //
                            const Eigen::Vector3d &pos) const = 0; //

  // Partitioned noise level estimation: the patch is split into P partitions, each of which
  //   contributes the independent eigenspectrum of its own decomposition (see Spectrum). The
  //   pooled spectrum yields a single noise level (Result::sigma2) shared by the patch, with a
  //   per-partition signal/noise cutoff (Result::cutoff_p_partition). Every estimator must
  //   implement this: partitioning is a property of the schedule, not of the estimator, and
  //   callers are free to combine any -estimator with any partition count.
  virtual Result operator()(const std::vector<Spectrum> &spectra,  //
                            const Eigen::Vector3d &pos) const = 0; //

  // Predicted relative RMSE of this estimator's noise level estimate for a Casorati matrix of m
  //   volumes, n voxels and signal rank r (sigma = 1). Used by the RMSE-tolerance kernel to decide
  //   how large to grow the patch. Data-driven estimators override this with a calibrated model
  //   (log RMSE = c + p*log m + q*log n + s*log r + g*log m*log n; coefficients from numerical
  //   simulation across m, n and signal-rank density). The default (non-data-driven estimators,
  //   e.g. imposed sigma / unity) returns NaN: such estimators cannot size an RMSE kernel.
  virtual double predicted_rmse(const ssize_t /*m*/, const ssize_t /*n*/, const double /*r*/) const {
    return std::numeric_limits<double>::quiet_NaN();
  }
};

} // namespace MR::Denoise::Estimator
