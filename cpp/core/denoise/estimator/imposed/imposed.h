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
#include <cassert>
#include <cmath>
#include <memory>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"
#include "image.h"
#include "math/math.h"

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
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s, //
                    const ssize_t m,                              //
                    const ssize_t n,                              //
                    const ssize_t rp,                             //
                    const Eigen::Vector3d &pos) const final {     //
    assert(s.size() == std::min(m, n));
    const ssize_t qnz = dimlong_nonzero(m, n, rp);
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t rnz = rank_nonzero(m, n, rp);
    Result result;
    double sigma_sq;
    if (!get_sigma_sq(pos, sigma_sq))
      return result;
    result.sigma2 = sigma_sq;
    // From this noise level,
    //   get the upper bound of the MP distribution and rank of signal
    //   given the ordered list of eigenvalues
    result.lamplus = Math::pow2(1.0 + std::sqrt(double(rnz) / double(qnz))) * result.sigma2;
    result.cutoff_p = rz;
    for (ssize_t p = rz; p != s.size(); ++p) {
      if (s[p] / qnz > result.lamplus)
        break;
      result.cutoff_p = p + 1;
    }
    return result;
  }

  // Partitioned form: a single externally-provided sigma^2 applies to the whole patch; each
  //   partition's signal/noise boundary is the Marchenko-Pastur edge for that partition's own
  //   aspect ratio beta_p (which may differ slightly between partitions).
  Result operator()(const std::vector<eigenvalues_type> &s, //
                    const std::vector<ssize_t> &m,           //
                    const std::vector<ssize_t> &n,           //
                    const std::vector<ssize_t> &rp,          //
                    const Eigen::Vector3d &pos) const final {
    Result result;
    double sigma_sq;
    if (!get_sigma_sq(pos, sigma_sq))
      return result;
    result.sigma2 = sigma_sq;
    const std::vector<PartitionDims> d = partition_dims(m, n, rp);
    std::vector<double> lamplus_p(d.size());
    for (size_t p = 0; p != d.size(); ++p)
      lamplus_p[p] = Math::pow2(1.0 + std::sqrt(d[p].beta)) * sigma_sq;
    double noise_sum = 0.0;
    ssize_t noise_count = 0;
    apply_threshold_per_partition(s, d, lamplus_p, result, noise_sum, noise_count);
    // Representative boundary (normalised units) for export, using the pooled mean aspect ratio.
    result.lamplus = Math::pow2(1.0 + std::sqrt(mean_beta(d))) * sigma_sq;
    return result;
  }

  bool supports_partitioning() const final { return true; }

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
