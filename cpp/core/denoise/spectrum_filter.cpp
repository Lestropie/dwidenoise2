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

#include "denoise/spectrum_filter.h"

#include <cassert>
#include <cmath>

#include "math/math.h"

namespace MR::Denoise {

Moments &Moments::operator+=(const Moments &that) {
  sum_weights += that.sum_weights;
  sum_sqweights += that.sum_sqweights;
  sum_quadweights += that.sum_quadweights;
  sum_variance += that.sum_variance;
  sum_eigenvalues += that.sum_eigenvalues;
  rank += that.rank;
  return *this;
}

bool SpectrumFilter::weights(const Spectrum &spectrum,
                             const Estimator::Result &threshold,
                             const ssize_t partition,
                             vector_type &w,
                             Moments &moments) const {

  const ssize_t r = spectrum.r();
  const ssize_t rz = spectrum.rz();
  const ssize_t qnz = spectrum.qnz();
  const double beta = spectrum.beta();
  const auto &s = spectrum.eigenvalues();
  if (r > w.size())
    w.resize(r);

  moments = Moments();
  moments.sum_eigenvalues = s.sum();

  // Erring on the conservative side:
  //   If the decomposition fails, or a threshold can't be found,
  //   the caller copies the input data to the output data as-is,
  //   regardless of whether performing overcomplete local PCA
  if (!threshold) {
    w.head(r).setOnes();
    moments.rank = r;
    moments.sum_weights = moments.sum_sqweights = moments.sum_quadweights = double(r);
    moments.sum_variance = moments.sum_eigenvalues;
    return false;
  }

  switch (type_) {
  case filter_type::OPTSHRINK: {
    w.head(rz).setZero();
    const double transition = 1.0 + spectrum.sqrt_beta();
    for (ssize_t i = rz; i != r; ++i) {
      // TODO For non-binary determination of weights for optimal shrinkage,
      //   should the expression be identical between BDCSVD and SelfAdjointEigenSolver?
      //   Or eg. is one equivalent to scaling singular values whereas the other is equivalent to scaling eigenvalues?
      const double lam = spectrum.normalized(i);
      const double y = std::sqrt(lam / threshold.sigma2);
      // const double y = lam / threshold.sigma2;
      double nu = 0.0;
      if (y > transition) {
        // Occasionally floating-point precision will drive this calculation to fractionally greater than y,
        //   which will erroneously yield a weight fractionally greater than 1.0
        nu = std::min(y, std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y);
        ++moments.rank;
      }
      w[i] = lam > 0.0 ? (nu / y) : 0.0;
      // Precondition: threshold.sigma2 > 0, guaranteed by Estimator::Result::operator bool()
      //   under which this branch is entered. A zero sigma2 would give y = inf and hence a NaN
      //   weight here; that is an unusable threshold, not a shrinkage weight, and is rejected as
      //   an invalid Result rather than caught by this assertion.
      assert(w[i] >= 0.0 && w[i] <= 1.0);
      moments.sum_weights += w[i];
      moments.sum_sqweights += Math::pow2(w[i]);
      moments.sum_quadweights += Math::pow4(w[i]);
      moments.sum_variance += w[i] * s[i];
    }
  } break;
  case filter_type::OPTTHRESH: {
    const std::map<double, double>::const_iterator it = beta2lambdastar.find(beta);
    double lambda_star = 0.0;
    if (it == beta2lambdastar.end()) {
      lambda_star =
          sqrt(2.0 * (beta + 1.0) + ((8.0 * beta) / (beta + 1.0 + std::sqrt(Math::pow2(beta) + 14.0 * beta + 1.0))));
      beta2lambdastar[beta] = lambda_star;
    } else {
      lambda_star = it->second;
    }
    // Gavish & Donoho (2014) optimal hard threshold, tau* = lambda*(beta) sqrt(qnz) sigma. It is
    //   a threshold on *singular values*; the spectrum holds their squares, so the two are compared
    //   as tau*^2 = lambda*(beta)^2 qnz sigma^2. Equivalently, and consistently with the optimal
    //   shrinkage branch above, this retains a component iff its scaled singular value
    //   y = sqrt(lambda / sigma^2) exceeds lambda*(beta).
    const double tau_star = lambda_star * std::sqrt(double(qnz)) * std::sqrt(threshold.sigma2);
    const double cut = Math::pow2(tau_star);
    w.head(rz).setZero();
    for (ssize_t i = rz; i != r; ++i) {
      if (s[i] >= cut) {
        w[i] = 1.0;
        ++moments.rank;
        moments.sum_variance += s[i];
      } else {
        w[i] = 0.0;
      }
    }
    // Hard filter: every retained weight is exactly 1.0, so all spectral moments equal the rank.
    moments.sum_weights = moments.sum_sqweights = moments.sum_quadweights = double(moments.rank);
  } break;
  case filter_type::TRUNCATE: {
    const ssize_t cutoff = threshold.cutoff(partition);
    moments.rank = r - cutoff;
    w.head(cutoff).setZero();
    w.segment(cutoff, moments.rank).setOnes();
    moments.sum_weights = moments.sum_sqweights = moments.sum_quadweights = double(moments.rank);
    moments.sum_variance = w.head(r).matrix().dot(s.matrix());
  } break;
  default:
    assert(false);
  }
  assert(std::isfinite(moments.sum_weights));
  assert(w.head(r).allFinite());
  return true;
}

} // namespace MR::Denoise
