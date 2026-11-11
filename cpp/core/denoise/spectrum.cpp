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

#include "denoise/spectrum.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "math/math.h"

namespace MR::Denoise {

Spectrum::Spectrum(const eigenvalues_type &storage, const ssize_t m, const ssize_t n, const ssize_t rp)
    : s(storage.data(), std::min(m, n)),  //
      m_(m),                              //
      n_(n),                              //
      rp_(rp),                            //
      r_(std::min(m, n)),                 //
      rz_(rank_zero(m, n, rp)),           //
      rnz_(rank_nonzero(m, n, rp)),       //
      qnz_(dimlong_nonzero(m, n, rp)),    //
      beta_(double(rnz_) / double(qnz_)), //
      sqrtbeta_(std::sqrt(beta_)),        //
      edge_(Math::pow2(1.0 + sqrtbeta_)) {
  assert(storage.size() >= r_);
  assert(rz_ + rnz_ == r_);
}

ssize_t Spectrum::noise_count(const double z) const {
  ssize_t c = rz_;
  for (ssize_t i = rz_; i != r_; ++i) {
    if (standardized(i) > z)
      break;
    c = i + 1;
  }
  return c;
}

double Spectrum::raw_sum(const ssize_t first, const ssize_t count) const {
  assert(first >= 0 && count >= 0 && first + count <= r_);
  return count > 0 ? s.segment(first, count).sum() : 0.0;
}

} // namespace MR::Denoise
