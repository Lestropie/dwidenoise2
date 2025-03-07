/* Required Notice: Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;
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

#include <string>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

// This class assumes that in a prior iteration,
//   a noise level image has been computed,
//   and that image is being used for both variance-stabilising transform
//   and as a noise level estimate
// Where this occurs,
//   the levels for the a priori noise level estimate and the VST are always identical,
//   and so sigma^2 == 1.0 always
class Unity : public Base {
public:
  Unity() {}
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s, //
                    const ssize_t m,                              //
                    const ssize_t n,                              //
                    const ssize_t rp,                             //
                    const Eigen::Vector3d &pos) const final {     //
    assert(s.size() == r);
    const ssize_t qnz = dimlong_nonzero(m, n, rp);
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t rnz = rank_nonzero(m, n, rp);
    Result result;
    result.sigma2 = 1.0;
    // From this noise level,
    //   get the upper bound of the MP distribution and rank of signal
    //   given the ordered list of eigenvalues
    result.lamplus = Math::pow2(1.0 + std::sqrt(double(rnz) / double(qnz)));
    result.cutoff_p = rz;
    for (ssize_t p = rz; p != s.size(); ++p) {
      if (s[p] / qnz > result.lamplus)
        break;
      result.cutoff_p = p + 1;
    }

    return result;
  }
};

} // namespace MR::Denoise::Estimator
