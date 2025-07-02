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

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

class Rank : public Base {
public:
  Rank(const ssize_t r) : rank(r) {}
  Result operator()(const Eigen::VectorBlock<eigenvalues_type> s,     //
                    const ssize_t m,                                  //
                    const ssize_t n,                                  //
                    const ssize_t rp,                                 //
                    const Eigen::Vector3d & /*unused*/) const final { //
    assert(s.size() == std::min(m, n));
    const ssize_t rz = rank_zero(m, n, rp);
    const ssize_t rnz = rank_nonzero(m, n, rp);
    const ssize_t qnz = dimlong_nonzero(m, n, rp);
    Result result;
    // Bear in mind that any assumed-zero singular values "rz" due to preconditioning "rp"
    //   must be assumed to contribute to the rank
    if (rnz == rank) {
      // All components contribute (even the assumed-zero ones)
      result.cutoff_p = 0;
      result.lamplus = 0.0;
      result.sigma2 = 0.0;
    } else if (rnz > rank) {
      result.cutoff_p = s.size() - (rank - rz);
      result.sigma2 = s.segment(rz, result.cutoff_p - rz).sum() / (qnz * (result.cutoff_p + 1 - rz));
      result.lamplus = s[result.cutoff_p - 1] / qnz;
    } // If requested rank is greater than available rank, leave "result" completely uninitialised
    return result;
  }

protected:
  const ssize_t rank;
};

} // namespace MR::Denoise::Estimator
