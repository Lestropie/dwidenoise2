/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#pragma once

#include <limits>

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "math/math.h"

namespace MR::Denoise::Estimator {

class MRM2022 : public Base {
public:
  MRM2022() = default;
  Result operator()(const eigenvalues_type &s, const ssize_t m, const ssize_t n) const final {
    Result result;
    const ssize_t mprime = std::min(m, n);
    const ssize_t nprime = std::max(m, n);
    const double sigmasq_to_lamplus = Math::pow2(std::sqrt(nprime) + std::sqrt(mprime));
    double clam = 0.0;
    for (ssize_t i = 0; i != mprime; ++i)
      clam += std::max(s[i], 0.0);
    clam /= nprime;
    // Unlike Exp# code,
    //   MRM2022 article uses p to index number of signal components,
    //   and here doing a direct translation of the manuscript content to code
    double lamplusprev = -std::numeric_limits<double>::infinity();
    for (ssize_t p = 0; p < mprime; ++p) {
      const ssize_t i = mprime - 1 - p;
      const double lam = std::max(s[i], 0.0) / nprime;
      if (lam < lamplusprev)
        return result;
      clam -= lam;
      const double sigmasq = clam / ((mprime - p) * (nprime - p));
      lamplusprev = sigmasq * sigmasq_to_lamplus;
      result.cutoff_p = i;
      result.sigma2 = sigmasq;
    }
    return result;
  }
};

} // namespace MR::Denoise::Estimator
