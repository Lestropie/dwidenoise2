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

#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

// TODO Move to .cpp
template <ssize_t version> class Exp : public Base {
public:
  Exp() = default;
  Result operator()(const eigenvalues_type &s, const ssize_t m, const ssize_t n) const final {
    Result result;
    const ssize_t r = std::min(m, n);
    const ssize_t q = std::max(m, n);
    const double lam_r = std::max(s[0], 0.0) / q;
    double clam = 0.0;
    for (ssize_t p = 0; p < r; ++p) // p+1 is the number of noise components
    {                               // (as opposed to the paper where p is defined as the number of signal components)
      const double lam = std::max(s[p], 0.0) / q;
      clam += lam;
      double denominator = std::numeric_limits<double>::signaling_NaN();
      switch (version) {
      case 1:
        denominator = q;
        break;
      case 2:
        denominator = q - (r - p - 1);
        break;
      default:
        assert(false);
      }
      const double gam = double(p + 1) / denominator;
      const double sigsq1 = clam / double(p + 1);
      const double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
      // sigsq2 > sigsq1 if signal else noise
      if (sigsq2 < sigsq1) {
        result.sigma2 = sigsq1;
        result.cutoff_p = p + 1;
      }
    }
    return result;
  }
};

} // namespace MR::Denoise::Estimator
