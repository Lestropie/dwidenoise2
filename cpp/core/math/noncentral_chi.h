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

#include "types.h"

// The non-central chi distribution: the distribution of the magnitude sqrt(sum_k |z_k|^2) of a
//   sum of squares of "dof" real Gaussian variates of common standard deviation sigma, whose means
//   have combined magnitude nu. In MR it describes magnitude images reconstructed by sum-of-squares
//   combination of dof/2 receive channels; dof == 2 is the Rician distribution (see math/rician.h
//   for its likelihood), and dof == 1 the half-Gaussian.
//
// The degrees of freedom are real-valued, not integral: writing N = dof/2, every expression here is
//   analytically valid for real N > 0. This matters in practice because the effective number of
//   independent channels in sum-of-squares data is in general fractional (noise correlation between
//   channels and overlap of their sensitivities; e.g. Sakaie & Lowe 2017 measured 3.76 effective
//   channels for a 12-element coil).

namespace MR::Math::NonCentralChi {

// Moments in normalised form: as functions of the normalised underlying signal level a = nu / sigma
//   and of the degrees of freedom alone, with the noise level scaled out (mean / sigma and
//   variance / sigma^2 respectively).
//
// - normalised_mean() = sqrt(2) Gamma(N+1/2) / Gamma(N) 1F1(-1/2; N; -a^2/2), evaluated through the
//     Kummer transform as a positive-term series (no cancellation); for large a the asymptote
//     sqrt(a^2 + dof - 1) is used, which carries the correct leading bias and gives variance -> 1.
// - normalised_variance() = (a^2 + dof) - normalised_mean()^2, floored at a small positive value
//     to absorb rounding where the two terms nearly cancel.
//
// Both tend to dof-independent limits at high SNR (mean -> a, variance -> 1): it is only near the
//   noise floor that one value of dof is distinguishable from another, and there the variance
//   carries essentially all of that information.
default_type normalised_mean(const default_type a, const default_type dof);
default_type normalised_variance(const default_type a, const default_type dof);

} // namespace MR::Math::NonCentralChi
