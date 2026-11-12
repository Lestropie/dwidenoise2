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

namespace MR::Denoise::Precondition::NoiseModel::GammaStats {

// Special functions of the Gamma distribution required by the estimation of the magnitude noise
//   model's degrees of freedom from background voxels (background.h).
//
// WHY THESE ARE WRITTEN HERE RATHER THAN TAKEN FROM A LIBRARY. The estimator's central operation
//   is the Gamma quantile P^{-1}(alpha, p), which fixes the interval within which a voxel's summed
//   statistic must fall for that voxel to be accepted as background. No inverse regularised
//   incomplete gamma exists in the C++ standard library, in MRtrix3, or in Eigen (supported or
//   unsupported modules alike), so it has to be written; and writing it requires P(alpha, x) for
//   the iteration that refines the initial estimate. Once P is in hand, the digamma and trigamma
//   functions that the maximum-likelihood estimator needs are the smaller half of what remains.
//   Eigen's unsupported/SpecialFunctions module does provide digamma / polygamma / igamma, but
//   MRtrix3's CMake never defines MRTRIX_HAVE_EIGEN_UNSUPPORTED_SPECIAL_FUNCTIONS and every use of
//   that module in MRtrix3 core is paired with a hand-written fallback; depending on it
//   unconditionally from here would break the configuration core takes care to support, and would
//   in any case not supply the one function that cannot be done without.
//
// Everything below is real-valued and defined for strictly positive arguments; each function
//   documents what it does with an argument outside its domain. All are re-entrant (the only
//   shared state is the lgamma mutex that MRtrix3 core already maintains for the same purpose).

// psi(x) = d/dx log Gamma(x), for x > 0.
// Evaluated by the recurrence psi(x) = psi(x+1) - 1/x until the argument is large enough for the
//   asymptotic Bernoulli series to be accurate to machine precision.
default_type digamma(const default_type x);

// psi'(x), the trigamma function, for x > 0.
// Same construction: the recurrence psi'(x) = psi'(x+1) + 1/x^2, then the asymptotic series.
default_type trigamma(const default_type x);

// The inverse of psi: the unique x > 0 with psi(x) == y, for any real y.
// Minka's approximation supplies the starting value, and Newton's method with psi' refines it;
//   the shape parameter of a Gamma distribution is recovered from its log-likelihood equations
//   this way (background.h, estimator L).
default_type inverse_digamma(const default_type y);

// P(alpha, x): the regularised LOWER incomplete gamma function, i.e. the cumulative distribution
//   function of Gamma(alpha, 1) evaluated at x. Zero for x <= 0; alpha must be positive.
// Series representation for x < alpha + 1 and a Lentz continued fraction for the complement
//   otherwise, which is where each converges quickly.
default_type gamma_p(const default_type alpha, const default_type x);

// P^{-1}(alpha, p): the quantile of Gamma(alpha, 1), i.e. the x for which P(alpha, x) == p.
// Zero at p == 0 and +infinity at p == 1; alpha must be positive and p must lie in [0, 1].
// The Wilson-Hilferty cube-root normal approximation provides the initial estimate for
//   alpha >= 1 and the leading-order inversion x ~ (p Gamma(alpha+1))^(1/alpha) for smaller alpha;
//   bracketed Halley iteration on P(alpha, x) - p then takes it to machine precision.
//
// NAMING. It is the inverse of P, and not of the complement Q = 1 - P, so it is named for P; the
//   spelling "gamma_q_inv" is used for this function elsewhere in the literature and is misleading.
default_type gamma_p_inv(const default_type alpha, const default_type p);

} // namespace MR::Denoise::Precondition::NoiseModel::GammaStats
