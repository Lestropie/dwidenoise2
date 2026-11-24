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

namespace MR::Math {

// Special functions related to the Gamma function and the Gamma distribution: the digamma and
//   trigamma functions and the inverse of the digamma function (as required to fit a Gamma
//   distribution by maximum likelihood), and the regularised lower and upper incomplete gamma
//   functions together with their inverses (the cumulative distribution function, survival
//   function and quantiles of a Gamma distribution).
//
// WHY THESE ARE WRITTEN HERE RATHER THAN TAKEN FROM A LIBRARY. No inverse regularised incomplete
//   gamma exists in the C++ standard library, in MRtrix3, or in Eigen (supported or unsupported
//   modules alike), so it has to be written; and writing it requires P(alpha, x) and Q(alpha, x)
//   for the iteration that refines the initial estimate. Once those are in hand, the digamma and
//   trigamma functions are the smaller half of what remains. Eigen's unsupported/SpecialFunctions
//   module (lgamma, digamma, polygamma, igamma, igammac) is available to any MRtrix3 build: Eigen
//   3.4 is required, and core already includes the module unconditionally for that version
//   (dwi/tractography/tracking/early_exit.h). Taking these functions from it would nonetheless
//   gain nothing. In Eigen 3.4 none of them is vectorised on the CPU (the HasLGamma, HasDiGamma,
//   HasIGamma, HasIGammac and HasPolygamma packet traits are 0 for SSE, AVX, AVX-512 and NEON), so
//   applied to an array they evaluate the same scalar series per element; they are in any case
//   called here a handful of times per group of voxels and not once per voxel; and the quantile
//   would still have to be written. (MRTRIX_HAVE_EIGEN_UNSUPPORTED_SPECIAL_FUNCTIONS, which
//   math/betainc.h and math/zstatistic.cpp test, is a remnant of the configure script that preceded
//   CMake and is no longer defined by anything, so those files always take their hand-written
//   paths.)
//
// Everything below is real-valued and defined for strictly positive arguments; each function
//   documents what it does with an argument outside its domain. All are re-entrant (the only
//   shared state is the lgamma mutex that MRtrix3 core already maintains for the same purpose).
//   Code using log-gamma anywhere that may run concurrently should call log_gamma() below rather
//   than std::lgamma().

// log |Gamma(x)|, safe to call concurrently from multiple threads (unlike std::lgamma(), which
//   writes the sign of Gamma(x) to a global). Uses the re-entrant lgamma_r() where MRtrix3's CMake
//   has detected it (MRTRIX_HAVE_LGAMMA_R), and otherwise serialises std::lgamma() on the mutex
//   that MRtrix3 core provides for the same purpose (Math::mutex_lgamma, math/betainc.h).
default_type log_gamma(const default_type x);

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
//   this way.
default_type inverse_digamma(const default_type y);

// P(alpha, x): the regularised LOWER incomplete gamma function, i.e. the cumulative distribution
//   function of Gamma(alpha, 1) evaluated at x. Zero for x <= 0; alpha must be positive.
// Series representation for x < alpha + 1 and a Lentz continued fraction for the complement
//   otherwise, which is where each converges quickly.
default_type gamma_p(const default_type alpha, const default_type x);

// Q(alpha, x) = 1 - P(alpha, x): the regularised UPPER incomplete gamma function, i.e. the
//   survival function of Gamma(alpha, 1). One for x <= 0; alpha must be positive.
// Computed directly rather than as 1 - P, so that it retains full relative precision in the upper
//   tail, where Q is small and 1 - P would be dominated by the rounding of P.
default_type gamma_q(const default_type alpha, const default_type x);

// P^{-1}(alpha, p): the quantile of Gamma(alpha, 1), i.e. the x for which P(alpha, x) == p.
// Zero at p == 0 and +infinity at p == 1; alpha must be positive and p must lie in [0, 1].
// The Wilson-Hilferty cube-root normal approximation provides the initial estimate for
//   alpha >= 1 and the leading-order inversion x ~ (p Gamma(alpha+1))^(1/alpha) for smaller alpha;
//   bracketed Newton iteration on the logarithm of the smaller of the two tail probabilities
//   (log P(alpha, x) in the lower half of the distribution, log Q(alpha, x) in the upper half) then
//   takes it to machine precision.
default_type gamma_p_inv(const default_type alpha, const default_type p);

// Q^{-1}(alpha, q): the x for which Q(alpha, x) == q, i.e. the quantile of Gamma(alpha, 1) at
//   probability 1 - q. +infinity at q == 0 and zero at q == 1; alpha must be positive and q must
//   lie in [0, 1].
// The same iteration as gamma_p_inv(), but given the upper-tail probability itself, which is
//   therefore not first rounded by forming 1 - q: an upper-tail quantile should be requested
//   through this function rather than as gamma_p_inv(alpha, 1 - q).
default_type gamma_q_inv(const default_type alpha, const default_type q);

} // namespace MR::Math
