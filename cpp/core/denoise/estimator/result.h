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

#include <cmath>
#include <vector>

#include "types.h"

namespace MR::Denoise::Estimator {

class Result {
public:
  Result();
  // A noise level of zero is not a usable threshold: lambda+ = (1 + sqrt(beta))^2 * sigma^2 is then
  //   zero, so every component is classified as signal, and the optimal-shrinkage weights divide by
  //   sigma2 (yielding NaN weights, and hence NaN output data in a build without assertions).
  //   Estimators signal "no boundary could be found" by returning sigma2 == 0 (MRM2023's
  //   fall-through when the peel-off scan classifies every component as signal;
  //   Estimator::Rank when the imposed rank equals the available rank), so a strictly positive
  //   sigma2 is required here. Recon and Estimate then take their documented conservative path
  //   for an unusable threshold: copy the input data through unchanged.
  operator bool() const {
    return cutoff_p >= 0                               //
           && std::isfinite(sigma2) && sigma2 > 0.0    //
           && std::isfinite(edge) && edge > 0.0        //
           && std::isfinite(lamplus) && lamplus >= 0.0 //
           && std::isfinite(lammax_noise) && lammax_noise >= 0.0;
  }
  bool operator!() const { return !bool(*this); }
  // From dwidenoise code / estimator::Exp :
  //   cutoff_p is the *number of noise components considered to be part of the MP distribution*.
  // For the partitioned path cutoff_p carries the pooled total noise count across partitions,
  //   and sigma2 is the single noise level shared by the patch.
  ssize_t cutoff_p;
  // Noise variance on the normalised eigenvalue scale, and PER CHANNEL: for complex input the
  //   eigenvalues reaching the estimator have already been halved (Estimate::to_per_channel), so
  //   sqrt(sigma2) is the standard deviation of one real channel and not that of the complex sample
  //   as a whole. This is the currency in which every noise level in this software is expressed.
  double sigma2;

  // The two members below are BOTH "the top of the noise part of the eigenspectrum", and they are
  //   not the same number. They were formerly conflated in a single "lamplus" field, which
  //   different estimators filled with one or the other; they are now stored separately, and every
  //   estimator writes both.
  //
  //   - lamplus is the THEORETICAL boundary: the Marchenko-Pastur upper bulk edge implied by the
  //     estimated noise level and the shape of the Casorati matrix. It is a property of the model,
  //     and carries no information beyond sigma2 and edge.
  //   - lammax_noise is the EMPIRICAL boundary: the largest eigenvalue that this result's
  //     classification actually places in the noise part of the spectrum. It is a property of the
  //     data, is bounded above by the largest eigenvalue present, and coincides with lamplus only
  //     to the resolution of the discrete spectrum (and only for those estimators that cut at the
  //     edge at all: Exp cuts where its two noise-variance estimates cross, and Rank cuts wherever
  //     the imposed rank falls, so for those two the empirical boundary is under no obligation to
  //     sit anywhere near lambda+).
  //
  // Both are on the NORMALISED eigenvalue scale (Spectrum::normalized(), lambda = s / qnz), the
  //   same scale as sigma2, so that lammax_noise and lamplus are directly comparable with one
  //   another within a patch. Dividing either by "edge" places it on the beta-free STANDARDISED
  //   scale on which patches of differing aspect ratio are comparable (Spectrum::standardized()).

  // Ratio of the Marchenko-Pastur upper bulk edge to the noise level for the decomposition(s) this
  //   result describes: Spectrum::edge() in the single-PCA path, and the representative
  //   (rnz-weighted mean-beta) edge Estimator::mean_edge() in the partitioned one. Stored rather
  //   than recovered as lamplus / sigma2, because a consumer needing to place either boundary on a
  //   beta-independent scale can only do so by dividing by it, and because the fall-through
  //   "no usable threshold" results have sigma2 == 0.
  double edge;
  // Theoretical boundary: lamplus == sigma2 * edge, always.
  double lamplus;
  // Empirical boundary: the largest normalised eigenvalue classified as noise, or zero where the
  //   noise set holds no nonzero component.
  double lammax_noise;
  // Partitioned results (empty for the single-PCA path; otherwise length P). With the patch
  //   split into P partitions, a single sigma2 applies but the signal rank may differ per
  //   partition: cutoff_p_partition[p] is the number of noise components in partition p (its
  //   smallest cutoff_p_partition[p] eigenvalues, ascending, are noise; a partition may be all
  //   noise, i.e. rank 0), and lamplus_partition[p] is the (normalised) boundary that the pooled
  //   standardised cut denotes in partition p. Note that the latter is the boundary APPLIED, which
  //   is the theoretical lambda+ of that partition only for those estimators whose cut is sigma2
  //   itself; it is retained for diagnostic completeness and currently has no reader.
  std::vector<ssize_t> cutoff_p_partition;
  std::vector<double> lamplus_partition;

  // Number of noise components in one decomposition of the patch: partition "partition", or the
  //   patch's single decomposition where that index is -1 (for which the pooled cutoff_p is that
  //   decomposition's own count). Consumers of a decomposition are written once against this,
  //   rather than once per partitioning state against the two members separately.
  ssize_t cutoff(const ssize_t partition) const;

  // Record the Marchenko-Pastur edge of the decomposition(s) from which this result came, and with
  //   it both signal/noise boundaries, from the standardised value z_max_noise of the largest
  //   component classified as noise (Spectrum::max_noise_standardized() /
  //   Estimator::NoiseStats::z_max). sigma2 must already have been set; this is the one place at
  //   which lamplus and lammax_noise are formed, so that no estimator can populate one and leave
  //   the other stale, inconsistently scaled, or signalling NaN.
  void set_boundaries(const double mp_edge, const double z_max_noise);

  // The empirical boundary on the beta-free standardised scale, ie. expressed in units of sigma^2:
  //   comparable across patches, kernels and iterations whose Casorati aspect ratio differs, and
  //   equal to sigma2 exactly when the classification cut falls on the Marchenko-Pastur edge.
  double lammax_noise_standardized() const;
};

} // namespace MR::Denoise::Estimator
