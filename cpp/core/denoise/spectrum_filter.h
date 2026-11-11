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

#include <map>

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise {

// Spectral moments of the filter applied to one decomposition: the quantities that summarise what
//   the filter did to the spectrum, as distinct from the filtered data themselves.
//
// The per-partition filters of a partitioned patch occupy disjoint blocks of volume rows, so the
//   composite filter of the patch is their direct sum and every one of these quantities is additive
//   across partitions; that is what operator+= expresses, and it is why the partitioned and
//   single-decomposition paths can share one accumulator.
class Moments {
public:
  Moments()
      : sum_weights(0.0), sum_sqweights(0.0), sum_quadweights(0.0), sum_variance(0.0), sum_eigenvalues(0.0), rank(0) {}

  Moments &operator+=(const Moments &that);

  // Fraction of the decomposition's total eigenvalue energy that the filter discarded.
  double variance_removed() const { return 1.0 - sum_variance / sum_eigenvalues; }

  // First, second and fourth spectral moments sum_i w_i^q of the filter. The first is the
  //   contribution to the effective number of degrees of freedom retained; the latter two feed the
  //   chi-squared-matching degrees-of-freedom measure. For a hard filter all three equal the rank.
  double sum_weights;
  double sum_sqweights;
  double sum_quadweights;
  // Retained eigenvalue energy sum_i w_i lambda_i, on the RAW eigenvalue scale (not normalised by
  //   the long dimension), because it is summed across partitions of differing qnz and only an
  //   energy is additive.
  double sum_variance;
  // Total raw eigenvalue energy sum_i lambda_i presented to the filter, against which the retained
  //   energy is expressed. Held alongside it so that the fraction removed is a property of the
  //   filtering rather than something each consumer re-derives from a spectrum it must keep to hand.
  double sum_eigenvalues;
  // Number of components the filter retained (weight strictly positive; for a hard filter, the
  //   output rank).
  ssize_t rank;
};

// The eigenspectrum filter: given a decomposition's spectrum and the noise level estimated for the
//   patch, the weight to apply to each component.
//
// It is deliberately ignorant of images, patches, aggregation and even of the data matrix -- it
//   acts on real eigenvalues alone -- so that every consumer of a filtered decomposition applies
//   the same filter by construction rather than by parallel implementation. Reconstruction is one
//   such consumer; estimation of the noise model's degrees of freedom, which measures the residual
//   of the filtered reconstruction, is another, and the value it estimates is only meaningful
//   relative to the filter actually in force.
//
// Non-template: unlike the data, the eigenvalues are real whatever the precision or complexity of
//   the input.
//
// Not thread-shared: the optimal-hard-threshold cache below is mutated on use, and the class is
//   held by value in the per-thread functors that own it (one copy per thread, as for the
//   Estimate/Recon state it sits alongside).
class SpectrumFilter {
public:
  explicit SpectrumFilter(const filter_type type) : type_(type) {}

  // Weights for one decomposition, and the spectral moments derived from them.
  // - "spectrum" describes the decomposition: its eigenvalues and every dimension that gives them
  //     meaning.
  // - "threshold" is the noise level estimated for the patch as a whole (a single sigma^2 is shared
  //     by all of its partitions), together with the per-decomposition noise-component count that
  //     the truncation filter cuts at.
  // - "partition" indexes that count: the position of this decomposition within the patch's
  //     partitioning, or -1 for an unpartitioned patch.
  // - "w" is resized as necessary; on return its leading spectrum.r() entries are the weights, in
  //     the same (ascending eigenvalue) order as the spectrum.
  // Returns false where the threshold is unusable, in which case the weights are all unity and the
  //   moments are those of the identity filter, and the caller is to pass the patch through
  //   unchanged rather than project it (reconstructing with unit weights is not the same operation:
  //   it re-synthesises the data through the decomposition instead of copying them).
  bool weights(const Spectrum &spectrum,
               const Estimator::Result &threshold,
               const ssize_t partition,
               vector_type &w,
               Moments &moments) const;

  filter_type type() const { return type_; }

private:
  filter_type type_;
  // Gavish & Donoho optimal hard threshold lambda*(beta), cached per aspect ratio: it is a fixed
  //   function of beta alone, and the number of distinct beta values encountered is small.
  mutable std::map<double, double> beta2lambdastar;
};

} // namespace MR::Denoise
