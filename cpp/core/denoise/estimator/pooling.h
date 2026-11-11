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

// Shared machinery for partition-aware (pooled) noise level estimation.
//
// Each PCA patch is split into P partitions; partition p yields an independent, ascending
// eigenspectrum (a Denoise::Spectrum, which carries with it the dimensions of the decomposition
// from which it came). Partitions are NOT guaranteed to hold equal numbers of volumes, nor to
// regress equal numbers of demeaning-group means: partition_volumes() balances the volume load
// subject to keeping every group it touches at two volumes or more, and a group too small to
// split lands wholly in one partition. So in general m_p, rp_p, and hence
//     rnz_p = min(m_p - rp_p, n),  qnz_p = max(m_p - rp_p, n),  beta_p = rnz_p / qnz_p
// all differ between the partitions of one patch.
//
// Two consequences follow, and between them they dictate everything in this header.
//
// 1. Normalising by qnz_p is necessary but NOT sufficient. It places the eigenvalues on a common
//    scale (lambda = s / qnz_p, with E[lambda] = sigma^2 in every partition irrespective of
//    beta_p), but it does not make them commensurable: partition p's noise occupies
//    [sigma^2 (1-sqrt(beta_p))^2, sigma^2 (1+sqrt(beta_p))^2], so one and the same lambda is
//    unremarkable noise in a wide-bulk partition and conspicuous signal in a narrow one.
//    Concatenating normalised eigenvalues and sorting therefore ranks a large partition's
//    ordinary noise above a small partition's genuine weak signal, and a single normalised
//    threshold cuts each partition at a different point within its own bulk -- systematically
//    over-assigning rank to the large partitions and under-assigning it to the small ones.
//
//    The remedy is to pool and to threshold on the *standardised* eigenvalue
//        z = lambda / (1 + sqrt(beta_p))^2 = Spectrum::standardized()
//    which is free of sigma (so it can order the pool before any noise level is known) and for
//    which the Marchenko-Pastur boundary is the partition-independent value z == sigma^2. One
//    scalar cut z* is simultaneously a per-partition boundary z* (1 + sqrt(beta_p))^2 in
//    normalised units. Every estimator here reduces to the choice of one such z*.
//
// 2. Noise levels must be pooled as energy over degrees of freedom, not as a mean of normalised
//    eigenvalues. The sum of k_p raw eigenvalues of partition p has expectation
//    sigma^2 k_p qnz_p, so
//        sigma^2 = sum_p raw_sum_p / sum_p k_p qnz_p
//    is unbiased and weights each partition by its information content (its number of matrix
//    entries), which is the minimum-variance combination. Averaging normalised eigenvalues
//    instead weights by component count alone; the two agree only when every qnz_p is equal.
//    Both forms collapse to the single-PCA expression when P == 1.
//
// The mixture also forbids evaluating a Marchenko-Pastur shape constant at any single "average"
// beta: the pooled sample is a mixture sum_p w_p MP(beta_p), which is not MP(beta) for any beta.
// Constants that are *moments* average linearly over the mixture, so the weighted mean of the
// per-partition constants is exact (TBME2022's quarter-circle moments; the median that Med
// divides out); constants describing the *support* are set by the widest partition alone, so
// max_beta() is the right one (the bulk width in Exp, the edge difference in TBME2022).
// mean_beta() is consequently for reporting only.

#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"
#include "denoise/spectrum.h"

namespace MR::Denoise::Estimator {

// One entry of the concatenated eigenspectrum, tagged with the partition that produced it.
//   Provenance is not incidental book-keeping: every quantity an estimator forms as it walks the
//   pool -- the running degrees of freedom, the per-partition candidate counts that set the
//   Marchenko-Pastur bulk widths, the deflation of a partition's long dimension by its own peeled
//   components -- is a per-partition quantity, and none of them is recoverable from a bare list
//   of numbers.
struct PooledEigenvalue {
  double z;          // standardised: lambda / (1 + sqrt(beta_p))^2
  double lambda;     // normalised: s / qnz_p
  ssize_t partition; // index into the spectra vector
  ssize_t index;     // index within that partition's eigenvalues
};

// rnz-weighted mean of the per-partition aspect ratios. For reporting only: it is the beta of no
//   actual decomposition, and using it for a Marchenko-Pastur shape constant is the very error
//   this header exists to prevent. Its legitimate use is converting a standardised boundary into
//   a single representative normalised value for export (Result::lamplus), for which it
//   reproduces the single-PCA value exactly when P == 1.
double mean_beta(const std::vector<Spectrum> &spectra);

// Ratio of the representative Marchenko-Pastur edge to the noise level, for converting a
//   standardised quantity into one normalised value for export.
double mean_edge(const std::vector<Spectrum> &spectra);

// Largest per-partition aspect ratio. The support of the pooled mixture is that of the widest
//   partition alone -- (1 + sqrt(beta))^2 increases and (1 - sqrt(beta))^2 decreases with beta,
//   so both extremes come from the same partition -- so any statistic formed from the observed
//   *range* of the pool (a Marchenko-Pastur bulk width 4 sigma^2 sqrt(beta), or a quarter-circle
//   edge difference) must use this rather than an average.
double max_beta(const std::vector<Spectrum> &spectra);

ssize_t total_rnz(const std::vector<Spectrum> &spectra);

ssize_t total_qnz(const std::vector<Spectrum> &spectra);

// The concatenated nonzero eigenspectrum, ascending in standardised value z, each entry tagged
//   with its partition of origin.
std::vector<PooledEigenvalue> pool_standardized(const std::vector<Spectrum> &spectra);

// The concatenated nonzero eigenspectrum in normalised units, each partition's values divided by
//   a per-partition constant, sorted ascending. For the case where the statistic of interest is
//   not the bulk edge but some other feature of each partition's law whose position depends on
//   beta_p: in practice Med, which divides by the Marchenko-Pastur median mu(beta_p) so that
//   every partition contributes values whose own median is sigma^2.
std::vector<double> pool_rescaled(const std::vector<Spectrum> &spectra, const std::vector<double> &divisor);

// Pooled noise statistics accumulated over a classification: the raw eigenvalue energy of the
//   noise components and the degrees of freedom they carry, from which the noise level follows as
//   sigma^2 = sum / dof (see note 2 at the head of this file).
struct NoiseStats {
  double sum = 0.0;  // sum_p sum_{i in noise_p} s_{p,i}, ie. energy
  double dof = 0.0;  // sum_p k_p qnz_p
  ssize_t count = 0; // sum_p k_p, counting nonzero noise components only
  bool empty() const { return count == 0; }
  double sigma2() const { return sum / dof; }
};

// Classify every partition against the standardised cut z, which is simultaneously a boundary
//   z (1 + sqrt(beta_p))^2 in each partition's own normalised units. Fills the per-partition
//   noise counts and boundaries and the pooled noise count of "result", and returns the pooled
//   energy and degrees of freedom of the nonzero noise components (the rz_p components that
//   demeaning forces to zero count as noise, as in the single-PCA estimators, but carry no energy
//   and no degrees of freedom).
//
// This one function replaces both of the thresholding paths that preceded it: applying a single
//   normalised threshold to every partition (correct only where every beta_p is equal), and
//   applying an explicitly-constructed vector of per-partition Marchenko-Pastur edges (which is
//   what the choice z == sigma^2 amounts to here).
NoiseStats apply_threshold_standardized(const std::vector<Spectrum> &spectra, const double z, Result &result);

} // namespace MR::Denoise::Estimator
