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

// Shared helpers for partition-aware (pooled) noise level estimation.
//
// Each PCA patch is split into P partitions; partition p yields an independent, ascending
// eigenspectrum eigenvalues[p] of length r_p = min(m_p, n_p). Because raw eigenvalue magnitudes
// scale with the partition's long dimension qnz_p, partitions are pooled only after normalising
// each eigenvalue by its own qnz_p (lambda = s_p[i] / qnz_p). With the partitions sized so that
// each shares a comparable aspect ratio beta_p, the pooled normalised noise eigenvalues form a
// larger sample of the same Marchenko-Pastur law, giving a lower-variance single noise level
// estimate; the signal/noise boundary (rank) is then resolved per partition.

#include <algorithm>
#include <cmath>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"

namespace MR::Denoise::Estimator {

// Per-partition derived matrix dimensions (mirrors the m,n,rp -> qnz/rnz/rz/beta resolution
//   used throughout the single-PCA estimators).
struct PartitionDims {
  ssize_t r;   // min(m,n): length of the partition's eigenvalue vector
  ssize_t rz;  // assumed-zero components (from preconditioner demeaning)
  ssize_t rnz; // nonzero components
  ssize_t qnz; // long (scaling) dimension
  double beta; // rnz / qnz
};

inline std::vector<PartitionDims> partition_dims(const std::vector<ssize_t> &m,
                                                 const std::vector<ssize_t> &n,
                                                 const std::vector<ssize_t> &rp) {
  const size_t P = m.size();
  std::vector<PartitionDims> d(P);
  for (size_t p = 0; p != P; ++p) {
    d[p].r = std::min(m[p], n[p]);
    d[p].qnz = dimlong_nonzero(m[p], n[p], rp[p]);
    d[p].rnz = rank_nonzero(m[p], n[p], rp[p]);
    d[p].rz = rank_zero(m[p], n[p], rp[p]);
    d[p].beta = double(d[p].rnz) / double(d[p].qnz);
  }
  return d;
}

// rnz-weighted mean of the per-partition aspect ratios; used as the single shape parameter
//   for the (beta-dependent) Marchenko-Pastur shape constants when pooling. With balanced
//   partitions every beta_p is close to this value.
inline double mean_beta(const std::vector<PartitionDims> &d) {
  double num = 0.0;
  double den = 0.0;
  for (const auto &x : d) {
    num += double(x.rnz) * x.beta;
    den += double(x.rnz);
  }
  return den > 0.0 ? num / den : 0.0;
}

inline ssize_t total_rnz(const std::vector<PartitionDims> &d) {
  ssize_t s = 0;
  for (const auto &x : d)
    s += x.rnz;
  return s;
}

inline ssize_t total_qnz(const std::vector<PartitionDims> &d) {
  ssize_t s = 0;
  for (const auto &x : d)
    s += x.qnz;
  return s;
}

// Sorted-ascending pool of normalised nonzero eigenvalues lambda = s_p[i]/qnz_p over
//   i in [rz_p, r_p) across all partitions.
inline std::vector<double> pool_normalized(const std::vector<eigenvalues_type> &s,
                                           const std::vector<PartitionDims> &d) {
  std::vector<double> pooled;
  pooled.reserve(size_t(std::max<ssize_t>(0, total_rnz(d))));
  for (size_t p = 0; p != d.size(); ++p)
    for (ssize_t i = d[p].rz; i < d[p].r; ++i)
      pooled.push_back(s[p][i] / double(d[p].qnz));
  std::sort(pooled.begin(), pooled.end());
  return pooled;
}

// Classify each partition's eigenvalues against a single normalised threshold t (ascending
//   eigenvalues with lambda <= t are noise, INCLUDING the rz_p assumed-zero components), filling
//   result.cutoff_p_partition (per-partition noise count), result.lamplus_partition (= t for all),
//   result.cutoff_p (pooled noise count), and accumulating the pooled normalised noise sum and
//   nonzero noise count (excluding assumed-zero components, to mirror the single-PCA estimators).
inline void apply_threshold(const std::vector<eigenvalues_type> &s,
                            const std::vector<PartitionDims> &d,
                            const double t,
                            Result &result,
                            double &noise_sum,
                            ssize_t &noise_count) {
  const size_t P = d.size();
  result.cutoff_p_partition.assign(P, 0);
  result.lamplus_partition.assign(P, t);
  noise_sum = 0.0;
  noise_count = 0;
  ssize_t total_cutoff = 0;
  for (size_t p = 0; p != P; ++p) {
    // The rz_p assumed-zero (demean) components are always noise, as in the single-PCA path.
    ssize_t c = d[p].rz;
    for (ssize_t i = d[p].rz; i < d[p].r; ++i) {
      const double lam = s[p][i] / double(d[p].qnz);
      if (lam > t)
        break;
      ++c;
      noise_sum += lam;
      ++noise_count;
    }
    result.cutoff_p_partition[p] = c;
    total_cutoff += c;
  }
  result.cutoff_p = total_cutoff;
}

// As apply_threshold, but with a per-partition normalised threshold t_p (used by the imposed-
//   sigma estimators, where the Marchenko-Pastur edge (1+sqrt(beta_p))^2 sigma^2 differs per
//   partition because beta_p differs).
inline void apply_threshold_per_partition(const std::vector<eigenvalues_type> &s,
                                          const std::vector<PartitionDims> &d,
                                          const std::vector<double> &t,
                                          Result &result,
                                          double &noise_sum,
                                          ssize_t &noise_count) {
  const size_t P = d.size();
  result.cutoff_p_partition.assign(P, 0);
  result.lamplus_partition = t;
  noise_sum = 0.0;
  noise_count = 0;
  ssize_t total_cutoff = 0;
  for (size_t p = 0; p != P; ++p) {
    // The rz_p assumed-zero (demean) components are always noise, as in the single-PCA path.
    ssize_t c = d[p].rz;
    for (ssize_t i = d[p].rz; i < d[p].r; ++i) {
      const double lam = s[p][i] / double(d[p].qnz);
      if (lam > t[p])
        break;
      ++c;
      noise_sum += lam;
      ++noise_count;
    }
    result.cutoff_p_partition[p] = c;
    total_cutoff += c;
  }
  result.cutoff_p = total_cutoff;
}

} // namespace MR::Denoise::Estimator
