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

#include <algorithm>

#include <Eigen/Dense>

#include "denoise/denoise.h"

namespace MR::Denoise {

// Everything there is to know about the eigenspectrum of one partition of volumes: the
//   eigenvalues themselves, the shape of the Casorati matrix that produced them, and every
//   quantity derived from that shape.
//
// It sits here, alongside the dimension resolution functions it invokes, rather than under
//   estimator/, because it is not the estimators' private currency: Estimate constructs one per
//   partition when it decomposes a patch, Recon reads each partition's dimensions back from it
//   when reconstructing, and the estimators are simply the consumers in between.
//
// The eigenvalues and the dimensions that give them meaning are inseparable: an eigenvalue is
//   uninterpretable without the long dimension qnz that sets its scale, the count rz of leading
//   components that preconditioner demeaning has forced to zero, and the aspect ratio beta that
//   fixes the width of the Marchenko-Pastur bulk. Binding them into a single object is what
//   keeps them in correspondence: a partition is described by one value rather than by the same
//   index into four parallel containers, and the derived quantities are resolved once, in this
//   constructor, rather than re-derived (identically, and by hand) by every consumer.
//
// It is a class rather than an aggregate because the quantities of interest are no longer the
//   stored ones. Partitions of a patch are not interchangeable: they need hold neither equal
//   numbers of volumes nor equal numbers of regressed group means, so beta differs between them
//   and their eigenvalues are drawn from differently-shaped Marchenko-Pastur laws. Every
//   cross-partition operation is therefore a question addressed to a partition -- where does this
//   eigenvalue sit within *your* bulk, how many of *your* components fall below this boundary,
//   how many degrees of freedom do *your* noise components carry -- rather than a read of a
//   shared field. Answering those here, once, is what keeps the estimators free of open-coded
//   (and silently divergent) per-partition arithmetic; see denoise/estimator/pooling.h for the
//   pooling built on top of them.
//
// The non-partitioned (single-PCA) case is simply the P == 1 instance of this: one partition
//   spanning every volume of the series. It is described by this same type, which is why
//   Estimator::Base needs only two overloads distinguished by arity of partitioning, each
//   taking two arguments.
//
// The eigenvalues are on the PER-CHANNEL scale, whatever the type of the input data: for complex
//   input they have already been halved by Estimate::to_per_channel(), so that E[normalized(i)] is
//   the variance of one real noise channel rather than that of a whole complex sample. Every noise
//   level derived from them therefore has the same meaning for magnitude and complex data, which is
//   what permits one noise map to be estimated from either and consumed by the other (and what
//   distinguishes this software from MRtrix3 dwidenoise; see casorati_entry_channels()).
//
// The eigenvalues are held as a non-owning view of externally-managed storage (in practice the
//   reusable per-patch buffers of Denoise::Estimate); that storage must outlive the Spectrum,
//   and must not be resized while a Spectrum referencing it is alive. It may legitimately be
//   longer than the r == min(m, n) components of this partition -- those buffers are grown to
//   the largest patch encountered and never shrunk -- so the view is truncated to r here, and
//   the tail beyond it is not addressable through this class.
class Spectrum {
public:
  // - storage : ascending eigenvalues, of which the leading min(m, n) are those of this partition
  // - m       : number of image volumes (rows of the Casorati matrix) in this partition
  // - n       : number of voxels in the patch (columns)
  // - rp      : preconditioner / null rank, ie. number of means regressed from these volumes
  Spectrum(const eigenvalues_type &storage, const ssize_t m, const ssize_t n, const ssize_t rp);

  // Ascending eigenvalues of this partition's Casorati matrix; length r, of which the leading
  //   rz are those that the preconditioner's demeaning forces to zero.
  const Eigen::Map<const eigenvalues_type> &eigenvalues() const { return s; }
  double eigenvalue(const ssize_t i) const { return s[i]; }

  ssize_t m() const { return m_; }      // number of volumes
  ssize_t n() const { return n_; }      // number of voxels in the patch
  ssize_t rp() const { return rp_; }    // rank of the preconditioner (number of regressed means)
  ssize_t r() const { return r_; }      // min(m, n): total number of components
  ssize_t rz() const { return rz_; }    // components assumed zero as a consequence of that regression
  ssize_t rnz() const { return rnz_; }  // r - rz: components that may be nonzero
  ssize_t qnz() const { return qnz_; }  // long dimension of the nonzero sub-block; scales the eigenvalues
  double beta() const { return beta_; } // rnz / qnz: aspect ratio parameterising Marchenko-Pastur
  double sqrt_beta() const { return sqrtbeta_; }

  // Ratio of this partition's Marchenko-Pastur upper bulk edge to the noise level, ie.
  //   lambda+ = edge() * sigma^2. A function of beta alone, so partitions of unequal volume count
  //   have different edges even though they share one noise level.
  double edge() const { return edge_; }

  // Eigenvalue i on the normalised scale lambda = s[i] / qnz. This removes the dependence on the
  //   long dimension -- E[lambda] = sigma^2 whatever beta may be, with sigma the per-channel noise
  //   level -- and is the scale on which noise levels and signal/noise boundaries are expressed. It
  //   is necessary but not sufficient for comparing partitions: two partitions at one noise level
  //   still have differently-shaped bulks, so equal lambda does not mean equal significance.
  double normalized(const ssize_t i) const { return s[i] / double(qnz_); }

  // Eigenvalue i as a fraction of this partition's own upper bulk edge, z = lambda / edge(). This
  //   is the scale on which the eigenvalues of different partitions ARE comparable:
  //   - it is free of sigma, so it can order a concatenated spectrum before any noise level is
  //     known;
  //   - the Marchenko-Pastur boundary is the partition-independent value z == sigma^2, whatever
  //     beta may be, so one scalar cut classifies every partition against its own edge;
  //   - ordering by z ranks components by significance within their own bulk, so a large
  //     partition's ordinary noise no longer outranks a small partition's genuine weak signal.
  double standardized(const ssize_t i) const { return normalized(i) / edge_; }

  // The normalised boundary that a standardised cut z denotes in this partition, ie. this
  //   partition's entry of Result::lamplus_partition.
  double lambda_of(const double z) const { return z * edge_; }

  // Number of this partition's components classified as noise by the standardised cut z: the
  //   count of leading (ascending) eigenvalues with standardized() <= z. The rz components that
  //   demeaning forces to zero are always noise, as in the single-PCA estimators. Monotone in z,
  //   since edge() is constant within a partition.
  ssize_t noise_count(const double z) const;

  // Sum of raw eigenvalues over [first, first + count). Raw rather than normalised because sums
  //   are accumulated across partitions of differing qnz: a raw sum is an energy, and energies
  //   are additive where normalised eigenvalues are not.
  double raw_sum(const ssize_t first, const ssize_t count) const;

  // Degrees of freedom carried by "count" of this partition's noise components: under Gaussian
  //   noise of per-channel variance sigma^2 each carries qnz, so the sum of count raw eigenvalues
  //   has expectation sigma^2 * dof(count). That the relation holds for complex data too is a
  //   consequence of the per-channel rescaling described above; what the rescaling does not alter
  //   is the *fluctuation* of that sum, which is smaller for complex entries than for real ones. Pairing raw_sum() with
  //   dof() across partitions weights each by its information content (its number of matrix entries), which is the
  //   minimum-variance combination; averaging normalised eigenvalues instead weights by component
  //   count alone, and the two agree only when every qnz is equal.
  double dof(const ssize_t count) const { return double(count) * double(qnz_); }

  // As dof(), for a decomposition from which "removed" leading components have been peeled off as
  //   signal, each costing one direction of the long dimension. Generalises the (n' - p) deflation
  //   of the MRM2023 recursion to a partition that has had its own components removed, rather than
  //   deflating one notional aggregate dimension by every partition's removals.
  double dof_deflated(const ssize_t count, const ssize_t removed) const {
    return double(count) * std::max(1.0, double(qnz_ - removed));
  }

private:
  const Eigen::Map<const eigenvalues_type> s;
  const ssize_t m_;
  const ssize_t n_;
  const ssize_t rp_;
  const ssize_t r_;
  const ssize_t rz_;
  const ssize_t rnz_;
  const ssize_t qnz_;
  const double beta_;
  const double sqrtbeta_;
  const double edge_;
};

} // namespace MR::Denoise
