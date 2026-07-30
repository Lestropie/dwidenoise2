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
#include <cassert>

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
// The non-partitioned (single-PCA) case is simply the P == 1 instance of this: one partition
//   spanning every volume of the series. It is described by this same type, which is why
//   Estimator::Base needs only two overloads distinguished by arity of partitioning, each
//   taking two arguments.
//
// The eigenvalues are held as a non-owning view of externally-managed storage (in practice the
//   reusable per-patch buffers of Denoise::Estimate); that storage must outlive the Spectrum,
//   and must not be resized while a Spectrum referencing it is alive. It may legitimately be
//   longer than the r == min(m, n) components of this partition -- those buffers are grown to
//   the largest patch encountered and never shrunk -- so the view is truncated to r here, and
//   the tail beyond it is not addressable through this class.
struct Spectrum {
  // - storage : ascending eigenvalues, of which the leading min(m, n) are those of this partition
  // - m       : number of image volumes (rows of the Casorati matrix) in this partition
  // - n       : number of voxels in the patch (columns)
  // - rp      : preconditioner / null rank, ie. number of means regressed from these volumes
  Spectrum(const eigenvalues_type &storage, const ssize_t m, const ssize_t n, const ssize_t rp)
      : eigenvalues(storage.data(), std::min(m, n)), //
        m(m),                                        //
        n(n),                                        //
        rp(rp),                                      //
        r(std::min(m, n)),                           //
        rz(rank_zero(m, n, rp)),                     //
        rnz(rank_nonzero(m, n, rp)),                 //
        qnz(dimlong_nonzero(m, n, rp)),              //
        beta(double(rnz) / double(qnz)) {            //
    assert(storage.size() >= r);
    assert(rz + rnz == r);
  }

  // Ascending eigenvalues of this partition's Casorati matrix; length r, of which the leading
  //   rz are those that the preconditioner's demeaning forces to zero.
  const Eigen::Map<const eigenvalues_type> eigenvalues;
  const ssize_t m;   // number of volumes
  const ssize_t n;   // number of voxels in the patch
  const ssize_t rp;  // rank of the preconditioner (number of regressed means)
  const ssize_t r;   // min(m, n): total number of components
  const ssize_t rz;  // components assumed zero as a consequence of that regression
  const ssize_t rnz; // r - rz: components that may be nonzero
  const ssize_t qnz; // long dimension of the nonzero sub-block; scales the eigenvalues
  const double beta; // rnz / qnz: aspect ratio parameterising the Marchenko-Pastur distribution

  // Eigenvalue i expressed on the normalised scale lambda = s[i] / qnz. This is the scale on
  //   which noise levels (sigma^2) and signal/noise boundaries (lambda+) are expressed, and the
  //   only scale on which the eigenvalues of different partitions are commensurable.
  double normalized(const ssize_t i) const { return eigenvalues[i] / double(qnz); }
};

} // namespace MR::Denoise
