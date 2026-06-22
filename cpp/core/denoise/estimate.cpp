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

#include "denoise/estimate.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#include "interp/cubic.h"
#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Estimate<F>::Estimate(const Image<F> &image,
                      std::shared_ptr<SpatialSubsample> subsample,
                      std::shared_ptr<Kernel::Base> kernel,
                      decomp_type decomp,
                      std::shared_ptr<Estimator::Base> estimator,
                      Exports &exports,
                      const ssize_t preconditioner_rank,
                      const bool enable_recon,
                      std::shared_ptr<const Partitioning> level_partitioning,
                      std::vector<ssize_t> volume_group,
                      const ssize_t kernel_num_partitions)
    : m(Denoise::num_volumes(image)),
      subsample(subsample),
      kernel(kernel),
      decomp(decomp),
      estimator(estimator),
      preconditioner_rank(preconditioner_rank),
      enable_recon(enable_recon),
      X(m, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? m : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      s(std::min(m, kernel->estimated_size())),
      level_partitioning(level_partitioning),
      kernel_num_partitions(kernel_num_partitions),
      active_part(nullptr),
      volume_group(std::move(volume_group)),
      num_demean_groups(this->volume_group.empty()
                            ? 0
                            : (*std::max_element(this->volume_group.begin(), this->volume_group.end()) + 1)),
      exports(exports) {
  // If input image is > 4D, should have been preconditioned into 4D
  assert(image.ndim() == 4);
  pca_failure_counter.store(0, std::memory_order_release);
  allocate_partition_storage();
}

template <typename F>
Estimate<F>::Estimate(const Estimate<F> &that)
    : m(that.m),
      subsample(that.subsample),
      kernel(that.kernel),
      decomp(that.decomp),
      estimator(that.estimator),
      preconditioner_rank(that.preconditioner_rank),
      enable_recon(that.enable_recon),
      X(m, kernel->estimated_size()),
      SVD(decomp == decomp_type::BDCSVD ? m : 0,
          decomp == decomp_type::BDCSVD ? kernel->estimated_size() : 0,
          enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly),
      XtX(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0,
          decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      eig(decomp == decomp_type::SELFADJOINT ? std::min(m, kernel->estimated_size()) : 0),
      s(std::min(m, kernel->estimated_size())),
      level_partitioning(that.level_partitioning),
      kernel_num_partitions(that.kernel_num_partitions),
      active_part(nullptr),
      volume_group(that.volume_group),
      num_demean_groups(that.num_demean_groups),
      exports(that.exports) {
  allocate_partition_storage();
}

// Choose the partitioning for the patch centred at "voxel": the shared per-level assignment, a
//   freshly drawn per-kernel assignment (seeded deterministically by the voxel so it is identical
//   between the Estimate and Recon evaluations and reproducible across runs/threads), or none.
template <typename F> void Estimate<F>::select_partitioning(const Kernel::Voxel::index_type &voxel) {
  if (level_partitioning && level_partitioning->num_partitions() > 1) {
    active_part = level_partitioning.get();
  } else if (kernel_num_partitions > 1) {
    // Spatial hash of the patch centre (Teschner et al.) → reproducible per-voxel seed.
    const uint32_t seed = uint32_t((uint32_t(voxel[0]) * 73856093u) ^ (uint32_t(voxel[1]) * 19349663u) ^
                                   (uint32_t(voxel[2]) * 83492791u));
    Math::RNG rng(seed);
    patch_partition = partition_volumes(m, volume_group, kernel_num_partitions, rng);
    active_part = (patch_partition.num_partitions() > 1) ? &patch_partition : nullptr;
  } else {
    active_part = nullptr;
  }
}

template <typename F> void Estimate<F>::allocate_partition_storage() {
  if (!partitioning_enabled())
    return;
  const ssize_t P = configured_partitions();
  Xsub_partition.resize(P);
  s_partition.resize(P);
  part_m.assign(P, 0);
  part_n.assign(P, 0);
  part_rp.assign(P, 0);
  // means_partition is used as demean scratch in both the estimation and reconstruction passes
  //   (and read back by Recon to re-add the per-group means), so allocate it unconditionally.
  means_partition.resize(P);
  if (enable_recon) {
    U_partition.resize(P);
    V_partition.resize(P);
    evec_partition.resize(P);
    sv_partition.resize(P);
  }
}

template <typename F> void Estimate<F>::operator()(Image<F> &dwi) {

  // There are two options here for looping in the presence of subsampling:
  // 1. Loop over the input image
  //    Skip voxels that don't lie at the centre of a patch
  //    Have to transform input image voxel indices to subsampled image voxel indices for some optional outputs
  // 2. Loop over the subsampled image
  //    In some use cases there may not be any image created that conforms to this voxel grid
  //    Have to transform the subsampled voxel index into an input image voxel index for the centre of the patch
  // Going to go with 1. for now, as for 2. may not have a suitable image over which to loop
  Kernel::Voxel::index_type voxel({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (!subsample->process(voxel))
    return;

  // Select this patch's partitioning (shared per-level, or a per-kernel assignment drawn from a
  //   voxel-seeded RNG, or none). active_part is then used throughout this evaluation.
  select_partitioning(voxel);

  // Load list of voxels from which to import data
  patch = (*kernel)(voxel);
  const ssize_t n = patch.voxels.size();
  const ssize_t r = std::min(m, n);

  // Expand local storage if necessary
  if (n > X.cols()) {
    DEBUG("Expanding data matrix storage from " + str(m) + "x" + str(X.cols()) + " to " + str(m) + "x" + str(n));
    X.resize(m, n);
  }
  if (decomp == decomp_type::SELFADJOINT && r > XtX.cols()) {
    DEBUG("Expanding decomposition matrix storage from " + str(X.rows()) + " to " + str(r));
    XtX.resize(r, r);
  }
  if (r > s.size()) {
    DEBUG("Expanding eigenvalue storage from " + str(s.size()) + " to " + str(r));
    s.resize(r);
  }

  // Fill matrices with NaN when in debug mode;
  //   make sure results from one voxel are not creeping into another
  //   due to use of block oberations to prevent memory re-allocation
  //   in the presence of variation in kernel sizes
#ifndef NDEBUG
  X.fill(std::numeric_limits<F>::signaling_NaN());
  XtX.fill(std::numeric_limits<F>::signaling_NaN());
  s.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  load_data(dwi);
  assert(X.leftCols(n).allFinite());

  // Compute the eigendecomposition(s) and estimate the signal/noise threshold.
  bool successful_decomposition = false;
  if (partitioned()) {
    // Split the volumes into partitions, decompose each independently, and estimate a single
    //   noise level (with a per-partition rank) from the pooled spectrum.
    successful_decomposition = compute_partitions(n);
    if (successful_decomposition)
      threshold = (*estimator)(s_partition, part_m, part_n, part_rp, patch.centre_realspace);
    else {
      threshold = Estimator::Result();
      pca_failure_counter.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    switch (decomp) {
    case decomp_type::BDCSVD: {
      SVD.compute(X.leftCols(n), enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly);
      successful_decomposition = SVD.info() == Eigen::Success;
      if (successful_decomposition) {
        // eigenvalues sorted in increasing order:
        s.head(r) = SVD.singularValues().array().reverse().square().template cast<double>();
      }
    } break;
    case decomp_type::SELFADJOINT: {
      if (m <= n)
        XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n) * X.leftCols(n).adjoint();
      else
        XtX.topLeftCorner(r, r).template triangularView<Eigen::Lower>() = X.leftCols(n).adjoint() * X.leftCols(n);
      eig.compute(XtX.topLeftCorner(r, r), enable_recon ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
      successful_decomposition = eig.info() == Eigen::Success;
      if (successful_decomposition) {
        // eigenvalues sorted in increasing order,
        //   additionally clamping any negtive values to zero:
        s.head(r) = eig.eigenvalues().template cast<double>().cwiseMax(0.0);
      }
    } break;
    }

    if (successful_decomposition) {
      // Threshold determination, possibly via Marchenko-Pastur
      threshold = (*estimator)(s.head(r), m, n, preconditioner_rank, patch.centre_realspace);
    } else {
      s.head(r).fill(std::numeric_limits<double>::signaling_NaN());
      threshold = Estimator::Result();
      pca_failure_counter.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Store additional output maps if requested
  auto ss_index = subsample->in2ss(voxel);
  if (exports.noise_out.valid()) {
    assign_pos_of(ss_index).to(exports.noise_out);
    exports.noise_out.value() = bool(threshold)                                //
                                    ? float(std::sqrt(threshold.sigma2))       //
                                    : std::numeric_limits<float>::quiet_NaN(); //
  }
  if (exports.lamplus.valid()) {
    assign_pos_of(ss_index).to(exports.lamplus);
    exports.lamplus.value() = threshold.lamplus;
  }
  if (exports.rank_pcanonzero.valid()) {
    assign_pos_of(ss_index).to(exports.rank_pcanonzero);
    if (partitioned()) {
      // Pooled count of nonzero components across partitions (sum_p rank_nonzero(m_p,n,rp_p)).
      ssize_t total = 0;
      for (ssize_t p = 0; p != active_part->num_partitions(); ++p)
        total += rank_nonzero(part_m[p], part_n[p], part_rp[p]);
      exports.rank_pcanonzero.value() = total;
    } else {
      exports.rank_pcanonzero.value() = rank_nonzero(m, n, preconditioner_rank);
    }
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(ss_index).to(exports.rank_input);
    if (!successful_decomposition)
      exports.rank_input.value() = 0;
    else if (partitioned()) {
      // Total *signal* rank across partitions = pooled total rank - pooled noise count. The
      //   regressed group-mean components are deliberately excluded here (mirroring the single-PCA
      //   path, whose per-iteration value is also signal-only): they are added back, as a constant
      //   group count, only to the final exported maps (see the reconciliation in the commands).
      //   Keeping this value mean-free also makes it the correct per-partition signal density once
      //   divided by the partition count when forming rank_per_mm for the next iteration's kernel.
      ssize_t total_r = 0;
      for (ssize_t p = 0; p != active_part->num_partitions(); ++p)
        total_r += std::min(part_m[p], part_n[p]);
      exports.rank_input.value() = bool(threshold) ? (total_r - threshold.cutoff_p) : total_r;
    } else if (bool(threshold))
      exports.rank_input.value() = r - threshold.cutoff_p;
    else
      exports.rank_input.value() = r;
  }
  if (exports.max_dist.valid()) {
    assign_pos_of(ss_index).to(exports.max_dist);
    exports.max_dist.value() = patch.max_distance;
  }
  if (exports.voxelcount.valid()) {
    assign_pos_of(ss_index).to(exports.voxelcount);
    exports.voxelcount.value() = n;
  }
  if (exports.patchcount.valid() || exports.saving_eigenspectra()) {
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    if (exports.patchcount.valid()) {
      for (const auto &v : patch.voxels) {
        assign_pos_of(v.index).to(exports.patchcount);
        exports.patchcount.value() = exports.patchcount.value() + 1;
      }
    }
    if (exports.saving_eigenspectra()) {
      if (partitioned()) {
        // Export the pooled, descending, qnz-normalised spectrum across partitions (raw
        //   per-partition eigenvalues are on incommensurable scales).
        ssize_t total = 0;
        for (ssize_t p = 0; p != active_part->num_partitions(); ++p)
          total += std::min(part_m[p], part_n[p]);
        eigenvalues_type pooled(total);
        ssize_t idx = 0;
        for (ssize_t p = 0; p != active_part->num_partitions(); ++p) {
          const ssize_t qnz = dimlong_nonzero(part_m[p], part_n[p], part_rp[p]);
          const ssize_t rp_r = std::min(part_m[p], part_n[p]);
          for (ssize_t i = 0; i != rp_r; ++i)
            pooled[idx++] = s_partition[p][i] / double(qnz);
        }
        std::sort(pooled.data(), pooled.data() + pooled.size(), std::greater<double>());
        exports.add_eigenspectrum(pooled);
      } else {
        exports.add_eigenspectrum(s);
      }
    }
  }
}

template <typename F> void Estimate<F>::report_warnings() const {
  const ssize_t count = pca_failure_counter.load(std::memory_order_acquire);
  if (count > 0) {
    WARN("A total of " + str(count) + " PCA kernels failed to converge");
  }
}

template <typename F> bool Estimate<F>::compute_partitions(const ssize_t n) {
  assert(active_part->num_volumes() == m);
  const ssize_t P = active_part->num_partitions();
  bool all_ok = true;
  for (ssize_t p = 0; p != P; ++p) {
    const std::vector<ssize_t> &rows = active_part->volumes(p);
    const ssize_t m_p = ssize_t(rows.size());
    part_m[p] = m_p;
    part_n[p] = n;

    // Extract this partition's sub-block: its m_p volume rows, all n patch-voxel columns.
    MatrixType &Xp = Xsub_partition[p];
    Xp.resize(m_p, n);
    for (ssize_t li = 0; li != m_p; ++li)
      Xp.row(li) = X.row(rows[li]).head(n);

    // Per-(group) per-column demeaning within the partition, keeping the subtracted mean
    //   orthogonal to this partition's PCA; rp_p counts the demeaning groups present (each
    //   contributes one regressed-out / assumed-zero component). The means are retained for
    //   Recon to re-add to the reconstructed data.
    ssize_t rp_p = 0;
    if (!volume_group.empty()) {
      MatrixType &means = means_partition[p];
      means.setZero(num_demean_groups, n);
      std::vector<ssize_t> counts(num_demean_groups, 0);
      for (ssize_t li = 0; li != m_p; ++li) {
        const ssize_t g = volume_group[rows[li]];
        means.row(g) += Xp.row(li);
        ++counts[g];
      }
      for (ssize_t g = 0; g != num_demean_groups; ++g) {
        if (counts[g] > 0) {
          means.row(g) /= F(double(counts[g]));
          ++rp_p;
        }
      }
      for (ssize_t li = 0; li != m_p; ++li)
        Xp.row(li) -= means.row(volume_group[rows[li]]);
    }
    part_rp[p] = rp_p;

    const ssize_t r_p = std::min(m_p, n);
    if (s_partition[p].size() < r_p)
      s_partition[p].resize(r_p);

    bool ok = false;
    switch (decomp) {
    case decomp_type::BDCSVD: {
      SVD.compute(Xp, enable_recon ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : Eigen::EigenvaluesOnly);
      ok = SVD.info() == Eigen::Success;
      if (ok) {
        s_partition[p].head(r_p) = SVD.singularValues().array().reverse().square().template cast<double>();
        if (enable_recon) {
          U_partition[p] = SVD.matrixU();
          V_partition[p] = SVD.matrixV();
          sv_partition[p] = SVD.singularValues();
        }
      }
    } break;
    case decomp_type::SELFADJOINT: {
      if (r_p > XtX.cols())
        XtX.resize(r_p, r_p);
      if (m_p <= n)
        XtX.topLeftCorner(r_p, r_p).template triangularView<Eigen::Lower>() = Xp * Xp.adjoint();
      else
        XtX.topLeftCorner(r_p, r_p).template triangularView<Eigen::Lower>() = Xp.adjoint() * Xp;
      eig.compute(XtX.topLeftCorner(r_p, r_p), enable_recon ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
      ok = eig.info() == Eigen::Success;
      if (ok) {
        s_partition[p].head(r_p) = eig.eigenvalues().template cast<double>().cwiseMax(0.0);
        if (enable_recon)
          evec_partition[p] = eig.eigenvectors();
      }
    } break;
    }
    if (!ok) {
      all_ok = false;
      s_partition[p].head(r_p).fill(std::numeric_limits<double>::signaling_NaN());
    }
  }
  return all_ok;
}

template <typename F> void Estimate<F>::load_data(Image<F> &image) {
  const Kernel::Voxel::index_type pos({image.index(0), image.index(1), image.index(2)});
  for (ssize_t i = 0; i != patch.voxels.size(); ++i) {
    assign_pos_of(patch.voxels[i].index, 0, 3).to(image);
    X.col(i) = image.row(3);
  }
  assign_pos_of(pos, 0, 3).to(image);
}

template class Estimate<float>;
template class Estimate<cfloat>;
template class Estimate<double>;
template class Estimate<cdouble>;

} // namespace MR::Denoise
