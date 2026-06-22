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

// Need to import this first to get relevant precompiler definitions
#include "denoise/denoise.h"

#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/result.h"
#include "denoise/exports.h"
#include "denoise/kernel/base.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "denoise/partition.h"
#include "denoise/spatial_subsample.h"
#include "header.h"
#include "image.h"
#include "transform.h"

namespace MR::Denoise {

template <typename F> class Estimate {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;

  Estimate(const Image<F> &image,
           std::shared_ptr<SpatialSubsample> subsample,
           std::shared_ptr<Kernel::Base> kernel,
           decomp_type decomp,
           std::shared_ptr<Estimator::Base> estimator,
           Exports &exports,
           const ssize_t preconditioner_rank = 0,
           const bool enable_recon = false,
           std::shared_ptr<const Partitioning> level_partitioning = nullptr,
           std::vector<ssize_t> volume_group = {},
           const ssize_t kernel_num_partitions = 1);

  Estimate(const Estimate &);

  void operator()(Image<F> &dwi);

  void report_warnings() const;

protected:
  const ssize_t m;

  // Denoising configuration
  std::shared_ptr<SpatialSubsample> subsample;
  std::shared_ptr<Kernel::Base> kernel;
  decomp_type decomp;
  std::shared_ptr<Estimator::Base> estimator;
  ssize_t preconditioner_rank;
  bool enable_recon;

  // Reusable memory
  Kernel::Data patch;
  MatrixType X;

  // TODO For both BDCSVD and SelfAdjointEigenSolver,
  //   the template type is MatrixType,
  //   and it doesn't seem to be possible to define an Eigen::Block as this template type;
  //   as such, most likely in both circumstances it is actually constructing a MatrixType from Eigen::Block
  //   in order to construct the decomposition
  // What could conceivably be done instead,
  //   given that these matrices are relatively small
  //   and the number of unique patch sizes is small (though not necessarily one),
  //   would be to construct a std::map<> from patch size to PCA memory;
  //   each processing thread would allocate new memory for new patch sizes not yet encountered by it,
  //   but the total memory consumption should still be relatively small;
  //   note that "X" would be subsumed within such a mechanism also
  Eigen::BDCSVD<MatrixType> SVD;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;

  eigenvalues_type s;
  Estimator::Result threshold;

  // Volume partitioning (large-series acceleration). When partitioning is in effect, each
  //   patch's volumes (rows of X) are split into P partitions and decomposed independently; the
  //   pooled spectrum is passed to the partition-aware estimator and the per-partition results
  //   are retained for reconstruction. Two modes:
  //   - per-level: a single shared assignment (level_partitioning, P>1) is reused for every
  //       patch in the iteration;
  //   - per-kernel: each patch draws its own balanced assignment of the m' volumes into
  //       kernel_num_partitions partitions, from an RNG seeded deterministically by the patch
  //       centre (so the assignment is reproducible and identical between the Estimate and Recon
  //       evaluations of the same patch). This averages each output voxel over many partitionings.
  //   A null shared assignment with kernel_num_partitions <= 1 selects the single-PCA path.
  std::shared_ptr<const Partitioning> level_partitioning;
  ssize_t kernel_num_partitions;        // P for per-kernel mode (1 ⇒ not per-kernel)
  Partitioning patch_partition;         // per-kernel scratch, rebuilt per patch
  const Partitioning *active_part;      // partitioning in force for the current patch (or nullptr)
  // Per-(output) volume demeaning-group label (length m'); empty ⇒ no demeaning. Drives the
  //   per-partition per-group mean subtraction (kept orthogonal to each partition's PCA) and
  //   the per-partition preconditioner rank rp_p. (Under partitioning the preconditioner itself
  //   performs no demeaning; see Precondition::set_partitioning_active.)
  std::vector<ssize_t> volume_group;
  ssize_t num_demean_groups; // max(volume_group)+1, or 0 when volume_group is empty

  using RealVectorType = Eigen::Matrix<typename Eigen::NumTraits<F>::Real, Eigen::Dynamic, 1>;
  // Reusable per-partition decomposition storage (sized to P; filled by compute_partitions()).
  std::vector<MatrixType> Xsub_partition;       // demeaned sub-block (m_p x n) per partition
  std::vector<eigenvalues_type> s_partition;    // ascending eigenvalues (length r_p) per partition
  std::vector<ssize_t> part_m, part_n, part_rp; // per-partition (m_p, n, rp_p)
  // Populated only when enable_recon (used by Recon to forward-project each partition):
  std::vector<MatrixType> U_partition, V_partition, evec_partition;
  std::vector<RealVectorType> sv_partition;
  std::vector<MatrixType> means_partition; // (num_demean_groups x n) per-column group means

  // Whether partitioning is configured at all (known at construction; either mode).
  bool partitioning_enabled() const {
    return (level_partitioning && level_partitioning->num_partitions() > 1) || kernel_num_partitions > 1;
  }
  // The (maximum) number of partitions, for sizing the reusable per-partition storage.
  ssize_t configured_partitions() const {
    return level_partitioning ? level_partitioning->num_partitions() : kernel_num_partitions;
  }
  // Whether the current patch is being partitioned (active_part selected for it).
  bool partitioned() const { return active_part != nullptr; }
  const Partitioning &active_partitioning() const { return *active_part; }
  // Select the partitioning in force for the patch centred at "voxel": the shared per-level
  //   assignment, or a freshly drawn per-kernel one (seeded by the voxel), or none.
  void select_partitioning(const Kernel::Voxel::index_type &voxel);
  // Size the per-partition storage vectors to P (called from the constructors).
  void allocate_partition_storage();
  // Split X (already filled by load_data) into the partition sub-blocks, demean each per group,
  //   and decompose each; fills s_partition / part_* (and the recon outputs when enable_recon).
  //   Returns false if any partition's decomposition failed to converge.
  bool compute_partitions(const ssize_t n);

  // Export images
  // Note: One instance created per thread,
  //   so that when possible output image data can be written without mutex-locking
  Exports exports;

  // Some data can only be written in a thread-safe manner
  static std::mutex mutex;

  static std::atomic<ssize_t> pca_failure_counter;

  void load_data(Image<F> &image);
};

template <typename F> std::mutex Estimate<F>::mutex;
template <typename F> std::atomic<ssize_t> Estimate<F>::pca_failure_counter;

} // namespace MR::Denoise
