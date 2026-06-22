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

#include "denoise/recon.h"

#include "denoise/denoise.h"
#include "math/math.h"

namespace MR::Denoise {

template <typename F>
Recon<F>::Recon(const Image<F> &image,
                std::shared_ptr<SpatialSubsample> subsample,
                std::shared_ptr<Kernel::Base> kernel,
                const decomp_type decomposition,
                std::shared_ptr<Estimator::Base> estimator,
                filter_type filter,
                aggregator_type aggregator,
                Exports &exports,
                const ssize_t preconditioner_rank,
                std::shared_ptr<const Partitioning> level_partitioning,
                std::vector<ssize_t> volume_group,
                const ssize_t kernel_num_partitions)
    : Estimate<F>(image,
                  subsample,
                  kernel,
                  decomposition,
                  estimator,
                  exports,
                  preconditioner_rank,
                  true,
                  level_partitioning,
                  std::move(volume_group),
                  kernel_num_partitions),
      filter(filter),
      aggregator(aggregator),
      // FWHM = 2 x cube root of spacings between kernels
      gaussian_multiplier(-std::log(2.0) /                                                          //
                          Math::pow2(std::cbrt(subsample->get_factors()[0] * image.spacing(0)       //
                                               * subsample->get_factors()[1] * image.spacing(1)     //
                                               * subsample->get_factors()[2] * image.spacing(2)))), //
      w(std::min(Estimate<F>::m, kernel->estimated_size())),
      Xr(Estimate<F>::m, aggregator == aggregator_type::EXCLUSIVE ? 1 : kernel->estimated_size()) {}

template <typename F> void Recon<F>::operator()(Image<F> &dwi, Image<F> &out) {

  if (!Estimate<F>::subsample->process({dwi.index(0), dwi.index(1), dwi.index(2)}))
    return;

  Estimate<F>::operator()(dwi);

  if (Estimate<F>::partitioned()) {
    recon_partitioned(dwi, out);
    return;
  }

  const ssize_t n = Estimate<F>::patch.voxels.size();
  const ssize_t r = std::min(Estimate<F>::m, n);
  const ssize_t rz = rank_zero(Estimate<F>::m, n, Estimate<F>::preconditioner_rank);
  const ssize_t rnz = rank_nonzero(Estimate<F>::m, n, Estimate<F>::preconditioner_rank);
  const ssize_t qnz = dimlong_nonzero(Estimate<F>::m, n, Estimate<F>::preconditioner_rank);
  const double beta = double(rnz) / double(qnz);

  if (r > w.size())
    w.resize(r);
  if (aggregator != aggregator_type::EXCLUSIVE && n > Xr.cols())
    Xr.resize(Estimate<F>::m, n);
#ifndef NDEBUG
  w.fill(std::numeric_limits<default_type>::signaling_NaN());
  Xr.fill(std::numeric_limits<default_type>::signaling_NaN());
#endif

  // Generate weights vector
  double sum_weights = 0.0;
  double sum_variance = 0.0;
  ssize_t out_rank = 0;
  if (bool(Estimate<F>::threshold)) {
    switch (filter) {
    case filter_type::OPTSHRINK: {
      w.head(rz).setZero();
      const double transition = 1.0 + std::sqrt(beta);
      for (ssize_t i = rz; i != r; ++i) {
        // TODO For non-binary determination of weights for optimal shrinkage,
        //   should the expression be identical between BDCSVD and SelfAdjointEigenSolver?
        //   Or eg. is one equivalent to scaling singular values whereas the other is equivalent to scaling eigenvalues?
        const double lam = Estimate<F>::s[i] / qnz;
        const double y = std::sqrt(lam / Estimate<F>::threshold.sigma2);
        // const double y = lam / Estimate<F>::threshold.sigma2;
        double nu = 0.0;
        if (y > transition) {
          // Occasionally floating-point precision will drive this calculation to fractionally greater than y,
          //   which will erroneously yield a weight fractionally greater than 1.0
          nu = std::min(y, std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y);
          ++out_rank;
        }
        w[i] = lam > 0.0 ? (nu / y) : 0.0;
        assert(w[i] >= 0.0 && w[i] <= 1.0);
        sum_weights += w[i];
        sum_variance += w[i] * Estimate<F>::s[i];
      }
    } break;
    case filter_type::OPTTHRESH: {
      const std::map<double, double>::const_iterator it = beta2lambdastar.find(beta);
      double lambda_star = 0.0;
      if (it == beta2lambdastar.end()) {
        lambda_star =
            sqrt(2.0 * (beta + 1.0) + ((8.0 * beta) / (beta + 1.0 + std::sqrt(Math::pow2(beta) + 14.0 * beta + 1.0))));
        beta2lambdastar[beta] = lambda_star;
      } else {
        lambda_star = it->second;
      }
      const double tau_star = lambda_star * std::sqrt(qnz) * std::sqrt(Estimate<F>::threshold.sigma2);
      // TODO Unexpected requisite square applied to qnz here
      const double threshold = tau_star * Math::pow2(qnz);
      w.head(rz).setZero();
      for (ssize_t i = rz; i != r; ++i) {
        if (Estimate<F>::s[i] >= threshold) {
          w[i] = 1.0;
          ++out_rank;
          sum_variance += Estimate<F>::s[i];
        } else {
          w[i] = 0.0;
        }
      }
      sum_weights = out_rank;
    } break;
    case filter_type::TRUNCATE:
      out_rank = r - Estimate<F>::threshold.cutoff_p;
      w.head(Estimate<F>::threshold.cutoff_p).setZero();
      w.segment(Estimate<F>::threshold.cutoff_p, out_rank).setOnes();
      sum_weights = double(out_rank);
      sum_variance += w.head(r).matrix().dot(Estimate<F>::s.head(r).matrix());
      break;
    default:
      assert(false);
    }
    assert(std::isfinite(sum_weights));
  } else { // Threshold for this patch is invalid
    // Erring on the conservative side:
    //   If the decomposition fails, or a threshold can't be found,
    //   copy the input data to the output data as-is,
    //   regardless of whether performing overcomplete local PCA
    w.head(r).setOnes();
    out_rank = r;
    sum_weights = r;
    sum_variance = Estimate<F>::s.sum();
  }
  assert(w.head(r).allFinite());
  const double variance_removed = 1.0 - sum_variance / Estimate<F>::s.sum();

  // Recombine data using only eigenvectors above threshold
  // If only the data computed when this voxel was the centre of the patch
  //   is to be used for synthesis of the output image,
  //   then only that individual column needs to be reconstructed;
  //   if however the result from this patch is to contribute to the synthesized image
  //   for all voxels that were utilised within this patch,
  //   then we need to instead compute the full projection
  switch (aggregator) {
  case aggregator_type::EXCLUSIVE: {
    assert(Estimate<F>::patch.centre_index >= 0);
    if (bool(Estimate<F>::threshold)) {
      switch (Estimate<F>::decomp) {
      case decomp_type::BDCSVD: {
        assert(Estimate<F>::SVD.matrixU().allFinite());
        assert(Estimate<F>::SVD.matrixV().allFinite());
        assert(w.head(r).allFinite());
        assert(Estimate<F>::SVD.singularValues().allFinite());
        // TODO Re-try reconstruction without use of V:
        //   https://github.com/MRtrix3/mrtrix3/pull/2906/commits/eb34f3c57dd460d2b3bd86b9653066be15e916c6
        // It might be that in the case of anything other than EXCLUSIVE,
        //   computing V is no more expensive than doing the full patch reconstruction in its absence,
        //   whereas for EXCLUSIVE since only a small portion of V is used it's worthwhile
        Xr.noalias() =                                                                 //
            Estimate<F>::SVD.matrixU() *                                               //
            (w.head(r).reverse().template cast<F>().array() *                          //
             Estimate<F>::SVD.singularValues().array()).matrix().asDiagonal() *        //
            Estimate<F>::SVD.matrixV().row(Estimate<F>::patch.centre_index).adjoint(); //
      } break;
      case decomp_type::SELFADJOINT: {
        if (Estimate<F>::m <= n)
          Xr.noalias() =                                               //
              Estimate<F>::eig.eigenvectors() *                        //
              (w.head(r).template cast<F>().matrix().asDiagonal() *    //
               (Estimate<F>::eig.eigenvectors().adjoint() *            //
                Estimate<F>::X.col(Estimate<F>::patch.centre_index))); //
        else
          Xr.noalias() =                                                                          //
              Estimate<F>::X.leftCols(n) *                                                        //
              (Estimate<F>::eig.eigenvectors() *                                                  //
               (w.head(r).template cast<F>().matrix().asDiagonal() *                              //
                Estimate<F>::eig.eigenvectors().adjoint().col(Estimate<F>::patch.centre_index))); //
      } break;
      }
      assert(Xr.allFinite());
    } else {
      // In the case of -aggregator exclusive,
      //   where a decomposition fails or we can't find a threshold,
      //   we simply copy the input data into the output image
      Xr.noalias() = Estimate<F>::X.col(Estimate<F>::patch.centre_index);
    }
    assign_pos_of(dwi).to(out);
    out.row(3) = Xr.col(0);
    if (Estimate<F>::exports.sum_aggregation.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.sum_aggregation);
      Estimate<F>::exports.sum_aggregation.value() = 1.0;
    }
    if (Estimate<F>::exports.rank_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.rank_output);
      Estimate<F>::exports.rank_output.value() = out_rank;
    }
  } break;
  default: { // All aggregators other than EXCLUSIVE
    if (!Estimate<F>::threshold) {
      Xr.leftCols(n).noalias() = Estimate<F>::X.leftCols(n);
    } else {
      switch (Estimate<F>::decomp) {
      case decomp_type::BDCSVD:
        Xr.leftCols(n).noalias() =                                              //
            Estimate<F>::SVD.matrixU() *                                        //
            (w.head(r).reverse().template cast<F>().array() *                   //
             Estimate<F>::SVD.singularValues().array()).matrix().asDiagonal() * //
            Estimate<F>::SVD.matrixV().adjoint();                               //
        break;
      case decomp_type::SELFADJOINT:
        if (Estimate<F>::m <= n) {
          Xr.leftCols(n).noalias() =                                //
              Estimate<F>::eig.eigenvectors() *                     //
              (w.head(r).template cast<F>().matrix().asDiagonal() * //
               (Estimate<F>::eig.eigenvectors().adjoint() *         //
                Estimate<F>::X.leftCols(n)));                       //
        } else {
          Xr.leftCols(n).noalias() =                                 //
              Estimate<F>::X.leftCols(n) *                           //
              (Estimate<F>::eig.eigenvectors() *                     //
               (w.head(r).template cast<F>().matrix().asDiagonal() * //
                Estimate<F>::eig.eigenvectors().adjoint()));         //
        }
        break;
      }
    }
    assert(Xr.leftCols(n).allFinite());
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    for (size_t voxel_index = 0; voxel_index != Estimate<F>::patch.voxels.size(); ++voxel_index) {
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(out);
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index).to(Estimate<F>::exports.sum_aggregation);
      double weight = std::numeric_limits<double>::signaling_NaN();
      switch (aggregator) {
      case aggregator_type::EXCLUSIVE:
        assert(false);
        break;
      case aggregator_type::GAUSSIAN:
        weight = std::exp(gaussian_multiplier * Estimate<F>::patch.voxels[voxel_index].sq_distance);
        break;
      case aggregator_type::INVL0:
        weight = 1.0 / (1 + out_rank);
        break;
      case aggregator_type::RANK:
        weight = out_rank;
        break;
      case aggregator_type::UNIFORM:
        weight = 1.0;
        break;
      }
      out.row(3) += weight * Xr.col(voxel_index);
      Estimate<F>::exports.sum_aggregation.value() += weight;
      if (Estimate<F>::exports.rank_output.valid()) {
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.rank_output);
        Estimate<F>::exports.rank_output.value() += weight * out_rank;
      }
    }
  } break;
  }

  auto ss_index = Estimate<F>::subsample->in2ss({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (Estimate<F>::exports.sum_optshrink.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.sum_optshrink);
    Estimate<F>::exports.sum_optshrink.value() = sum_weights;
  }
  if (Estimate<F>::exports.variance_removed.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.variance_removed);
    Estimate<F>::exports.variance_removed.value() = variance_removed;
  }
}

template <typename F> void Recon<F>::recon_partitioned(Image<F> &dwi, Image<F> &out) {
  using MatrixType = typename Estimate<F>::MatrixType;
  const ssize_t P = Estimate<F>::active_partitioning().num_partitions();
  const ssize_t n = Estimate<F>::patch.voxels.size();
  const ssize_t centre = Estimate<F>::patch.centre_index;
  const bool valid = bool(Estimate<F>::threshold);
  const double sigma2 = Estimate<F>::threshold.sigma2;
  const std::vector<ssize_t> &cutoff_part = Estimate<F>::threshold.cutoff_p_partition;
  const bool exclusive = (aggregator == aggregator_type::EXCLUSIVE);
  const ssize_t n_eff = exclusive ? 1 : n;

  if (Xr.rows() != Estimate<F>::m || Xr.cols() < n_eff)
    Xr.resize(Estimate<F>::m, n_eff);

  double sum_weights = 0.0;
  double sum_variance = 0.0;
  double total_s = 0.0;
  ssize_t out_rank = 0;

  for (ssize_t p = 0; p != P; ++p) {
    const std::vector<ssize_t> &rows = Estimate<F>::active_partitioning().volumes(p);
    const ssize_t m_p = Estimate<F>::part_m[p];
    const ssize_t rp_p = Estimate<F>::part_rp[p];
    const ssize_t r_p = std::min(m_p, n);
    const ssize_t rz = rank_zero(m_p, n, rp_p);
    const ssize_t rnz = rank_nonzero(m_p, n, rp_p);
    const ssize_t qnz = dimlong_nonzero(m_p, n, rp_p);
    const double beta = double(rnz) / double(qnz);
    const eigenvalues_type &sp = Estimate<F>::s_partition[p];
    total_s += sp.head(r_p).sum();

    Xr_partition.resize(m_p, n_eff);

    // Threshold invalid for this patch: copy the (un-denoised) input rows through unchanged.
    if (!valid) {
      for (ssize_t li = 0; li != m_p; ++li) {
        if (exclusive)
          Xr(rows[li], 0) = Estimate<F>::X(rows[li], centre);
        else
          Xr.row(rows[li]).head(n) = Estimate<F>::X.row(rows[li]).head(n);
      }
      out_rank += r_p;
      sum_weights += double(r_p);
      sum_variance += sp.head(r_p).sum();
      continue;
    }

    // Per-partition filter weights from the single pooled noise level and this partition's beta.
    if (w.size() < r_p)
      w.resize(r_p);
    double sw_p = 0.0;
    double sv_p = 0.0;
    ssize_t orank_p = 0;
    switch (filter) {
    case filter_type::OPTSHRINK: {
      w.head(rz).setZero();
      const double transition = 1.0 + std::sqrt(beta);
      for (ssize_t i = rz; i != r_p; ++i) {
        const double lam = sp[i] / qnz;
        const double y = std::sqrt(lam / sigma2);
        double nu = 0.0;
        if (y > transition) {
          nu = std::min(y, std::sqrt(Math::pow2(Math::pow2(y) - beta - 1.0) - (4.0 * beta)) / y);
          ++orank_p;
        }
        w[i] = lam > 0.0 ? (nu / y) : 0.0;
        sw_p += w[i];
        sv_p += w[i] * sp[i];
      }
    } break;
    case filter_type::OPTTHRESH: {
      const std::map<double, double>::const_iterator it = beta2lambdastar.find(beta);
      double lambda_star = 0.0;
      if (it == beta2lambdastar.end()) {
        lambda_star =
            sqrt(2.0 * (beta + 1.0) + ((8.0 * beta) / (beta + 1.0 + std::sqrt(Math::pow2(beta) + 14.0 * beta + 1.0))));
        beta2lambdastar[beta] = lambda_star;
      } else {
        lambda_star = it->second;
      }
      const double tau_star = lambda_star * std::sqrt(double(qnz)) * std::sqrt(sigma2);
      const double thresh = tau_star * Math::pow2(double(qnz));
      w.head(rz).setZero();
      for (ssize_t i = rz; i != r_p; ++i) {
        if (sp[i] >= thresh) {
          w[i] = 1.0;
          ++orank_p;
          sv_p += sp[i];
        } else {
          w[i] = 0.0;
        }
      }
      sw_p = double(orank_p);
    } break;
    case filter_type::TRUNCATE: {
      const ssize_t cp = cutoff_part[p];
      orank_p = r_p - cp;
      w.head(cp).setZero();
      w.segment(cp, orank_p).setOnes();
      sw_p = double(orank_p);
      sv_p = w.head(r_p).matrix().dot(sp.head(r_p).matrix());
    } break;
    default:
      assert(false);
    }
    out_rank += orank_p;
    sum_weights += sw_p;
    sum_variance += sv_p;

    // Forward-project this partition's filtered spectrum (in the demeaned domain). The diagonal
    //   weight expressions are written inline (not stored) to avoid dangling references to the
    //   temporary coefficient-wise array, matching the non-partitioned reconstruction.
    switch (Estimate<F>::decomp) {
    case decomp_type::BDCSVD: {
      const MatrixType &U = Estimate<F>::U_partition[p];
      const MatrixType &V = Estimate<F>::V_partition[p];
      if (exclusive)
        Xr_partition.noalias() =
            U *
            (w.head(r_p).reverse().template cast<F>().array() * Estimate<F>::sv_partition[p].array())
                .matrix()
                .asDiagonal() *
            V.row(centre).adjoint();
      else
        Xr_partition.noalias() =
            U *
            (w.head(r_p).reverse().template cast<F>().array() * Estimate<F>::sv_partition[p].array())
                .matrix()
                .asDiagonal() *
            V.adjoint();
    } break;
    case decomp_type::SELFADJOINT: {
      const MatrixType &evec = Estimate<F>::evec_partition[p];
      const MatrixType &Xp = Estimate<F>::Xsub_partition[p];
      if (m_p <= n) {
        if (exclusive)
          Xr_partition.noalias() =
              evec * (w.head(r_p).template cast<F>().matrix().asDiagonal() * (evec.adjoint() * Xp.col(centre)));
        else
          Xr_partition.noalias() =
              evec * (w.head(r_p).template cast<F>().matrix().asDiagonal() * (evec.adjoint() * Xp));
      } else {
        if (exclusive)
          Xr_partition.noalias() =
              Xp * (evec * (w.head(r_p).template cast<F>().matrix().asDiagonal() * evec.adjoint().col(centre)));
        else
          Xr_partition.noalias() = Xp * (evec * (w.head(r_p).template cast<F>().matrix().asDiagonal() * evec.adjoint()));
      }
    } break;
    }

    // Re-add the per-(group) stabilised-domain means subtracted prior to decomposition.
    if (!Estimate<F>::volume_group.empty()) {
      const MatrixType &means = Estimate<F>::means_partition[p];
      for (ssize_t li = 0; li != m_p; ++li) {
        const ssize_t g = Estimate<F>::volume_group[rows[li]];
        if (exclusive)
          Xr_partition(li, 0) += means(g, centre);
        else
          Xr_partition.row(li).head(n) += means.row(g);
      }
    }

    // Scatter this partition's reconstructed rows into the full set of volumes.
    for (ssize_t li = 0; li != m_p; ++li) {
      if (exclusive)
        Xr(rows[li], 0) = Xr_partition(li, 0);
      else
        Xr.row(rows[li]).head(n) = Xr_partition.row(li).head(n);
    }
  }
  const double variance_removed = 1.0 - sum_variance / total_s;

  // Spatial aggregation over the patch (mirrors the non-partitioned path, operating on Xr).
  if (exclusive) {
    assign_pos_of(dwi).to(out);
    out.row(3) = Xr.col(0);
    if (Estimate<F>::exports.sum_aggregation.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.sum_aggregation);
      Estimate<F>::exports.sum_aggregation.value() = 1.0;
    }
    if (Estimate<F>::exports.rank_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.rank_output);
      Estimate<F>::exports.rank_output.value() = out_rank;
    }
  } else {
    std::lock_guard<std::mutex> lock(Estimate<F>::mutex);
    for (size_t voxel_index = 0; voxel_index != Estimate<F>::patch.voxels.size(); ++voxel_index) {
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(out);
      assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index).to(Estimate<F>::exports.sum_aggregation);
      double weight = std::numeric_limits<double>::signaling_NaN();
      switch (aggregator) {
      case aggregator_type::EXCLUSIVE:
        assert(false);
        break;
      case aggregator_type::GAUSSIAN:
        weight = std::exp(gaussian_multiplier * Estimate<F>::patch.voxels[voxel_index].sq_distance);
        break;
      case aggregator_type::INVL0:
        weight = 1.0 / (1 + out_rank);
        break;
      case aggregator_type::RANK:
        weight = out_rank;
        break;
      case aggregator_type::UNIFORM:
        weight = 1.0;
        break;
      }
      out.row(3) += weight * Xr.col(voxel_index);
      Estimate<F>::exports.sum_aggregation.value() += weight;
      if (Estimate<F>::exports.rank_output.valid()) {
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.rank_output);
        Estimate<F>::exports.rank_output.value() += weight * out_rank;
      }
    }
  }

  auto ss_index = Estimate<F>::subsample->in2ss({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (Estimate<F>::exports.sum_optshrink.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.sum_optshrink);
    Estimate<F>::exports.sum_optshrink.value() = sum_weights;
  }
  if (Estimate<F>::exports.variance_removed.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.variance_removed);
    Estimate<F>::exports.variance_removed.value() = variance_removed;
  }
}

template class Recon<float>;
template class Recon<cfloat>;
template class Recon<double>;
template class Recon<cdouble>;

} // namespace MR::Denoise
