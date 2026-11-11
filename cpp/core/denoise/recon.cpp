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

#include "denoise/recon.h"

#include "denoise/denoise.h"
#include "denoise/spectrum.h"
#include "denoise/spectrum_filter.h"
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
                const default_type aggregator_fwhm,
                Exports &exports,
                const ssize_t null_rank,
                std::shared_ptr<const Partitioning> level_partitioning,
                std::vector<ssize_t> volume_group,
                const demean_site_t demean_site,
                const ssize_t kernel_num_partitions)
    : Estimate<F>(image,
                  subsample,
                  kernel,
                  decomposition,
                  estimator,
                  exports,
                  null_rank,
                  true,
                  level_partitioning,
                  std::move(volume_group),
                  demean_site,
                  kernel_num_partitions),
      filter(filter),
      aggregator(aggregator),
      // Gaussian aggregator width. The reference length is the geometric mean over the three axes
      //   of the physical spacing between adjacent PCA patch centres (reconstruction sub-sample
      //   factor x voxel spacing); the FWHM is aggregator_fwhm times that spacing. The weight of a
      //   voxel at distance d (mm) from the patch centre is
      //     weight(d) = exp(gaussian_multiplier * d^2) = 2^(-(2d/FWHM)^2),
      //   so gaussian_multiplier = -4 ln2 / FWHM^2. The default aggregator_fwhm == 2 places the
      //   half-maximum exactly on the neighbouring patch centre and reproduces the width that was
      //   previously hard-coded (then written as "FWHM = 2 x cube root of spacings between kernels").
      gaussian_multiplier(-4.0 * std::log(2.0) /                                                    //
                          Math::pow2(aggregator_fwhm *                                              //
                                     std::cbrt(subsample->get_factors()[0] * image.spacing(0)       //
                                               * subsample->get_factors()[1] * image.spacing(1)     //
                                               * subsample->get_factors()[2] * image.spacing(2)))), //
      w(std::min(Estimate<F>::m, kernel->estimated_size())),
      Xr(Estimate<F>::m, aggregator == aggregator_type::EXCLUSIVE ? 1 : kernel->estimated_size()) {}

// Note on the exactness of the exported spectral moments.
//   Reconstruction applies to each voxel's own column a symmetric positive semi-definite contraction
//   S_k = U_k W_k U_k^H in volume space, and aggregation forms the composite filter
//   Sbar = (sum_k a_k S_k) / (sum_k a_k). Because tr() is linear, the aggregation-weighted mean of
//   the per-patch first moments is exactly tr(Sbar), so trace_output is exact. The higher moments are
//   not linear in the filter: tr(Sbar^2) involves inter-patch cross terms tr(S_j S_k) that do not
//   factorise, so the weighted means accumulated into trace2_accum / trace4_accum neglect those terms
//   and dof_output is correspondingly approximate (it is exact for -aggregator exclusive, where a
//   single patch contributes per voxel). Computing them exactly would require materialising the
//   m x m matrix Sbar per voxel, which is not affordable. The tractable exact route is Hutchinson
//   probing: append K fixed random probe vectors (Rademacher or Gaussian, in R^m, shared globally
//   across all voxels) as extra columns of X before the reconstruction multiply, and accumulate
//   sum_k a_k S_k z_j per output voxel exactly as the data is accumulated, yielding Sbar z_j. Then
//   tr(Sbar) = E[z^H Sbar z] and tr(Sbar^2) = E[||Sbar z||^2] are unbiased over the probes, with
//   relative standard error ~sqrt(2 / (r K)) on the trace of a rank-r projector (~8% at r=10, K=32).
//   Cost is K extra columns in an already-formed matmul plus K/m of the output image in storage; the
//   probes also absorb the demeaning projector and the partition block structure automatically.
template <typename F> void Recon<F>::operator()(Image<F> &dwi, Image<F> &out) {

  if (!Estimate<F>::subsample->process({dwi.index(0), dwi.index(1), dwi.index(2)}))
    return;

  Estimate<F>::operator()(dwi);

  // If only the data computed when this voxel was the centre of the patch
  //   is to be used for synthesis of the output image,
  //   then only that individual column needs to be reconstructed;
  //   if however the result from this patch is to contribute to the synthesized image
  //   for all voxels that were utilised within this patch,
  //   then we need to instead compute the full projection
  const bool exclusive = (aggregator == aggregator_type::EXCLUSIVE);
  Moments moments;
  Estimate<F>::project_patch(filter, exclusive, Xr, w, moments);

  if (exclusive) {
    assert(Estimate<F>::patch.centre_index >= 0);
    assign_pos_of(dwi).to(out);
    out.row(3) = Xr.col(0);
    if (Estimate<F>::exports.sum_aggregation.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.sum_aggregation);
      Estimate<F>::exports.sum_aggregation.value() = 1.0;
    }
    if (Estimate<F>::exports.trace_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.trace_output);
      Estimate<F>::exports.trace_output.value() = moments.sum_weights;
    }
    if (Estimate<F>::exports.dof_output.valid()) {
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.trace2_accum);
      assign_pos_of(dwi, 0, 3).to(Estimate<F>::exports.trace4_accum);
      Estimate<F>::exports.trace2_accum.value() = moments.sum_sqweights;
      Estimate<F>::exports.trace4_accum.value() = moments.sum_quadweights;
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
        weight = 1.0 / (1 + moments.rank);
        break;
      case aggregator_type::RANK:
        weight = moments.rank;
        break;
      case aggregator_type::UNIFORM:
        weight = 1.0;
        break;
      }
      out.row(3) += weight * Xr.col(voxel_index);
      Estimate<F>::exports.sum_aggregation.value() += weight;
      // First moment aggregates exactly; higher moments do not (see the note above operator()).
      if (Estimate<F>::exports.trace_output.valid()) {
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.trace_output);
        Estimate<F>::exports.trace_output.value() += weight * moments.sum_weights;
      }
      if (Estimate<F>::exports.dof_output.valid()) {
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.trace2_accum);
        assign_pos_of(Estimate<F>::patch.voxels[voxel_index].index, 0, 3).to(Estimate<F>::exports.trace4_accum);
        Estimate<F>::exports.trace2_accum.value() += weight * moments.sum_sqweights;
        Estimate<F>::exports.trace4_accum.value() += weight * moments.sum_quadweights;
      }
    }
  }

  auto ss_index = Estimate<F>::subsample->in2ss({dwi.index(0), dwi.index(1), dwi.index(2)});
  if (Estimate<F>::exports.trace_patch.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.trace_patch);
    Estimate<F>::exports.trace_patch.value() = moments.sum_weights;
  }
  if (Estimate<F>::exports.variance_removed.valid()) {
    assign_pos_of(ss_index, 0, 3).to(Estimate<F>::exports.variance_removed);
    Estimate<F>::exports.variance_removed.value() = moments.variance_removed();
  }
}

template class Recon<float>;
template class Recon<cfloat>;
template class Recon<double>;
template class Recon<cdouble>;

} // namespace MR::Denoise
