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

#include "denoise/loop.h"

#include "image_helpers.h"

namespace MR::Denoise {

bool Sender::operator()(Kernel::Voxel::index_type &out) {
  if (index[0] >= size[0] || index[1] >= size[1] || index[2] >= size[2])
    return false;
  if (index.minCoeff() == -1) // First call after instance construction
    index = {0, 0, 0};
  else
    increment();
  while (valid() && !subsample->process(index))
    increment();
  out = index;
  return valid();
}

bool Sender::valid() const {
  return index[0] >= 0 && index[0] < size[0] && //
         index[1] >= 0 && index[1] < size[1] && //
         index[2] >= 0 && index[2] < size[2];
}

void Sender::increment() {
  if (++index[0] == size[0]) {
    index[0] = 0;
    if (++index[1] == size[1]) {
      index[1] = 0;
      if (++index[2] == size[2]) {
        // Gone past the final voxel in the image; explicitly flag
        index[0] = size[0];
        index[1] = size[1];
      }
    }
  }
}

const std::unordered_map<operation_type, std::string> ReceiverEstimate::progress_messages{
  {operation_type::ESTIMATE_ONLY, "Running PCA noise level estimation"},
  {operation_type::RECON_ONLY, "Running PCA denoising"},
  {operation_type::ESTIMATE_AND_RECON, "Running PCA noise level estimation and denoising"}
};

ReceiverEstimate::ReceiverEstimate(std::shared_ptr<Subsample> subsample,
                                   Exports &exports,
                                  const operation_type operation) :
    subsample(subsample),
    exports (exports),
    pca_invalid_count(0),
    progress(progress_messages.at(operation), voxel_count(subsample->header())) {}

ReceiverEstimate::~ReceiverEstimate() {
  if (pca_invalid_count > 0) {
    WARN("A total of " + str(pca_invalid_count) + " PCA kernels failed to converge");
  }
}

bool ReceiverEstimate::operator()(const EstimatedPatch &in) {
  if (!in.valid)
    ++pca_invalid_count;
  // Store additional output maps if requested
  auto ss_index = subsample->in2ss(in.patch.seed_voxel);
  if (exports.noise_out.valid()) {
    assign_pos_of(ss_index).to(exports.noise_out);
    exports.noise_out.value() = static_cast<bool>(in.threshold)                          //
                                    ? static_cast<float>(std::sqrt(in.threshold.sigma2)) //
                                    : std::numeric_limits<float>::quiet_NaN();           //
  }
  if (exports.lamplus.valid()) {
    assign_pos_of(ss_index).to(exports.lamplus);
    exports.lamplus.value() = in.threshold.lamplus;
  }
  if (exports.rank_pcanonzero.valid()) {
    assign_pos_of(ss_index).to(exports.rank_pcanonzero);
    exports.rank_pcanonzero.value() = in.rank_pcanonzero;
  }
  if (exports.rank_input.valid()) {
    assign_pos_of(ss_index).to(exports.rank_input);
    if (!in.valid)
      exports.rank_input.value() = 0;
    else if (static_cast<bool>(in.threshold))
      exports.rank_input.value() = in.rank_pca - in.threshold.cutoff_p;
    else
      exports.rank_input.value() = in.rank_pca;
  }
  if (exports.max_dist.valid()) {
    assign_pos_of(ss_index).to(exports.max_dist);
    exports.max_dist.value() = in.patch.max_distance;
  }
  if (exports.voxelcount.valid()) {
    assign_pos_of(ss_index).to(exports.voxelcount);
    exports.voxelcount.value() = in.num_voxels();
  }
  if (exports.patchcount.valid()) {
    for (const auto &v : in.patch.voxels) {
      assign_pos_of(v.index).to(exports.patchcount);
      exports.patchcount.value() = exports.patchcount.value() + 1;
    }
  }
  if (exports.saving_eigenspectra())
    exports.add_eigenspectrum(in.eigenspectrum);
  ++progress;
  return true;
}

template <typename F>
ReceiverRecon<F>::ReceiverRecon(Image<F> &output,
                                std::shared_ptr<Subsample> subsample,
                                Exports &exports,
                                const operation_type operation) :
    ReceiverEstimate(subsample, exports, operation),
    output(output) {
  assert(operation != operation_type::ESTIMATE_ONLY);
}

template <typename F>
bool ReceiverRecon<F>::operator()(const ReconstructedPatch<F> &in) {

  for (size_t voxel_index = 0; voxel_index != in.num_voxels(); ++voxel_index) {
    assign_pos_of(in.patch.voxels[voxel_index].index, 0, 3).to(output);
    assign_pos_of(in.patch.voxels[voxel_index].index).to(exports.sum_aggregation);
    const double weight = in.aggregation_weights[voxel_index];
    output.row(3) += weight * in.Xr.col(voxel_index);
    exports.sum_aggregation.value() += weight;
    if (exports.rank_output.valid()) {
      assign_pos_of(in.patch.voxels[voxel_index].index, 0, 3).to(exports.rank_output);
      exports.rank_output.value() += weight * in.rank_recon;
    }
  }

  auto ss_index = subsample->in2ss(in.patch.seed_voxel);
  if (exports.sum_optshrink.valid()) {
    assign_pos_of(ss_index, 0, 3).to(exports.sum_optshrink);
    exports.sum_optshrink.value() = in.sum_shrinkage_weights;
  }
  if (exports.variance_removed.valid()) {
    assign_pos_of(ss_index, 0, 3).to(exports.variance_removed);
    exports.variance_removed.value() = in.variance_removed;
  }

  // Run this last only so that the progress bar increments at the end of the function
  (*this).ReceiverEstimate::operator()(in);

  return true;
}

template class ReceiverRecon<float>;
template class ReceiverRecon<cfloat>;
template class ReceiverRecon<double>;
template class ReceiverRecon<cdouble>;

} // namespace MR::Denoise
