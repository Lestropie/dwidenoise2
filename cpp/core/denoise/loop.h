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

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "denoise/estimate.h"
#include "denoise/exports.h"
#include "denoise/kernel/voxel.h"
#include "denoise/recon.h"
#include "denoise/subsample.h"
#include "header.h"
#include "image.h"
#include "progressbar.h"

namespace MR::Denoise {

enum class operation_type { ESTIMATE_ONLY, RECON_ONLY, ESTIMATE_AND_RECON };


// TODO New classes for changing how the image FoV is looped over:
// 1. A class that reads in the subsampling information,
//    and populates a Thread::Queue of Kernel::Voxel::index_type's
//    with the locations of the PCA patch centres
// 2. Classes that receive as input the outcomes of the PCA decompositions
//    and write the results into the respective classes
//    This will differ between:
//    1. Estimation of noise level only
//       (plus potentially other images)
//    2. All of that, but also denoised image data
//       (plus potentially other images)

class Sender {
public:
  template <class HeaderType>
  Sender(const HeaderType &voxel_grid, std::shared_ptr<Subsample> subsample) :
      size({voxel_grid.size(0), voxel_grid.size(1), voxel_grid.size(2)}),
      subsample(subsample),
      index({-1, -1, -1}) {}
  bool operator()(Kernel::Voxel::index_type &);
private:
  // TODO Change to Kernel::Voxel::index_type?
  const std::array<ssize_t, 3> size;
  std::shared_ptr<Subsample> subsample;
  Kernel::Voxel::index_type index;

  bool valid() const;
  void increment();
};

class ReceiverEstimate {
public:
  ReceiverEstimate(std::shared_ptr<Subsample> subsample,
                   Exports &exports,
                   const operation_type operation = operation_type::ESTIMATE_ONLY);
  ~ReceiverEstimate();
  bool operator()(const EstimatedPatch &);
protected:
  std::shared_ptr<Subsample> subsample;
  // TODO If this wider-scale refactoring works,
  //   then this shouldn't need to do any kind of copy-construction
  Exports exports;
  ssize_t pca_invalid_count;
  // TODO This needs to construct a progress bar;
  //   ReceiverRecon also needs to have the ability to modify the progress message
  ProgressBar progress;
private:
  static const std::unordered_map<operation_type, std::string> progress_messages;
};

template <typename F>
class ReceiverRecon : private ReceiverEstimate {
public:
  ReceiverRecon(Image<F> &output,
                std::shared_ptr<Subsample> subsample,
                Exports &exports,
                const operation_type operation);
  bool operator()(const ReconstructedPatch<F> &);
private:
  Image<F> output;
};

} // namespace MR::Denoise
