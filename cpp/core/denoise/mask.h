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

#include <cstddef>

#include "header.h"
#include "image.h"

namespace MR::Denoise {

// Threshold on the number of voxels with invalid data above which a warning
//   (rather than merely an informational message) is justified.
// This is set to the largest number of voxels contained within a single slice
//   taken orthogonal to any of the three image axes.
// Where the slice encoding direction is known from the image metadata,
//   that axis is excluded from the quantification,
//   such that the presence of invalid data spanning as little as a single acquired slice
//   is sufficient to trigger the warning.
size_t invalid_data_warning_threshold(const Header &H);

// Report on the presence of voxels with invalid data,
//   escalating from an informational message to a warning
//   only where the prevalence of such voxels strictly exceeds the threshold defined above
void report_invalid_data(const size_t excluded_count, const Header &H);

// Need to sweep through the input data,
//   identify voxels that cannot be utilised in PCA,
//   and generate a mask that will preclude them from contributing
// This can only be done after an Image<> instance has been created,
//   which is typically templated based on data / user input

// TODO These functions should also take an optional VST image as input,
//   and exclude from the mask those voxels where a valid noise level reading can't be obtained
// Note however that this may change between iterations,
//   which conflicts with how this is currently managed,
//   where the mask is computed only once before the first iteration

// TODO This operation may run faster if,
//   rather than looping over voxels as an outer loop then volumes as an inner loop,
//   the image were instead looped over along contiguous strides,
//   with local scratch buffers tracking presence of non-finite values and minima/maxima
//   (inded maxima / minima might not be required;
//    just anything that is neither zero nor an existing value might suffice?)

// Per-spatial-voxel mask-generation functor for ThreadedLoop. The 3-D mask is the looped image;
//   each invocation examines all volumes at that voxel (held as a private per-thread copy of the
//   input image) and marks the voxel valid iff every volume is finite and the values are not all
//   identical (zero variance). "excluded_count" is a reduction: each thread accumulates its own
//   tally and adds it to the shared grand total in the destructor, which runs only after all
//   threads have rejoined (the RMS idiom documented in ThreadedLoop). Handles complex and real
//   input through a single compile-time branch.
template <typename T> class MaskFunctor {
public:
  MaskFunctor(Image<T> &image, size_t &grand_excluded)
      : image(image), excluded(0), grand_excluded(grand_excluded) {}
  ~MaskFunctor() { grand_excluded += excluded; }
  void operator()(Image<bool> &mask);

private:
  Image<T> image;
  size_t excluded;
  size_t &grand_excluded;
};

template <typename T> Image<bool> generate_mask(Image<T> &image);

} // namespace MR::Denoise
