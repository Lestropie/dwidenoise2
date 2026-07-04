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

#include <algorithm>
#include <limits>
#include <string>

#include "algo/loop.h"
#include "algo/threaded_loop.h"
#include "datatype.h"
#include "exception.h"
#include "header.h"
#include "image.h"
#include "metadata/bids.h"

namespace MR::Denoise {

// Threshold on the number of voxels with invalid data above which a warning
//   (rather than merely an informational message) is justified.
// This is set to the largest number of voxels contained within a single slice
//   taken orthogonal to any of the three image axes.
// Where the slice encoding direction is known from the image metadata,
//   that axis is excluded from the quantification,
//   such that the presence of invalid data spanning as little as a single acquired slice
//   is sufficient to trigger the warning.
inline size_t invalid_data_warning_threshold(const Header &H) {
  size_t slice_axis = 3; // Sentinel: no slice encoding direction identified from metadata
  auto slice_encoding_it = H.keyval().find("SliceEncodingDirection");
  if (slice_encoding_it != H.keyval().end()) {
    const Metadata::BIDS::axis_vector_type dir = Metadata::BIDS::axisid2vector(slice_encoding_it->second);
    for (size_t axis = 0; axis != 3; ++axis) {
      if (dir[axis]) {
        slice_axis = axis;
        break;
      }
    }
  }
  size_t threshold = 0;
  for (size_t axis = 0; axis != 3; ++axis) {
    if (axis == slice_axis)
      continue;
    // Number of voxels within a single slice orthogonal to this axis
    size_t slice_voxels = 1;
    for (size_t a = 0; a != 3; ++a) {
      if (a != axis)
        slice_voxels *= H.size(a);
    }
    threshold = std::max(threshold, slice_voxels);
  }
  return threshold;
}

// Report on the presence of voxels with invalid data,
//   escalating from an informational message to a warning
//   only where the prevalence of such voxels strictly exceeds the threshold defined above
inline void report_invalid_data(const size_t excluded_count, const Header &H) {
  if (excluded_count == 0)
    return;
  const std::string message = "A total of " + str(excluded_count) +     //
                              " voxels were found with invalid data;"    //
                              " these will be excluded from processing"; //
  if (excluded_count > invalid_data_warning_threshold(H))
    WARN(message);
  else
    INFO(message);
}

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
  void operator()(Image<bool> &mask) {
    assign_pos_of(mask).to(image);
    bool all_finite = true;
    if constexpr (is_complex<T>::value) {
      using R = typename T::value_type;
      T min_value(std::numeric_limits<R>::infinity(), std::numeric_limits<R>::infinity());
      T max_value(-std::numeric_limits<R>::infinity(), -std::numeric_limits<R>::infinity());
      for (auto l_inner = Loop(image, 3, image.ndim())(image); l_inner; ++l_inner) {
        const T value = image.value();
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
          all_finite = false;
        } else {
          min_value = {std::min(min_value.real(), value.real()), std::min(min_value.imag(), value.imag())};
          max_value = {std::max(max_value.real(), value.real()), std::max(max_value.imag(), value.imag())};
        }
      }
      if (all_finite && min_value != max_value)
        mask.value() = true;
      else
        ++excluded;
    } else {
      T min_value(std::numeric_limits<T>::infinity());
      T max_value(-std::numeric_limits<T>::infinity());
      for (auto l_inner = Loop(image, 3, image.ndim())(image); l_inner; ++l_inner) {
        const T value = image.value();
        if (!std::isfinite(value)) {
          all_finite = false;
        } else {
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }
      }
      if (all_finite && min_value != max_value)
        mask.value() = true;
      else
        ++excluded;
    }
  }

private:
  Image<T> image;
  size_t excluded;
  size_t &grand_excluded;
};

template <typename T> Image<bool> generate_mask(Image<T> &image) {
  Header H(image);
  H.ndim() = 3;
  H.datatype() = DataType::Bit;
  Image<bool> mask = Image<bool>::scratch(H, "Scratch mask of voxels with valid data for denoising");
  size_t excluded_count(0);
  ThreadedLoop("Scanning image for invalid voxels", mask).run(MaskFunctor<T>(image, excluded_count), mask);
  report_invalid_data(excluded_count, H);
  return mask;
}

//template <typename T> typename std::enable_if<is_complex<T>::value, Image<bool>>::type generate_mask(Image<T> &image) {
//  Header H(image);
//  H.ndim() = 3;
//  Image<T> data = Image<T>::scratch(H, "Scratch data for detecting inequal values");
//  for (auto l = Loop(data)(data); l; ++l)
//    data.value() = T(std::numeric_limits<typename T::value_type>::quiet_NaN(),
//                     std::numeric_limits<typename T::value_type>::quiet_NaN());
//  H.datatype() = DataType::Bit;
//  Image<bool> nonzerovar_mask = Image<bool>::scratch(H, "Scratch mask of voxels with data with non-zero variance");
//  Image<bool> nonfinite_mask = Image<bool>::scratch(H, "Scratch mask of voxels with non-finite data");
//  size_t excluded_count(0);
//  for (auto l = Loop("Scanning image for invalid voxels")(image); l; ++l) {
//    const T value(static_cast<T>(image.value()));
//    if (std::isfinite(value.real()) && std::isfinite(value.imag())) {
//      assign_pos_of(image, 0, 3).to(data);
//      if (!std::isfinite(static_cast<T>(data.value()).real() && !std::isfinite(static_cast<T>(data.value()).imag()))) {
//        data.value() = image.value();
//      } else if (image.value() != data.value()) {
//        assign_pos_of(image, 0, 3).to(nonzerovar_mask);
//        nonzerovar_mask.value() = true;
//      }
//    } else {
//      assign_pos_of(image, 0, 3).to(nonfinite_mask);
//      nonfinite_mask.value() = true;
//    }
//  }
//  Image<bool> mask = Image<bool>::scratch(H, "Scratch mask of voxels with valid data");
//  for (auto l = Loop(mask)(nonzerovar_mask, nonfinite_mask, mask); l; ++l)
//    mask.value() = static_cast<bool>(nonzerovar_mask.value()) && !static_cast<bool>(nonfinite_mask.value());
//  return mask;
//}

//template <typename T> typename std::enable_if<!is_complex<T>::value, Image<bool>>::type generate_mask(Image<T> &image) {
//  Header H(image);
//  H.ndim() = 3;
//  Image<T> data = Image<T>::scratch(H, "Scratch data for detecting inequal values");
//  for (auto l = Loop(data)(data); l; ++l)
//    data.value() = std::numeric_limits<typename T::value_type>::quiet_NaN();
//  H.datatype() = DataType::Bit;
//  Image<bool> nonzerovar_mask = Image<bool>::scratch(H, "Scratch mask of voxels with data with non-zero variance");
//  Image<bool> nonfinite_mask = Image<bool>::scratch(H, "Scratch mask of voxels with non-finite data");
//  size_t excluded_count(0);
//  for (auto l = Loop("Scanning image for invalid voxels")(image); l; ++l) {
//    const T value(static_cast<T>(image.value()));
//    if (std::isfinite(value)) {
//      assign_pos_of(image, 0, 3).to(data);
//      if (!std::isfinite(static_cast<T>(data.value()))) {
//        data.value() = image.value();
//      } else if (image.value() != data.value()) {
//        assign_pos_of(image, 0, 3).to(nonzerovar_mask);
//        nonzerovar_mask.value() = true;
//      }
//    } else {
//      assign_pos_of(image, 0, 3).to(nonfinite_mask);
//      nonfinite_mask.value() = true;
//    }
//  }
//  Image<bool> mask = Image<bool>::scratch(H, "Scratch mask of voxels with valid data");
//  for (auto l = Loop(mask)(nonzerovar_mask, nonfinite_mask, mask); l; ++l)
//    mask.value() = static_cast<bool>(nonzerovar_mask.value()) && !static_cast<bool>(nonfinite_mask.value());
//  return mask;
//}

} // namespace MR::Denoise
