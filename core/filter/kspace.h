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

#include <string>
#include <vector>

#include "adapter/replicate.h"
#include "filter/base.h"
#include "header.h"
#include "image.h"
#include "math/fft.h"
#include "types.h"

namespace MR::Filter {

const std::vector<std::string> kspace_window_choices({"tukey"});
enum class kspace_windowfn_t { TUKEY };
constexpr default_type default_tukey_width = 0.5;

/** \addtogroup Filters
@{ */

/*! Apply (or reverse) k-space filtering
 */
// TODO Add ability to undo prior filtering
class KSpace : public Base {
public:
  KSpace(const Header &H, Image<double> &window) : Base(H), window(window) {
    for (size_t axis = 0; axis != H.ndim(); ++axis) {
      if (axis < window.ndim() && window.size(axis) > 1)
        inner_axes.push_back(axis);
      else
        outer_axes.push_back(axis);
    }
  }

  template <class InputImageType, class OutputImageType>
  typename std::enable_if<std::is_same<typename InputImageType::value_type, std::complex<double>>::value, void>::type
  operator()(InputImageType &in, OutputImageType &out) {
    Image<cdouble> kspace;
    Image<cdouble> temp;
    {
      for (ssize_t n = 0; n != inner_axes.size(); ++n) {
        switch (n) {
        case 0:
          kspace = Image<cdouble>::scratch(in,
                                           "Scratch k-space for \"" + in.name() + "\""        //
                                               + (inner_axes.size() > 1 ? " (1 of 2)" : "")); //
          Math::FFT(in, kspace, inner_axes[0], FFTW_FORWARD, false);
          break;
        case 1:
          temp = kspace;
          kspace = Image<cdouble>::scratch(in,
                                           "Scratch k-space for \"" + in.name() + "\" " //
                                               + "(2 of 2)");                           //
          Math::FFT(temp, kspace, inner_axes[n], FFTW_FORWARD, false);
          break;
        default:
          std::swap(temp, kspace);
          Math::FFT(temp, kspace, inner_axes[n], FFTW_FORWARD, false);
          break;
        }
      }
    }

    if (outer_axes.size()) {
      Adapter::Replicate<Image<double>> replicating_window(window, in);
      for (auto l = Loop(in)(kspace, replicating_window); l; ++l)
        kspace.value() *= double(replicating_window.value());
    } else {
      for (auto l = Loop(in)(kspace, window); l; ++l)
        kspace.value() *= double(window.value());
    }

    for (ssize_t n = 0; n != inner_axes.size(); ++n) {
      if (n == inner_axes.size() - 1) {
        // Final FFT:
        //   use output image if applicable;
        //   perform amplitude transform if necessary
        do_final_fft(out, kspace, temp, inner_axes[n]);
      } else {
        Math::FFT(kspace, temp, inner_axes[n], FFTW_BACKWARD, false);
        std::swap(kspace, temp);
      }
    }
  }

  template <class InputImageType, class OutputImageType>
  typename std::enable_if<!std::is_same<typename InputImageType::value_type, cdouble>::value, void>::type
  operator()(InputImageType &in, OutputImageType &out) {
    Image<cdouble> temp = Image<cdouble>::scratch(in, "Scratch \"" + in.name() + "\" converted to cdouble for FFT");
    for (auto l = Loop(in)(in, temp); l; ++l)
      temp.value() = {double(in.value()), double(0)};
    (*this)(temp, out);
  }

  // TODO Static functions to create different windows
  //   (different windows may require different input parameters)
  // TODO Currently extending 1D window functions to ND via outer product;
  //   this is however not a unique choice
  static Image<double> window_tukey(const Header &header,                  //
                                    const std::vector<size_t> &inner_axes, //
                                    const default_type cosine_frac) {      //
    assert(cosine_frac >= 0.0 && cosine_frac <= 1.0);
    Image<double> window = Image<double>::scratch(make_window_header(header, inner_axes),
                                                  "Scratch Tukey filter window with alpha=" + str(cosine_frac));
    for (auto l = Loop(window)(window); l; ++l)
      window.value() = 1.0;
    for (auto axis : inner_axes) {
      const size_t N = header.size(axis);
      Eigen::Array<double, Eigen::Dynamic, 1> window1d = Eigen::Array<double, Eigen::Dynamic, 1>::Ones(N);
      const default_type transition_lower = 0.5 - 0.5 * cosine_frac;
      const default_type transition_upper = 0.5 + 0.5 * cosine_frac;
      // Beware of FFT being non-centred
      for (size_t n = 0; n != N; ++n) {
        const default_type pos = default_type(n) / default_type(N);
        if (pos > transition_lower && pos < transition_upper)
          window1d[n] = 0.5 + 0.5 * std::cos(2.0 * Math::pi * (pos - transition_lower) / cosine_frac);
      }
      window1d *= 1.0 / double(N);
      // Need to loop over all inner axes other than the current one
      std::vector<size_t> inner_excluding_axis;
      for (auto a : inner_axes) {
        if (a != axis)
          inner_excluding_axis.push_back(a);
      }
      window.reset();
      for (auto l = Loop(inner_excluding_axis)(window); l; ++l) {
        for (size_t n = 0; n != N; ++n) {
          window.index(axis) = n;
          window.value() *= window1d[n];
        }
      }
    }
    return window;
  }

protected:
  Image<double> window;
  std::vector<size_t> inner_axes;
  std::vector<size_t> outer_axes;

  template <class ImageType>
  typename std::enable_if<std::is_same<typename ImageType::value_type, cdouble>::value, void>::type
  do_final_fft(ImageType &out, Image<cdouble> &kspace, Image<cdouble> &scratch, const size_t axis) {
    TRACE;
    Math::FFT(kspace, out, axis, FFTW_BACKWARD, false);
  }
  template <class ImageType>
  typename std::enable_if<std::is_same<typename ImageType::value_type, cfloat>::value, void>::type
  do_final_fft(ImageType &out, Image<cdouble> &kspace, Image<cdouble> &scratch, const size_t axis) {
    TRACE;
    assert(scratch.valid());
    Math::FFT(kspace, scratch, axis, FFTW_BACKWARD, false);
    for (auto l = Loop(out)(scratch, out); l; ++l)
      out.value() = {float(cdouble(scratch.value()).real()), float(cdouble(scratch.value()).imag())};
  }
  template <class ImageType>
  typename std::enable_if<!MR::is_complex<typename ImageType::value_type>::value, void>::type
  do_final_fft(ImageType &out, Image<cdouble> &kspace, Image<cdouble> &scratch, const size_t axis) {
    TRACE;
    assert(scratch.valid());
    Math::FFT(kspace, scratch, axis, FFTW_BACKWARD, false);
    for (auto l = Loop(out)(scratch, out); l; ++l)
      out.value() = std::abs(cdouble(scratch.value()));
  }

  static Header make_window_header(const Header &header, const std::vector<size_t> &inner_axes) {
    Header H(header);
    H.datatype() = DataType::Float64;
    H.datatype().set_byte_order_native();
    std::vector<size_t>::const_iterator it = inner_axes.begin();
    for (size_t axis = 0; axis != header.ndim(); ++axis) {
      if (it != inner_axes.end() && *it == axis)
        ++it;
      else
        H.size(axis) = 1;
    }
    squeeze_dim(H);
    return H;
  }
};
//! @}

} // namespace MR::Filter
