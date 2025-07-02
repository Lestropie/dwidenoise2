/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#include "denoise/demodulate.h"

#include "app.h"
#include "axes.h"

using namespace MR::App;

namespace MR::Denoise {

const char *const demodulation_description = "If the input data are of complex type, "
                                             "then a linear phase term will be removed from each k-space prior to PCA. "
                                             "In the absence of metadata indicating otherwise, "
                                             "it is inferred that the first two axes correspond to acquired slices, "
                                             "and different slices / volumes will be demodulated individually; "
                                             "this behaviour can be modified using the -demod_axes option.";

const OptionGroup demodulation_options =
    OptionGroup("Options for phase demodulation of complex data") +
    Option("nodemod", "disable phase demodulation")
    // TODO Consider option to disable the remodulation of the output denoised series;
    //   would need to turn this into a function call,
    //   as that option would need to be omitted from dwi2noise
    // Perhaps -nodemod, -noremod could be combined into a type_choice()?
    // This wouldn't be able to also cover the future prospect of linear vs. non-linear phase demodulation;
    //   maybe input phase demodulation being none / linear / nonlinear would be the better type_choice()?
    +
    Option("demod_axes", "comma-separated list of axis indices along which FFT can be applied for phase demodulation") +
    Argument("axes").type_sequence_int();

std::vector<size_t> get_demodulation_axes(const Header &H) {
  const bool complex = H.datatype().is_complex();
  auto opt = App::get_options("nodemod");
  if (!opt.empty()) {
    if (!App::get_options("demod_axes").empty())
      throw Exception("Options -nodemod and -demod_axes are mutually exclusive");
    return std::vector<size_t>();
  }
  opt = App::get_options("demod_axes");
  if (opt.empty()) {
    if (complex) {
      auto slice_encoding_it = H.keyval().find("SliceEncodingDirection");
      if (slice_encoding_it == H.keyval().end()) {
        INFO("No header information on slice encoding; assuming first two axes are within-slice");
        return {0, 1};
      } else {
        auto dir = Axes::id2dir(slice_encoding_it->second);
        std::vector<size_t> result;
        for (size_t axis = 0; axis != 3; ++axis) {
          if (!dir[axis])
            result.push_back(axis);
        }
        INFO("For header SliceEncodingDirection=\"" + slice_encoding_it->second +
             "\", "
             "chose demodulation axes: " +
             join(result, ","));
        return result;
      }
    }
  } else {
    if (!complex)
      throw Exception("Cannot perform phase demodulation on magnitude input image");
    auto result = parse_ints<size_t>(opt[0][0]);
    for (auto axis : result) {
      if (axis > 2)
        throw Exception("Phase demodulation implementation not yet robust to non-spatial axes");
    }
    return result;
  }
  return std::vector<size_t>();
}

} // namespace MR::Denoise
