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

#include "denoise/precondition.h"

#include <limits>

#include "algo/copy.h"
#include "app.h"
#include "axes.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "transform.h"

using namespace MR::App;

namespace MR::Denoise {

const char *const demodulation_description =
    "If the input data are of complex type, "
    "then a smooth non-linear phase will be demodulated removed from each k-space prior to PCA. "
    "In the absence of metadata indicating otherwise, "
    "it is inferred that the first two axes correspond to acquired slices, "
    "and different slices / volumes will be demodulated individually; "
    "this behaviour can be modified using the -demod_axes option. "
    "A strictly linear phase term can instead be regressed from each k-space, "
    "similarly to performed in Cordero-Grande et al. 2019, "
    "by specifying -demodulate linear.";

// clang-format off
const OptionGroup precondition_options = OptionGroup("Options for preconditioning data prior to PCA")
  + Option("demodulate",
           "select form of phase demodulation; "
           "options are: " + join(demodulation_choices, ",") + " "
           "(default: nonlinear)")
    + Argument("mode").type_choice(demodulation_choices)
  + Option("demod_axes",
           "comma-separated list of axis indices along which FFT can be applied for phase demodulation")
    + Argument("axes").type_sequence_int()
  + Option("demean",
           "select method of demeaning prior to PCA; "
           "options are: " + join(demean_choices, ",") + " "
           "(default: 'shells' if DWI gradient table available, 'all' otherwise)")
    + Argument("mode").type_choice(demean_choices)
  + Option("vst",
           "apply a within-patch variance-stabilising transformation based on a pre-estimated noise level map")
    + Argument("image").type_image_in()
  + Option("preconditioned",
           "export the preconditioned version of the input image that is the input to PCA")
    + Argument("image").type_image_out();
// clang-format on

Demodulation select_demodulation(const Header &H) {
  const bool complex = H.datatype().is_complex();
  auto opt_mode = get_options("demodulate");
  auto opt_axes = get_options("demod_axes");
  Demodulation result;
  if (opt_mode.empty()) {
    if (complex) {
      result.mode = demodulation_t::NONLINEAR;
    } else {
      if (!opt_axes.empty()) {
        throw Exception("Option -demod_axes cannot be specified: "
                        "no phase demodulation of magnitude data");
      }
    }
  } else {
    result.mode = demodulation_t(int(opt_mode[0][0]));
    if (!complex) {
      switch (result.mode) {
      case demodulation_t::NONE:
        WARN("Specifying -demodulate none is redundant: "
             "never any phase demodulation for magnitude input data");
        break;
      default:
        throw Exception("Phase modulation cannot be utilised for magnitude-only input data");
      }
    }
  }
  if (!complex)
    return result;
  if (opt_axes.empty()) {
    auto slice_encoding_it = H.keyval().find("SliceEncodingDirection");
    if (slice_encoding_it == H.keyval().end()) {
      // TODO Ideally this would be the first two axes *on disk*,
      //   not following transform realignment
      INFO("No header information on slice encoding; "
           "assuming first two axes are within-slice");
      result.axes = {0, 1};
    } else {
      auto dir = Axes::id2dir(slice_encoding_it->second);
      for (size_t axis = 0; axis != 3; ++axis) {
        if (!dir[axis])
          result.axes.push_back(axis);
      }
      INFO("For header SliceEncodingDirection=\"" + slice_encoding_it->second +
           "\", "
           "chose demodulation axes: " +
           join(result.axes, ","));
    }
  } else {
    result.axes = parse_ints<size_t>(opt_axes[0][0]);
    for (auto axis : result.axes) {
      if (axis > 2)
        throw Exception("Phase demodulation implementation not yet robust to non-spatial axes");
    }
  }
  return result;
}

demean_type select_demean(const Header &H) {
  auto opt = get_options("demean");
  if (opt.empty()) {
    try {
      auto grad = DWI::get_DW_scheme(H);
      auto shells = DWI::Shells(grad);
      INFO("Choosing to demean per b-value shell based on input gradient table");
      return demean_type::SHELLS;
    } catch (Exception &) {
      INFO("Choosing to demean across all volumes based on absent / non-shelled gradient table");
      return demean_type::ALL;
    }
  }
  return demean_type(int(opt[0][0]));
}

template <typename T>
Precondition<T>::Precondition(Image<T> &image,
                              const Demodulation &demodulation,
                              const demean_type demean,
                              Image<float> &vst_image)
    : H(image),              //
      vst_image(vst_image) { //

  // Step 1: Phase demodulation
  Image<T> dephased;
  if (demodulation.mode == demodulation_t::NONE) {
    dephased = image;
  } else {
    typename DemodulatorSelector<T>::type demodulator(image,                                        //
                                                      demodulation.axes,                            //
                                                      demodulation.mode == demodulation_t::LINEAR); //
    phase_image = demodulator();
    // Only actually perform the dephasing of the input image
    //   if that result needs to be utilised in calculation of the mean
    if (demean != demean_type::NONE) {
      dephased = Image<T>::scratch(H, "Scratch dephased version of \"" + image.name() + "\" for mean calculation");
      demodulator(image, dephased, false);
    }
  }

  // Step 2: Demeaning
  Header H_mean(H);
  switch (demean) {
  case demean_type::NONE:
    break;
  case demean_type::SHELLS: {
    Eigen::Matrix<default_type, Eigen::Dynamic, Eigen::Dynamic> grad;
    try {
      grad = DWI::get_DW_scheme(H_mean);
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to obtain valid gradient table");
    }
    try {
      DWI::Shells shells(grad);
      vol2shellidx.resize(image.size(3), -1);
      for (ssize_t shell_idx = 0; shell_idx != shells.count(); ++shell_idx) {
        for (auto v : shells[shell_idx].get_volumes())
          vol2shellidx[v] = shell_idx;
      }
      assert(*std::min_element(vol2shellidx.begin(), vol2shellidx.end()) == 0);
      H_mean.size(3) = shells.count();
      DWI::stash_DW_scheme(H_mean, grad);
      mean_image = Image<T>::scratch(H_mean, "Scratch image for per-shell mean intensity");
      for (auto l_voxel = Loop("Computing mean intensities within shells", H_mean, 0, 3)(dephased, mean_image); //
           l_voxel;                                                                                             //
           ++l_voxel) {                                                                                         //
        for (ssize_t volume_idx = 0; volume_idx != image.size(3); ++volume_idx) {
          dephased.index(3) = volume_idx;
          mean_image.index(3) = vol2shellidx[volume_idx];
          mean_image.value() += dephased.value();
        }
        for (ssize_t shell_idx = 0; shell_idx != shells.count(); ++shell_idx) {
          mean_image.index(3) = shell_idx;
          mean_image.value() /= T(shells[shell_idx].count());
        }
      }
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to establish b-value shell structure");
    }
  } break;
  case demean_type::ALL: {
    H_mean.ndim() = 3;
    DWI::clear_DW_scheme(H_mean);
    mean_image = Image<T>::scratch(H_mean, "Scratch image for mean intensity across all volumes");
    for (auto l_voxel = Loop("Computing mean intensity across all volumes", H_mean)(dephased, mean_image); //
         l_voxel;                                                                                          //
         ++l_voxel) {                                                                                      //
      T mean(T(0));
      for (auto l_volume = Loop(3)(dephased); l_volume; ++l_volume)
        mean += T(dephased.value());
      mean_image.value() = mean / T(image.size(3));
    }
  } break;
  }

  // Step 3: Variance-stabilising transform
  // Image<float> vst is already set within constructor definition;
  //   nothing to do here
}

namespace {
// Private functions to prevent compiler attempting to create complex functions for real types
template <typename T>
typename std::enable_if<std::is_same<T, cfloat>::value, T>::type demodulate(const cfloat in, const cfloat phase) {
  return in * std::conj(phase);
}
template <typename T>
typename std::enable_if<std::is_same<T, cdouble>::value, T>::type demodulate(const cdouble in, const cfloat phase) {
  return in * std::conj(cdouble(phase));
}
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type demodulate(const T in, const cfloat phase) {
  assert(false);
  return in;
}
template <typename T>
typename std::enable_if<std::is_same<T, cfloat>::value, T>::type modulate(const cfloat in, const cfloat phase) {
  return in * phase;
}
template <typename T>
typename std::enable_if<std::is_same<T, cdouble>::value, T>::type modulate(const cdouble in, const cfloat phase) {
  return in * cdouble(phase);
}
template <typename T> typename std::enable_if<!is_complex<T>::value, T>::type modulate(const T in, const cfloat phase) {
  assert(false);
  return in;
}
} // namespace

template <typename T> void Precondition<T>::operator()(Image<T> input, Image<T> output, const bool inverse) const {

  // For thread-safety / const-ness
  const Transform transform(input);
  Image<cfloat> phase(phase_image);
  Image<T> mean(mean_image);
  std::unique_ptr<Interp::Cubic<Image<float>>> vst;
  if (vst_image.valid())
    vst.reset(new Interp::Cubic<Image<float>>(vst_image));

  Eigen::Array<T, Eigen::Dynamic, 1> data(input.size(3));
  if (inverse) {
    for (auto l_voxel = Loop("Reversing data preconditioning", H, 0, 3)(input, output); l_voxel; ++l_voxel) {

      // Step 3: Reverse variance-stabilising transform
      if (vst) {
        vst->scanner(transform.voxel2scanner *                         //
                     Eigen::Vector3d({default_type(input.index(0)),    //
                                      default_type(input.index(1)),    //
                                      default_type(input.index(2))})); //
        const T multiplier = T(vst->value());
        for (ssize_t v = 0; v != input.size(3); ++v) {
          input.index(3) = v;
          data[v] = T(input.value()) * multiplier;
        }
      } else {
        for (ssize_t v = 0; v != input.size(3); ++v) {
          input.index(3) = v;
          data[v] = input.value();
        }
      }

      // Step 2: Reverse demeaning
      if (mean.valid()) {
        assign_pos_of(input, 0, 3).to(mean);
        if (mean.ndim() == 3) {
          const T mean_value = mean.value();
          data += mean_value;
        } else {
          for (ssize_t v = 0; v != input.size(3); ++v) {
            mean.index(3) = vol2shellidx[v];
            data[v] += T(mean.value());
          }
        }
      }

      // Step 1: Reverse phase demodulation
      if (phase.valid()) {
        assign_pos_of(input, 0, 3).to(phase);
        for (ssize_t v = 0; v != input.size(3); ++v) {
          phase.index(3) = v;
          data[v] = modulate<T>(data[v], phase.value());
        }
      }

      // Write to output
      for (ssize_t v = 0; v != input.size(3); ++v) {
        output.index(3) = v;
        output.value() = data[v];
      }
    }
    return;
  }

  // Applying forward preconditioning
  for (auto l_voxel = Loop("Applying data preconditioning", H, 0, 3)(input, output); l_voxel; ++l_voxel) {

    // Step 1: Phase demodulation
    if (phase.valid()) {
      assign_pos_of(input, 0, 3).to(phase);
      for (ssize_t v = 0; v != input.size(3); ++v) {
        input.index(3) = v;
        phase.index(3) = v;
        data[v] = demodulate<T>(input.value(), phase.value());
      }
    } else {
      for (ssize_t v = 0; v != input.size(3); ++v) {
        input.index(3) = v;
        data[v] = input.value();
      }
    }

    // Step 2: Demeaning
    if (mean.valid()) {
      assign_pos_of(input, 0, 3).to(mean);
      if (mean.ndim() == 3) {
        const T mean_value = mean.value();
        for (ssize_t v = 0; v != input.size(3); ++v)
          data[v] -= mean_value;
      } else {
        for (ssize_t v = 0; v != input.size(3); ++v) {
          mean.index(3) = vol2shellidx[v];
          data[v] -= T(mean.value());
        }
      }
    }

    // Step 3: Variance-stabilising transform
    if (vst) {
      vst->scanner(transform.voxel2scanner                             //
                   * Eigen::Vector3d({default_type(input.index(0)),    //
                                      default_type(input.index(1)),    //
                                      default_type(input.index(2))})); //
      const default_type multiplier = 1.0 / vst->value();
      data *= multiplier;
    }

    // Write to output
    for (ssize_t v = 0; v != input.size(3); ++v) {
      output.index(3) = v;
      output.value() = data[v];
    }
  }
}

} // namespace MR::Denoise
