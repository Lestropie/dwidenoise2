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

#include <algorithm>
#include <limits>
#include <memory>

#include "algo/copy.h"
#include "app.h"
#include "axes.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "interp/cubic.h"
#include "metadata/bids.h"
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
OptionGroup precondition_options(const bool include_output)
{
  OptionGroup result ("Options for preconditioning data prior to PCA");
  result
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
           "(default: 'shells' if DWI gradient table available; 'volume_groups' if volume groups present; 'all' otherwise)")
    + Argument("mode").type_choice(demean_choices)
  + Option("vst",
           "apply a within-patch variance-stabilising transformation based on a pre-estimated noise level map")
    + Argument("image").type_image_in();
  if (include_output) {
    result
    + Option("preconditioned_input",
             "export the preconditioned version of the input image that is the input to PCA")
      + Argument("image").type_image_out()
    + Option("preconditioned_output",
             "export the denoised data prior to reversal of preconditioning")
      + Argument("image").type_image_out();
  } else {
    result
    + Option("preconditioned",
             "export the preconditioned version of the input image that is the input to PCA")
      + Argument("image").type_image_out();
  }
  return result;
}
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
      auto dir = Metadata::BIDS::axisid2vector(slice_encoding_it->second);
      for (size_t axis = 0; axis != 3; ++axis) {
        if (!dir[axis])
          result.axes.push_back(axis);
      }
      INFO("For header SliceEncodingDirection=\"" + slice_encoding_it->second + "\", " + //
           "chose demodulation axes: " + join(result.axes, ","));                        //
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
  bool shells_available = false;
  try {
    auto grad = DWI::get_DW_scheme(H);
    auto shells = DWI::Shells(grad);
    shells_available = true;
  } catch (Exception &) {
  }
  const bool volume_groups_available = H.ndim() > 4;
  auto opt = get_options("demean");
  if (opt.empty()) {
    if (shells_available) {
      if (volume_groups_available)
        throw Exception("Cannot automatically determine how to demean, "
                        "as input series is >4D and also has a diffusion gradient table");
      INFO("Automatically demeaning per b-value shell based on input gradient table");
      return demean_type::SHELLS;
    }
    if (volume_groups_available) {
      INFO("Automatically demeaning by volume groups");
      return demean_type::VOLUME_GROUPS;
    }
    INFO("Automatically demeaning across all volumes");
    return demean_type::ALL;
  }
  const demean_type user_selection = demean_type(int(opt[0][0]));
  if (user_selection == demean_type::SHELLS && !shells_available)
    throw Exception("Cannot demean by b-value shells as shell structure could not be inferred");
  if (user_selection == demean_type::VOLUME_GROUPS && !volume_groups_available)
    throw Exception("Cannot demean by volume groups as image does not possess volume groups");
  return user_selection;
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

// Forward variance-stabilising transform applied to a single datum.
// For complex (Gaussian) data the transform is applied independently to the
//   real and imaginary channels, as documented by the NoiseModel interface.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_forward(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.stabilise(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_forward(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.stabilise(default_type(in.real()), sigma)), R(model.stabilise(default_type(in.imag()), sigma)));
}

// Algebraic inverse of the forward variance-stabilising transform,
//   recovering the conventional (still-biased) intensity scale.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_inverse(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.inverse_algebraic(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_inverse(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.inverse_algebraic(default_type(in.real()), sigma)),
           R(model.inverse_algebraic(default_type(in.imag()), sigma)));
}
} // namespace

template <typename T>
Precondition<T>::Precondition(Image<T> &image,
                              const Demodulation &demodulation,
                              const demean_type demean,
                              Image<float> &vst_noise,
                              std::shared_ptr<NoiseModel::Base> noise_model)
    : H_in(image),                   //
      H_out(image),                  //
      num_volume_groups(1),          //
      noise_model(noise_model),      //
      vst_noise_image(vst_noise) {   //

  for (ssize_t axis = 4; axis != H_in.ndim(); ++axis) {
    num_volume_groups *= H_in.size(axis);
    H_out.size(3) *= H_in.size(axis);
  }
  H_out.ndim() = 4;
  Stride::set(H_out, Stride::contiguous_along_axis(3));
  H_out.datatype() = DataType::from<T>();
  H_out.datatype().set_byte_order_native();

  if (H_in.ndim() > 4) {
    Header H_serialise(H_in);
    for (ssize_t axis = 3; axis != H_in.ndim(); ++axis) {
      H_serialise.size(axis - 3) = H_in.size(axis);
      H_serialise.stride(axis - 3) = axis - 2;
      H_serialise.spacing(axis - 3) = 1.0;
    }
    for (ssize_t axis = H_in.ndim() - 3; axis != 3; ++axis) {
      H_serialise.size(axis) = 1;
      H_serialise.stride(axis) = axis + 1;
      H_serialise.spacing(axis) = 1.0;
    }
    H_serialise.ndim() = std::max(size_t(3), H_in.ndim() - 3);
    H_serialise.datatype() = DataType::from<uint32_t>();
    H_serialise.datatype().set_byte_order_native();
    H_serialise.transform().setIdentity();
    serialise_image = Image<uint32_t>::scratch(                                     //
        H_serialise,                                                               //
        "Scratch image for serialising non-spatial indices into Casorati matrix"); //

    uint32_t output_index = 0;
    for (auto l = Loop(serialise_image)(serialise_image); l; ++l)
      serialise_image.value() = output_index++;
    serialise_image.reset();
    assert(output_index == H_out.size(3));
  }

  // Step 1: Phase demodulation
  // Only the smooth phase needs to be retained here;
  //   the actual demodulation of the data is performed on-the-fly,
  //   both for the Casorati fill and for the stabilised-domain mean computation.
  if (demodulation.mode != demodulation_t::NONE) {
    typename DemodulatorSelector<T>::type demodulator(image,                                        //
                                                      demodulation.axes,                            //
                                                      demodulation.mode == demodulation_t::LINEAR); //
    phase_image = demodulator();
  }

  // Step 2: Demeaning
  // Here only the structure is established (group indexing and storage allocation);
  //   the stabilised-domain mean values are (re)computed by compute_means(),
  //   which is sensitive to the current noise level map.
  Header H_mean(H_out);
  switch (demean) {
  case demean_type::NONE:
    break;
  case demean_type::VOLUME_GROUPS: {
    assert(serialise_image.valid());
    if (H_in.ndim() < 5)
      throw Exception("Cannot demean by volume groups if input image is <= 4D");
    index2group.resize(H_out.size(3));
    ssize_t group_index = 0;
    for (auto l_group = Loop(serialise_image, 1)(serialise_image); l_group; ++l_group, ++group_index) {
      for (auto l_volumes = Loop(serialise_image, 0, 1)(serialise_image); l_volumes; ++l_volumes)
        index2group[serialise_image.value()] = group_index;
    }
    serialise_image.reset();
    assert(group_index == num_volume_groups);
    H_mean.size(3) = num_volume_groups;
    vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for per-volume-group stabilised-domain mean");
  } break;
  case demean_type::SHELLS: {
    Eigen::Matrix<default_type, Eigen::Dynamic, Eigen::Dynamic> grad;
    try {
      grad = DWI::get_DW_scheme(H_mean);
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to obtain valid gradient table");
    }
    try {
      DWI::Shells shells(grad);
      index2shell.resize(image.size(3), -1);
      for (ssize_t shell_idx = 0; shell_idx != shells.count(); ++shell_idx) {
        for (auto v : shells[shell_idx].get_volumes())
          index2shell[v] = shell_idx;
      }
      assert(*std::min_element(index2shell.begin(), index2shell.end()) == 0);
      H_mean.size(3) = shells.count();
      DWI::stash_DW_scheme(H_mean, grad);
      vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for per-shell stabilised-domain mean");
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to establish b-value shell structure");
    }
  } break;
  case demean_type::ALL: {
    H_mean.ndim() = 3;
    DWI::clear_DW_scheme(H_mean);
    vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for stabilised-domain mean across all volumes");
  } break;
  }

  // Compute the initial stabilised-domain means.
  // With no noise level map available at construction (the iteration-1 bootstrap),
  //   the forward transform is the identity, so this reduces to the empirical means;
  //   when a noise level map is supplied (e.g. -vst / -noise_in),
  //   the means are formed in the stabilised domain (vst_plan.md section 3.2).
  compute_means(image);
}

// Serialise this voxel's volumes into "data", applying phase demodulation and,
//   where a noise level map is available, the forward variance-stabilising transform.
template <typename T>
void Precondition<T>::serialise_and_stabilise(Image<T> &input,
                                              Image<cfloat> &phase,
                                              Image<uint32_t> &serialise,
                                              const Transform &transform,
                                              Interp::Cubic<Image<float>> *vst,
                                              Eigen::Array<T, Eigen::Dynamic, 1> &data) const {
  // Load all volumes within this voxel into "data"
  if (H_in.ndim() == 4) {
    for (ssize_t v = 0; v != H_out.size(3); ++v) {
      input.index(3) = v;
      data[v] = input.value();
    }
  } else {
    for (auto l = Loop(H_in, 3)(input); l; ++l) {
      for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
        serialise.index(axis - 3) = input.index(axis);
      data[serialise.value()] = input.value();
    }
  }

  // Phase demodulation
  if (phase.valid()) {
    assign_pos_of(input, 0, 3).to(phase);
    if (H_in.ndim() == 4) {
      for (ssize_t v = 0; v != H_out.size(3); ++v) {
        phase.index(3) = v;
        data[v] = demodulate<T>(data[v], phase.value());
      }
    } else {
      for (auto l = Loop(H_in, 3)(phase); l; ++l) {
        for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
          serialise.index(axis - 3) = phase.index(axis);
        data[serialise.value()] = demodulate<T>(data[serialise.value()], phase.value());
      }
    }
  }

  // Forward variance-stabilising transform
  if (vst != nullptr) {
    vst->scanner(transform.voxel2scanner *                         //
                 Eigen::Vector3d({default_type(input.index(0)),    //
                                  default_type(input.index(1)),    //
                                  default_type(input.index(2))})); //
    const default_type sigma = vst->value();
    for (ssize_t v = 0; v != H_out.size(3); ++v)
      data[v] = vst_forward<T>(*noise_model, data[v], sigma);
  }
}

template <typename T> void Precondition<T>::compute_means(Image<T> input_arg) {
  if (!vst_mean_image.valid())
    return;

  const Transform transform(input_arg);
  Image<T> input(input_arg);
  Image<cfloat> phase(phase_image);
  Image<uint32_t> serialise(serialise_image);
  Image<T> mean(vst_mean_image);
  std::unique_ptr<Interp::Cubic<Image<float>>> vst;
  if (vst_noise_image.valid())
    vst.reset(new Interp::Cubic<Image<float>>(vst_noise_image));

  // Volume counts per group, required as divisors.
  // - across all volumes: the total number of volumes;
  // - per volume group: the number of volumes within each (equally-sized) group;
  // - per shell: the per-shell volume count.
  std::vector<ssize_t> group_counts;
  if (mean.ndim() > 3) {
    group_counts.assign(mean.size(3), 0);
    if (!index2shell.empty()) {
      for (const ssize_t shell_index : index2shell)
        ++group_counts[shell_index];
    } else {
      assert(!index2group.empty());
      std::fill(group_counts.begin(), group_counts.end(), H_in.size(3));
    }
  }

  Eigen::Array<T, Eigen::Dynamic, 1> data(H_out.size(3));
  // Per-group accumulator, reused across voxels to avoid repeated allocation.
  std::vector<T> sums(mean.ndim() > 3 ? mean.size(3) : 0);
  for (auto l_voxel = Loop("Computing stabilised-domain mean intensities", H_in, 0, 3)(input); l_voxel; ++l_voxel) {
    serialise_and_stabilise(input, phase, serialise, transform, vst.get(), data);
    assign_pos_of(input, 0, 3).to(mean);
    if (mean.ndim() == 3) {
      T sum(T(0));
      for (ssize_t v = 0; v != H_out.size(3); ++v)
        sum += data[v];
      mean.value() = sum / T(H_out.size(3));
    } else {
      std::fill(sums.begin(), sums.end(), T(0));
      for (ssize_t v = 0; v != H_out.size(3); ++v)
        sums[!index2shell.empty() ? index2shell[v] : index2group[v]] += data[v];
      for (ssize_t group = 0; group != mean.size(3); ++group) {
        mean.index(3) = group;
        mean.value() = group_counts[group] > 0 ? sums[group] / T(group_counts[group]) : T(0);
      }
    }
  }
}

template <typename T> void Precondition<T>::operator()(Image<T> input, Image<T> output, const bool inverse) const {

  // For thread-safety / const-ness
  const Transform transform(input);
  Image<uint32_t> serialise(serialise_image);
  Image<cfloat> phase(phase_image);
  Image<T> mean(vst_mean_image);
  std::unique_ptr<Interp::Cubic<Image<float>>> vst;
  if (vst_noise_image.valid())
    vst.reset(new Interp::Cubic<Image<float>>(vst_noise_image));

  Eigen::Array<T, Eigen::Dynamic, 1> data(H_out.size(3));
  if (inverse) {

    assert(dimensions_match(H_out, input));
    assert(dimensions_match(H_in, output));

    // The forward order is: phase demodulation -> variance-stabilising transform -> demeaning.
    // Reversal therefore proceeds in the opposite order:
    //   re-add the stabilised-domain mean -> invert the variance-stabilising transform -> re-modulate phase.
    for (auto l_voxel = Loop("Reversing data preconditioning", H_in, 0, 3)(input, output); l_voxel; ++l_voxel) {

      for (ssize_t v = 0; v != H_out.size(3); ++v) {
        input.index(3) = v;
        data[v] = input.value();
      }

      // Step 3 reversal: re-add the stabilised-domain mean
      if (mean.valid()) {
        assign_pos_of(input, 0, 3).to(mean);
        if (mean.ndim() == 3) {
          const T mean_value = mean.value();
          data += mean_value;
        } else if (!index2shell.empty()) {
          for (ssize_t v = 0; v != H_out.size(3); ++v) {
            mean.index(3) = index2shell[v];
            data[v] += T(mean.value());
          }
        } else if (!index2group.empty()) {
          for (ssize_t v = 0; v != H_out.size(3); ++v) {
            mean.index(3) = index2group[v];
            data[v] += T(mean.value());
          }
        } else {
          assert(false);
          data.fill(std::numeric_limits<T>::signaling_NaN());
        }
      }

      // Step 2 reversal: invert the variance-stabilising transform
      if (vst) {
        vst->scanner(transform.voxel2scanner *                         //
                     Eigen::Vector3d({default_type(input.index(0)),    //
                                      default_type(input.index(1)),    //
                                      default_type(input.index(2))})); //
        const default_type sigma = vst->value();
        for (ssize_t v = 0; v != H_out.size(3); ++v)
          data[v] = vst_inverse<T>(*noise_model, data[v], sigma);
      }

      // Step 1 reversal: re-modulate phase
      if (phase.valid()) {
        assign_pos_of(input, 0, 3).to(phase);
        if (serialise.valid()) {
          for (auto l = Loop(H_in, 3)(phase); l; ++l) {
            for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
              serialise.index(axis - 3) = phase.index(axis);
            data[serialise.value()] = modulate<T>(data[serialise.value()], phase.value());
          }
        } else {
          for (ssize_t v = 0; v != H_out.size(3); ++v) {
            phase.index(3) = v;
            data[v] = modulate<T>(data[v], phase.value());
          }
        }
      }

      // Write to output
      if (serialise.valid()) {
        for (auto l = Loop(H_in, 3)(output); l; ++l) {
          for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
            serialise.index(axis - 3) = output.index(axis);
          output.value() = data[serialise.value()];
        }
      } else {
        for (ssize_t v = 0; v != H_out.size(3); ++v) {
          output.index(3) = v;
          output.value() = data[v];
        }
      }
    }
    return;
  }

  assert(dimensions_match(H_in, input));
  assert(dimensions_match(H_out, output));

  // Applying forward preconditioning.
  // Order: phase demodulation -> variance-stabilising transform -> demeaning;
  //   demeaning is performed in the stabilised domain so that the Casorati entries
  //   are zero-mean per group (vst_plan.md section 3.1).
  for (auto l_voxel = Loop("Applying data preconditioning", H_in, 0, 3)(input, output); l_voxel; ++l_voxel) {

    // Steps 1 & 2: serialise, phase demodulation, variance-stabilising transform
    serialise_and_stabilise(input, phase, serialise, transform, vst.get(), data);

    // Step 3: demeaning (in the stabilised domain)
    if (mean.valid()) {
      assign_pos_of(input, 0, 3).to(mean);
      if (mean.ndim() == 3) {
        const T mean_value = mean.value();
        for (ssize_t v = 0; v != H_out.size(3); ++v)
          data[v] -= mean_value;
      } else if (!index2shell.empty()) {
        for (ssize_t v = 0; v != H_out.size(3); ++v) {
          mean.index(3) = index2shell[v];
          data[v] -= T(mean.value());
        }
      } else if (!index2group.empty()) {
        for (ssize_t v = 0; v != H_out.size(3); ++v) {
          mean.index(3) = index2group[v];
          data[v] -= T(mean.value());
        }
      } else {
        assert(false);
        data.fill(std::numeric_limits<T>::signaling_NaN());
      }
    }

    // Write to output
    for (ssize_t v = 0; v != H_out.size(3); ++v) {
      output.index(3) = v;
      output.value() = data[v];
    }
  }
}

template class Precondition<float>;
template class Precondition<double>;
template class Precondition<cfloat>;
template class Precondition<cdouble>;

} // namespace MR::Denoise
