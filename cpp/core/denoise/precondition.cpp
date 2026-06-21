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
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

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
  + Option("noise_dof",
           "the number of receive channels N combined by sum-of-squares reconstruction of magnitude data, "
           "such that the noise follows a non-central chi distribution with 2N degrees of freedom "
           "(default: 1, i.e. Rician; ignored for complex input data)")
    + Argument("count").type_integer(1)
  + Option("vst_method",
           "the variance-stabilising transform to apply prior to PCA; "
           "options are: " + join(NoiseModel::vst_methods, ",") + "; "
           "'none' applies no transform; "
           "'linear' divides by the noise level (the appropriate transform for Gaussian-distributed, "
           "e.g. complex or phase-demodulated, data); "
           "'foi', 'koay' and 'mom' construct a non-linear transform with bias-corrected inverse "
           "for magnitude data, differing only in the inverse / debias strategy "
           "(default: foi for magnitude data; complex data always use the linear transform)")
    + Argument("method").type_choice(NoiseModel::vst_methods);
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

std::shared_ptr<NoiseModel::Base> make_noise_model(const bool complex) {
  auto opt_dof = get_options("noise_dof");
  auto opt_method = get_options("vst_method");
  const NoiseModel::vst_method_t vst_method = opt_method.empty()                                  //
                                                  ? NoiseModel::vst_method_t::FOI                  //
                                                  : NoiseModel::vst_method_t(int(opt_method[0][0])); //
  // -vst_method none: no variance-stabilising transform (identity), for any data type.
  //   The noise distribution and -noise_dof are irrelevant, and the data reach PCA
  //   unmodified (save for any demeaning); see the single-pass fallback / warning in
  //   the calling commands.
  if (vst_method == NoiseModel::vst_method_t::NONE) {
    if (!opt_dof.empty())
      WARN("Option -noise_dof is ignored when -vst_method none is specified: "
           "no variance-stabilising transform is applied");
    return NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 1, vst_method);
  }
  if (complex) {
    if (!opt_dof.empty()) {
      WARN("Option -noise_dof is ignored for complex input data: "
           "the demodulated noise is Gaussian (one degree of freedom per channel)");
    }
    // The forward transform for Gaussian data is linear, so foi/koay/mom collapse to
    //   the linear transform; the default is harmless and no warning is emitted.
    return NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 1, vst_method);
  }
  // -vst_method linear: the simple linear transform u = m / sigma for magnitude data too.
  //   This only normalises the scale (the magnitude noise remains heteroscedastic near
  //   the floor) and performs no noise-floor debiasing; -noise_dof has no effect.
  if (vst_method == NoiseModel::vst_method_t::LINEAR) {
    if (!opt_dof.empty())
      WARN("Option -noise_dof is ignored when -vst_method linear is specified: "
           "the linear transform does not model the magnitude noise distribution");
    return NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 1, vst_method);
  }
  const ssize_t num_channels = opt_dof.empty() ? 1 : ssize_t(opt_dof[0][0]);
  const NoiseModel::distribution_t distribution = num_channels == 1                          //
                                                      ? NoiseModel::distribution_t::RICIAN   //
                                                      : NoiseModel::distribution_t::NONCENTRALCHI; //
  return NoiseModel::make(distribution, num_channels, vst_method);
}

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

// INVESTIGATION REQUIRED (ongoing need for explicit demeaning under BDCSVD):
//   Demeaning prior to PCA was originally introduced to condition the self-adjoint
//   (Gram-matrix) eigendecomposition at single precision, where a large common-mean /
//   low inter-volume-variance series (e.g. fMRI) made rank estimation unstable. The
//   default decomposition is now BDCSVD, which operates on the data matrix directly rather
//   than its Gram matrix and is far better conditioned against a dominant mean (the mean
//   appears as a single well-separated singular value). Whether explicit demeaning is still
//   required at all under the default BDCSVD decomposition has not been re-evaluated and
//   should be investigated; it may be reducible to optional / off-by-default.
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
      // Default reverted to per-b-value-shell demeaning when a gradient table is present.
      //   VERIFICATION FROM REAL DATA REQUIRED: the prior default of whole-dataset demeaning
      //   ('all') was adopted on the basis of subjective image assessment. Per-shell demeaning
      //   is expected to be the better-justified default, particularly now that noise-floor
      //   debiasing is anchored per-sample (debias_anchor == SAMPLE) and is therefore no longer
      //   sensitive to the demeaning grouping; this should be confirmed empirically.
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

// Exact-unbiased inverse of the forward variance-stabilising transform,
//   mapping a stabilised-domain (group) mean to the bias-free underlying level.
// Applied only to the per-group DC term so that the magnitude noise-floor bias
//   is not re-introduced into the denoised output.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_inverse_unbiased(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.inverse_unbiased(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_inverse_unbiased(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.inverse_unbiased(default_type(in.real()), sigma)),
           R(model.inverse_unbiased(default_type(in.imag()), sigma)));
}

// Local gain of the inverse transform at the operating point: the linear factor
//   by which a stabilised-domain residual is mapped back to the intensity scale.
// For complex (Gaussian) data the gain is identical across the real and imaginary
//   channels and independent of the operating point, so a single real factor suffices.
template <typename T>
typename std::enable_if<!is_complex<T>::value, default_type>::type
vst_jacobian(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return model.jacobian(default_type(in), sigma);
}
template <typename T>
typename std::enable_if<is_complex<T>::value, default_type>::type
vst_jacobian(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return model.jacobian(default_type(in.real()), sigma);
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

  // The stabilised-domain mean *values* are intentionally not computed here, only their
  //   storage allocated above. Every iteration selects its temporal subset and (re)computes
  //   the means against the current noise level map via set_temporal_subsample() +
  //   update_vst_parameters() before the preconditioner is applied (see the run() loops in
  //   dwidenoise2 / dwi2noise). Computing them at construction would therefore always be
  //   immediately overwritten before use; doing so previously manifested as two back-to-back
  //   "Computing stabilised-domain mean intensities" passes. Callers must invoke
  //   update_vst_parameters() to populate the means before applying the forward/inverse transform.
}

template <typename T>
void Precondition<T>::set_temporal_subsample(default_type fraction, Math::RNG &rng, ssize_t min_per_group) {
  temporal_subset.clear();
  if (fraction >= 1.0)
    return;
  assert(fraction > 0.0);
  assert(min_per_group >= 1);

  const ssize_t num_vols = H_out.size(3);

  // Partition volume indices into strata according to the active demeaning grouping,
  //   so the random subset preserves the relative proportions of each group.
  //   (-demean none / all: a single stratum spanning all volumes.)
  std::vector<std::vector<ssize_t>> strata;
  if (!index2shell.empty()) {
    strata.resize(*std::max_element(index2shell.begin(), index2shell.end()) + 1);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[index2shell[v]].push_back(v);
  } else if (!index2group.empty()) {
    strata.resize(num_volume_groups);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[index2group[v]].push_back(v);
  } else {
    strata.resize(1);
    strata[0].resize(num_vols);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[0][v] = v;
  }

  std::vector<ssize_t> per_stratum_counts(strata.size(), 0);
  bool floor_raised = false;
  for (size_t g = 0; g != strata.size(); ++g) {
    std::vector<ssize_t> &stratum = strata[g];
    const ssize_t s = stratum.size();
    if (s == 0)
      continue;
    const ssize_t k_target = ssize_t(std::lround(fraction * default_type(s)));
    ssize_t k = std::min(std::max(k_target, min_per_group), s);
    if (k > k_target)
      floor_raised = true;
    // Partial Fisher-Yates: draw k distinct indices from this stratum without replacement.
    for (ssize_t i = 0; i != k; ++i) {
      std::uniform_int_distribution<ssize_t> dist(i, s - 1);
      std::swap(stratum[i], stratum[dist(rng)]);
      temporal_subset.push_back(stratum[i]);
    }
    per_stratum_counts[g] = k;
  }

  std::sort(temporal_subset.begin(), temporal_subset.end());
  assert(ssize_t(temporal_subset.size()) <= num_vols);

  H_out_subset = H_out;
  H_out_subset.size(3) = ssize_t(temporal_subset.size());

  std::string counts_str;
  for (size_t g = 0; g != per_stratum_counts.size(); ++g)
    counts_str += (g ? "," : "") + str(per_stratum_counts[g]);
  INFO("Temporal sub-sampling: retaining " + str(temporal_subset.size()) + " of " + str(num_vols) +
       " volumes (fraction " + str(fraction) + ") across " + str(strata.size()) +
       " group(s); per-group counts: " + counts_str);
  if (floor_raised)
    WARN("Temporal sub-sampling floor of " + str(min_per_group) +
         " volume(s) per group raised the effective subset above the requested fraction "
         "for one or more groups");
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

  // When a temporal subset is active, the means are formed over only the subset volumes,
  //   so that the stabilised Casorati matrix (which holds only those volumes) is exactly
  //   zero-mean per group, keeping it rank-deficient by null_rank().
  const bool subsampled = !temporal_subset.empty();
  const ssize_t neff = subsampled ? ssize_t(temporal_subset.size()) : H_out.size(3);
  auto eff_index = [&](const ssize_t idx) -> ssize_t { return subsampled ? temporal_subset[idx] : idx; };

  // Volume counts per group, required as divisors: the number of (subset) volumes in each group.
  std::vector<ssize_t> group_counts;
  if (mean.ndim() > 3) {
    group_counts.assign(mean.size(3), 0);
    for (ssize_t idx = 0; idx != neff; ++idx) {
      const ssize_t v = eff_index(idx);
      ++group_counts[!index2shell.empty() ? index2shell[v] : index2group[v]];
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
      for (ssize_t idx = 0; idx != neff; ++idx)
        sum += data[eff_index(idx)];
      mean.value() = sum / T(neff);
    } else {
      std::fill(sums.begin(), sums.end(), T(0));
      for (ssize_t idx = 0; idx != neff; ++idx) {
        const ssize_t v = eff_index(idx);
        sums[!index2shell.empty() ? index2shell[v] : index2group[v]] += data[v];
      }
      for (ssize_t group = 0; group != mean.size(3); ++group) {
        mean.index(3) = group;
        mean.value() = group_counts[group] > 0 ? sums[group] / T(group_counts[group]) : T(0);
      }
    }
  }
}

template <typename T>
void Precondition<T>::operator()(Image<T> input,
                                 Image<T> output,
                                 const bool inverse,
                                 const bias_handling_t bias_handling,
                                 const debias_anchor_t debias_anchor) const {

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
    // Reversal therefore proceeds in the opposite order.
    // Where a noise level map is available, demeaning is treated purely as PCA conditioning:
    //   the SAMPLE anchor (default) undoes the stored demean offset first to form each
    //   volume's own denoised operating point (group mean + denoised residual), then applies
    //   the chosen non-linear inverse pointwise at that operating point. This makes debiasing
    //   independent of the demeaning grouping and restores the natural heteroscedasticity on
    //   the output scale. The GROUP_MEAN anchor instead reproduces the prior behaviour for the
    //   DEBIAS handling, linearising the unbiased inverse about the per-group mean.
    //   The PRESERVE handling is always inverted pointwise (a faithful algebraic reversal).
    // Where no noise level map is available (the demean-only bootstrap),
    //   reversal reduces to re-addition of the stored (empirical) mean.

    // Per-group quantities, sized once and reused across voxels to avoid repeated allocation:
    //   the stabilised-domain demean offset (SAMPLE), and the mapped DC value + local gain
    //   (GROUP_MEAN legacy debiasing).
    const ssize_t num_groups = mean.valid() ? (mean.ndim() == 3 ? 1 : mean.size(3)) : 0;
    std::vector<T> group_offset(std::max<ssize_t>(num_groups, ssize_t(1)), T(0));
    std::vector<T> group_dc(num_groups);
    std::vector<default_type> group_gain(num_groups);

    if (vst && !mean.valid()) {
      INFO("Reversing preconditioning without a demeaning reference: "
           "the non-linear inverse variance-stabilising transform is applied directly "
           "to each denoised sample");
    }

    // Describe in the progress message which corrections are being reversed: re-addition of
    //   the demeaning offset, the noise model governing the inverse variance-stabilising
    //   transform, and (only for a non-linear model, where the distinction has an effect)
    //   whether the noise-floor bias is being removed (DEBIAS) or preserved (PRESERVE).
    std::string reversal_message = "Reversing data preconditioning";
    {
      std::string detail;
      const auto append = [&detail](const std::string &item) { detail += (detail.empty() ? "" : ", ") + item; };
      if (mean.valid())
        append("reverting demeaning");
      if (vst) {
        append(noise_model->description() + " noise model");
        if (!noise_model->is_linear())
          append(bias_handling == bias_handling_t::DEBIAS ? "removing noise-floor bias"
                                                          : "preserving noise-floor bias");
      }
      if (!detail.empty())
        reversal_message += " (" + detail + ")";
    }

    for (auto l_voxel = Loop(reversal_message, H_in, 0, 3)(input, output); l_voxel; ++l_voxel) {

      for (ssize_t v = 0; v != H_out.size(3); ++v) {
        input.index(3) = v;
        data[v] = input.value();
      }

      if (vst) {
        // Interpolate the noise level (variance-stabilising-transform scale) at this voxel.
        vst->scanner(transform.voxel2scanner *                         //
                     Eigen::Vector3d({default_type(input.index(0)),    //
                                      default_type(input.index(1)),    //
                                      default_type(input.index(2))})); //
        const default_type sigma = vst->value();

        if (bias_handling == bias_handling_t::DEBIAS && debias_anchor == debias_anchor_t::GROUP_MEAN &&
            mean.valid()) {
          // GROUP_MEAN (legacy) debiasing: linearise the unbiased inverse about the per-group
          //   stabilised-domain mean (the demeaning offset), mapping the denoised residual
          //   through the local Jacobian. Debiasing accuracy then depends on the proximity of
          //   each volume to its group mean; see the SAMPLE default below and debias_anchor_t.
          assign_pos_of(input, 0, 3).to(mean);
          for (ssize_t group = 0; group != num_groups; ++group) {
            if (mean.ndim() > 3)
              mean.index(3) = group;
            const T op = mean.value();
            group_dc[group] = vst_inverse_unbiased<T>(*noise_model, op, sigma);
            group_gain[group] = vst_jacobian<T>(*noise_model, op, sigma);
          }
          for (ssize_t v = 0; v != H_out.size(3); ++v) {
            const ssize_t group = num_groups == 1                                //
                                      ? 0                                        //
                                      : (!index2shell.empty() ? index2shell[v]   //
                                                              : index2group[v]); //
            data[v] = group_dc[group] + T(group_gain[group]) * data[v];
          }
        } else {
          // SAMPLE (default) reversal, and all PRESERVE reversal:
          //   Undo the demeaning offset (it exists only to condition the PCA), forming each
          //   volume's own denoised operating point u_recon = (group mean) + (denoised
          //   residual), then apply the chosen non-linear inverse pointwise at u_recon. The
          //   inverse is evaluated per volume (not linearised about a group mean), so debiasing
          //   does not depend on the demeaning grouping and the natural signal-dependent
          //   heteroscedasticity is restored. With -demean none there is no stored offset, so
          //   u_recon == data[v] and the inverse is applied directly to each denoised sample.
          if (mean.valid()) {
            assign_pos_of(input, 0, 3).to(mean);
            for (ssize_t group = 0; group != num_groups; ++group) {
              if (mean.ndim() > 3)
                mean.index(3) = group;
              group_offset[group] = mean.value();
            }
          } else {
            group_offset[0] = T(0);
          }
          for (ssize_t v = 0; v != H_out.size(3); ++v) {
            const ssize_t group = num_groups <= 1                                //
                                      ? 0                                        //
                                      : (!index2shell.empty() ? index2shell[v]   //
                                                              : index2group[v]); //
            const T u_recon = data[v] + group_offset[group];
            // DEBIAS: bias-free underlying level; PRESERVE: conventional biased-magnitude level.
            //   (A Jensen second-moment correction for the post-denoising residual noise was
            //   considered but deliberately omitted: its stabilised-domain variance is not
            //   estimable from the data within this reversal, and a user-supplied value would
            //   contradict the data-driven premise of the tool.)
            data[v] = (bias_handling == bias_handling_t::DEBIAS)        //
                          ? vst_inverse_unbiased<T>(*noise_model, u_recon, sigma)  //
                          : vst_inverse<T>(*noise_model, u_recon, sigma);          //
          }
        }
      } else if (mean.valid()) {
        // No variance-stabilising transform (demean-only bootstrap):
        //   reversal is simply re-addition of the stored mean.
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
  // While a temporal subset is active the output holds only the m' selected volumes,
  //   so it conforms to header() (size(3) == m'), not to the full H_out.
  assert(dimensions_match(header(), output));

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

    // Write to output.
    // Without temporal sub-sampling, emit every volume; with sub-sampling, emit only the
    //   m' selected volumes (data was demeaned by the subset means, so the emitted columns
    //   are exactly zero-mean per group).
    if (temporal_subset.empty()) {
      for (ssize_t v = 0; v != H_out.size(3); ++v) {
        output.index(3) = v;
        output.value() = data[v];
      }
    } else {
      for (ssize_t k = 0; k != ssize_t(temporal_subset.size()); ++k) {
        output.index(3) = k;
        output.value() = data[temporal_subset[k]];
      }
    }
  }
}

template class Precondition<float>;
template class Precondition<double>;
template class Precondition<cfloat>;
template class Precondition<cdouble>;

} // namespace MR::Denoise
