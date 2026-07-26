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

#include "denoise/precondition/precondition.h"

#include <memory>
#include <string>
#include <vector>

#include "app.h"
#include "axes.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "metadata/bids.h"

using namespace MR::App;

namespace MR::Denoise::Precondition {

const char *const demodulation_description =
    "If the input data are of complex type, "
    "then a smooth phase is demodulated from each k-space prior to PCA, "
    "so that the residual phase is coherent across volumes and the Casorati matrix remains low-rank. "
    "In the absence of metadata indicating otherwise, "
    "it is inferred that the first two axes correspond to acquired slices, "
    "and different slices / volumes will be demodulated individually; "
    "this behaviour can be modified using the -demod_axes option. "
    "The method of phase estimation is selected with -demodulate. "
    "The default 'apc' performs noise-adaptive phase correction (Pizzolato et al. 2020): "
    "the background phase is re-estimated at every noise level estimation iteration "
    "from the empirical complex data via a noise-weighted total-variation smoothing, "
    "the strength of which is driven by the current noise level map "
    "(the first iteration, which has no noise map yet, "
    "instead self-calibrates from a data-derived global noise level with uniform spatial weighting). "
    "The alternative 'hann' uses a fixed full-extent Hann-window non-linear phase estimated only once. "
    "'linear' instead regresses a strictly linear phase term from each k-space, "
    "similarly to performed in Cordero-Grande et al. 2019. "
    "'none' disables phase demodulation.";

// clang-format off
OptionGroup precondition_options(const bool include_output)
{
  OptionGroup result ("Options for preconditioning data prior to PCA");
  result
  + Option("demodulate",
           "select form of phase demodulation; "
           "options are: " + Enum::join<demodulation_t>(",") + " "
           "(default: apc for complex data)")
    + Argument("mode").type_choice<demodulation_t>()
  + Option("demod_axes",
           "comma-separated list of axis indices along which FFT can be applied for phase demodulation")
    + Argument("axes").type_sequence_int()
  + Option("demean",
           "select method of demeaning prior to PCA; "
           "options are: " + Enum::join<demean_type>(",") + " "
           "(default: 'shells' if DWI gradient table available; 'volume_groups' if volume groups present; 'all' otherwise)")
    + Argument("mode").type_choice<demean_type>()
  + Option("noise_dof",
           "the number of receive channels N combined by sum-of-squares reconstruction of magnitude data, "
           "such that the noise follows a non-central chi distribution with 2N degrees of freedom "
           "(default: 1, i.e. Rician; ignored for complex input data)")
    + Argument("count").type_integer(1)
  + Option("vst_method",
           "the variance-stabilising transform to apply prior to PCA; "
           "options are: " + Enum::join<NoiseModel::vst_method_t>(",") + "; "
           "'none' applies no transform; "
           "'linear' divides by the noise level (the appropriate transform for Gaussian-distributed, "
           "e.g. complex or phase-demodulated, data); "
           "'foi', 'koay' and 'mom' construct a non-linear transform with bias-corrected inverse "
           "for magnitude data, differing only in the inverse / debias strategy "
           "(default: foi for magnitude data; complex data always use the linear transform)")
    + Argument("method").type_choice<NoiseModel::vst_method_t>();
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
  const NoiseModel::vst_method_t vst_method =
      get_option_choice("vst_method", NoiseModel::vst_method_t::FOI);
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
      result.mode = demodulation_t::APC;
    } else {
      if (!opt_axes.empty()) {
        throw Exception("Option -demod_axes cannot be specified: "
                        "no phase demodulation of magnitude data");
      }
    }
  } else {
    result.mode = Enum::from_name<demodulation_t>(std::string_view(opt_mode[0][0]));
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
  const demean_type user_selection = Enum::from_name<demean_type>(std::string_view(opt[0][0]));
  if (user_selection == demean_type::SHELLS && !shells_available)
    throw Exception("Cannot demean by b-value shells as shell structure could not be inferred");
  if (user_selection == demean_type::VOLUME_GROUPS && !volume_groups_available)
    throw Exception("Cannot demean by volume groups as image does not possess volume groups");
  return user_selection;
}

} // namespace MR::Denoise::Precondition
