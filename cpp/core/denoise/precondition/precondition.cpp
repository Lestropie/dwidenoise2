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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "app.h"
#include "axes.h"
#include "algo/loop.h"
#include "denoise/precondition/noise_model/coils.h"
#include "file/ofstream.h"
#include "image.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "metadata/bids.h"

using namespace MR::App;

namespace MR::Denoise::Precondition {

const char *const demodulation_description =
    "If the input data are of complex type, "
    "then a smooth phase is demodulated from each k-space prior to PCA, "
    "so that the residual phase is coherent across volumes and the Casorati matrix remains low-rank. "
    "Each slice of each volume is demodulated individually, "
    "the axis that separates slices being determined as described for option -slice_axis; "
    "alternatively, the axes along which demodulation is performed can be named directly using -demod_axes; "
    "the 'apc' estimator is two-dimensional and therefore requires exactly two axes, "
    "whereas 'linear' and 'hann' accept any number of spatial axes. "
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

const char *const slice_axis_description =
    "Both the estimation of the noise distribution from the background of magnitude data (-noise_dof_method)"
    " and the phase demodulation of complex data (-demodulate) operate on individual slices."
    " The axis that separates slices is taken from the \"SliceEncodingDirection\" header key-value"
    " where present, and otherwise is assumed to be the third spatial axis of the image as stored on disk;"
    " it can instead be nominated using -slice_axis."
    " As for every axis index on the MRtrix3 command line,"
    " the axis indices of -slice_axis and -demod_axes refer to the image axes"
    " as realigned on load to approximate RAS (as reported by mrinfo),"
    " which may differ from the axes of the image as stored on disk (as reported by mrinfo -ondisk);"
    " to specify axis indices as stored on disk, use -config RealignTransform false,"
    " under which every axis index refers to the image as stored on disk."
    " A single slice axis is used throughout, so -slice_axis and -demod_axes are mutually exclusive.";

const char *const noise_channels_description =
    "Options -noise_channels and -noise_dof configure the noise model for magnitude data"
    " (both are ignored for complex input data)."
    " -noise_channels takes the EFFECTIVE number of receive channels N"
    " combined by sum-of-squares reconstruction of magnitude data,"
    " such that the noise follows a non-central chi distribution with 2N degrees of freedom;"
    " -noise_dof takes those degrees of freedom directly."
    " The latter is exactly twice the former,"
    " as each complex channel contributes two real Gaussian components."
    " The two options are therefore two parameterisations of the same quantity and are mutually exclusive;"
    " supply whichever of the two the value to hand is expressed in"
    " (-noise_channels 4 and -noise_dof 8 are equivalent)."
    " The effective channel count is not the number of elements in the receive coil,"
    " nor the number of receiver channels:"
    " coil elements may be combined into fewer channels before reconstruction,"
    " and correlation of the noise between channels together with overlap of their sensitivity profiles"
    " means that the channels combined do not contribute independent degrees of freedom."
    " The effective count is consequently smaller than the element count,"
    " often by a large factor, and need not be a whole number"
    " (for example, 3.76 has been measured for a nominally 12-element head coil)."
    " It is the quantity that a fit of the noise distribution to the data estimates,"
    " as performed by -noise_dof_method (see below)."
    " If neither option is given, and no such fit was requested,"
    " the count is inferred from the image header metadata where they permit it:"
    " the coil combination mode is read from \"CoilCombinationMethod\""
    " (or from a declaration of SENSE1 combination in the image comments,"
    " as written by the CMRR multi-band sequences, which takes precedence over it),"
    " and, only where that mode is sum of squares,"
    " the effective channel count is resolved from \"Manufacturer\", \"ManufacturersModelName\","
    " \"ReceiveCoilName\" and \"ReceiveCoilActiveElements\""
    " against a table of receive coils built into this software."
    " That table holds counts measured experimentally for a coil where such a measurement exists,"
    " and otherwise, for coils whose active elements are reported as enumerated clusters of the receive array"
    " (\"HC1-7;NC1,2\"), the number of clusters active,"
    " on the hypothesis that each contributes one independent channel;"
    " a value obtained the latter way is applied but reported as a hypothesis, with a warning."
    " Any phase-sensitive combination (adaptive combine, SENSE1) yields N = 1."
    " Where no inference can be drawn, a warning is issued and N = 1 (Rician) is assumed."
    " A measured count is applied with a statement of its provenance;"
    " the Rician value of a phase-sensitive combination is applied without a warning."
    " An explicitly provided value is always used,"
    " with a warning if it disagrees with a value that the metadata establish rather than hypothesise,"
    " or by more than 5% with the value estimated from the data via -noise_dof_method."
    " Note also that these degrees of freedom, including those tabulated by -noise_dof_out,"
    " are unrelated to the -dof_output option of the dwidenoise2 command,"
    " which concerns the (Satterthwaite) degrees of freedom of the residual energy remaining after denoising;"
    " the two quantities merely share a name.";

const char *const noise_dof_estimation_description =
    "Option -noise_dof_method estimates the degrees of freedom of the magnitude noise distribution"
    " from the voxels of the input data that contain noise alone, following St-Jean et al. (2020)."
    " The effective channel count is seldom known for any particular receive hardware,"
    " and the header metadata from which it might otherwise be inferred are frequently absent,"
    " such that for the great majority of real data a Rician distribution would otherwise be assumed."
    " For a voxel carrying no signal,"
    " the squared magnitude divided by twice the noise variance follows a Gamma distribution"
    " whose shape parameter is precisely the effective channel count N (half the degrees of freedom);"
    " as the sum of such values across volumes is itself Gamma-distributed,"
    " each voxel can be tested individually against that model."
    " The voxels carrying no signal are identified automatically,"
    " by iteratively rejecting those whose summed statistic falls outside the central 95%"
    " of the Gamma distribution implied by the current estimates of noise level and degrees of freedom,"
    " and re-fitting both to the voxels retained."
    " This rejection discards not only voxels containing tissue signal,"
    " but equally those corrupted by ghosting, slice leakage or spikes,"
    " which is what makes the procedure automatic rather than dependent on a mask."
    " The rejection cannot by itself establish that what remains is background:"
    " the iteration converges on whatever compact cluster of voxels it begins at."
    " Once it has converged in a slice, the voxels selected are therefore tested"
    " for consistency with a single Gamma distribution,"
    " by fitting both the method-of-moments and the maximum-likelihood solutions to them:"
    " these agree for a Gamma sample, whereas for a mixture,"
    " such as tissue whose signal varies between voxels and between volumes,"
    " or a noise level that differs between volumes,"
    " the method-of-moments estimate is the lower of the two."
    " A slice whose ratio of the two estimates is below 0.95,"
    " by more than 3.5 times the standard error of that ratio, yields no estimate."
    " In each slice the iteration is initialised at the lowest mode"
    " of the distribution of voxel-wise mean squared magnitudes on a logarithmic scale"
    " that comprises enough voxels to yield an estimate,"
    " found in a histogram whose bin width follows from that slice's own voxels (by the Freedman-Diaconis rule)"
    " and disregarding local maxima attributable to sampling fluctuation of the histogram counts;"
    " background voxels form the darkest such cluster whatever the proportions of tissue and air in the slice."
    " Two alternative solutions of this same Gamma model are offered:"
    " 'ml' solves the maximum likelihood equations, and has the lower variance of the two;"
    " 'moments' solves the closed-form method-of-moments equations."
    " 'none', the default, disables the estimation,"
    " leaving the degrees of freedom to be taken from -noise_channels / -noise_dof"
    " or inferred from the image header metadata as described above."
    " Where an estimator is selected but the estimation does not succeed,"
    " the command terminates with an error rather than falling back on the header metadata or the Rician default,"
    " unless -noise_channels or -noise_dof is also given, in which case that value is used"
    " and the failure to corroborate it is reported with a warning."
    " The estimation is applicable only to magnitude input data,"
    " and is performed only where a non-linear variance-stabilising transform (-vst_method foi, koay or mom)"
    " is to make use of its result.";

const char *const noise_dof_selection_description =
    "Options -noise_dof_scope, -noise_dof_out and -noise_voxels_out"
    " each require that an estimator be selected via -noise_dof_method,"
    " as does -slice_axis for magnitude input data."
    " Estimation of the degrees of freedom from the data is performed independently for each slice,"
    " with slices separated along the axis determined as described for option -slice_axis."
    " A slice from which too few background voxels are identified,"
    " or whose voxels selected fail the test of consistency with a single Gamma distribution (see below),"
    " yields no estimate;"
    " so too does a slice that holds no data at all (every sample zero or non-finite, as in the slices"
    " that padding or resampling of the field of view leaves), which is distinguished from a failure."
    " A slice that contains no tissue is not of this kind: it is background throughout, and is fitted as such."
    " The procedure declines altogether if fewer than three slices yield an estimate,"
    " or if the majority of the field of view is identically zero in every volume."
    " Option -noise_dof_scope determines whether one value of the degrees of freedom is used"
    " for the whole image or one per slice;"
    " the estimation itself is performed per slice in either case."
    " By default (-noise_dof_scope global), the per-slice estimates are reduced to their median,"
    " which is robust to slices at the extremities of the field of view"
    " and to those contaminated by wrap-around or slice leakage;"
    " any slice holding data that yields no estimate is excluded from that median,"
    " and the number of such slices is reported with a warning."
    " With -noise_dof_scope slice, the per-slice estimates are instead retained,"
    " and a separate variance-stabilising transform is constructed for each slice;"
    " every slice that holds data must then yield an estimate,"
    " the command terminating with an error otherwise,"
    " and only slices that hold no data at all take the global median."
    " Because parallel imaging makes the effective channel count vary over space,"
    " slice-wise estimates may be closer to the truth;"
    " they are however individually noisier, being fitted from the background voxels of one slice only."
    " A large spread of the per-slice estimates is the signature of a spatially varying distribution,"
    " but equally of slices in which the voxels selected are not background (see below);"
    " a spread greater than sampling variation explains is reported with a warning."
    " Two diagnostic exports are available."
    " -noise_dof_out writes a tab-separated text file with one row per slice,"
    " giving the estimated degrees of freedom, the equivalent effective receive channel count,"
    " the estimated background noise level, the number of voxels identified as containing noise alone,"
    " the number of iterations of voxel rejection,"
    " whether a converged estimate was obtained,"
    " the ratio of the method-of-moments to the maximum-likelihood estimate of the degrees of freedom"
    " from the voxels selected (see below),"
    " and the reason where no estimate was obtained;"
    " comment lines preceding the table report the slice axis"
    " and the global median and interquartile range of the per-slice estimates."
    " -noise_voxels_out writes a mask of the voxels identified as containing noise alone,"
    " from which the noise distribution was estimated;"
    " inspection of this mask is the single most informative check on whether the procedure did what it should.";

const char *const noise_dof_limitations_description =
    "Three limitations are intrinsic to estimating the noise distribution from background voxels,"
    " rather than being defects of this implementation."
    " Firstly, it requires that background voxels exist:"
    " data with no air in the field of view, or whose reconstruction has already masked the background,"
    " carry no information for this estimator, and the procedure declines rather than returning a value;"
    " where there is only a little air, voxels within the head may be selected instead;"
    " most such slices fail the test of consistency with a single Gamma distribution and yield no estimate,"
    " but low-signal tissue under a Rician distribution can pass it,"
    " which typically yields too few degrees of freedom and per-slice estimates that disagree."
    " Data whose magnitudes have been smoothed or resampled spatially,"
    " as by distortion correction or registration,"
    " can still appear Gamma-distributed, but with inflated degrees of freedom;"
    " estimation should be performed on data prior to any such processing."
    " Secondly, under k-space parallel imaging (e.g. GRAPPA) the effective channel count depends on the local signal,"
    " such that a value estimated in the background is an upper bound on that which applies within the head."
    " Thirdly, simultaneous multi-slice acquisition aliases signal into what should be background,"
    " which inflates the estimate."
    " It nonetheless remains a large improvement on assuming a Rician distribution."
    " Separately, zero-valued samples receive particular treatment."
    " A voxel that is zero in every volume is disregarded."
    " Otherwise, a zero is taken to be a noise magnitude that was too small to be represented,"
    " whether rounded to zero by integer storage of data with a low noise level"
    " or clipped to zero by the reconstruction,"
    " and enters the fit as a sample known only to lie below a threshold:"
    " half the storage quantum (detected from the data) below the smallest positive magnitude of the slice."
    " Simply excluding such zeros instead would truncate the distribution and inflate the estimate,"
    " by around 20% for a half-Gaussian background stored as integers at a noise level of ten quanta."
    " Zeros that are not noise,"
    " such as regions filled with zeros by reconstruction or resampling,"
    " or negative values clipped from a real-valued reconstruction,"
    " would conversely deflate the estimate if treated in this way;"
    " in each slice, the number of zeros is therefore compared against that predicted"
    " from a fit to the positive magnitudes alone,"
    " and where there are far more than predicted, the zeros of that slice are instead excluded from the fit,"
    " with a warning issued where more than 1% of the samples of the voxels selected were excluded."
    " Note also that while the procedure necessarily estimates a background noise level"
    " alongside the degrees of freedom,"
    " that value is not used as the noise level in any subsequent computation:"
    " under parallel imaging a background estimate understates the noise level within the head,"
    " and the patch-wise Marchenko-Pastur estimate remains its sole source;"
    " the background noise level is exported via -noise_dof_out as a diagnostic only.";

// clang-format off
OptionGroup precondition_options(const option_scope_t scope)
{
  OptionGroup result ("Options for preconditioning data prior to PCA");
  result
  + Option("demodulate",
           "select form of phase demodulation; "
           "options are: " + Enum::join<demodulation_t>(",") + " "
           "(default: apc for complex data)")
    + Argument("mode").type_choice<demodulation_t>()
  + Option("slice_axis",
           "the image axis that separates slices, "
           "used for estimation of the noise distribution (magnitude data with -noise_dof_method) "
           "and for phase demodulation (complex data); "
           "indexed against the image axes as realigned on load to approximate RAS, as reported by mrinfo "
           "(default: from the \"SliceEncodingDirection\" header key-value where present, "
           "otherwise the third spatial axis of the image as stored on disk; "
           "see the Description section of the help page for details)")
    + Argument("index").type_integer(0, 2)
  + Option("demod_axes",
           "comma-separated list of axis indices along which FFT can be applied for phase demodulation, "
           "indexed against the image axes as realigned on load to approximate RAS, as reported by mrinfo "
           "(mutually exclusive with -slice_axis)")
    + Argument("axes").type_sequence_int()
  + Option("demean",
           "select method of demeaning prior to PCA; "
           "options are: " + Enum::join<demean_type>(",") + " "
           "(default: 'shells' if DWI gradient table available; 'volume_groups' if volume groups present; 'all' otherwise)")
    + Argument("mode").type_choice<demean_type>()
  + Option("noise_channels",
           "the EFFECTIVE number of receive channels combined by sum-of-squares reconstruction of magnitude data, "
           "which is not the number of coil elements "
           "(see the Description section of the help page for details)")
    + Argument("count").type_float(0.0)
  + Option("noise_dof",
           "the degrees of freedom of the non-central chi noise distribution of sum-of-squares magnitude data, "
           "equal to twice the effective receive channel count "
           "(see the Description section of the help page for details)")
    + Argument("count").type_float(0.0)
  + Option("vst_method",
           "the variance-stabilising transform to apply prior to PCA; "
           "options are: " + Enum::join<NoiseModel::vst_method_t>(",") + "; "
           "'none' applies no transform; "
           "'linear' divides by the noise level (the appropriate transform for Gaussian-distributed, "
           "e.g. complex or phase-demodulated, data); "
           "'foi', 'koay' and 'mom' construct a non-linear transform with bias-corrected inverse "
           "for magnitude data, differing only in the inverse / debias strategy "
           "(default: foi for magnitude data; "
           "for complex data the noise is Gaussian and the non-linear methods foi/koay/mom "
           "reduce to 'linear', though 'none' is still honoured)")
    + Argument("method").type_choice<NoiseModel::vst_method_t>()
  + Option("noise_dof_method",
           "estimate the degrees of freedom of the magnitude noise distribution "
           "from the voxels of the input data that contain noise alone; "
           "options are: " + Enum::join<NoiseModel::background_method_t>(",") + " "
           "(default: none; see the Description section of the help page for details)")
    + Argument("method").type_choice<NoiseModel::background_method_t>()
  + Option("noise_dof_scope",
           "whether one value of the degrees of freedom of the noise distribution is used for the whole image, "
           "or one per slice; "
           "options are: " + Enum::join<NoiseModel::background_scope_t>(",") + " "
           "(default: global; requires -noise_dof_method; "
           "see the Description section of the help page for details)")
    + Argument("scope").type_choice<NoiseModel::background_scope_t>()
  + Option("noise_dof_out",
           "export a text file tabulating the per-slice estimates of the noise distribution"
           + std::string(scope == option_scope_t::DENOISING ? ", which are unrelated to -dof_output" : "") + " "
           "(requires -noise_dof_method; see the Description section of the help page for details)")
    + Argument("file").type_file_out()
  + Option("noise_voxels_out",
           "export a mask of the voxels identified as containing noise alone, "
           "from which the noise distribution was estimated "
           "(requires -noise_dof_method; see the Description section of the help page for details)")
    + Argument("image").type_image_out();
  if (scope == option_scope_t::DENOISING) {
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

namespace {

// One phrase describing how a data estimate was obtained, for the messages that report it.
std::string describe_estimate(const NoiseModelOptions &options, const NoiseModel::BackgroundResult &background) {
  size_t estimated = 0;
  for (const auto &slice : background.slices)
    estimated += slice.estimated() ? 1 : 0;
  return std::string(options.background_method == NoiseModel::background_method_t::ML
                         ? "maximum likelihood fit"
                         : "method-of-moments fit") +
         " to " + str(background.num_selected_total) + " voxels over " + str(estimated) + " of " +
         str(background.slices.size()) + " slices, median across slices, interquartile range " +
         str(background.dof_iqr, 3) + " degrees of freedom";
}

// The per-slice table of -noise_dof_out. Slices that declined, or that hold no data, are written
//   with the voxel count they did achieve, with empty estimate fields and with the reason as the
//   comment, so that the file distinguishes "estimated as this" from "could not estimate" rather
//   than conflating the two into a blank row.
void write_dof_table(const std::string &path,
                     const NoiseModelOptions &options,
                     const NoiseModel::BackgroundResult &background) {
  File::OFStream out(path);
  out << "# " << App::command_history_string << "\n";
  out << "# Estimation of the degrees of freedom of the magnitude noise distribution from the voxels\n"
         "#   of each slice that contain noise alone (St-Jean et al. 2020).\n"
         "# The noise level sigma is reported as a diagnostic only: it is NOT the noise level used by\n"
         "#   the denoising, which comes from the per-patch Marchenko-Pastur estimate, and under\n"
         "#   parallel imaging a background estimate understates the value within the head.\n";
  out << "# slice axis: " << background.slice_axis << "\n";
  if (background)
    out << "# global (median across slices): dof " << background.dof << ", channels " << 0.5 * background.dof
        << ", interquartile range " << background.dof_iqr << "\n";
  else
    out << "# no global estimate: " << background.decline << "\n";
  out << "slice\tdof\tchannels\tsigma\tvoxels\titerations\tconverged\tgamma_ratio\tcomment\n";
  for (const auto &slice : background.slices) {
    out << slice.index << "\t";
    if (slice.estimated())
      out << slice.dof << "\t" << 0.5 * slice.dof << "\t" << slice.sigma << "\t";
    else
      out << "\t\t\t";
    out << slice.num_selected << "\t" << slice.num_iterations << "\t" << (slice.estimated() ? 1 : 0) << "\t";
    if (std::isfinite(slice.gamma_ratio))
      out << slice.gamma_ratio;
    out << "\t" << (slice.estimated() ? std::string() : slice.decline) << "\n";
  }
  INFO("Exported per-slice noise distribution estimates to \"" + path + "\"");
}

// The mask of -noise_voxels_out: which voxels the acceptance step chose. Written even where the
//   estimation declined, since a picture of what it selected is exactly what is wanted in order to
//   understand why.
template <class ImageType>
void write_voxel_mask(const std::string &path, ImageType &input, const NoiseModel::BackgroundResult &background) {
  Header H(input);
  H.ndim() = 3;
  H.datatype() = DataType::Bit;
  Image<bool> mask = Image<bool>::create(path, H);
  const std::array<ssize_t, 3> size{{H.size(0), H.size(1), H.size(2)}};
  const bool populated =
      background.selected.size() == size_t(size[0]) * size_t(size[1]) * size_t(size[2]);
  for (auto l = Loop(mask, 0, 3)(mask); l; ++l) {
    mask.value() =
        populated &&
        background.selected[NoiseModel::voxel_index(size, mask.index(0), mask.index(1), mask.index(2))] != 0;
  }
  if (!populated)
    WARN("Option -noise_voxels_out was requested, but no voxel was identified as containing noise alone;"
         " the exported mask is empty");
}

// Every axis index named on the command line, as throughout MRtrix3, refers to the image axes as
//   realigned on load to approximate RAS (as reported by mrinfo, not mrinfo -ondisk).
//   Header::Realignment::applied_transform() maps a direction expressed against the axes as stored on
//   disk to one expressed against the realigned axes; it is a signed permutation, so its transpose
//   maps back.
size_t nonzero_axis(const Metadata::BIDS::axis_vector_type &dir) {
  for (size_t axis = 0; axis != 3; ++axis) {
    if (dir[axis])
      return axis;
  }
  return 3;
}
Metadata::BIDS::axis_vector_type unit_vector(const size_t axis) {
  Metadata::BIDS::axis_vector_type result(Metadata::BIDS::axis_vector_type::Zero());
  result[axis] = 1;
  return result;
}
Metadata::BIDS::axis_vector_type to_ondisk(const Header &H, const Metadata::BIDS::axis_vector_type &realigned) {
  return Metadata::BIDS::axis_vector_type(H.realignment().applied_transform().transpose() * realigned);
}
size_t ondisk_axis(const Header &H, const size_t realigned_axis) {
  return nonzero_axis(to_ondisk(H, unit_vector(realigned_axis)));
}
size_t realigned_axis(const Header &H, const size_t ondisk_axis) {
  return nonzero_axis(Metadata::BIDS::axis_vector_type(H.realignment().applied_transform() * unit_vector(ondisk_axis)));
}
// Where realignment moved the axis, a note of which axis it is on disk; otherwise empty.
std::string ondisk_note(const Header &H, const size_t axis) {
  if (!H.realignment().applied())
    return std::string();
  const size_t ondisk = ondisk_axis(H, axis);
  return ondisk == axis ? std::string() : " (axis " + str(ondisk) + " of the image as stored on disk)";
}
std::vector<size_t> within_slice_axes(const size_t slice_axis) {
  std::vector<size_t> result;
  for (size_t axis = 0; axis != 3; ++axis) {
    if (axis != slice_axis)
      result.push_back(axis);
  }
  return result;
}

// The slice axis as it would be inferred from the header alone, without reference to -slice_axis,
//   together with the message that reports that inference. "purpose" names what the axis is used for.
struct SliceAxisInference {
  enum class level_t { INFO, CONSOLE, WARN };
  size_t axis{2};
  bool from_metadata{false};
  level_t level{level_t::INFO};
  std::string message;
  void report() const {
    switch (level) {
    case level_t::INFO:
      INFO(message);
      break;
    case level_t::CONSOLE:
      CONSOLE(message);
      break;
    case level_t::WARN:
      WARN(message);
      break;
    }
  }
};
SliceAxisInference infer_slice_axis(const Header &H, const std::string &purpose) {
  SliceAxisInference result;
  const Header::Realignment &realignment = H.realignment();
  auto slice_encoding_it = H.keyval().find("SliceEncodingDirection");
  if (slice_encoding_it != H.keyval().end()) {
    const Metadata::BIDS::axis_vector_type dir = Metadata::BIDS::axisid2vector(slice_encoding_it->second);
    const size_t axis = nonzero_axis(dir);
    if (axis != 3) {
      result.axis = axis;
      result.from_metadata = true;
      result.message = "For header SliceEncodingDirection=\"" + slice_encoding_it->second + "\", chose axis " +
                       str(axis) + " to separate slices for " + purpose;
      if (realignment.applied()) {
        const Metadata::BIDS::axis_vector_type ondisk_dir = to_ondisk(H, dir);
        if (ondisk_dir != dir) {
          const auto orig_it = realignment.orig_keyval().find("SliceEncodingDirection");
          result.message += " (image axes were realigned on load to approximate RAS; as stored on disk, ";
          result.message += orig_it == realignment.orig_keyval().end()
                                ? "the image has no SliceEncodingDirection field, \"" + slice_encoding_it->second +
                                      "\" having been inferred on load from field SliceTiming"
                                : "SliceEncodingDirection=\"" + orig_it->second + "\"";
          result.message +=
              ", corresponding to axis " + str(nonzero_axis(ondisk_dir)) + " of the image as stored on disk)";
        }
      }
      return result;
    }
  }
  // Slices are conventionally stored along the third axis of the image on disk, which need not be
  //   the third axis of the image as realigned on load to approximate RAS.
  result.axis = realigned_axis(H, 2);
  if (result.axis == 2) {
    result.level = SliceAxisInference::level_t::CONSOLE;
    result.message = "No header information on slice encoding; "
                     "assuming the third spatial axis separates slices for " +
                     purpose;
  } else {
    result.level = SliceAxisInference::level_t::WARN;
    result.message = "No header information on slice encoding; "
                     "assuming the third spatial axis of the image as stored on disk separates slices for " +
                     purpose + ", which following realignment of the image axes on load to approximate RAS is axis " +
                     str(result.axis) + "; use -slice_axis to select a different axis if this is incorrect";
  }
  return result;
}

// A user who reads an axis index from the image as stored on disk (from another software package,
//   or from the sidecar JSON) and supplies it on the command line, where it is interpreted against
//   the realigned axes, names the wrong axis. Where realignment has moved axes, the signature of
//   that mistake is a nominated slice axis that differs from the one inferred from the header, but
//   that is the on-disk index of the inferred one. "remedy" is the option as it would then be
//   correctly specified.
void check_axis_frame(const Header &H,
                      const std::string &nomination,
                      const size_t nominated_slice_axis,
                      const SliceAxisInference &inferred,
                      const std::string &remedy) {
  if (!H.realignment().applied() || nominated_slice_axis == inferred.axis ||
      ondisk_axis(H, inferred.axis) != nominated_slice_axis)
    return;
  WARN(nomination + " is interpreted against the image axes as realigned on load to approximate RAS," +
       " under which it implies that axis " + str(nominated_slice_axis) + ondisk_note(H, nominated_slice_axis) +
       " separates slices; but " +
       (inferred.from_metadata ? "header field SliceEncodingDirection indicates"
                               : "in the absence of slice encoding metadata, it would be inferred to be") +
       " axis " + str(inferred.axis) + ", which is axis " + str(nominated_slice_axis) +
       " of the image as stored on disk. If on-disk axis " + str(nominated_slice_axis) + " was intended, specify " +
       remedy +
       " instead, or use -config RealignTransform false, under which every axis index refers to the image as"
       " stored on disk");
}

// The axis that separates slices: from -slice_axis where specified, otherwise inferred from the header.
size_t select_slice_axis(const Header &H, const std::string &purpose) {
  const SliceAxisInference inferred = infer_slice_axis(H, purpose);
  auto opt = get_options("slice_axis");
  if (opt.empty()) {
    inferred.report();
    return inferred.axis;
  }
  const size_t axis = size_t(int(opt[0][0]));
  check_axis_frame(H, "-slice_axis " + str(axis), axis, inferred, "-slice_axis " + str(inferred.axis));
  INFO("Using axis " + str(axis) + ondisk_note(H, axis) + ", as nominated by -slice_axis, to separate slices for " +
       purpose);
  return axis;
}

} // namespace

NoiseModelOptions noise_model_options(const bool complex, const Header &H) {
  NoiseModelOptions result;
  result.complex = complex;

  // -noise_channels and -noise_dof are two parameterisations of one quantity, related by
  //   dof = 2 * channels. They are resolved here, once, into the degrees of freedom that the model
  //   is built from; everything downstream sees only that. Supplying both is not a redundancy to be
  //   tolerated -- the two arguments could contradict one another, and there is no basis on which to
  //   prefer either -- so it is refused.
  auto opt_channels = get_options("noise_channels");
  auto opt_dof = get_options("noise_dof");
  if (!opt_channels.empty() && !opt_dof.empty())
    throw Exception("Options -noise_channels and -noise_dof cannot be used together: "
                    "they express the same quantity in the two conventions related by "
                    "dof = 2 x channels (-noise_channels 4 is -noise_dof 8). Supply whichever one "
                    "matches the value you have");
  // The option through which the user expressed the count, for messages that quote it back.
  result.user_option = opt_channels.empty() ? std::string("noise_dof") : std::string("noise_channels");
  // Reduced to the common currency -- degrees of freedom, in which every internal computation is
  //   expressed -- at the point of parsing, so that the doubling of a channel-count request happens
  //   exactly once and in one place, and nothing downstream has to know which option was used.
  result.user_dof = !opt_channels.empty() ? std::optional<default_type>(2.0 * default_type(opt_channels[0][0]))
                    : !opt_dof.empty()    ? std::optional<default_type>(default_type(opt_dof[0][0]))
                                          : std::nullopt;
  const bool user_specified = result.user_dof.has_value();

  result.vst_method = get_option_choice("vst_method", NoiseModel::vst_method_t::FOI);
  // Options that exist only to parameterise the non-linear magnitude noise models have no
  //   meaning for complex data, whose demodulated noise is Gaussian: the variance-stabilising
  //   transform is then linear and carries no distribution bias. Requesting one is an incoherent
  //   combination rather than a redundant one -- the user is asking for a magnitude noise model
  //   that cannot be applied -- so it is refused rather than silently overridden.
  //   -vst_method {none, linear} are excluded: both are genuinely honoured for complex data.
  if (complex) {
    if (user_specified)
      throw Exception("Option -" + result.user_option +
                      " cannot be used with complex input data: "
                      "it parameterises the magnitude (Rician / non-central chi) noise model, whereas the "
                      "demodulated complex noise is Gaussian and carries two degrees of freedom per sample "
                      "however many channels were combined to produce it");
    if (!get_options("vst_method").empty() &&                //
        result.vst_method != NoiseModel::vst_method_t::LINEAR && //
        result.vst_method != NoiseModel::vst_method_t::NONE)     //
      throw Exception("Option -vst_method " + Enum::lowercase_name(result.vst_method) +
                      " cannot be used with complex input data: "
                      "it is a non-linear transform for magnitude data, whereas the noise here is "
                      "Gaussian-distributed and its variance-stabilising transform is linear. "
                      "Specify -vst_method linear, or -vst_method none, or omit the option");
  }

  // Estimation of the degrees of freedom from the data themselves.
  const bool method_named = !get_options("noise_dof_method").empty();
  NoiseModel::background_method_t method =
      get_option_choice("noise_dof_method", NoiseModel::background_method_t::NONE);
  auto opt_scope = get_options("noise_dof_scope");
  auto opt_axis = get_options("slice_axis");
  const bool exports_requested = !get_options("noise_dof_out").empty() || !get_options("noise_voxels_out").empty();
  result.background_scope = get_option_choice("noise_dof_scope", NoiseModel::background_scope_t::GLOBAL);

  // Everything that governs or reports the estimation presupposes that an estimation is happening.
  //   Where none was asked for, naming one of them is an incoherent request rather than a redundant
  //   one, and is refused rather than silently ignored.
  if (method == NoiseModel::background_method_t::NONE) {
    std::vector<std::string> dependent;
    if (!opt_scope.empty())
      dependent.push_back("-noise_dof_scope");
    // For complex input, -slice_axis governs the phase demodulation instead, and is validated there.
    if (!opt_axis.empty() && !complex)
      dependent.push_back("-slice_axis");
    if (!get_options("noise_dof_out").empty())
      dependent.push_back("-noise_dof_out");
    if (!get_options("noise_voxels_out").empty())
      dependent.push_back("-noise_voxels_out");
    if (!dependent.empty())
      throw Exception("Option" + std::string(dependent.size() > 1 ? "s " : " ") + join(dependent, ", ") +
                      (dependent.size() > 1 ? " require" : " requires") +
                      " -noise_dof_method to select an estimator: " +
                      (method_named ? "-noise_dof_method none was specified, so there is no estimation for "
                                      "them to govern or to report"
                                    : "no estimation of the noise distribution from the data is performed "
                                      "unless one is requested"));
  }
  // Complex data have two degrees of freedom per sample by construction; there is nothing to
  //   estimate, and an estimate made from the magnitude of complex data would parameterise a model
  //   that is not applied to them.
  if (complex && method != NoiseModel::background_method_t::NONE)
    throw Exception("Option -noise_dof_method " + Enum::lowercase_name(method) +
                    " cannot be used with complex input data: "
                    "the degrees of freedom it estimates parameterise the magnitude (Rician / non-central "
                    "chi) noise model, whereas the demodulated complex noise is Gaussian and carries two "
                    "degrees of freedom per sample however many channels were combined to produce it");
  // -vst_method none / linear do not use the degrees of freedom at all, so the pass that estimates
  //   them would be a read of the whole image to produce a number that nothing consumes. Reported
  //   as ignored, exactly as -noise_channels is on those same paths.
  if (method != NoiseModel::background_method_t::NONE &&
      (result.vst_method == NoiseModel::vst_method_t::NONE ||
       result.vst_method == NoiseModel::vst_method_t::LINEAR)) {
    WARN("Option -noise_dof_method is ignored when -vst_method " + Enum::lowercase_name(result.vst_method) +
         " is specified: that transform does not model the magnitude noise distribution, so the estimated "
         "degrees of freedom would not be used" +
         (exports_requested ? ", and the requested diagnostic exports are not produced" : ""));
    method = NoiseModel::background_method_t::NONE;
  }
  // An explicitly named estimator is run even where the user has imposed a value, so that the two
  //   can be compared; but where the estimator is merely the default, running it would cost a read
  //   of the data to produce a number that the user's own value supersedes.
  if (method != NoiseModel::background_method_t::NONE && user_specified && !method_named) {
    INFO("Not estimating the noise distribution from the data: " + result.user_option +
         " was specified, and its value is what would be used. Name -noise_dof_method explicitly to "
         "estimate anyway and have the two compared");
    method = NoiseModel::background_method_t::NONE;
  }
  // A scalar imposed on the command line cannot vary by slice, so there is nothing for a slice-wise
  //   scope to be slice-wise about.
  if (!opt_scope.empty() && result.background_scope == NoiseModel::background_scope_t::SLICE && user_specified)
    throw Exception("Option -noise_dof_scope slice cannot be used together with -" + result.user_option +
                    ": a value imposed on the command line is one number for the whole image, and it is "
                    "what would be used at every slice");
  result.background_method = method;

  if (result.background_method != NoiseModel::background_method_t::NONE) {
    result.background_slice_axis = select_slice_axis(H, "noise distribution estimation");
  }
  return result;
}

template <class ImageType>
NoiseModel::BackgroundResult estimate_background_dof(ImageType &input, const NoiseModelOptions &options) {
  NoiseModel::BackgroundResult result;
  if (options.background_method == NoiseModel::background_method_t::NONE)
    return result;

  // One pass over the 4D data, forming four sufficient statistics per spatial voxel; everything
  //   thereafter is on a 3D array (see noise_model/background.h).
  const NoiseModel::Accumulation accumulation =
      NoiseModel::accumulate_background(input, options.background_slice_axis);
  result = NoiseModel::estimate_background(accumulation, options.background_method);

  // Braced throughout: INFO() and WARN() each expand to an unbraced if, to which an else following
  //   them would otherwise bind.
  for (const auto &slice : result.slices) {
    if (slice.estimated()) {
      INFO("Noise distribution at slice " + str(slice.index) + ": " + str(slice.dof, 4) + " degrees of freedom, " +
           "sigma " + str(slice.sigma, 4) + ", from " + str(slice.num_selected) + " voxels in " +
           str(slice.num_iterations) + " iterations, Gamma consistency ratio " + str(slice.gamma_ratio, 3) +
           (slice.zeros_excluded ? std::string("; zero-valued samples excluded")
            : slice.num_zero > 0 ? "; " + str(slice.num_zero) + " zero-valued samples treated as censored below " +
                                       str(slice.censoring_threshold, 4)
                                 : std::string()));
    } else {
      INFO("No noise distribution estimate at slice " + str(slice.index) + ": " + slice.decline);
    }
  }
  for (const std::string &caveat : NoiseModel::estimate_caveats(result))
    WARN(caveat);

  // Estimation from the data is opt-in, and while it awaits validation on a broader range of real
  //   data every imperfection of it is made prominent. Where it was requested and did not succeed,
  //   nothing falls back silently to the header metadata or to the Rician default: that is a choice
  //   for the user to make, by omitting -noise_dof_method. The exception is a value imposed via
  //   -noise_channels / -noise_dof, which is what would be used whatever the estimate, the estimate
  //   then serving only to corroborate it.
  const bool user_specified = options.user_dof.has_value();
  const bool slicewise = options.background_scope == NoiseModel::background_scope_t::SLICE;
  const size_t num_slices = result.slices.size();
  const size_t num_failed = result.count(NoiseModel::slice_outcome_t::FAILED);
  const size_t num_no_signal = result.count(NoiseModel::slice_outcome_t::NO_SIGNAL);
  const std::string no_signal_phrase = num_no_signal > 0
                                           ? str(num_no_signal) + " of " + str(num_slices) +
                                                 " slices hold no data at all (every sample zero or non-finite)"
                                           : std::string();
  std::optional<Exception> failure;
  if (!result) {
    if (user_specified) {
      WARN("Unable to estimate the noise distribution from the voxels of these data that contain noise alone: " +
           result.decline + ". The value supplied via -" + options.user_option +
           " is used, but cannot be corroborated against the data");
    } else {
      failure = Exception("Unable to estimate the noise distribution from the voxels of these data that contain "
                          "noise alone, as requested via -noise_dof_method: " +
                          result.decline +
                          ". Per-slice outcomes are reported at -info, and -noise_dof_out / -noise_voxels_out"
                          " export the per-slice estimates and the voxels selected. To proceed without"
                          " estimation, supply -noise_channels or -noise_dof, or omit -noise_dof_method to have"
                          " the degrees of freedom inferred from the image header metadata where they permit"
                          " it and the Rician default assumed otherwise");
    }
  } else if (slicewise && num_failed > 0) {
    // A slice-wise transform cannot be constructed for a slice whose distribution is not known, and
    //   substituting the median there would apply a value that the slice-wise scope was requested in
    //   order to avoid.
    Exception e("Option -noise_dof_scope slice requires an estimate of the noise distribution from every slice"
                " that holds data, but " +
                str(num_failed) + " of " + str(num_slices) + " slices yielded none" +
                (num_no_signal > 0 ? " (a further " + no_signal_phrase + ")" : std::string()) +
                ". -noise_dof_scope global would apply the median over the " +
                str(result.count(NoiseModel::slice_outcome_t::ESTIMATED)) +
                " slices that did; -noise_dof_out and -noise_voxels_out export the per-slice estimates"
                " and the voxels selected. The slices that yielded no estimate:");
    for (const auto &slice : result.slices) {
      if (slice.outcome == NoiseModel::slice_outcome_t::FAILED)
        e.push_back("slice " + str(slice.index) + ": " + slice.decline);
    }
    failure = std::move(e);
  } else if (num_failed > 0) {
    WARN("Estimation of the noise distribution from the data yielded no estimate in " + str(num_failed) + " of " +
         str(num_slices) + " slices" + (num_no_signal > 0 ? " (a further " + no_signal_phrase + ")" : std::string()) +
         "; proceeding with the median over the " + str(result.count(NoiseModel::slice_outcome_t::ESTIMATED)) +
         " slices that yielded one (per-slice reasons are reported at -info)");
  } else if (num_no_signal > 0) {
    INFO(no_signal_phrase + " and yield no estimate of the noise distribution" +
         (slicewise ? "; their transform takes the global median, which has no effect on voxels that are "
                      "constant across volumes"
                    : "; the median is taken over the remaining slices"));
  }

  // .as_text() rather than an implicit conversion: App::ParsedArgument asserts in debug builds when
  //   a filesystem-typed argument is converted to std::string.
  auto opt = get_options("noise_dof_out");
  if (!opt.empty())
    write_dof_table(opt[0][0].as_text(), options, result);
  opt = get_options("noise_voxels_out");
  if (!opt.empty())
    write_voxel_mask(opt[0][0].as_text(), input, result);
  // Thrown only once the diagnostic exports are written, as they are what is needed to understand
  //   the failure.
  if (failure)
    throw *failure;
  return result;
}

template NoiseModel::BackgroundResult estimate_background_dof<Image<float>>(Image<float> &, const NoiseModelOptions &);
template NoiseModel::BackgroundResult estimate_background_dof<Image<double>>(Image<double> &,
                                                                             const NoiseModelOptions &);

NoiseModel::Set make_noise_model(const NoiseModelOptions &options,
                                 const Header &H,
                                 const NoiseModel::BackgroundResult &background) {
  const bool user_specified = options.user_dof.has_value();
  // -vst_method none: no variance-stabilising transform (identity), for any data type.
  //   The noise distribution and the channel count are irrelevant, and the data reach PCA
  //   unmodified (save for any demeaning); see the single-pass fallback / warning in
  //   the calling commands.
  if (options.vst_method == NoiseModel::vst_method_t::NONE) {
    if (user_specified)
      WARN("Option -" + options.user_option +
           " is ignored when -vst_method none is specified: "
           "no variance-stabilising transform is applied");
    return NoiseModel::Set(NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 2.0, options.vst_method));
  }
  // Any magnitude-specific request has already been refused above; the default (foi) simply
  //   collapses to the linear transform for Gaussian noise.
  if (options.complex)
    return NoiseModel::Set(NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 2.0, options.vst_method));
  // -vst_method linear: the simple linear transform u = m / sigma for magnitude data too.
  //   This only normalises the scale (the magnitude noise remains heteroscedastic near
  //   the floor) and performs no noise-floor debiasing; the channel count has no effect.
  if (options.vst_method == NoiseModel::vst_method_t::LINEAR) {
    if (user_specified)
      WARN("Option -" + options.user_option +
           " is ignored when -vst_method linear is specified: "
           "the linear transform does not model the magnitude noise distribution");
    return NoiseModel::Set(NoiseModel::make(NoiseModel::distribution_t::GAUSSIAN, 2.0, options.vst_method));
  }
  // This is the only code path on which the degrees of freedom have any effect: a non-linear
  //   transform for magnitude data, whose forward transform and bias-corrected inverse are both
  //   built from the noise distribution. It is therefore also the only path on which the header is
  //   interrogated for that count, so that no claim is made about a quantity that the requested
  //   transform would then ignore. Where the user named a count it is used regardless, but both the
  //   data estimate and the inference are still reported so that any disagreement is surfaced.
  const std::optional<default_type> data_dof =
      background ? std::optional<default_type>(background.dof) : std::nullopt;
  const std::string data_evidence = background ? describe_estimate(options, background) : std::string();
  const default_type dof =
      NoiseModel::resolve_noise_dof(H,
                                    options.user_dof,
                                    options.user_option,
                                    data_dof,
                                    data_evidence,
                                    options.background_method != NoiseModel::background_method_t::NONE);

  // Slice-wise scope, which only a successful estimate that the user has not overridden can supply;
  //   estimate_background_dof() has already refused one in which any slice holding data failed, so
  //   the global median stands in only for slices that hold no data at all.
  if (options.background_scope == NoiseModel::background_scope_t::SLICE && background && !user_specified) {
    std::vector<default_type> dof_per_slice(background.slices.size());
    size_t num_fallback = 0;
    for (size_t slice = 0; slice != background.slices.size(); ++slice) {
      assert(background.slices[slice].outcome != NoiseModel::slice_outcome_t::FAILED);
      dof_per_slice[slice] = background.dof_at_slice(ssize_t(slice));
      num_fallback += background.slices[slice].estimated() ? 0 : 1;
    }
    NoiseModel::Set models =
        NoiseModel::make_slicewise(dof_per_slice, background.slice_axis, options.vst_method);
    INFO("Built " + str(models.num_distinct_models()) + " distinct variance-stabilising transforms for " +
         str(dof_per_slice.size()) + " slices" +
         (num_fallback > 0 ? "; " + str(num_fallback) + " slices hold no data and take the global median of " +
                                 str(dof, 4) + " degrees of freedom"
                           : ""));
    return models;
  }

  // The Rician label is reserved for exactly two degrees of freedom (one effective channel); any
  //   other value, integral or not, is described as what it is.
  const NoiseModel::distribution_t distribution = dof == default_type(2)                           //
                                                      ? NoiseModel::distribution_t::RICIAN         //
                                                      : NoiseModel::distribution_t::NONCENTRALCHI; //
  return NoiseModel::Set(NoiseModel::make(distribution, dof, options.vst_method));
}

Demodulation select_demodulation(const Header &H) {
  const bool complex = H.datatype().is_complex();
  auto opt_mode = get_options("demodulate");
  auto opt_axes = get_options("demod_axes");
  auto opt_slice_axis = get_options("slice_axis");
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
  // -demod_axes names the axes along which the phase demodulation is performed. With no
  //   demodulation there are no such axes for it to govern, so supplying it is an incoherent
  //   request rather than a redundant one, and is rejected for both magnitude and complex input.
  //   (Magnitude input with -demod_axes but no -demodulate is caught above, with a message
  //   specific to that case.)
  if (result.mode == demodulation_t::NONE && !opt_axes.empty())
    throw Exception("Option -demod_axes cannot be specified together with -demodulate none: "
                    "there are no demodulation axes when no phase demodulation is performed");
  // For magnitude input, -slice_axis governs the estimation of the noise distribution, and is
  //   validated alongside it. Complex input has no such estimation, so without demodulation
  //   -slice_axis has nothing to govern.
  if (!complex)
    return result;
  if (result.mode == demodulation_t::NONE && !opt_slice_axis.empty())
    throw Exception("Option -slice_axis cannot be specified together with -demodulate none for complex input data: "
                    "the slice axis is used only by the phase demodulation, which is not performed");
  // -demod_axes and -slice_axis would each determine the slices that are demodulated individually;
  //   both being accepted would allow two different definitions of the slice axis.
  if (!opt_axes.empty() && !opt_slice_axis.empty())
    throw Exception("Options -demod_axes and -slice_axis are mutually exclusive: "
                    "the phase demodulation is performed along the axes that do not separate slices, "
                    "so either option alone determines those axes");
  if (opt_axes.empty()) {
    result.axes = within_slice_axes(select_slice_axis(H, "phase demodulation"));
    INFO("Phase demodulation axes: " + join(result.axes, ","));
  } else {
    result.axes = parse_ints<size_t>(opt_axes[0][0]);
    // A demodulation axis list names the axes along which an FFT is taken; naming an axis twice is
    //   not a second transform but a malformed request. It must be rejected here: downstream,
    //   duplicate inner axes make ThreadedLoop() iterate an axis within itself and never terminate.
    std::vector<size_t> sorted_axes(result.axes);
    std::sort(sorted_axes.begin(), sorted_axes.end());
    if (std::adjacent_find(sorted_axes.begin(), sorted_axes.end()) != sorted_axes.end())
      throw Exception("Option -demod_axes may not name the same axis more than once "
                      "(got \"" +
                      std::string(opt_axes[0][0]) + "\")");
    for (auto axis : result.axes) {
      if (axis > 2)
        throw Exception("Phase demodulation implementation not yet robust to non-spatial axes");
    }
    // Two demodulation axes imply that the third separates slices; check that implication for
    //   the signature of axis indices read from the image as stored on disk.
    if (result.axes.size() == 2) {
      const size_t implied_slice_axis = 3 - result.axes[0] - result.axes[1];
      const SliceAxisInference inferred = infer_slice_axis(H, "phase demodulation");
      check_axis_frame(H,
                       "-demod_axes " + std::string(opt_axes[0][0]),
                       implied_slice_axis,
                       inferred,
                       "-demod_axes " + join(within_slice_axes(inferred.axis), ","));
    }
  }
  // The adaptive phase correction solver is structurally two-dimensional (it solves an in-slice
  //   problem); any other axis count is impermissible usage rather than an implementation fault,
  //   and must therefore be rejected here rather than reaching the solver's own assertion.
  //   The linear / Hann demodulators go through Filter::Demodulate, which accepts any number of
  //   FFT axes, so the constraint is specific to 'apc'.
  if (result.mode == demodulation_t::APC && result.axes.size() != 2)
    throw Exception("Option -demod_axes must name exactly two axes "
                    "when adaptive phase correction is in use "
                    "(-demodulate apc, the default for complex input data): "
                    "the adaptive phase estimator solves a two-dimensional in-slice problem "
                    "(" +
                    str(result.axes.size()) +
                    " axes were provided). "
                    "Either name the two within-slice axes (e.g. -demod_axes 0,1), "
                    "or select a demodulator that accepts a different number of axes "
                    "(-demodulate linear or -demodulate hann)");
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
demean_type select_demean(const Header &H, const Eigen::MatrixXd &grad) {
  // Whether a b-value shell structure can be inferred. The table itself is resolved once by the
  //   calling command; only the shell decomposition is probed here, and its failure is a
  //   legitimate reason to fall back to another demeaning mode.
  bool shells_available = false;
  if (grad.rows() > 0) {
    try {
      auto shells = DWI::Shells(grad);
      shells_available = true;
    } catch (Exception &) {
    }
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

// Partitioning is what moves the demeaning: each partition's own mean subtraction must be
//   orthogonal to that partition's decomposition, which a single subtraction over the whole volume
//   series does not achieve. Without partitioning the preconditioner does it once for the series.
demean_site_t resolve_demean_site(const demean_type demean, const ssize_t num_partitions) {
  assert(num_partitions >= 1);
  if (demean == demean_type::NONE)
    return demean_site_t::NONE;
  return num_partitions > 1 ? demean_site_t::ESTIMATOR : demean_site_t::PRECONDITIONER;
}

} // namespace MR::Denoise::Precondition
