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

#include "denoise/denoise.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "algo/threaded_loop.h"
#include "axes.h"
#include "exception.h"
#include "filter/smooth.h"
#include "mrtrix.h"
#include "transform.h"

namespace MR::Denoise {

using namespace App;

const char *patent_description =
    "Estimation of the noise level of an image series"
    " based on the Marchenko-Pastur distribution (-estimator exp1/exp2)"
    " is protected by the following patent: \n"
    "US10698065B2."
    " System, method and computer accessible medium for noise estimation, noise removal and gibbs ringing removal."
    " Dmitry Novikov, Jelle Veraart, Els Fieremans."
    " Contact: https://tov.med.nyu.edu/about/contact-us/";

const char *first_step_description =
    "Important note:"
    " image denoising must be performed as the first step of the image processing pipeline."
    " The routine will not operate correctly if interpolation or smoothing"
    " has been applied to the data prior to denoising."
    " Where metadata of DICOM provenance are preserved in the image header,"
    " they are interrogated for evidence of such prior processing"
    "---specifically gradient non-linearity distortion correction,"
    " and reconstruction of a finer image matrix than the acquired k-space supports"
    " (k-space zero-filling)---"
    "and a warning is issued for any that is found."
    " The absence of such a warning is however not evidence that the data are suitable:"
    " the relevant fields seldom survive conversion from DICOM,"
    " and processing applied after that conversion leaves no trace in the image header at all.";

const char *non_gaussian_noise_description =
    "Magnitude-reconstructed MRI data exhibit non-Gaussian noise statistics"
    " (Rician for a single receive channel,"
    " or non-central chi for multiple channels combined by sum-of-squares reconstruction),"
    " which deviate from the Gaussian noise model that underlies PCA-based denoising,"
    " most prominently for data close to the noise floor."
    " This command accounts for these statistics through a noise-model-aware variance-stabilising transform"
    " (configurable via the -vst_method and -noise_dof options)"
    " that renders the data approximately Gaussian and homoscedastic prior to PCA."
    " Nonetheless, where the complex image data are available,"
    " including the MRI phase as part of a complex input image remains preferable,"
    " as phase demodulation yields zero-mean Gaussian noise"
    " and thereby avoids the non-Gaussian regime altogether.";

const char *decomposition_description =
    "By default, the command uses Eigen's Bidirectional Divide-and-Conquer"
    " Singular Value Decomposition (BDCSVD) algorithm"
    " for deriving the eigenspectrum for each PCA patch."
    " This is more numerically precise than the Self-Adjoint solver used"
    " in the original MRtrix3 dwidenoise command;"
    " it does however come at increased computational expense,"
    " particularly for very large patches."
    " If runtime is prohibitive for data with a very large number of volumes,"
    " it may be preferred to revert to the original decomposition method"
    " using command-line option -decomposition selfajoint.";

const char *filter_description =
    "By default, optimal value shrinkage based on minimisation of the Frobenius norm "
    "will be used to attenuate eigenvectors based on the estimated noise level. "
    "Hard truncation of sub-threshold components and inclusion of supra-threshold components"
    "---which was the behaviour of the dwidenoise command in version 3.0.x---"
    "can be activated using -filter truncate."
    "Alternatively, optimal truncation as described in Gavish and Donoho 2014 "
    "can be utilised by specifying -filter optthresh. "
    "For denoising of functional MRI data, "
    "it is hypothesized (though not exhaustively tested) "
    "that use of -filter truncate may be a superior choice to the default optimal shrinkage "
    "as it minimises risk of loss of BOLD signal fluctuations near the noise floor.";

const char *aggregation_description =
    "-aggregation exclusive corresponds to the behaviour of the dwidenoise command in version 3.0.x, "
    "where the output intensities for a given image voxel are determined exclusively "
    "from the PCA decomposition where the sliding spatial window is centred at that voxel. "
    "In all other use cases, so-called \"overcomplete local PCA\" is performed, "
    "where the intensities for an output image voxel are some combination of all PCA decompositions "
    "for which that voxel is included in the local spatial kernel. "
    "There are multiple algebraic forms that modulate the weight with which each decomposition "
    "contributes with greater or lesser strength toward the output image intensities. "
    "The various options are: "
    "'gaussian': A Gaussian weighting centred on each decomposition's kernel, "
    "such that decompositions centred more closely to the output voxel have greater influence; "
    "its full width at half maximum defaults to twice the spacing between adjacent PCA patches "
    "and may be modulated via the -aggregator_fwhm option; "
    "'invl0': The inverse of the L0 norm (ie. rank) of each decomposition, "
    "as used in Manjon et al. 2013; "
    "'rank': The rank of each decomposition, "
    "such that high-rank decompositions contribute more strongly to the output intensities "
    "regardless of distance between the output voxel and the centre of the decomposition kernel; "
    "'uniform': All decompositions that include the output voxel in the sliding spatial window contribute equally.";

const Option datatype_option = Option("datatype",
                                      "Datatype for the eigenvalue decomposition"
                                      " (single or double precision). "
                                      "For complex input data,"
                                      " this will select complex float32 or complex float64 datatypes.")
                               + Argument("float32/float64").type_choice<dtype_t>();

const Option decomposition_option = Option("decomposition",
                                           "Method used for the decomposition of the data in each patch;"
                                           " options are: " + Enum::join<decomp_type>(", ") +
                                           " (default: BDCSVD)")
                                    + Argument("choice").type_choice<decomp_type>();

ssize_t dimlong_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp) { return std::max(m - rp, n); }
ssize_t rank_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp) { return std::min(m - rp, n); }
ssize_t rank_zero(const ssize_t m, const ssize_t n, const ssize_t rp) {
  return (std::min(m, n) - rank_nonzero(m, n, rp));
}

size_t num_volumes(const Header& H) {
  switch (H.ndim()) {
    case 0:
    case 1:
    case 2:
      throw Exception("Attempt to compute number of volumes for a <3D image");
    case 3:
      return 1;
    default: {
      size_t result = H.size(3);
      for (size_t axis = 4; axis != H.ndim(); ++axis)
        result *= H.size(axis);
      return result;
    }
  }
}

namespace {

// Header key-values within which the free-text / multi-valued descriptors of an image's
//   reconstruction may be found, in order of decreasing reliability:
// - "ImageType": DICOM Image Type (0008,0008); dcm2niix exports it as a JSON array, which
//     File::JSON::read() imports as newline-separated values.
// - "ImageTypeText": Siemens private (0021,1175), exported by recent dcm2niix versions; on XA
//     software it is here, rather than in (0008,0008), that the descriptors appear.
// - "ImageComments": DICOM Image Comments (0020,4000).
// - "comments": where MRtrix3's own DICOM reader deposits the Image Type values, embedded within
//     its "study: <name> [ <image type> ]" line (File::Dicom::dicom_to_mapper()).
// - "ImageHistory": Siemens private (0021,1176).
// - "DerivationDescription": DICOM (0008,2111), which describes how a derived image was produced.
// - "SeriesDescription", "ProtocolName": not intended to carry such descriptors, but some sites
//     append them to the series name, and reading them costs nothing.
const std::vector<std::string> descriptive_keys{"ImageType",
                                                "ImageTypeText",
                                                "ImageComments",
                                                "comments",
                                                "ImageHistory",
                                                "DerivationDescription",
                                                "SeriesDescription",
                                                "ProtocolName"};

// Siemens' zero-fill interpolation flags, as exported by dcm2niix from the CSA header.
const std::vector<std::string> interpolation_keys{"Interpolation2D", "Interpolation3D"};

// Header key-values reporting the in-plane matrix that was actually acquired:
// - "AcquisitionMatrix": DICOM Acquisition Matrix (0018,1310), four values -- frequency rows,
//     frequency columns, phase rows, phase columns -- of which the two that do not apply are zero.
// - "BaseResolution": the Siemens readout base resolution, exported by dcm2niix from the CSA header
//     ("# of acquisition lines").
const std::vector<std::string> acquired_matrix_keys{"AcquisitionMatrix", "BaseResolution"};

// Split a header key-value into upper-case tokens. Which character separates the values of a
//   multi-valued field depends on the route by which it reached the header: MRtrix3's own DICOM
//   reader joins Image Type values with spaces and wraps them in brackets, a JSON sidecar array
//   arrives newline-separated, a raw DICOM multi-valued string retains the standard backslash,
//   and other converters use commas; so all of them are treated as separators.
std::vector<std::string> metadata_tokens(const std::string &value) {
  std::vector<std::string> result;
  for (const auto &token : split(value, " \t\n\r\\,;:[]()", true))
    result.push_back(uppercase(token));
  return result;
}

// Every (key, token) pair for which one of the nominated tokens appears among the values of one of
//   the nominated keys, each rendered as a phrase for inclusion in a warning message.
std::vector<std::string> find_metadata_tokens(const KeyValues &keyval,
                                              const std::vector<std::string> &keys,
                                              const std::vector<std::string> &tokens) {
  std::vector<std::string> result;
  for (const auto &key : keys) {
    const auto entry = keyval.find(key);
    if (entry == keyval.end())
      continue;
    const std::vector<std::string> value_tokens = metadata_tokens(entry->second);
    for (const auto &token : tokens) {
      if (std::find(value_tokens.begin(), value_tokens.end(), token) != value_tokens.end())
        result.push_back("\"" + key + "\" contains \"" + token + "\"");
    }
  }
  return result;
}

// Boolean interpretation of a header key-value, or nullopt if it is not one. to<bool>() accepts
//   both the "true"/"false" spelling into which File::JSON::read() converts a JSON boolean and the
//   integer flag that a converter reading a vendor field may write instead.
std::optional<bool> metadata_bool(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  try {
    return to<bool>(entry->second);
  } catch (Exception &) {
    return std::nullopt;
  }
}

// Numeric interpretation of a header key-value, or nullopt if it is absent or is not a single
//   finite number.
std::optional<default_type> metadata_float(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  try {
    const default_type value = to<default_type>(entry->second);
    if (!std::isfinite(value))
      return std::nullopt;
    return value;
  } catch (Exception &) {
    return std::nullopt;
  }
}

// The largest of the numbers in a multi-valued numeric header key-value, or nullopt if the key is
//   absent, or any of its values is not a finite number, or all of them are non-positive.
std::optional<default_type> metadata_max(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  default_type result = 0.0;
  for (const auto &token : split(entry->second, " \t\n\r\\,;", true)) {
    try {
      const default_type value = to<default_type>(token);
      if (!std::isfinite(value))
        return std::nullopt;
      result = std::max(result, value);
    } catch (Exception &) {
      return std::nullopt;
    }
  }
  return result > 0.0 ? std::optional<default_type>(result) : std::nullopt;
}

// Matrix sizes are integers, but arrive via the generic floating-point interpretation above.
std::string str_matrix_size(const default_type value) { return str(ssize_t(std::lround(value))); }

// Voxel-wise functors for ThreadedLoop. ThreadedLoop copies the functor per thread, so each
//   thread receives independent Image voxel accessors (deep-copied); the operations below are
//   strictly voxel-local, so no locking is required.

// Copy (and optionally condition) a noise-level map into a scratch image, with either padded
//   edge-clamped sampling (pad) or a direct positional copy, and optional NaN-to-zero imputation.
//   Holds its own copy of the source image so each thread samples it independently.
class ConditionNoiseMapFunctor {
public:
  ConditionNoiseMapFunctor(Image<float> &in,
                           const noise_impute_type nan_to_zero,
                           const noise_pad_type pad,
                           const std::array<ssize_t, 3> &pad_lo)
      : in(in), nan_to_zero(nan_to_zero), pad(pad), pad_lo(pad_lo) {}
  void operator()(Image<float> &out) {
    if (pad == noise_pad_type::PAD) {
        // Padding may be asymmetric (sized per edge to reach the coverage target), so the offset
        //   is per-axis. Clamping replicates the nearest edge value into the padded region.
        for (ssize_t axis = 0; axis != 3; ++axis)
          in.index(axis) = std::max(ssize_t(0), std::min(in.size(axis) - 1, out.index(axis) - pad_lo[axis]));
    } else {
      assign_pos_of(out).to(in);
    }
    if (nan_to_zero == noise_impute_type::NAN_TO_ZERO)
      out.value() = std::isfinite(in.value()) ? in.value() : 0.0F;
    else
      out.value() = in.value();
  }

private:
  Image<float> in;
  noise_impute_type nan_to_zero;
  noise_pad_type pad;
  std::array<ssize_t, 3> pad_lo;
};

// Fill every voxel of a scratch image with a fixed value.
class FillFloatFunctor {
public:
  explicit FillFloatFunctor(const float value) : value(value) {}
  void operator()(Image<float> &image) { image.value() = value; }

private:
  float value;
};

// Per-voxel signal-rank density (rank per mm of kernel radius): rank_input / (P * radius).
class RankPerMMFunctor {
public:
  explicit RankPerMMFunctor(const default_type partition_scale) : partition_scale(partition_scale) {}
  void operator()(Image<float> &result, Image<uint16_t> &rank_input, Image<float> &max_dist) {
    const float radius = max_dist.value();
    result.value() = radius > 0.0F ? float(rank_input.value() / (partition_scale * radius)) : 0.0F;
  }

private:
  default_type partition_scale;
};
} // namespace

void warn_if_metadata_precludes_mp(const Header &H) {
  const KeyValues &keyval = H.keyval();

  // 1. Resampling of the image data by interpolation, gradient non-linearity distortion correction
  //    being the instance that image metadata most often reports.
  //    Siemens declares the correction it has applied among the Image Type descriptors: "DIS2D" for
  //    the in-plane-only correction of pre-XA software, "DIS3D" for the through-plane correction of
  //    XA30 and later, and "ND" ("no distortion [correction]") as the corresponding negative
  //    declaration. dcm2niix derives the BIDS "NonlinearGradientCorrection" field from those same
  //    descriptors, but only for some software versions, so both are interrogated.
  std::vector<std::string> resampled = find_metadata_tokens(keyval, descriptive_keys, {"DIS2D", "DIS3D"});
  if (metadata_bool(keyval, "NonlinearGradientCorrection").value_or(false))
    resampled.push_back("\"NonlinearGradientCorrection\" is set");
  if (resampled.empty()) {
    // The negative declaration is worth reporting: it distinguishes "the data are known not to have
    //   been distortion-corrected" from the far more common "the header says nothing either way".
    if (!find_metadata_tokens(keyval, descriptive_keys, {"ND"}).empty())
      INFO("Image header explicitly declares that no gradient non-linearity distortion correction"
           " has been applied to the input data");
  } else {
    WARN("Image header indicates that gradient non-linearity distortion correction"
         " has been applied to the input data (" + join(resampled, "; ") + ");"
         " the interpolation intrinsic to such resampling correlates the noise between"
         " neighbouring voxels, so the eigenspectrum of each patch will not conform to the"
         " Marchenko-Pastur distribution and the noise level will be underestimated;"
         " denoising should instead be performed on the data as reconstructed by the scanner,"
         " prior to any such correction");
  }

  // 2. Reconstruction of a finer image matrix than the acquired k-space supports, ie. zero-filling
  //    of unacquired k-space.
  std::vector<std::string> zerofilled;
  // Siemens writes its zero-fill interpolation flags only when the feature was active, so the mere
  //   presence of one of these fields is the indication; a value that is explicitly false or zero is
  //   nonetheless respected.
  for (const auto &key : interpolation_keys) {
    if (keyval.find(key) != keyval.end() && metadata_bool(keyval, key).value_or(true))
      zerofilled.push_back("\"" + key + "\" is set");
  }
  // dcm2niix exports both the number of phase-encode lines actually acquired (from DICOM MR
  //   Acquisition Phase Encoding Steps In-plane (0018,9231)) and the reconstructed matrix size along
  //   that same axis. Partial Fourier and in-plane parallel imaging each reduce the former relative
  //   to the latter without the outer k-space having been zero-filled, so both are discounted before
  //   the residual discrepancy is judged.
  const std::optional<default_type> acquired_pe = metadata_float(keyval, "AcquisitionMatrixPE");
  const std::optional<default_type> recon_pe = metadata_float(keyval, "ReconMatrixPE");
  if (acquired_pe.has_value() && recon_pe.has_value() && acquired_pe.value() > 0.0) {
    default_type partial_fourier = metadata_float(keyval, "PartialFourier").value_or(1.0);
    if (partial_fourier > 1.0 && partial_fourier <= 100.0)
      partial_fourier *= 0.01; // Written as a percentage rather than the BIDS fraction
    if (!(partial_fourier > 0.0 && partial_fourier <= 1.0))
      partial_fourier = 1.0;
    default_type acceleration = metadata_float(keyval, "ParallelReductionFactorInPlane").value_or(1.0);
    if (!(acceleration >= 1.0))
      acceleration = 1.0;
    const default_type supported = acquired_pe.value() * acceleration / partial_fourier;
    // The tolerance absorbs the rounding of the acquired line count and the parallel imaging
    //   reference lines, which inflate that count above the exact quotient.
    if (recon_pe.value() > 1.05 * supported)
      zerofilled.push_back("the reconstructed phase-encode matrix (\"ReconMatrixPE\" = " +
                           str_matrix_size(recon_pe.value()) + ") exceeds what the " +
                           str_matrix_size(acquired_pe.value()) +
                           " acquired phase-encode lines (\"AcquisitionMatrixPE\") support");
  }
  // Where the acquired in-plane matrix is reported instead as a whole, it is compared against the
  //   reconstructed in-plane matrix. In each case the larger of the two values is used: whichever
  //   way round the readout and phase-encode axes lie, the larger acquired count is the readout base
  //   resolution (the phase-encode count being the one reduced by a rectangular field of view), and
  //   the readout direction is subject to neither parallel imaging nor, ordinarily, partial Fourier,
  //   so this comparison requires no discounting.
  std::vector<std::pair<std::string, default_type>> acquired_inplane;
  for (const auto &key : acquired_matrix_keys) {
    const std::optional<default_type> value = metadata_max(keyval, key);
    if (value.has_value())
      acquired_inplane.emplace_back(key, value.value());
  }
  if (!acquired_inplane.empty()) {
    const ssize_t recon_inplane = std::max(H.size(0), H.size(1));
    for (const auto &acquired : acquired_inplane) {
      // Zero-fill interpolation doubles the reconstructed matrix, so requiring an excess of more
      //   than half again retains that signature while tolerating the more modest enlargement
      //   produced by phase oversampling (conventionally no more than 50%), which is not
      //   zero-filling of unacquired k-space.
      if (default_type(recon_inplane) > 1.5 * acquired.second)
        zerofilled.push_back("the reconstructed in-plane matrix (" + str(recon_inplane) +
                             ") exceeds the acquired matrix (\"" + acquired.first + "\" = " +
                             str_matrix_size(acquired.second) + ")");
    }
  }
  if (!zerofilled.empty())
    WARN("Image header indicates that the input data were reconstructed at a finer image matrix"
         " than the acquired k-space supports (" + join(zerofilled, "; ") + ");"
         " zero-filling of unacquired k-space correlates the noise between neighbouring voxels,"
         " so the eigenspectrum of each patch will not conform to the Marchenko-Pastur distribution"
         " and the noise level will be underestimated");
}

Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type nan_to_zero,
                                 const noise_pad_type pad,
                                 const noise_smooth_type smooth,
                                 const Header *coverage) {
  Header H(in);
  // Margin required either side for cubic interpolation to be well-posed.
  constexpr ssize_t interp_margin = 2;
  std::array<ssize_t, 3> pad_lo{0, 0, 0};
  std::array<ssize_t, 3> pad_hi{0, 0, 0};
  if (pad == noise_pad_type::PAD) {
    pad_lo = {interp_margin, interp_margin, interp_margin};
    pad_hi = pad_lo;
    if (coverage != nullptr) {
      // A supplied map need not share the DWI voxel grid, but every position the DWI grid can
      //   present must lie within the map's sampleable region. Project the eight corners of the
      //   target field of view into this map's voxel space and widen the padding until they are
      //   all enclosed with the interpolation margin to spare. Expressing the requirement in the
      //   map's own voxel space handles differing voxel sizes, offsets and orientations alike.
      const Transform T_target(*coverage);
      const Transform T_map(H);
      Eigen::Vector3d vmin = Eigen::Vector3d::Constant(std::numeric_limits<default_type>::infinity());
      Eigen::Vector3d vmax = Eigen::Vector3d::Constant(-std::numeric_limits<default_type>::infinity());
      for (ssize_t c = 0; c != 8; ++c) {
        // Voxel-centre indices are 0 .. size-1; the sampled extent runs half a voxel beyond each.
        const Eigen::Vector3d corner((c & 1) ? coverage->size(0) - 0.5 : -0.5,  //
                                     (c & 2) ? coverage->size(1) - 0.5 : -0.5,  //
                                     (c & 4) ? coverage->size(2) - 0.5 : -0.5); //
        const Eigen::Vector3d in_map = T_map.scanner2voxel * (T_target.voxel2scanner * corner);
        vmin = vmin.cwiseMin(in_map);
        vmax = vmax.cwiseMax(in_map);
      }
      for (ssize_t axis = 0; axis != 3; ++axis) {
        pad_lo[axis] = std::max(pad_lo[axis], ssize_t(std::ceil(interp_margin - vmin[axis])));
        pad_hi[axis] = std::max(pad_hi[axis], ssize_t(std::ceil(vmax[axis] + interp_margin - (H.size(axis) - 1))));
        pad_lo[axis] = std::max<ssize_t>(0, pad_lo[axis]);
        pad_hi[axis] = std::max<ssize_t>(0, pad_hi[axis]);
      }
    }
    for (ssize_t axis = 0; axis != 3; ++axis)
      H.size(axis) += pad_lo[axis] + pad_hi[axis];
    H.transform().translation() = Transform(H).voxel2scanner * Eigen::Vector3d{default_type(-pad_lo[0]), //
                                                                               default_type(-pad_lo[1]), //
                                                                               default_type(-pad_lo[2])};
  }
  Image<float> out = Image<float>::scratch(H, "Conditioned version of \"" + std::string(in.name()) + "\"");
  ThreadedLoop(out).run(ConditionNoiseMapFunctor(in, nan_to_zero, pad, pad_lo), out);
  if (smooth == noise_smooth_type::SMOOTH) {
    Filter::Smooth smooth_filter(out);
    smooth_filter(out);
  }
  return out;
}

std::optional<default_type> imposed_scalar_value(const App::ParsedArgument &arg) {
  bool is_scalar = false;
  try {
    to<default_type>(arg.as_text());
    is_scalar = true;
  } catch (Exception &) {
  }
  if (!is_scalar)
    return std::nullopt;
  // Deliberately outside the try: as_float() enforces the option's declared range, and that
  //   "out of bounds" exception must reach the user rather than be caught and downgraded into an
  //   attempt to open the value as an image.
  return arg.as_float();
}

Image<float> import_vst_noise_map(const App::ParsedArgument &arg, const Header &H_spatial) {
  // Detect whether a scalar value or an image path was provided,
  //   matching the dispatch used by Estimator::make_imposed().
  // The raw text decides only *which* interpretation applies; once it is known to be a number,
  //   the conversion goes through ParsedArgument::as_float() so that the range declared on the
  //   option itself (Argument::type_float) is what rejects an out-of-range value, with the
  //   parser's own "out of bounds (valid range: ...)" message. Doing the range check here
  //   instead would either duplicate the declared bound or, if omitted, send a negative number
  //   to the image loader to be misreported as an unreadable filename.
  const std::optional<default_type> scalar = imposed_scalar_value(arg);
  Image<float> in;
  if (scalar.has_value()) {
    const default_type scalar_value = scalar.value();
    // The declared bound is inclusive, so it admits zero; a zero noise level is not a usable
    //   scale for the variance-stabilising transform and must still be excluded explicitly.
    if (scalar_value == 0.0)
      throw Exception("Noise level provided via -noise_in must be greater than zero");
    Header H(H_spatial);
    H.ndim() = 3;
    H.reset_intensity_scaling();
    H.datatype() = DataType::Float32;
    H.datatype().set_byte_order_native();
    in = Image<float>::scratch(H, "Spatially-constant noise level map");
    ThreadedLoop(in).run(FillFloatFunctor(float(scalar_value)), in);
  } else {
    in = Image<float>::open(std::string(arg));
    if (in.ndim() != 3)
      throw Exception("Noise level image provided via -noise_in must be 3-dimensional");
  }
  // Pad out to cover the DWI field of view: the map is interpolated at patch centres drawn from
  //   the DWI grid, so it need not share that grid but must be sampleable across all of it.
  return condition_noise_map(in, noise_impute_type::NONE, noise_pad_type::PAD, noise_smooth_type::NONE, &H_spatial);
}

Image<float> compute_rank_per_mm(Image<uint16_t> &rank_input, Image<float> &max_dist, const ssize_t num_partitions) {
  assert(num_partitions >= 1);
  Image<float> result = Image<float>::scratch(max_dist, "Scratch image for rank per mm kernel radius");
  const default_type partition_scale = default_type(num_partitions);
  ThreadedLoop(result).run(RankPerMMFunctor(partition_scale), result, rank_input, max_dist);
  return result;
}

Image<float> import_rank_per_mm_map(const App::ParsedArgument &arg, const Header &H_spatial) {
  // Detect whether a scalar value or an image path was provided, matching import_vst_noise_map()
  //   (including its reason for deferring the range check to the declared bound).
  const std::optional<default_type> scalar = imposed_scalar_value(arg);
  Image<float> in;
  if (scalar.has_value()) {
    // Unlike -noise_in, zero is a legitimate rank density, so the declared (inclusive) lower
    //   bound of zero expresses this option's domain exactly and needs no further check.
    const default_type scalar_value = scalar.value();
    Header H(H_spatial);
    H.ndim() = 3;
    H.reset_intensity_scaling();
    H.datatype() = DataType::Float32;
    H.datatype().set_byte_order_native();
    in = Image<float>::scratch(H, "Spatially-constant rank-per-mm map");
    ThreadedLoop(in).run(FillFloatFunctor(float(scalar_value)), in);
  } else {
    in = Image<float>::open(std::string(arg));
    if (in.ndim() != 3)
      throw Exception("Rank-per-mm image provided via -rankpermm_in must be 3-dimensional");
  }
  // As for the noise level map: cover the DWI field of view so interpolation is well-posed at
  //   every patch position, rather than requiring the supplied map to be grid-conformant.
  return condition_noise_map(
      in, noise_impute_type::NAN_TO_ZERO, noise_pad_type::PAD, noise_smooth_type::NONE, &H_spatial);
}

} // namespace MR::Denoise
