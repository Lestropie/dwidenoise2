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

#include "denoise/metadata.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise {

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

// The verbatim (whitespace-stripped) value of a header key-value, or nullopt if the key is absent
//   or its value is empty. Unlike the interpretations below, no conversion is attempted: this is
//   for fields whose value is a free-form identifier (a coil name, a reconstruction mode) that is
//   to be matched or reported as written.
std::optional<std::string> metadata_string(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  const std::string value = strip(entry->second);
  if (value.empty())
    return std::nullopt;
  return value;
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

namespace {

// Siemens' zero-fill interpolation flags, as exported by dcm2niix from the CSA header.
const std::vector<std::string> interpolation_keys{"Interpolation2D", "Interpolation3D"};

// Header key-values reporting the in-plane matrix that was actually acquired:
// - "AcquisitionMatrix": DICOM Acquisition Matrix (0018,1310), four values -- frequency rows,
//     frequency columns, phase rows, phase columns -- of which the two that do not apply are zero.
// - "BaseResolution": the Siemens readout base resolution, exported by dcm2niix from the CSA header
//     ("# of acquisition lines").
const std::vector<std::string> acquired_matrix_keys{"AcquisitionMatrix", "BaseResolution"};

// Matrix sizes are integers, but arrive via the generic floating-point interpretation above.
std::string str_matrix_size(const default_type value) { return str(ssize_t(std::lround(value))); }
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

} // namespace MR::Denoise
