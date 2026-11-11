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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "types.h"

// Generic interrogation of image header key-values.
//
// These are the primitives shared by every part of this software that reasons about the
//   provenance of the input data from metadata of (typically) DICOM origin: the check for
//   prior processing that invalidates the Marchenko-Pastur assumptions
//   (Denoise::warn_if_metadata_precludes_mp) and the inference of the degrees of freedom
//   that parameterise the magnitude noise model
//   (Denoise::Precondition::NoiseModel::infer_noise_dof).
// They are deliberately forgiving: any key may be absent, may carry a value of a type other
//   than the one sought, and may have reached the header by any of several routes (a JSON
//   sidecar, MRtrix3's own DICOM reader, a third-party converter), each with its own
//   conventions for separating the values of a multi-valued field. Every function therefore
//   reports failure by returning an empty result rather than by throwing: absent or
//   unintelligible metadata are the norm, not an error.

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
extern const std::vector<std::string> descriptive_keys;

// Split a header key-value into upper-case tokens. Which character separates the values of a
//   multi-valued field depends on the route by which it reached the header: MRtrix3's own DICOM
//   reader joins Image Type values with spaces and wraps them in brackets, a JSON sidecar array
//   arrives newline-separated, a raw DICOM multi-valued string retains the standard backslash,
//   and other converters use commas; so all of them are treated as separators.
std::vector<std::string> metadata_tokens(const std::string &value);

// Every (key, token) pair for which one of the nominated tokens appears among the values of one of
//   the nominated keys, each rendered as a phrase for inclusion in a warning message.
std::vector<std::string> find_metadata_tokens(const KeyValues &keyval,
                                              const std::vector<std::string> &keys,
                                              const std::vector<std::string> &tokens);

// The verbatim (whitespace-stripped) value of a header key-value, or nullopt if the key is absent
//   or its value is empty. Unlike the interpretations below, no conversion is attempted: this is
//   for fields whose value is a free-form identifier (a coil name, a reconstruction mode) that is
//   to be matched or reported as written.
std::optional<std::string> metadata_string(const KeyValues &keyval, const std::string &key);

// Boolean interpretation of a header key-value, or nullopt if it is not one. to<bool>() accepts
//   both the "true"/"false" spelling into which File::JSON::read() converts a JSON boolean and the
//   integer flag that a converter reading a vendor field may write instead.
std::optional<bool> metadata_bool(const KeyValues &keyval, const std::string &key);

// Numeric interpretation of a header key-value, or nullopt if it is absent or is not a single
//   finite number.
std::optional<default_type> metadata_float(const KeyValues &keyval, const std::string &key);

// The largest of the numbers in a multi-valued numeric header key-value, or nullopt if the key is
//   absent, or any of its values is not a finite number, or all of them are non-positive.
std::optional<default_type> metadata_max(const KeyValues &keyval, const std::string &key);

} // namespace MR::Denoise
