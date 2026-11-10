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

#include "header.h"

namespace MR::Denoise {

// Interrogate the input image header key-values for evidence that upstream image processing has
//   already invalidated the assumptions on which Marchenko-Pastur-based noise level estimation
//   rests, issuing a WARN() for each class of violation for which evidence is found.
// Two such classes are discoverable from image metadata of DICOM provenance:
// - Resampling of the image data by interpolation, of which vendor gradient non-linearity
//   distortion correction is the most prevalent instance. Interpolation correlates the noise
//   between neighbouring voxels, breaking the independence on which the Marchenko-Pastur
//   distribution of the noise eigenvalues depends.
// - Reconstruction of a finer image matrix than the acquired k-space supports, ie. zero-filling of
//   unacquired k-space. The unmeasured outer k-space contributes signal but no noise, which
//   likewise correlates the noise between neighbouring voxels.
// In both cases the consequence for this command is the same: the noise portion of each patch's
//   eigenspectrum is compressed relative to the Marchenko-Pastur prediction, and the noise level
//   is consequently underestimated.
// The absence of any warning is *not* evidence that the input data are suitable. Most of the
//   relevant fields do not survive conversion from DICOM (that for the acquisition matrix
//   especially rarely does), the vendor conventions interrogated here are not exhaustive, and
//   resampling performed after conversion (eg. gradient unwarping or upsampling by a third-party
//   tool) leaves no trace in the image header at all.
void warn_if_metadata_precludes_mp(const Header &H);

} // namespace MR::Denoise
