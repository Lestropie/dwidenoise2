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

#include "header.h"
#include "types.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Inference, from image header key-value metadata, of the EFFECTIVE number of receive channels N
//   combined into magnitude input data, which sets the degrees of freedom (2N) of the noise
//   distribution that the variance-stabilising transform is built to model (see noise_model.h).
//
// EFFECTIVE, NOT PHYSICAL. The quantity required is not the number of coil elements, nor even the
//   number of receiver channels, but the number of statistically INDEPENDENT channels that the
//   sum-of-squares combination effectively averaged. Three distinct effects separate it from the
//   element count printed on the coil, and they compound:
//   - Elements are not channels. Siemens reports elements in "ReceiveCoilActiveElements", but a
//     coil's elements may be combined into fewer receiver channels before reconstruction (on Tim
//     systems with a Mode Matrix, only the highest mode gives a one-to-one correspondence). A
//     12-element head coil read out through four channel groups presents four channels, not twelve.
//   - Correlation. The channels' noise is correlated and their sensitivity profiles overlap, so
//     even genuinely distinct channels do not contribute independent degrees of freedom; the
//     equivalent count is the participation ratio of the noise-covariance eigenvalues, and is
//     fractional (Aja-Fernandez & Tristan-Vega 2012).
//   - Reconstruction. Parallel imaging correlates the channels further and makes the effective
//     count vary over space (Aja-Fernandez et al. 2011); a single scalar is then an approximation.
//   Empirically the gap is large: Sakaie & Lowe (2017) measured 3.76 +/- 0.07 effective channels
//   for a nominally 12-element coil. The table below therefore stores MEASURED effective counts
//   only, and holds an entry for a coil only where such a measurement has been reported.
//
// The determination proceeds in two stages, because the channel count is only relevant to the
//   noise statistics if the channels were combined by sum of squares:
//
// 1. How the channels were combined.
//    - The CMRR multi-band sequences perform their own SENSE1 (phase-sensitive, unit-weight
//      SENSE) coil combination and record it in DICOM Image Comments (0020,4000), e.g.
//      "Unaliased MB3/PE3/LB SENSE1". This is done inside the sequence and therefore overrides
//      the coil combine mode selected in the protocol, which the Siemens CSA header (and hence
//      the "CoilCombinationMethod" field derived from it) continues to report unchanged; real
//      data exist carrying "SENSE1" in Image Comments alongside "Sum of Squares" in
//      CoilCombinationMethod. The sequence's declaration is therefore given precedence.
//    - Otherwise, BIDS "CoilCombinationMethod", which dcm2niix derives from the Siemens CSA
//      "ucCoilCombineMode" and writes as either "Sum of Squares" or "Adaptive Combine".
//    Any phase-sensitive combination (SENSE1, adaptive combine) collapses the multi-channel data
//      to a single complex image before the magnitude is taken, so the result is Rician (N = 1)
//      irrespective of how many channels were combined. This is not merely theoretical for
//      adaptive combine: Sakaie & Lowe (2017), fitting the effective channel count to data from
//      a 12-channel coil, obtained 1.03 +/- 0.01 for adaptive combine against 3.76 +/- 0.07 for
//      sum of squares.
//
// 2. Only where sum-of-squares combination is established, the effective channel count, resolved
//    against a lookup table built into this software. The key is all four of BIDS "Manufacturer",
//    "ManufacturersModelName", "ReceiveCoilName" (the coil's marketing name, e.g. "HeadNeck_64")
//    and "ReceiveCoilActiveElements" (the selected element groups, e.g. "HC1-7;NC1,2"). All four
//    are required, because none of them identifies the hardware on its own: a coil name is unique
//    only within a vendor and generation; the same element string denotes different coils (both
//    the 32-element Head_32 and the 12-element HeadMatrix report "HEA;HEP"), so it is meaningful
//    only once the coil is known; and the effective count belongs to the coil together with the
//    receiver hardware and the reconstruction, so even one coil design does not carry one count
//    across system generations. Where "ReceiveCoilActiveElements" is absent, dcm2niix's own
//    "CoilString" is consulted in its place, as on some data it is that field that carries the
//    element selection.
//
//    The table itself is in coil_database.{h,cpp}, which also documents how to add an entry.
//
//    The table holds values of two kinds, which the software keeps distinct at every point:
//    - MEASURED: an effective channel count established experimentally for that coil under
//      sum-of-squares combination.
//    - HYPOTHESISED: where the element string enumerates the CLUSTERS of the receive array
//      ("HC1-7;NC1,2" naming seven head clusters and two neck clusters), the number of clusters
//      active is taken as the effective channel count, on the hypothesis that each cluster
//      contributes about one independent channel. Elements within a cluster are adjacent, strongly
//      correlated and read out together, so the cluster count is far closer to the effective count
//      than the element count is; but the hypothesis is untested, so a value obtained this way is
//      always accompanied by a warning saying so. Where the element string names only regions of
//      the array rather than its clusters ("HEA;HEP"), no hypothesised value is offered.
//
// Where any step cannot be completed the inference fails, rather than a value being guessed;
//   the calling command then proceeds with the Rician default and says so. Note that what is never
//   admitted is an effective count taken from an ELEMENT count, or from any scaling of one: that
//   is not an approximation of the right answer but an unrelated number, and one that would
//   silently misparameterise the transform.
class CoilInference {
public:
  // The degrees of freedom where they could be established: exactly 2 (one effective channel)
  //   wherever the coil combination is phase-sensitive, or twice the measured effective channel
  //   count where the channels were combined by sum of squares. Empty where the metadata do not
  //   permit the determination. Expressed as degrees of freedom, in common with everything else
  //   internal to the noise model; the equivalent channel count is half of it.
  std::optional<default_type> dof;
  // Whether "dof" rests on an effective channel count measured for this coil, or on the number of
  //   element clusters the header reports as active (see the discussion of the two kinds of table
  //   entry above); false where "dof" is empty, and false for the Rician value that any
  //   phase-sensitive combination implies, that being a property of the combination rather than a
  //   claim about the coil. A hypothesised value is applied in full, but is reported to the user
  //   as a hypothesis.
  bool hypothesised{false};
  // One phrase naming the metadata on which the outcome rests, for inclusion in the message that
  //   reports it; where "dof" is empty this instead names what was missing.
  std::string evidence;
  // In-plane parallel imaging acceleration factor reported by the header, where it reports one.
  //   Under parallel imaging the reconstruction correlates the noise between channels, so the
  //   effective degrees of freedom fall below the nominal 2N and vary over space; the count
  //   inferred above is then an upper bound.
  std::optional<default_type> acceleration;
};

// Interrogate the header key-values as described above.
CoilInference infer_noise_dof(const Header &H);

// Reconcile the degrees of freedom inferred from the header with any value the user specified
//   explicitly, reporting the outcome, and return the value to be used:
// - user value supplied: it is always what is returned, as an explicit instruction outranks an
//     inference drawn from metadata that may not describe the data as they now stand. Where an
//     inference was also drawn and disagrees, that disagreement is reported as a warning. A value
//     below the model's admissible range is clamped, with a warning naming both values.
// - no user value, inference successful: the inferred value, reported at INFO level; where it was
//     hypothesised from the coil's cluster count rather than measured for the coil, a warning says
//     so, since the value parameterises the noise model exactly as a measured one would and
//     nothing in the results would otherwise distinguish the two.
// - no user value, no inference: 2 (Rician), with a warning stating as much.
// "user_option" names the command-line option through which "user_value" was supplied
//   ("noise_channels" or "noise_dof"), so that messages quote back the option the user actually
//   used; it is ignored when "user_value" is empty. "user_value" is in every case a
//   degrees-of-freedom value: a channel count is doubled by the caller before arriving here.
default_type
resolve_noise_dof(const Header &H, const std::optional<default_type> &user_value, const std::string &user_option);

} // namespace MR::Denoise::Precondition::NoiseModel
