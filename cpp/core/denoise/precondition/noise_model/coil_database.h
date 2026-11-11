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

#include <vector>

#include "types.h"

namespace MR::Denoise::Precondition::NoiseModel {

// DATABASE OF KNOWN RECEIVE COILS
//
// The built-in lookup table from receive coil to effective channel count, consulted when magnitude
//   input data are established to have been combined by sum of squares (see coils.h for the whole
//   inference, and for why it is the EFFECTIVE rather than the physical channel count that is
//   needed). The entries themselves are in coil_database.cpp; this header defines only the form
//   they take. The code that matches header metadata against the table is in coils.cpp, and nothing
//   there needs to change when an entry is added or amended.
//
// The table holds entries of two kinds, which are never conflated and which the user is never left
//   to tell apart by inspection:
//
// 1. MEASURED. An effective channel count established experimentally for that coil under
//    sum-of-squares combination and reported -- in the literature, or by measurement contributed
//    to this software. Applied without apology, and reported with its source.
//
// 2. HYPOTHESISED, from the clustering of the elements within the receive array. Where the header
//    enumerates the active element clusters ("HC1-7;NC1,2"), the count of those clusters is taken
//    as the effective channel count: 7 for "HC1-7", 9 for "HC1-7;NC1,2". The reasoning is that the
//    elements within one cluster are adjacent, strongly correlated, and read out together, so a
//    cluster is much closer to one independent channel than the elements within it are; the
//    element count is known to be far too large (see coils.h), and the cluster count is the next
//    quantity down that the header actually reports.
//    THIS IS A HYPOTHESIS, NOT A MEASUREMENT, and the software says so every time it applies one
//    (see resolve_noise_dof()). The single calibration point available bounds neither its accuracy
//    nor even its direction: on the Tim Head Matrix the header names two element groups where 3.76
//    effective channels were measured, i.e. the array's own clustering (four receiver groups of
//    three elements) is not what its element string enumerates. That coil's element string is
//    consequently NOT cluster-enumerated, and no coil whose string names only regions of the array
//    ("HEA;HEP") is given a hypothesised count -- the count of such groups is not a cluster count
//    and is known to understate the measured value.
//
// HOW TO ADD OR AMEND AN ENTRY
//
// Each entry is one CoilDefinition in coil_database.cpp. Every entry requires:
//   - The four identifying fields, as they appear in the BIDS sidecar / image header:
//     "Manufacturer", "ManufacturersModelName" (one or more system models), "ReceiveCoilName", and,
//     for each measured configuration, "ReceiveCoilActiveElements". See CoilDefinition below for
//     why all four are needed, and for the exact form in which each must be written.
//   - At least one of:
//     - a MEASURED configuration: the element selection, the measured effective channel count, and a
//       source string naming the measurement. The source is mandatory and is quoted back to the user
//       whenever the value is applied, so it must be specific enough for them to judge whether the
//       measurement transfers to their data (publication or contributor, coil, system, and the
//       reconstruction under which it was measured). Write the value as an explicit doubling of the
//       published channel count ("2.0 * 3.76"); see CoilConfiguration.
//     - a description of the coil's CLUSTER naming, which permits a hypothesised value. Add this only
//       where the cluster naming has been confirmed against real headers from that coil.
//   - A comment above the entry recording what the coil is, where the value comes from, and any
//     restriction on the systems to which it applies.
//
// Extending an existing entry to a further system model is as legitimate a contribution as adding
//   a coil, and carries the same requirement that the coil be known to ship with that system. The
//   list of system models is not exhaustive; the same coil on a system not listed simply fails to
//   match, and the command falls back to the Rician default with a message naming what it saw.
//
// Values NOT admitted under either heading, recorded here so that they are not mistakenly added:
//   - Element counts of any coil (Head_32 = 32 elements, HeadNeck_64 = 64 elements of which 52
//     cover the head, HeadNeck_20 = 16 head + 4 neck). These are physical element counts, larger
//     than the effective count by a factor that is neither known nor constant.
//   - Any scaling of an element count (its square root, its division by a fixed factor): observed
//     effective counts are not monotone in element count, so no function of it alone can serve.
//   - "N = 4 for a 12-channel head coil with GRAPPA" and "around N = 9 seems to work for a
//     32-channel head coil" (NLSAM documentation, citing Brion et al. and Varadarajan et al.).
//     The first is the coil's receiver-group count rather than a measurement; the second is an
//     acknowledged working value, is not attributed to a named coil, and describes a GRAPPA
//     reconstruction, whose effective count is a property of the reconstruction rather than of the
//     coil and varies over space.
//   - Degrees of freedom of roughly 4 to 10 estimated on a 64-channel Prisma head coil (Bas et al.
//     2024): the coil combination mode is not reported, so the value cannot be attributed to
//     sum of squares.
//   - The Siemens 32-channel head coil (Head_32). It reports "HEA;HEP", which names the anterior
//     and posterior halves of the array rather than its clusters, so neither kind of entry can be
//     made for it: no measurement exists, and the hypothesised rule has nothing to count.

// One selection of coil element groups, the measured effective channel count for it, and where
//   that measurement comes from. The provenance is not decoration: it is quoted back to the user
//   in the message reporting the inference, because the number is an experimental result whose
//   applicability to their acquisition they must be able to judge.
class CoilConfiguration {
public:
  // The active element groups in canonical form: upper case, without whitespace or any coil-connector
  //   prefix ("T:", "C:"), groups separated by ';' and sorted alphabetically, members within a group
  //   separated by ','. E.g. "HEA;HEP", "HC1-7;NC1,2". Header values are brought to this same form
  //   before comparison, so the several spellings under which a selection reaches the header
  //   ("T:HEA;HEP", "NC1_2;HC1-7") all match the one entry.
  const char *elements;
  // Degrees of freedom, in common with the rest of the noise model. Sources report an effective
  //   CHANNEL count, so entries are written as an explicit doubling of the published figure: the
  //   value transcribed from the source and the one conversion applied to it are then both visible
  //   at the point of entry, and no conversion is left to happen anywhere downstream.
  default_type dof;
  // Provenance of the measurement; mandatory (see above).
  const char *source;
};

// One receive coil on one family of systems. The key is all four of manufacturer, system model,
//   coil name and active elements, none of which identifies the coil on its own:
//   - "ReceiveCoilName" is a marketing name, unique only within a vendor and generation. Nothing
//     stops two vendors, or two generations of one vendor, from shipping different hardware under
//     names that reduce to the same string.
//   - The element string is a selection within a named coil, and the same string denotes different
//     hardware on different coils: both the 32-element Head_32 and the 12-element Tim Head Matrix
//     report "HEA;HEP".
//   - Even one coil design does not have one effective channel count: the count is a property of
//     the coil TOGETHER WITH the receiver hardware and the reconstruction, so the same coil on a
//     different system generation need not behave the same way. This is not hypothetical -- the
//     one measured entry in the table owes its value to the receiver grouping of the Trio
//     generation rather than to the coil.
//   Requiring the system to be identified as well as the coil is what keeps a tabulated value from
//   being applied to hardware it was never established on.
//
// The three name fields are written in ALPHANUMERIC form: upper case, with every character that is
//   not a letter or digit removed ("HeadNeck_64" -> "HEADNECK64", "MAGNETOM Prisma_fit" ->
//   "MAGNETOMPRISMAFIT"). Header values are reduced the same way before comparison, so punctuation,
//   spacing and case are immaterial.
class CoilDefinition {
public:
  // Sought WITHIN the header "Manufacturer" ("SIEMENS" matches "Siemens Healthineers"), so it should
  //   be the distinctive core of the vendor name rather than any one spelling of it.
  const char *manufacturer;
  // One of these must appear WITHIN the header "ManufacturersModelName" ("PRISMA" matches both
  //   "Prisma" and "Prisma_fit"); likewise the distinctive core of each system name.
  std::vector<const char *> models;
  // "ReceiveCoilName", matched IN FULL.
  const char *name;
  // Element selections for which an effective channel count has been MEASURED. Consulted first,
  //   as a measurement always outranks the hypothesis below. May be empty.
  std::vector<CoilConfiguration> measured;
  // Where the coil's element string enumerates the clusters of the array -- a family prefix
  //   followed by cluster indices, as a range or list ("HC1-7", "NC1,2") -- a description of that
  //   naming; nullptr where it does not, which suppresses the hypothesised count entirely for this
  //   coil. The string is quoted in the message reporting a hypothesised value, so it should say
  //   what the families are and how many physical elements they hold, that being the information
  //   from which the user can judge the hypothesis.
  const char *clusters;
};

// The table itself; see coil_database.cpp.
const std::vector<CoilDefinition> &coil_database();

} // namespace MR::Denoise::Precondition::NoiseModel
