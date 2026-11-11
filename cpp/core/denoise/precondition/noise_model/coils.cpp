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

#include "denoise/precondition/noise_model/coils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "denoise/metadata.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise::Precondition::NoiseModel {

namespace {

// How the individual receive channels were combined into the image data.
// - PHASE_SENSITIVE: combined as complex data before the magnitude was taken (adaptive combine,
//     SENSE1, or any other linear combination), leaving Rician magnitude statistics whatever the
//     number of channels.
// - SUM_OF_SQUARES: combined as magnitudes, leaving non-central chi statistics with 2N degrees of
//     freedom for N channels; the channel count must then be established.
// - UNKNOWN: the header says nothing from which either could be concluded.
enum class combination_t { UNKNOWN, PHASE_SENSITIVE, SUM_OF_SQUARES };

// Reduce a free-form identifier to a form in which incidental differences of punctuation, spacing
//   and case do not defeat comparison: upper case with every non-alphanumeric character removed.
//   Applied both to coil names ("HeadNeck_64", "HeadNeck 64" and "Head/Neck-64" all become
//   "HEADNECK64") and to the value of "CoilCombinationMethod" ("Sum of Squares" -> "SUMOFSQUARES").
std::string alphanumeric(const std::string &text) {
  std::string result;
  for (const char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0)
      result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return result;
}

// Canonical form of a string of active coil-element groups, so that the several spellings under
//   which the same selection reaches the header all resolve to one table key:
// - Some values carry a coil-connector prefix ("T:HEA;HEP", "C:A32"), which names the plug rather
//   than the elements and is discarded.
// - The "CoilString" field substitutes an underscore for the comma that separates the members of
//   a group ("HC1-7;NC1_2" for "HC1-7;NC1,2"), the comma being illegal in its DICOM value
//   representation.
// - The groups themselves are an unordered selection, so they are sorted; a header that reported
//   "NC1,2;HC1-7" denotes the same selection as "HC1-7;NC1,2".
std::string canonical_elements(const std::string &elements) {
  std::string text = uppercase(strip(elements));
  if (text.size() > 2 && text[1] == ':')
    text = text.substr(2);
  std::replace(text.begin(), text.end(), '_', ',');
  text.erase(std::remove_if(text.begin(),
                            text.end(),
                            [](const char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }),
             text.end());
  std::vector<std::string> groups = split(text, ";", true);
  std::sort(groups.begin(), groups.end());
  return join(groups, ";");
}

// Whether a tabulated identifier appears within the corresponding header value, both already
//   reduced by alphanumeric(). Containment rather than equality, because the same system and the
//   same vendor reach the header under several names: "Prisma" and "Prisma_fit", "Vida" and
//   "MAGNETOM Vida", "SIEMENS" and "Siemens Healthineers". Each tabulated string is therefore the
//   distinctive core of the name rather than any one of its spellings.
bool contains(const std::string &header_value, const char *const tabulated) {
  return header_value.find(tabulated) != std::string::npos;
}

bool contains_any(const std::string &header_value, const std::vector<const char *> &tabulated) {
  for (const char *const item : tabulated) {
    if (contains(header_value, item))
      return true;
  }
  return false;
}

// A whole number as written in an element string, or nullopt where the text is not one.
std::optional<size_t> parse_index(const std::string &text) {
  if (text.empty() || text.size() > 3)
    return {};
  for (const char c : text) {
    if (std::isdigit(static_cast<unsigned char>(c)) == 0)
      return {};
  }
  return size_t(std::stoul(text));
}

// No coil presents more clusters than this; a larger count is taken as evidence that the string
//   has been misparsed rather than as a coil of unprecedented size, since the consequence of
//   accepting it would be a grossly misparameterised noise model.
constexpr size_t max_plausible_clusters = 64;

// The distinct element clusters named by an active-element string (which must already be in the
//   form returned by canonical_elements()), or nullopt where the string does not enumerate
//   clusters.
//
// Siemens names the active elements as groups: a family prefix identifying a region of the array
//   ("HC" head and "NC" neck on the 64-channel Head/Neck coil, "HE" and "NE" on the 20-channel,
//   "SP" spine), followed by the indices selected within that family, written either as a range
//   ("HC1-7") or as a list ("NC1,2", which reaches some headers as "NC1_2" and which
//   canonical_elements() has already normalised). Each index denotes one cluster of elements that
//   the array reads out as a unit, so the number of indices is the number of clusters combined:
//   7 for "HC1-7", 9 for "HC1-7;NC1,2".
//
// Returning nullopt is a substantive outcome rather than a parse failure. An element string such
//   as "HEA;HEP" names two anatomical halves of the array, not its clusters, and how many clusters
//   lie within them cannot be recovered from the string; see coil_database() for why that
//   distinction decides whether a coil may be given a hypothesised count at all.
std::optional<size_t> count_element_clusters(const std::string &canonical, std::vector<std::string> &clusters) {
  std::set<std::string> unique;
  for (const std::string &group : split(canonical, ";", true)) {
    // Family prefix, then the indices selected within it. A group with no index at all names a
    //   region rather than a cluster, and a group beginning with a digit names no family.
    const size_t first_digit = group.find_first_of("0123456789");
    if (first_digit == std::string::npos || first_digit == 0)
      return {};
    const std::string family = group.substr(0, first_digit);
    for (const std::string &item : split(group.substr(first_digit), ",", true)) {
      const size_t dash = item.find('-');
      const std::optional<size_t> first = parse_index(dash == std::string::npos ? item : item.substr(0, dash));
      const std::optional<size_t> last = dash == std::string::npos ? first : parse_index(item.substr(dash + 1));
      if (!first.has_value() || !last.has_value())
        return {};
      if (first.value() < 1 || last.value() < first.value() || last.value() > max_plausible_clusters)
        return {};
      for (size_t index = first.value(); index != last.value() + 1; ++index)
        unique.insert(family + str(index));
    }
  }
  if (unique.empty() || unique.size() > max_plausible_clusters)
    return {};
  clusters.assign(unique.begin(), unique.end());
  return unique.size();
}

// One selection of coil element groups, the measured effective channel count for it, and where
//   that measurement comes from. The provenance is not decoration: it is quoted back to the user
//   in the message reporting the inference, because the number is an experimental result whose
//   applicability to their acquisition they must be able to judge.
class CoilConfiguration {
public:
  const char *elements; // as returned by canonical_elements()
  // Degrees of freedom, in common with the rest of the noise model. Sources report an effective
  //   CHANNEL count, so entries are written as an explicit doubling of the published figure: the
  //   value transcribed from the source and the one conversion applied to it are then both visible
  //   at the point of entry, and no conversion is left to happen anywhere downstream.
  default_type dof;
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
//     one measured entry below owes its value to the receiver grouping of the Trio generation
//     rather than to the coil.
//   Requiring the system to be identified as well as the coil is what keeps a tabulated value from
//   being applied to hardware it was never established on.
class CoilDefinition {
public:
  const char *manufacturer;         // alphanumeric(); sought within the header "Manufacturer"
  std::vector<const char *> models; // alphanumeric(); one must appear within the header model name
  const char *name;                 // alphanumeric() of "ReceiveCoilName"; matched in full
  // Element selections for which an effective channel count has been MEASURED. Consulted first,
  //   as a measurement always outranks the hypothesis below.
  std::vector<CoilConfiguration> measured;
  // Where the coil's element string enumerates the clusters of the array (see
  //   count_element_clusters()), a description of that naming; nullptr where it does not, which
  //   suppresses the hypothesised count entirely for this coil. The string is quoted in the
  //   message reporting a hypothesised value, so it should say what the families are and how many
  //   physical elements they hold, that being the information from which the user can judge the
  //   hypothesis.
  const char *clusters;
};

// The built-in lookup table from receive coil to effective channel count, holding entries of two
// kinds, which are never conflated and which the user is never left to tell apart by inspection:
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
//
// The list of system models against which an entry may match names the systems on which that coil
//   is known to be offered, and is not exhaustive; the same coil on a system not listed simply
//   fails to match, and the command falls back to the Rician default with a message naming what it
//   saw. Extending an entry to a further system is therefore as legitimate a contribution as adding
//   a coil, and carries the same requirement that the coil be known to ship with it.
//
// Adding a measured entry requires the four identifying fields as they appear in the header, the
//   element selection (canonicalised as above), the measured effective channel count, and a source
//   string naming the measurement; that string should be specific enough for a user to judge
//   whether the measurement transfers to their data. Adding a hypothesised entry requires only the
//   identifying fields and a description of the coil's cluster naming, but should be made only
//   where that naming has been confirmed against real headers from the coil.
const std::vector<CoilDefinition> &coil_database() {
  static const std::vector<CoilDefinition> result{
      // Siemens Tim Head Matrix, the 12-element head coil of the Tim Trio generation. Sakaie & Lowe
      //   (2017) reconstructed raw diffusion data from five subjects twice -- once by sum of
      //   squares, once by adaptive combine -- and fitted a central chi to the background of the
      //   b = 0 images, obtaining 3.76 +/- 0.07 effective channels for the sum-of-squares
      //   reconstruction (and 1.03 +/- 0.01 for adaptive combine, consistent with the Rician case
      //   that any phase-sensitive combination produces).
      //   Note that the figure is close to four rather than twelve because on that generation the
      //   twelve elements are read out through four groups of three; a Tim Head Matrix on a system
      //   presenting one channel per element would not be expected to give the same value, which is
      //   why the entry is confined to the Trio. Data from this coil have been observed to carry
      //   the element string in "CoilString" ("T:HEA;HEP") with "ReceiveCoilActiveElements" absent,
      //   which the caller's fallback accommodates. No cluster rule: "HEA;HEP" names the two halves
      //   of the array, not the four receiver groups through which it is read.
      {"SIEMENS",
       {"TRIO"},
       "HEADMATRIX",
       {{"HEA;HEP",
         2.0 * 3.76,
         "Sakaie & Lowe 2017, sum-of-squares reconstruction of diffusion data from a "
         "12-element Tim Head Matrix on a 3T Siemens Magnetom Tim Trio"}},
       nullptr},
      // Siemens Head/Neck 20, the 20-channel head/neck coil of the Tim 4G generation onwards
      //   (Aera, Avanto fit, Skyra, Prisma, and the BioMatrix systems that followed). Its 16 head
      //   elements are read out as the four clusters "HE1" to "HE4" and its 4 neck elements as the
      //   two clusters "NE1" and "NE2"; a whole-coil acquisition therefore reports "HE1-4;NE1,2"
      //   and is hypothesised at 6 effective channels. No effective count has been measured for it.
      {"SIEMENS",
       {"AERA", "AVANTO", "SKYRA", "PRISMA", "VIDA", "SOLA", "LUMINA", "ALTEA"},
       "HEADNECK20",
       {},
       "the 16 head elements are read out as the four clusters HE1 to HE4, and the 4 neck elements "
       "as the two clusters NE1 and NE2"},
      // Siemens Head/Neck 64, the 64-channel head/neck coil introduced with the Prisma. Its 52 head
      //   elements are read out as the seven clusters "HC1" to "HC7" and its 12 neck elements as the
      //   two clusters "NC1" and "NC2", so a whole-coil acquisition reports "HC1-7;NC1,2" and is
      //   hypothesised at 9 effective channels, and the common head-only selection "HC1-7" at 7. No
      //   effective count has been measured for it.
      {"SIEMENS",
       {"PRISMA", "SKYRA", "VIDA", "CIMA"},
       "HEADNECK64",
       {},
       "the 52 head elements are read out as the seven clusters HC1 to HC7, and the 12 neck elements "
       "as the two clusters NC1 and NC2"},
  };
  return result;
}

// Outcome of consulting the table above. The ways of failing are distinguished because they call
//   for different advice: an unrecognised coil (or system) is a gap in the table, whereas a
//   recognised coil that could not be resolved tells the user either which selections would have
//   been resolved or that its element string does not enumerate clusters.
class Lookup {
public:
  std::optional<default_type> dof;
  // Whether "dof" is a measured value or the hypothesised cluster count; meaningless where "dof"
  //   is empty. Carried all the way out to the user, because the two warrant different confidence.
  bool hypothesised{false};
  // What the value rests on: the source of the measurement, or the clusters that were counted.
  std::string basis;
  bool coil_known{false};
  std::vector<std::string> known_configurations;
  // Set where the coil is one whose elements are cluster-enumerated but the string at hand does
  //   not enumerate them, that being a different failure from having no rule at all.
  bool cluster_rule_defeated{false};
};

Lookup lookup_dof(const std::string &manufacturer,
                  const std::string &model,
                  const std::string &coil_name,
                  const std::string &elements) {
  Lookup result;
  const std::string manufacturer_key = alphanumeric(manufacturer);
  const std::string model_key = alphanumeric(model);
  const std::string name_key = alphanumeric(coil_name);
  const std::string elements_key = canonical_elements(elements);
  for (const auto &coil : coil_database()) {
    if (name_key != coil.name || !contains(manufacturer_key, coil.manufacturer) ||
        !contains_any(model_key, coil.models))
      continue;
    result.coil_known = true;
    for (const auto &configuration : coil.measured) {
      if (elements_key == configuration.elements) {
        result.dof = configuration.dof;
        result.basis = configuration.source;
        return result;
      }
    }
    if (coil.clusters != nullptr) {
      std::vector<std::string> clusters;
      const std::optional<size_t> count = count_element_clusters(elements_key, clusters);
      if (count.has_value()) {
        result.dof = 2.0 * default_type(count.value());
        result.hypothesised = true;
        result.basis = std::string("the element cluster") + (count.value() == 1 ? " active is " : "s active are ") +
                       join(clusters, ", ") + "; on this coil " + coil.clusters;
        return result;
      }
      result.cluster_rule_defeated = true;
    }
    // No match: report what this coil could have been recognised as, so that a header reporting a
    //   selection one group away from a tabulated one is diagnosable from the message alone.
    for (const auto &configuration : coil.measured)
      result.known_configurations.push_back(configuration.elements);
    return result;
  }
  return result;
}

// Classify the coil combination, together with the phrase describing the evidence for it.
combination_t combination_mode(const KeyValues &keyval, std::string &evidence) {
  // The CMRR multi-band sequences' own SENSE1 combination takes precedence over the protocol's
  //   coil combine mode; see the discussion in coils.h.
  const std::vector<std::string> sense1 = find_metadata_tokens(keyval, descriptive_keys, {"SENSE1"});
  if (!sense1.empty()) {
    evidence = "image metadata declare SENSE1 coil combination (" + join(sense1, "; ") +
               "), which the CMRR multi-band sequences perform in place of the coil combination"
               " selected in the protocol, and which yields Rician magnitude data";
    return combination_t::PHASE_SENSITIVE;
  }
  const std::optional<std::string> method = metadata_string(keyval, "CoilCombinationMethod");
  if (!method.has_value()) {
    evidence = "the image header does not report how the receive channels were combined"
               " (no \"CoilCombinationMethod\" field, and no declaration of SENSE1 combination)";
    return combination_t::UNKNOWN;
  }
  // BIDS constrains neither the vocabulary nor the spelling of this field. dcm2niix writes exactly
  //   "Sum of Squares" or "Adaptive Combine" (from the Siemens CSA "ucCoilCombineMode"), but the
  //   BIDS specification itself describes the common case as "root-sum-of-squares (rSOS)", so the
  //   abbreviations a hand-written or third-party sidecar might use are admitted alongside.
  //   Comparison is against alphanumeric(), so punctuation, spacing and case are immaterial.
  static const std::vector<std::string> sum_of_squares{"SUMOFSQUARES", "SOS", "ROOTSUMOFSQUARES", "RSOS", "RSS"};
  // Any combination performed on the complex data before the magnitude is taken; the result is a
  //   single complex image, hence Rician, whatever the number of channels combined.
  static const std::vector<std::string> phase_sensitive{
      "ADAPTIVECOMBINE", "ADAPTIVECOMBINATION", "ADAPTIVE", "SENSE1", "ROEMER"};
  const std::string key = alphanumeric(method.value());
  if (std::find(sum_of_squares.begin(), sum_of_squares.end(), key) != sum_of_squares.end()) {
    evidence = "\"CoilCombinationMethod\" is \"" + method.value() + "\"";
    return combination_t::SUM_OF_SQUARES;
  }
  if (std::find(phase_sensitive.begin(), phase_sensitive.end(), key) != phase_sensitive.end()) {
    evidence = "\"CoilCombinationMethod\" is \"" + method.value() +
               "\", a phase-sensitive combination that leaves the magnitude data Rician"
               " (Sakaie & Lowe 2017 fit an effective channel count of 1.03 +/- 0.01 to"
               " adaptive-combine data from a 12-channel coil)";
    return combination_t::PHASE_SENSITIVE;
  }
  evidence = "\"CoilCombinationMethod\" is \"" + method.value() +
             "\", which is not a value from which the noise distribution can be determined";
  return combination_t::UNKNOWN;
}

} // namespace

CoilInference infer_noise_dof(const Header &H) {
  const KeyValues &keyval = H.keyval();
  CoilInference result;

  // Reported whenever a channel count greater than one is ultimately applied, so it is gathered
  //   irrespective of which branch below is taken.
  const std::optional<default_type> acceleration = metadata_float(keyval, "ParallelReductionFactorInPlane");
  if (acceleration.has_value() && acceleration.value() > 1.0)
    result.acceleration = acceleration;

  std::string evidence;
  switch (combination_mode(keyval, evidence)) {
  case combination_t::UNKNOWN:
    result.evidence = evidence;
    return result;
  case combination_t::PHASE_SENSITIVE:
    // Exactly one, not the 1.03 that Sakaie & Lowe measured: the model here is that a
    //   phase-sensitive combination yields a single complex image, whose magnitude is Rician by
    //   construction. Their 1.03 +/- 0.01 corroborates that model rather than refining it, and
    //   adopting it would substitute false precision for the exact Rician case.
    result.dof = 2.0;
    result.evidence = evidence;
    return result;
  case combination_t::SUM_OF_SQUARES:
    break;
  }

  // Sum-of-squares combination established; the channel count now has to be found, which requires
  //   the coil, the system on which it was used, and the elements of it that were active all to be
  //   identifiable from the header. See CoilDefinition for why the system is as necessary as the
  //   coil: a tabulated count belongs to hardware and reconstruction together, not to a coil name.
  const std::optional<std::string> manufacturer = metadata_string(keyval, "Manufacturer");
  // BIDS spells the field with the plural possessive; DICOM (0008,1090) and some converters that
  //   follow it do not, and the field is too central to this determination to lose to a spelling.
  std::string model_key = "ManufacturersModelName";
  std::optional<std::string> model = metadata_string(keyval, model_key);
  if (!model.has_value()) {
    model_key = "ManufacturerModelName";
    model = metadata_string(keyval, model_key);
  }
  const std::optional<std::string> coil_name = metadata_string(keyval, "ReceiveCoilName");
  // "CoilString" is dcm2niix's own field, written from the Siemens CSA coil identifier; on data
  //   where the element selection did not reach "ReceiveCoilActiveElements" it may carry that
  //   selection instead (e.g. "T:HEA;HEP"). It may equally carry only the coil name again, in
  //   which case it simply fails to match any tabulated selection.
  std::string elements_source = "ReceiveCoilActiveElements";
  std::optional<std::string> elements = metadata_string(keyval, elements_source);
  if (!elements.has_value()) {
    elements_source = "CoilString";
    elements = metadata_string(keyval, elements_source);
  }

  // Reported together rather than one at a time: a header missing one of these is usually missing
  //   several, and a user who has to rerun the command to discover each in turn is being told the
  //   same thing four times over.
  std::vector<std::string> missing;
  if (!manufacturer.has_value())
    missing.push_back("\"Manufacturer\"");
  if (!model.has_value())
    missing.push_back("\"ManufacturersModelName\"");
  if (!coil_name.has_value())
    missing.push_back("\"ReceiveCoilName\"");
  if (!elements.has_value())
    missing.push_back("\"ReceiveCoilActiveElements\" (nor \"CoilString\")");
  if (!missing.empty()) {
    result.evidence = evidence +
                      ", but the image header lacks metadata necessary to identify the receive coil"
                      " and the system on which it was used (no " +
                      join(missing, ", nor ") + "), so the effective number of channels combined cannot be looked up";
    return result;
  }

  // The system and coil as the messages below name them, so that the user is shown exactly what
  //   was matched (or not matched) against the table.
  const std::string hardware = "receive coil \"" + coil_name.value() + "\" on a \"" + manufacturer.value() + "\" \"" +
                               model.value() + "\" (\"" + model_key + "\")";

  const Lookup lookup = lookup_dof(manufacturer.value(), model.value(), coil_name.value(), elements.value());
  if (lookup.dof.has_value()) {
    result.dof = lookup.dof;
    result.hypothesised = lookup.hypothesised;
    result.evidence = evidence + ", and for " + hardware + " with active elements \"" + elements.value() + "\" (\"" +
                      elements_source + "\") the " + (lookup.hypothesised ? "hypothesised" : "tabulated measured") +
                      " effective channel count is " + str(0.5 * lookup.dof.value(), 4) + " (" + lookup.basis + ")";
    return result;
  }
  if (lookup.cluster_rule_defeated) {
    result.evidence = evidence + ", and " + hardware + " is present in the built-in table, but its active elements \"" +
                      elements.value() + "\" (\"" + elements_source +
                      "\") do not enumerate the element clusters that were active: they name regions of the receive"
                      " array, or are not in the form this coil's element groups take, so the number of clusters"
                      " combined cannot be determined from them";
    return result;
  }
  if (lookup.coil_known) {
    result.evidence =
        evidence + ", and " + hardware + " is present in the built-in table, but its active elements \"" +
        elements.value() + "\" (\"" + elements_source +
        "\") are not a configuration for which an effective channel count has been measured" +
        (lookup.known_configurations.empty() ? std::string()
                                             : std::string(" (measured configurations for this coil: \"") +
                                                   join(lookup.known_configurations, "\", \"") + "\")");
    return result;
  }
  result.evidence = evidence + ", but " + hardware + " (active elements \"" + elements.value() +
                    "\") is not present in the built-in table of receive coils for which an effective"
                    " channel count is available";
  return result;
}

namespace {
// Render a channel count together with the degrees of freedom it implies, so that every message
//   states both and the factor of two relating them is never left for the reader to supply.
std::string dof_and_channels(const default_type dof) {
  const default_type channels = 0.5 * dof;
  return str(dof, 4) + " degree" + (dof == 1.0 ? "" : "s") + " of freedom (" + str(channels, 4) + " effective channel" +
         (channels == 1.0 ? "" : "s") + ")";
}

// Whether two real-valued channel counts are to be regarded as the same value. Both sides may be
//   non-integral, and the user's value reaches this point through decimal text, so exact equality
//   is not a usable test.
bool same_value(const default_type a, const default_type b) {
  return std::abs(a - b) <= 1.0e-4 * std::max(default_type(1), std::max(std::abs(a), std::abs(b)));
}
} // namespace

default_type
resolve_noise_dof(const Header &H, const std::optional<default_type> &user_value, const std::string &user_option) {
  const CoilInference inference = infer_noise_dof(H);

  // An explicit instruction outranks an inference drawn from metadata: the header describes the
  //   data as reconstructed, which need not be the data as they now stand (channels may have been
  //   recombined, or the series assembled from more than one acquisition), and the user may know
  //   an effective channel count that no metadata report. The inference is nonetheless performed,
  //   so that a disagreement can be surfaced rather than passing unnoticed.
  if (user_value.has_value()) {
    // Clamped before it is reported, so that every message downstream quotes the value that will
    //   actually be used. A request below the bound is not refused outright -- it is a plausible
    //   thing to arrive at from a data-driven fit, and the clamped model remains usable -- but it
    //   is never applied silently.
    const default_type requested = user_value.value();
    const default_type applied = clamp_dof(requested);
    if (!same_value(applied, requested)) {
      WARN("The value supplied via -" + user_option + " corresponds to " + dof_and_channels(requested) +
           ", which is fewer than the magnitude noise model admits:"
           " fewer than one degree of freedom cannot arise from summing squares of Gaussian variates."
           " Proceeding with " +
           dof_and_channels(applied) + " (the half-Gaussian limit) instead of the value specified");
    }
    // Whether the header-derived figure was measured or merely hypothesised decides both the
    //   wording and the severity of what follows: a value of the user's own departing from a
    //   measurement is worth an alert, whereas one departing from this software's hypothesis is
    //   the expected way for a user who knows their own effective count to use these options.
    const std::string derivation = inference.hypothesised ? "hypothesised from" : "inferred from";
    if (!inference.dof.has_value()) {
      INFO("The " + dof_and_channels(applied) + " specified via -" + user_option +
           " could not be corroborated from the image header metadata: " + inference.evidence);
    } else if (same_value(inference.dof.value(), applied)) {
      INFO("The " + dof_and_channels(applied) + " specified via -" + user_option + " agrees with what is " +
           derivation + " the image header metadata: " + inference.evidence);
    } else if (inference.hypothesised) {
      INFO("The " + dof_and_channels(applied) + " specified via -" + user_option + " differs from the " +
           dof_and_channels(inference.dof.value()) +
           " that would have been hypothesised from the image header metadata (" + inference.evidence +
           "); proceeding with the value specified on the command line, the hypothesis being the weaker basis"
           " of the two");
    } else {
      WARN("The " + dof_and_channels(applied) + " specified via -" + user_option + " is inconsistent with the " +
           dof_and_channels(inference.dof.value()) + " inferred from the image header metadata (" + inference.evidence +
           "); proceeding with the value specified on the command line");
    }
    return applied;
  }

  if (!inference.dof.has_value()) {
    WARN("The image header metadata do not contain all of the information required to configure"
         " automatically the effective number of receive channels combined into these magnitude data: " +
         inference.evidence +
         ". Proceeding with a Rician noise model (one effective channel; 2 degrees of freedom)."
         " If the channels were instead combined by sum of squares, the noise floor and the"
         " variance-stabilising transform will both be misparameterised; supply the count using"
         " -noise_channels (or the degrees of freedom using -noise_dof). What is required there is the"
         " EFFECTIVE number of channels, which is neither the number of elements in the receive coil nor"
         " the number of receiver channels: elements may be combined into fewer channels before"
         " reconstruction, and correlation of the noise between those channels together with overlap of"
         " their sensitivity profiles means that they do not contribute independent degrees of freedom."
         " The effective count is consequently smaller than the element count, often by a large factor,"
         " and need not be a whole number: 3.76 has been measured for a nominally 12-element head coil");
    return default_type(2);
  }

  const default_type num_dof = clamp_dof(inference.dof.value());
  INFO("Inferred " + dof_and_channels(num_dof) + " from image header metadata: " + inference.evidence +
       "; applying a " + (same_value(num_dof, 2.0) ? std::string("Rician") : std::string("non-central chi")) +
       " noise model");

  // Where the count was not measured for this coil but obtained by counting the element clusters
  //   that the header reports as active, the user is told so in as many words. The value is very
  //   much better founded than the element count it replaces, but it remains a hypothesis about
  //   how a receive array behaves, and it parameterises the noise floor and the
  //   variance-stabilising transform exactly as a measured value would; a user given no notice
  //   would have no way to tell the two apart in their results.
  if (inference.hypothesised) {
    WARN("The " + dof_and_channels(num_dof) +
         " applied to these data are HYPOTHESISED, not measured: no effective channel count has been"
         " measured for this receive coil, and the value applied is instead the number of element"
         " clusters that the image header reports as active (" +
         inference.evidence +
         "), on the hypothesis that each cluster of the array contributes one independent channel"
         " to the sum-of-squares combination. That hypothesis is untested: on the one coil for which"
         " a measurement exists, 3.76 effective channels were measured where the element string names"
         " only two groups. A count obtained this way may therefore err in either direction, and the"
         " noise floor and the variance-stabilising transform err with it. Supply -noise_channels or"
         " -noise_dof to impose a value of your own, which suppresses this warning");
  }

  // A count applied from the table needs no apology as far as the coil goes. What does need saying
  //   is that it belongs to some particular acquisition: the effective count is a property of the
  //   coil together with the scanner configuration and the reconstruction, not of the coil alone.
  //   Parallel imaging is the case in which that caveat bites hardest, because it correlates the
  //   channels further and makes the effective count vary over space, so no single scalar --
  //   measured, hypothesised or otherwise -- is strictly correct.
  if (!same_value(num_dof, 2.0) && inference.acceleration.has_value()) {
    WARN(std::string(inference.hypothesised
                         ? "The effective channel count applied to these data counts the clusters of the receive"
                           " array as though each were an independent channel of an unaccelerated acquisition,"
                           " whereas"
                         : "The tabulated effective channel count applied to these data was measured on a different"
                           " acquisition, whereas") +
         " the image header reports in-plane parallel imaging (acceleration factor " +
         str(inference.acceleration.value(), 4) +
         "). Parallel imaging correlates the noise between channels and makes the effective degrees of"
         " freedom vary over space, so the value applied need not transfer and a single scalar is in"
         " any case an approximation. Supply -noise_channels or -noise_dof to impose a different value,"
         " which suppresses this warning");
  }

  return num_dof;
}

} // namespace MR::Denoise::Precondition::NoiseModel
