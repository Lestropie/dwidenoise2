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

#include "denoise/precondition/noise_model/coil_database.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Entries of the database of known receive coils.
// See coil_database.h for the form each entry takes, the two kinds of value it may hold, what is
//   required to add one, and the values that are deliberately NOT admitted.
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

} // namespace MR::Denoise::Precondition::NoiseModel
