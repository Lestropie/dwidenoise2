/* Required Notice: Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;
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

#include "axes.h"

namespace MR::Denoise {

using namespace App;

const char *first_step_description =
    "Important note:"
    " image denoising must be performed as the first step of the image processing pipeline."
    " The routine will not operate correctly if interpolation or smoothing"
    " has been applied to the data prior to denoising.";

const char *non_gaussian_noise_description =
    "Note that this function does not correct for non-Gaussian noise biases"
    " present in magnitude-reconstructed MRI images."
    " If available, including the MRI phase data as part of a complex input image"
    " can reduce such non-Gaussian biases.";

const char *filter_description =
    "By default, optimal value shrinkage based on minimisation of the Frobenius norm "
    "will be used to attenuate eigenvectors based on the estimated noise level. "
    "Hard truncation of sub-threshold components and inclusion of supra-threshold components"
    "---which was the behaviour of the dwidenoise command in version 3.0.x---"
    "can be activated using -filter truncate."
    "Alternatively, optimal truncation as described in Gavish and Donoho 2014 "
    "can be utilised by specifying -filter optthresh.";

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
    "'gaussian': A Gaussian distribution with FWHM equal to twice the voxel size, "
    "such that decompisitions centred more closely to the output voxel have greater influence; "
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
                                      " this will select complex float32 or complex float64 datatypes.") +
                               Argument("float32/float64").type_choice(dtypes);

} // namespace MR::Denoise
