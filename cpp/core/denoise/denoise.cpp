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

#include <optional>

#include "axes.h"
#include "filter/smooth.h"
#include "transform.h"

namespace MR::Denoise {

using namespace App;

const char *patent_description =
    "Estimation of the noise level of an image series"
    " based on the Marchenk--Pastur distribution (-estimator exp1/exp2)"
    " is protected by the following patent: \n"
    "US10698065B2."
    " System, method and computer accessible medium for noise estimation, noise removal and gibbs ringing removal."
    " Dmitry Novikov, Jelle Veraart, Els Fieremans."
    " Contact: https://tov.med.nyu.edu/about/contact-us/";

const char *first_step_description =
    "Important note:"
    " image denoising must be performed as the first step of the image processing pipeline."
    " The routine will not operate correctly if interpolation or smoothing"
    " has been applied to the data prior to denoising.";

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
                                      " this will select complex float32 or complex float64 datatypes.")
                               + Argument("float32/float64").type_choice(dtypes);

const Option decomposition_option = Option("decomposition",
                                           "Method used for the decomposition of the data in each patch;"
                                           " options are: " + join(decompositions, ", ") +
                                           " (default: BDCSVD)")
                                    + Argument("choice").type_choice(decompositions);

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

Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type nan_to_zero,
                                 const noise_pad_type pad,
                                 const noise_smooth_type smooth) {
  Header H(in);
  if (pad == noise_pad_type::PAD) {
    // Just pad by 2 voxels at all edges;
    //   that should make cubic interpolation safe
    for (ssize_t axis = 0; axis != 3; ++axis)
      H.size(axis) += 4;
    H.transform().translation() = Transform(H).voxel2scanner * Eigen::Vector3d{-2.0, -2.0, -2.0};
  }
  Image<float> out = Image<float>::scratch(H, "Conditioned version of \"" + std::string(in.name()) + "\"");
  for (auto l = Loop(out)(out); l; ++l) {
    if (pad == noise_pad_type::PAD) {
      for (ssize_t axis = 0; axis != 3; ++axis)
        in.index(axis) = std::max(ssize_t(0), std::min(in.size(axis) - 1, out.index(axis) - 2));
    } else {
      assign_pos_of(out).to(in);
    }
    if (nan_to_zero == noise_impute_type::NAN_TO_ZERO)
      out.value() = std::isfinite(in.value()) ? in.value() : 0.0F;
    else
      out.value() = in.value();
  }
  if (smooth == noise_smooth_type::SMOOTH) {
    Filter::Smooth smooth_filter(out);
    smooth_filter(out);
  }
  return out;
}

Image<float> import_vst_noise_map(const App::ParsedArgument &arg, const Header &H_spatial) {
  // Detect whether a scalar value or an image path was provided,
  //   matching the dispatch used by Estimator::make_estimator().
  std::optional<default_type> scalar_value;
  try {
    scalar_value = default_type(arg);
  } catch (Exception &) {
  }
  Image<float> in;
  if (scalar_value.has_value()) {
    if (scalar_value.value() <= 0.0)
      throw Exception("Noise level provided via -noise_in must be greater than zero");
    Header H(H_spatial);
    H.ndim() = 3;
    H.reset_intensity_scaling();
    H.datatype() = DataType::Float32;
    H.datatype().set_byte_order_native();
    in = Image<float>::scratch(H, "Spatially-constant noise level map");
    for (auto l = Loop(in)(in); l; ++l)
      in.value() = float(scalar_value.value());
  } else {
    in = Image<float>::open(std::string(arg));
    if (in.ndim() != 3)
      throw Exception("Noise level image provided via -noise_in must be 3-dimensional");
  }
  return condition_noise_map(in, noise_impute_type::NONE, noise_pad_type::PAD, noise_smooth_type::NONE);
}

} // namespace MR::Denoise
