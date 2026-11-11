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
    " (configurable via the -vst_method option, and parameterised by -noise_channels or -noise_dof)"
    " that renders the data approximately Gaussian and homoscedastic prior to PCA."
    " The distribution is fixed by a single quantity,"
    " expressible either as the effective number of receive channels N combined by sum of squares"
    " or as the 2N degrees of freedom of the resulting distribution;"
    " the two options above are the two ways of stating it, and are mutually exclusive."
    " Note that N is the number of channels that were effectively combined,"
    " which is not the number of elements in the receive coil:"
    " elements may be combined into fewer channels before reconstruction,"
    " and correlation of the noise between channels together with overlap of their sensitivity profiles"
    " means that those channels do not contribute independent degrees of freedom."
    " N is consequently smaller than the element count, often by a large factor, and need not be a whole number."
    " Where neither option is provided,"
    " the image header key-values are interrogated for the manner in which the receive channels were combined,"
    " and, where that combination was by sum of squares,"
    " for the effective channel count measured for that coil and element selection"
    " (resolved against a table of measured values built into this software);"
    " a warning is issued, and Rician statistics assumed,"
    " if no such determination can be made."
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
