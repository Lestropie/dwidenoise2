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
#include <atomic>
#include <cmath>
#include <limits>
#include <optional>

#include "algo/loop.h"
#include "algo/threaded_loop.h"
#include "axes.h"
#include "denoise/mask.h"
#include "denoise/metadata.h"
#include "exception.h"
#include "filter/smooth.h"
#include "math/math.h"
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

// Map a position in the (possibly padded) output onto the corresponding position in the source,
//   clamping into range so that the padded region replicates the nearest edge value. Padding may
//   be asymmetric (sized per edge to reach a coverage target), so the offset is per-axis.
inline void assign_padded_pos(Image<float> &out,
                              Image<float> &in,
                              const noise_pad_type pad,
                              const std::array<ssize_t, 3> &pad_lo) {
  if (pad == noise_pad_type::PAD) {
    for (ssize_t axis = 0; axis != 3; ++axis)
      in.index(axis) = std::max(ssize_t(0), std::min(in.size(axis) - 1, out.index(axis) - pad_lo[axis]));
  } else {
    assign_pos_of(out, 0, 3).to(in);
  }
}

// Copy a map into a scratch image, with either padded edge-clamped sampling (pad) or a direct
//   positional copy, sanitising as it goes. Holds its own copy of the source image so each thread
//   samples it independently.
// The sanitisation is what establishes the single sentinel the rest of the conditioning relies on:
//   for a noise level map, anything failing noise_level_is_valid() -- NaN, an infinity, a zero, a
//   negative value -- becomes NaN. Estimated maps do not otherwise contain zeros once the former
//   NaN-to-zero conversion is gone; this defends chiefly against a user-supplied -noise_in image
//   and against float underflow. The NAN_TO_ZERO form is the legacy behaviour, retained for the
//   rank-per-mm density alone.
class ConditionNoiseMapFunctor {
public:
  ConditionNoiseMapFunctor(Image<float> &in,
                           const noise_impute_type impute,
                           const noise_pad_type pad,
                           const std::array<ssize_t, 3> &pad_lo)
      : in(in), impute(impute), pad(pad), pad_lo(pad_lo) {}
  void operator()(Image<float> &out) {
    assign_padded_pos(out, in, pad, pad_lo);
    const float value = in.value();
    if (impute == noise_impute_type::NAN_TO_ZERO)
      out.value() = std::isfinite(value) ? value : 0.0F;
    else
      out.value() = noise_level_is_valid(value) ? value : std::numeric_limits<float>::quiet_NaN();
  }

private:
  Image<float> in;
  noise_impute_type impute;
  noise_pad_type pad;
  std::array<ssize_t, 3> pad_lo;
};

// sigma -> x = log(sigma^2) = 2 log(sigma), and back. Invalid voxels carry NaN through both.
class ToLogNoiseFunctor {
public:
  void operator()(Image<float> &image) {
    const float sigma = image.value();
    image.value() = noise_level_is_valid(sigma) ? float(2.0 * std::log(default_type(sigma)))
                                                : std::numeric_limits<float>::quiet_NaN();
  }
};
class FromLogNoiseFunctor {
public:
  void operator()(Image<float> &image) {
    const float x = image.value();
    // A voxel still invalid here survived every fill mechanism, and only the RETAIN policy lets one
    //   through; exp() of NaN preserves it rather than manufacturing a value.
    image.value() = std::isfinite(x) ? float(std::exp(0.5 * default_type(x))) : x;
  }
};

// The two halves of a normalised convolution: split the value image into the numerator product
//   w .* x and the 0/1 weight field w, then recombine them after each has been smoothed.
class NormalisedConvolutionSplitFunctor {
public:
  void operator()(Image<float> &x, Image<float> &numerator, Image<float> &denominator) {
    const float value = x.value();
    const bool valid = std::isfinite(value);
    // The zero written at an invalid voxel is a placeholder in the NUMERATOR PRODUCT, not an
    //   imputed value: the same voxel contributes zero to the denominator, so it never survives the
    //   division. It is chosen because it is finite, exactly representable and annihilates under
    //   multiplication; any other finite constant would give the same answer. It must not be read
    //   as imputing sigma^2 == 1 at holes.
    numerator.value() = valid ? value : 0.0F;
    denominator.value() = valid ? 1.0F : 0.0F;
  }
};
class NormalisedConvolutionCombineFunctor {
public:
  explicit NormalisedConvolutionCombineFunctor(const default_type epsilon) : epsilon(epsilon) {}
  void operator()(Image<float> &x, Image<float> &numerator, Image<float> &denominator) {
    const default_type weight = default_type(denominator.value());
    // No valid support anywhere within the kernel: leave the voxel invalid for the imputation
    //   passes that follow, rather than dividing two near-zeros.
    x.value() = weight > epsilon ? float(default_type(numerator.value()) / weight)
                                 : std::numeric_limits<float>::quiet_NaN();
  }

private:
  default_type epsilon;
};

// Fill invalid voxels from the valid ones among their 26 immediate neighbours, weighted by a
//   Gaussian whose full width at half maximum is one mean voxel spacing of the map's own grid.
//   Isotropic in millimetres, and therefore anisotropic in voxels wherever the grid is.
//
// The kernel is deliberately very tight. With FWHM equal to the mean spacing the standard deviation
//   is ~0.425 voxels, so a face neighbour weighs exp(-2.77) ~ 0.063 against an edge neighbour's
//   0.0039 and a corner's 0.00024: an imputed value is essentially the weighted mean of its face
//   neighbours, and one pass erodes a hole by one voxel. That is the intended behaviour -- a hole of
//   half-width k voxels closes in k passes -- and is what removes any need to enlarge the kernel to
//   cover the largest hole a map might contain.
//
// Double-buffered: reads the previous pass's image and writes a new one. Updating in place would
//   make the result depend on traversal order (a voxel filled early in a sweep would feed a later
//   one within the same sweep) and would not be thread-safe under ThreadedLoop.
class ImputeNoiseFunctor {
public:
  ImputeNoiseFunctor(Image<float> &previous, const std::array<default_type, 3> &spacing, std::atomic<size_t> &filled)
      : previous(previous), filled(&filled) {
    const default_type fwhm = std::cbrt(spacing[0] * spacing[1] * spacing[2]);
    const default_type stdev = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    const default_type denominator = 2.0 * stdev * stdev;
    for (ssize_t dz = -1; dz != 2; ++dz) {
      for (ssize_t dy = -1; dy != 2; ++dy) {
        for (ssize_t dx = -1; dx != 2; ++dx) {
          const default_type distance_sq = Math::pow2(default_type(dx) * spacing[0])  //
                                           + Math::pow2(default_type(dy) * spacing[1]) //
                                           + Math::pow2(default_type(dz) * spacing[2]);
          weights[tap_index(dx, dy, dz)] = std::exp(-distance_sq / denominator);
        }
      }
    }
  }
  void operator()(Image<float> &out) {
    const std::array<ssize_t, 3> pos{out.index(0), out.index(1), out.index(2)};
    for (ssize_t axis = 0; axis != 3; ++axis)
      previous.index(axis) = pos[axis];
    const float centre = previous.value();
    if (std::isfinite(centre)) {
      out.value() = centre;
      return;
    }
    default_type sum = 0.0;
    default_type sum_weights = 0.0;
    for (ssize_t dz = -1; dz != 2; ++dz) {
      const ssize_t z = pos[2] + dz;
      if (z < 0 || z >= previous.size(2))
        continue;
      for (ssize_t dy = -1; dy != 2; ++dy) {
        const ssize_t y = pos[1] + dy;
        if (y < 0 || y >= previous.size(1))
          continue;
        for (ssize_t dx = -1; dx != 2; ++dx) {
          const ssize_t x = pos[0] + dx;
          if (x < 0 || x >= previous.size(0))
            continue;
          if (dx == 0 && dy == 0 && dz == 0)
            continue;
          previous.index(0) = x;
          previous.index(1) = y;
          previous.index(2) = z;
          const default_type value = default_type(previous.value());
          if (!std::isfinite(value))
            continue;
          const default_type weight = weights[tap_index(dx, dy, dz)];
          sum += weight * value;
          sum_weights += weight;
        }
      }
    }
    if (sum_weights > 0.0) {
      out.value() = float(sum / sum_weights);
      filled->fetch_add(1, std::memory_order_relaxed);
    } else {
      out.value() = std::numeric_limits<float>::quiet_NaN();
    }
  }

private:
  static ssize_t tap_index(const ssize_t dx, const ssize_t dy, const ssize_t dz) {
    return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
  }
  Image<float> previous;
  std::array<default_type, 27> weights;
  std::atomic<size_t> *filled;
};

// Last-resort fill from the patch-derived upper bounds (Estimator::Result::sigma2_bound, exported
//   as a noise level). Applied only to voxels that no imputation pass could reach, and only in the
//   log domain the surrounding map is already in.
class BoundFillFunctor {
public:
  BoundFillFunctor(Image<float> &bound,
                   const noise_pad_type pad,
                   const std::array<ssize_t, 3> &pad_lo,
                   std::atomic<size_t> &filled)
      : bound(bound), pad(pad), pad_lo(pad_lo), filled(&filled) {}
  void operator()(Image<float> &out) {
    if (std::isfinite(out.value()))
      return;
    assign_padded_pos(out, bound, pad, pad_lo);
    const float sigma = bound.value();
    if (!noise_level_is_valid(sigma))
      return;
    out.value() = float(2.0 * std::log(default_type(sigma)));
    filled->fetch_add(1, std::memory_order_relaxed);
  }

private:
  Image<float> bound;
  noise_pad_type pad;
  std::array<ssize_t, 3> pad_lo;
  std::atomic<size_t> *filled;
};

// Copy one map into another of identical dimensions.
class CopyFloatFunctor {
public:
  void operator()(Image<float> &destination, Image<float> &source) { destination.value() = source.value(); }
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
// Number of voxels of a map holding no usable value: NaN at any point after sanitisation, and
//   equivalently anything failing noise_level_is_valid() before it. Serial: the maps this is
//   applied to are the sub-sampled grid, and the count is needed as a scalar between passes.
size_t count_invalid_noise_levels(Image<float> &image) {
  size_t count = 0;
  for (auto l = Loop(image)(image); l; ++l) {
    if (!noise_level_is_valid(image.value()))
      ++count;
  }
  return count;
}

// Number of voxels of a LOG-domain map holding no usable value. The predicate differs from the one
//   above by necessity: x = log(sigma^2) is legitimately zero or negative, so only finiteness
//   distinguishes a hole there.
size_t count_nonfinite(Image<float> &image) {
  size_t count = 0;
  for (auto l = Loop(image)(image); l; ++l) {
    if (!std::isfinite(image.value()))
      ++count;
  }
  return count;
}

// Normalised convolution (Knutsson-Westin) of a log-domain map holding NaN at invalid voxels, at
//   the given full width at half maximum in millimetres.
//
// Both convolutions are performed on images that are finite everywhere, so the smoothing filter's
//   own missing-data fallback never triggers and the result does not depend on it. That matters:
//   MRtrix3's smoother renormalises over the finite neighbours of each ONE-DIMENSIONAL pass
//   (Filter::SmoothFunctor1D::operator(), Adapter::Gaussian1D::value()), which applied separably is
//   not 3D normalised convolution -- a voxel filled during the x pass is treated as full-weight
//   data by the y pass, so the result silently extrapolates between passes and depends on axis
//   order. Here the weight field is fixed and the Gaussian separable, so each convolution is an
//   exact separable 3D convolution and their ratio is the exact 3D normalised convolution,
//   independent of axis order.
//
// The 0/1 weight field is a placeholder for a precision weight w = nu/2, the inverse variance of
//   log(sigma^2) with nu the number of noise components the patch pooled (Result::cutoff_p, the
//   -rank_pcanonzero export). That would turn the operation from an unweighted geometric mean into
//   the best linear unbiased estimate of the log-noise field; the formulation is chosen so that it
//   is a one-line substitution when that becomes available.
//
// Deliberately written free of any dwidenoise2-specific dependency -- a value image, a weight
//   image and a width -- so that it can be lifted upstream verbatim if MRtrix3's smoothing filter
//   ever grows an optional weight image.
void normalised_convolution(Image<float> &x, const default_type fwhm) {
  // Filter::Smooth is parameterised by the standard deviation of the Gaussian in millimetres, one
  //   value per spatial axis; the schedule specifies the FWHM of a spatially isotropic Gaussian, so
  //   the same converted width is imposed on all three axes irrespective of the (possibly
  //   anisotropic) voxel size of the grid on which the map was estimated.
  const default_type stdev = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  Header H(x);
  Image<float> numerator = Image<float>::scratch(H, "Numerator of normalised convolution");
  Image<float> denominator = Image<float>::scratch(H, "Denominator of normalised convolution");
  ThreadedLoop(x).run(NormalisedConvolutionSplitFunctor(), x, numerator, denominator);
  for (Image<float> *image : {&numerator, &denominator}) {
    Filter::Smooth smooth_filter(*image);
    smooth_filter.set_stdev(stdev);
    smooth_filter(*image);
  }
  // The normalised Gaussian sums to at most one, so a denominator this small marks a voxel with no
  //   valid support anywhere within the kernel rather than a poorly-covered one.
  constexpr default_type coverage_epsilon = 1e-6;
  ThreadedLoop(x).run(NormalisedConvolutionCombineFunctor(coverage_epsilon), x, numerator, denominator);
}

} // namespace

Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type impute,
                                 const noise_pad_type pad,
                                 const default_type smooth_fwhm,
                                 const Header *coverage,
                                 Image<float> *bound,
                                 const noise_unfilled_type unfilled,
                                 const std::string &context) {
  assert(smooth_fwhm >= 0.0);
  Header H(in);
  // Imputation weights are formed from the map's OWN voxel spacings, which for an iterative
  //   schedule differ from one iteration to the next; padding does not change them.
  const std::array<default_type, 3> spacing{H.spacing(0), H.spacing(1), H.spacing(2)};
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
  ThreadedLoop(out).run(ConditionNoiseMapFunctor(in, impute, pad, pad_lo), out);
  // The rank-per-mm density is not a noise level: zero is a legitimate value for it, and the log
  //   domain is meaningless. Padding and the NaN-to-zero conversion are all it wants from here.
  if (impute == noise_impute_type::NAN_TO_ZERO)
    return out;

  // Report the number of voxels the map itself holds no estimate at, before padding replicates any
  //   of them outward. The threshold escalating INFO to WARN is a property of the map's own grid.
  report_imputed_noise_levels(count_invalid_noise_levels(in), Header(in), context);

  // Everything from here to the inverse transform below operates on x = log(sigma^2).
  ThreadedLoop(out).run(ToLogNoiseFunctor(), out);

  size_t remaining = count_nonfinite(out);
  ssize_t passes = 0;
  if (smooth_fwhm > 0.0) {
    // Pass zero is the requested smoothing, which -- being a normalised convolution -- fills every
    //   hole with any valid support inside its (much wider) kernel as a by-product. Passes one
    //   onward are the tight imputation kernel, which erodes whatever remains one voxel at a time.
    normalised_convolution(out, smooth_fwhm);
    remaining = count_nonfinite(out);
    ++passes;
  }
  size_t imputed = 0;
  Image<float> previous;
  while (remaining > 0) {
    if (!previous.valid())
      previous = Image<float>::scratch(H, "Previous pass of noise level map imputation");
    ThreadedLoop(previous).run(CopyFloatFunctor(), previous, out);
    std::atomic<size_t> filled{0};
    ThreadedLoop(out).run(ImputeNoiseFunctor(previous, spacing, filled), out);
    const size_t filled_count = filled.load(std::memory_order_acquire);
    // A pass that fills nothing means no hole has a valid neighbour; since holes shrink
    //   monotonically, that means the map holds no valid voxel at all. Break rather than spin:
    //   the per-voxel fallback below is what such a map is left to.
    if (filled_count == 0)
      break;
    assert(filled_count <= remaining);
    remaining -= filled_count;
    imputed += filled_count;
    ++passes;
  }
  if (passes > 0) {
    INFO(context + ": conditioning completed in " + str(passes) + " pass(es), " + str(imputed) +
         " voxel(s) filled by imputation");
  }

  if (remaining > 0 && bound != nullptr && bound->valid()) {
    std::atomic<size_t> filled{0};
    ThreadedLoop(out).run(BoundFillFunctor(*bound, pad, pad_lo, filled), out);
    const size_t filled_count = filled.load(std::memory_order_acquire);
    assert(filled_count <= remaining);
    remaining -= filled_count;
    if (filled_count > 0) {
      INFO(context + ": " + str(filled_count) +
           " voxel(s) beyond the reach of any valid estimate were filled from the patch-derived"
           " upper bound on the noise level");
    }
  }

  if (remaining > 0) {
    const std::string message = context + " holds " + str(remaining) +
                                " voxel(s) at which no noise level could be established:"
                                " no valid estimate anywhere in the map is able to reach them,"
                                " and no patch-derived bound is available for them";
    if (unfilled == noise_unfilled_type::THROW)
      throw Exception(message);
    WARN(message + "; they are left undefined in the exported map");
  }

  ThreadedLoop(out).run(FromLogNoiseFunctor(), out);
#ifndef NDEBUG
  if (unfilled == noise_unfilled_type::THROW) {
    for (auto l = Loop(out)(out); l; ++l)
      assert(noise_level_is_valid(out.value()));
  }
#endif
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
  // A supplied map is conditioned exactly as an estimated one is. This is the route by which
  //   non-positive and non-finite noise levels most plausibly enter the software, and cubic
  //   interpolation would otherwise spread a single one of them across a 4^3 neighbourhood.
  return condition_noise_map(in,
                             noise_impute_type::NONE,
                             noise_pad_type::PAD,
                             0.0,
                             &H_spatial,
                             nullptr,
                             noise_unfilled_type::THROW,
                             "the noise level map supplied via -noise_in");
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
  // NAN_TO_ZERO is retained here, and here alone. A rank density is not a noise level: zero is a
  //   legitimate value for it (a patch with no signal rank), so there is nothing to impute and
  //   nothing the log domain would mean.
  return condition_noise_map(in, noise_impute_type::NAN_TO_ZERO, noise_pad_type::PAD, 0.0, &H_spatial);
}

void NoiseMapVSTRescaleFunctor::operator()(Image<float> &map) {
  vst_interp.scanner(transform.voxel2scanner *
                     Eigen::Vector3d({double(map.index(0)), double(map.index(1)), double(map.index(2))}));
  const float sigma = vst_interp.value();
  map.value() *= (power == 2) ? (sigma * sigma) : sigma;
}

} // namespace MR::Denoise
