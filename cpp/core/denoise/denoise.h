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

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "app.h"
#include "header.h"
#include "image.h"
#include "interp/cubic.h"
#include "transform.h"

namespace MR::Denoise {

constexpr ssize_t default_spatial_subsample_ratio = 2;

using eigenvalues_type = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using vector_type = Eigen::Array<double, Eigen::Dynamic, 1>;

extern const char *patent_description;
extern const char *first_step_description;
extern const char *non_gaussian_noise_description;
extern const char *decomposition_description;
extern const char *filter_description;
extern const char *aggregation_description;

// Eigenvalue-decomposition working precision selected by -datatype
//   (the enumerator order maps to the float32/float64 processing path index).
enum class dtype_t { FLOAT32, FLOAT64 };
extern const App::Option datatype_option;

enum class decomp_type { BDCSVD, SELFADJOINT };
extern const App::Option decomposition_option;
constexpr decomp_type default_decomposition = decomp_type::BDCSVD;

enum class filter_type { OPTSHRINK, OPTTHRESH, TRUNCATE };

enum class aggregator_type { EXCLUSIVE, GAUSSIAN, INVL0, RANK, UNIFORM };

// Default full width at half maximum of the Gaussian aggregator (option -aggregator_fwhm),
//   expressed as a multiple of the spacing between adjacent PCA patch centres (the reconstruction
//   sub-sample grid). A value of 2 places the half-maximum exactly on the neighbouring patch centre.
//   It is both the value that was historically hard-coded and, from first principles, the smallest
//   FWHM for which sampling the Gaussian on the overcomplete patch lattice leaves negligible residual
//   ripple in the summed aggregation weights (an approximate partition of unity), so it maximises
//   variance reduction without imprinting the sub-sampling grid or over-blurring the reconstruction.
constexpr default_type default_aggregator_fwhm = 2.0;

// These functions resolve dimensions of the matrix decomposition
//   in the presence of precoditioning that make the data rank-deficient
// - m = number of volumes
// - n = number of voxels in patch
// - rp = rank of preconditioner
ssize_t dimlong_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp);
ssize_t rank_nonzero(const ssize_t m, const ssize_t n, const ssize_t rp);
ssize_t rank_zero(const ssize_t m, const ssize_t n, const ssize_t rp);

// Convenience function for determining the total number of volumes
//   whether the input is 4D or higher
size_t num_volumes(const Header&);

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

// Function for performing filtering operations on a noise map:
// - May need to replace NaNs with zeros
//   (want to persist with putting NaNs in the noise map as indication of PCA failure rather than rank estimation failure,
//   but this is deleterious in an iterative environment)
// - May need to perform padding in order to facilitate interpolation at the next resolution
//   (while it would be preferable for this to be managed using EdgeHandlers in MRtrix3/mrtrix3#2278,
//    for the sake of progress we will here just do manual explicit padding)
// - May be instructed to perform explicit smoothing of the noise map prior to the next iteration
enum class noise_impute_type{ NAN_TO_ZERO, NONE };
enum class noise_pad_type{ PAD, NONE };
enum class noise_smooth_type{ SMOOTH, NONE };
// When "coverage" is non-null, the padding is widened as necessary so that the returned map's
//   field of view encloses that header's field of view (plus the interpolation margin). An
//   externally-supplied map need not share the DWI voxel grid -- the estimator interpolates it --
//   but it must be sampleable at every patch position the DWI grid can produce, which a map on a
//   smaller or offset grid otherwise is not. Padded voxels replicate the nearest edge value.
Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type impute = noise_impute_type::NAN_TO_ZERO,
                                 const noise_pad_type pad = noise_pad_type::PAD,
                                 const noise_smooth_type smooth = noise_smooth_type::NONE,
                                 const Header *coverage = nullptr);

// Construct the externally-provided noise level map (the variance-stabilising-transform scale)
//   from a -noise_in command-line argument, which may be either:
// - a scalar value, yielding a spatially-constant map over the spatial grid of H_spatial; or
// - the filesystem path to a pre-estimated 3D noise level image.
// The scalar-vs-image dispatch mirrors that of Estimator::make_imposed().
// The returned map is conditioned (padded) for safe cubic interpolation.
Image<float> import_vst_noise_map(const App::ParsedArgument &arg, const Header &H_spatial);

// Decide whether a value-or-image argument (-noise_in, -rankpermm_in) was given as a plain
//   number rather than an image path, returning the value if so. The raw text decides only which
//   interpretation applies; the conversion then goes through ParsedArgument::as_float(), so the
//   range declared on the option is what rejects an out-of-range value (and that exception
//   propagates rather than being downgraded into an attempt to open the value as a filename).
std::optional<default_type> imposed_scalar_value(const App::ParsedArgument &arg);

// Per-voxel signal-rank density (estimated signal rank per mm of kernel radius), used to size the
//   rank-adaptive spherical kernel (Kernel::SphereRankAdaptive). Computed as
//   rank_input / (num_partitions * max_dist), the scalar quantity accumulated across iterations in
//   the dwidenoise2 / dwi2noise run() loops and (optionally) exported via -rankpermm_out. The
//   returned scratch image shares the grid of max_dist (the subsample grid); voxels with a
//   non-positive patch radius yield zero density.
Image<float> compute_rank_per_mm(Image<uint16_t> &rank_input, Image<float> &max_dist, const ssize_t num_partitions);

// Construct an externally-provided rank-per-mm map from a -rankpermm_in command-line argument
//   (scalar value or 3D image path), mirroring import_vst_noise_map(). The map is conditioned
//   (NaN->0 and padded) so it can be safely interpolated to size the reconstruction kernel.
Image<float> import_rank_per_mm_map(const App::ParsedArgument &arg, const Header &H_spatial);

// Multiply a 3-D map in place by the noise level (variance-stabilising-transform scale)
//   interpolated at each voxel's scanner position. When the data were stabilised with a noise
//   map, an estimate derived on the stabilised data is a unit-less correction factor; multiplying
//   by the noise level recovers the first-order refinement on the native scale (see
//   Iterative::estimate and the dwidenoise2 final noise_out / lamplus exports). ThreadedLoop
//   copies the functor per thread, giving each thread its own interpolator; the map being scaled
//   is the looped image, so writes are voxel-disjoint. "grid" is the header defining the map's
//   voxel-to-scanner transform (the sub-sample grid).
class NoiseMapVSTRescaleFunctor {
public:
  NoiseMapVSTRescaleFunctor(Image<float> &vst_image, const Header &grid)
      : vst_interp(vst_image), transform(grid) {}
  void operator()(Image<float> &map) {
    vst_interp.scanner(transform.voxel2scanner *
                       Eigen::Vector3d({double(map.index(0)), double(map.index(1)), double(map.index(2))}));
    map.value() *= vst_interp.value();
  }

private:
  Interp::Cubic<Image<float>> vst_interp;
  Transform transform;
};

} // namespace MR::Denoise
