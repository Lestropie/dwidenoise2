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
#include <cassert>
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

// Number of statistically independent real Gaussian noise components carried by one entry of the
//   Casorati matrix: one for real-valued input, two (real and imaginary) for complex.
//
// A complex patch's eigenvalues are accordingly twice those a real patch at the same *per-channel*
//   noise level would produce, since E[|x|^2] == 2 sigma^2 where the real and imaginary parts each
//   carry variance sigma^2. Dividing the eigenvalues by this factor as soon as they are computed
//   (Estimate::to_per_channel) is what makes every eigenvalue -- and hence every noise level derived
//   from one, anywhere downstream -- a per-channel quantity, so that a scalar noise level means one
//   thing whatever the type of the input data. Nothing downstream undoes it, and nothing needs to:
//   the eigenspectrum filter depends only on the ratio of an eigenvalue to sigma^2, and
//   reconstruction is projected through the singular values / eigenvectors rather than through the
//   eigenvalues.
//
// This is the point at which dwidenoise2 departs from MRtrix3 dwidenoise, whose noise level for
//   complex input is the aggregate across the two channels and is therefore larger by sqrt(2).
template <typename F> constexpr double casorati_entry_channels() { return is_complex<F>::value ? 2.0 : 1.0; }

// Which command a shared OptionGroup is being constructed for:
//   ESTIMATION for noise level estimation alone (dwi2noise),
//   DENOISING for denoising with optional noise level estimation (dwidenoise2).
//   Options pertaining only to the reconstruction of a denoised image series are omitted for the former.
enum class option_scope_t { ESTIMATION, DENOISING };

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

// Which component actually subtracts the group means, and hence which one is responsible for
//   accounting for the rank they regress out. It is declared here, rather than alongside the other
//   preconditioning enumerations, because it is the contract *between* the preconditioner and the
//   Estimate/Recon classes: both sides key their own behaviour on it.
// The rank removed is the same in either case -- Precondition::Preconditioner::partition_null_rank()
//   -- because the same group count is regressed from each Casorati matrix that is decomposed. What
//   differs is where it happens, and that must be stated rather than inferred from the partition
//   count: reading a zero from a "how much rank did the preconditioner remove" query is
//   indistinguishable from "no demeaning was performed at all" unless the caller already knows the
//   partitioning state.
// - NONE          : -demean none; no group means are regressed anywhere.
// - PRECONDITIONER: subtracted once over the whole volume series by Preconditioner::operator(),
//                     so the data reaching PCA are already mean-free.
// - ESTIMATOR     : subtracted per partition inside Estimate/Recon, keeping each partition's mean
//                     subtraction orthogonal to that partition's own decomposition. Used whenever
//                     volume partitioning is active.
enum class demean_site_t { NONE, PRECONDITIONER, ESTIMATOR };

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

// Multiply a 3-D map in place by a power of the noise level (variance-stabilising-transform scale)
//   interpolated at each voxel's scanner position. When the data were stabilised with a noise
//   map, an estimate derived on the stabilised data is a unit-less correction factor; multiplying
//   by the noise level recovers the first-order refinement on the native scale (see
//   Iterative::estimate and the dwidenoise2 final noise_out / boundary-map exports). ThreadedLoop
//   copies the functor per thread, giving each thread its own interpolator; the map being scaled
//   is the looped image, so writes are voxel-disjoint. "grid" is the header defining the map's
//   voxel-to-scanner transform (the sub-sample grid).
//
// "power" is the power of sigma in which the map being rescaled is expressed, and must be stated by
//   every caller rather than assumed: a noise level map carries sigma (power 1), whereas a map of
//   eigenvalue units carries sigma^2 (power 2). Both signal/noise boundaries of Estimator::Result
//   are proportional to sigma^2 -- lamplus is sigma^2 * edge(), and the standardised maximal noise
//   eigenvalue is that same quantity divided by a shape constant -- so removing the effect of the
//   stabilisation from either exported map requires the square of the noise level; using the first
//   power leaves a map that is neither the stabilised quantity nor the native one, and does so
//   silently, differing only in magnitude.
class NoiseMapVSTRescaleFunctor {
public:
  NoiseMapVSTRescaleFunctor(Image<float> &vst_image, const Header &grid, const int power)
      : vst_interp(vst_image), transform(grid), power(power) {
    assert(power == 1 || power == 2);
  }
  void operator()(Image<float> &map);

private:
  Interp::Cubic<Image<float>> vst_interp;
  Transform transform;
  int power;
};

} // namespace MR::Denoise
