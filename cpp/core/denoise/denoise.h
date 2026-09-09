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
#include <cmath>
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

// Whether a value stored in a noise level map is a usable noise level.
//
// A single sentinel governs every noise level map in this software: NaN means "no valid noise
//   level here". Zero and negative values are not sentinels but corruption -- a zero noise level
//   blanks the data it stabilises (NoiseModel::Gaussian::stabilise returns 0 for sigma <= 0), and a
//   negative one is silently laundered into a plausible positive sigma^2 by the squaring in
//   Estimator::Import::get_sigma_sq -- so they are treated here as equivalent to NaN and converted
//   to it before anything else reads the map.
//
// Note the deliberate asymmetry with Estimator::Result, which uses sigma2 == 0 as its own internal
//   "no usable threshold" sentinel (see Result::operator bool()). The conversion between the two
//   conventions happens at exactly one place, where Estimate writes the estimated level into
//   exports.noise_out.
inline bool noise_level_is_valid(const float value) { return std::isfinite(value) && value > 0.0F; }

// Report the number of voxels of a noise level map at which no valid noise level was estimated and
//   which conditioning consequently had to fill. It does not name the mechanism, because it is
//   emitted before any of them has run and more than one may be involved; the INFO messages that
//   follow report how many voxels each of them accounted for. Escalates from INFO to WARN by the same
//   rule as Denoise::report_invalid_data(): the threshold is the largest number of voxels within a
//   single slice orthogonal to any axis, excluding the slice-encoding axis where the metadata name
//   it. "H" must be the header of the map being conditioned (the sub-sampled grid), so that "one
//   slice" is one slice of that map. "context" names the map in the message.
void report_imputed_noise_levels(const size_t imputed_count, const Header &H, const std::string &context);

// Make a noise level map fit to be used: pad it for interpolation, and guarantee that every voxel
//   of it holds a usable noise level.
//
// Conditioning is performed on log(sigma^2), and this is the only place in the software at which
//   that domain appears. Both operations below -- the Gaussian smoothing and the imputation -- are
//   linear filters, and applying a linear filter to log(sigma^2) rather than to sigma is what makes
//   them well posed:
//
// - Power-invariance. The map stores sigma, the estimator produces sigma^2, and the eigenvalue
//   model is linear in sigma^2. Arithmetic smoothing does not commute with the square root
//   (sqrt(G*sigma^2) != G*sigma), so the result would depend on the arbitrary choice of which power
//   of the noise level the image happens to hold. In the log domain the two agree exactly, the
//   operation being equivariant under any power law. (Smoothing log(sigma^2) and log(sigma) are for
//   the same reason the same operation: they differ by a constant factor 2, which passes unchanged
//   through a normalised linear filter.)
// - The estimator error is multiplicative. A patch's sigma^2 is a scaled chi-squared-type variate
//   with roughly constant relative error (sd ~ sqrt(2/nu)) -- strongly heteroscedastic in absolute
//   terms, homoscedastic on the log scale, which is where a linear filter is the minimum-variance
//   combiner. log is exactly the variance-stabilising transform of the constant-CV family, the same
//   reasoning this pipeline already applies to the data themselves.
// - Dynamic range. Across a receive array the noise field spans a large multiplicative range. A
//   fixed-width Gaussian in the linear domain distributes ABSOLUTE error uniformly, concentrating
//   the relative damage in the low-sigma interior while the high-sigma periphery bleeds inward. In
//   the log domain the same kernel imposes a uniform RELATIVE error, which is what the downstream
//   use of sigma -- as a divisor in the variance-stabilising transform -- cares about.
// - Outlier robustness. A single near-failed patch at ten times the surrounding level dominates an
//   arithmetic neighbourhood mean; on the log scale it contributes ln 10.
// - Consistency with the iteration's own algebra. The update is multiplicative,
//   sigma_k = sigma_{k-1} * sigma_post-VST, so the quantity that composes additively across
//   iterations is log sigma. Log-domain smoothing commutes with that update; arithmetic smoothing
//   does not.
// - Positivity by construction. exp() of anything finite is positive, so no combination of
//   filtering, imputation or interpolation can emit a zero or negative sigma into a map that is
//   subsequently cubic-interpolated.
//
// The counter-argument, stated honestly: if the intent were to POOL the variance of a genuinely
//   heterogeneous neighbourhood, the arithmetic mean of sigma^2 would be correct, because variances
//   add. The intent here is to regularise an estimate of a smooth field -- to reduce estimator
//   variance without moving the field -- and for that the log domain is right. The consequence is a
//   geometric rather than arithmetic mean, which is smaller by approximately exp(s^2/2) in the
//   within-kernel spread s: below 0.1% from estimator noise alone, a few percent where the true
//   field varies steeply within the kernel. Small, systematic, and always downward.
//
// "impute" selects what becomes of a voxel holding no valid value:
// - NONE imputes it, which is the behaviour every noise level map takes. Voxels failing
//   noise_level_is_valid() are converted to NaN first, so that a single sentinel governs the whole
//   operation whatever the map's provenance.
// - NAN_TO_ZERO replaces it with zero, and bypasses everything else here: no log domain, no
//   imputation, only the padding. This exists for the ONE map passing through this function that is
//   not a noise level -- the rank-per-mm density of import_rank_per_mm_map(), for which zero is a
//   legitimate value and the log domain is meaningless.
enum class noise_impute_type{ NAN_TO_ZERO, NONE };
enum class noise_pad_type{ PAD, NONE };
// What to do with a voxel that no mechanism could fill: one with no valid voxel anywhere in the map
//   to reach it, and no patch-derived bound of its own.
// - THROW: the map is to seed a subsequent iteration's variance-stabilising transform, so it must
//   be usable; a wholly unfillable one is a broken run.
// - RETAIN: the map is only being exported. Leaving the NaN visible is right there, because a
//   configuration that produces no estimate anywhere is legitimately degenerate (-fixed_rank set to
//   the available rank is exactly that), and aborting the run over an output image would be wrong.
enum class noise_unfilled_type { THROW, RETAIN };
// When "coverage" is non-null, the padding is widened as necessary so that the returned map's
//   field of view encloses that header's field of view (plus the interpolation margin). An
//   externally-supplied map need not share the DWI voxel grid -- the estimator interpolates it --
//   but it must be sampleable at every patch position the DWI grid can produce, which a map on a
//   smaller or offset grid otherwise is not. Padded voxels replicate the nearest edge value.
// "fwhm" is the full width at half maximum, in millimetres, of the isotropic Gaussian with
//   which the map is smoothed once conditioned; zero (the default) bypasses smoothing entirely.
//   Expressing the width in millimetres rather than in voxels makes the extent of the smoothing a
//   property of the anatomy rather than of the grid the map happens to be estimated on, which for
//   an iterative schedule differs from one iteration to the next. Note that smoothing is no longer
//   the only thing this function does, nor a precondition for it doing anything: the imputation
//   below runs whether or not a width was requested.
// "bound" is the optional map of patch-derived upper bounds (Exports::noise_bound), consulted only
//   for voxels that no number of imputation passes could reach.
// "context" names the map in any message this function emits.
//
// POSTCONDITION, asserted: for impute == NONE with unfilled == THROW, the returned map is finite
//   and strictly positive at every voxel, padding included. That is what Interp::Cubic and
//   NoiseModel::stabilise have always assumed of it and never been given.
Image<float> condition_noise_map(Image<float> &in,
                                 const noise_impute_type impute = noise_impute_type::NONE,
                                 const noise_pad_type pad = noise_pad_type::PAD,
                                 const default_type fwhm = 0.0,
                                 const Header *coverage = nullptr,
                                 Image<float> *bound = nullptr,
                                 const noise_unfilled_type unfilled = noise_unfilled_type::THROW,
                                 const std::string &context = "the estimated noise level map");

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

// Log-domain twin of a conditioned noise level map: x = log(sigma^2) = 2 log(sigma), and NaN
//   wherever the source holds no usable level (which for a conditioned map is nowhere).
//
// Every point at which a noise level map is sampled between voxel centres samples this instead,
//   through LogNoiseInterp below. Two reasons, and they are the reasons conditioning itself is
//   performed in the log domain:
// - Cubic interpolation is not monotone: it undershoots at a step, and near the edge of the object
//   the noise field presents a large one. In the linear domain that undershoot reaches zero and
//   below, and the consequences are silent -- NoiseModel::Gaussian::stabilise() returns 0 for
//   sigma <= 0, blanking the data at that voxel; Estimator::Import squares the interpolated value,
//   laundering a negative sigma into a plausible positive sigma^2; Estimator::Constant divides by
//   its square, so a zero yields an infinity. exp() of an undershooting log value is merely small.
// - It makes interpolation geometric, matching the domain in which the map was conditioned. A map
//   smoothed as a geometric mean and then sampled as an arithmetic one is inconsistent with itself.
Image<float> log_noise_map(Image<float> &sigma_map);

// Cubic interpolation of a noise level map performed on its log-domain twin, returning the noise
//   level. Constructed over the image log_noise_map() returns, never over the map itself.
//   Out-of-field-of-view sampling yields NaN, as the underlying interpolator does; callers test the
//   value they receive.
class LogNoiseInterp {
public:
  explicit LogNoiseInterp(const Image<float> &log_map) : interp(log_map) {}
  bool scanner(const Eigen::Vector3d &position) { return interp.scanner(position); }
  default_type value() { return std::exp(0.5 * default_type(interp.value())); }

private:
  Interp::Cubic<Image<float>> interp;
};

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
  // The log-domain twin is built here, once per construction, and held so that the scratch buffer
  //   outlives the per-thread copies ThreadedLoop makes of this functor (an Image copy shares the
  //   buffer, and each copy's interpolator gets its own voxel accessor). Building it here rather
  //   than requiring it of the caller keeps every call site expressed in the map it actually holds.
  NoiseMapVSTRescaleFunctor(Image<float> &vst_image, const Header &grid, const int power)
      : vst_log(log_noise_map(vst_image)), vst_interp(vst_log), transform(grid), power(power) {
    assert(power == 1 || power == 2);
  }
  void operator()(Image<float> &map);

private:
  Image<float> vst_log;
  LogNoiseInterp vst_interp;
  Transform transform;
  int power;
};

} // namespace MR::Denoise
