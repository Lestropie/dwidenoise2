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

#include <memory>

#include "algo/threaded_loop.h"
#include "axes.h"
#include "command.h"
#include "denoise/denoise.h"
#include "denoise/estimate.h"
#include "denoise/estimator/estimator.h"
#include "denoise/exports.h"
#include "denoise/iterative.h"
#include "denoise/kernel/kernel.h"
#include "denoise/mask.h"
#include "denoise/metadata.h"
#include "denoise/partition.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "denoise/precondition/preconditioner.h"
#include "denoise/schedule.h"
#include "denoise/spatial_subsample.h"
#include "dwi/gradient.h"
#include "exception.h"
#include "filter/demodulate.h"
#include "filter/smooth.h"
#include "interp/linear.h"
#include "math/rng.h"

using namespace MR;
using namespace App;
using namespace MR::Denoise;
using namespace MR::Denoise::Precondition;

// clang-format off
void usage() {

  SYNOPSIS = "Noise level estimation using Marchenko-Pastur PCA";

  DESCRIPTION
  + "DWI data noise map estimation"
    " by interrogating data redundancy in the PCA domain"
    " using the prior knowledge that the eigenspectrum of random covariance matrices"
    " is described by the universal Marchenko-Pastur (MP) distribution."
    " Fitting the MP distribution to the spectrum of patch-wise signal matrices"
    " hence provides an estimator of the noise level 'sigma'."

  + "Unlike the MRtrix3 command dwidenoise,"
    " or command dwidenoise2 with which this command dwi2noise is provided,"
    " dwi2noise does not generate a denoised version of the input image series;"
    " its primary output is instead a map of the estimated noise level."
    " While this can also be obtained from the dwidenoise2 command using option -noise_out,"
    " using instead the dwi2noise command gives the ability to obtain a noise map"
    " to which filtering can be applied,"
    " which can then be utilised for the actual image series denoising,"
    " without computing an unwanted intermediate denoised image series."
    " The resulting (optionally filtered) noise map can subsequently be supplied"
    " to the dwidenoise2 -noise_in option to perform that denoising."

  + Denoise::patent_description

  + Denoise::non_gaussian_noise_description

  + noise_channels_description

  + "Important note:"
    " noise level estimation should only be performed as the first step of an image processing pipeline."
    " The routine is invalid if interpolation or smoothing has been applied to the data prior to denoising."
    " Where metadata of DICOM provenance are preserved in the image header,"
    " they are interrogated for evidence of such prior processing"
    "---specifically gradient non-linearity distortion correction,"
    " and reconstruction of a finer image matrix than the acquired k-space supports"
    " (k-space zero-filling)---"
    "and a warning is issued for any that is found."
    " The absence of such a warning is however not evidence that the data are suitable:"
    " the relevant fields seldom survive conversion from DICOM,"
    " and processing applied after that conversion leaves no trace in the image header at all."

  + "The estimated noise level is always that of a single real-valued channel,"
    " for complex input data as much as for magnitude."
    " The value estimated from a complex series is therefore that which would be estimated"
    " from the corresponding magnitude series at high SNR,"
    " and the two are interchangeable:"
    " a noise level map estimated from one may be used to denoise the other"
    " (via the \"-noise_in\" option of the dwidenoise2 command) without rescaling."
    " Note that this differs from the \"dwidenoise\" command in MRtrix3,"
    " whose noise map for complex input data is instead"
    " the aggregate level across the real and imaginary channels,"
    " and is therefore larger by a factor of sqrt(2);"
    " such a map must be divided by sqrt(2) before being supplied to this software."

  + "By default, the noise map estimate is derived through an iterative multi-resolution pyramid."
    " In the earlier iterations, there are fewer PCAs performed"
    " in order to obtain a low-resolution and smooth noise map estimate."
    " These estimates are then used as input to subsequent iterations"
    " to apply a variance-stabilising transform prior to PCA,"
    " deriving a new noise map at a higher spatial resolution."
    " The noise map estimated at the final iteration is additionally smoothed prior to export."

  + demodulation_description

  + Kernel::shape_description

  + Kernel::default_size_description

  + Kernel::cuboid_size_description

  + noise_boundary_description

  + Schedule::schedule_description;

  EXAMPLES
  + Example("Estimate a noise map, filter it, then denoise using the filtered map",
            "dwi2noise DWI.mif noise.mif;"
            " mrfilter noise.mif smooth noise_smooth.mif;"
            " dwidenoise2 DWI.mif DWI_denoised.mif -noise_in noise_smooth.mif",
            "Estimating the noise map with dwi2noise as a separate step makes it possible"
            " to inspect and post-process (e.g. smooth) the map before it is used for denoising;"
            " the curated map is then supplied to dwidenoise2 via -noise_in,"
            " which both parameterises the variance-stabilising transform"
            " and fixes the noise level so that no further data-driven estimation"
            " is performed during denoising.");

  AUTHOR = "Robert E. Smith (robert.smith@florey.edu.au)";

  COPYRIGHT =
  "Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;"
  " The Florey Institute of Neuroscience and Mental Health."
  " Licensed under the PolyForm Noncommercial License 1.0.0 (the \"License\");"
  " you may not use this file except in compliance with the License."
  " You may obtain a copy of the License at:"
  " https://polyformproject.org/licenses/noncommercial/1.0.0."
  " Unless required by applicable law or agreed to in writing,"
  " software distributed under the License is distributed on an \"AS IS\" BASIS,"
  " WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,"
  " either express or implied."
  " See the License of the specific language"
  " governing permissions and limitations under the License.";

  REFERENCES
  + "Smith, R.E. "
    "dwidenoise2. "
    "Zenodo, 2026, doi: 10.5281/zenodo.21601472"

  + "Veraart, J.; Fieremans, E. & Novikov, D.S. "
    "Diffusion MRI noise mapping using random matrix theory. "
    "Magn. Res. Med., 2016, 76(5), 1582-1593"

  + "Cordero-Grande, L.; Christiaens, D.; Hutter, J.; Price, A.N.; Hajnal, J.V. "
    "Complex diffusion-weighted image estimation via matrix recovery under general noise models. "
    "NeuroImage, 2019, 200, 391-404, doi: 10.1016/j.neuroimage.2019.06.039"

  + "* If estimating noise from complex data without changing -demodulate from its default (apc): "
    "Pizzolato, M.; Gilbert, G.; Thiran, J.-P.; Descoteaux, M.; Deriche, R. "
    "Adaptive phase correction of diffusion-weighted images. "
    "NeuroImage, 2020, 206, 116274, doi: 10.1016/j.neuroimage.2019.116274"

  + "* If using -demodulate hann: "
    "Manzano Patron, J.P.; Moeller, S.; Andersson, J.L.R.; Ugurbil, K.; Yacoub, E.; Sotiropoulos, S.N. "
    "Denoising diffusion MRI: Considerations and implications for analysis. "
    "Imaging Neuroscience, 2024, 2, 00060"

  + "* If estimating noise from magnitude data with any non-linear variance-stabilising transform:  "
    "Foi A. "
    "Noise estimation and removal in MR imaging: The variance-stabilization approach. "
    "IEEE International Symposium on Biomedical Imaging, 2011, 1809-1814, doi: 10.1109/ISBI.2011.5872758"

  + "* If using -vst koay: "
    "Koay, C.G.; Basser, P.J. "
    "Analytically exact correction scheme for signal extraction from noisy magnitude MR signals. "
    "Journal of Magnetic Resonance, 2006, 179(2), 317-322"

  + "* If estimating noise from magnitude data with any non-linear variance-stabilising transform: "
    "Ma, X.; Ugurbil, K.; Wu, X. "
    "Denoise magnitude diffusion magnetic resonance images "
    "via variance-stabilizing transformation and optimal singular-value manipulation. "
    "NeuroImage, 2020, 215, 116852"

  + "* If using -estimator mrm2023 (the default): "
    "Olesen, J.L.; Ianus, A.; Ostergaard, L.; Shemesh, N.; Jespersen, S.N. "
    "Tensor denoising of multidimensional MRI data. "
    "Magnetic Resonance in Medicine, 2023, 89(3), 1160-1172"

  + "* If using -estimator tbme2022: "
    "Zhu, W.; Ma, X.; Zhu, X.-H.; Ugurbil, K.; Chen, W.; Wu, X. "
    "Denoise Functional Magnetic Resonance Imaging With Random Matrix Theory Based Principal Component Analysis. "
    "IEEE Transactions on Biomedical Engineering, 2022, 69(11), 3377-3388, doi: 10.1109/TBME.2022.3168592"

  + "* If using -estimator med: "
    "Gavish, M.; Donoho, D.L. "
    "The Optimal Hard Threshold for Singular Values is 4/sqrt(3). "
    "IEEE Transactions on Information Theory, 2014, 60(8), 5040-5053.";

  ARGUMENTS
  + Argument("dwi", "the input diffusion-weighted image").type_image_in()
  + Argument("noise", "the output estimated noise level map").type_image_out();

  OPTIONS
  + OptionGroup("Options for modifying PCA computations")
  + datatype_option
  + decomposition_option
  + Estimator::estimator_option
  + Option("noise_in",
           "provide a pre-estimated noise level (scalar value or 3D image) to seed the"
           " variance-stabilising transform for the first iteration of the schedule;"
           " the schedule then refines the estimate (for a multi-stage schedule the supplied"
           " image is only an initial seed, not the final estimated output)."
           " The value or map must be the noise level of a single real-valued channel,"
           " which is the convention in which this command's own output is expressed"
           " (see the description above)")
    + Argument("value/image").type_float(0.0).type_image_in()
  + Schedule::schedule_option
  + precondition_options(option_scope_t::ESTIMATION)

  + DWI::GradImportOptions()
  + DWI::GradExportOptions()

  + export_options(option_scope_t::ESTIMATION)
  + kernel_debug_options(option_scope_t::ESTIMATION);
}
// clang-format on

namespace {
// Voxel-wise functors for the final-export loops. ThreadedLoop copies the functor per thread, so
//   each thread holds independent Image accessors; every operation below is voxel-local.

// Copy one image's value into another, voxel for voxel.
struct CopyValueFunctor {
  template <class ImageOut, class ImageIn> void operator()(ImageOut &out, ImageIn &in) {
    out.value() = in.value();
  }
};

// Add the preconditioner's demean null-rank back into the reported input rank (clamped to
//   max_rank), only for voxels carrying a non-zero estimated signal rank.
class RankInputAddbackFunctor {
public:
  RankInputAddbackFunctor(const uint16_t null_rank, const uint16_t max_rank)
      : null_rank(null_rank), max_rank(max_rank) {}
  void operator()(Image<uint16_t> &rank_input) {
    rank_input.value() = std::min<uint16_t>(uint16_t(rank_input.value()) + null_rank, max_rank);
  }

private:
  uint16_t null_rank;
  uint16_t max_rank;
};
} // namespace

template <typename T>
void run(Header &dwi,
         const Demodulation &demodulation,
         const demean_type demean,
         const Eigen::MatrixXd &grad,
         Image<float> &user_vst_image,
         const decomp_type decomposition,
         std::shared_ptr<Estimator::Base> estimator,
         const std::vector<Iterative::Iteration> &iterations,
         Exports &final_exports) {

  Image<T> input(dwi.get_image<T>());
  Image<bool> mask = generate_mask(input);
  Image<float> vst_image(user_vst_image);

  // Noise model governing the variance-stabilising transform, configured from
  //   the -noise_channels / -noise_dof and -vst_method options (Gaussian for complex data). Where
  //   magnitude data receive a non-linear transform and neither channel-count option was provided,
  //   the effective receive-channel count is instead inferred from the header key-values.
  std::shared_ptr<NoiseModel::Base> noise_model = make_noise_model(is_complex<T>::value, dwi);

  Preconditioner<T> preconditioner(input, demodulation, demean, user_vst_image, noise_model, grad);
  // Cold adaptive-phase-correction solves within the noise-estimation iterations below are
  //   bootstraps: the final iteration re-estimates at native resolution the phase of every volume
  //   it uses, so they may be solved on a 2x-downsampled grid. Cleared before that final iteration
  //   (and never set for a single-row schedule, which has no such iterations).
  preconditioner.set_apc_refine_later(iterations.size() > 1);
  Image<T> input_preconditioned;

  Image<float> rank_per_mm;

  // Random number generator for drawing temporal sub-sampling subsets.
  Math::RNG rng;
  constexpr ssize_t temporal_min_per_group = 2;

  // All but the last iteration
  for (ssize_t iteration = 0; iteration != iterations.size() - 1; ++iteration) {
    // Draw this iteration's temporal subset (stratified by demeaning group) and (re)compute
    //   the stabilised-domain per-group means over that subset, under the current noise level
    //   map; the preconditioned data then hold only the m' subset volumes.
    preconditioner.set_temporal_subsample(iterations[iteration].temporal_subsample, rng, temporal_min_per_group);
    const ssize_t mprime = preconditioner.header().size(3);
    const ssize_t num_partitions =
        Iterative::resolve_num_partitions(iterations[iteration], mprime, "Schedule iteration " + str(iteration + 1));
    // With partitioning the demeaning moves into Estimate (per partition); impose the resolved site
    //   before update_parameters() so the preconditioner skips the then-unused mean pass.
    const demean_site_t demean_site = resolve_demean_site(demean, num_partitions);
    preconditioner.set_demean_site(demean_site);
    preconditioner.update_parameters(vst_image, input);
    estimator->update_vst_image(vst_image);
    input_preconditioned =
        Image<T>::scratch(preconditioner.header(), "Preconditioned version of \"" + dwi.name() + "\"");

    // Per-kernel partitioning: pass demeaning-group labels + partition count; the per-patch
    //   assignment is drawn inside Estimate from a voxel-seeded RNG.
    std::vector<ssize_t> volume_group;
    std::shared_ptr<const Partitioning> partitioning; // null ⇒ per-kernel
    if (num_partitions > 1)
      volume_group = preconditioner.output_volume_to_group();

    std::shared_ptr<SpatialSubsample> subsample =
        std::make_shared<SpatialSubsample>(dwi, iterations[iteration].spatial_subsample_ratios);
    // For internal iterations, we only save the output noise level estimate
    Exports iteration_exports(dwi, subsample->header());
    iteration_exports.set_noise_out();
    iteration_exports.set_rank_input();
    iteration_exports.set_max_dist();
    Iterative::estimate(input,
                        input_preconditioned,
                        mask,
                        vst_image,
                        rank_per_mm,
                        iterations[iteration],
                        iteration,
                        subsample,
                        decomposition,
                        estimator,
                        preconditioner,
                        iteration_exports,
                        num_partitions,
                        partitioning,
                        volume_group,
                        demean_site);
    // Propagate result to next iteration
    vst_image = Denoise::condition_noise_map(iteration_exports.noise_out,
                                             noise_impute_type::NAN_TO_ZERO,
                                             noise_pad_type::PAD,
                                             iterations[iteration].smooth_noiseout);

    // Per-partition signal-rank density: rank_input is summed over the iteration's num_partitions
    //   partitions, while the next iteration's rank-based kernel is sized from the smallest
    //   partition (m'/P), so divide by the partition count (which may differ between iterations).
    rank_per_mm = Denoise::compute_rank_per_mm(iteration_exports.rank_input, iteration_exports.max_dist, num_partitions);
  }

  // Last iteration. Unlike dwidenoise2, dwi2noise may sub-sample the volumes of its final
  //   (output) iteration: it produces a noise level map, which is volume-count-independent
  //   in expectation. The spatial subsampling follows the final entry of the schedule.
  preconditioner.set_temporal_subsample(iterations.back().temporal_subsample, rng, temporal_min_per_group);
  // No APC pass follows this one: a volume this final subset draws in for the first time (earlier
  //   iterations having sub-sampled it away) has its cold coarse solve followed immediately by the
  //   warm native refinement, rather than stopping at the coarse bootstrap grid.
  preconditioner.set_apc_refine_later(false);
  const ssize_t mprime_final = preconditioner.header().size(3);
  const ssize_t num_partitions =
      Iterative::resolve_num_partitions(iterations.back(), mprime_final, "The final schedule row");
  const demean_site_t demean_site = resolve_demean_site(demean, num_partitions);
  preconditioner.set_demean_site(demean_site);
  preconditioner.update_parameters(vst_image, input);
  estimator->update_vst_image(vst_image);
  // -preconditioned exports the preconditioned data that are the input to PCA. The final
  //   iteration's are the ones that correspond to the exported noise map, so those are what is
  //   written (mirroring dwidenoise2 -preconditioned_input).
  auto opt_precond = get_options("preconditioned");
  if (!opt_precond.empty())
    input_preconditioned = Image<T>::create(opt_precond[0][0], preconditioner.header());
  else
    input_preconditioned =
        Image<T>::scratch(preconditioner.header(), "Preconditioned version of \"" + dwi.name() + "\"");
  std::vector<ssize_t> volume_group;
  std::shared_ptr<const Partitioning> partitioning; // null ⇒ per-kernel in Estimate
  if (num_partitions > 1)
    volume_group = preconditioner.output_volume_to_group();
  auto subsample = std::make_shared<SpatialSubsample>(dwi, iterations.back().spatial_subsample_ratios);
  Iterative::estimate(input,
                      input_preconditioned,
                      mask,
                      vst_image,
                      rank_per_mm,
                      iterations.back(),
                      iterations.size() - 1,
                      subsample,
                      decomposition,
                      estimator,
                      preconditioner,
                      final_exports,
                      num_partitions,
                      partitioning,
                      volume_group,
                      demean_site);

  // Smooth the exported noise map if (and only if) the final schedule row requests it. The non-final
  //   iterations have their smoothing applied by condition_noise_map when propagating the estimate to
  //   the next iteration; the final iteration's map is the command output, so its smoothing must be
  //   applied here, before it reaches the filesystem. This mirrors the smoothing dwidenoise2 applies
  //   internally to its penultimate map, so the two-phase route (dwi2noise, then
  //   dwidenoise2 -noise_in -schedule apriori) denoises with the same map a single dwidenoise2 run
  //   would. pad=NONE keeps the result on the (unpadded) output grid.
  if (iterations.back().smooth_noiseout == noise_smooth_type::SMOOTH) {
    Image<float> smoothed = Denoise::condition_noise_map(final_exports.noise_out,
                                                         noise_impute_type::NAN_TO_ZERO,
                                                         noise_pad_type::NONE,
                                                         noise_smooth_type::SMOOTH);
    ThreadedLoop(final_exports.noise_out).run(CopyValueFunctor(), final_exports.noise_out, smoothed);
  }

  // Optionally export the per-voxel signal-rank density (rank per mm of kernel radius) of this final
  //   iteration. This is the same quantity dwidenoise2 accumulates across its iterations to size its
  //   rank-adaptive reconstruction kernel; exporting it lets a single-row dwidenoise2 schedule
  //   (-rankpermm_in) reproduce that kernel. It is derived from the raw signal rank, so it must be
  //   computed before the demean null-rank addback below (which adjusts only the reported rank_input
  //   export). final_exports.rank_input / .max_dist are forced valid in run() when -rankpermm_out is
  //   requested.
  {
    auto opt_rpm = get_options("rankpermm_out");
    if (!opt_rpm.empty()) {
      assert(final_exports.rank_input.valid() && final_exports.max_dist.valid());
      Image<float> rpm =
          Denoise::compute_rank_per_mm(final_exports.rank_input, final_exports.max_dist, num_partitions);
      Header H_rpm(final_exports.max_dist);
      Image<float> out = Image<float>::create(opt_rpm[0][0].as_text(), H_rpm);
      ThreadedLoop(out).run(CopyValueFunctor(), out, rpm);
    }
  }

  // Add the regressed group-mean components back to the reported (signal-only) input rank, on the
  //   final output map only. partition_null_rank() yields the group count whether the demeaning was applied
  //   by the preconditioner or per partition within Estimate, so the contribution matches the
  //   non-partitioned case. Clamp to the available rank (max_rank).
  const uint16_t null_rank = uint16_t(preconditioner.partition_null_rank());
  const uint16_t max_rank = Denoise::num_volumes(dwi);
  if (null_rank > 0 && final_exports.rank_input.valid()) {
    ThreadedLoop(final_exports.rank_input)
        .run(RankInputAddbackFunctor(null_rank, max_rank), final_exports.rank_input);
  }
}

void run() {
  auto dwi = Header::open(argument[0]);
  if (dwi.ndim() < 4)
    throw Exception("input image must be at least 4-dimensional");
  if (Denoise::num_volumes(dwi) == 1)
    throw Exception("input image must be non-singleton across non-spatial dimensions");
  bool complex = dwi.datatype().is_complex();

  // Report any evidence in the image metadata that upstream processing has already invalidated the
  //   assumptions underpinning Marchenko-Pastur-based noise level estimation.
  Denoise::warn_if_metadata_precludes_mp(dwi);

  // Resolve the DW gradient table once, here, from the Header returned by Header::open(): only
  //   that header carries the transform-realignment metadata that -fslgrad import needs in order
  //   to rotate the bvecs. Headers derived downstream (e.g. the preconditioner's, which is built
  //   from an Image rather than a Header) lose it, and re-deriving the table from one of those
  //   would assert in a debug build and silently misrotate the bvecs in a release build.
  // Resolving it here also means a table supplied on the command line is validated: MRtrix3
  //   already diagnoses a wrong row count, a non-4-column table, -grad together with -fslgrad, and
  //   a bvecs/bvals length mismatch, but until now every one of those diagnostics was swallowed by
  //   the shell-availability probe in select_demean(), and the command silently fell back to
  //   another demeaning mode. An absent or malformed table *within the image header* remains a
  //   legitimate reason to fall back; a table the user explicitly supplied is a hard error.
  Eigen::MatrixXd grad;
  const bool grad_from_commandline = !get_options("grad").empty() || !get_options("fslgrad").empty();
  try {
    grad = DWI::get_DW_scheme(dwi);
  } catch (Exception &e) {
    if (grad_from_commandline)
      throw Exception(e, "Invalid diffusion gradient table provided on the command line");
    grad.resize(0, 0);
  }

  const Demodulation demodulation = select_demodulation(dwi);
  const demean_type demean = select_demean(dwi, grad);
  const decomp_type decomposition = get_option_choice("decomposition", default_decomposition);

  // Resolve the requested variance-stabilising transform up-front:
  //   -vst_method none disables the transform entirely, which makes iterative
  //   refinement of the noise level pointless (handled below) and is incompatible
  //   with -noise_in (which exists only to parameterise the transform).
  const NoiseModel::vst_method_t vst_method =
      get_option_choice("vst_method", NoiseModel::vst_method_t::FOI);
  const bool vst_none = (vst_method == NoiseModel::vst_method_t::NONE);

  // A pre-estimated noise level provided via -noise_in seeds the variance-stabilising
  //   transform (the VST scale) as a prior; unlike dwidenoise2 it does not bypass
  //   estimation (make_estimator below keeps permit_bypass == false), so a genuine
  //   estimator still runs and the output map is the refinement sigma_in * (post-VST sigma).
  //   The forward transform is well-posed without a mean, so -demean none is permitted
  //   (see vst_plan.md sections 2.1 and 6.2).
  Image<float> user_vst_image;
  auto opt = get_options("noise_in");
  if (!opt.empty()) {
    if (vst_none)
      throw Exception("Options -noise_in and -vst_method none are incompatible: "
                      "a pre-estimated noise level is used to parameterise the variance-stabilising transform, "
                      "which is disabled by -vst_method none");
    user_vst_image = Denoise::import_vst_noise_map(opt[0][0], dwi);
  }

  auto estimator = Estimator::make_estimator(user_vst_image, false);

  // Resolve the iteration schedule. When the user supplies -schedule it is the single
  //   source of truth for the spatial and temporal sub-sampling of each iteration; otherwise
  //   the command's bundled default schedule is loaded from file (Schedule::load_default).
  //   "one-pass" operation is simply a single-row schedule. -vst_method none removes the
  //   coupling between iterations and so reduces to a single-iteration schedule.
  std::vector<Iterative::Iteration> iterations;
  if (Schedule::requested()) {
    // dwi2noise has no -rankpermm_in option and no prior density on its first row.
    iterations = Schedule::load("dwi2noise", false);
    // Without a variance-stabilising transform there is no coupling between iterations, so the
    //   per-iteration refinement of a multi-row schedule cannot propagate and is wasted
    //   computation; a single-row schedule is, however, perfectly meaningful under
    //   -vst_method none.
    if (vst_none && iterations.size() > 1)
      throw Exception("Option -vst_method none cannot be combined with a -schedule of more than "
                      "one row: without a variance-stabilising transform there is no coupling "
                      "between iterations for the additional schedule rows to control; "
                      "use a single-row schedule");
  } else if (vst_none) {
    WARN("-vst_method none: no variance-stabilising transform is applied, so iteratively "
         "refining the noise level estimate would have no effect on subsequent iterations "
         "and is therefore wasted computation; the bundled single-row \"novst\" schedule is "
         "used in place of the bundled \"default\" schedule");
    iterations = Schedule::load_bundled("dwi2noise", "novst", false);
  } else {
    iterations = Schedule::load_default("dwi2noise", false);
    Schedule::warn_if_default_schedule_slow(Denoise::num_volumes(dwi));
  }
  std::shared_ptr<SpatialSubsample> final_subsample =
      std::make_shared<SpatialSubsample>(dwi, iterations.back().spatial_subsample_ratios);

  // dwi2noise (re)estimates the noise level in every iteration, including the final one
  //   (its output is the estimate), so update_noise is unconditionally true here. The schedule
  //   parser rejects the "update_noise" column outright for dwi2noise whatever its value
  //   (Schedule::parse(), permit_update_noise), so no iteration can arrive with it already set
  //   and there is no final-row case left to check.
  for (auto &it : iterations) {
    assert(!it.update_noise.has_value());
    it.update_noise = true;
  }

  Exports final_exports(dwi, final_subsample->header());
  final_exports.set_noise_out(argument[1].as_text());
  opt = get_options("lamplus");
  if (!opt.empty())
    final_exports.set_lamplus(opt[0][0].as_text());
  opt = get_options("lammax_noise");
  if (!opt.empty())
    final_exports.set_lammax_noise(opt[0][0].as_text());
  opt = get_options("rank_pcanonzero");
  if (!opt.empty())
    final_exports.set_rank_pcanonzero(opt[0][0].as_text());
  opt = get_options("rank_input");
  if (!opt.empty())
    final_exports.set_rank_input(opt[0][0].as_text());
  opt = get_options("max_dist");
  if (!opt.empty())
    final_exports.set_max_dist(opt[0][0].as_text());
  opt = get_options("voxelcount");
  if (!opt.empty())
    final_exports.set_voxelcount(opt[0][0].as_text());
  opt = get_options("patchcount");
  if (!opt.empty())
    final_exports.set_patchcount(opt[0][0].as_text());
  opt = get_options("eigenspectra");
  if (!opt.empty())
    final_exports.set_eigenspectra_path(opt[0][0].as_text());
  // Exporting the rank-per-mm map requires the per-patch signal rank and patch radius of the final
  //   iteration; ensure both are populated (as scratch if the user did not request them as outputs).
  //   The map itself is computed and written in run<T>() once those values exist.
  if (!get_options("rankpermm_out").empty()) {
    if (!final_exports.rank_input.valid())
      final_exports.set_rank_input();
    if (!final_exports.max_dist.valid())
      final_exports.set_max_dist();
  }

  // -export_grad_mrtrix / -export_grad_fsl are declared by DWI::GradExportOptions(); this is the
  //   function that consumes them. It reads the "dw_scheme" keyval, which the gradient resolution
  //   above writes back into the header whenever -grad / -fslgrad was given, and which otherwise
  //   is the image's own; with neither, it reports that no gradient information is available.
  DWI::export_grad_commandline(dwi);

  int prec = (get_option_choice("datatype", dtype_t::FLOAT32) == dtype_t::FLOAT64) ? 1 : 0;
  if (complex)
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    assert(demodulation.axes.empty());
    INFO("select real float32 for processing");
    run<float>(dwi, demodulation, demean, grad, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 1:
    assert(demodulation.axes.empty());
    INFO("select real float64 for processing");
    run<double>(dwi, demodulation, demean, grad, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(dwi, demodulation, demean, grad, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(dwi, demodulation, demean, grad, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  }
}
