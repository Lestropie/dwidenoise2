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

  + "Important note:"
    " noise level estimation should only be performed as the first step of an image processing pipeline."
    " The routine is invalid if interpolation or smoothing has been applied to the data prior to denoising."

  + "Note that on complex input data,"
    " the output will be the total noise level across real and imaginary channels,"
    " so a scale factor sqrt(2) applies."

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
           " image is only an initial seed, not the final estimated output)")
    + Argument("value/image").type_float(0.0).type_image_in()
  + Schedule::schedule_option
  + precondition_options(false)

  + DWI::GradImportOptions()
  + DWI::GradExportOptions()

  + OptionGroup("Options for exporting additional data regarding PCA behaviour")
  + Option("rank_pcanonzero",
           "The (non-zero) rank of the PCA decomposition prior to any noise level estimation")
    + Argument("image").type_image_out()
  + Option("lamplus",
           "The estimated upper bound of the noise portion of the eigenspectrum (\"lambda-plus\")")
    + Argument("image").type_image_out()
  + Option("rank_input",
           "The estimated rank of the input data for each denoising patch")
    + Argument("image").type_image_out()
  + Option("rankpermm_out",
           "Export the per-voxel signal-rank density (estimated signal rank per mm of kernel radius)."
           " This is the quantity dwidenoise2 uses to size its rank-adaptive reconstruction kernel;"
           " exporting it here and importing it into dwidenoise2 via -rankpermm_in allows a"
           " single-row (e.g. \"apriori\") dwidenoise2 schedule to reproduce that kernel.")
    + Argument("image").type_image_out()
  + Option("eigenspectra",
           "Output a matrix containing the spectra of eigenvalues across patches"
           " (one row per patch). Where volume partitioning is in effect, each row is the"
           " per-partition eigenspectra concatenated and sorted, normalised by partition size,"
           " rather than a single decomposition's spectrum.")
    + Argument("file").type_file_out()

  + OptionGroup("Options for debugging the operation of sliding window kernels")
  + Option("max_dist",
           "The maximum distance between the centre of the patch and a voxel that was included within that patch")
    + Argument("image").type_image_out()
  + Option("voxelcount",
           "The number of voxels that contributed to the PCA for processing of each patch")
    + Argument("image").type_image_out()
  + Option("patchcount",
           "The number of unique patches to which an input image voxel contributes")
    + Argument("image").type_image_out();
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
         Image<float> &user_vst_image,
         const decomp_type decomposition,
         std::shared_ptr<Estimator::Base> estimator,
         const std::vector<Iterative::Iteration> &iterations,
         Exports &final_exports) {

  Image<T> input(dwi.get_image<T>());
  Image<bool> mask = generate_mask(input);
  Image<float> vst_image(user_vst_image);

  // Noise model governing the variance-stabilising transform, configured from
  //   the -noise_dof and -vst_method options (Gaussian for complex data).
  std::shared_ptr<NoiseModel::Base> noise_model = make_noise_model(is_complex<T>::value);

  Preconditioner<T> preconditioner(input, demodulation, demean, user_vst_image, noise_model);
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
    const ssize_t num_partitions = Iterative::resolve_num_partitions(iterations[iteration], mprime);
    if (num_partitions > 1 && !estimator->supports_partitioning())
      throw Exception("The selected noise level estimator does not support volume partitioning "
                      "(schedule iteration " + str(iteration + 1) + " requests " + str(num_partitions) +
                      " partitions); choose a different -estimator or remove partitioning from the schedule");
    preconditioner.set_partitioning_active(num_partitions > 1);
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
                        volume_group);
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
  const ssize_t num_partitions = Iterative::resolve_num_partitions(iterations.back(), mprime_final);
  if (num_partitions > 1 && !estimator->supports_partitioning())
    throw Exception("The selected noise level estimator does not support volume partitioning "
                    "(final schedule row requests " + str(num_partitions) +
                    " partitions); choose a different -estimator or remove partitioning from the schedule");
  preconditioner.set_partitioning_active(num_partitions > 1);
  preconditioner.update_parameters(vst_image, input);
  estimator->update_vst_image(vst_image);
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
                      volume_group);

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
  //   final output map only. demean_rank() yields the group count whether the demeaning was applied
  //   by the preconditioner or per partition within Estimate, so the contribution matches the
  //   non-partitioned case. Clamp to the available rank (max_rank).
  const uint16_t null_rank = uint16_t(preconditioner.demean_rank());
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

  const Demodulation demodulation = select_demodulation(dwi);
  const demean_type demean = select_demean(dwi);
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
    iterations = Schedule::load("dwi2noise");
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
         "and is therefore wasted computation; performing a single pass instead");
    Iterative::Iteration config;
    config.spatial_subsample_ratios = {default_spatial_subsample_ratio,
                                       default_spatial_subsample_ratio,
                                       default_spatial_subsample_ratio};
    // Single pass, no prior rank map available: size by a (rank-naive) aspect ratio (n ~ 2m).
    config.kernel.type = Kernel::kernel_spec_type::ASPECT_RATIO;
    config.kernel.param = 2.0;
    config.smooth_noiseout = noise_smooth_type::NONE;
    config.temporal_subsample = 1.0;
    config.update_noise = true;
    iterations.push_back(config);
  } else {
    iterations = Schedule::load_default("dwi2noise");
    Schedule::warn_if_default_schedule_slow(Denoise::num_volumes(dwi));
  }
  std::shared_ptr<SpatialSubsample> final_subsample =
      std::make_shared<SpatialSubsample>(dwi, iterations.back().spatial_subsample_ratios);

  // dwi2noise (re)estimates the noise level in every iteration, including the final one
  //   (its output is the estimate). Resolve any unset update_noise to true and reject a
  //   schedule that disables estimation in the final iteration.
  for (auto &it : iterations)
    if (!it.update_noise.has_value())
      it.update_noise = true;
  if (!iterations.back().update_noise.value())
    throw Exception("dwi2noise requires the final schedule iteration to estimate the noise "
                    "level (update_noise = true), as that iteration produces the output map");

  Exports final_exports(dwi, final_subsample->header());
  final_exports.set_noise_out(argument[1].as_text());
  opt = get_options("lamplus");
  if (!opt.empty())
    final_exports.set_lamplus(opt[0][0].as_text());
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

  int prec = (get_option_choice("datatype", dtype_t::FLOAT32) == dtype_t::FLOAT64) ? 1 : 0;
  if (complex)
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    assert(demodulation.axes.empty());
    INFO("select real float32 for processing");
    run<float>(dwi, demodulation, demean, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 1:
    assert(demodulation.axes.empty());
    INFO("select real float64 for processing");
    run<double>(dwi, demodulation, demean, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(dwi, demodulation, demean, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(dwi, demodulation, demean, user_vst_image, decomposition, estimator, iterations, final_exports);
    break;
  }
}
