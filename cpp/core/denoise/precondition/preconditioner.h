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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "denoise/precondition/apc.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "denoise/precondition/precondition.h"
#include "filter/demodulate.h"
#include "header.h"
#include "image.h"
#include "interp/cubic.h"
#include "math/rng.h"
#include "transform.h"
#include "types.h"

namespace MR::Denoise::Precondition {

// Need to SFINAE define the demodulator type,
//   so that it does not attempt to compile the demodulation filter for non-complex types
class DummyDemodulator {
public:
  template <class ImageType> DummyDemodulator(ImageType &, const std::vector<size_t> &, const bool) {}
  template <class InputImageType, class OutputImageType>
  void operator()(InputImageType &, OutputImageType &, const bool) {
    assert(false);
  }
  Image<cfloat> operator()() { return Image<cfloat>(); }
};
template <typename T> struct DemodulatorSelector {
  using type = DummyDemodulator;
};
template <typename T> struct DemodulatorSelector<std::complex<T>> {
  using type = Filter::Demodulate;
};

template <typename T> class Preconditioner {
public:
  Preconditioner(Image<T> &image,
                 const Demodulation &demodulation,
                 const demean_type demean,
                 Image<float> &vst_noise,
                 std::shared_ptr<NoiseModel::Base> noise_model);
  Preconditioner(Preconditioner &) = default;
  // Refresh the preconditioning parameters together, each iteration of the noise-estimation
  //   schedule: the noise level map (VST scale), the stabilised-domain per-group means (VST
  //   offset, derived from it; see vst_plan.md section 3.2), and -- when adaptive phase
  //   correction is active (-demodulate apc) -- the background phase map.
  // A pass over the input is required to recompute the stabilised-domain means; when APC runs
  //   it must precede that pass, because the means are formed from the demodulated data
  //   (serialise_and_stabilise) and so must reflect the updated phase.
  // APC guards (all must hold): (1) complex input (compile-time gate; the phase is ill-defined
  //   for real T); (2) -demodulate apc selected (apc_enabled); (3) a phase map is maintained
  //   (phase_image.valid()). APC runs every call, including the first: the incoming noise map
  //   may be absent on the first iteration, in which case the estimator self-calibrates a
  //   global noise level from the data with uniform weighting (see AdaptivePhaseEstimator); later
  //   iterations pass the refined spatially-varying map. When the schedule has more than one
  //   iteration (set_apc_coarse_first), the first APC pass is solved on a 2x-downsampled grid
  //   (cold start; a background phase is smooth, so this is near-lossless at ~1/4 the cost) and
  //   every subsequent pass runs at native resolution but warm-started from the previous estimate
  //   with a reduced iteration budget (see AdaptivePhaseEstimator). A single-iteration schedule
  //   solves its sole, authoritative pass at native resolution. apc_first_pass tracks which regime
  //   applies.
  void update_parameters(Image<float> new_vst_noise, Image<T> input) {
    vst_noise_image = new_vst_noise;
    if constexpr (is_complex<T>::value) {          // (1)
      if (apc_enabled && phase_image.valid())      // (2), (3)
        update_phase(input);
    }
    compute_means(input);
  }
  // The bias_handling and debias_anchor arguments apply only to the inverse transform
  //   (inverse == true) and are ignored for the forward transform:
  // - bias_handling selects how the noise-distribution bias is treated in the
  //     reconstructed output (see bias_handling_t);
  // - debias_anchor selects the operating point for the (non-linear) inverse
  //     (see debias_anchor_t); the default SAMPLE undoes demeaning then inverts pointwise.
  void operator()(Image<T> input,
                  Image<T> output,
                  const bool inverse = false,
                  const bias_handling_t bias_handling = bias_handling_t::DEBIAS,
                  const debias_anchor_t debias_anchor = debias_anchor_t::SAMPLE) const;

  // Enable/disable preconditioner-side demeaning for the current iteration. When volume
  //   partitioning is in effect (P > 1), demeaning is instead performed per partition by the
  //   Estimate/Recon classes (per-partition per-group means keep the mean subtraction orthogonal
  //   to each partition's PCA); the preconditioner then applies only phase demodulation and the
  //   (non-linear) variance-stabilising transform, and its inverse reduces to the pointwise
  //   inverse VST (the demean offset re-added by Recon already lives in the stabilised domain).
  //   While active, null_rank() reports 0 (no globally-regressed mean) and the forward/inverse
  //   transforms skip the demean step. The GROUP_MEAN inverse anchor is unavailable in this mode.
  void set_partitioning_active(const bool active) { partitioning_active = active; }
  bool demean_active() const { return vst_mean_image.valid() && !partitioning_active; }

  // Enable solving the first adaptive-phase-correction pass on a 2x-downsampled grid (its phase
  //   upsampled to native). Set from the run loop to (schedule length > 1): downsampling is only
  //   worthwhile when a later pass refines the phase at native resolution, so a single-iteration
  //   schedule leaves it disabled and solves its sole, authoritative pass natively. No effect
  //   unless -demodulate apc is active on complex data.
  void set_apc_coarse_first(const bool enable) { apc_coarse_first = enable; }

  // Per-(preconditioned/output) row demeaning-group labels for the current data layout (length
  //   header().size(3) == m', accounting for any active temporal subset): the b-value shell or
  //   5th-dimension volume-group index of each output volume. Returns an empty vector when no
  //   grouped demeaning applies (-demean none), and an all-zero vector for -demean all (a single
  //   group). Used to (a) stratify the volume partitioning so each group is balanced across
  //   partitions, and (b) drive the per-partition per-group demeaning within Estimate/Recon.
  std::vector<ssize_t> output_volume_to_group() const;

  // Select a stratified random subset of "fraction" of the volumes (along the
  //   supra-spatial axes) for the next forward preconditioning + mean computation.
  //   The subset is stratified by the active demeaning grouping (b-value shells or
  //   volume groups; a single stratum for -demean none/all), so the relative
  //   proportions of volumes from each group are preserved; each stratum keeps at
  //   least min_per_group volumes. fraction >= 1 clears any subset (use all volumes).
  // While a subset is active, header().size(3) and the forward output hold only the
  //   m' selected volumes, so downstream code (Estimate / kernel sizing) sees m'
  //   volumes and need not know that sub-sampling occurred.
  // A subsequent update_parameters() / compute_means() forms the per-group means
  //   over this same subset, keeping the stabilised Casorati matrix exactly
  //   rank-deficient by null_rank().
  void set_temporal_subsample(default_type fraction, Math::RNG &rng, ssize_t min_per_group);

  // Header of the preconditioned data: the full serialised header, or, while a
  //   temporal subset is active, the subset header (size(3) == m').
  const Header &header() const { return temporal_subset.empty() ? H_out : H_out_subset; }

  // Rank removed by group-mean demeaning: the number of demeaning groups (1 for -demean all;
  //   shell/volume-group count otherwise; 0 for -demean none). This is reported irrespective of
  //   whether the demeaning is applied here (no partitioning) or per partition within
  //   Estimate/Recon (partitioning) — the degrees of freedom regressed out are the same group
  //   count either way (per the partitioned-model expectation that the demean-induced rank change
  //   matches the non-partitioned case). Used to add the regressed-mean components back to the
  //   final exported rank maps.
  ssize_t demean_rank() const {
    if (!vst_mean_image.valid())
      return 0;
    if (vst_mean_image.ndim() == 3)
      return 1;
    return vst_mean_image.size(3);
  }

  ssize_t null_rank() const {
    // The rank that *this* (preconditioner-side) demeaning regresses out of the Casorati matrix.
    //   When partitioning is active the preconditioner performs no demeaning (it is done per
    //   partition inside Estimate/Recon, which account for it via their own per-partition rank),
    //   so it is 0; otherwise it equals demean_rank().
    return partitioning_active ? 0 : demean_rank();
  }

  bool noop() const {
    return (temporal_subset.empty() && num_volume_groups == 1 && !phase_image.valid() &&
            !vst_mean_image.valid() && !vst_noise_image.valid());
  }

private:
  const Header H_in;
  Header H_out;
  // Subset header (a copy of H_out with size(3) == m') returned by header() while a
  //   temporal subset is active; the preconditioned data then hold only the subset volumes.
  Header H_out_subset;
  // Current temporal subset: sorted original (serialised) volume indices used for noise
  //   level estimation in this iteration; empty ⇒ all volumes (no temporal sub-sampling).
  std::vector<ssize_t> temporal_subset;
  // When true, demeaning is delegated to the per-partition Estimate/Recon path and the
  //   preconditioner applies only phase demodulation and the variance-stabilising transform.
  bool partitioning_active = false;
  // For serialisation of >4D images
  ssize_t num_volume_groups;
  Image<uint32_t> serialise_image;
  // Noise distribution governing the variance-stabilising transform (VST);
  //   scalar configuration shared by forward stabilisation and inverse mapping.
  std::shared_ptr<NoiseModel::Base> noise_model;
  // First step: Phase demodulation
  Image<cfloat> phase_image;
  // Adaptive phase correction (Pizzolato APC): each update_parameters() call re-estimates
  //   phase_image from the empirical complex input, driven by the current noise map (or, on
  //   the first iteration when no map exists yet, a self-calibrated global noise level). The
  //   estimator itself is stateless/cheap to copy; its per-thread scratch is allocated inside
  //   its threaded driver, so Preconditioner remains trivially copyable.
  AdaptivePhaseEstimator apc;
  bool apc_enabled = false;     // -demodulate apc on complex data
  bool apc_first_pass = true;   // false after the first APC estimation: switches the estimator
                                //   from the cold solve to warm-started native-resolution
                                //   refinement (see update_parameters / AdaptivePhaseEstimator)
  bool apc_coarse_first = false; // solve the first (cold) APC pass on a 2x-downsampled grid; set
                                 //   by set_apc_coarse_first iff the schedule has >1 iteration
  // Second step (forward): variance-stabilising transform.
  //   The noise level map is the VST scale / dispersion parameter.
  Image<float> vst_noise_image;
  // Third step (forward): demeaning, performed in the stabilised domain.
  //   The stored means are the per-group means of the stabilised data
  //   (the VST offset parameter; sigma-dependent), not the empirical magnitude means.
  std::vector<ssize_t> index2shell;
  std::vector<ssize_t> index2group;
  Image<T> vst_mean_image;

  // Serialise this voxel's volumes into "data", applying phase demodulation and,
  //   where a noise level is available, the forward variance-stabilising transform.
  // This is the common forward preprocessing shared by the Casorati fill
  //   (operator() forward) and the stabilised-domain mean computation.
  void serialise_and_stabilise(Image<T> &input,
                               Image<cfloat> &phase,
                               Image<uint32_t> &serialise,
                               const Transform &transform,
                               Interp::Cubic<Image<float>> *vst,
                               Eigen::Array<T, Eigen::Dynamic, 1> &data) const;

  // (Re)compute the stabilised-domain per-group means from the input,
  //   using the currently stored noise level map for stabilisation.
  void compute_means(Image<T> input);

  // Re-estimate phase_image in place from the empirical complex input and the current noise
  //   level map (adaptive phase correction). A no-op for real T (the phase is ill-defined),
  //   guarded internally by `if constexpr (is_complex<T>::value)` so the explicit
  //   instantiations for real T compile.
  void update_phase(Image<T> &input);
};

} // namespace MR::Denoise::Precondition
