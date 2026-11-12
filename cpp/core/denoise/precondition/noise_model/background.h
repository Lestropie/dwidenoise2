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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "header.h"
#include "types.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Estimation of the magnitude noise distribution's degrees of freedom from the voxels of the data
//   that carry no signal, following St-Jean et al. (2020), "Automated characterization of noise
//   distributions in diffusion MRI data", Medical Image Analysis 65:101758.
//
// THE IDEA. For a voxel carrying no signal, the magnitude m of a sum-of-squares reconstruction
//   over dof/2 effective channels satisfies t = m^2 / (2 sigma^2) ~ Gamma(dof/2, 1) exactly (Koay
//   et al. 2009b; the change of variable PIESNO is built on). A Gamma(alpha, 1) has both mean and
//   variance equal to alpha, so the SHAPE parameter -- which is precisely the quantity wanted --
//   is identified by the sample moments of t alone. Summing over K volumes at one voxel gives
//   Gamma(K dof/2, 1) by the additivity of Gamma variates at fixed rate, which is what makes a
//   per-voxel acceptance test possible.
//
// WHY IT IS THE RIGHT PLACE TO ESTIMATE. The alternative -- fitting the degrees of freedom to the
//   residual of a filtered PCA reconstruction -- fails because what the filter does not capture
//   leaks into that residual in an SNR-dependent way several times larger than the signature being
//   measured. Every mechanism of that failure is absent here: there is no PCA, no filter and no
//   projector; no variance-stabilising transform has been applied, so the change of variable above
//   is exact rather than first-order; and the pass is a one-shot pre-pass outside the iteration
//   schedule, coupled to nothing. The same underlying estimator recovers the truth to within 9% on
//   patches carrying no signal subspace, and the acceptance test below is a principled way of
//   selecting exactly such samples.
//
// WHAT IT DOES NOT SOLVE, stated up front and repeated in the command-line help:
//   1. It requires background voxels. Data with no air in the field of view, or whose background
//      the reconstruction has already zeroed, carry no information for this estimator; the guards
//      below detect that and decline rather than return a number.
//   2. Degrees of freedom in the background are not degrees of freedom in tissue under k-space
//      parallel imaging (GRAPPA), where the effective count depends on the local signal (the
//      paper's Section 4.1 and Fig. 6). A background estimate is then systematically an
//      over-estimate in the interior. The paper calls accurate estimation over signal regions an
//      open challenge; this remains a large improvement on assuming Rician.
//   3. Multiband slice leakage inflates the estimate (the paper's Figs. 7A/7B), by folding aliased
//      signal into what should be background.
//   4. It estimates sigma too, and that value is NOT used to set the noise level. Background sigma
//      understates in-brain sigma under parallel imaging (the paper's Section 4.3); the per-patch
//      Marchenko-Pastur noise map that the iteration schedule produces remains the sole source of
//      the variance-stabilising transform's scale. The sigma here is computed because the degrees
//      of freedom cannot be estimated without it, is exported as a diagnostic, and is otherwise
//      discarded.

// Which of the two solutions of the same model is used to fit a selected set of background
//   samples; see estimate_from_sums().
// - NONE: no estimation is performed (the degrees of freedom come from the command line or from
//     the header metadata, as before).
// - MOMENTS: the closed-form method-of-moments equations (the paper's Eqs. 8 and 9).
// - ML: the maximum-likelihood equations (the paper's Eqs. 11 and 12 and its Appendix A), which
//     the paper reports as having the lower variance in every comparison where the two differ.
enum class background_method_t { NONE, MOMENTS, ML };

// Whether the per-slice estimates are reduced to one value for the whole image or retained.
// - GLOBAL: their median (see BackgroundResult::dof).
// - SLICE: each slice keeps its own, with the global median standing in for slices that declined.
enum class background_scope_t { GLOBAL, SLICE };

// Tuning constants of the algorithm. The first group follows the paper and its reference
//   implementation; the second group is this implementation's guards against answering where the
//   data cannot support an answer (there is no such thing as a failed estimate that is harmless
//   here: the value silently parameterises the noise floor of every subsequent computation).
class BackgroundParameters {
public:
  // Probability mass excluded by the acceptance interval: a voxel is accepted as background only
  //   if its summed statistic falls within the central 1 - p of the Gamma distribution implied by
  //   the current parameters. This is a REJECTION step, and it is what makes the method automatic
  //   rather than mask-dependent: it discards voxels containing tissue signal, and equally
  //   discards voxels corrupted by ghosting, slice leakage or spikes.
  default_type p{0.05};
  // Number of candidate noise levels in the initial search grid (the paper's l).
  ssize_t num_candidates{50};
  // Bounds of the REJECTION WINDOW on the first outer iteration, not of the estimate: the interval
  //   is then evaluated with its lower end at dof_min and its upper end at dof_max so as to span
  //   the whole admissible range (the paper's Fig. 1C). An estimate outside these bounds is
  //   perfectly admissible -- the paper's own GE result of 0.45 effective channels was obtained
  //   with a lower bound of one.
  default_type dof_min{2.0};
  default_type dof_max{24.0};
  // Outer-iteration cap and the convergence tolerance on (dof, sigma), absolute or relative.
  ssize_t max_iter{100};
  default_type eps{1.0e-3};
  // Guards. A slice yielding fewer than min_voxels_slice background voxels yields no estimate:
  //   PIESNO's failure mode on the paper's GE data was identifying "about 10 noise only voxels",
  //   and a floor makes that visible rather than silent. The image as a whole must furnish
  //   min_voxels_total selected voxels across at least min_slices slices.
  size_t min_voxels_slice{100};
  size_t min_voxels_total{1000};
  size_t min_slices{3};
  // Plausibility ceiling on the estimate, as a multiple of dof_max: a value above it indicates
  //   that what was selected is not background at all.
  default_type dof_ceiling_factor{2.0};
  // Fraction of the field of view that may be identically zero across every volume before the
  //   estimator declines outright. A reconstruction that has masked the background leaves nothing
  //   to estimate from, and what remains adjacent to the mask boundary is not background either.
  default_type max_zeroed_fraction{0.5};
};

// Sufficient statistics of one spatial voxel, accumulated over the K volumes of the input.
//   These four numbers are sufficient for BOTH estimators and for the acceptance test, which is
//   what makes the algorithm cheap: the 4D data are read exactly once and everything thereafter
//   operates on a 3D array.
class VoxelStats {
public:
  // Volumes at which the magnitude was finite and strictly positive. Zero and non-finite samples
  //   are excluded rather than counted, since log(m^2) is what the likelihood is written in.
  uint32_t count{0};
  default_type sum_sq{0.0};     // sum_k m^2
  default_type sum_quad{0.0};   // sum_k m^4
  default_type sum_log_sq{0.0}; // sum_k log(m^2)
};

// The 3D array of per-voxel statistics, together with the geometry needed to walk it slice by
//   slice.
//
// NOTE ON SCALE. The paper normalises the magnitudes by the dataset median before accumulating,
//   so that m^4 can neither overflow nor lose precision. Accumulating in double precision instead
//   makes that unnecessary -- m^4 would have to exceed 1e300 to overflow, and the relative
//   precision of a floating-point sum does not depend on its scale -- and it saves the extra pass
//   over the 4D data that computing the median beforehand would require. The one place a typical
//   magnitude is genuinely needed, namely the top of the initial noise-level search grid, is
//   instead derived from these accumulators (see estimate_background()).
// Linear index of a spatial voxel within Accumulation::voxels and BackgroundResult::selected: the
//   first axis varies fastest. A free function so that a consumer of the exported mask can address
//   it without constructing an Accumulation, which would allocate the whole statistics array (forty
//   bytes per voxel) merely to perform this arithmetic.
inline size_t voxel_index(const std::array<ssize_t, 3> &size, //
                          const ssize_t i0,                   //
                          const ssize_t i1,                   //
                          const ssize_t i2) {                 //
  return size_t((i2 * size[1] + i1) * size[0] + i0);
}

class Accumulation {
public:
  Accumulation() : size{{0, 0, 0}}, slice_axis(2) {}
  Accumulation(const std::array<ssize_t, 3> &size, const size_t slice_axis);
  size_t index(const ssize_t i0, const ssize_t i1, const ssize_t i2) const {
    return voxel_index(size, i0, i1, i2);
  }
  size_t num_voxels() const { return voxels.size(); }
  // Proportion of spatial voxels at which no volume provided a usable magnitude.
  default_type zeroed_fraction() const;
  std::vector<VoxelStats> voxels;
  std::array<ssize_t, 3> size;
  size_t slice_axis;
};

// Accumulate the statistics above from a magnitude image, in one pass. Explicitly instantiated for
//   the real-valued image types the commands process; complex data have two degrees of freedom per
//   sample by construction and are never passed here.
template <class ImageType> Accumulation accumulate_background(ImageType &input, const size_t slice_axis);

// The sums over a SELECTED SET of samples, from which either estimator is evaluated. "count" is
//   the number of samples (voxel-volume pairs), not of voxels.
class SampleSums {
public:
  size_t count{0};
  default_type sum_sq{0.0};
  default_type sum_quad{0.0};
  default_type sum_log_sq{0.0};
};

// One fit of the Gamma model to a selected set.
class Estimate {
public:
  explicit operator bool() const { return valid; }
  bool valid{false};
  default_type sigma{0.0};
  default_type dof{0.0};
  // Why the fit was refused, where it was; empty otherwise.
  std::string failure;
};

// Fit (sigma, dof) to a selected set of background samples by one of the two methods.
//   Exposed rather than kept internal to the algorithm because it is the piece whose statistical
//   behaviour has to be characterised on its own (bias and scatter against a known truth), and
//   because it is the cheapest possible cross-check of one method against the other once the sums
//   exist.
Estimate estimate_from_sums(const SampleSums &sums, const background_method_t method);

// The outcome for one slice.
class SliceEstimate {
public:
  ssize_t index{0};
  bool estimated{false};
  default_type dof{0.0};
  default_type sigma{0.0};
  // Voxels accepted as background, and outer iterations taken. Both are reported even where the
  //   slice declined, since "how many voxels did it find" is the first question a failure raises.
  size_t num_selected{0};
  ssize_t num_iterations{0};
  // Why this slice yielded no estimate; empty iff estimated.
  std::string decline;
};

// The outcome for the image.
class BackgroundResult {
public:
  explicit operator bool() const { return estimated; }
  bool estimated{false};
  // The median of the per-slice estimates over the slices that produced one.
  //   Median, not mean: the extreme slices of a volume are partly outside the head, and slices
  //   contaminated by wrap or leakage are exactly what a median is robust to. Explicitly NOT a
  //   pooled refit over all slices' selected voxels -- sigma genuinely varies between slices,
  //   which is the whole reason the paper works per slice, so pooling the magnitudes would fit one
  //   sigma to a heterogeneous sample and bias dof through it. dof is pooled by taking a median of
  //   per-slice fits; sigma is never pooled at all.
  default_type dof{0.0};
  // The median of the per-slice noise levels. A DIAGNOSTIC ONLY: it is not the noise level used
  //   by anything (see point 4 of the discussion at the head of this file).
  default_type sigma{0.0};
  // Interquartile range of the per-slice degrees of freedom. A large spread is the signature of a
  //   spatially varying distribution -- which is what parallel imaging produces -- and is the cue
  //   to reach for slice-wise scope.
  default_type dof_iqr{0.0};
  size_t slice_axis{2};
  std::vector<SliceEstimate> slices;
  size_t num_selected_total{0};
  default_type zeroed_fraction{0.0};
  // Why no global estimate was produced; empty iff estimated.
  std::string decline;
  // Per spatial voxel: was this voxel accepted as background by the iteration that converged?
  //   Laid out as Accumulation::index() and sized only where an estimate was attempted. This is
  //   the diagnostic the paper leans on throughout (its Figs. 4D, 5D, 8C, 9A, 10D): the single
  //   most informative check on whether the method did what it should is a picture of which voxels
  //   it chose.
  std::vector<uint8_t> selected;
  std::array<ssize_t, 3> size{{0, 0, 0}};
  // The degrees of freedom to apply at a given slice: that slice's own estimate where it has one,
  //   and the global median otherwise. Requires estimated == true.
  default_type dof_at_slice(const ssize_t slice) const;
};

// Run the paper's Algorithm 1 over every slice of an accumulated image and reduce the result.
BackgroundResult estimate_background(const Accumulation &accumulation,
                                     const background_method_t method,
                                     const BackgroundParameters &parameters = BackgroundParameters());

// The image axis along which slices are separated, taken from "SliceEncodingDirection" where the
//   header provides it and otherwise assumed to be the third spatial axis.
//   The caveat recorded at precondition.cpp for the demodulation axes applies equally here: ideally
//   this would be resolved against the axes ON DISK, before transform realignment.
size_t select_slice_axis(const Header &H);

} // namespace MR::Denoise::Precondition::NoiseModel
