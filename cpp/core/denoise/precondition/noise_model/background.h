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
#include <limits>
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
// INITIALISATION, which departs from the paper. Algorithm 1 begins each slice by trying a grid of
//   candidate noise levels from zero up to a ceiling derived from the median of the data, keeping
//   whichever accepts the most voxels. Where background does not dominate a slice, that maximum lies
//   within tissue; where it does, a ceiling derived from one global statistic can fall short of the
//   background. Each slice is instead initialised at the lowest mode of its own histogram of
//   voxel-wise mean squared magnitudes that holds enough voxels to be background (see
//   estimate_background()).
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
//   5. Zero-valued samples are treated as censored below a threshold where their number is
//      consistent with that (see estimate_background()), which corrects the overestimate that
//      excluding zeros produced from integer-rounded or clipped data. It cannot correct rounding
//      bias in the samples that are not zero, which remains where the noise level is within a
//      couple of storage quanta of zero; nor does it follow a clip level that varies WITHIN a
//      slice, where the zeros are tested against the lowest and are typically then excluded.
//   6. A slice holding fewer than min_voxels_slice background voxels, but enough dark tissue to
//      form a mode of that size, is initialised within the tissue and converges there. The
//      selection of every converged slice is tested for consistency with a single Gamma
//      distribution (see BackgroundParameters::gamma_ratio_floor), which declines most such slices;
//      but low-signal tissue under a Rician (two degree of freedom) distribution can pass, as its
//      shape is then close to the Gamma it is fitted with. Those that pass are flagged only where
//      they make the per-slice estimates disagree (see estimate_caveats()). The test is equally
//      blind to background whose magnitudes were averaged spatially with uniform weights, which
//      remains Gamma-distributed but with inflated degrees of freedom.

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
// - SLICE: each slice keeps its own; every slice holding data must yield one (see slice_outcome_t).
enum class background_scope_t { GLOBAL, SLICE };

// What one slice yielded.
// - ESTIMATED: a converged estimate that passed every check.
// - NO_SIGNAL: the slice holds no data at all -- every sample of every voxel is zero or non-finite,
//     as in the zero-filled slices that padding or resampling of the field of view leaves. Nothing
//     can be estimated there, but neither is anything lost: whatever the transform applied, the
//     voxels of the slice are constant across volumes. A slice of pure noise is NOT of this kind; it
//     is background throughout, and is fitted like any other.
// - FAILED: the slice holds data, but no estimate could be obtained from them.
enum class slice_outcome_t { ESTIMATED, NO_SIGNAL, FAILED };

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
  // dof_min sets the width of the REJECTION WINDOW on the first outer iteration, which is the
  //   central 1 - p interval of the broadest admissible background distribution, centred on the
  //   lowest mode of the slice (see estimate_background()). dof_max bounds the plausibility ceiling
  //   below. Neither bounds the estimate: an estimate outside them is perfectly admissible -- the
  //   paper's own GE result of 0.45 effective channels was obtained with a lower bound of one.
  default_type dof_min{2.0};
  default_type dof_max{24.0};
  // Family-wise probability that sampling fluctuation of the counts alone produces a local maximum
  //   of a slice's histogram that is accepted as a mode (see estimate_background()).
  default_type mode_significance{0.05};
  // Outer-iteration cap and the convergence tolerance on (dof, sigma), absolute or relative.
  ssize_t max_iter{100};
  default_type eps{1.0e-3};
  // Guards. A slice yielding fewer than min_voxels_slice background voxels yields no estimate:
  //   PIESNO's failure mode on the paper's GE data was identifying "about 10 noise only voxels",
  //   and a floor makes that visible rather than silent. The same floor is the least number of
  //   voxels a mode of a slice's histogram must comprise to initialise the iteration there. The
  //   image as a whole must furnish min_voxels_total selected voxels across at least min_slices
  //   slices.
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
  // Consistency test of zero-valued samples with censoring (see estimate_background()). The zeros
  //   among a slice's selected samples are judged NOT to be censored noise where a one-sided test
  //   rejects, at significance zero_consistency_p, the hypothesis that they occur at no more than
  //   zero_excess_factor times the rate predicted by a fit to the positive samples alone. The factor is a tolerance for
  //   the model of censoring being itself approximate: integer rounding also displaces the
  //   magnitudes that do not round to zero, so the observed zero rate departs from the fitted one
  //   by a few percent even where the zeros are genuinely rounded noise, and with 1e5 samples per
  //   slice any departure at all is statistically significant. Zeros that are not noise (regions
  //   zero-filled by reconstruction or resampling, or negative values clipped from a real-valued
  //   reconstruction) exceed the predicted rate by an order of magnitude or more.
  default_type zero_consistency_p{1.0e-6};
  default_type zero_excess_factor{2.0};
  // Consistency of a converged slice's selection with a single Gamma distribution. Nothing in the
  //   iteration otherwise tests this: the window follows the fit and the fit follows the window, so
  //   it converges on whatever compact cluster the first window lands on, background or not. The
  //   two estimators are fitted to the same selected samples, and the ratio of their degrees of
  //   freedom, moments over maximum likelihood, is formed. For a Gamma sample both are consistent
  //   and the ratio tends to one. A mixture pulls them apart, because the method of moments sees
  //   the mixture's excess variance in E[t^2] while maximum likelihood sees it only through E[log t]:
  //   for a mixture of noise levels whose log-variance has standard deviation s, the ratio is about
  //   1 - s^2. Tissue whose signal varies between voxels and between volumes, and a noise level
  //   that differs between volumes, are both such mixtures. The slice declines where the ratio lies
  //   below gamma_ratio_floor AND more than gamma_ratio_z standard errors below one. The standard
  //   error is that of the difference between an inefficient and an efficient estimator of the
  //   same parameter, Var(log moments) - Var(log ML): with N = dof/2 and n samples, that is
  //   [2 (N+1) / N - 1 / (N^2 (psi'(N) - 1/N))] / n. Neither condition suffices alone: at 1e5
  //   samples the estimators' small-sample biases are themselves significant, and at 1e3 samples a
  //   ratio of 0.95 is within sampling variation.
  // Calibrated on synthetic phantoms only (K = 2 to 200 volumes, 1 to 16 degrees of freedom, 16^2 to
  //   384^2 matrices, ghosting, scalp, in-plane and through-plane gradients of noise level, scarce
  //   background). Over 1720 slices whose selection held no tissue, the lowest ratio was 0.947 and
  //   the lowest standardised value -2.55, with no slice satisfying both conditions. The window
  //   truncation biases the ratio upward, not downward, at few volumes (up to 1.10 at K = 2).
  //   Over 102 slices whose selection was more than 5% tissue, both conditions held for 99, with
  //   ratios of 0.70 to 0.94. Those missed are low-SNR tissue at two degrees of freedom (ratios 0.96
  //   to 1.02). On the six resampled series of MRtrix3's script_test_data, the selections of all
  //   but one series carried signal (their mean power in b = 0 volumes was 9 to 18 times that in
  //   diffusion-weighted volumes), and those slices gave ratios of 0.20 to 0.82. Not calibrated
  //   against raw multi-coil data, whose noise level may vary within a slice more than any phantom's.
  default_type gamma_ratio_floor{0.95};
  default_type gamma_ratio_z{3.5};
};

// Sufficient statistics of one spatial voxel, accumulated over the K volumes of the input.
//   These numbers are sufficient for BOTH estimators and for the acceptance test, which is what
//   makes the algorithm cheap: the 4D data are read exactly once and everything thereafter
//   operates on a 3D array.
class VoxelStats {
public:
  // Volumes at which the magnitude was finite and strictly positive; the three sums run over
  //   these alone, log(m^2) being what the likelihood is written in.
  uint32_t count{0};
  // Volumes at which the magnitude was exactly zero. These are not discarded: a stored zero states
  //   that the magnitude lay below the censoring threshold (see Accumulation::censoring_threshold()),
  //   and enters the fit as such. Non-finite samples are neither counted here nor in count.
  uint32_t num_zero{0};
  // The smallest positive magnitude, from which the censoring threshold of the voxel's slice is
  //   determined. Single precision suffices, and keeps the structure at forty bytes.
  float min_positive{std::numeric_limits<float>::infinity()};
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
//   over the 4D data that computing the median beforehand would require. (The paper also uses that
//   median to bound its initial noise-level search; this implementation initialises differently,
//   and needs no global statistic at all.)
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
  // Number of volumes accumulated per voxel, from which the number of samples that VoxelStats::count
  //   excluded can be recovered. Zero where unknown (an Accumulation not filled by
  //   accumulate_background()), in which case nothing that depends on it is reported.
  uint32_t num_volumes{0};
  // The storage quantum: the constant of which every magnitude is an integer multiple, where the
  //   data are evidently so (integer storage, with or without intensity scaling), and zero otherwise.
  //   Detected from the smallest distinct magnitudes of the image (see accumulate_background()).
  default_type quantum{0.0};
  // The magnitude below which a stored zero is taken to have lain, given the smallest positive
  //   magnitude m_min among the samples concerned: tau = m_min - quantum/2. For integer data whose
  //   zeros arise from rounding, this is half the quantum; for floating-point data clipped at a
  //   level t, it is (the sample estimate of) t; for both together, t - quantum/2. Zero where it
  //   cannot be determined, in which case zeros cannot be censored and are excluded.
  //   It is evaluated per slice (see estimate_background()), so that a clip level that differs
  //   between slices -- one set relative to a noise level that does, for instance -- is followed.
  default_type censoring_threshold(const default_type min_positive) const;
};

// Accumulate the statistics above from a magnitude image, in one pass, together with the
//   smallest positive magnitude and the storage quantum. Explicitly instantiated for
//   the real-valued image types the commands process; complex data have two degrees of freedom per
//   sample by construction and are never passed here.
template <class ImageType> Accumulation accumulate_background(ImageType &input, const size_t slice_axis);

// The sums over a SELECTED SET of samples, from which either estimator is evaluated. "count" is
//   the number of exactly observed (positive) samples, voxel-volume pairs rather than voxels, over
//   which the three sums run. "num_zero" is the number of samples observed only to lie below the
//   censoring threshold, whose square is "threshold_sq"; where it is zero the fit is precisely that
//   of the exactly observed samples, and threshold_sq is not consulted.
class SampleSums {
public:
  size_t count{0};
  default_type sum_sq{0.0};
  default_type sum_quad{0.0};
  default_type sum_log_sq{0.0};
  size_t num_zero{0};
  default_type threshold_sq{0.0};
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

// Fit (sigma, dof) to a selected set of background samples by one of the two methods, treating
//   any zero-valued samples as censored at the threshold the sums carry.
//   Exposed rather than kept internal to the algorithm because it is the piece whose statistical
//   behaviour has to be characterised on its own (bias and scatter against a known truth), and
//   because it is the cheapest possible cross-check of one method against the other once the sums
//   exist.
Estimate estimate_from_sums(const SampleSums &sums, const background_method_t method);

// The outcome for one slice.
class SliceEstimate {
public:
  ssize_t index{0};
  slice_outcome_t outcome{slice_outcome_t::FAILED};
  bool estimated() const { return outcome == slice_outcome_t::ESTIMATED; }
  default_type dof{0.0};
  default_type sigma{0.0};
  // Voxels accepted as background, and outer iterations taken. Both are reported even where the
  //   slice declined, since "how many voxels did it find" is the first question a failure raises.
  size_t num_selected{0};
  ssize_t num_iterations{0};
  // The censoring threshold of this slice (see Accumulation::censoring_threshold()); zero where the
  //   slice holds no zeros, or none could be determined.
  default_type censoring_threshold{0.0};
  // Samples (voxel-volume pairs) of the final selection: all of them, and the zeros among them,
  //   which are censored samples unless zeros_excluded is set.
  size_t num_samples{0};
  size_t num_zero{0};
  // The sums of the final selection, from which the consistency of the zeros with censoring is
  //   judged.
  SampleSums sums;
  // Whether zeros were present in the slice but excluded from its fit: either their number was
  //   inconsistent with their being censored noise, or no censoring threshold could be determined,
  //   or the censored fit declined. The estimate is then that of the positive samples alone, as the
  //   paper's reference implementation computes it everywhere.
  bool zeros_excluded{false};
  // The test of the final selection's consistency with a single Gamma distribution (see
  //   BackgroundParameters::gamma_ratio_floor): the ratio of the degrees of freedom fitted to it by
  //   the method of moments to that fitted by maximum likelihood, and the logarithm of that ratio
  //   in units of its standard error. NaN where the iteration did not converge, or where either fit
  //   was refused. Set whether the test passed or not, so that a declined slice reports by how much.
  default_type gamma_ratio{std::numeric_limits<default_type>::quiet_NaN()};
  default_type gamma_ratio_z{std::numeric_limits<default_type>::quiet_NaN()};
  // Why this slice yielded no estimate; empty iff estimated().
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
  // Of all samples (voxel-volume pairs) of the voxels selected as background, the proportion that
  //   were excluded from the fit: non-finite samples, and zeros in slices where they were not
  //   treated as censored (SliceEstimate::zeros_excluded). Not a decline criterion but a diagnostic
  //   (see estimate_caveats()).
  default_type excluded_fraction{0.0};
  // Of the same samples, the proportion that were zero and entered the fit as censored.
  default_type censored_fraction{0.0};
  // As Accumulation::quantum.
  default_type quantum{0.0};
  // Number of slices whose zeros were excluded rather than censored (SliceEstimate::zeros_excluded).
  size_t num_slices_zeros_excluded{0};
  // As Accumulation::num_volumes.
  uint32_t num_volumes{0};
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
  //   and the global median otherwise (which slice-wise scope permits only for NO_SIGNAL slices).
  //   Requires estimated == true.
  default_type dof_at_slice(const ssize_t slice) const;
  // Number of slices with the given outcome.
  size_t count(const slice_outcome_t outcome) const;
};

// Run the paper's Algorithm 1 over every slice of an accumulated image and reduce the result.
BackgroundResult estimate_background(const Accumulation &accumulation,
                                     const background_method_t method,
                                     const BackgroundParameters &parameters = BackgroundParameters());

// Warnings about a SUCCESSFUL estimate, one string per condition detected, for the caller to issue.
//   These flag (and do not correct) the two ways in which the estimator has been observed to return
//   a confident but wrong value on synthetic data:
// - The per-slice estimates are spread far more widely than sampling variation explains. Where the
//   field of view holds too little air, the voxels selected in some slices lie within the head;
//   the plausibility ceiling does not catch this because the value obtained is low rather than
//   high (around two to three degrees of freedom whatever the truth), but it varies from slice to
//   slice with the tissue selected, whereas genuine background gives per-slice estimates that agree
//   to within about 2% even from few volumes. A genuinely spatially varying distribution may also
//   produce a spread, which the message acknowledges. Most slices whose selection lies within the
//   head are declined by the test of consistency with a single Gamma distribution
//   (BackgroundParameters::gamma_ratio_floor), and this caveat is what remains for those that pass
//   it (low-signal tissue at around two degrees of freedom), and for disagreement between slices
//   that each pass it.
// - More than a small proportion of the samples of the selected voxels are zeros that were excluded
//   from the fit (BackgroundResult::excluded_fraction) because they could not be explained as noise
//   rounded or clipped below the censoring threshold. If they are nonetheless low magnitudes (clipped
//   at a level that varies across the image, say), the low tail of the distribution is missing from
//   the fit and the degrees of freedom are overestimated.
// Empty for a result that did not produce an estimate.
std::vector<std::string> estimate_caveats(const BackgroundResult &result);

// The image axis along which slices are separated, taken from "SliceEncodingDirection" where the
//   header provides it and otherwise assumed to be the third spatial axis.
//   The caveat recorded at precondition.cpp for the demodulation axes applies equally here: ideally
//   this would be resolved against the axes ON DISK, before transform realignment.
size_t select_slice_axis(const Header &H);

} // namespace MR::Denoise::Precondition::NoiseModel
