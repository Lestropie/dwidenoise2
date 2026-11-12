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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "app.h"
#include "denoise/denoise.h"
#include "denoise/precondition/noise_model/background.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "header.h"

namespace MR::Denoise::Precondition {

extern const char *const demodulation_description;
extern const char *const noise_channels_description;

// Phase demodulation applied to complex data before PCA. This [none, linear, hann, apc]
//   classification refines the historical [none, linear, nonlinear]: both "hann" and "apc"
//   are nonlinear-phase methods (they are not an independent axis of choice).
// - NONE: no phase demodulation.
// - LINEAR: regress a strictly linear phase ramp from each k-space (Cordero-Grande et al.
//     2019); estimated once and held for every iteration.
// - HANN: a fixed full-extent Hann-window nonlinear k-space phase, estimated once and held
//     for every iteration (the previous "nonlinear" behaviour; retained for comparison).
// - APC: noise-adaptive nonlinear phase (Pizzolato et al. 2020; see
//     MR::Denoise::Precondition::AdaptivePhaseEstimator).
//     Re-estimated every noise-estimation iteration from the empirical complex data by a
//     noise-weighted total-variation smoothing. The first iteration, which has no noise map
//     yet, self-calibrates from a data-derived global noise level with uniform weighting.
//     This is the default for complex input data.
enum class demodulation_t { NONE, LINEAR, HANN, APC };

enum class demean_type { NONE, VOLUME_GROUPS, SHELLS, ALL };

// Handling of the noise-distribution bias when reversing preconditioning
//   (the inverse variance-stabilising transform); see vst_plan.md section 3.3.
// - DEBIAS: map the per-group operating point (DC term) to the bias-free
//     underlying signal level, removing the magnitude noise-floor bias (the
//     residual "haze"); the denoised fluctuations are mapped by a linear gain,
//     so their Gaussian character and the homoscedasticity correction are retained.
// - PRESERVE: map the operating point to the conventional biased-magnitude level,
//     reproducing magnitude-scale output with the noise floor retained.
// For complex (Gaussian) data there is no distribution bias and the two modes coincide.
enum class bias_handling_t { DEBIAS, PRESERVE };

// Operating point at which the non-linear inverse variance-stabilising transform (and its
//   noise-floor debiasing) is evaluated when reversing preconditioning:
// - SAMPLE: undo the demeaning offset first, then apply the inverse pointwise at each
//     volume's own denoised value (operating point = group mean + denoised residual). The
//     demeaning is treated purely as PCA conditioning and is reversed exactly before the
//     inverse transform; debiasing is then independent of the demeaning grouping, and the
//     natural (signal-dependent) heteroscedasticity is restored on the output scale. (default)
// - GROUP_MEAN: linearise the inverse about the per-group stabilised-domain mean (the
//     demeaning offset), mapping the denoised residual through the local Jacobian. Reproduces
//     the prior behaviour; debiasing accuracy then depends on how far each volume departs from
//     its group mean. Applies only to the DEBIAS bias handling: the PRESERVE (algebraic)
//     inverse is always evaluated pointwise (a faithful, grouping-independent reversal).
enum class debias_anchor_t { SAMPLE, GROUP_MEAN };

App::OptionGroup precondition_options(const option_scope_t scope);

// Everything the command line says about the magnitude noise model, parsed and cross-validated in
//   one place.
//
// It is resolved BEFORE the input image is read, because one of the things it decides is whether a
//   pass over that image is made at all: estimating the degrees of freedom from the data costs a
//   read of them, and there is no point paying for it to produce a number that the requested
//   transform would ignore or that an explicit -noise_dof would override. Separating the parsing
//   from the construction of the model is what makes that possible; make_noise_model() then
//   receives both this and whatever the estimation produced.
class NoiseModelOptions {
public:
  // The degrees of freedom the user imposed, and the name of the option through which they did so
  //   ("noise_channels" or "noise_dof"), for messages that quote it back. A channel count is
  //   doubled at the point of parsing, so everything below deals only in degrees of freedom.
  std::optional<default_type> user_dof;
  std::string user_option;
  // Whether the input data are complex, for which the demodulated noise is Gaussian and the
  //   magnitude noise model does not apply. Recorded here, having been established once by the
  //   calling command, so that make_noise_model() need not be told a second time.
  bool complex{false};
  NoiseModel::vst_method_t vst_method{NoiseModel::vst_method_t::FOI};
  // Estimation of the degrees of freedom from the voxels of the data that contain noise alone:
  //   which estimator (NONE where no such pass is to be performed, whether because none was asked
  //   for or because something else makes it pointless), the scope over which its result is
  //   applied, and the image axis that separates slices.
  NoiseModel::background_method_t background_method{NoiseModel::background_method_t::NONE};
  NoiseModel::background_scope_t background_scope{NoiseModel::background_scope_t::GLOBAL};
  size_t background_slice_axis{2};
};

// Parse and cross-validate the options above, refusing combinations that cannot be honoured
//   coherently and warning about those that are merely redundant.
// "complex" reports whether the input data are complex, for which the noise is Gaussian and every
//   magnitude-specific option is meaningless; "H" is the header of the input image as opened by the
//   calling command, consulted only for the slice encoding direction.
NoiseModelOptions noise_model_options(const bool complex, const Header &H);

// Estimate the degrees of freedom of the magnitude noise distribution from the voxels of the input
//   that contain noise alone (St-Jean et al. 2020; see noise_model/background.h), and write the
//   diagnostics requested via -noise_dof_out / -noise_voxels_out.
// Returns an unpopulated result where no estimation was requested, and one carrying the reason
//   where it was requested but could not be completed; in both cases the caller falls through to
//   the next route by which the degrees of freedom may be obtained.
// Declared for the real-valued image types only: complex data have two degrees of freedom per
//   sample by construction, so the calling command excludes this pass at compile time.
template <class ImageType>
NoiseModel::BackgroundResult estimate_background_dof(ImageType &input, const NoiseModelOptions &options);

// Construct the noise model(s) governing the variance-stabilising transform.
// For complex (or phase-demodulated) data the Gaussian model is returned, as a set of one.
// For magnitude data the degrees of freedom select a Rician (dof == 2) or non-central chi (any
//   other value) model, built with the requested -vst_method strategy. The degrees of freedom are
//   real-valued and are twice the EFFECTIVE receive-channel count, which is not a coil element
//   count. They are resolved in order of precedence from -noise_dof / -noise_channels, then
//   "background" (the estimate made from the data themselves), then the key-values of "H" (see
//   NoiseModel::resolve_noise_dof), falling back to the Rician default where none of those permits
//   a value to be established.
//   Where -noise_dof_scope slice is in effect and the estimation produced per-slice values, the
//   result holds one model per slice; otherwise it holds one.
//   "H" is the header of the input image as opened by the calling command, whose key-values are
//   those of DICOM provenance; it is interrogated only on the code path where the degrees of
//   freedom have an effect, so no message is emitted about a quantity that the selected transform
//   does not use.
NoiseModel::Set make_noise_model(const NoiseModelOptions &options,
                                 const Header &H,
                                 const NoiseModel::BackgroundResult &background);

class Demodulation {
public:
  Demodulation(demodulation_t mode) : mode(mode) {}
  Demodulation() : mode(demodulation_t::NONE) {}
  explicit operator bool() const { return mode != demodulation_t::NONE; }
  bool operator!() const { return mode == demodulation_t::NONE; }
  demodulation_t mode;
  std::vector<size_t> axes;
};
Demodulation select_demodulation(const Header &);

// Select the demeaning mode from -demean and the data.
// "grad" is the DW gradient table already resolved by the calling command from the header it
//   opened (empty if the image has none and none was supplied). It must be passed in rather than
//   re-derived here: only the opened Header carries the transform-realignment metadata that
//   -fslgrad import needs, and a table supplied on the command line has already been validated by
//   the caller, so a failure to obtain one here would hide a user error.
demean_type select_demean(const Header &, const Eigen::MatrixXd &grad);

// Resolve which component is to subtract the group means for one iteration (see demean_site_t),
//   from the demeaning mode and that iteration's partition count. This is the sole place the rule is
//   applied: the resulting value is then handed to every component whose behaviour depends on it
//   (Preconditioner::set_demean_site(), Estimate / Recon, Iterative::estimate), so that all of them
//   act on one decision rather than each re-deriving it from a proxy such as the partition count or
//   the presence of the demeaning-group labels.
demean_site_t resolve_demean_site(const demean_type demean, const ssize_t num_partitions);

} // namespace MR::Denoise::Precondition
