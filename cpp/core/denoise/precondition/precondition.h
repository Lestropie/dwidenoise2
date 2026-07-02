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

#include "app.h"
#include "denoise/precondition/noise_model/noise_model.h"
#include "header.h"

namespace MR::Denoise::Precondition {

extern const char *const demodulation_description;

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

App::OptionGroup precondition_options(const bool include_output);

// Construct the noise model governing the variance-stabilising transform
//   from the -noise_dof and -vst_method command-line options.
// For complex (or phase-demodulated) data, pass complex == true to obtain the
//   Gaussian model; -noise_dof is then ignored (with a warning if specified).
// For magnitude data, the receive-channel count N from -noise_dof selects a
//   Rician (N == 1) or non-central chi (N > 1) model, built with the requested
//   -vst_method strategy.
std::shared_ptr<NoiseModel::Base> make_noise_model(const bool complex);

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

demean_type select_demean(const Header &);

} // namespace MR::Denoise::Precondition
