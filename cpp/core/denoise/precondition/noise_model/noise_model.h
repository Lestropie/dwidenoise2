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

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

#include "denoise/precondition/noise_model/base.h"

namespace MR::Denoise::Precondition::NoiseModel {

// Relation between receive channels and degrees of freedom, which holds throughout this library
//   and is a definition rather than an approximation:
//
//     dof = 2 * channels
//
// Each complex receive channel contributes two real Gaussian components (real and imaginary), so
//   N channels combined by sum of squares give a non-central chi distribution with 2N degrees of
//   freedom. N = 1 (2 DOF) is the Rician case, which is also what any PHASE-SENSITIVE combination
//   (adaptive combine, SENSE, SENSE1, Roemer) yields no matter how many physical channels it
//   combines, since those collapse the channels to a single complex image before the magnitude is
//   taken. For complex (non-magnitude) input the data have already been combined, so a sample
//   carries 2 DOF irrespective of the channel count.
//
// N is the EFFECTIVE channel count, not the physical element count, and is real-valued: see
//   NonCentralChi in noncentralchi.h and the discussion in coils.h.

// Statistical distribution governing the raw image intensities prior to PCA:
// - GAUSSIAN: complex (or phase-demodulated) data; zero-mean Gaussian per channel.
// - RICIAN: magnitude data from a single receive channel (non-central chi, 2 DOF).
// - NONCENTRALCHI: magnitude data combined across N channels by sum-of-squares (2N DOF).
const std::vector<std::string> distributions = {"gaussian", "rician", "noncentralchi"};
enum class distribution_t { GAUSSIAN, RICIAN, NONCENTRALCHI };

// Variance-stabilising transform (VST) applied to the raw data prior to PCA.
// Two of the options select the transform directly, independent of the noise
//   distribution:
// - NONE:   identity; no transform is applied (the data reach PCA unmodified
//             save for any demeaning). As nothing is divided by the noise level,
//             refining that level across iterations has no effect on subsequent
//             processing, so the calling commands fall back to a single pass.
// - LINEAR: the simple linear transform u = m / sigma (the Gaussian model);
//             exact for additive Gaussian (complex / phase-demodulated) noise,
//             but only a scale normalisation for magnitude data (which remains
//             heteroscedastic near the floor) and with no noise-floor debiasing.
// The remaining options build the nonlinear transform for magnitude data and,
//   in particular, the exact-unbiased inverse that debiases the noise floor:
// - FOI:  Foi (2011)-style exact-unbiased inverse via numerical integration;
//           smooth and well-defined across the entire SNR range including nu=0.
// - KOAY: Koay-Basser (2006) first-moment inverse with a hard floor clamp
//           (no unique solution below SNR ~1.913).
// - MOM:  method-of-moments (closed-form) first-moment inverse.
// For these three the forward stabilising transform itself is shared;
//   they differ only in how the stabilised domain is mapped back to a
//   bias-free underlying level (the inverse / debias step).
enum class vst_method_t { NONE, LINEAR, FOI, KOAY, MOM };

// The normalised moments of the non-central chi distribution are free functions in
//   math/noncentral_chi.h (Math::NonCentralChi::normalised_mean() / normalised_variance()) rather
//   than being reachable only through a constructed model, because the degrees of freedom are
//   something to be *estimated*: evaluating a candidate value must not require building that
//   value's tabulations (a one-time cost per model, and the dominant cost by far relative to a
//   single evaluation of either function).

// Constrain a degrees-of-freedom value to the range within which the magnitude noise model is
//   well-posed (see NonCentralChi::min_dof), returning the value that will actually be used.
//   Separated from the model itself so that a caller can detect that a requested value has been
//   modified and say so, rather than silently substituting one.
default_type clamp_dof(const default_type dof);

// Construct a noise model for the requested distribution.
// - dof is the real-valued degrees of freedom, used only for NONCENTRALCHI (ignored for GAUSSIAN /
//     RICIAN, both of which are 2 by construction). It must already lie within the admissible
//     range; pass it through clamp_dof() first.
// - vst_method is used only for magnitude distributions (ignored for GAUSSIAN).
std::shared_ptr<Base> make(const distribution_t distribution, //
                           const default_type dof,            //
                           const vst_method_t vst_method);    //

// The noise model, or models, governing the variance-stabilising transform across one image.
//
// WHY A SET RATHER THAN A MODEL. Under parallel imaging the effective channel count is not a
//   property of the coil alone but of the reconstruction, and it varies over space (Aja-Fernandez
//   et al. 2011); estimating it from the data therefore yields one value per slice, not one value.
//   Whether those values are reduced to their median or retained is the user's choice
//   (-noise_dof_scope), and this class is what allows both to be expressed without the rest of the
//   software needing to know which was chosen: a scalar model is a set of one, and at() ignores its
//   argument.
//
// Kernels span slices, and that is fine: the point of the transform is that each sample is rendered
//   approximately unit-variance by ITS OWN model, and the inverse selects the same model at the
//   same voxel. What must not happen is a mismatch between the forward and the inverse, and holding
//   the set in the one object that applies both is what prevents it.
class Set {
public:
  Set() = default;
  // One model for the whole image.
  explicit Set(std::shared_ptr<Base> model);
  // One model per slice along "slice_axis"; entries may be, and ordinarily are, shared between
  //   slices whose degrees of freedom quantise to the same value.
  Set(std::vector<std::shared_ptr<Base>> models, const size_t slice_axis);

  bool valid() const { return !models.empty(); }
  bool slicewise() const { return models.size() > 1; }
  size_t slice_axis() const { return axis; }
  // Transforms actually constructed, which is what the quantisation of a slice-wise set bounds.
  size_t num_distinct_models() const;

  // The model governing a given slice. The argument is ignored where the set holds one model, so a
  //   caller that has the voxel position in hand can pass it unconditionally and need not know
  //   whether the set is slice-wise.
  const Base &at(const ssize_t slice) const {
    assert(!models.empty());
    return *models[models.size() == 1 ? 0 : size_t(slice)];
  }

  // Whether the transform is linear (see Base::is_linear). Every model of a set is built from the
  //   same distribution family and the same -vst_method, so the property belongs to the set.
  bool is_linear() const;
  std::string description() const;

private:
  std::vector<std::shared_ptr<Base>> models;
  size_t axis{2};
};

// Build a slice-wise set from one degrees-of-freedom value per slice.
//
// The values are QUANTISED to a grid before the transforms are built, because constructing a
//   NonCentralChi is a one-time Poisson-mixture quadrature whose cost is quadratic in the
//   tabulation extent, and one per slice for sixty to ninety slices is a real startup cost. The
//   default step of 0.25 degrees of freedom (0.125 effective channels) is far below the estimator's
//   own slice-to-slice scatter, and the transform's sensitivity to the degrees of freedom at that
//   resolution is negligible; slices whose values fall in the same cell therefore share one model.
//   Each value is clamped to the admissible range before quantisation.
Set make_slicewise(const std::vector<default_type> &dof_per_slice, //
                   const size_t slice_axis,                        //
                   const vst_method_t vst_method,                  //
                   const default_type quantum = 0.25);             //

} // namespace MR::Denoise::Precondition::NoiseModel
