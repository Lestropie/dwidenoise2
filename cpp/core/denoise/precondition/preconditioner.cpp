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

#include "denoise/precondition/preconditioner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "algo/copy.h"
#include "algo/threaded_loop.h"
#include "app.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "interp/cubic.h"
#include "transform.h"

#include "denoise/precondition/vst.h"

using namespace MR::App;

namespace MR::Denoise::Precondition {

namespace {
// Private functions to prevent compiler attempting to create complex functions for real types
template <typename T>
typename std::enable_if<std::is_same<T, cfloat>::value, T>::type demodulate(const cfloat in, const cfloat phase) {
  return in * std::conj(phase);
}
template <typename T>
typename std::enable_if<std::is_same<T, cdouble>::value, T>::type demodulate(const cdouble in, const cfloat phase) {
  return in * std::conj(cdouble(phase));
}
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type demodulate(const T in, const cfloat phase) {
  assert(false);
  return in;
}
template <typename T>
typename std::enable_if<std::is_same<T, cfloat>::value, T>::type modulate(const cfloat in, const cfloat phase) {
  return in * phase;
}
template <typename T>
typename std::enable_if<std::is_same<T, cdouble>::value, T>::type modulate(const cdouble in, const cfloat phase) {
  return in * cdouble(phase);
}
template <typename T> typename std::enable_if<!is_complex<T>::value, T>::type modulate(const T in, const cfloat phase) {
  assert(false);
  return in;
}

// Fill every voxel of the adaptive-phase-correction phase map with a unit real phase (a no-op
//   demodulator) before the first APC estimation pass overwrites it. Voxel-wise; ThreadedLoop
//   copies the functor per thread.
class FillUnitPhaseFunctor {
public:
  void operator()(Image<cfloat> &phase) { phase.value() = cfloat(1.0f, 0.0f); }
};

} // namespace

template <typename T>
Preconditioner<T>::Preconditioner(Image<T> &image,
                              const Demodulation &demodulation,
                              const demean_type demean,
                              Image<float> &vst_noise,
                              std::shared_ptr<NoiseModel::Base> noise_model)
    : H_in(image),                   //
      H_out(image),                  //
      num_volume_groups(1),          //
      noise_model(noise_model),      //
      vst_noise_image(vst_noise) {   //

  for (ssize_t axis = 4; axis != H_in.ndim(); ++axis) {
    num_volume_groups *= H_in.size(axis);
    H_out.size(3) *= H_in.size(axis);
  }
  H_out.ndim() = 4;
  Stride::set(H_out, Stride::contiguous_along_axis(3));
  H_out.datatype() = DataType::from<T>();
  H_out.datatype().set_byte_order_native();

  if (H_in.ndim() > 4) {
    Header H_serialise(H_in);
    for (ssize_t axis = 3; axis != H_in.ndim(); ++axis) {
      H_serialise.size(axis - 3) = H_in.size(axis);
      H_serialise.stride(axis - 3) = axis - 2;
      H_serialise.spacing(axis - 3) = 1.0;
    }
    for (ssize_t axis = H_in.ndim() - 3; axis != 3; ++axis) {
      H_serialise.size(axis) = 1;
      H_serialise.stride(axis) = axis + 1;
      H_serialise.spacing(axis) = 1.0;
    }
    H_serialise.ndim() = std::max(size_t(3), H_in.ndim() - 3);
    H_serialise.datatype() = DataType::from<uint32_t>();
    H_serialise.datatype().set_byte_order_native();
    H_serialise.transform().setIdentity();
    serialise_image = Image<uint32_t>::scratch(                                     //
        H_serialise,                                                               //
        "Scratch image for serialising non-spatial indices into Casorati matrix"); //

    // Intentionally single-threaded: this assigns each voxel of the (small, non-spatial)
    //   serialisation image a monotonically increasing index in traversal order, defining the
    //   multi-index -> Casorati-column map. The value is a running counter, so it is inherently
    //   order-dependent (not a per-voxel-independent computation) and carries no parallelisation
    //   benefit; it is therefore left as a serial Loop rather than a ThreadedLoop.
    uint32_t output_index = 0;
    for (auto l = Loop(serialise_image)(serialise_image); l; ++l)
      serialise_image.value() = output_index++;
    serialise_image.reset();
    assert(output_index == H_out.size(3));
  }

  // Step 1: Phase demodulation
  // Only the smooth phase needs to be retained here; the actual demodulation of the data is
  //   performed on-the-fly, both for the Casorati fill and for the stabilised-domain mean
  //   computation.
  // - LINEAR / HANN: a fixed phase (Cordero-Grande linear ramp, or full-extent Hann-window
  //     non-linear phase) estimated once here and held unchanged for every iteration.
  // - APC: no fixed bootstrap. The phase is (re-)estimated from the empirical complex data by
  //     update_parameters() on every iteration, including the first (which self-calibrates a
  //     global noise level; see AdaptivePhaseEstimator). A unit-magnitude phase is allocated now so
  //     the map is valid before the first pass (demodulation is a no-op until that pass
  //     overwrites it) and so it serves as the per-slice fallback for any slice APC cannot
  //     calibrate.
  switch (demodulation.mode) {
  case demodulation_t::NONE:
    break;
  case demodulation_t::LINEAR:
  case demodulation_t::HANN: {
    typename DemodulatorSelector<T>::type demodulator(image,                                        //
                                                      demodulation.axes,                            //
                                                      demodulation.mode == demodulation_t::LINEAR); //
    phase_image = demodulator();
  } break;
  case demodulation_t::APC:
    phase_image =
        Image<cfloat>::scratch(image, "Scratch image storing adaptive-phase-correction background phase");
    ThreadedLoop(phase_image).run(FillUnitPhaseFunctor(), phase_image);
    break;
  }
  apc = AdaptivePhaseEstimator(demodulation.axes);
  apc_enabled = (demodulation.mode == demodulation_t::APC);

  // Step 2: Demeaning
  // Here only the structure is established (group indexing and storage allocation);
  //   the stabilised-domain mean values are (re)computed by compute_means(),
  //   which is sensitive to the current noise level map.
  Header H_mean(H_out);
  switch (demean) {
  case demean_type::NONE:
    break;
  case demean_type::VOLUME_GROUPS: {
    assert(serialise_image.valid());
    if (H_in.ndim() < 5)
      throw Exception("Cannot demean by volume groups if input image is <= 4D");
    index2group.resize(H_out.size(3));
    // Intentionally single-threaded: builds the (small) index-to-group lookup table with a running
    //   group counter over the non-spatial serialisation image. Like the serialisation fill above,
    //   the group index is order-dependent and the table is tiny, so parallelisation is neither
    //   well-defined nor beneficial.
    ssize_t group_index = 0;
    for (auto l_group = Loop(serialise_image, 1)(serialise_image); l_group; ++l_group, ++group_index) {
      for (auto l_volumes = Loop(serialise_image, 0, 1)(serialise_image); l_volumes; ++l_volumes)
        index2group[serialise_image.value()] = group_index;
    }
    serialise_image.reset();
    assert(group_index == num_volume_groups);
    H_mean.size(3) = num_volume_groups;
    vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for per-volume-group stabilised-domain mean");
  } break;
  case demean_type::SHELLS: {
    Eigen::Matrix<default_type, Eigen::Dynamic, Eigen::Dynamic> grad;
    try {
      grad = DWI::get_DW_scheme(H_mean);
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to obtain valid gradient table");
    }
    try {
      DWI::Shells shells(grad);
      index2shell.resize(image.size(3), -1);
      for (ssize_t shell_idx = 0; shell_idx != shells.count(); ++shell_idx) {
        for (auto v : shells[shell_idx].get_volumes())
          index2shell[v] = shell_idx;
      }
      assert(*std::min_element(index2shell.begin(), index2shell.end()) == 0);
      H_mean.size(3) = shells.count();
      DWI::stash_DW_scheme(H_mean, grad);
      vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for per-shell stabilised-domain mean");
    } catch (Exception &e) {
      throw Exception(e, "Cannot demean by shells as unable to establish b-value shell structure");
    }
  } break;
  case demean_type::ALL: {
    H_mean.ndim() = 3;
    DWI::clear_DW_scheme(H_mean);
    vst_mean_image = Image<T>::scratch(H_mean, "Scratch image for stabilised-domain mean across all volumes");
  } break;
  }

  // The stabilised-domain mean *values* are intentionally not computed here, only their
  //   storage allocated above. Every iteration selects its temporal subset and (re)computes
  //   the means against the current noise level map via set_temporal_subsample() +
  //   update_parameters() before the preconditioner is applied (see the run() loops in
  //   dwidenoise2 / dwi2noise). Computing them at construction would therefore always be
  //   immediately overwritten before use; doing so previously manifested as two back-to-back
  //   "Computing stabilised-domain mean intensities" passes. Callers must invoke
  //   update_parameters() to populate the means before applying the forward/inverse transform.
}

// Adaptive phase correction: re-estimate phase_image in place from the empirical complex
//   input and the current noise level map. Complex T only; a compiled no-op for real T so the
//   explicit template instantiations at the foot of this file remain well-formed.
template <typename T> void Preconditioner<T>::update_phase(Image<T> &input) {
  if constexpr (is_complex<T>::value) {
    // First pass: cold solve; every pass thereafter: warm-started native refinement. Downsample
    //   the first pass only when a later pass will refine it (multi-iteration schedule); a
    //   single-iteration schedule solves its sole pass natively (apc_coarse_first == false).
    const bool warm_start = !apc_first_pass;
    const bool downsample = apc_first_pass && apc_coarse_first;
    apc(input, vst_noise_image, phase_image, warm_start, downsample);
    apc_first_pass = false;
  }
}

template <typename T>
void Preconditioner<T>::set_temporal_subsample(default_type fraction, Math::RNG &rng, ssize_t min_per_group) {
  temporal_subset.clear();
  if (fraction >= 1.0)
    return;
  assert(fraction > 0.0);
  assert(min_per_group >= 1);

  const ssize_t num_vols = H_out.size(3);

  // Partition volume indices into strata according to the active demeaning grouping,
  //   so the random subset preserves the relative proportions of each group.
  //   (-demean none / all: a single stratum spanning all volumes.)
  std::vector<std::vector<ssize_t>> strata;
  if (!index2shell.empty()) {
    strata.resize(*std::max_element(index2shell.begin(), index2shell.end()) + 1);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[index2shell[v]].push_back(v);
  } else if (!index2group.empty()) {
    strata.resize(num_volume_groups);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[index2group[v]].push_back(v);
  } else {
    strata.resize(1);
    strata[0].resize(num_vols);
    for (ssize_t v = 0; v != num_vols; ++v)
      strata[0][v] = v;
  }

  std::vector<ssize_t> per_stratum_counts(strata.size(), 0);
  bool floor_raised = false;
  for (size_t g = 0; g != strata.size(); ++g) {
    std::vector<ssize_t> &stratum = strata[g];
    const ssize_t s = stratum.size();
    if (s == 0)
      continue;
    const ssize_t k_target = ssize_t(std::lround(fraction * default_type(s)));
    ssize_t k = std::min(std::max(k_target, min_per_group), s);
    if (k > k_target)
      floor_raised = true;
    // Partial Fisher-Yates: draw k distinct indices from this stratum without replacement.
    for (ssize_t i = 0; i != k; ++i) {
      std::uniform_int_distribution<ssize_t> dist(i, s - 1);
      std::swap(stratum[i], stratum[dist(rng)]);
      temporal_subset.push_back(stratum[i]);
    }
    per_stratum_counts[g] = k;
  }

  std::sort(temporal_subset.begin(), temporal_subset.end());
  assert(ssize_t(temporal_subset.size()) <= num_vols);

  H_out_subset = H_out;
  H_out_subset.size(3) = ssize_t(temporal_subset.size());

  std::string counts_str;
  for (size_t g = 0; g != per_stratum_counts.size(); ++g)
    counts_str += (g ? "," : "") + str(per_stratum_counts[g]);
  INFO("Temporal sub-sampling: retaining " + str(temporal_subset.size()) + " of " + str(num_vols) +
       " volumes (fraction " + str(fraction) + ") across " + str(strata.size()) +
       " group(s); per-group counts: " + counts_str);
  if (floor_raised)
    WARN("Temporal sub-sampling floor of " + str(min_per_group) +
         " volume(s) per group raised the effective subset above the requested fraction "
         "for one or more groups");
}

template <typename T> std::vector<ssize_t> Preconditioner<T>::output_volume_to_group() const {
  const ssize_t mprime = header().size(3);
  std::vector<ssize_t> result;
  // Output row k corresponds to the original (serialised) volume orig(k); under a temporal
  //   subset only the selected volumes are emitted, in sorted order.
  const auto orig = [&](const ssize_t k) -> ssize_t { return temporal_subset.empty() ? k : temporal_subset[k]; };
  if (!index2shell.empty()) {
    result.resize(mprime);
    for (ssize_t k = 0; k != mprime; ++k)
      result[k] = index2shell[orig(k)];
  } else if (!index2group.empty()) {
    result.resize(mprime);
    for (ssize_t k = 0; k != mprime; ++k)
      result[k] = index2group[orig(k)];
  } else if (vst_mean_image.valid()) {
    // -demean all: a single demeaning group spanning every volume.
    result.assign(mprime, 0);
  }
  // else -demean none: leave empty (no grouped demeaning).
  return result;
}

// Serialise this voxel's volumes into "data", applying phase demodulation and,
//   where a noise level map is available, the forward variance-stabilising transform.
template <typename T>
void Preconditioner<T>::serialise_and_stabilise(Image<T> &input,
                                              Image<cfloat> &phase,
                                              Image<uint32_t> &serialise,
                                              const Transform &transform,
                                              Interp::Cubic<Image<float>> *vst,
                                              Eigen::Array<T, Eigen::Dynamic, 1> &data) const {
  // Load all volumes within this voxel into "data"
  if (H_in.ndim() == 4) {
    for (ssize_t v = 0; v != H_out.size(3); ++v) {
      input.index(3) = v;
      data[v] = input.value();
    }
  } else {
    for (auto l = Loop(H_in, 3)(input); l; ++l) {
      for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
        serialise.index(axis - 3) = input.index(axis);
      data[serialise.value()] = input.value();
    }
  }

  // Phase demodulation
  if (phase.valid()) {
    assign_pos_of(input, 0, 3).to(phase);
    if (H_in.ndim() == 4) {
      for (ssize_t v = 0; v != H_out.size(3); ++v) {
        phase.index(3) = v;
        data[v] = demodulate<T>(data[v], phase.value());
      }
    } else {
      for (auto l = Loop(H_in, 3)(phase); l; ++l) {
        for (ssize_t axis = 3; axis != H_in.ndim(); ++axis)
          serialise.index(axis - 3) = phase.index(axis);
        data[serialise.value()] = demodulate<T>(data[serialise.value()], phase.value());
      }
    }
  }

  // Forward variance-stabilising transform
  if (vst != nullptr) {
    vst->scanner(transform.voxel2scanner *                         //
                 Eigen::Vector3d({default_type(input.index(0)),    //
                                  default_type(input.index(1)),    //
                                  default_type(input.index(2))})); //
    const default_type sigma = vst->value();
    for (ssize_t v = 0; v != H_out.size(3); ++v)
      data[v] = vst_forward<T>(*noise_model, data[v], sigma);
  }
}

namespace detail {

// One spatial voxel per invocation: serialise + stabilise the voxel's volume column and write its
//   per-group stabilised-domain mean(s). Mirrors the former serial body of
//   Preconditioner::compute_means (see there and the passages it references for the subset /
//   grouping / divisor rationale). The interpolator, the serialised "data" column and the group
//   accumulators are per-thread scratch held by the functor, which ThreadedLoop copies per thread.
template <typename T> class ComputeMeansFunctor {
public:
  ComputeMeansFunctor(const Preconditioner<T> &pre, Image<T> &input)
      : pre(pre),
        phase(pre.phase_image),
        serialise(pre.serialise_image),
        mean(pre.vst_mean_image),
        transform(input),
        subsampled(!pre.temporal_subset.empty()),
        neff(subsampled ? ssize_t(pre.temporal_subset.size()) : pre.H_out.size(3)),
        data(pre.H_out.size(3)),
        sums(mean.ndim() > 3 ? mean.size(3) : 0) {
    if (pre.vst_noise_image.valid())
      vst.emplace(pre.vst_noise_image);
    if (mean.ndim() > 3) {
      group_counts.assign(mean.size(3), 0);
      for (ssize_t idx = 0; idx != neff; ++idx) {
        const ssize_t v = eff_index(idx);
        ++group_counts[!pre.index2shell.empty() ? pre.index2shell[v] : pre.index2group[v]];
      }
    }
  }

  void operator()(Image<T> &input) {
    pre.serialise_and_stabilise(input, phase, serialise, transform, vst ? &vst.value() : nullptr, data);
    assign_pos_of(input, 0, 3).to(mean);
    if (mean.ndim() == 3) {
      T sum(T(0));
      for (ssize_t idx = 0; idx != neff; ++idx)
        sum += data[eff_index(idx)];
      mean.value() = sum / T(neff);
    } else {
      std::fill(sums.begin(), sums.end(), T(0));
      for (ssize_t idx = 0; idx != neff; ++idx) {
        const ssize_t v = eff_index(idx);
        sums[!pre.index2shell.empty() ? pre.index2shell[v] : pre.index2group[v]] += data[v];
      }
      for (ssize_t group = 0; group != mean.size(3); ++group) {
        mean.index(3) = group;
        mean.value() = group_counts[group] > 0 ? sums[group] / T(group_counts[group]) : T(0);
      }
    }
  }

private:
  ssize_t eff_index(const ssize_t idx) const { return subsampled ? pre.temporal_subset[idx] : idx; }

  const Preconditioner<T> &pre;
  Image<cfloat> phase;
  Image<uint32_t> serialise;
  Image<T> mean;
  Transform transform;
  std::optional<Interp::Cubic<Image<float>>> vst;
  bool subsampled;
  ssize_t neff;
  Eigen::Array<T, Eigen::Dynamic, 1> data;
  std::vector<T> sums;
  std::vector<ssize_t> group_counts;
};

// One spatial voxel per invocation: forward preconditioning (phase demodulation -> variance-
//   stabilising transform -> stabilised-domain demeaning) of the voxel's volume column, written to
//   the output. Mirrors the former serial forward body of Preconditioner::operator().
template <typename T> class ForwardApplyFunctor {
public:
  ForwardApplyFunctor(const Preconditioner<T> &pre, Image<T> &input)
      : pre(pre),
        phase(pre.phase_image),
        serialise(pre.serialise_image),
        mean(pre.vst_mean_image),
        transform(input),
        data(pre.H_out.size(3)) {
    if (pre.vst_noise_image.valid())
      vst.emplace(pre.vst_noise_image);
  }

  void operator()(Image<T> &input, Image<T> &output) {
    // Steps 1 & 2: serialise, phase demodulation, variance-stabilising transform
    pre.serialise_and_stabilise(input, phase, serialise, transform, vst ? &vst.value() : nullptr, data);

    // Step 3: demeaning (in the stabilised domain). Skipped when partitioning is active (the
    //   per-partition per-group demeaning is then performed within Estimate/Recon).
    if (mean.valid() && !pre.partitioning_active) {
      assign_pos_of(input, 0, 3).to(mean);
      if (mean.ndim() == 3) {
        const T mean_value = mean.value();
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v)
          data[v] -= mean_value;
      } else if (!pre.index2shell.empty()) {
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          mean.index(3) = pre.index2shell[v];
          data[v] -= T(mean.value());
        }
      } else if (!pre.index2group.empty()) {
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          mean.index(3) = pre.index2group[v];
          data[v] -= T(mean.value());
        }
      } else {
        assert(false);
        data.fill(std::numeric_limits<T>::signaling_NaN());
      }
    }

    // Write to output. Without temporal sub-sampling, emit every volume; with sub-sampling, emit
    //   only the m' selected volumes (data was demeaned by the subset means).
    if (pre.temporal_subset.empty()) {
      for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
        output.index(3) = v;
        output.value() = data[v];
      }
    } else {
      for (ssize_t k = 0; k != ssize_t(pre.temporal_subset.size()); ++k) {
        output.index(3) = k;
        output.value() = data[pre.temporal_subset[k]];
      }
    }
  }

private:
  const Preconditioner<T> &pre;
  Image<cfloat> phase;
  Image<uint32_t> serialise;
  Image<T> mean;
  Transform transform;
  std::optional<Interp::Cubic<Image<float>>> vst;
  Eigen::Array<T, Eigen::Dynamic, 1> data;
};

// One spatial voxel per invocation: inverse preconditioning of the voxel's volume column (undo
//   demeaning / variance-stabilising transform, then re-modulate the background phase), written to
//   the output. Mirrors the former serial inverse body of Preconditioner::operator(); the one-off
//   setup (use_mean, num_groups, messages) is performed by the caller and passed in. Per-thread
//   scratch (interpolator, "data" column, per-group accumulators) is held by the functor.
template <typename T> class InverseApplyFunctor {
public:
  InverseApplyFunctor(const Preconditioner<T> &pre,
                      Image<T> &input,
                      const bool use_mean,
                      const ssize_t num_groups,
                      const bias_handling_t bias_handling,
                      const debias_anchor_t debias_anchor)
      : pre(pre),
        phase(pre.phase_image),
        serialise(pre.serialise_image),
        mean(pre.vst_mean_image),
        transform(input),
        use_mean(use_mean),
        num_groups(num_groups),
        bias_handling(bias_handling),
        debias_anchor(debias_anchor),
        data(pre.H_out.size(3)),
        group_offset(std::max<ssize_t>(num_groups, ssize_t(1)), T(0)),
        group_dc(num_groups),
        group_gain(num_groups) {
    if (pre.vst_noise_image.valid())
      vst.emplace(pre.vst_noise_image);
  }

  void operator()(Image<T> &input, Image<T> &output) {

    for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
      input.index(3) = v;
      data[v] = input.value();
    }

    if (vst) {
      // Interpolate the noise level (variance-stabilising-transform scale) at this voxel.
      vst->scanner(transform.voxel2scanner *                         //
                   Eigen::Vector3d({default_type(input.index(0)),    //
                                    default_type(input.index(1)),    //
                                    default_type(input.index(2))})); //
      const default_type sigma = vst->value();

      if (bias_handling == bias_handling_t::DEBIAS && debias_anchor == debias_anchor_t::GROUP_MEAN &&
          use_mean) {
        // GROUP_MEAN (legacy) debiasing: linearise the unbiased inverse about the per-group
        //   stabilised-domain mean, mapping the denoised residual through the local Jacobian.
        assign_pos_of(input, 0, 3).to(mean);
        for (ssize_t group = 0; group != num_groups; ++group) {
          if (mean.ndim() > 3)
            mean.index(3) = group;
          const T op = mean.value();
          group_dc[group] = vst_inverse_unbiased<T>(*pre.noise_model, op, sigma);
          group_gain[group] = vst_jacobian<T>(*pre.noise_model, op, sigma);
        }
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          const ssize_t group = num_groups == 1                                    //
                                    ? 0                                            //
                                    : (!pre.index2shell.empty() ? pre.index2shell[v]   //
                                                                : pre.index2group[v]); //
          data[v] = group_dc[group] + T(group_gain[group]) * data[v];
        }
      } else {
        // SAMPLE (default) reversal, and all PRESERVE reversal: undo the demeaning offset, form
        //   each volume's own denoised operating point, then apply the chosen non-linear inverse
        //   pointwise at that point.
        if (use_mean) {
          assign_pos_of(input, 0, 3).to(mean);
          for (ssize_t group = 0; group != num_groups; ++group) {
            if (mean.ndim() > 3)
              mean.index(3) = group;
            group_offset[group] = mean.value();
          }
        } else {
          group_offset[0] = T(0);
        }
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          const ssize_t group = num_groups <= 1                                    //
                                    ? 0                                            //
                                    : (!pre.index2shell.empty() ? pre.index2shell[v]   //
                                                                : pre.index2group[v]); //
          const T u_recon = data[v] + group_offset[group];
          data[v] = (bias_handling == bias_handling_t::DEBIAS)                //
                        ? vst_inverse_unbiased<T>(*pre.noise_model, u_recon, sigma) //
                        : vst_inverse<T>(*pre.noise_model, u_recon, sigma);         //
        }
      }
    } else if (use_mean) {
      // No variance-stabilising transform (demean-only bootstrap): reversal is re-addition of mean.
      assign_pos_of(input, 0, 3).to(mean);
      if (mean.ndim() == 3) {
        const T mean_value = mean.value();
        data += mean_value;
      } else if (!pre.index2shell.empty()) {
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          mean.index(3) = pre.index2shell[v];
          data[v] += T(mean.value());
        }
      } else if (!pre.index2group.empty()) {
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          mean.index(3) = pre.index2group[v];
          data[v] += T(mean.value());
        }
      } else {
        assert(false);
        data.fill(std::numeric_limits<T>::signaling_NaN());
      }
    }

    // Step 1 reversal: re-modulate phase
    if (phase.valid()) {
      assign_pos_of(input, 0, 3).to(phase);
      if (serialise.valid()) {
        for (auto l = Loop(pre.H_in, 3)(phase); l; ++l) {
          for (ssize_t axis = 3; axis != pre.H_in.ndim(); ++axis)
            serialise.index(axis - 3) = phase.index(axis);
          data[serialise.value()] = modulate<T>(data[serialise.value()], phase.value());
        }
      } else {
        for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
          phase.index(3) = v;
          data[v] = modulate<T>(data[v], phase.value());
        }
      }
    }

    // Write to output
    if (serialise.valid()) {
      for (auto l = Loop(pre.H_in, 3)(output); l; ++l) {
        for (ssize_t axis = 3; axis != pre.H_in.ndim(); ++axis)
          serialise.index(axis - 3) = output.index(axis);
        output.value() = data[serialise.value()];
      }
    } else {
      for (ssize_t v = 0; v != pre.H_out.size(3); ++v) {
        output.index(3) = v;
        output.value() = data[v];
      }
    }
  }

private:
  const Preconditioner<T> &pre;
  Image<cfloat> phase;
  Image<uint32_t> serialise;
  Image<T> mean;
  Transform transform;
  std::optional<Interp::Cubic<Image<float>>> vst;
  bool use_mean;
  ssize_t num_groups;
  bias_handling_t bias_handling;
  debias_anchor_t debias_anchor;
  Eigen::Array<T, Eigen::Dynamic, 1> data;
  std::vector<T> group_offset;
  std::vector<T> group_dc;
  std::vector<default_type> group_gain;
};

} // namespace detail

template <typename T> void Preconditioner<T>::compute_means(Image<T> input_arg) {
  if (!vst_mean_image.valid())
    return;
  // When partitioning is active, demeaning is performed per partition inside Estimate/Recon, so
  //   no preconditioner-side means are needed (and the stored values are left unused).
  if (partitioning_active)
    return;

  // One independent problem per spatial voxel: serialise + stabilise the voxel's volume column and
  //   write its per-group stabilised-domain mean(s). The subset / grouping / divisor handling (and
  //   the reason the means are formed over only the temporal-subset volumes) lives in
  //   detail::ComputeMeansFunctor.
  ThreadedLoop("Computing stabilised-domain mean intensities", input_arg, 0, 3)
      .run(detail::ComputeMeansFunctor<T>(*this, input_arg), input_arg);
}

template <typename T>
void Preconditioner<T>::operator()(Image<T> input,
                                 Image<T> output,
                                 const bool inverse,
                                 const bias_handling_t bias_handling,
                                 const debias_anchor_t debias_anchor) const {

  // Only the stored layout is queried here, for the one-off setup below; the actual per-voxel
  //   transform -- each thread's own image accessors, interpolator and scratch -- is performed by
  //   the ThreadedLoop functors (detail::ForwardApplyFunctor / detail::InverseApplyFunctor).
  Image<T> mean(vst_mean_image);
  if (inverse) {

    assert(dimensions_match(H_out, input));
    assert(dimensions_match(H_in, output));

    // The forward order is: phase demodulation -> variance-stabilising transform -> demeaning.
    // Reversal therefore proceeds in the opposite order.
    // Where a noise level map is available, demeaning is treated purely as PCA conditioning:
    //   the SAMPLE anchor (default) undoes the stored demean offset first to form each
    //   volume's own denoised operating point (group mean + denoised residual), then applies
    //   the chosen non-linear inverse pointwise at that operating point. This makes debiasing
    //   independent of the demeaning grouping and restores the natural heteroscedasticity on
    //   the output scale. The GROUP_MEAN anchor instead reproduces the prior behaviour for the
    //   DEBIAS handling, linearising the unbiased inverse about the per-group mean.
    //   The PRESERVE handling is always inverted pointwise (a faithful algebraic reversal).
    // Where no noise level map is available (the demean-only bootstrap),
    //   reversal reduces to re-addition of the stored (empirical) mean.

    // When partitioning is active the demean offset was applied (and is reversed) per partition
    //   inside Estimate/Recon; the preconditioner therefore reverses only the VST (pointwise),
    //   exactly as for -demean none. Treat the stored mean as absent in that case.
    const bool use_mean = mean.valid() && !partitioning_active;

    // Per-group quantities (the stabilised-domain demean offset for SAMPLE, and the mapped DC
    //   value + local gain for GROUP_MEAN legacy debiasing) are sized per thread inside the functor.
    const ssize_t num_groups = use_mean ? (mean.ndim() == 3 ? 1 : mean.size(3)) : 0;

    if (vst_noise_image.valid() && !mean.valid()) {
      INFO("Reversing preconditioning without a demeaning reference: "
           "the non-linear inverse variance-stabilising transform is applied directly "
           "to each denoised sample");
    }

    // Describe which corrections are being reversed: re-addition of the demeaning offset,
    //   the noise model governing the inverse variance-stabilising transform, (only for a
    //   non-linear model, where the distinction has an effect) whether the noise-floor bias
    //   is being removed (DEBIAS) or preserved (PRESERVE), and re-modulation of the estimated
    //   background phase for complex data that were phase-demodulated during preconditioning.
    //   This detail is esoteric relative to the progress bar, so it is reported via INFO()
    //   rather than appended to the (per-voxel-updated) progress message itself.
    const std::string reversal_message = "Reversing data preconditioning";
    {
      std::string detail;
      const auto append = [&detail](const std::string &item) { detail += (detail.empty() ? "" : ", ") + item; };
      if (use_mean)
        append("reverting demeaning");
      if (vst_noise_image.valid()) {
        append(noise_model->description() + " noise model");
        if (!noise_model->is_linear())
          append(bias_handling == bias_handling_t::DEBIAS ? "removing noise-floor bias"
                                                          : "preserving noise-floor bias");
      }
      if (phase_image.valid())
        append("re-modulating estimated background phase");
      if (!detail.empty())
        INFO(reversal_message + ": " + detail);
    }

    ThreadedLoop(reversal_message, input, 0, 3)
        .run(detail::InverseApplyFunctor<T>(*this, input, use_mean, num_groups, bias_handling, debias_anchor),
             input,
             output);
    return;
  }

  assert(dimensions_match(H_in, input));
  // While a temporal subset is active the output holds only the m' selected volumes,
  //   so it conforms to header() (size(3) == m'), not to the full H_out.
  assert(dimensions_match(header(), output));

  // Applying forward preconditioning.
  // Order: phase demodulation -> variance-stabilising transform -> demeaning;
  //   demeaning is performed in the stabilised domain so that the Casorati entries
  //   are zero-mean per group (vst_plan.md section 3.1).
  ThreadedLoop("Applying data preconditioning", input, 0, 3)
      .run(detail::ForwardApplyFunctor<T>(*this, input), input, output);
}

template class Preconditioner<float>;
template class Preconditioner<double>;
template class Preconditioner<cfloat>;
template class Preconditioner<cdouble>;

} // namespace MR::Denoise::Precondition
