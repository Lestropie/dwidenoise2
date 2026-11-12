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

#include "denoise/precondition/noise_model/background.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

#include "algo/loop.h"
#include "algo/threaded_loop.h"
#include "denoise/precondition/noise_model/gamma_stats.h"
#include "exception.h"
#include "image.h"
#include "metadata/bids.h"
#include "mrtrix.h"

namespace MR::Denoise::Precondition::NoiseModel {

using GammaStats::digamma;
using GammaStats::gamma_p_inv;
using GammaStats::inverse_digamma;
using GammaStats::trigamma;

Accumulation::Accumulation(const std::array<ssize_t, 3> &size, const size_t slice_axis)
    : voxels(size_t(size[0]) * size_t(size[1]) * size_t(size[2])), size(size), slice_axis(slice_axis) {
  assert(slice_axis < 3);
}

default_type Accumulation::zeroed_fraction() const {
  if (voxels.empty())
    return 0.0;
  size_t zeroed = 0;
  for (const VoxelStats &stats : voxels) {
    if (stats.count == 0)
      ++zeroed;
  }
  return default_type(zeroed) / default_type(voxels.size());
}

namespace {

// One spatial voxel per invocation: walk that voxel's full column of volumes and write its four
//   sufficient statistics. Every voxel is visited by exactly one thread and writes only its own
//   entry, so the shared array needs no locking.
template <class ImageType> class AccumulateFunctor {
public:
  AccumulateFunctor(Accumulation &accumulation) : accumulation(accumulation) {}
  void operator()(ImageType &input) {
    VoxelStats stats;
    for (auto l = Loop(input, 3)(input); l; ++l) {
      const default_type m = default_type(input.value());
      // Zero and non-finite samples carry no information for a likelihood written in log(m^2), and
      //   a zero is in any case not a draw from the distribution being fitted but the mark of a
      //   voxel the reconstruction has emptied. Both are excluded from the count rather than
      //   contributing to it, so that every downstream sample total is a count of usable samples.
      if (!std::isfinite(m) || !(m > 0.0))
        continue;
      const default_type m2 = m * m;
      ++stats.count;
      stats.sum_sq += m2;
      stats.sum_quad += m2 * m2;
      stats.sum_log_sq += std::log(m2);
    }
    accumulation.voxels[accumulation.index(input.index(0), input.index(1), input.index(2))] = stats;
  }

private:
  Accumulation &accumulation;
};

} // namespace

template <class ImageType> Accumulation accumulate_background(ImageType &input, const size_t slice_axis) {
  Accumulation result({input.size(0), input.size(1), input.size(2)}, slice_axis);
  ThreadedLoop("Accumulating magnitude statistics for noise distribution estimation", input, 0, 3)
      .run(AccumulateFunctor<ImageType>(result), input);
  return result;
}

template Accumulation accumulate_background<Image<float>>(Image<float> &, const size_t);
template Accumulation accumulate_background<Image<double>>(Image<double> &, const size_t);

Estimate estimate_from_sums(const SampleSums &sums, const background_method_t method) {
  Estimate result;
  if (method == background_method_t::NONE)
    return result;
  if (sums.count < 2) {
    result.failure = "fewer than two samples";
    return result;
  }
  const default_type num_samples = default_type(sums.count);

  switch (method) {

  // Method of moments (the paper's Eqs. 8 and 9). Writing N = dof/2, a background sample has
  //   E[m^2] = 2 sigma^2 N and E[m^4] = 4 sigma^4 N (N+1), so
  //     sum(m^4)/sum(m^2) - sum(m^2)/V  ->  2 sigma^2 (N+1) - 2 sigma^2 N  =  2 sigma^2,
  //   which isolates sigma with no iteration and no special functions, and the first moment then
  //   gives N. The bracket is a difference of two positive quantities and a small or contaminated
  //   sample can drive it non-positive, which is a refusal rather than a number.
  case background_method_t::MOMENTS: {
    if (!(sums.sum_sq > 0.0)) {
      result.failure = "no positive magnitudes among the selected samples";
      return result;
    }
    const default_type bracket = sums.sum_quad / sums.sum_sq - sums.sum_sq / num_samples;
    if (!(bracket > 0.0)) {
      result.failure = "the method-of-moments variance estimate is non-positive"
                       " (the selected sample is too small, or is not background)";
      return result;
    }
    const default_type variance = 0.5 * bracket;
    result.sigma = std::sqrt(variance);
    result.dof = sums.sum_sq / (num_samples * variance);
    break;
  }

  // Maximum likelihood (the paper's Eqs. 11 and 12, and its Appendix A).
  //
  // With A = sum(m^2) / (2V) and B = sum(log m^2) / V, the two likelihood equations reduce to the
  //   single scalar equation in sigma
  //     f(sigma)  = psi(A/sigma^2) - B + log(2 sigma^2)                                = 0,
  //     f'(sigma) = -(2A/sigma^3) psi'(A/sigma^2) + 2/sigma,
  //   and the shape is then recovered by inverting the digamma function,
  //     dof = 2 psi^{-1}(B - log(2 sigma^2)).
  //
  // f is strictly decreasing on sigma > 0: f'(sigma) = (2/sigma) (1 - x psi'(x)) with x = A/sigma^2,
  //   and x psi'(x) > 1 for every x > 0 since psi'(x) > 1/x. It tends to log(2A) - B as sigma -> 0
  //   and to -infinity as sigma grows, so a root exists exactly when B < log(2A) -- which Jensen's
  //   inequality guarantees for any sample that is not degenerate, B being the mean of log(m^2) and
  //   log(2A) the log of their mean. The root is therefore unique and bracketable, and the Newton
  //   iteration below is run inside a bracket obtained by doubling / halving from the starting
  //   value, so that it cannot be defeated by a poor start.
  case background_method_t::ML: {
    const default_type a_term = sums.sum_sq / (2.0 * num_samples);
    const default_type b_term = sums.sum_log_sq / num_samples;
    if (!(a_term > 0.0) || !std::isfinite(b_term)) {
      result.failure = "the selected samples do not admit a likelihood (non-positive or non-finite)";
      return result;
    }
    if (!(b_term < std::log(2.0 * a_term))) {
      result.failure = "the selected magnitudes are degenerate (all of one value), so the likelihood has no maximum";
      return result;
    }
    const auto f = [a_term, b_term](const default_type sigma) {
      return digamma(a_term / (sigma * sigma)) - b_term + std::log(2.0 * sigma * sigma);
    };
    // The correctly-scaled root mean square of the selected magnitudes. This is the value sigma
    //   would take were there exactly one effective channel, and is a better-conditioned start
    //   than the sample standard deviation the paper names (which would in any case require a
    //   fifth accumulator, the first moment, that nothing else needs).
    default_type sigma = std::sqrt(a_term);
    // Bracket the root by expansion in the direction the sign of f indicates.
    default_type lower = sigma;
    default_type upper = sigma;
    if (f(sigma) > 0.0) {
      for (int i = 0; i != 200 && f(upper) > 0.0; ++i) {
        lower = upper;
        upper *= 2.0;
      }
    } else {
      for (int i = 0; i != 200 && f(lower) < 0.0; ++i) {
        upper = lower;
        lower *= 0.5;
      }
    }
    if (!(f(lower) >= 0.0) || !(f(upper) <= 0.0)) {
      result.failure = "the likelihood equation for the noise level could not be bracketed";
      return result;
    }
    bool converged = false;
    for (int iteration = 0; iteration != 100; ++iteration) {
      const default_type x = a_term / (sigma * sigma);
      const default_type value = digamma(x) - b_term + std::log(2.0 * sigma * sigma);
      if (value > 0.0)
        lower = sigma;
      else
        upper = sigma;
      const default_type derivative = -(2.0 * a_term / (sigma * sigma * sigma)) * trigamma(x) + 2.0 / sigma;
      default_type next = derivative < 0.0 ? sigma - value / derivative
                                           : std::numeric_limits<default_type>::quiet_NaN();
      if (std::isfinite(next) && next == sigma) {
        converged = true;
        break;
      }
      if (!std::isfinite(next) || next <= lower || next >= upper)
        next = 0.5 * (lower + upper);
      const default_type change = std::abs(next - sigma);
      sigma = next;
      if (change <= 1.0e-14 * sigma) {
        converged = true;
        break;
      }
    }
    if (!converged || !std::isfinite(sigma) || !(sigma > 0.0)) {
      result.failure = "the maximum-likelihood iteration for the noise level did not converge";
      return result;
    }
    const default_type num_channels = inverse_digamma(b_term - std::log(2.0 * sigma * sigma));
    if (!std::isfinite(num_channels) || !(num_channels > 0.0)) {
      result.failure = "the maximum-likelihood iteration for the degrees of freedom did not converge";
      return result;
    }
    result.sigma = sigma;
    result.dof = 2.0 * num_channels;
    break;
  }

  case background_method_t::NONE:
    assert(false);
    return result;
  }

  if (!std::isfinite(result.sigma) || !(result.sigma > 0.0) || !std::isfinite(result.dof) || !(result.dof > 0.0)) {
    result.failure = "the fit produced a non-finite or non-positive value";
    result.valid = false;
    return result;
  }
  result.valid = true;
  return result;
}

namespace {

// The voxels of one slice sharing a common number of usable volumes, ordered by their summed
//   squared magnitude, with prefix sums taken along that order.
//
// THIS IS WHERE ALGORITHM 1'S COST GOES. As written in the manuscript the algorithm is, per slice,
//   up to max_iter iterations x l candidate noise levels x n voxels of interval tests, followed by
//   a re-scan of the accepted voxels to form the moment sums -- of order 1e8 voxel visits per
//   slice, which would dominate the command's run time (and is presumably why the reference
//   implementation parallelises over slices). The work is entirely avoidable, EXACTLY, because the
//   acceptance test has been rearranged from "lambda_lo <= S2v / (2 sigma_c^2) <= lambda_hi" to
//   "2 sigma_c^2 lambda_lo <= S2v <= 2 sigma_c^2 lambda_hi": the candidate noise level then appears
//   only in the two bounds, so within a group of constant Kv the accepted set is a CONTIGUOUS
//   INTERVAL in S2v, located by two binary searches, and each of the four sums the estimators need
//   is one subtraction of prefix sums. The per-slice cost collapses from O(max_iter x l x n) to
//   O(n log n) for the sort plus O(max_iter x l x log n) for the searches.
//
// This is a reformulation and not an approximation: it computes the identical selected set and the
//   identical sums, which is what lets the implementation be faithful to the manuscript's iteration
//   counts rather than truncating them.
class Group {
public:
  Group(const uint32_t count) : count(count) {}
  void sort_and_accumulate(const std::vector<VoxelStats> &all) {
    std::sort(voxels.begin(), voxels.end(), [&all](const size_t a, const size_t b) {
      return all[a].sum_sq < all[b].sum_sq;
    });
    const size_t n = voxels.size();
    sum_sq.resize(n);
    prefix_sq.assign(n + 1, 0.0);
    prefix_quad.assign(n + 1, 0.0);
    prefix_log_sq.assign(n + 1, 0.0);
    for (size_t i = 0; i != n; ++i) {
      const VoxelStats &stats = all[voxels[i]];
      sum_sq[i] = stats.sum_sq;
      prefix_sq[i + 1] = prefix_sq[i] + stats.sum_sq;
      prefix_quad[i + 1] = prefix_quad[i] + stats.sum_quad;
      prefix_log_sq[i + 1] = prefix_log_sq[i] + stats.sum_log_sq;
    }
  }
  // Half-open index range [first, last) of the voxels whose summed squared magnitude lies within
  //   the acceptance interval implied by the candidate noise level and the current quantiles.
  std::pair<size_t, size_t> accepted(const default_type lower_bound, const default_type upper_bound) const {
    const auto first = std::lower_bound(sum_sq.begin(), sum_sq.end(), lower_bound);
    const auto last = std::upper_bound(first, sum_sq.end(), upper_bound);
    return {size_t(first - sum_sq.begin()), size_t(last - sum_sq.begin())};
  }
  uint32_t count;
  std::vector<size_t> voxels;
  std::vector<default_type> sum_sq;
  std::vector<default_type> prefix_sq;
  std::vector<default_type> prefix_quad;
  std::vector<default_type> prefix_log_sq;
};

// The state of one candidate selection: the accepted index range within each group, and the sums
//   over all of them.
class Selection {
public:
  size_t num_voxels{0};
  SampleSums sums;
  std::vector<std::pair<size_t, size_t>> ranges;
};

Selection select(const std::vector<Group> &groups,
                 const std::vector<default_type> &lambda_lo,
                 const std::vector<default_type> &lambda_hi,
                 const default_type sigma_candidate) {
  Selection result;
  result.ranges.resize(groups.size());
  const default_type scale = 2.0 * sigma_candidate * sigma_candidate;
  for (size_t g = 0; g != groups.size(); ++g) {
    const Group &group = groups[g];
    const auto range = group.accepted(scale * lambda_lo[g], scale * lambda_hi[g]);
    result.ranges[g] = range;
    const size_t n = range.second - range.first;
    result.num_voxels += n;
    result.sums.count += n * size_t(group.count);
    result.sums.sum_sq += group.prefix_sq[range.second] - group.prefix_sq[range.first];
    result.sums.sum_quad += group.prefix_quad[range.second] - group.prefix_quad[range.first];
    result.sums.sum_log_sq += group.prefix_log_sq[range.second] - group.prefix_log_sq[range.first];
  }
  return result;
}

// One iteration's state, retained so that a limit cycle can be recognised.
//
// WHY THIS IS NECESSARY. Algorithm 1's iterate is not really the pair (sigma, dof) but the
//   DISCRETE selected set that produces it, and a discrete map can enter a cycle that no tolerance
//   on the parameters will ever certify. It does so in practice: on ordinary synthetic background
//   the iteration is regularly observed to alternate indefinitely between two selections differing
//   by a handful of voxels out of a few thousand, whose degrees of freedom differ by 0.04% and
//   whose noise levels differ by 0.2% -- comfortably converged by any standard except the paper's
//   own, which compares consecutive iterates only and therefore reads the alternation as failure.
//   Left unhandled, such slices exhaust the iteration cap and decline, discarding a perfectly good
//   estimate. Recognising the recurrence of a selection is the exact statement that the iteration
//   has reached its terminal behaviour, whatever the period of that behaviour.
class IterationState {
public:
  std::vector<std::pair<size_t, size_t>> ranges;
  size_t num_voxels{0};
  default_type dof{0.0};
  default_type sigma{0.0};
};

// Whether two successive iterations agree to within the tolerance, on either an absolute or a
//   relative reading of it (the paper's stopping rule is the disjunction of the two).
bool has_converged(const default_type dof,
                   const default_type sigma,
                   const default_type previous_dof,
                   const default_type previous_sigma,
                   const default_type eps) {
  const default_type d_dof = std::abs(dof - previous_dof);
  const default_type d_sigma = std::abs(sigma - previous_sigma);
  if (d_dof < eps && d_sigma < eps)
    return true;
  return d_dof < eps * dof && d_sigma < eps * sigma;
}

// Algorithm 1 for one slice: alternate between selecting the voxels whose summed statistic falls
//   inside the central 1-p interval of the Gamma distribution implied by the current parameters,
//   and re-estimating those parameters from the voxels selected.
SliceEstimate estimate_slice(std::vector<Group> &groups,
                             const std::vector<default_type> &initial_candidates,
                             const background_method_t method,
                             const BackgroundParameters &parameters) {
  SliceEstimate result;
  if (groups.empty()) {
    result.decline = "no voxel of this slice provided a usable magnitude in any volume";
    return result;
  }

  // On the first outer iteration the acceptance interval is deliberately broad, spanning the whole
  //   admissible range: its lower end is evaluated at dof_min and its upper end at dof_max (the
  //   paper's Fig. 1C). Thereafter both ends are evaluated at the current estimate.
  default_type dof_lo = parameters.dof_min;
  default_type dof_hi = parameters.dof_max;
  std::vector<default_type> candidates(initial_candidates);
  std::vector<default_type> lambda_lo(groups.size());
  std::vector<default_type> lambda_hi(groups.size());
  default_type previous_dof = std::numeric_limits<default_type>::quiet_NaN();
  default_type previous_sigma = std::numeric_limits<default_type>::quiet_NaN();
  std::vector<IterationState> history;

  for (ssize_t iteration = 0; iteration != parameters.max_iter; ++iteration) {
    result.num_iterations = iteration + 1;
    // The quantiles depend on the voxel only through its number of usable volumes, so they are
    //   evaluated once per distinct value of that count rather than once per voxel.
    for (size_t g = 0; g != groups.size(); ++g) {
      const default_type volumes = default_type(groups[g].count);
      lambda_lo[g] = gamma_p_inv(volumes * dof_lo / 2.0, 0.5 * parameters.p);
      lambda_hi[g] = gamma_p_inv(volumes * dof_hi / 2.0, 1.0 - 0.5 * parameters.p);
    }
    // The candidate noise level is the one that accepts the most voxels: a level too small or too
    //   large places the interval away from the bulk of the background and admits few voxels,
    //   whereas the true level centres it on them.
    Selection best;
    for (const default_type candidate : candidates) {
      const Selection trial = select(groups, lambda_lo, lambda_hi, candidate);
      if (trial.num_voxels > best.num_voxels)
        best = trial;
    }
    result.num_selected = best.num_voxels;
    if (best.num_voxels < parameters.min_voxels_slice) {
      result.decline = "only " + str(best.num_voxels) + " voxels were identified as containing noise alone" +
                       " (at least " + str(parameters.min_voxels_slice) + " are required)";
      return result;
    }
    const Estimate estimate = estimate_from_sums(best.sums, method);
    if (!estimate) {
      result.decline = estimate.failure;
      return result;
    }
    const default_type ceiling = parameters.dof_ceiling_factor * parameters.dof_max;
    if (estimate.dof > ceiling) {
      result.decline = "the estimate of " + str(estimate.dof, 4) + " degrees of freedom exceeds the plausibility" +
                       " ceiling of " + str(ceiling, 4) + ", indicating that what was selected is not background";
      return result;
    }
    result.dof = estimate.dof;
    result.sigma = estimate.sigma;
    // Both ends of the interval now come from the current estimate, and the search for the noise
    //   level is narrowed to its neighbourhood.
    dof_lo = dof_hi = estimate.dof;
    candidates.resize(11);
    for (size_t i = 0; i != candidates.size(); ++i)
      candidates[i] = (0.95 + 0.01 * default_type(i)) * estimate.sigma;
    if (iteration > 0 && has_converged(estimate.dof, estimate.sigma, previous_dof, previous_sigma, parameters.eps)) {
      result.estimated = true;
      return result;
    }
    previous_dof = estimate.dof;
    previous_sigma = estimate.sigma;

    // Has this exact selection been visited before? If so the iteration has closed a cycle and
    //   will now repeat forever. The states of that cycle are all equally the algorithm's answer;
    //   the one built from the most voxels is taken, that being the selection the maximisation
    //   step would itself prefer, and it is a deterministic choice rather than an arbitrary one.
    const auto recurrence = std::find_if(history.begin(), history.end(), [&best](const IterationState &state) {
      return state.ranges == best.ranges;
    });
    if (recurrence != history.end()) {
      const auto preferred = std::max_element(
          recurrence, history.end(), [](const IterationState &a, const IterationState &b) {
            return a.num_voxels < b.num_voxels;
          });
      if (preferred->num_voxels > best.num_voxels) {
        result.dof = preferred->dof;
        result.sigma = preferred->sigma;
        result.num_selected = preferred->num_voxels;
      }
      result.estimated = true;
      return result;
    }
    IterationState state;
    state.ranges = best.ranges;
    state.num_voxels = best.num_voxels;
    state.dof = estimate.dof;
    state.sigma = estimate.sigma;
    history.push_back(std::move(state));
  }

  result.decline = "the estimate did not converge within " + str(parameters.max_iter) + " iterations";
  return result;
}

// Recover the voxels the converged parameters accept, for the diagnostic mask. Re-derived after
//   the fact rather than carried through the iteration so that what is exported is the selection
//   implied by the values actually reported.
void mark_selected(const std::vector<Group> &groups,
                   const SliceEstimate &slice,
                   const BackgroundParameters &parameters,
                   std::vector<uint8_t> &selected) {
  if (!slice.estimated)
    return;
  for (const Group &group : groups) {
    const default_type alpha = default_type(group.count) * slice.dof / 2.0;
    const default_type scale = 2.0 * slice.sigma * slice.sigma;
    const auto range = group.accepted(scale * gamma_p_inv(alpha, 0.5 * parameters.p),
                                      scale * gamma_p_inv(alpha, 1.0 - 0.5 * parameters.p));
    for (size_t i = range.first; i != range.second; ++i)
      selected[group.voxels[i]] = 1;
  }
}

} // namespace

default_type BackgroundResult::dof_at_slice(const ssize_t slice) const {
  assert(estimated);
  if (slice < 0 || size_t(slice) >= slices.size())
    return dof;
  return slices[slice].estimated ? slices[slice].dof : dof;
}

BackgroundResult estimate_background(const Accumulation &accumulation,
                                     const background_method_t method,
                                     const BackgroundParameters &parameters) {
  BackgroundResult result;
  result.slice_axis = accumulation.slice_axis;
  result.size = accumulation.size;
  result.zeroed_fraction = accumulation.zeroed_fraction();
  if (method == background_method_t::NONE) {
    result.decline = "no estimation was requested";
    return result;
  }

  // A field of view whose background the reconstruction has already zeroed carries no information
  //   for this estimator, and what remains adjacent to the mask boundary is not background either.
  //   This deserves its own message because it is a property of the data that the user can do
  //   nothing about and that no amount of parameter tuning will fix.
  if (result.zeroed_fraction > parameters.max_zeroed_fraction) {
    result.decline = str(std::round(100.0 * result.zeroed_fraction)) +
                     "% of the field of view is identically zero in every volume,"
                     " which is the signature of a reconstruction that has already masked the background;"
                     " there are no background voxels from which to estimate the noise distribution";
    return result;
  }

  // The top of the initial noise-level search grid. A background voxel at the largest admissible
  //   degrees of freedom has median magnitude sigma sqrt(2 P^{-1}(dof_max/2, 1/2)); evaluating that
  //   relation at a typical magnitude of the whole image, which is no smaller than a typical
  //   background magnitude, therefore bounds sigma from above. It bounds a grid that the algorithm
  //   then refines, so a loose bound costs only resolution on the first iteration.
  std::vector<default_type> typical;
  typical.reserve(accumulation.voxels.size());
  for (const VoxelStats &stats : accumulation.voxels) {
    if (stats.count > 0)
      typical.push_back(std::sqrt(stats.sum_sq / default_type(stats.count)));
  }
  if (typical.empty()) {
    result.decline = "the image contains no positive finite magnitudes";
    return result;
  }
  std::nth_element(typical.begin(), typical.begin() + typical.size() / 2, typical.end());
  const default_type median_magnitude = typical[typical.size() / 2];
  const default_type sigma_hi = median_magnitude / std::sqrt(2.0 * gamma_p_inv(0.5 * parameters.dof_max, 0.5));
  if (!std::isfinite(sigma_hi) || !(sigma_hi > 0.0)) {
    result.decline = "a noise level search range could not be established from the image intensities";
    return result;
  }
  std::vector<default_type> initial_candidates(parameters.num_candidates);
  for (ssize_t i = 0; i != parameters.num_candidates; ++i)
    initial_candidates[i] = default_type(i + 1) * sigma_hi / default_type(parameters.num_candidates);

  const size_t slice_axis = accumulation.slice_axis;
  const std::array<size_t, 2> plane_axes = slice_axis == 0   ? std::array<size_t, 2>{{1, 2}}
                                           : slice_axis == 1 ? std::array<size_t, 2>{{0, 2}}
                                                             : std::array<size_t, 2>{{0, 1}};
  const ssize_t num_slices = accumulation.size[slice_axis];
  result.slices.resize(size_t(num_slices));
  result.selected.assign(accumulation.voxels.size(), 0);

  for (ssize_t slice = 0; slice != num_slices; ++slice) {
    // Voxels are grouped by their number of usable volumes, because the Gamma distribution the
    //   acceptance test is applied to has shape K dof/2 and so differs between them. Ordinarily
    //   every voxel is in one group.
    std::map<uint32_t, Group> by_count;
    std::array<ssize_t, 3> position{{0, 0, 0}};
    position[slice_axis] = slice;
    for (ssize_t a = 0; a != accumulation.size[plane_axes[0]]; ++a) {
      position[plane_axes[0]] = a;
      for (ssize_t b = 0; b != accumulation.size[plane_axes[1]]; ++b) {
        position[plane_axes[1]] = b;
        const size_t index = accumulation.index(position[0], position[1], position[2]);
        const uint32_t count = accumulation.voxels[index].count;
        if (count == 0)
          continue;
        auto it = by_count.find(count);
        if (it == by_count.end())
          it = by_count.emplace(count, Group(count)).first;
        it->second.voxels.push_back(index);
      }
    }
    std::vector<Group> groups;
    groups.reserve(by_count.size());
    for (auto &entry : by_count)
      groups.push_back(std::move(entry.second));
    for (Group &group : groups)
      group.sort_and_accumulate(accumulation.voxels);

    SliceEstimate &estimate = result.slices[size_t(slice)];
    estimate = estimate_slice(groups, initial_candidates, method, parameters);
    estimate.index = slice;
    if (estimate.estimated) {
      result.num_selected_total += estimate.num_selected;
      mark_selected(groups, estimate, parameters, result.selected);
    }
  }

  std::vector<default_type> dofs;
  std::vector<default_type> sigmas;
  for (const SliceEstimate &slice : result.slices) {
    if (!slice.estimated)
      continue;
    dofs.push_back(slice.dof);
    sigmas.push_back(slice.sigma);
  }
  if (dofs.size() < parameters.min_slices) {
    result.decline = "only " + str(dofs.size()) + " of " + str(num_slices) +
                     " slices yielded an estimate of the noise distribution (at least " +
                     str(parameters.min_slices) + " are required)";
    return result;
  }
  if (result.num_selected_total < parameters.min_voxels_total) {
    result.decline = "only " + str(result.num_selected_total) +
                     " voxels in total were identified as containing noise alone (at least " +
                     str(parameters.min_voxels_total) + " are required)";
    return result;
  }

  std::sort(dofs.begin(), dofs.end());
  std::sort(sigmas.begin(), sigmas.end());
  const auto median_of = [](const std::vector<default_type> &sorted) {
    const size_t n = sorted.size();
    return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  };
  const auto quantile_of = [](const std::vector<default_type> &sorted, const default_type q) {
    const default_type position = q * default_type(sorted.size() - 1);
    const size_t below = size_t(std::floor(position));
    const size_t above = std::min(below + 1, sorted.size() - 1);
    const default_type weight = position - default_type(below);
    return (1.0 - weight) * sorted[below] + weight * sorted[above];
  };
  result.dof = median_of(dofs);
  result.sigma = median_of(sigmas);
  result.dof_iqr = quantile_of(dofs, 0.75) - quantile_of(dofs, 0.25);
  result.estimated = true;
  return result;
}

size_t select_slice_axis(const Header &H) {
  auto slice_encoding_it = H.keyval().find("SliceEncodingDirection");
  if (slice_encoding_it == H.keyval().end()) {
    // TODO Ideally this would be resolved against the axes *on disk*,
    //   not following transform realignment
    INFO("No header information on slice encoding; "
         "assuming the third spatial axis separates slices for noise distribution estimation");
    return 2;
  }
  const Metadata::BIDS::axis_vector_type dir = Metadata::BIDS::axisid2vector(slice_encoding_it->second);
  for (size_t axis = 0; axis != 3; ++axis) {
    if (dir[axis]) {
      INFO("For header SliceEncodingDirection=\"" + slice_encoding_it->second + "\", " +
           "chose axis " + str(axis) + " to separate slices for noise distribution estimation");
      return axis;
    }
  }
  return 2;
}

} // namespace MR::Denoise::Precondition::NoiseModel
