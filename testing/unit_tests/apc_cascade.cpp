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

// Self-contained unit test for the multi-resolution cold solve of adaptive phase correction
//   (AdaptivePhaseEstimator::Params::cascade_depth).
//
// A cold solve descends a stack of 2x-decimated grids, solves the coarsest, and carries the phase
//   back up. How deep it may descend is bounded twice over: by the requested depth, and by the
//   plane itself, since a grid under 4 voxels along either axis is not decimated further and a
//   plane under 4 voxels to begin with is solved cold at its native size. Those two bounds are what
//   this test pins, because they are invisible in any per-plane accuracy metric: a plane solved at
//   the wrong depth still solves, still returns a unit-magnitude phase, and differs only in how much
//   of the long-wavelength content it recovered.
//
// A plain executable whose non-zero exit status indicates failure, following
//   testing/unit_tests/{noise_convention,noise_model,dof_estimation,apc_lambda_carry}.cpp. It drives
//   the real AdaptivePhaseEstimator through its public full-image entry point, so what is exercised
//   is the estimator's own level stack and not a re-implementation of it.
//
// What is asserted:
//
//   1. Depth 2 reproduces the default configuration bit for bit, on both cold regimes. Depth 2 is
//      the shipped arrangement -- one coarse grid and the native one -- so anything else would mean
//      the level machinery had altered the path it was generalising.
//   2. Depth 1 differs from depth 2, so the setting is live and the test above is not vacuous.
//   3. Descent stops at the plane's own floor: on a 12x12 plane, which admits 12 -> 6 -> 3 and no
//      further, depths 3, 4, 6 and 12 all give bit-identical results.
//   4. A plane too small to decimate at all is solved identically at every depth, including depths
//      that ask for levels it cannot supply.
//   5. Every plane is solved (the returned phase is unit-magnitude everywhere and differs from the
//      incoming one) at every depth, on both plane sizes.
//
// The iteration budgets are cut far below the shipped ones throughout: this test is about which
//   grids a solve visits, not about how well any plane is solved.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "algo/loop.h"
#include "app.h"
#include "denoise/precondition/apc.h"
#include "header.h"
#include "image.h"
#include "types.h"

using MR::Denoise::Precondition::AdaptivePhaseEstimator;
using regime_t = AdaptivePhaseEstimator::regime_t;
using VolumePlan = AdaptivePhaseEstimator::VolumePlan;

namespace {

int failures = 0;

void check(const bool condition, const std::string &what) {
  std::printf("   [%s] %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition)
    ++failures;
}

constexpr ssize_t Nz = 2;
constexpr ssize_t Nvol = 2;

// Budgets small enough that the whole test is a few hundred thousand voxel-sweeps even in a debug
//   build, but large enough that every level moves the phase away from the one it was handed.
AdaptivePhaseEstimator::Params test_params(const ssize_t depth) {
  AdaptivePhaseEstimator::Params params;
  params.cp_iter = 8;
  params.max_lambda_iter = 3;
  params.polish_iter = 4;
  params.cp_iter_warm = 6;
  params.max_lambda_iter_warm = 2;
  params.polish_iter_warm = 4;
  if (depth > 0)
    params.cascade_depth = depth;
  return params;
}

MR::Header make_header(const ssize_t N) {
  MR::Header H;
  H.ndim() = 4;
  H.size(0) = N;
  H.size(1) = N;
  H.size(2) = Nz;
  H.size(3) = Nvol;
  for (size_t axis = 0; axis != 4; ++axis) {
    H.spacing(axis) = 1.0;
    H.stride(axis) = ssize_t(axis) + 1;
  }
  H.transform().setIdentity();
  H.datatype() = MR::DataType::from<MR::cfloat>();
  return H;
}

// A smooth magnitude and a smooth background phase plus noise, varying from volume to volume and
//   from slice to slice so that no two planes are the same problem.
MR::Image<MR::cfloat> make_data(const ssize_t N) {
  MR::Image<MR::cfloat> image = MR::Image<MR::cfloat>::scratch(make_header(N), "test data");
  std::mt19937_64 rng(20260806);
  std::normal_distribution<double> noise(0.0, 4.0);
  for (ssize_t v = 0; v != Nvol; ++v) {
    image.index(3) = v;
    for (ssize_t z = 0; z != Nz; ++z) {
      image.index(2) = z;
      const double scale = 60.0 * (1.0 + 0.35 * double(v)) * (1.0 + 0.2 * double(z));
      const double twist = 0.5 + 0.3 * double(v) - 0.15 * double(z);
      for (ssize_t j = 0; j != N; ++j) {
        image.index(1) = j;
        for (ssize_t i = 0; i != N; ++i) {
          image.index(0) = i;
          const double x = (double(i) - 0.5 * double(N - 1)) / double(N);
          const double y = (double(j) - 0.5 * double(N - 1)) / double(N);
          const double magnitude = scale * std::exp(-3.0 * (x * x + y * y));
          const double phase = twist * (1.7 * x + 1.1 * y) + 0.6 * std::sin(4.0 * x) * std::cos(3.0 * y);
          image.value() = MR::cfloat(float(magnitude * std::cos(phase) + noise(rng)),
                                     float(magnitude * std::sin(phase) + noise(rng)));
        }
      }
    }
  }
  return image;
}

MR::Image<MR::cfloat> unit_phase(const ssize_t N) {
  MR::Image<MR::cfloat> phase = MR::Image<MR::cfloat>::scratch(make_header(N), "test phase");
  for (auto l = MR::Loop(phase)(phase); l; ++l)
    phase.value() = MR::cfloat(1.0f, 0.0f);
  return phase;
}

// One pass over every volume in the requested regime, at the requested depth (0 = leave the default
//   in place), returning the phase map it produced.
MR::Image<MR::cfloat> solve(MR::Image<MR::cfloat> &image,
                            const ssize_t N,
                            const ssize_t depth,
                            const regime_t regime) {
  MR::Image<MR::cfloat> phase = unit_phase(N);
  std::vector<VolumePlan> plan(Nvol, VolumePlan{true, regime});
  const AdaptivePhaseEstimator estimator(std::vector<size_t>{0, 1}, test_params(depth));
  MR::Image<float> no_sigma;
  MR::Image<uint32_t> no_serialise;
  estimator(image, no_sigma, phase, no_serialise, plan);
  return phase;
}

bool identical(MR::Image<MR::cfloat> &a, MR::Image<MR::cfloat> &b) {
  for (auto l = MR::Loop(a)(a, b); l; ++l)
    if (MR::cfloat(a.value()) != MR::cfloat(b.value()))
      return false;
  return true;
}

// Every voxel carries a unit-magnitude phase, and at least one differs from the unit real phase the
//   map was initialised with -- i.e. the plane was actually solved and written back.
bool solved_everywhere(MR::Image<MR::cfloat> &phase) {
  bool moved = false;
  for (auto l = MR::Loop(phase)(phase); l; ++l) {
    const MR::cfloat value = phase.value();
    if (std::abs(std::abs(value) - 1.0f) > 1e-5f)
      return false;
    if (value != MR::cfloat(1.0f, 0.0f))
      moved = true;
  }
  return moved;
}

} // namespace

int main() {
  // Deterministic and free of any dependence on the host's core count: the estimator's threaded
  //   loop distributes planes over threads, and while each plane is an independent problem writing
  //   to its own region of the phase map, a single thread makes the test's behaviour identical
  //   everywhere.
  setenv("MRTRIX_NTHREADS", "1", 1);
  // The estimator's threaded driver announces itself with a progress bar and an informational
  //   summary; neither belongs in this executable's output, which is a list of checks.
  MR::App::log_level = 0;

  std::printf("== Adaptive phase correction: multi-resolution cold solve ==\n");

  // -- A plane large enough for three levels: 12 -> 6 -> 3, and no further. ------------------
  constexpr ssize_t Nlarge = 12;
  MR::Image<MR::cfloat> large = make_data(Nlarge);

  for (const regime_t regime : {regime_t::COLD_COARSE, regime_t::COLD_COARSE_THEN_WARM}) {
    const std::string name = regime == regime_t::COLD_COARSE ? "COLD_COARSE" : "COLD_COARSE_THEN_WARM";
    MR::Image<MR::cfloat> shipped = solve(large, Nlarge, 0, regime);
    MR::Image<MR::cfloat> depth2 = solve(large, Nlarge, 2, regime);
    MR::Image<MR::cfloat> depth1 = solve(large, Nlarge, 1, regime);
    check(identical(shipped, depth2), name + ": depth 2 reproduces the default bit for bit");
    check(!identical(shipped, depth1), name + ": depth 1 differs from depth 2");
    check(solved_everywhere(shipped), name + ": the default solves every plane");
    check(solved_everywhere(depth1), name + ": depth 1 solves every plane");
  }

  {
    MR::Image<MR::cfloat> depth3 = solve(large, Nlarge, 3, regime_t::COLD_COARSE_THEN_WARM);
    MR::Image<MR::cfloat> depth2 = solve(large, Nlarge, 2, regime_t::COLD_COARSE_THEN_WARM);
    check(!identical(depth3, depth2), "12x12: depth 3 differs from depth 2");
    check(solved_everywhere(depth3), "12x12: depth 3 solves every plane");
    for (const ssize_t depth : {4, 6, 12}) {
      MR::Image<MR::cfloat> deeper = solve(large, Nlarge, depth, regime_t::COLD_COARSE_THEN_WARM);
      check(identical(depth3, deeper),
            "12x12: depth " + std::to_string(depth) + " is capped at the plane's floor of 3 levels");
    }
  }

  // -- A plane too small to decimate at all: every depth must solve it cold at native size. ---
  constexpr ssize_t Nsmall = 3;
  MR::Image<MR::cfloat> small = make_data(Nsmall);
  {
    MR::Image<MR::cfloat> depth1 = solve(small, Nsmall, 1, regime_t::COLD_COARSE_THEN_WARM);
    check(solved_everywhere(depth1), "3x3: the plane is solved");
    for (const ssize_t depth : {2, 3, 5}) {
      MR::Image<MR::cfloat> deeper = solve(small, Nsmall, depth, regime_t::COLD_COARSE_THEN_WARM);
      check(identical(depth1, deeper),
            "3x3: depth " + std::to_string(depth) + " matches the native cold solve");
    }
    MR::Image<MR::cfloat> shipped = solve(small, Nsmall, 0, regime_t::COLD_COARSE);
    check(identical(depth1, shipped), "3x3: the default also solves it cold at native size");
  }

  if (failures != 0) {
    std::printf("\n%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
