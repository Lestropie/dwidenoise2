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

// Self-contained unit test for the per-plane fidelity weight that adaptive phase correction carries
//   between passes (AdaptivePhaseEstimator::Params::carry_lambda and the io_lambda argument of
//   operator()).
//
// The state is one scalar per (serialised volume, slice). What makes that indexing load-bearing is
//   temporal sub-sampling: a noise-estimation schedule may estimate a different subset of volumes
//   in each pass, so a volume can reach its first estimation at any pass, and the position of a
//   volume within a pass's subset -- or the order in which the threaded loop happens to visit
//   planes -- is not stable between passes. Indexing the state by anything but the serialised volume
//   index would therefore hand one volume's converged weight to another. Nothing in a per-plane
//   accuracy metric would show that: both volumes solve, both return a plausible phase, and the
//   only symptom is that the phase of one of them silently depends on the data of the other.
//
// A plain executable whose non-zero exit status indicates failure, following
//   testing/unit_tests/{noise_convention,noise_model,dof_estimation}.cpp. It drives the real
//   AdaptivePhaseEstimator through its public full-image entry point, so what is exercised is the
//   estimator's own plane-to-slot mapping and not a re-implementation of it.
//
// What is asserted:
//
//   1. A pass that estimates a subset of volumes writes state for exactly those volumes' planes and
//      leaves every other plane's entry empty. This is what fails if the state is indexed by a
//      running counter over estimated planes, or by position within the subset.
//   2. The value stored for a given plane does not depend on which other volumes the pass estimated:
//      estimating one volume alone reproduces, bit for bit, the entry that estimating it alongside
//      others produced.
//   3. Perturbing one plane's stored weight changes the phase that plane subsequently solves to, and
//      changes no other plane's -- neither another volume's, nor another slice of the same volume.
//      This pins both axes of the index at once, and is the direct statement of the property that a
//      volume reaching a later pass with no history must not inherit another volume's weight.
//   4. With Params::carry_lambda unset the same perturbation changes nothing at all, so the state is
//      recorded but not consumed.
//
// The iteration budgets are cut far below the shipped ones throughout: this test is about which
//   plane a scalar reaches, not about how well any plane is solved, and the phase's dependence on
//   the starting weight is established within a handful of sweeps.

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

constexpr ssize_t Nx = 16;
constexpr ssize_t Ny = 16;
constexpr ssize_t Nz = 3;
constexpr ssize_t Nvol = 6;

// Budgets small enough that the whole test is a few hundred thousand voxel-sweeps even in a debug
//   build, but large enough that the solved phase depends on the fidelity weight it started from.
AdaptivePhaseEstimator::Params test_params(const bool carry) {
  AdaptivePhaseEstimator::Params params;
  params.cp_iter = 8;
  params.max_lambda_iter = 3;
  params.polish_iter = 4;
  params.cp_iter_warm = 6;
  params.max_lambda_iter_warm = 2;
  params.polish_iter_warm = 4;
  params.carry_lambda = carry;
  return params;
}

MR::Header make_header() {
  MR::Header H;
  H.ndim() = 4;
  H.size(0) = Nx;
  H.size(1) = Ny;
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

// A smooth magnitude and a smooth background phase plus noise, with the magnitude scale and the
//   phase geometry both varying from volume to volume and from slice to slice. That variation is
//   what makes each plane's converged fidelity weight distinct, and hence what lets a mis-indexed
//   store be detected at all.
MR::Image<MR::cfloat> make_data() {
  MR::Image<MR::cfloat> image = MR::Image<MR::cfloat>::scratch(make_header(), "test data");
  std::mt19937_64 rng(20260806);
  std::normal_distribution<double> noise(0.0, 4.0);
  for (ssize_t v = 0; v != Nvol; ++v) {
    image.index(3) = v;
    for (ssize_t z = 0; z != Nz; ++z) {
      image.index(2) = z;
      const double scale = 60.0 * (1.0 + 0.35 * double(v)) * (1.0 + 0.2 * double(z));
      const double twist = 0.5 + 0.3 * double(v) - 0.15 * double(z);
      for (ssize_t j = 0; j != Ny; ++j) {
        image.index(1) = j;
        for (ssize_t i = 0; i != Nx; ++i) {
          image.index(0) = i;
          const double x = (double(i) - 0.5 * double(Nx - 1)) / double(Nx);
          const double y = (double(j) - 0.5 * double(Ny - 1)) / double(Ny);
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

MR::Image<MR::cfloat> unit_phase() {
  MR::Image<MR::cfloat> phase = MR::Image<MR::cfloat>::scratch(make_header(), "test phase");
  for (auto l = MR::Loop(phase)(phase); l; ++l)
    phase.value() = MR::cfloat(1.0f, 0.0f);
  return phase;
}

MR::Image<MR::cfloat> copy_of(MR::Image<MR::cfloat> &source) {
  MR::Image<MR::cfloat> result = MR::Image<MR::cfloat>::scratch(make_header(), "test phase copy");
  for (auto l = MR::Loop(result)(result, source); l; ++l)
    result.value() = source.value();
  return result;
}

// Run one pass over the image, in place on "phase" and "store".
void run_pass(MR::Image<MR::cfloat> &image,
              MR::Image<MR::cfloat> &phase,
              std::vector<double> &store,
              const std::vector<VolumePlan> &plan,
              const bool carry) {
  const AdaptivePhaseEstimator estimator(std::vector<size_t>{0, 1}, test_params(carry));
  MR::Image<float> no_sigma;
  MR::Image<uint32_t> no_serialise;
  estimator(image, no_sigma, phase, no_serialise, plan, &store);
}

std::vector<VolumePlan> plan_for(const std::vector<ssize_t> &volumes, const regime_t regime) {
  std::vector<VolumePlan> plan(Nvol);
  for (const ssize_t v : volumes)
    plan[size_t(v)] = {true, regime};
  return plan;
}

// Number of planes at which two phase maps differ, and whether a specific plane is among them.
ssize_t count_differing_planes(MR::Image<MR::cfloat> &a, MR::Image<MR::cfloat> &b, std::vector<bool> &which) {
  which.assign(size_t(Nvol * Nz), false);
  for (ssize_t v = 0; v != Nvol; ++v) {
    a.index(3) = v;
    b.index(3) = v;
    for (ssize_t z = 0; z != Nz; ++z) {
      a.index(2) = z;
      b.index(2) = z;
      for (ssize_t j = 0; j != Ny && !which[size_t(v * Nz + z)]; ++j) {
        a.index(1) = j;
        b.index(1) = j;
        for (ssize_t i = 0; i != Nx; ++i) {
          a.index(0) = i;
          b.index(0) = i;
          if (MR::cfloat(a.value()) != MR::cfloat(b.value())) {
            which[size_t(v * Nz + z)] = true;
            break;
          }
        }
      }
    }
  }
  ssize_t total = 0;
  for (const bool differs : which)
    if (differs)
      total += 1;
  return total;
}

} // namespace

int main() {
  // Deterministic and free of any dependence on the host's core count: the estimator's threaded
  //   loop distributes planes over threads, and while each plane is an independent problem writing
  //   to its own slot, a single thread makes the test's behaviour identical everywhere.
  setenv("MRTRIX_NTHREADS", "1", 1);
  // The estimator's threaded driver announces itself with a progress bar and an informational
  //   summary; neither belongs in this executable's output, which is a list of checks.
  MR::App::log_level = 0;

  std::printf("== Adaptive phase correction: fidelity weight carried between passes ==\n");

  MR::Image<MR::cfloat> image = make_data();

  // ---- 1. A sub-sampled pass writes state for exactly the volumes it estimated ------------------
  // The first pass of a schedule that sub-samples: three of six volumes, each with no prior
  //   estimate and no later pass to refine it, so each is solved cold on the coarse grid and then
  //   refined at native resolution within the same pass. It is that native stage that converges the
  //   weight recorded here.
  const std::vector<ssize_t> subset{0, 2, 4};
  std::vector<double> store(size_t(Nvol * Nz), 0.0);
  MR::Image<MR::cfloat> phase = unit_phase();
  run_pass(image, phase, store, plan_for(subset, regime_t::COLD_COARSE_THEN_WARM), true);

  bool estimated_all_positive = true;
  bool unestimated_all_zero = true;
  for (ssize_t v = 0; v != Nvol; ++v) {
    const bool estimated = (v % 2 == 0);
    for (ssize_t z = 0; z != Nz; ++z) {
      const double value = store[size_t(v * Nz + z)];
      if (estimated && !(value > 0.0))
        estimated_all_positive = false;
      if (!estimated && value != 0.0)
        unestimated_all_zero = false;
    }
  }
  check(estimated_all_positive, "every estimated volume's planes record a fidelity weight");
  check(unestimated_all_zero, "no unestimated volume's plane records one");

  // Distinct planes must hold distinct values, or checks 2 and 3 could pass by coincidence.
  bool values_distinct = true;
  for (const ssize_t v : subset)
    for (ssize_t z = 0; z != Nz; ++z)
      for (const ssize_t v2 : subset)
        for (ssize_t z2 = 0; z2 != Nz; ++z2)
          if ((v != v2 || z != z2) && store[size_t(v * Nz + z)] == store[size_t(v2 * Nz + z2)])
            values_distinct = false;
  check(values_distinct, "each estimated plane converged to its own distinct fidelity weight");

  // ---- 2. The value stored for a plane is independent of what else the pass estimated -----------
  std::vector<double> store_alone(size_t(Nvol * Nz), 0.0);
  MR::Image<MR::cfloat> phase_alone = unit_phase();
  run_pass(image, phase_alone, store_alone, plan_for({2}, regime_t::COLD_COARSE_THEN_WARM), true);
  bool matches_alone = true;
  for (ssize_t z = 0; z != Nz; ++z)
    if (store_alone[size_t(2 * Nz + z)] != store[size_t(2 * Nz + z)])
      matches_alone = false;
  check(matches_alone, "estimating one volume alone reproduces the weight it recorded alongside others");

  // ---- 3. A perturbed entry reaches exactly one plane -------------------------------------------
  // The second pass of the schedule: the volumes estimated above are refined warm, and the three
  //   that the first pass sub-sampled away reach their own first estimation here, with no history.
  const std::vector<ssize_t> rest{1, 3, 5};
  std::vector<VolumePlan> second_pass = plan_for(subset, regime_t::WARM);
  for (const ssize_t v : rest)
    second_pass[size_t(v)] = {true, regime_t::COLD_COARSE_THEN_WARM};

  constexpr ssize_t perturbed_volume = 2;
  constexpr ssize_t perturbed_slice = 1;
  const size_t perturbed = size_t(perturbed_volume * Nz + perturbed_slice);

  MR::Image<MR::cfloat> phase_reference = copy_of(phase);
  std::vector<double> store_reference = store;
  run_pass(image, phase_reference, store_reference, second_pass, true);

  MR::Image<MR::cfloat> phase_perturbed = copy_of(phase);
  std::vector<double> store_perturbed = store;
  store_perturbed[perturbed] *= 4.0;
  run_pass(image, phase_perturbed, store_perturbed, second_pass, true);

  std::vector<bool> differs;
  const ssize_t n_differing = count_differing_planes(phase_reference, phase_perturbed, differs);
  std::printf("   planes whose phase changed when one plane's stored weight was perturbed: %ld\n",
              long(n_differing));
  check(n_differing == 1, "exactly one plane's phase responds to a perturbation of one stored weight");
  check(n_differing == 1 && differs[perturbed],
        "the plane that responds is the one whose weight was perturbed");

  // ---- 4. The state is not consumed unless carrying is enabled ---------------------------------
  MR::Image<MR::cfloat> phase_off = copy_of(phase);
  std::vector<double> store_off = store;
  run_pass(image, phase_off, store_off, second_pass, false);

  MR::Image<MR::cfloat> phase_off_perturbed = copy_of(phase);
  std::vector<double> store_off_perturbed = store;
  store_off_perturbed[perturbed] *= 4.0;
  run_pass(image, phase_off_perturbed, store_off_perturbed, second_pass, false);

  const ssize_t n_differing_off = count_differing_planes(phase_off, phase_off_perturbed, differs);
  check(n_differing_off == 0, "with carrying disabled, a perturbed stored weight changes no phase");

  // And the state is still recorded, so that enabling the carry is the only change required.
  bool recorded_when_disabled = true;
  for (ssize_t v = 0; v != Nvol; ++v)
    for (ssize_t z = 0; z != Nz; ++z)
      if (!(store_off[size_t(v * Nz + z)] > 0.0))
        recorded_when_disabled = false;
  check(recorded_when_disabled, "with carrying disabled, every solved plane still records its weight");

  if (failures != 0) {
    std::printf("\n%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
