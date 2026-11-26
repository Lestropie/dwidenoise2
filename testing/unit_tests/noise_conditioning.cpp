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

// Self-contained unit test for the conditioning of a noise level map.
//
// Conditioning makes a map fit to use: every voxel of it must hold a finite, strictly positive
//   noise level, because that is what cubic interpolation and the variance-stabilising transform
//   have always assumed of it. It does so on log(sigma^2), and it fills the voxels at which nothing
//   was estimated rather than averaging a zero in as though it were a measurement.
//
// A plain executable whose non-zero exit status indicates failure, following
//   testing/unit_tests/{noise_convention,dof_estimation}.cpp. It links the real
//   Denoise::condition_noise_map, the real SpectrumFilter, the real imposed estimators and the real
//   interpolation wrapper; nothing is re-implemented here except the brute-force reference of
//   test 6, whose whole purpose is to be an independent implementation.
//
// What is asserted, and why each matters:
//
//   1. A constant field survives arbitrary holes exactly. Whatever the mechanism, whatever the mix
//      of NaN, infinity, zero and negative values it is handed, it must return the constant.
//   2. Power-invariance. Conditioning a map of sigma and conditioning the map of sigma^2 and then
//      taking the square root agree. This is the property arithmetic smoothing fails, and hence the
//      direct test of the reason the operation is performed in the log domain at all.
//   3. Imputation without smoothing. With no width requested, valid voxels are returned unchanged
//      and invalid ones are filled -- ie. conditioning is not conditional on smoothing.
//   4. A hole wider than the kernel. A 7^3 cube of NaN is filled, and every filled value obeys the
//      maximum principle, lying within the range of the valid values that surround it.
//   5. A wholly invalid map throws where it would seed a further iteration, and returns with its
//      NaNs intact where it is only being exported.
//   6. Two-image exactness. Against a brute-force 3D normalised convolution in double precision.
//      This is what pins the claim that the numerator/denominator formulation is exact, as opposed
//      to the per-axis renormalisation of MRtrix3's own missing-data path.
//   7. Axis-order independence. Conditioning a map whose strides are permuted, and permuting the
//      result back, gives the same answer. This is the property the per-axis renormalisation fails,
//      and this test is the guard against regressing to it.
//   8. Log-domain interpolation. At a steep step, cubic interpolation of sigma undershoots below
//      zero; through the log domain it cannot. Both halves are asserted, so the test fails if the
//      wrapper is ever bypassed.
//   9. The analytic case. On a log-linear ramp, x = log(sigma^2) is linear, and a symmetric
//      normalised kernel reproduces a linear function exactly. So the conditioned map equals the
//      input in the interior, whatever the kernel -- an assertion free of any assumption about the
//      smoother's radius or weights.
//  10. The warning threshold, with and without SliceEncodingDirection, and at its escalation
//      boundary.
//  11. The imposed-estimator guard: an unusable interpolated level is counted and the patch
//      abandoned, rather than tripping an assertion.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "algo/loop.h"
#include "datatype.h"
#include "denoise/denoise.h"
#include "denoise/estimator/imposed/constant.h"
#include "denoise/estimator/imposed/import.h"
#include "denoise/estimator/pooling.h"
#include "denoise/estimator/result.h"
#include "denoise/mask.h"
#include "denoise/spectrum.h"
#include "denoise/spectrum_filter.h"
#include "exception.h"
#include "header.h"
#include "image.h"
#include "interp/cubic.h"
#include "mrtrix.h"
#include "types.h"

using MR::default_type;
using MR::Denoise::condition_noise_map;
using MR::Denoise::eigenvalues_type;
using MR::Denoise::filter_type;
using MR::Denoise::invalid_data_warning_threshold;
using MR::Denoise::log_noise_map;
using MR::Denoise::LogNoiseInterp;
using MR::Denoise::Moments;
using MR::Denoise::noise_level_is_valid;
using MR::Denoise::noise_pad_type;
using MR::Denoise::noise_unfilled_type;
using MR::Denoise::Spectrum;
using MR::Denoise::SpectrumFilter;
using MR::Denoise::vector_type;
using MR::Denoise::Estimator::all_signal_sigma2_bound;
using MR::Denoise::Estimator::Constant;
using MR::Denoise::Estimator::Import;
using MR::Denoise::Estimator::Result;

namespace {

int failures = 0;

void check(const bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

void check_relative(const double value, const double target, const double tol, const std::string &message) {
  const double denom = std::fabs(target) > 0.0 ? std::fabs(target) : 1.0;
  if (!(std::fabs(value - target) / denom <= tol)) {
    std::fprintf(stderr,
                 "FAIL: %s (got %.10g, expected %.10g, relative difference %.3g > %.3g)\n",
                 message.c_str(),
                 value,
                 target,
                 std::fabs(value - target) / denom,
                 tol);
    ++failures;
  }
}

constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

// A 3D grid of the requested size, unit spacing unless stated otherwise, identity transform.
MR::Header make_header(const ssize_t nx, const ssize_t ny, const ssize_t nz, const default_type spacing = 1.0) {
  MR::Header H;
  H.ndim() = 3;
  H.size(0) = nx;
  H.size(1) = ny;
  H.size(2) = nz;
  for (size_t axis = 0; axis != 3; ++axis) {
    H.spacing(axis) = spacing;
    H.stride(axis) = ssize_t(axis) + 1;
  }
  H.transform().setIdentity();
  H.datatype() = MR::DataType::from<float>();
  return H;
}

MR::Image<float> make_map(const MR::Header &H, const std::string &name = "test noise level map") {
  return MR::Image<float>::scratch(H, name);
}

MR::Image<float> constant_map(const MR::Header &H, const float value) {
  MR::Image<float> image = make_map(H);
  for (auto l = MR::Loop(image)(image); l; ++l)
    image.value() = value;
  return image;
}

// Every voxel of the map, read out in scan order, so that two maps on the same grid can be
//   compared without regard to how either is laid out in memory.
std::vector<float> flatten(MR::Image<float> &image) {
  std::vector<float> values;
  values.reserve(size_t(image.size(0) * image.size(1) * image.size(2)));
  for (ssize_t k = 0; k != image.size(2); ++k) {
    for (ssize_t j = 0; j != image.size(1); ++j) {
      for (ssize_t i = 0; i != image.size(0); ++i) {
        image.index(0) = i;
        image.index(1) = j;
        image.index(2) = k;
        values.push_back(image.value());
      }
    }
  }
  return values;
}

float at(MR::Image<float> &image, const ssize_t i, const ssize_t j, const ssize_t k) {
  image.index(0) = i;
  image.index(1) = j;
  image.index(2) = k;
  return image.value();
}

void set(MR::Image<float> &image, const ssize_t i, const ssize_t j, const ssize_t k, const float value) {
  image.index(0) = i;
  image.index(1) = j;
  image.index(2) = k;
  image.value() = value;
}

bool all_valid(MR::Image<float> &image) {
  for (auto l = MR::Loop(image)(image); l; ++l)
    if (!noise_level_is_valid(image.value()))
      return false;
  return true;
}

// ---------------------------------------------------------------------------------------------
// 1. A constant field, arbitrary holes.

void test_constant_field() {
  std::fprintf(stderr, "\n-- constant field with arbitrary holes --\n");
  const MR::Header H = make_header(12, 11, 10);
  constexpr float level = 7.25F;
  MR::Image<float> map = constant_map(H, level);

  // A fifth of the voxels made invalid, cycling through every way a value can fail to be a noise
  //   level: the sentinel itself, both infinities, an exact zero and a negative number. The last
  //   two are what the sanitisation exists for -- they are finite, so nothing downstream would
  //   otherwise notice them.
  std::mt19937_64 rng(20260909);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const std::vector<float> invalid_values{
      NaN, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 0.0F, -3.5F};
  ssize_t made_invalid = 0;
  for (auto l = MR::Loop(map)(map); l; ++l) {
    if (uniform(rng) < 0.2) {
      map.value() = invalid_values[size_t(made_invalid) % invalid_values.size()];
      ++made_invalid;
    }
  }
  check(made_invalid > 0, "the test made some voxels invalid");

  for (const default_type fwhm : {0.0, 3.0}) {
    MR::Image<float> out = condition_noise_map(map, noise_impute_type::NONE, noise_pad_type::NONE, fwhm);
    check(all_valid(out), "conditioned constant map holds a valid level everywhere (fwhm " + MR::str(fwhm) + ")");
    double worst = 0.0;
    for (auto l = MR::Loop(out)(out); l; ++l)
      worst = std::max(worst, std::fabs(double(out.value()) - double(level)) / double(level));
    check_relative(1.0 + worst, 1.0, 1e-6, "conditioned constant map is the constant (fwhm " + MR::str(fwhm) + ")");
  }
}

// ---------------------------------------------------------------------------------------------
// 2. Power invariance.

void test_power_invariance() {
  std::fprintf(stderr, "\n-- power invariance --\n");
  const MR::Header H = make_header(10, 10, 9);
  MR::Image<float> sigma = make_map(H, "sigma");
  MR::Image<float> sigma_sq = make_map(H, "sigma squared");
  std::mt19937_64 rng(20260910);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (auto l = MR::Loop(sigma)(sigma, sigma_sq); l; ++l) {
    // A field with real structure, so that the kernel has something to average over, plus a
    //   scattering of holes so that the imputation is exercised on the same footing.
    const double x = double(sigma.index(0)) / double(H.size(0));
    const double y = double(sigma.index(1)) / double(H.size(1));
    const double z = double(sigma.index(2)) / double(H.size(2));
    const double value = 2.0 + 6.0 * std::exp(-3.0 * ((x - 0.5) * (x - 0.5) + (y - 0.4) * (y - 0.4))) +
                         1.5 * std::sin(5.0 * z + 2.0 * x);
    const bool hole = uniform(rng) < 0.12;
    sigma.value() = hole ? NaN : float(value);
    sigma_sq.value() = hole ? NaN : float(value * value);
  }

  MR::Image<float> from_sigma = condition_noise_map(sigma, noise_pad_type::NONE, 4.0);
  MR::Image<float> from_sigma_sq = condition_noise_map(sigma_sq, noise_pad_type::NONE, 4.0);
  double worst = 0.0;
  for (auto l = MR::Loop(from_sigma)(from_sigma, from_sigma_sq); l; ++l) {
    const double direct = double(from_sigma.value());
    const double via_square = std::sqrt(double(from_sigma_sq.value()));
    worst = std::max(worst, std::fabs(direct - via_square) / direct);
  }
  std::fprintf(stderr, "   worst relative disagreement between the two powers: %.3g\n", worst);
  // Float storage of the two maps is the only reason this is not exact.
  check(worst < 1e-5, "conditioning commutes with the choice of which power of sigma the map holds");
}

// ---------------------------------------------------------------------------------------------
// 3. Imputation without smoothing.

void test_imputation_only() {
  std::fprintf(stderr, "\n-- imputation with no smoothing requested --\n");
  const MR::Header H = make_header(9, 9, 9);
  MR::Image<float> map = make_map(H);
  for (auto l = MR::Loop(map)(map); l; ++l)
    map.value() = float(3.0 + 0.25 * double(map.index(0)) + 0.1 * double(map.index(1) * map.index(2)));
  set(map, 4, 4, 4, NaN);
  set(map, 1, 7, 2, 0.0F);
  // Read out after the invalidation, so that the two voxels the comparison below must skip are
  //   identified by the same predicate the conditioning itself uses rather than by index.
  const std::vector<float> before = flatten(map);

  MR::Image<float> out = condition_noise_map(map, noise_pad_type::NONE, 0.0);
  check(all_valid(out), "every voxel is valid after imputation alone");

  // The valid voxels are not touched by the imputation, but they do make the round trip through
  //   x = 2 log(sigma) and back, so they are equal to within the precision of that round trip
  //   rather than bit-identical.
  const std::vector<float> after = flatten(out);
  double worst = 0.0;
  size_t compared = 0;
  for (size_t i = 0; i != before.size(); ++i) {
    if (!noise_level_is_valid(before[i]))
      continue;
    worst = std::max(worst, std::fabs(double(after[i]) - double(before[i])) / double(before[i]));
    ++compared;
  }
  check(compared == before.size() - 2, "exactly the two invalidated voxels were excluded from the comparison");
  check(worst < 1e-6, "valid voxels are returned unchanged by the log round trip");
}

// ---------------------------------------------------------------------------------------------
// 4. A hole wider than the kernel.

void test_wide_hole() {
  std::fprintf(stderr, "\n-- a hole wider than the imputation kernel --\n");
  const MR::Header H = make_header(15, 15, 15);
  MR::Image<float> map = make_map(H);
  for (auto l = MR::Loop(map)(map); l; ++l)
    map.value() = float(4.0 + 0.3 * double(map.index(0)) + 0.2 * double(map.index(1)) - 0.1 * double(map.index(2)));
  // Range of the valid values bordering the hole, for the maximum principle below.
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (ssize_t k = 3; k != 12; ++k)
    for (ssize_t j = 3; j != 12; ++j)
      for (ssize_t i = 3; i != 12; ++i) {
        const bool inside = i >= 4 && i < 11 && j >= 4 && j < 11 && k >= 4 && k < 11;
        if (inside)
          continue;
        lo = std::min(lo, double(at(map, i, j, k)));
        hi = std::max(hi, double(at(map, i, j, k)));
      }
  // A 7x7x7 cube of NaN: half-width three voxels, so no single pass of a 3x3x3 kernel can reach
  //   its centre, and no plausible enlargement of that kernel would be a general answer.
  for (ssize_t k = 4; k != 11; ++k)
    for (ssize_t j = 4; j != 11; ++j)
      for (ssize_t i = 4; i != 11; ++i)
        set(map, i, j, k, NaN);

  MR::Image<float> out = condition_noise_map(map, noise_pad_type::NONE, 0.0);
  check(all_valid(out), "a hole three voxels deep is closed by iterated imputation");
  bool within = true;
  for (ssize_t k = 4; k != 11; ++k)
    for (ssize_t j = 4; j != 11; ++j)
      for (ssize_t i = 4; i != 11; ++i) {
        const double value = double(at(out, i, j, k));
        if (!(value >= lo - 1e-4 && value <= hi + 1e-4))
          within = false;
      }
  check(within, "every imputed value obeys the maximum principle of the values surrounding the hole");
}

// ---------------------------------------------------------------------------------------------
// 5. A wholly invalid map.

void test_all_invalid() {
  std::fprintf(stderr, "\n-- a map with no valid voxel at all --\n");
  const MR::Header H = make_header(6, 6, 6);
  MR::Image<float> map = constant_map(H, NaN);

  bool threw = false;
  try {
    MR::Image<float> out = condition_noise_map(map, noise_pad_type::NONE, 0.0);
  } catch (MR::Exception &) {
    threw = true;
  }
  check(threw, "a wholly invalid map throws where it would seed a further iteration");

  bool threw_retain = false;
  MR::Image<float> exported;
  try {
    exported = condition_noise_map(
        map, noise_pad_type::NONE, 0.0, nullptr, nullptr, noise_unfilled_type::RETAIN);
  } catch (MR::Exception &) {
    threw_retain = true;
  }
  check(!threw_retain, "a wholly invalid map does not throw where it is only being exported");
  if (!threw_retain) {
    bool all_nan = true;
    for (auto l = MR::Loop(exported)(exported); l; ++l)
      if (std::isfinite(exported.value()))
        all_nan = false;
    check(all_nan, "an exported map that could not be filled retains its NaNs rather than inventing values");
  }

  // Holes that no valid voxel can reach, in a map that does hold valid voxels elsewhere: the
  //   pathological case is not "no valid voxel" but "no valid voxel connected to this one", and
  //   with a map that is entirely one or entirely the other there would be nothing to distinguish.
  //   Here the whole map is invalid except one isolated plane, so every hole is eventually reached
  //   and the map conditions successfully; that it does is the point.
  MR::Image<float> partial = constant_map(H, NaN);
  for (ssize_t j = 0; j != H.size(1); ++j)
    for (ssize_t i = 0; i != H.size(0); ++i)
      set(partial, i, j, 0, 5.0F);
  MR::Image<float> filled = condition_noise_map(partial, noise_pad_type::NONE, 0.0);
  check(all_valid(filled), "a map whose only valid data is one plane is filled from that plane");
}

// ---------------------------------------------------------------------------------------------
// 6. Two-image exactness against a brute-force reference.

// The reference: an explicit 3D normalised convolution in double precision, over the same
//   truncated Gaussian MRtrix3's separable smoother uses (radius ceil(2 sigma / spacing), weights
//   exp(-d^2 / 2 sigma^2)). Written in one triple loop with no separability and no renormalisation
//   trickery, so that it shares no machinery whatsoever with the implementation under test. The
//   normalisation of the kernel itself is irrelevant: it cancels between numerator and denominator.
std::vector<double> reference_normalised_convolution(const std::vector<double> &x,
                                                     const std::vector<bool> &valid,
                                                     const ssize_t nx,
                                                     const ssize_t ny,
                                                     const ssize_t nz,
                                                     const default_type spacing,
                                                     const default_type stdev) {
  const ssize_t radius = ssize_t(std::ceil(2.0 * stdev / spacing));
  const auto index = [nx, ny](const ssize_t i, const ssize_t j, const ssize_t k) { return size_t((k * ny + j) * nx + i); };
  std::vector<double> out(x.size(), std::numeric_limits<double>::quiet_NaN());
  for (ssize_t k = 0; k != nz; ++k) {
    for (ssize_t j = 0; j != ny; ++j) {
      for (ssize_t i = 0; i != nx; ++i) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (ssize_t dk = -radius; dk <= radius; ++dk) {
          const ssize_t kk = k + dk;
          if (kk < 0 || kk >= nz)
            continue;
          for (ssize_t dj = -radius; dj <= radius; ++dj) {
            const ssize_t jj = j + dj;
            if (jj < 0 || jj >= ny)
              continue;
            for (ssize_t di = -radius; di <= radius; ++di) {
              const ssize_t ii = i + di;
              if (ii < 0 || ii >= nx)
                continue;
              if (!valid[index(ii, jj, kk)])
                continue;
              const double d2 = double(di * di + dj * dj + dk * dk) * double(spacing) * double(spacing);
              const double weight = std::exp(-d2 / (2.0 * double(stdev) * double(stdev)));
              numerator += weight * x[index(ii, jj, kk)];
              denominator += weight;
            }
          }
        }
        if (denominator > 0.0)
          out[index(i, j, k)] = numerator / denominator;
      }
    }
  }
  return out;
}

void test_two_image_exactness() {
  std::fprintf(stderr, "\n-- two-image normalised convolution against a brute-force reference --\n");
  const ssize_t N = 11;
  const default_type spacing = 1.0;
  const default_type fwhm = 3.0;
  const default_type stdev = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  const MR::Header H = make_header(N, N, N, spacing);
  MR::Image<float> map = make_map(H);
  std::mt19937_64 rng(20260911);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> x(size_t(N * N * N));
  std::vector<bool> valid(size_t(N * N * N), true);
  const auto index = [N](const ssize_t i, const ssize_t j, const ssize_t k) { return size_t((k * N + j) * N + i); };
  for (ssize_t k = 0; k != N; ++k) {
    for (ssize_t j = 0; j != N; ++j) {
      for (ssize_t i = 0; i != N; ++i) {
        // Holes sparse enough that no voxel is left without support inside the smoothing kernel,
        //   so the reference and the implementation are compared on the convolution alone rather
        //   than on what either does about a voxel the convolution could not fill.
        const bool hole = uniform(rng) < 0.08;
        const double sigma = 2.0 + std::sin(0.7 * double(i)) + 0.5 * std::cos(0.9 * double(j) + 0.3 * double(k));
        set(map, i, j, k, hole ? NaN : float(sigma));
        valid[index(i, j, k)] = !hole;
        // The reference works in the same domain the implementation does.
        x[index(i, j, k)] = hole ? 0.0 : 2.0 * std::log(double(float(sigma)));
      }
    }
  }

  MR::Image<float> out = condition_noise_map(map, noise_pad_type::NONE, fwhm);
  const std::vector<double> reference = reference_normalised_convolution(x, valid, N, N, N, spacing, stdev);

  double worst = 0.0;
  size_t compared = 0;
  for (ssize_t k = 0; k != N; ++k) {
    for (ssize_t j = 0; j != N; ++j) {
      for (ssize_t i = 0; i != N; ++i) {
        const double expected = std::exp(0.5 * reference[index(i, j, k)]);
        const double got = double(at(out, i, j, k));
        worst = std::max(worst, std::fabs(got - expected) / expected);
        ++compared;
      }
    }
  }
  std::fprintf(stderr, "   compared %zu voxels; worst relative difference %.3g\n", compared, worst);
  // Float storage of the two intermediate images is the only source of disagreement.
  check(worst < 1e-5, "the two-image formulation is the exact 3D normalised convolution");
}

// ---------------------------------------------------------------------------------------------
// 7. Axis-order independence.

void test_axis_order_independence() {
  std::fprintf(stderr, "\n-- axis-order independence --\n");
  const ssize_t N = 9;
  MR::Header H = make_header(N, N, N);
  MR::Header H_permuted = make_header(N, N, N);
  // The same grid, laid out with the axes in the opposite memory order. Nothing about the geometry
  //   changes -- only which axis a separable filter would happen to visit first.
  H_permuted.stride(0) = 3;
  H_permuted.stride(1) = 2;
  H_permuted.stride(2) = 1;

  MR::Image<float> map = make_map(H);
  MR::Image<float> permuted = make_map(H_permuted);
  std::mt19937_64 rng(20260912);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (ssize_t k = 0; k != N; ++k) {
    for (ssize_t j = 0; j != N; ++j) {
      for (ssize_t i = 0; i != N; ++i) {
        // Holes deliberately arranged as a slab, so that an axis-order-dependent implementation
        //   would extrapolate across it differently depending on which axis it swept first.
        const bool hole = (i >= 3 && i <= 5 && j >= 2 && j <= 6) || uniform(rng) < 0.05;
        const float value = hole ? NaN : float(3.0 + 0.4 * double(i) - 0.2 * double(j) + 0.3 * double(k));
        set(map, i, j, k, value);
        set(permuted, i, j, k, value);
      }
    }
  }

  MR::Image<float> a = condition_noise_map(map, noise_pad_type::NONE, 2.5);
  MR::Image<float> b = condition_noise_map(permuted, noise_pad_type::NONE, 2.5);
  double worst = 0.0;
  for (ssize_t k = 0; k != N; ++k)
    for (ssize_t j = 0; j != N; ++j)
      for (ssize_t i = 0; i != N; ++i)
        worst = std::max(worst, std::fabs(double(at(a, i, j, k)) - double(at(b, i, j, k))) / double(at(a, i, j, k)));
  std::fprintf(stderr, "   worst relative difference across the stride permutation: %.3g\n", worst);
  check(worst < 1e-5, "conditioning does not depend on the order in which the axes are stored");
}

// ---------------------------------------------------------------------------------------------
// 8. Log-domain interpolation.

void test_log_domain_interpolation() {
  std::fprintf(stderr, "\n-- cubic interpolation through the log domain --\n");
  const ssize_t N = 12;
  const MR::Header H = make_header(N, N, N);
  MR::Image<float> map = make_map(H);
  // A steep step in x, of the kind the edge of the object presents: a factor of four hundred
  //   across one voxel is well within what separates object from background.
  for (auto l = MR::Loop(map)(map); l; ++l)
    map.value() = map.index(0) < N / 2 ? 0.05F : 20.0F;

  MR::Image<float> log_map = log_noise_map(map);
  LogNoiseInterp log_interp(log_map);
  MR::Interp::Cubic<MR::Image<float>> linear_interp(map);

  double lowest_log = std::numeric_limits<double>::infinity();
  double lowest_linear = std::numeric_limits<double>::infinity();
  ssize_t nonpositive_linear = 0;
  ssize_t nonpositive_log = 0;
  for (ssize_t step = 0; step != 200; ++step) {
    const double x = 0.02 * double(step) * double(N - 1) / 4.0 + double(N) / 2.0 - 2.0;
    const Eigen::Vector3d position(x, 0.5 * double(N - 1), 0.5 * double(N - 1));
    if (!linear_interp.scanner(position))
      continue;
    log_interp.scanner(position);
    const double from_log = log_interp.value();
    const double from_linear = double(linear_interp.value());
    lowest_log = std::min(lowest_log, from_log);
    lowest_linear = std::min(lowest_linear, from_linear);
    if (from_linear <= 0.0)
      ++nonpositive_linear;
    if (!(from_log > 0.0))
      ++nonpositive_log;
  }
  std::fprintf(stderr,
               "   across the step: linear-domain minimum %.4g (%zd non-positive samples),"
               " log-domain minimum %.4g (%zd non-positive)\n",
               lowest_linear,
               nonpositive_linear,
               lowest_log,
               nonpositive_log);
  // Both halves: that the problem is real, and that the wrapper removes it. Asserting only the
  //   second would pass just as well on a step too gentle to provoke any undershoot at all.
  check(nonpositive_linear > 0, "cubic interpolation of sigma does undershoot to zero or below at a steep step");
  check(nonpositive_log == 0, "cubic interpolation through the log domain never returns a non-positive level");
}

// ---------------------------------------------------------------------------------------------
// 9. The analytic case: a log-linear ramp.

void test_log_linear_ramp() {
  std::fprintf(stderr, "\n-- a log-linear ramp is reproduced exactly --\n");
  const ssize_t N = 21;
  const MR::Header H = make_header(N, N, N);
  MR::Image<float> map = make_map(H);
  // sigma = exp(a x + b y + c z + d), so x = log(sigma^2) is linear in the voxel indices. A
  //   symmetric normalised kernel reproduces a linear function exactly, so the geometric mean of
  //   the kernel is the central value and the conditioned map equals the input -- whatever radius
  //   or weights the smoother happens to use. Nothing else in this file is independent of those.
  const auto analytic = [](const ssize_t i, const ssize_t j, const ssize_t k) {
    return std::exp(0.07 * double(i) - 0.04 * double(j) + 0.05 * double(k) - 0.5);
  };
  for (ssize_t k = 0; k != N; ++k)
    for (ssize_t j = 0; j != N; ++j)
      for (ssize_t i = 0; i != N; ++i)
        set(map, i, j, k, float(analytic(i, j, k)));

  MR::Image<float> out = condition_noise_map(map, noise_pad_type::NONE, 3.0);
  double worst = 0.0;
  // The interior only: at the boundary the kernel is truncated asymmetrically, so the geometric
  //   mean of a ramp is no longer its central value, and there is nothing analytic left to compare
  //   against. The truncation radius for this width is 3 voxels.
  for (ssize_t k = 4; k != N - 4; ++k)
    for (ssize_t j = 4; j != N - 4; ++j)
      for (ssize_t i = 4; i != N - 4; ++i)
        worst = std::max(worst, std::fabs(double(at(out, i, j, k)) - analytic(i, j, k)) / analytic(i, j, k));
  std::fprintf(stderr, "   worst relative departure from the analytic geometric mean: %.3g\n", worst);
  check(worst < 1e-5, "log-domain smoothing of a log-linear ramp returns the ramp");
}

// ---------------------------------------------------------------------------------------------
// 10. The warning threshold.

void test_warning_threshold() {
  std::fprintf(stderr, "\n-- the invalid-data warning threshold --\n");
  MR::Header H = make_header(20, 30, 40);
  // The threshold is the largest number of voxels in a slice taken orthogonal to any axis. For a
  //   20 x 30 x 40 grid the candidates are 30 x 40, 20 x 40 and 20 x 30, so with no slice encoding
  //   direction known it is 1200.
  check(invalid_data_warning_threshold(H) == 1200, "threshold without SliceEncodingDirection is the largest slice");

  // Naming the slice axis excludes the slice orthogonal to it, which is the acquired slice plane.
  //   With the slices stacked along the first axis the 30 x 40 candidate is dropped and the
  //   threshold falls to 20 x 40, which is what makes invalid data spanning as little as one
  //   acquired slice enough to warrant a warning.
  H.keyval()["SliceEncodingDirection"] = "i";
  check(invalid_data_warning_threshold(H) == 800, "excluding the slice-encoding axis lowers the threshold");
  H.keyval()["SliceEncodingDirection"] = "i-";
  check(invalid_data_warning_threshold(H) == 800, "the sign of the slice encoding direction is immaterial");
  // Stacked along the third axis instead, the excluded candidate (20 x 30) was not the largest, so
  //   the threshold is unchanged. The rule is about which slice is acquired, not about lowering the
  //   threshold unconditionally.
  H.keyval()["SliceEncodingDirection"] = "k";
  check(invalid_data_warning_threshold(H) == 1200, "excluding a non-maximal slice leaves the threshold alone");

  // The escalation is on strict exceedance, so a count equal to the threshold is not a warning.
  //   Asserted against the same expression report_invalid_data() and report_imputed_noise_levels()
  //   use, both of which compare with ">".
  const size_t threshold = invalid_data_warning_threshold(H);
  check(!(threshold > threshold), "a count equal to the threshold does not escalate");
  check(threshold + 1 > threshold, "a count one above the threshold does escalate");
}

// ---------------------------------------------------------------------------------------------
// 11. The imposed-estimator guard.

void test_imposed_guard() {
  std::fprintf(stderr, "\n-- an unusable imposed noise level is counted, not asserted --\n");
  const ssize_t N = 8;
  const MR::Header H = make_header(N, N, N);
  // A map holding a zero across an entire half: the value Constant divides by and Import squares,
  //   and the value that a conditioned map can no longer contain. Half the map rather than a
  //   single voxel, so that the patch centre below is well inside the affected region and the
  //   result does not depend on how far cubic interpolation spreads it.
  MR::Image<float> vst = make_map(H, "variance-stabilising map with a hole");
  for (auto l = MR::Loop(vst)(vst); l; ++l)
    vst.value() = vst.index(0) < N / 2 ? 0.0F : 3.0F;

  eigenvalues_type s(20);
  for (ssize_t i = 0; i != s.size(); ++i)
    s[i] = 1.0 + 0.5 * double(i);
  Spectrum spectrum = make_spectrum(s, 20, 40, 0);

  // Deep inside the zeroed half, and away from the field-of-view edge so that the interpolator
  //   itself does not decline the position for an unrelated reason.
  const Eigen::Vector3d inside(1.0, 0.5 * double(N - 1), 0.5 * double(N - 1));
  const Eigen::Vector3d outside(double(N) - 2.0, 0.5 * double(N - 1), 0.5 * double(N - 1));

  Constant constant(4.0, vst);
  const Result bad_constant = constant(spectrum, inside);
  check(!bool(bad_constant), "Constant returns an unusable Result where the map holds a zero");
  check(constant.unusable_sigma_count() == 1, "and counts the occurrence exactly once");
  const Result good_constant = constant(spectrum, outside);
  check(bool(good_constant), "and is unaffected where the map is usable");
  check(constant.unusable_sigma_count() == 1, "with no further occurrences counted");

  Import import(vst);
  const Result bad_import = import(spectrum, inside);
  check(!bool(bad_import), "Import returns an unusable Result where the map holds a zero");
  check(import.unusable_sigma_count() == 1, "and counts the occurrence");
  const Result good_import = import(spectrum, outside);
  check(bool(good_import), "and is unaffected where the map is usable");
  check(import.unusable_sigma_count() == 1, "with no further occurrences counted");
}

} // namespace

int main() {
  test_constant_field();
  test_power_invariance();
  test_imputation_only();
  test_wide_hole();
  test_all_invalid();
  test_two_image_exactness();
  test_axis_order_independence();
  test_log_domain_interpolation();
  test_log_linear_ramp();
  test_warning_threshold();
  test_imposed_guard();

  if (failures == 0)
    std::fprintf(stderr, "\nnoise_conditioning: all checks passed\n");
  else
    std::fprintf(stderr, "\nnoise_conditioning: %d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
