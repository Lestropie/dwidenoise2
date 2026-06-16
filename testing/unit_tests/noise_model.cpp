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

// Self-contained unit test for the denoise NoiseModel / VST library.
//
// It depends only on the public NoiseModel interface plus the standard library,
//   so it can be compiled and registered with CTest directly (a plain executable
//   whose non-zero exit status indicates failure). If wired into the MRtrix3
//   unit-test harness, replace main() with the project's test scaffold; the
//   check_* helpers and tolerances are otherwise unchanged.
//
// Validates (cf. implementation plan section 10):
//   - normalised floor bias mean(0) = sqrt(pi/2) for Rician;
//   - Var[Phi(m)|nu] ~ 1 across SNR (homoscedasticity);
//   - forward / algebraic-inverse round-trip;
//   - exact-unbiased (foi) debias removes the floor where nu ~ 0 and recovers nu;
//   - method-of-moments / Koay inverses clamp below the floor;
//   - near-Gaussianity of the stabilised residual at moderate-to-high SNR;
//   - Gaussian (complex) model is exactly the prior linear transform.

#include "denoise/noise_model/noise_model.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace MR::Denoise::NoiseModel;

namespace {

int failures = 0;

void check(const bool condition, const char *const message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

void check_near(const double value, const double target, const double tol, const char *const message) {
  if (!(std::fabs(value - target) <= tol)) {
    std::fprintf(stderr, "FAIL: %s (got %.6f, expected %.6f +/- %.6f)\n", message, value, target, tol);
    ++failures;
  }
}

// Draw m/sigma samples from a non-central chi with 2N DOF and SNR a (sigma = 1).
std::vector<double> samples(const double a, const int N, const int n, std::mt19937 &rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> out(n);
  for (int s = 0; s != n; ++s) {
    double acc = 0.0;
    for (int c = 0; c != N; ++c) {
      const double re = nd(rng) + (c == 0 ? a : 0.0);
      const double im = nd(rng);
      acc += re * re + im * im;
    }
    out[s] = std::sqrt(acc);
  }
  return out;
}

struct Moments {
  double mean;
  double variance;
  double skewness;
};

Moments moments(const std::vector<double> &v) {
  Moments m{0.0, 0.0, 0.0};
  for (const double x : v)
    m.mean += x;
  m.mean /= double(v.size());
  double m3 = 0.0;
  for (const double x : v) {
    const double d = x - m.mean;
    m.variance += d * d;
    m3 += d * d * d;
  }
  m.variance /= double(v.size());
  m3 /= double(v.size());
  m.skewness = m3 / std::pow(m.variance, 1.5);
  return m;
}

void test_magnitude(const int N, std::mt19937 &rng) {
  const distribution_t dist = (N == 1) ? distribution_t::RICIAN : distribution_t::NONCENTRALCHI;
  auto foi = make(dist, N, vst_method_t::FOI);
  auto mom = make(dist, N, vst_method_t::MOM);
  auto koay = make(dist, N, vst_method_t::KOAY);

  check(foi->num_channels() == N, "num_channels");
  check(foi->dof() == 2 * N, "dof");

  // Floor bias and second moment (closed form vs known values).
  check_near(foi->variance(0.0, 1.0) + foi->mean(0.0, 1.0) * foi->mean(0.0, 1.0), double(2 * N), 1e-6,
             "E[m^2|0] = 2N");
  if (N == 1) {
    const double pi = std::acos(-1.0);
    check_near(foi->mean(0.0, 1.0), std::sqrt(pi / 2.0), 1e-6, "Rician floor mean = sqrt(pi/2)");
  }

  // Forward transform monotonic increasing.
  double prev = foi->stabilise(0.0, 1.0);
  bool monotonic = true;
  for (double m = 0.05; m <= 40.0; m += 0.05) {
    const double u = foi->stabilise(m, 1.0);
    if (u <= prev)
      monotonic = false;
    prev = u;
  }
  check(monotonic, "forward VST monotonic");

  for (const double a : {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0}) {
    auto s = samples(a, N, 300000, rng);
    std::vector<double> z(s.size());
    for (size_t i = 0; i != s.size(); ++i)
      z[i] = foi->stabilise(s[i], 1.0);
    const Moments mz = moments(z);
    // Homoscedasticity: loose near the floor, tight at high SNR.
    if (a >= 4.0)
      check_near(mz.variance, 1.0, 0.05, "Var[Phi] ~ 1 (high SNR)");
    else
      check(mz.variance > 0.6 && mz.variance < 1.3, "Var[Phi] within [0.6,1.3] (low SNR)");
    // Near-Gaussianity of the stabilised residual at moderate-to-high SNR.
    if (a >= 2.0)
      check(std::fabs(mz.skewness) < 0.25, "stabilised residual near-symmetric (moderate-high SNR)");
  }

  // Debias: recover nu from the stabilised-domain group mean.
  for (const double a : {0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0}) {
    auto s = samples(a, N, 600000, rng);
    double ubar = 0.0;
    for (const double x : s)
      ubar += foi->stabilise(x, 1.0);
    ubar /= double(s.size());
    const double nu_foi = foi->inverse_unbiased(ubar, 1.0);
    if (a == 0.0) {
      check(nu_foi < 0.15, "foi removes the floor (nu ~ 0 at a = 0)");
      check_near(mom->inverse_unbiased(ubar, 1.0), 0.0, 1e-9, "mom clamps at floor");
      check_near(koay->inverse_unbiased(ubar, 1.0), 0.0, 1e-9, "koay clamps at floor");
    } else if (a >= 1.0) {
      check_near(nu_foi, a, 0.05 * a + 0.02, "foi exact-unbiased recovery of nu");
      check(foi->jacobian(ubar, 1.0) > 0.0, "Jacobian positive");
    }
  }

  // Forward / algebraic-inverse round-trip (sigma = 3 to exercise scaling).
  double rt_err = 0.0;
  for (const double m : {0.5, 1.0, 3.0, 9.0, 30.0}) {
    const double rt = foi->inverse_algebraic(foi->stabilise(m, 3.0), 3.0);
    rt_err = std::max(rt_err, std::fabs(rt - m));
  }
  check(rt_err < 1e-3, "forward/inverse_algebraic round-trip");
}

void test_gaussian() {
  auto g = make(distribution_t::GAUSSIAN, 1, vst_method_t::FOI);
  // Must reproduce the prior linear transform exactly.
  check_near(g->stabilise(7.0, 2.0), 3.5, 1e-12, "Gaussian stabilise = m/sigma");
  check_near(g->inverse_algebraic(3.5, 2.0), 7.0, 1e-12, "Gaussian inverse_algebraic = sigma*u");
  check_near(g->inverse_unbiased(3.5, 2.0), 7.0, 1e-12, "Gaussian inverse_unbiased = sigma*u (no bias)");
  check_near(g->jacobian(123.0, 2.0), 2.0, 1e-12, "Gaussian Jacobian = sigma");
  check_near(g->mean(5.0, 2.0), 5.0, 1e-12, "Gaussian mean = nu");
  check_near(g->variance(5.0, 2.0), 4.0, 1e-12, "Gaussian variance = sigma^2");
}

} // namespace

int main() {
  std::mt19937 rng(1);
  test_gaussian();
  for (const int N : {1, 2, 4})
    test_magnitude(N, rng);
  if (failures == 0)
    std::printf("noise_model: all checks passed\n");
  return failures == 0 ? 0 : 1;
}
