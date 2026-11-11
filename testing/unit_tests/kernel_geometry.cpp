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

// Geometry of the patches returned by the sliding-window kernels: the physical distances recorded
//   for each member voxel.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "algo/loop.h"
#include "datatype.h"
#include "denoise/kernel/cuboid.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/voxel.h"
#include "header.h"
#include "image.h"
#include "math/math.h"
#include "mrtrix.h"
#include "types.h"

using MR::default_type;
using MR::Denoise::Kernel::Cuboid;
using MR::Denoise::Kernel::Data;
using MR::Denoise::Kernel::Voxel;

namespace {

int failures = 0;

void check(const bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

void check_close(const double value, const double target, const double tol, const std::string &message) {
  if (!(std::fabs(value - target) <= tol)) {
    std::fprintf(stderr, "FAIL: %s (got %.10g, expected %.10g)\n", message.c_str(), value, target);
    ++failures;
  }
}

const std::array<ssize_t, 3> no_subsampling{1, 1, 1};

// A 7x7x7 grid with a different voxel spacing along each axis, identity orientation.
MR::Header make_header() {
  MR::Header H;
  H.ndim() = 3;
  const std::array<default_type, 3> spacing{1.5, 2.0, 3.0};
  for (size_t axis = 0; axis != 3; ++axis) {
    H.size(axis) = 7;
    H.spacing(axis) = spacing[axis];
    H.stride(axis) = ssize_t(axis) + 1;
  }
  H.transform().setIdentity();
  H.datatype() = MR::DataType::Bit;
  return H;
}

MR::Image<bool> make_mask(const MR::Header &H) {
  MR::Image<bool> mask = MR::Image<bool>::scratch(H, "kernel geometry test mask");
  for (auto l = MR::Loop(mask)(mask); l; ++l)
    mask.value() = true;
  return mask;
}

const Voxel::index_type centre_pos(3, 3, 3);

// Every member of a cuboid patch records its squared distance from the patch centre in mm^2, on a
//   grid whose spacing differs per axis; the patch radius is the distance to its corner.
void test_cuboid_distances() {
  const MR::Header H = make_header();
  MR::Image<bool> mask = make_mask(H);
  Cuboid kernel(H, no_subsampling, {3, 3, 3});
  kernel.set_mask(mask);
  const Data patch = kernel(centre_pos);
  check(patch.voxels.size() == 27, "a 3x3x3 cuboid with nothing masked holds 27 voxels");
  default_type worst = 0.0;
  for (const auto &v : patch.voxels) {
    default_type expected = 0.0;
    for (size_t axis = 0; axis != 3; ++axis)
      expected += MR::Math::pow2(default_type(v.index[axis] - centre_pos[axis]) * H.spacing(axis));
    worst = std::max(worst, std::fabs(v.sq_distance - expected));
  }
  check_close(worst, 0.0, 1e-9, "each cuboid member records its squared distance in mm^2");
  check_close(patch.max_distance,
              std::sqrt(MR::Math::pow2(1.5) + MR::Math::pow2(2.0) + MR::Math::pow2(3.0)),
              1e-9,
              "the cuboid patch radius is the distance to its corner in mm");
}

} // namespace

int main() {
  test_cuboid_distances();

  if (failures == 0)
    std::fprintf(stderr, "\nkernel_geometry: all checks passed\n");
  else
    std::fprintf(stderr, "\nkernel_geometry: %d check(s) failed\n", failures);
  return failures == 0 ? 0 : 1;
}
