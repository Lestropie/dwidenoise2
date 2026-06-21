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

#include "denoise/spatial_subsample.h"

namespace MR::Denoise {

SpatialSubsample::SpatialSubsample(const Header &in, const std::array<ssize_t, 3> &factors)
    : H_in(make_input_header(in)),
      factors(factors),
      size({(in.size(0) + factors[0] - 1) / factors[0],
            (in.size(1) + factors[1] - 1) / factors[1],
            (in.size(2) + factors[2] - 1) / factors[2]}),
      origin({(in.size(0) - factors[0] * (size[0] - 1) - 1) / 2,
              (in.size(1) - factors[1] * (size[1] - 1) - 1) / 2,
              (in.size(2) - factors[2] * (size[2] - 1) - 1) / 2}),
      H_ss(make_subsample_header()) {}

bool SpatialSubsample::process(const Kernel::Voxel::index_type &pos) const {
  for (ssize_t axis = 0; axis != 3; ++axis) {
    if (pos[axis] % factors[axis] != origin[axis])
      return false;
  }
  return true;
}

std::array<ssize_t, 3> SpatialSubsample::in2ss(const Kernel::Voxel::index_type &pos) const {
  // Do not attempt to map an unprocessed voxel to a voxel index in subsampled space
  assert(process(pos));
  assert(!is_out_of_bounds(H_in, pos, 0, 3));
  return std::array<ssize_t, 3>({(pos[0] - origin[0]) / factors[0],   //
                                 (pos[1] - origin[1]) / factors[1],   //
                                 (pos[2] - origin[2]) / factors[2]}); //
}

std::array<ssize_t, 3> SpatialSubsample::ss2in(const Kernel::Voxel::index_type &pos) const {
  assert(!is_out_of_bounds(H_ss, pos));
  return std::array<ssize_t, 3>({pos[0] * factors[0] + origin[0],   //
                                 pos[1] * factors[1] + origin[1],   //
                                 pos[2] * factors[2] + origin[2]}); //
}

Header SpatialSubsample::make_input_header(const Header &H_in) const {
  Header H(H_in);
  H.ndim() = 3;
  H.reset_intensity_scaling();
  H.datatype() = DataType::Float32;
  H.datatype().set_byte_order_native();
  return H;
}

Header SpatialSubsample::make_subsample_header() const {
  Header H(H_in);
  H.ndim() = 3;
  H.reset_intensity_scaling();
  H.datatype() = DataType::Float32;
  H.datatype().set_byte_order_native();
  std::array<double, 3> halfvoxel_offsets;
  for (ssize_t axis = 0; axis != 3; ++axis) {
    H.size(axis) = size[axis];
    H.spacing(axis) *= factors[axis];
    halfvoxel_offsets[axis] = factors[axis] & 1 ? 0.0 : 0.5;
  }
  H.transform().translation() =
      H_in.transform() * Eigen::Matrix<default_type, 3, 1>({(origin[0] + halfvoxel_offsets[0]) * H_in.spacing(0),   //
                                                            (origin[1] + halfvoxel_offsets[1]) * H_in.spacing(1),   //
                                                            (origin[2] + halfvoxel_offsets[2]) * H_in.spacing(2)}); //
  return H;
}

} // namespace MR::Denoise
