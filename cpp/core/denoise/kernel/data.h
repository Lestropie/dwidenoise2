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

#include <vector>

#include <Eigen/Dense>

#include "denoise/kernel/voxel.h"
#include "types.h"

namespace MR::Denoise::Kernel {

class Data {
public:
  Data(const Voxel::index_type seed_voxel,
       const Eigen::Vector3d &pos,
       const ssize_t i)
      : seed_voxel(seed_voxel),
        centre_realspace(pos),
        centre_index(i),
        max_distance(-std::numeric_limits<default_type>::infinity()),
        centre_noise(std::numeric_limits<double>::signaling_NaN()) {}
  Data(const ssize_t i)
      : seed_voxel({-1, -1, -1}),
        centre_realspace(Eigen::Vector3d::Constant(std::numeric_limits<double>::signaling_NaN())),
        centre_index(i),
        max_distance(-std::numeric_limits<default_type>::infinity()),
        centre_noise(std::numeric_limits<double>::signaling_NaN()) {}
  Data()
      : seed_voxel({-1, -1, -1}),
        centre_realspace(Eigen::Vector3d::Constant(std::numeric_limits<double>::signaling_NaN())),
        centre_index(-1),
        max_distance(-std::numeric_limits<default_type>::infinity()),
        centre_noise(std::numeric_limits<double>::signaling_NaN()) {}
  ssize_t num_voxels() const { return voxels.size(); }
  Voxel::index_type seed_voxel;
  Eigen::Vector3d centre_realspace;
  std::vector<Voxel> voxels;
  ssize_t centre_index;
  default_type max_distance;
  double centre_noise;
};

} // namespace MR::Denoise::Kernel
