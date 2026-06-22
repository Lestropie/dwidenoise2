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

#include <cassert>
#include <vector>

#include "denoise/denoise.h"
#include "denoise/estimator/result.h"
#include "exception.h"

namespace MR::Denoise::Estimator {

class Base {
public:
  Base() = default;
  Base(const Base &) = delete;
  virtual ~Base() = default;
  virtual void update_vst_image(Image<float> &) {}
  // Single-PCA (non-partitioned) noise level estimation.
  // m = Number of image volumes;
  // n = Number of voxels in patch;
  // rp = Preconditioner rank = number of means regressed from the data;
  // pos = realspace position of the centre of the patch
  virtual Result operator()(const Eigen::VectorBlock<eigenvalues_type> eigenvalues, //
                            const ssize_t m,                                        //
                            const ssize_t n,                                        //
                            const ssize_t rp,                                       //
                            const Eigen::Vector3d &pos) const = 0;                  //

  // Partitioned noise level estimation: the patch is split into P partitions, with partition p
  //   contributing an independent ascending eigenspectrum eigenvalues[p] (length min(m[p],n[p]))
  //   from a decomposition of dimensions (m[p], n[p]) and preconditioner rank rp[p]. The pooled
  //   spectrum yields a single noise level (Result::sigma2) shared by the patch, with a
  //   per-partition signal/noise cutoff (Result::cutoff_p_partition). Estimators that support
  //   partitioning override this; the default rejects it (callers must gate on
  //   supports_partitioning()).
  virtual Result operator()(const std::vector<eigenvalues_type> &eigenvalues, //
                            const std::vector<ssize_t> &m,                     //
                            const std::vector<ssize_t> &n,                     //
                            const std::vector<ssize_t> &rp,                    //
                            const Eigen::Vector3d &pos) const {                //
    assert(false);
    throw Exception("Selected noise level estimator does not support volume partitioning");
  }

  // Whether operator() for multiple partitions is implemented for this estimator.
  virtual bool supports_partitioning() const { return false; }
};

} // namespace MR::Denoise::Estimator
