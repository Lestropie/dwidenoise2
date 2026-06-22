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

#include "math/rng.h"
#include "types.h"

namespace MR::Denoise {

// A partition of m "volumes" (the rows of a PCA Casorati matrix) into P disjoint partitions,
//   used to accelerate large-series PCA: each partition is decomposed independently and the
//   eigenspectra pooled. The assignment is balanced across demeaning groups (b-value shells or
//   5th-dimension volume groups) so each partition keeps a comparable Casorati aspect ratio and
//   group composition, and so that no (group, partition) cell is left with exactly one volume
//   (which would become identically zero after group-mean subtraction; see partition_volumes).
class Partitioning {
public:
  // Trivial single partition over m volumes (the P==1, no-partitioning case).
  explicit Partitioning(const ssize_t m = 0);
  // Construct from a complete volume->partition map (each entry in [0, num_partitions)).
  //   groups_present[p] is the number of distinct demeaning groups represented in partition p
  //   (used downstream to derive that partition's preconditioner/null rank); pass an empty
  //   vector when grouping is not applicable, in which case it is derived as 1 per non-empty
  //   partition.
  Partitioning(std::vector<ssize_t> &&volume_to_partition,
               const ssize_t num_partitions,
               std::vector<ssize_t> &&groups_present);

  ssize_t num_partitions() const { return P; }
  ssize_t num_volumes() const { return ssize_t(v2p.size()); }

  // Partition index of a given volume row.
  ssize_t partition_of(const ssize_t volume) const { return v2p[volume]; }
  // Ascending list of volume rows belonging to partition p.
  const std::vector<ssize_t> &volumes(const ssize_t p) const { return p2v[p]; }
  ssize_t size(const ssize_t p) const { return ssize_t(p2v[p].size()); }
  // Number of distinct demeaning groups represented in partition p.
  ssize_t groups_in_partition(const ssize_t p) const { return groups_present_[p]; }

private:
  ssize_t P;
  std::vector<ssize_t> v2p;              // length m: volume -> partition
  std::vector<std::vector<ssize_t>> p2v; // length P: partition -> sorted volume list
  std::vector<ssize_t> groups_present_;  // length P: distinct demeaning groups per partition
};

// Build a balanced partitioning of m volumes into num_partitions partitions.
//   group_per_volume, if non-empty, must have length m and assign each volume a demeaning-group
//   label in [0, num_groups); an empty vector denotes a single group spanning all volumes.
//   Each group's volumes are randomly (rng) spread across as many partitions as possible while
//   keeping at least two volumes of that group in every partition it touches (groups too small
//   for that are concentrated into fewer partitions), avoiding singleton (group, partition)
//   cells. Partition total sizes are kept as balanced as the group constraints allow.
//   num_partitions is clamped to [1, max(1, m)].
Partitioning partition_volumes(const ssize_t m,
                               const std::vector<ssize_t> &group_per_volume,
                               const ssize_t num_partitions,
                               Math::RNG &rng);

} // namespace MR::Denoise
