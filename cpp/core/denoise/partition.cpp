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

#include "denoise/partition.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>

namespace MR::Denoise {

namespace {
// Rebuild the partition->volume lists and (if not supplied) the per-partition distinct-group
//   counts from a volume->partition map.
void build_inverse(const std::vector<ssize_t> &v2p,
                   const ssize_t P,
                   const std::vector<ssize_t> &group_per_volume,
                   std::vector<std::vector<ssize_t>> &p2v,
                   std::vector<ssize_t> &groups_present) {
  p2v.assign(P, {});
  for (ssize_t v = 0; v != ssize_t(v2p.size()); ++v) {
    assert(v2p[v] >= 0 && v2p[v] < P);
    p2v[v2p[v]].push_back(v);
  }
  if (groups_present.size() != size_t(P)) {
    groups_present.assign(P, 0);
    if (group_per_volume.empty()) {
      for (ssize_t p = 0; p != P; ++p)
        groups_present[p] = p2v[p].empty() ? 0 : 1;
    } else {
      for (ssize_t p = 0; p != P; ++p) {
        std::vector<ssize_t> g;
        g.reserve(p2v[p].size());
        for (const ssize_t v : p2v[p])
          g.push_back(group_per_volume[v]);
        std::sort(g.begin(), g.end());
        groups_present[p] = ssize_t(std::unique(g.begin(), g.end()) - g.begin());
      }
    }
  }
}
} // namespace

Partitioning::Partitioning(const ssize_t m) : P(1), v2p(std::max<ssize_t>(0, m), 0) {
  std::vector<ssize_t> empty;
  build_inverse(v2p, P, empty, p2v, groups_present_);
}

Partitioning::Partitioning(std::vector<ssize_t> &&volume_to_partition,
                           const ssize_t num_partitions,
                           std::vector<ssize_t> &&groups_present)
    : P(num_partitions), v2p(std::move(volume_to_partition)), groups_present_(std::move(groups_present)) {
  assert(P >= 1);
  std::vector<ssize_t> empty;
  build_inverse(v2p, P, empty, p2v, groups_present_);
}

Partitioning partition_volumes(const ssize_t m,
                               const std::vector<ssize_t> &group_per_volume,
                               const ssize_t num_partitions,
                               Math::RNG &rng) {
  assert(m >= 0);
  assert(group_per_volume.empty() || ssize_t(group_per_volume.size()) == m);
  const ssize_t P = std::max<ssize_t>(1, std::min(num_partitions, std::max<ssize_t>(1, m)));

  std::vector<ssize_t> v2p(m, 0);
  if (P == 1)
    return Partitioning(std::move(v2p), 1, {});

  // Strata = volume index lists per demeaning group (a single stratum when no grouping).
  std::vector<std::vector<ssize_t>> strata;
  if (group_per_volume.empty()) {
    strata.resize(1);
    strata[0].resize(m);
    std::iota(strata[0].begin(), strata[0].end(), ssize_t(0));
  } else {
    const ssize_t num_groups = *std::max_element(group_per_volume.begin(), group_per_volume.end()) + 1;
    strata.resize(num_groups);
    for (ssize_t v = 0; v != m; ++v)
      strata[group_per_volume[v]].push_back(v);
  }

  // Every volume of a demeaning group that lands in a partition has that group's within-partition
  //   mean subtracted from it (see Estimate::compute_partitions). A group represented in a
  //   partition by a single volume is therefore not demeanable: subtracting the mean of one volume
  //   from itself yields a row that is identically zero, which is neither the signal nor the noise
  //   the decomposition is asked to separate. The spread below keeps every populated
  //   (group, partition) cell at two or more volumes whenever the group has at least two volumes to
  //   give; a group of exactly one volume can satisfy that under no assignment, so refuse here
  //   rather than emit a degenerate Casorati matrix. As with the unpopulated-partition case below,
  //   this depends only on the group sizes and P, so it fires identically for every patch.
  std::vector<size_t> singleton_groups;
  for (size_t gi = 0; gi != strata.size(); ++gi) {
    if (strata[gi].size() == 1)
      singleton_groups.push_back(gi);
  }
  if (!singleton_groups.empty()) {
    std::string list;
    for (const size_t gi : singleton_groups)
      list += (list.empty() ? "" : ", ") + str(gi);
    throw Exception("Cannot split " + str(m) + " volumes into " + str(P) +
                    " partitions: demeaning group(s) [" + list +
                    "] contain only one volume each, so the partition holding such a volume would "
                    "hold fewer than two volumes of that group and could not be demeaned. "
                    "Disable volume partitioning, use a coarser -demean grouping, or use -demean none");
  }

  // Process larger groups first so the abundant groups establish a balanced base load
  //   into which the smaller (concentrated) groups are then slotted.
  std::vector<size_t> order(strata.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::sort(order.begin(), order.end(),
            [&](const size_t a, const size_t b) { return strata[a].size() > strata[b].size(); });

  std::vector<ssize_t> load(P, 0);                       // running per-partition volume count
  std::vector<ssize_t> groups_present(P, 0);             // distinct groups assigned per partition

  for (const size_t gi : order) {
    std::vector<ssize_t> &vols = strata[gi];
    const ssize_t s = ssize_t(vols.size());
    if (s == 0)
      continue;
    // Partial Fisher-Yates shuffle of this group's volumes (matching set_temporal_subsample).
    for (ssize_t i = 0; i + 1 < s; ++i) {
      std::uniform_int_distribution<ssize_t> dist(i, s - 1);
      std::swap(vols[i], vols[dist(rng)]);
    }
    // Spread across as many partitions as possible while keeping >=2 volumes per used
    //   partition; a singleton group (s==1) unavoidably lands in one partition.
    ssize_t k = (s == 1) ? 1 : std::min(P, s / 2);
    k = std::max<ssize_t>(1, k);
    // Choose the k least-loaded partitions to balance total sizes.
    std::vector<ssize_t> part_order(P);
    std::iota(part_order.begin(), part_order.end(), ssize_t(0));
    std::sort(part_order.begin(), part_order.end(),
              [&](const ssize_t a, const ssize_t b) { return load[a] < load[b]; });
    part_order.resize(k);
    // Distribute s volumes across the k chosen partitions as evenly as possible (counts differ
    //   by at most one; the extras go to the least-loaded). With k = s/2 every count is >= 2.
    ssize_t idx = 0;
    for (ssize_t j = 0; j != k; ++j) {
      const ssize_t count = s / k + (j < (s % k) ? 1 : 0);
      const ssize_t p = part_order[j];
      for (ssize_t c = 0; c != count; ++c, ++idx)
        v2p[vols[idx]] = p;
      load[p] += count;
      ++groups_present[p];
    }
    assert(idx == s);
  }

  // The invariant the singleton-group rejection above and the per-stratum spread together
  //   establish: no populated (group, partition) cell holds fewer than two volumes, so every
  //   group mean Estimate subtracts is formed from at least two volumes.
#ifndef NDEBUG
  {
    std::vector<std::vector<ssize_t>> cell_counts(P, std::vector<ssize_t>(strata.size(), 0));
    for (ssize_t v = 0; v != m; ++v)
      ++cell_counts[v2p[v]][group_per_volume.empty() ? 0 : group_per_volume[v]];
    for (ssize_t p = 0; p != P; ++p) {
      for (size_t gi = 0; gi != strata.size(); ++gi)
        assert(cell_counts[p][gi] == 0 || cell_counts[p][gi] >= 2);
    }
  }
#endif

  // A partition holding no volumes has no Casorati matrix to decompose, and an empty block reaching
  //   Eigen aborts. The per-stratum spread above uses at most min(P, s/2) partitions per stratum, so
  //   with several small strata (or with P close to m) partitions can be left unused however the
  //   caller clamps P. Compact the plan so that the returned Partitioning never contains one.
  ssize_t populated = 0;
  std::vector<ssize_t> remap(P, -1);
  for (ssize_t p = 0; p != P; ++p)
    if (load[p] > 0)
      remap[p] = populated++;
  if (populated != P) {
    assert(populated >= 1);
    for (ssize_t &p : v2p)
      p = remap[p];
    std::vector<ssize_t> compacted_groups(populated, 0);
    for (ssize_t p = 0; p != P; ++p)
      if (remap[p] >= 0)
        compacted_groups[remap[p]] = groups_present[p];
    return Partitioning(std::move(v2p), populated, std::move(compacted_groups));
  }

  return Partitioning(std::move(v2p), P, std::move(groups_present));
}

} // namespace MR::Denoise
