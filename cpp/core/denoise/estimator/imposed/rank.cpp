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

#include "denoise/estimator/imposed/rank.h"

#include <vector>

#include "denoise/estimator/pooling.h"
#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise::Estimator {

Result Rank::operator()(const Spectrum &spectrum, const Eigen::Vector3d & /*unused*/) const {
  const ssize_t rz = spectrum.rz();
  // The imposed rank counts the assumed-zero singular values "rz" that preconditioning ("rp"
  //   regressed means) forces to zero, as the arithmetic below (rank - rz) requires. The
  //   maximal attainable rank is therefore rz + rnz == min(m, n) == the number of components,
  //   NOT rnz: a patch whose demeaning removed rz directions still has min(m, n) components in
  //   total, rz of which are identically zero. Comparing against rnz alone understates the
  //   maximum by rz, which both mis-classified attainable ranks as no-ops and let unattainable
  //   ones fall through to a default-constructed Result.
  const ssize_t max_rank = spectrum.r();
  if (rank > max_rank)
    throw Exception("Imposed signal rank of " + str(rank) + " cannot be attained: this patch has " + str(spectrum.m()) +
                    " volumes and " + str(spectrum.n()) + " voxels, giving at most " + str(max_rank) +
                    " components (of which " + str(rz) +
                    " are forced to zero by the demeaning of the preconditioner). "
                    "Reduce -fixed_rank to " +
                    str(max_rank) + " or fewer");
  if (rank < rz)
    throw Exception("Imposed signal rank of " + str(rank) + " is below the " + str(rz) +
                    " component(s) that the preconditioner's demeaning has already forced to zero "
                    "in this patch, and which necessarily count towards the signal rank. "
                    "Increase -fixed_rank to at least " +
                    str(rz));
  Result result;
  if (rank == max_rank) {
    // Every component is signal, so there is no noise boundary to report. sigma2 == 0 is the
    //   established encoding of "no usable threshold" (see Result::operator bool), on which
    //   Estimate and Recon take their documented conservative path of copying the patch through
    //   unchanged -- which is exactly the no-op this case calls for. The Result is explicitly
    //   populated rather than left default-constructed.
    result.cutoff_p = 0;
    result.sigma2 = 0.0;
    result.set_boundaries(spectrum.edge(), 0.0);
    return result;
  }
  result.cutoff_p = spectrum.r() - (rank - rz);
  const ssize_t noise_nonzero = result.cutoff_p - rz;
  // NOTE: the denominator carries one component more than the numerator sums. Preserved as-is,
  //   and mirrored in the partitioned form below, rather than silently altered here.
  result.sigma2 = spectrum.raw_sum(rz, noise_nonzero) / spectrum.dof(noise_nonzero + 1);
  // An imposed rank cuts wherever it is told to; the empirical boundary that produces bears no
  //   necessary relation to the lambda+ that the resulting noise level implies, which is precisely
  //   why the two are recorded separately.
  result.set_boundaries(spectrum.edge(), spectrum.max_noise_standardized(result.cutoff_p));
  return result;
}

Result Rank::operator()(const std::vector<Spectrum> &spectra, //
                        const Eigen::Vector3d & /*unused*/) const {
  ssize_t total_rz = 0;
  for (const auto &spectrum : spectra)
    total_rz += spectrum.rz();
  const ssize_t total_nonzero = total_rnz(spectra);
  Result result;
  // Number of genuine (nonzero) signal components to retain across the patch.
  const ssize_t signal_nonzero = rank - total_rz;
  // As in the single-PCA form: an unattainable request is a fault to report, not a Result to
  //   leave default-constructed and pass on to callers that would read it as "no threshold".
  if (total_nonzero == 0)
    throw Exception("Imposed signal rank of " + str(rank) +
                    " cannot be attained: preconditioning has forced every component of this "
                    "partitioned patch to zero, leaving no components to classify");
  if (signal_nonzero < 0)
    throw Exception("Imposed signal rank of " + str(rank) + " is below the " + str(total_rz) +
                    " component(s) that the preconditioner's demeaning has already forced to zero "
                    "across this patch's partitions, and which necessarily count towards the "
                    "signal rank. Increase -fixed_rank to at least " +
                    str(total_rz));
  if (signal_nonzero > total_nonzero)
    throw Exception("Imposed signal rank of " + str(rank) +
                    " cannot be attained: this patch's partitions "
                    "provide at most " +
                    str(total_rz + total_nonzero) + " components (of which " + str(total_rz) +
                    " are forced to zero by the demeaning of the preconditioner). "
                    "Reduce -fixed_rank to " +
                    str(total_rz + total_nonzero) + " or fewer");
  if (signal_nonzero == total_nonzero) {
    // Keep all nonzero components as signal.
    result.cutoff_p = 0;
    result.sigma2 = 0.0;
    result.set_boundaries(mean_edge(spectra), 0.0);
    result.cutoff_p_partition.assign(spectra.size(), 0);
    result.lamplus_partition.assign(spectra.size(), 0.0);
    return result;
  }
  // Ascending in standardised value, length total_nonzero.
  const std::vector<PooledEigenvalue> pooled = pool_standardized(spectra);
  const ssize_t noise_nonzero = total_nonzero - signal_nonzero; // > 0 here
  // The most significant component retained as noise defines the boundary.
  const double zstar = pooled[noise_nonzero - 1].z;
  const NoiseStats stats = apply_threshold_standardized(spectra, zstar, result);
  // As in the single-PCA form, the denominator carries one component more than the numerator
  //   sums; that extra component is charged the mean degrees of freedom per noise component, so
  //   that a single partition reproduces qnz * (k + 1) exactly.
  result.sigma2 = stats.sum / (stats.dof * double(stats.count + 1) / double(stats.count));
  result.set_boundaries(mean_edge(spectra), stats.z_max);
  return result;
}

} // namespace MR::Denoise::Estimator
