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

#include "denoise/precondition/noise_model/noise_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <utility>

#include "mrtrix.h"

#include "denoise/precondition/noise_model/gaussian.h"
#include "denoise/precondition/noise_model/identity.h"
#include "denoise/precondition/noise_model/noncentralchi.h"
#include "denoise/precondition/noise_model/rician.h"

namespace MR::Denoise::Precondition::NoiseModel {

default_type clamp_dof(const default_type dof) { return std::max(dof, NonCentralChi::min_dof); }

std::shared_ptr<Base> make(const distribution_t distribution, //
                           const default_type dof,            //
                           const vst_method_t vst_method) {   //
  // -vst_method none / linear select the transform directly, independent of the
  //   underlying noise distribution:
  //   - none   : identity (no variance stabilisation);
  //   - linear : the simple linear scaling u = m / sigma (the Gaussian model),
  //              exact for additive Gaussian (complex / phase-demodulated) noise.
  switch (vst_method) {
  case vst_method_t::NONE:
    return std::make_shared<Identity>();
  case vst_method_t::LINEAR:
    return std::make_shared<Gaussian>();
  default:
    break;
  }
  switch (distribution) {
  case distribution_t::GAUSSIAN:
    return std::make_shared<Gaussian>();
  case distribution_t::RICIAN:
    return std::make_shared<Rician>(vst_method);
  case distribution_t::NONCENTRALCHI:
    return std::make_shared<NonCentralChi>(dof, vst_method);
  }
  assert(false);
  return nullptr;
}

Set::Set(std::shared_ptr<Base> model) : models{std::move(model)} { assert(models[0] != nullptr); }

Set::Set(std::vector<std::shared_ptr<Base>> models, const size_t slice_axis)
    : models(std::move(models)), axis(slice_axis) {
  assert(!this->models.empty());
  assert(slice_axis < 3);
}

size_t Set::num_distinct_models() const {
  std::vector<const Base *> distinct;
  distinct.reserve(models.size());
  for (const auto &model : models)
    distinct.push_back(model.get());
  std::sort(distinct.begin(), distinct.end());
  return size_t(std::unique(distinct.begin(), distinct.end()) - distinct.begin());
}

bool Set::is_linear() const {
  assert(!models.empty());
  return models[0]->is_linear();
}

std::string Set::description() const {
  assert(!models.empty());
  if (models.size() == 1)
    return models[0]->description();
  // For a slice-wise set the individual descriptions would be sixty near-identical strings, so
  //   what is reported instead is the family, the range of degrees of freedom it spans, and how
  //   many transforms were actually built (which the quantisation, not the slice count, decides).
  default_type lowest = models[0]->dof();
  default_type highest = models[0]->dof();
  for (const auto &model : models) {
    lowest = std::min(lowest, model->dof());
    highest = std::max(highest, model->dof());
  }
  return models[0]->description() + ", per slice (" + str(lowest, 4) + " to " + str(highest, 4) +
         " degrees of freedom; " + str(num_distinct_models()) + " distinct transforms)";
}

Set make_slicewise(const std::vector<default_type> &dof_per_slice, //
                   const size_t slice_axis,                        //
                   const vst_method_t vst_method,                  //
                   const default_type quantum) {                   //
  assert(!dof_per_slice.empty());
  assert(quantum > 0.0);
  std::map<default_type, std::shared_ptr<Base>> cache;
  std::vector<std::shared_ptr<Base>> models;
  models.reserve(dof_per_slice.size());
  for (const default_type requested : dof_per_slice) {
    const default_type quantised = clamp_dof(std::round(clamp_dof(requested) / quantum) * quantum);
    auto it = cache.find(quantised);
    if (it == cache.end()) {
      // The Rician label is reserved for exactly two degrees of freedom, as it is everywhere else.
      const distribution_t distribution =
          quantised == default_type(2) ? distribution_t::RICIAN : distribution_t::NONCENTRALCHI;
      it = cache.emplace(quantised, make(distribution, quantised, vst_method)).first;
    }
    models.push_back(it->second);
  }
  return Set(std::move(models), slice_axis);
}

} // namespace MR::Denoise::Precondition::NoiseModel
