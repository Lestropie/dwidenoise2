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

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "app.h"
#include "denoise/kernel/voxel.h"
#include "denoise/noise_model/noise_model.h"
#include "filter/demodulate.h"
#include "header.h"
#include "image.h"
#include "interp/cubic.h"
#include "transform.h"
#include "types.h"

namespace MR::Denoise {

extern const char *const demodulation_description;

const std::vector<std::string> demodulation_choices({"none", "linear", "nonlinear"});
enum class demodulation_t { NONE, LINEAR, NONLINEAR };

const std::vector<std::string> demean_choices = {"none", "volume_groups", "shells", "all"};
enum class demean_type { NONE, VOLUME_GROUPS, SHELLS, ALL };

App::OptionGroup precondition_options(const bool include_output);

class Demodulation {
public:
  Demodulation(demodulation_t mode) : mode(mode) {}
  Demodulation() : mode(demodulation_t::NONE) {}
  explicit operator bool() const { return mode != demodulation_t::NONE; }
  bool operator!() const { return mode == demodulation_t::NONE; }
  demodulation_t mode;
  std::vector<size_t> axes;
};
Demodulation select_demodulation(const Header &);

demean_type select_demean(const Header &);

// Need to SFINAE define the demodulator type,
//   so that it does not attempt to compile the demodulation filter for non-complex types
class DummyDemodulator {
public:
  template <class ImageType> DummyDemodulator(ImageType &, const std::vector<size_t> &, const bool) {}
  template <class InputImageType, class OutputImageType>
  void operator()(InputImageType &, OutputImageType &, const bool) {
    assert(false);
  }
  Image<cfloat> operator()() { return Image<cfloat>(); }
};
template <typename T> struct DemodulatorSelector {
  using type = DummyDemodulator;
};
template <typename T> struct DemodulatorSelector<std::complex<T>> {
  using type = Filter::Demodulate;
};

template <typename T> class Precondition {
public:
  Precondition(Image<T> &image,
               const Demodulation &demodulation,
               const demean_type demean,
               Image<float> &vst_noise,
               std::shared_ptr<NoiseModel::Base> noise_model);
  Precondition(Precondition &) = default;
  // Refresh both variance-stabilising-transform parameters together:
  //   the noise level map (VST scale) and, derived from it, the
  //   stabilised-domain per-group means (VST offset); see vst_plan.md section 3.2.
  // A pass over the input is required to recompute the stabilised-domain means.
  void update_vst_parameters(Image<float> new_vst_noise, Image<T> input) {
    vst_noise_image = new_vst_noise;
    compute_means(input);
  }
  void operator()(Image<T> input, Image<T> output, const bool inverse = false) const;
  const Header &header() const { return H_out; }

  ssize_t null_rank() const {
    if (!vst_mean_image.valid())
      return 0;
    if (vst_mean_image.ndim() == 3)
      return 1;
    return vst_mean_image.size(3);
  }

  bool noop() const {
    return (num_volume_groups == 1 && !phase_image.valid() && !vst_mean_image.valid() && !vst_noise_image.valid());
  }

private:
  const Header H_in;
  Header H_out;
  // For serialisation of >4D images
  ssize_t num_volume_groups;
  Image<uint32_t> serialise_image;
  // Noise distribution governing the variance-stabilising transform (VST);
  //   scalar configuration shared by forward stabilisation and inverse mapping.
  std::shared_ptr<NoiseModel::Base> noise_model;
  // First step: Phase demodulation
  Image<cfloat> phase_image;
  // Second step (forward): variance-stabilising transform.
  //   The noise level map is the VST scale / dispersion parameter.
  Image<float> vst_noise_image;
  // Third step (forward): demeaning, performed in the stabilised domain.
  //   The stored means are the per-group means of the stabilised data
  //   (the VST offset parameter; sigma-dependent), not the empirical magnitude means.
  std::vector<ssize_t> index2shell;
  std::vector<ssize_t> index2group;
  Image<T> vst_mean_image;

  // Serialise this voxel's volumes into "data", applying phase demodulation and,
  //   where a noise level is available, the forward variance-stabilising transform.
  // This is the common forward preprocessing shared by the Casorati fill
  //   (operator() forward) and the stabilised-domain mean computation.
  void serialise_and_stabilise(Image<T> &input,
                               Image<cfloat> &phase,
                               Image<uint32_t> &serialise,
                               const Transform &transform,
                               Interp::Cubic<Image<float>> *vst,
                               Eigen::Array<T, Eigen::Dynamic, 1> &data) const;

  // (Re)compute the stabilised-domain per-group means from the input,
  //   using the currently stored noise level map for stabilisation.
  void compute_means(Image<T> input);
};

} // namespace MR::Denoise
