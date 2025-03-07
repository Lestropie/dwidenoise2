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

#include "denoise/denoise.h"
#include "header.h"
#include "image.h"

namespace MR::Denoise {

class Exports {
public:
  Exports(const Header &in, const Header &ss);
  Exports(const Exports &that) = default;

  void set_noise_out(const std::string &path);
  void set_noise_out();
  void set_lamplus(const std::string &path);
  void set_rank_pcanonzero(const std::string &path);
  void set_rank_input(const std::string &path);
  void set_rank_output(const std::string &path);
  void set_sum_optshrink(const std::string &path);
  void set_max_dist(const std::string &path);
  void set_voxelcount(const std::string &path);
  void set_patchcount(const std::string &path);
  void set_sum_aggregation(const std::string &path);
  void set_sum_aggregation();
  void set_variance_removed(const std::string &path);
  void set_eigenspectra_path(const std::string &path);

  Image<float> noise_out;
  Image<float> lamplus;
  Image<uint16_t> rank_pcanonzero;
  Image<uint16_t> rank_input;
  Image<float> rank_output;
  Image<float> sum_optshrink;
  Image<float> max_dist;
  Image<uint16_t> voxelcount;
  Image<uint16_t> patchcount;
  Image<float> sum_aggregation;
  Image<float> variance_removed;

  // std::string eigenspectra_path;
  // std::vector<eigenvalues_type> eigenspectra_data;
  bool saving_eigenspectra() const { return bool(eigenspectra); }
  void add_eigenspectrum(const eigenvalues_type &s);

protected:
  std::shared_ptr<Header> H_in;
  std::shared_ptr<Header> H_ss;

  // Needs to be stored as a std::shared_ptr<> so that in a multi-threading environment
  //   only one attempt is made to write to the output file
  class Eigenspectra {
  public:
    Eigenspectra(const std::string &path);
    ~Eigenspectra();
    void add(const eigenvalues_type &s);

  private:
    const std::string path;
    std::vector<eigenvalues_type> data;
  };
  std::shared_ptr<Eigenspectra> eigenspectra;
};

} // namespace MR::Denoise
