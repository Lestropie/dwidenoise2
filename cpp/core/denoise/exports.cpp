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

#include "denoise/exports.h"

#include "app.h"
#include "dwi/gradient.h"
#include "file/ofstream.h"

namespace MR::Denoise {

using namespace App;

// clang-format off
const char *const noise_boundary_description =
    "Two distinct estimates of the upper bound of the noise portion of the PCA eigenspectrum"
    " can be exported, via options -lamplus and -lammax_noise;"
    " these are not the same quantity, and should not be used interchangeably."
    " The first, \"lambda-plus\" (-lamplus), is THEORETICAL:"
    " it is the upper edge of the Marchenko-Pastur distribution"
    " predicted from the estimated noise level 'sigma'"
    " and the aspect ratio 'beta' of the Casorati matrix of the patch,"
    " ie. sigma^2 multiplied by (1+sqrt(beta))^2;"
    " being a function of only these two quantities,"
    " it conveys no information about where the boundary between signal and noise components"
    " was actually placed."
    " The second (-lammax_noise) is EMPIRICAL:"
    " it is the largest eigenvalue that the estimator classified as noise."
    " Because the aspect ratio of the Casorati matrix,"
    " and with it the width of the Marchenko-Pastur distribution,"
    " varies between patches wherever the kernel size does,"
    " this latter quantity is exported in standardised form,"
    " divided by the Marchenko-Pastur upper edge factor (1+sqrt(beta))^2 of its own patch,"
    " such that it is comparable between voxels irrespective of patch geometry."
    " Both are eigenvalues, and are therefore in units of the SQUARE of the estimated noise level"
    " (which, as for the noise level map, is that of a single real-valued channel)."
    " The standardised maximal noise eigenvalue can furthermore be compared directly"
    " against the square of the noise level estimated for that same patch:"
    " the two are equal where the estimator's classification of components falls exactly"
    " on the Marchenko-Pastur upper edge;"
    " the former is smaller where the discrete eigenspectrum holds no component near that edge;"
    " and it may be larger for those mechanisms whose classification is not based on that edge"
    " (-estimator exp1 or exp2; or -fixed_rank where available).";

OptionGroup export_options(const option_scope_t scope) {
  OptionGroup result("Options for exporting additional data regarding PCA behaviour");
  if (scope == option_scope_t::DENOISING) {
    result
    + Option("noise_out",
             "The output noise map,"
             " i.e., the estimated noise level 'sigma' in the data. "
             "This is always the noise level of a single real-valued channel,"
             " for complex input data as much as for magnitude:"
             " the value estimated from a complex series is that which would be estimated"
             " from the corresponding magnitude series at high SNR,"
             " and the map may therefore be exchanged freely between the two"
             " (via -noise_in) without rescaling. "
             "Note that this differs from the \"dwidenoise\" command in MRtrix3,"
             " whose noise map for complex input data is instead"
             " the aggregate level across the real and imaginary channels,"
             " and is therefore larger by a factor of sqrt(2).")
      + Argument("image").type_image_out();
  }
  result
  + Option("lamplus",
           "The THEORETICAL upper bound of the noise portion of the eigenspectrum (\"lambda-plus\"):"
           " the upper edge of the Marchenko-Pastur distribution predicted from the estimated noise level"
           " and the aspect ratio of the Casorati matrix of each patch,"
           " in units of the square of the estimated noise level."
           " This is not the same quantity as -lammax_noise"
           " (see the Description section of the help page for details)")
    + Argument("image").type_image_out()
  + Option("lammax_noise",
           "The EMPIRICAL upper bound of the noise portion of the eigenspectrum:"
           " the largest eigenvalue that the estimator classified as noise,"
           " standardised by the Marchenko-Pastur upper edge factor of each patch,"
           " in units of the square of the estimated noise level."
           " This is not the same quantity as -lamplus"
           " (see the Description section of the help page for details)")
    + Argument("image").type_image_out()
  + Option("rank_pcanonzero",
           "The (non-zero) rank of the PCA decomposition of each patch,"
           " prior to the classification of any components as noise")
    + Argument("image").type_image_out()
  + Option("rank_input",
           "The estimated rank of the input data for each PCA patch")
    + Argument("image").type_image_out();
  if (scope == option_scope_t::DENOISING) {
    result
    + Option("trace_output",
             "The effective number of degrees of freedom retained per voxel in the output image data,"
             " accounting for multi-patch aggregation:"
             " the trace of the composite reconstruction filter (the sum of its eigenvector weights)."
             " For a hard filter (-filter truncate or -filter optthresh) every weight is 0 or 1 and this"
             " is exactly the rank of the output data;"
             " under the default optimal shrinkage the weights are fractional and the trace is smaller"
             " than the number of surviving components."
             " This is the appropriate divisor for converting a residual sum of squares computed on the"
             " denoised data into an unbiased noise level estimate (see also -dof_output)")
      + Argument("image").type_image_out()
    + Option("dof_output",
             "The (approximate) degrees of freedom of the residual energy of a statistical model"
             " subsequently fit to the output image data,"
             " for use in place of the image volume count when setting that model's degrees of freedom."
             " This is the chi-squared-matching (Satterthwaite) measure"
             " (sum of squared filter weights, squared, divided by the sum of their fourth powers),"
             " which is the quantity governing the reference distribution of a t- or F-statistic;"
             " it is identical to -trace_output for a hard filter."
             " Under optimal shrinkage the two differ, in a direction set by the shape of the weight"
             " spectrum: many components of similar sub-unity weight drive it above the trace,"
             " whereas a few near-unity weights accompanied by many small ones drive it below."
             " Approximate in two respects:"
             " the higher moments from which it is formed are averaged across the patches contributing"
             " to each voxel, which neglects inter-patch cross terms"
             " (it is exact for -aggregator exclusive only);"
             " and, as for -trace_output, it treats the PCA basis as fixed rather than as having been"
             " estimated from the same data, which understates the true degrees of freedom by a factor"
             " of roughly 1 + (num_volumes - rank) / num_patch_voxels")
      + Argument("image").type_image_out()
    + Option("rankpermm_out",
             "Export the per-voxel signal-rank density (signal rank per mm of kernel radius) that was"
             " used to size the rank-adaptive reconstruction kernel."
             " For a multi-resolution schedule this is the density derived from the penultimate"
             " iteration; for a single-row schedule it is whatever was supplied via -rankpermm_in"
             " (nothing is written if no such density is available).")
      + Argument("image").type_image_out()
    + Option("variance_removed",
             "the fraction of variance removed in producing the output denoised data "
             "(note that this applies strictly to the PCA and omits effects of preconditioning)")
      + Argument("image").type_image_out();
  } else {
    result
    + Option("rankpermm_out",
             "Export the per-voxel signal-rank density (estimated signal rank per mm of kernel radius)."
             " This is the quantity dwidenoise2 uses to size its rank-adaptive reconstruction kernel;"
             " exporting it here and importing it into dwidenoise2 via -rankpermm_in allows a"
             " single-row (e.g. \"apriori\") dwidenoise2 schedule to reproduce that kernel.")
      + Argument("image").type_image_out();
  }
  result
  + Option("eigenspectra",
           "Output a matrix containing the spectra of eigenvalues across patches"
           " (one row per patch). Where volume partitioning is in effect, each row is the"
           " per-partition eigenspectra concatenated and sorted, normalised by partition size,"
           " rather than a single decomposition's spectrum.")
    + Argument("file").type_file_out();
  if (scope == option_scope_t::DENOISING) {
    result
    + Option("residual_statistics",
             "export images containing statistics of the residuals between input and denoised data,"
             " with image paths in order representing: mean, standard deviation, maxabs."
             " The standard deviation is that of a single real-valued channel,"
             " matching the convention of the -noise_out map,"
             " and is therefore directly comparable to it"
             " wherever the residual consists of noise alone")
      + Argument("mean_image").type_image_out()
      + Argument("std_image").type_image_out()
      + Argument("maxabs_image").type_image_out();
  }
  return result;
}

OptionGroup kernel_debug_options(const option_scope_t scope) {
  OptionGroup result("Options for debugging the operation of sliding window kernels");
  result
  + Option("max_dist",
           "The maximum distance between the centre of the patch and a voxel that was included within that patch")
    + Argument("image").type_image_out()
  + Option("voxelcount",
           "The number of voxels that contributed to the PCA for processing of each patch")
    + Argument("image").type_image_out()
  + Option("patchcount",
           "The number of unique patches to which an input image voxel contributes")
    + Argument("image").type_image_out();
  if (scope == option_scope_t::DENOISING) {
    result
    + Option("sum_aggregation",
             "The sum of aggregation weights of those patches contributing to each output voxel")
      + Argument("image").type_image_out()
    + Option("trace_patch",
             "the sum of eigenvector weights computed for the denoising patch centred at each voxel; "
             "the unaggregated, per-patch counterpart to -trace_output, on the sub-sampled grid")
      + Argument("image").type_image_out();
  }
  return result;
}
// clang-format on

Exports::Exports(const Header &in, const Header &ss) //
    : H_in(std::make_shared<Header>(in)),            //
      H_ss(std::make_shared<Header>(ss)) {           //
  DWI::clear_DW_scheme(*H_in);
  H_in->ndim() = 3;
  H_in->reset_intensity_scaling();
  H_in->datatype() = DataType::Float32;
  H_in->datatype().set_byte_order_native();
}

void Exports::set_noise_out(const std::string &path) { noise_out = Image<float>::create(path, *H_ss); }
void Exports::set_noise_out() { noise_out = Image<float>::scratch(*H_ss, "Scratch image for noise level output"); }

void Exports::set_lamplus(const std::string &path) { lamplus = Image<float>::create(path, *H_ss); }

void Exports::set_lammax_noise(const std::string &path) { lammax_noise = Image<float>::create(path, *H_ss); }

void Exports::set_rank_pcanonzero(const std::string &path) {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  rank_pcanonzero = Image<uint16_t>::create(path, H);
}

void Exports::set_rank_input(const std::string &path) {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  rank_input = Image<uint16_t>::create(path, H);
}
void Exports::set_rank_input() {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  rank_input = Image<uint16_t>::scratch(H, "Scratch image for PCA patch rank");
}

void Exports::set_trace_output(const std::string &path) { trace_output = Image<float>::create(path, *H_in); }

void Exports::set_dof_output(const std::string &path) {
  dof_output = Image<float>::create(path, *H_in);
  // The second and fourth moments are accumulated separately because the degrees-of-freedom measure
  //   derived from them is non-linear: unlike the first moment it cannot be aggregated across
  //   patches directly, so the two moments are aggregated and only then combined (see the commands).
  trace2_accum = Image<float>::scratch(*H_in, "Scratch image for aggregated squared filter weights");
  trace4_accum = Image<float>::scratch(*H_in, "Scratch image for aggregated fourth-power filter weights");
}

void Exports::set_trace_patch(const std::string &path) { trace_patch = Image<float>::create(path, *H_ss); }

void Exports::set_max_dist(const std::string &path) { max_dist = Image<float>::create(path, *H_ss); }
void Exports::set_max_dist() { max_dist = Image<float>::scratch(*H_ss, "Scratch image for maximal patch radius"); }

void Exports::set_voxelcount(const std::string &path) {
  Header H(*H_ss);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  voxelcount = Image<uint16_t>::create(path, H);
}

void Exports::set_patchcount(const std::string &path) {
  Header H(*H_in);
  H.datatype() = DataType::UInt16;
  H.datatype().set_byte_order_native();
  patchcount = Image<uint16_t>::create(path, H);
}

void Exports::set_sum_aggregation(const std::string &path) { sum_aggregation = Image<float>::create(path, *H_in); }
void Exports::set_sum_aggregation() {
  sum_aggregation = Image<float>::scratch(*H_in, "Scratch image for patch aggregation sums");
}

void Exports::set_variance_removed(const std::string &path) { variance_removed = Image<float>::create(path, *H_ss); }

void Exports::set_eigenspectra_path(const std::string &path) {
  assert(!eigenspectra);
  eigenspectra.reset(new Eigenspectra(path));
}

void Exports::add_eigenspectrum(const eigenvalues_type &s) {
  assert(eigenspectra);
  eigenspectra->add(s);
}

Exports::Eigenspectra::Eigenspectra(const std::string &path) : path(path) {}

void Exports::Eigenspectra::add(const eigenvalues_type &s) { data.push_back(s); }

Exports::Eigenspectra::~Eigenspectra() {
  File::OFStream out(path);
  out << "# " << App::command_history_string << "\n";
  for (const auto &s : data) {
    out << s[0];
    for (ssize_t i = 1; i != s.size(); ++i)
      out << "," << s[i];
    out << "\n";
  }
}

} // namespace MR::Denoise
