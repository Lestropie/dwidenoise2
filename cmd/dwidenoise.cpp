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

#include <string>

#include "command.h"
#include "filter/demodulate.h"
#include "header.h"
#include "image.h"
#include "stride.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "denoise/demodulate.h"
#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/estimator.h"
#include "denoise/estimator/exp.h"
#include "denoise/estimator/mrm2022.h"
#include "denoise/estimator/result.h"
#include "denoise/exports.h"
#include "denoise/kernel/cuboid.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/kernel.h"
#include "denoise/kernel/sphere_radius.h"
#include "denoise/kernel/sphere_ratio.h"
#include "denoise/recon.h"
#include "denoise/subsample.h"

using namespace MR;
using namespace App;
using namespace MR::Denoise;

// clang-format off
void usage() {

  SYNOPSIS = "dMRI noise level estimation and denoising using Marchenko-Pastur PCA";

  DESCRIPTION
  + "DWI data denoising and noise map estimation"
    " by exploiting data redundancy in the PCA domain"
    " using the prior knowledge that the eigenspectrum of random covariance matrices"
    " is described by the universal Marchenko-Pastur (MP) distribution."
    " Fitting the MP distribution to the spectrum of patch-wise signal matrices"
    " hence provides an estimator of the noise level 'sigma';"
    " this noise level estimate then determines the optimal cut-off for PCA denoising."

  + "Important note:"
    " image denoising must be performed as the first step of the image processing pipeline."
    " The routine will fail if interpolation or smoothing has been applied to the data prior to denoising."

  + "Note that this function does not correct for non-Gaussian noise biases"
    " present in magnitude-reconstructed MRI images."
    " If available, including the MRI phase data can reduce such non-Gaussian biases,"
    " and the command now supports complex input data."

  + demodulation_description

  + Kernel::shape_description

  + Kernel::default_size_description

  + Kernel::cuboid_size_description

  + "By default, optimal value shrinkage based on minimisation of the Frobenius norm "
    "will be used to attenuate eigenvectors based on the estimated noise level. "
    "Hard truncation of sub-threshold components and inclusion of supra-threshold components"
    "---which was the behaviour of the dwidenoise command in version 3.0.x---"
    "can be activated using -filter truncate."
    "Alternatively, optimal truncation as described in Gavish and Donoho 2014 "
    "can be utilised by specifying -filter optthresh."

  + "-aggregation exclusive corresponds to the behaviour of the dwidenoise command in version 3.0.x, "
    "where the output intensities for a given image voxel are determined exclusively "
    "from the PCA decomposition where the sliding spatial window is centred at that voxel. "
    "In all other use cases, so-called \"overcomplete local PCA\" is performed, "
    "where the intensities for an output image voxel are some combination of all PCA decompositions "
    "for which that voxel is included in the local spatial kernel. "
    "There are multiple algebraic forms that modulate the weight with which each decomposition "
    "contributes with greater or lesser strength toward the output image intensities. "
    "The various options are: "
    "'gaussian': A Gaussian distribution with FWHM equal to twice the voxel size, "
      "such that decompisitions centred more closely to the output voxel have greater influence; "
    "'invl0': The inverse of the L0 norm (ie. rank) of each decomposition, "
      "as used in Manjon et al. 2013; "
    "'rank': The rank of each decomposition, "
      "such that high-rank decompositions contribute more strongly to the output intensities "
      "regardless of distance between the output voxel and the centre of the decomposition kernel; "
    "'uniform': All decompositions that include the output voxel in the sliding spatial window contribute equally.";

  AUTHOR = " Robert E. Smith (robert.smith@florey.edu.au)"
           " and Daan Christiaens (daan.christiaens@kcl.ac.uk)"
           " and Jelle Veraart (jelle.veraart@nyumc.org)"
           " and J-Donald Tournier (jdtournier@gmail.com)"
           " and Robert E. Smith (robert.smith@florey.edu.au)";

  COPYRIGHT =
  "Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;"
  " The Florey Institute of Neuroscience and Mental Health."
  " Licensed under the PolyForm Noncommercial License 1.0.0 (the \"License\");"
  " you may not use this file except in compliance with the License."
  " You may obtain a copy of the License at:"
  " https://polyformproject.org/licenses/noncommercial/1.0.0."
  " Unless required by applicable law or agreed to in writing,"
  " software distributed under the License is distributed on an \"AS IS\" BASIS,"
  " WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,"
  " either express or implied."
  " See the License of the specific language"
  " governing permissions and limitations under the License.";

  REFERENCES
  + "Veraart, J.; Novikov, D.S.; Christiaens, D.; Ades-aron, B.; Sijbers, J. & Fieremans, E. " // Internal
    "Denoising of diffusion MRI using random matrix theory. "
    "NeuroImage, 2016, 142, 394-406, doi: 10.1016/j.neuroimage.2016.08.016"

  + "Veraart, J.; Fieremans, E. & Novikov, D.S. " // Internal
    "Diffusion MRI noise mapping using random matrix theory. "
    "Magn. Res. Med., 2016, 76(5), 1582-1593, doi: 10.1002/mrm.26059"

  + "Cordero-Grande, L.; Christiaens, D.; Hutter, J.; Price, A.N.; Hajnal, J.V. " // Internal
    "Complex diffusion-weighted image estimation via matrix recovery under general noise models. "
    "NeuroImage, 2019, 200, 391-404, doi: 10.1016/j.neuroimage.2019.06.039"

  + "* If using -estimator mrm2022: "
    "Olesen, J.L.; Ianus, A.; Ostergaard, L.; Shemesh, N.; Jespersen, S.N. "
    "Tensor denoising of multidimensional MRI data. "
    "Magnetic Resonance in Medicine, 2022, 89(3), 1160-1172"

  + "* If using anything other than -aggregation exclusive: "
    "Manjon, J.V.; Coupe, P.; Concha, L.; Buades, A.; D. Collins, D.L.; Robles, M. "
    "Diffusion Weighted Image Denoising Using Overcomplete Local PCA. "
    "PLoS ONE, 2013, 8(9), e73021"

  + "* If using -estimator med or -filter optthresh: "
    "Gavish, M.; Donoho, D.L."
    "The Optimal Hard Threshold for Singular Values is 4/sqrt(3). "
    "IEEE Transactions on Information Theory, 2014, 60(8), 5040-5053.";

  ARGUMENTS
  + Argument("dwi", "the input diffusion-weighted image.").type_image_in()
  + Argument("out", "the output denoised DWI image.").type_image_out();

  OPTIONS
  + OptionGroup("Options for modifying PCA computations")
  + datatype_option
  + Estimator::option
  + Kernel::options
  + subsample_option
  + demodulation_options
  + Option("noise_in",
           "import a pre-estimated noise level map rather than estimating this level during denoising")
    + Argument("image").type_image_in()

  + OptionGroup("Options that affect reconstruction of the output image series")
  + Option("filter",
           "Modulate how component contributions are filtered "
           "based on the cumulative eigenvalues relative to the noise level; "
           "options are: " + join(filters, ",") + "; "
           "default: optshrink (Optimal Shrinkage based on minimisation of the Frobenius norm)")
    + Argument("choice").type_choice(filters)
  + Option("aggregator",
           "Select how the outcomes of multiple PCA outcomes centred at different voxels "
           "contribute to the reconstructed DWI signal in each voxel; "
           "options are: " + join(aggregators, ",") + "; default: Gaussian")
    + Argument("choice").type_choice(aggregators)
  // TODO For specifically the Gaussian aggregator,
  //   should ideally be possible to select the FWHM of the aggregator

  // TODO Consider restructuring & encapsulating in classes
  // Also bear in mind how use of subsampling could affect this
  // TODO If using downsampling,
  //   may be preferable to explicitly downsample the voxel grid on which to yield these data
  // - -noise: This is a patch-wise estimate;
  //           could however elect to make it an aggreate mean if aggregation is being performed
  //           (or just forbid it if downsampling is performed?)
  // - -rank: This is a patch-wise estimate (see -noise above)
  // - -weightedrank: This is explicitly a multi-patch aggregation,
  //   so doesn't apply to noise level estimation only;
  //   it may be better to distinguish between "input rank" and "output rank"?
  // - -sumweights: Explicitly a multi-patch aggregation
  // - -max_dist: This is a property of the local patch;
  //              this could be hidden behind a #define TBH
  // - -voxels: This is a property of the local patch (see -max_dist above)
  // - -aggregation_sum: This is explicitly a multi-patch aggregation
  // TODO Consider an option group for "debugging of sliding window kernel behaviour"
  // TODO Potential issue wherein optimal shrinkage may set to 0
  //   some components above the determined noise level...
  + OptionGroup("Options for exporting additional data regarding PCA behaviour")
  + Option("noise_out",
           "The output noise map,"
           " i.e., the estimated noise level 'sigma' in the data. "
           "Note that on complex input data,"
           " this will be the total noise level across real and imaginary channels,"
           " so a scale factor sqrt(2) applies.")
    + Argument("image").type_image_out()
  + Option("rank_input",
           "The signal rank estimated for each denoising patch")
    + Argument("image").type_image_out()
  + Option("rank_output",
           "An estimated rank for the output image data, accounting for multi-patch aggregation")
    + Argument("image").type_image_out()

  + OptionGroup("Options for debugging the operation of sliding window kernels")
  + Option("max_dist",
           "The maximum distance between a voxel and another voxel that was included in the local denoising patch")
    + Argument("image").type_image_out()
  + Option("voxelcount",
           "The number of voxels that contributed to the PCA for processing of each voxel")
    + Argument("image").type_image_out()
  + Option("patchcount",
           "The number of unique patches to which an image voxel contributes")
    + Argument("image").type_image_out()
  + Option("sum_aggregation",
           "The sum of aggregation weights of those patches contributing to each output voxel")
    + Argument("image").type_image_out()
  + Option("sum_optshrink",
           "the sum of eigenvector weights computed for the denoising patch centred at each voxel "
           "as a result of performing optimal shrinkage")
    + Argument("image").type_image_out();

  COPYRIGHT =
      "Copyright (c) 2016 New York University, University of Antwerp, and the MRtrix3 contributors \n \n"
      "Permission is hereby granted, free of charge, to any non-commercial entity ('Recipient') obtaining a copy of "
      "this software and "
      "associated documentation files (the 'Software'), to the Software solely for non-commercial research, including "
      "the rights to "
      "use, copy and modify the Software, subject to the following conditions: \n \n"
      "\t 1. The above copyright notice and this permission notice shall be included by Recipient in all copies or "
      "substantial portions of "
      "the Software. \n \n"
      "\t 2. THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT "
      "LIMITED TO THE WARRANTIES"
      "OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR "
      "COPYRIGHT HOLDERS BE"
      "LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING "
      "FROM, OUT OF OR"
      "IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. \n \n"
      "\t 3. In no event shall NYU be liable for direct, indirect, special, incidental or consequential damages in "
      "connection with the Software. "
      "Recipient will defend, indemnify and hold NYU harmless from any claims or liability resulting from the use of "
      "the Software by recipient. \n \n"
      "\t 4. Neither anything contained herein nor the delivery of the Software to recipient shall be deemed to grant "
      "the Recipient any right or "
      "licenses under any patents or patent application owned by NYU. \n \n"
      "\t 5. The Software may only be used for non-commercial research and may not be used for clinical care. \n \n"
      "\t 6. Any publication by Recipient of research involving the Software shall cite the references listed below.";
}
// clang-format on

// Necessary to allow normalisation by sum of aggregation weights
//   where the image type is cdouble, but aggregation weights are float
// (operations combining complex & real types not allowed to be of different precision)
std::complex<double> operator/(const std::complex<double> &c, const float n) { return c / double(n); }

template <typename T>
void run(Header &data,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         filter_type filter,
         aggregator_type aggregator,
         const std::string &output_name,
         Exports &exports) {
  auto input = data.get_image<T>().with_direct_io(3);
  // create output
  Header header(data);
  header.datatype() = DataType::from<T>();
  auto output = Image<T>::create(output_name, header);
  // run
  Recon<T> func(data, subsample, kernel, estimator, filter, aggregator, exports);
  ThreadedLoop("running MP-PCA denoising", data, 0, 3).run(func, input, output);
  // Rescale output if performing aggregation
  if (aggregator == aggregator_type::EXCLUSIVE)
    return;
  for (auto l_voxel = Loop(exports.sum_aggregation)(output, exports.sum_aggregation); l_voxel; ++l_voxel) {
    for (auto l_volume = Loop(3)(output); l_volume; ++l_volume)
      output.value() /= float(exports.sum_aggregation.value());
  }
  if (exports.rank_output.valid()) {
    for (auto l = Loop(exports.sum_aggregation)(exports.rank_output, exports.sum_aggregation); l; ++l)
      exports.rank_output.value() /= exports.sum_aggregation.value();
  }
}

template <typename T>
void run(Header &data,
         const std::vector<size_t> &demodulation_axes,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         filter_type filter,
         aggregator_type aggregator,
         const std::string &output_name,
         Exports &exports) {
  if (demodulation_axes.empty()) {
    run<T>(data, subsample, kernel, estimator, filter, aggregator, output_name, exports);
    return;
  }
  auto input = data.get_image<T>();
  // generate scratch version of DWI with phase demodulation
  Header H_scratch(data);
  Stride::set(H_scratch, Stride::contiguous_along_axis(3));
  H_scratch.datatype() = DataType::from<T>();
  H_scratch.datatype().set_byte_order_native();
  auto input_demodulated = Image<T>::scratch(H_scratch, "Phase-demodulated version of input DWI");
  Filter::Demodulate demodulate(input, demodulation_axes);
  demodulate(input, input_demodulated, false);
  input = Image<T>(); // free memory
  // create output
  Header header(data);
  header.datatype() = DataType::from<T>();
  auto output = Image<T>::create(output_name, header);
  // run
  Recon<T> func(data, subsample, kernel, estimator, filter, aggregator, exports);
  ThreadedLoop("running MP-PCA denoising", data, 0, 3).run(func, input_demodulated, output);
  // Re-apply phase ramps that were previously demodulated
  demodulate(output, true);
  // Rescale output if performing aggregation
  if (aggregator == aggregator_type::EXCLUSIVE)
    return;
  for (auto l_voxel = Loop(exports.sum_aggregation)(output, exports.sum_aggregation); l_voxel; ++l_voxel) {
    for (auto l_volume = Loop(3)(output); l_volume; ++l_volume)
      output.value() /= float(exports.sum_aggregation.value());
  }
  if (exports.rank_output.valid()) {
    for (auto l = Loop(exports.sum_aggregation)(exports.rank_output, exports.sum_aggregation); l; ++l)
      exports.rank_output.value() /= exports.sum_aggregation.value();
  }
}

void run() {
  auto dwi = Header::open(argument[0]);

  if (dwi.ndim() != 4 || dwi.size(3) <= 1)
    throw Exception("input image must be 4-dimensional");

  auto subsample = Subsample::make(dwi);
  assert(subsample);

  auto kernel = Kernel::make_kernel(dwi, subsample->get_factors());
  assert(kernel);

  auto estimator = Estimator::make_estimator();
  assert(estimator);

  filter_type filter = filter_type::OPTSHRINK;
  auto opt = get_options("filter");
  if (!opt.empty())
    filter = filter_type(int(opt[0][0]));

  aggregator_type aggregator = aggregator_type::GAUSSIAN;
  opt = get_options("aggregator");
  if (!opt.empty()) {
    aggregator = aggregator_type(int(opt[0][0]));
    if (aggregator == aggregator_type::EXCLUSIVE && subsample->get_factors() != std::array<ssize_t, 3>({1, 1, 1}))
      throw Exception("Cannot combine -aggregator exclusive with subsampling; "
                      "would result in empty output voxels");
  }

  Exports exports(dwi, subsample->header());
  opt = get_options("noise_out");
  if (!opt.empty())
    exports.set_noise_out(opt[0][0]);
  opt = get_options("rank_input");
  if (!opt.empty())
    exports.set_rank_input(opt[0][0]);
  opt = get_options("rank_output");
  if (!opt.empty()) {
    if (aggregator == aggregator_type::EXCLUSIVE && filter == filter_type::TRUNCATE) {
      WARN("When using -aggregator exclusive and -filter truncate, "
           "the output of -rank_output will be identical to the output of -rank_input, "
           "as there is no aggregation of multiple patches per output voxel "
           "and no optimal shrinkage to reduce output rank relative to estimated input rank");
    }
    exports.set_rank_output(opt[0][0]);
  }
  opt = get_options("sum_optshrink");
  if (!opt.empty()) {
    if (filter == filter_type::TRUNCATE) {
      WARN("Note that with a truncation filter, "
           "output image from -sumweights option will be equivalent to rank_input");
    }
    exports.set_sum_optshrink(opt[0][0]);
  }
  opt = get_options("max_dist");
  if (!opt.empty())
    exports.set_max_dist(opt[0][0]);
  opt = get_options("voxelcount");
  if (!opt.empty())
    exports.set_voxelcount(opt[0][0]);
  opt = get_options("patchcount");
  if (!opt.empty())
    exports.set_patchcount(opt[0][0]);

  opt = get_options("sum_aggregation");
  if (!opt.empty()) {
    if (aggregator == aggregator_type::EXCLUSIVE) {
      WARN("Output from -sum_aggregation will just contain 1 for every voxel processed: "
           "no patch aggregation takes place when output series comes exclusively from central patch");
    }
    exports.set_sum_aggregation(opt[0][0]);
  } else if (aggregator != aggregator_type::EXCLUSIVE) {
    exports.set_sum_aggregation("");
  }

  const std::vector<size_t> demodulation_axes = get_demodulation_axes(dwi);

  int prec = get_option_value("datatype", 0); // default: single precision
  if (dwi.datatype().is_complex())
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    assert(demodulation_axes.empty());
    INFO("select real float32 for processing");
    run<float>(dwi, subsample, kernel, estimator, filter, aggregator, argument[1], exports);
    break;
  case 1:
    assert(demodulation_axes.empty());
    INFO("select real float64 for processing");
    run<double>(dwi, subsample, kernel, estimator, filter, aggregator, argument[1], exports);
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(dwi, demodulation_axes, subsample, kernel, estimator, filter, aggregator, argument[1], exports);
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(dwi, demodulation_axes, subsample, kernel, estimator, filter, aggregator, argument[1], exports);
    break;
  }
}
