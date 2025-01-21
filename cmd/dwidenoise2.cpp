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

#include "denoise/denoise.h"
#include "denoise/estimator/base.h"
#include "denoise/estimator/estimator.h"
#include "denoise/estimator/exp.h"
#include "denoise/estimator/mrm2022.h"
#include "denoise/estimator/rank.h"
#include "denoise/estimator/result.h"
#include "denoise/exports.h"
#include "denoise/kernel/cuboid.h"
#include "denoise/kernel/data.h"
#include "denoise/kernel/kernel.h"
#include "denoise/kernel/sphere_radius.h"
#include "denoise/kernel/sphere_ratio.h"
#include "denoise/precondition.h"
#include "denoise/recon.h"
#include "denoise/subsample.h"

using namespace MR;
using namespace App;
using namespace MR::Denoise;

// clang-format off
void usage() {

  SYNOPSIS = "Improved dMRI denoising using Marchenko-Pastur PCA";

  AUTHOR = "Robert E. Smith (robert.smith@florey.edu.au)"
           " and Daan Christiaens (daan.christiaens@kcl.ac.uk)"
           " and Jelle Veraart (jelle.veraart@nyumc.org)"
           " and J-Donald Tournier (jdtournier@gmail.com)";

  DESCRIPTION
  + "This command performs DWI data denoising,"
    " additionally with data-driven noise map estimation if not provided explicitly."
    " The output denoised DWI data is formed based on filtering of eigenvectors of PCA decompositions:"
    " for a set of patches each of which consists of a set of proximal voxels,"
    " the PCA decomposition is applied,"
    " and the DWI signal for those voxels is reconstructed"
    " where the contribution of each eigenvector is modulated based on its classification as noise."
    " In many use cases,"
    " a threshold that classifies eigenvalues as belonging to signal of interest vs. random thermal noise"
    " is based on the prior knowledge that the eigenspectrum of random covariance matrices"
    " is described by the universal Marchenko-Pastur (MP) distribution."

  + "This command includes many capabilities absent from the original dwidenoise command. "
    "These include:"
    " - Multiple sliding window kernel shapes,"
      " including a spherical kernel that dilates at image edges to preserve aspect ratio;"
    " - A greater number of mechanisms for noise level estimation,"
      " including taking a pre-estimated noise map as input;"
    " - Preconditioning, including (per-shell) demeaning,"
      " phase demodulation (linear or nonlinear),"
      " and variance-stabilising transform to compensate for within-patch heteroscedasticity;"
    " - Overcomplete local PCA;"
    " - Subsampling (performing fewer PCAs than there are input voxels);"
    " - Optimal shrinkage of eigenvalues."

  + Denoise::first_step_description

  + Denoise::non_gaussian_noise_description

  + demodulation_description

  + Kernel::shape_description

  + Kernel::default_size_description

  + Kernel::cuboid_size_description

  + Denoise::filter_description

  + Denoise::aggregation_description;

  EXAMPLES
  + Example("To approximately replicate the behaviour of the original dwidenoise command",
            "dwidenoise2 DWI.mif out.mif -shape cuboid -subsample 1 -demodulate none -demean none -filter truncate -aggregator exclusive",
            "While this is neither guaranteed to match exactly the output of the original dwidenoise command"
            " nor is it a recommended use case,"
            " it may nevertheless be informative in demonstrating those advanced features of dwidenoise2 active by default"
            " that must be explicitly disabled in order to approximate that behaviour.");

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
  + Estimator::estimator_denoise_options
  + Kernel::options
  + subsample_option
  + precondition_options

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

}
// clang-format on

// Necessary to allow normalisation by sum of aggregation weights
//   where the image type is cdouble, but aggregation weights are float
// (operations combining complex & real types not allowed to be of different precision)
std::complex<double> operator/(const std::complex<double> &c, const float n) { return c / double(n); }

template <typename T>
void run(Image<T> &input,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         filter_type filter,
         aggregator_type aggregator,
         Image<T> &output,
         Exports &exports) {
  Recon<T> func(input, subsample, kernel, estimator, filter, aggregator, exports);
  ThreadedLoop("running MP-PCA denoising", input, 0, 3).run(func, input, output);
  // Rescale output if aggregation was performed
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
void run(Header &dwi,
         const Demodulation &demodulation,
         const demean_type demean,
         Image<float> &vst_noise_image,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         filter_type filter,
         aggregator_type aggregator,
         const std::string &output_name,
         Exports &exports) {
  auto opt_preconditioned = get_options("preconditioned");
  if (!demodulation && demean == demean_type::NONE && !vst_noise_image.valid()) {
    if (!opt_preconditioned.empty()) {
      WARN("-preconditioned option ignored: no preconditioning taking place");
    }
    auto input = dwi.get_image<T>().with_direct_io(3);
    Header H(dwi);
    H.datatype() = DataType::from<T>();
    auto output = Image<T>::create(output_name, H);
    run<T>(input, subsample, kernel, estimator, filter, aggregator, output, exports);
    return;
  }
  auto input = dwi.get_image<T>();
  // perform preconditioning
  const Precondition<T> preconditioner(input, demodulation, demean, vst_noise_image);
  Header H_preconditioned(dwi);
  Stride::set(H_preconditioned, Stride::contiguous_along_axis(3));
  H_preconditioned.datatype() = DataType::from<T>();
  H_preconditioned.datatype().set_byte_order_native();
  Image<T> input_preconditioned;
  input_preconditioned = opt_preconditioned.empty()
                             ? Image<T>::scratch(H_preconditioned, "Preconditioned version of \"" + dwi.name() + "\"")
                             : Image<T>::create(opt_preconditioned[0][0], H_preconditioned);
  preconditioner(input, input_preconditioned, false);
  // create output
  Header H(dwi);
  H.datatype() = DataType::from<T>();
  auto output = Image<T>::create(output_name, H);
  // run
  run(input_preconditioned, subsample, kernel, estimator, filter, aggregator, output, exports);
  // reverse effects of preconditioning
  Image<T> output2(output);
  preconditioner(output, output2, true);
  // compensate for effects of preconditioning where relevant
  if (exports.noise_out.valid() && vst_noise_image.valid()) {
    Interp::Cubic<Image<float>> vst(vst_noise_image);
    const Transform transform(exports.noise_out);
    for (auto l = Loop(exports.noise_out)(exports.noise_out); l; ++l) {
      vst.scanner(transform.voxel2scanner * Eigen::Vector3d{default_type(exports.noise_out.index(0)),
                                                            default_type(exports.noise_out.index(1)),
                                                            default_type(exports.noise_out.index(2))});
      exports.noise_out.value() *= vst.value();
    }
  }
  if (preconditioner.rank() == 1) {
    if (exports.rank_input.valid()) {
      for (auto l = Loop(exports.rank_input)(exports.rank_input); l; ++l)
        exports.rank_input.value() =
            std::min<uint16_t>(uint16_t(exports.rank_input.value()) + uint16_t(1), uint16_t(dwi.size(3)));
    }
    if (exports.rank_output.valid()) {
      for (auto l = Loop(exports.rank_output)(exports.rank_output); l; ++l)
        exports.rank_output.value() = std::min<float>(float(exports.rank_output.value()) + 1.0f, float(dwi.size(3)));
    }
    if (exports.sum_optshrink.valid()) {
      for (auto l = Loop(exports.sum_optshrink)(exports.sum_optshrink); l; ++l)
        exports.sum_optshrink.value() = float(exports.sum_optshrink.value()) + 1.0f;
    }
  }
}

void run() {
  auto dwi = Header::open(argument[0]);
  if (dwi.ndim() != 4 || dwi.size(3) <= 1)
    throw Exception("input image must be 4-dimensional");

  const Demodulation demodulation = select_demodulation(dwi);
  const demean_type demean = select_demean(dwi);
  Image<float> vst_noise_image;
  auto opt = get_options("vst");
  if (!opt.empty())
    vst_noise_image = Image<float>::open(opt[0][0]);

  auto subsample = Subsample::make(dwi);
  assert(subsample);

  auto kernel = Kernel::make_kernel(dwi, subsample->get_factors());
  assert(kernel);

  auto estimator = Estimator::make_estimator(vst_noise_image, true);
  assert(estimator);

  filter_type filter = get_options("fixed_rank").empty() ? filter_type::OPTSHRINK : filter_type::TRUNCATE;
  opt = get_options("filter");
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

  int prec = get_option_value("datatype", 0); // default: single precision
  if (dwi.datatype().is_complex())
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    assert(demodulation.axes.empty());
    INFO("select real float32 for processing");
    run<float>(          //
        dwi,             //
        demodulation,    //
        demean,          //
        vst_noise_image, //
        subsample,       //
        kernel,          //
        estimator,       //
        filter,          //
        aggregator,      //
        argument[1],     //
        exports);        //
    break;
  case 1:
    assert(demodulation.axes.empty());
    INFO("select real float64 for processing");
    run<double>(         //
        dwi,             //
        demodulation,    //
        demean,          //
        vst_noise_image, //
        subsample,       //
        kernel,          //
        estimator,       //
        filter,          //
        aggregator,      //
        argument[1],     //
        exports);        //
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(         //
        dwi,             //
        demodulation,    //
        demean,          //
        vst_noise_image, //
        subsample,       //
        kernel,          //
        estimator,       //
        filter,          //
        aggregator,      //
        argument[1],     //
        exports);        //
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(        //
        dwi,             //
        demodulation,    //
        demean,          //
        vst_noise_image, //
        subsample,       //
        kernel,          //
        estimator,       //
        filter,          //
        aggregator,      //
        argument[1],     //
        exports);        //
    break;
  }
}
