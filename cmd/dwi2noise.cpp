/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#include <memory>

#include "algo/threaded_loop.h"
#include "axes.h"
#include "command.h"
#include "denoise/denoise.h"
#include "denoise/estimate.h"
#include "denoise/estimator/estimator.h"
#include "denoise/exports.h"
#include "denoise/kernel/kernel.h"
#include "denoise/precondition.h"
#include "denoise/subsample.h"
#include "dwi/gradient.h"
#include "exception.h"
#include "filter/demodulate.h"

using namespace MR;
using namespace App;
using namespace MR::Denoise;

// clang-format off
void usage() {

  SYNOPSIS = "Noise level estimation using Marchenko-Pastur PCA";

  DESCRIPTION
  + "DWI data noise map estimation"
    " by interrogating data redundancy in the PCA domain"
    " using the prior knowledge that the eigenspectrum of random covariance matrices"
    " is described by the universal Marchenko-Pastur (MP) distribution."
    " Fitting the MP distribution to the spectrum of patch-wise signal matrices"
    " hence provides an estimator of the noise level 'sigma'."

  + "Unlike the MRtrix3 command dwidenoise,"
    " this command does not generate a denoised version of the input image series;"
    " its primary output is instead a map of the estimated noise level."
    " While this can also be obtained from the dwidenoise command using option -noise_out,"
    " using instead the dwi2noise command gives the ability to obtain a noise map"
    " to which filtering can be applied,"
    " which can then be utilised for the actual image series denoising,"
    " without generating an unwanted intermiedate denoised image series."

  + "Important note:"
    " noise level estimation should only be performed as the first step of an image processing pipeline."
    " The routine is invalid if interpolation or smoothing has been applied to the data prior to denoising."

  + "Note that on complex input data,"
    " the output will be the total noise level across real and imaginary channels,"
    " so a scale factor sqrt(2) applies."

  + demodulation_description

  + Kernel::shape_description

  + Kernel::default_size_description

  + Kernel::cuboid_size_description;

  AUTHOR = "Robert E. Smith (robert.smith@florey.edu.au)"
           " and Daan Christiaens (daan.christiaens@kcl.ac.uk)"
           " and Jelle Veraart (jelle.veraart@nyumc.org)"
           " and J-Donald Tournier (jdtournier@gmail.com)";

  REFERENCES
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

  + "* If using -estimator med: "
    "Gavish, M.; Donoho, D.L. "
    "The Optimal Hard Threshold for Singular Values is 4/sqrt(3). "
    "IEEE Transactions on Information Theory, 2014, 60(8), 5040-5053.";

  ARGUMENTS
  + Argument("dwi", "the input diffusion-weighted image").type_image_in()
  + Argument("noise", "the output estimated noise level map").type_image_out();

  OPTIONS
  + OptionGroup("Options for modifying PCA computations")
  + datatype_option
  + Estimator::estimator_option
  + Kernel::options
  + subsample_option
  + precondition_options

  + DWI::GradImportOptions()
  + DWI::GradExportOptions()

  + OptionGroup("Options for exporting additional data regarding PCA behaviour")
  + Option("rank",
           "The signal rank estimated for each denoising patch")
    + Argument("image").type_image_out()
  + OptionGroup("Options for debugging the operation of sliding window kernels")
  + Option("max_dist",
           "The maximum distance between the centre of the patch and a voxel that was included within that patch")
    + Argument("image").type_image_out()
  + Option("voxelcount",
           "The number of voxels that contributed to the PCA for processing of each patch")
    + Argument("image").type_image_out()
  + Option("patchcount",
           "The number of unique patches to which an input image voxel contributes")
    + Argument("image").type_image_out();

}
// clang-format on

template <typename T>
void run(Image<T> &input,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         Exports &exports) {
  Estimate<T> func(input, subsample, kernel, estimator, exports);
  ThreadedLoop("running MP-PCA noise level estimation", input, 0, 3).run(func, input);
}

template <typename T>
void run(Header &dwi,
         const Demodulation &demodulation,
         const demean_type demean,
         Image<float> &vst_noise_image,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         Exports &exports) {
  auto opt_preconditioned = get_options("preconditioned");
  if (!demodulation && demean == demean_type::NONE && !vst_noise_image.valid()) {
    if (!opt_preconditioned.empty()) {
      WARN("-preconditioned option ignored: no preconditioning taking place");
    }
    Image<T> input = dwi.get_image<T>().with_direct_io(3);
    run<T>(input, subsample, kernel, estimator, exports);
    return;
  }
  Image<T> input(dwi.get_image<T>());
  const Precondition<T> preconditioner(input, demodulation, demean, vst_noise_image);
  Header H_preconditioned(input);
  Stride::set(H_preconditioned, Stride::contiguous_along_axis(3, input));
  Image<T> input_preconditioned;
  input_preconditioned = opt_preconditioned.empty()
                             ? Image<T>::scratch(H_preconditioned, "Preconditioned version of \"" + input.name() + "\"")
                             : Image<T>::create(opt_preconditioned[0][0], H_preconditioned);
  preconditioner(input, input_preconditioned, false);
  run(input_preconditioned, subsample, kernel, estimator, exports);
  if (vst_noise_image.valid()) {
    Interp::Cubic<Image<float>> vst(vst_noise_image);
    const Transform transform(exports.noise_out);
    for (auto l = Loop(exports.noise_out)(exports.noise_out); l; ++l) {
      vst.scanner(transform.voxel2scanner * Eigen::Vector3d({default_type(exports.noise_out.index(0)),
                                                             default_type(exports.noise_out.index(1)),
                                                             default_type(exports.noise_out.index(2))}));
      exports.noise_out.value() *= vst.value();
    }
  }
  if (preconditioner.rank() == 1 && exports.rank_input.valid()) {
    for (auto l = Loop(exports.rank_input)(exports.rank_input); l; ++l)
      exports.rank_input.value() =
          std::max<uint16_t>(uint16_t(exports.rank_input.value()) + uint16_t(1), uint16_t(dwi.size(3)));
  }
}

void run() {
  auto dwi = Header::open(argument[0]);
  if (dwi.ndim() != 4 || dwi.size(3) <= 1)
    throw Exception("input image must be 4-dimensional");
  bool complex = dwi.datatype().is_complex();

  const Demodulation demodulation = select_demodulation(dwi);
  const demean_type demean = select_demean(dwi);
  Image<float> vst_noise_image;
  auto opt = get_options("vst");
  if (!opt.empty())
    vst_noise_image = Image<float>::open(opt[0][0]);

  auto subsample = Subsample::make(dwi, Denoise::default_subsample_ratio);
  assert(subsample);

  auto kernel = Kernel::make_kernel(dwi, subsample->get_factors());
  assert(kernel);

  auto estimator = Estimator::make_estimator(vst_noise_image, false);
  assert(estimator);

  Exports exports(dwi, subsample->header());
  exports.set_noise_out(argument[1]);
  opt = get_options("rank");
  if (!opt.empty())
    exports.set_rank_input(opt[0][0]);
  opt = get_options("max_dist");
  if (!opt.empty())
    exports.set_max_dist(opt[0][0]);
  opt = get_options("voxelcount");
  if (!opt.empty())
    exports.set_voxelcount(opt[0][0]);
  opt = get_options("patchcount");
  if (!opt.empty())
    exports.set_patchcount(opt[0][0]);

  int prec = get_option_value("datatype", 0); // default: single precision
  if (complex)
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    assert(demodulation.axes.empty());
    INFO("select real float32 for processing");
    run<float>(dwi, demodulation, demean, vst_noise_image, subsample, kernel, estimator, exports);
    break;
  case 1:
    assert(demodulation.axes.empty());
    INFO("select real float64 for processing");
    run<double>(dwi, demodulation, demean, vst_noise_image, subsample, kernel, estimator, exports);
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(dwi, demodulation, demean, vst_noise_image, subsample, kernel, estimator, exports);
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(dwi, demodulation, demean, vst_noise_image, subsample, kernel, estimator, exports);
    break;
  }
}
