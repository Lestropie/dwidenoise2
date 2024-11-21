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
#include "command.h"
#include "denoise/estimate.h"
#include "denoise/estimator/estimator.h"
#include "denoise/exports.h"
#include "denoise/kernel/kernel.h"
#include "denoise/subsample.h"
#include "exception.h"

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

  + Kernel::shape_description

  + Kernel::default_size_description

  + Kernel::cuboid_size_description;

  AUTHOR = "Daan Christiaens (daan.christiaens@kcl.ac.uk)"
           " and Jelle Veraart (jelle.veraart@nyumc.org)"
           " and J-Donald Tournier (jdtournier@gmail.com)"
           " and Robert E. Smith (robert.smith@florey.edu.au)";

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
    "Magnetic Resonance in Medicine, 2022, 89(3), 1160-1172";

  ARGUMENTS
  + Argument("dwi", "the input diffusion-weighted image").type_image_in()
  + Argument("noise", "the output estimated noise level map").type_image_out();

  OPTIONS
  + OptionGroup("Options for modifying PCA computations")
  + datatype_option
  + Estimator::option
  + Kernel::options
  + subsample_option

  // TODO Implement mask option
  // Note that behaviour of -mask for dwi2noise may be different to that of dwidenoise

  + OptionGroup("Options for exporting additional data regarding PCA behaviour")
  + Option("rank",
           "The signal rank estimated for the denoising patch centred at each input image voxel")
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

template <typename T>
void run(Header &data,
         std::shared_ptr<Subsample> subsample,
         std::shared_ptr<Kernel::Base> kernel,
         std::shared_ptr<Estimator::Base> estimator,
         Exports &exports) {
  auto input = data.get_image<T>().with_direct_io(3);
  Image<bool> mask; // unused
  Estimate<T> func(data, mask, subsample, kernel, estimator, exports);
  ThreadedLoop("running MP-PCA noise level estimation", data, 0, 3).run(func, input);
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

  Exports exports(dwi, subsample->header());
  exports.set_noise_out(argument[1]);
  auto opt = get_options("rank");
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
  if (dwi.datatype().is_complex())
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    INFO("select real float32 for processing");
    run<float>(dwi, subsample, kernel, estimator, exports);
    break;
  case 1:
    INFO("select real float64 for processing");
    run<double>(dwi, subsample, kernel, estimator, exports);
    break;
  case 2:
    INFO("select complex float32 for processing");
    run<cfloat>(dwi, subsample, kernel, estimator, exports);
    break;
  case 3:
    INFO("select complex float64 for processing");
    run<cdouble>(dwi, subsample, kernel, estimator, exports);
    break;
  }
}
