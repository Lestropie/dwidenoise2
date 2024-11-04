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

#include "command.h"
#include "image.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace MR;
using namespace App;

const std::vector<std::string> dtypes = {"float32", "float64"};
const std::vector<std::string> estimators = {"exp1", "exp2"};

// clang-format off
void usage() {

  SYNOPSIS = "dMRI noise level estimation and denoising using Marchenko-Pastur PCA";

  DESCRIPTION
  + "DWI data denoising and noise map estimation"
    " by exploiting data redundancy in the PCA domain"
    " using the prior knowledge that the eigenspectrum of random covariance matrices"
    " is described by the universal Marchenko-Pastur (MP) distribution."
    " Fitting the MP distribution to the spectrum of patch-wise signal matrices"
    " hence provides an estimator of the noise level 'sigma',"
    " as was first shown in Veraart et al. (2016)"
    " and later improved in Cordero-Grande et al. (2019)."
    " This noise level estimate then determines the optimal cut-off for PCA denoising."

  + "Important note:"
    " image denoising must be performed as the first step of the image processing pipeline."
    " The routine will fail if interpolation or smoothing has been applied to the data prior to denoising."

  + "Note that this function does not correct for non-Gaussian noise biases"
    " present in magnitude-reconstructed MRI images."
    " If available, including the MRI phase data can reduce such non-Gaussian biases,"
    " and the command now supports complex input data.";

  AUTHOR = " Robert E. Smith (robert.smith@florey.edu.au)"
           " and Daan Christiaens (daan.christiaens@kcl.ac.uk)"
           " and Jelle Veraart (jelle.veraart@nyumc.org)"
           " and J-Donald Tournier (jdtournier@gmail.com)";

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
    "NeuroImage, 2019, 200, 391-404, doi: 10.1016/j.neuroimage.2019.06.039";

  ARGUMENTS
  + Argument("dwi", "the input diffusion-weighted image.").type_image_in()
  + Argument("out", "the output denoised DWI image.").type_image_out();

  OPTIONS
  + Option("mask",
           "Only process voxels within the specified binary brain mask image.")
    + Argument("image").type_image_in()

  + Option("extent",
           "Set the patch size of the denoising filter."
           " By default, the command will select the smallest isotropic patch size"
           " that exceeds the number of DW images in the input data,"
           " e.g., 5x5x5 for data with <= 125 DWI volumes,"
           " 7x7x7 for data with <= 343 DWI volumes,"
           " etc.")
    + Argument("window").type_sequence_int()

  + Option("noise",
           "The output noise map,"
           " i.e., the estimated noise level 'sigma' in the data."
           "Note that on complex input data,"
           " this will be the total noise level across real and imaginary channels,"
           " so a scale factor sqrt(2) applies.")
    + Argument("level").type_image_out()

  + Option("rank",
           "The selected signal rank of the output denoised image.")
    + Argument("cutoff").type_image_out()

  + Option("datatype",
           "Datatype for the eigenvalue decomposition"
           " (single or double precision). "
           "For complex input data,"
           " this will select complex float32 or complex float64 datatypes.")
    + Argument("float32/float64").type_choice(dtypes)

  + Option("estimator",
           "Select the noise level estimator"
           " (default = Exp2),"
           " either: \n"
           "* Exp1: the original estimator used in Veraart et al. (2016), or \n"
           "* Exp2: the improved estimator introduced in Cordero-Grande et al. (2019).")
    + Argument("Exp1/Exp2").type_choice(estimators);

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

using real_type = float;

// Class to encode return information from kernel
template <class MatrixType> class KernelData {
public:
  KernelData(const size_t volumes, const size_t kernel_size)
      : centre_index(-1),                            //
        voxel_count(kernel_size),                    //
        X(MatrixType::Zero(volumes, kernel_size)) {} //
  size_t centre_index;
  size_t voxel_count;
  MatrixType X;
};

template <class MatrixType> class KernelBase {
public:
  KernelBase() : pos({-1, -1, -1}) {}
  KernelBase(const KernelBase &) : pos({-1, -1, -1}) {}

protected:
  // Store / restore position of image before / after data loading
  std::array<ssize_t, 3> pos;
  template <class ImageType> void stash_pos(const ImageType &image) {
    for (size_t axis = 0; axis != 3; ++axis)
      pos[axis] = image.index(axis);
  }
  template <class ImageType> void restore_pos(ImageType &image) {
    for (size_t axis = 0; axis != 3; ++axis)
      image.index(axis) = pos[axis];
  }
};

template <class MatrixType> class KernelCube : public KernelBase<MatrixType> {
public:
  KernelCube(const std::vector<uint32_t> &extent)
      : half_extent({int(extent[0] / 2), int(extent[1] / 2), int(extent[2] / 2)}) {
    for (auto e : extent) {
      if (!(e % 2))
        throw Exception("Size of cubic kernel must be an odd integer");
    }
  }
  KernelCube(const KernelCube &) = default;
  template <class ImageType> void operator()(ImageType &image, KernelData<MatrixType> &data) {
    assert(data.X.cols() == size());
    KernelBase<MatrixType>::stash_pos(image);
    size_t k = 0;
    for (int z = -half_extent[2]; z <= half_extent[2]; z++) {
      image.index(2) = wrapindex(z, 2, image.size(2));
      for (int y = -half_extent[1]; y <= half_extent[1]; y++) {
        image.index(1) = wrapindex(y, 1, image.size(1));
        for (int x = -half_extent[0]; x <= half_extent[0]; x++, k++) {
          image.index(0) = wrapindex(x, 0, image.size(0));
          data.X.col(k) = image.row(3);
        }
      }
    }
    KernelBase<MatrixType>::restore_pos(image);
    data.voxel_count = size();
    data.centre_index = size() / 2;
  }
  size_t size() const { return (2 * half_extent[0] + 1) * (2 * half_extent[1] + 1) * (2 * half_extent[2] + 1); }

private:
  const std::vector<int> half_extent;

  // patch handling at image edges
  inline size_t wrapindex(int r, int axis, int max) const {
    int rr = KernelBase<MatrixType>::pos[axis] + r;
    if (rr < 0)
      rr = half_extent[axis] - r;
    if (rr >= max)
      rr = (max - 1) - half_extent[axis] - r;
    return rr;
  }
};

template <typename F, class KernelType> class DenoisingFunctor {

public:
  using MatrixType = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;
  using SValsType = Eigen::VectorXd;

  DenoisingFunctor(
      int ndwi, KernelType &kernel, Image<bool> &mask, Image<real_type> &noise, Image<uint16_t> &rank, bool exp1)
      : data(ndwi, kernel.size()),
        kernel(kernel),
        m(ndwi),
        n(kernel.size()),
        r(std::min(m, n)),
        q(std::max(m, n)),
        exp1(exp1),
        XtX(r, r),
        eig(r),
        s(r),
        mask(mask),
        noise(noise),
        rankmap(rank) {}

  template <typename ImageType> void operator()(ImageType &dwi, ImageType &out) {
    // Process voxels in mask only
    if (mask.valid()) {
      assign_pos_of(dwi, 0, 3).to(mask);
      if (!mask.value())
        return;
    }

    // Load data in local window
    kernel(dwi, data);

    // Compute Eigendecomposition:
    if (m <= n)
      XtX.template triangularView<Eigen::Lower>() = data.X * data.X.adjoint();
    else
      XtX.template triangularView<Eigen::Lower>() = data.X.adjoint() * data.X;
    eig.compute(XtX);
    // eigenvalues sorted in increasing order:
    s = eig.eigenvalues().template cast<double>();

    // Marchenko-Pastur optimal threshold
    const double lam_r = std::max(s[0], 0.0) / q;
    double clam = 0.0;
    sigma2 = 0.0;
    ssize_t cutoff_p = 0;
    for (ssize_t p = 0; p < r; ++p) // p+1 is the number of noise components
    {                               // (as opposed to the paper where p is defined as the number of signal components)
      double lam = std::max(s[p], 0.0) / q;
      clam += lam;
      double gam = double(p + 1) / (exp1 ? q : q - (r - p - 1));
      double sigsq1 = clam / double(p + 1);
      double sigsq2 = (lam - lam_r) / (4.0 * std::sqrt(gam));
      // sigsq2 > sigsq1 if signal else noise
      if (sigsq2 < sigsq1) {
        sigma2 = sigsq1;
        cutoff_p = p + 1;
      }
    }

    if (cutoff_p > 0) {
      // recombine data using only eigenvectors above threshold:
      s.head(cutoff_p).setZero();
      s.tail(r - cutoff_p).setOnes();
      if (m <= n)
        data.X.col(data.centre_index) =
            eig.eigenvectors() *
            (s.cast<F>().asDiagonal() * (eig.eigenvectors().adjoint() * data.X.col(data.centre_index)));
      else
        data.X.col(data.centre_index) =
            data.X *
            (eig.eigenvectors() * (s.cast<F>().asDiagonal() * eig.eigenvectors().adjoint().col(data.centre_index)));
    }

    // Store output
    assign_pos_of(dwi).to(out);
    out.row(3) = data.X.col(data.centre_index);

    // store noise map if requested:
    if (noise.valid()) {
      assign_pos_of(dwi, 0, 3).to(noise);
      noise.value() = real_type(std::sqrt(sigma2));
    }
    // store rank map if requested:
    if (rankmap.valid()) {
      assign_pos_of(dwi, 0, 3).to(rankmap);
      rankmap.value() = uint16_t(r - cutoff_p);
    }
  }

private:
  KernelData<MatrixType> data;
  KernelType kernel;
  const ssize_t m, n, r, q;
  const bool exp1;
  MatrixType XtX;
  Eigen::SelfAdjointEigenSolver<MatrixType> eig;
  SValsType s;
  double sigma2;
  Image<bool> mask;
  Image<real_type> noise;
  Image<uint16_t> rankmap;
};

template <typename T, class KernelType>
void process_image(Header &data,
                   Image<bool> &mask,
                   Image<real_type> &noise,
                   Image<uint16_t> &rank,
                   const std::string &output_name,
                   KernelType &kernel,
                   bool exp1) {
  auto input = data.get_image<T>().with_direct_io(3);
  // create output
  Header header(data);
  header.datatype() = DataType::from<T>();
  auto output = Image<T>::create(output_name, header);
  // run
  DenoisingFunctor<T, KernelType> func(data.size(3), kernel, mask, noise, rank, exp1);
  ThreadedLoop("running MP-PCA denoising", data, 0, 3).run(func, input, output);
}

template <typename T>
void make_kernel(Header &data,
                 Image<bool> &mask,
                 Image<real_type> &noise,
                 Image<uint16_t> &rank,
                 const std::string &output_name,
                 bool exp1) {
  using KernelType = KernelCube<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

  auto opt = get_options("extent");
  std::vector<uint32_t> extent;
  if (!opt.empty()) {
    extent = parse_ints<uint32_t>(opt[0][0]);
    if (extent.size() == 1)
      extent = {extent[0], extent[0], extent[0]};
    if (extent.size() != 3)
      throw Exception("-extent must be either a scalar or a list of length 3");
    for (int i = 0; i < 3; i++) {
      if (!(extent[i] & 1))
        throw Exception("-extent must be a (list of) odd numbers");
      if (extent[i] > data.size(i))
        throw Exception("-extent must not exceed the image dimensions");
    }
  } else {
    uint32_t e = 1;
    while (Math::pow3(e) < data.size(3))
      e += 2;
    extent = {std::min(e, uint32_t(data.size(0))),  //
              std::min(e, uint32_t(data.size(1))),  //
              std::min(e, uint32_t(data.size(2)))}; //
  }
  INFO("selected patch size: " + str(extent[0]) + " x " + str(extent[1]) + " x " + str(extent[2]) + ".");

  if (std::min<uint32_t>(data.size(3), extent[0] * extent[1] * extent[2]) < 15) {
    WARN("The number of volumes or the patch size is small. "
         "This may lead to discretisation effects in the noise level "
         "and cause inconsistent denoising between adjacent voxels.");
  }

  KernelType kernel(extent);
  process_image<T, KernelType>(data, mask, noise, rank, output_name, kernel, exp1);
}

void run() {
  auto dwi = Header::open(argument[0]);

  if (dwi.ndim() != 4 || dwi.size(3) <= 1)
    throw Exception("input image must be 4-dimensional");

  Image<bool> mask;
  auto opt = get_options("mask");
  if (!opt.empty()) {
    mask = Image<bool>::open(opt[0][0]);
    check_dimensions(mask, dwi, 0, 3);
  }

  bool exp1 = get_option_value("estimator", 1) == 0; // default: Exp2 (unbiased estimator)

  Image<real_type> noise;
  opt = get_options("noise");
  if (!opt.empty()) {
    Header header(dwi);
    header.ndim() = 3;
    header.datatype() = DataType::Float32;
    noise = Image<real_type>::create(opt[0][0], header);
  }

  Image<uint16_t> rank;
  opt = get_options("rank");
  if (!opt.empty()) {
    Header header(dwi);
    header.ndim() = 3;
    header.datatype() = DataType::UInt16;
    header.reset_intensity_scaling();
    rank = Image<uint16_t>::create(opt[0][0], header);
  }

  int prec = get_option_value("datatype", 0); // default: single precision
  if (dwi.datatype().is_complex())
    prec += 2; // support complex input data
  switch (prec) {
  case 0:
    INFO("select real float32 for processing");
    make_kernel<float>(dwi, mask, noise, rank, argument[1], exp1);
    break;
  case 1:
    INFO("select real float64 for processing");
    make_kernel<double>(dwi, mask, noise, rank, argument[1], exp1);
    break;
  case 2:
    INFO("select complex float32 for processing");
    make_kernel<cfloat>(dwi, mask, noise, rank, argument[1], exp1);
    break;
  case 3:
    INFO("select complex float64 for processing");
    make_kernel<cdouble>(dwi, mask, noise, rank, argument[1], exp1);
    break;
  }
}
