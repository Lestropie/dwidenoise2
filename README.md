# `dwidenoise2`

This is a reworked implementation of Marchenko-Pastur (MP)
Principal Components Analysis (PCA) based denoising of >3D MRI data,
building upon the "`dwidenoise`" command in *MRtrix3*.

It integrates many technical developments in the domain
since the original derivation of this method and its implementation in *MRtrix3*
(see "enhancements" section below).

## Usage

Currently the simplest way to utilise the software is through a container.

The container itself can be built using eg.:

```ShellSession
docker build . -t dwidenoise2:latest
```

Within this container, the two most relevant commands are `dwidenoise2` and `dwi2noise`;
a limited subset of *MRtrix3* core commands are also compiled in the container
due to their utility in converting the image data that are input / output for these commands.

### `dwidenoise2` vs. `dwi2noise`

These two commands are very similar in function and operation.
The key difference is:

-   For `dwidenoise2`, the second compulsory positional command-line argument
    (ie. subsequent to the input image)
    is the denoised version of the input image series;
    the estimated noise map image can be *optionally* exported
    using the `-noise_out` option.

-   For `dwi2noise`, the second compulsory positional command-line argument
    is the estimated noise map image.
    No denoised version of the input image data can be produced.

### Denoising complex data

Both `dwidenoise2` and `dwi2noise` are capable of operating on complex data.
It is however necessary for the *singular* input image to be of data type *complex floating-point*.
This contrasts with typical scanner reconstructions that export image data
in the form of two distinct DICOIM series encoding magnitude and phase.
Further, a phase image may not be in the units of radians;
for instance, on Siemens platforms it is common for phase data to lie in the numerical range [-4096, +4094].
The following example shows how to combine magnitude and phase image series where this scaling applies
to form a complex image series for denoising:

```ShellSession
docker run -it --rm -v $(pwd):/data dwidenoise2:latest bash -c \
    "mrcalc /data/DICOM_Mag/ /data/DICOM_Phase/ pi 4096 -div -mult -polar - | \
    dwidenoise2 - ... "
```

### Denoising multi-echo data (eg. multi-echo fMRI)

Multi-echo fMRI data naturally form a 5D dataset,
as for each TR there is some fixed number of echoes acquired.
It is preferable to explicitly present such data as a 5D dataset to `dwidenoise2` / `dwi2noise`,
as demeaning will then be applied to each echo individually,
improving the efficacy of data preconditioning.
The following is an example of how data of such form may be processed,
based on the individual echoes being stored in individual NIfTI images
according the the Brain Imaging Data Structure (BIDS) specification:

```ShellSession
docker run -it --rm -v $(pwd):/data dwidenoise2:latest bash -c \
    "mrcat sub-01/func/sub-01_task-rest_echo-*_bold.nii.gz -axis 4 - | \
    dwidenoise2 - ... "
```

The output denoised image series can then, if necessary,
be split back into a 4D image series per echo using one of two approaches:

```ShellSession
docker run -it --rm -v $(pwd):/data dwidenoise2:latest bash -c \
    "mrconvert denoised.mif denoised_echo1.nii -coord 4 0 -axes 0,1,2,3 && \
    mrconvert denoised.mif denoised_echo2.nii -coord 4 1 -axes 0,1,2,3 && \
    ... "
```

Or:

```ShellSession
docker run -it --rm -v $(pwd):/data dwidenoise2:latest \
    dwidenoise2 ... denoised_echo[].nii
```

This "multi-file numbered image" format will split the 5D image along the final axis
across multiple 4D image files, numbering them consecutively from 0.

## Technical enhancements

The following is a list of technological enhancements present in the `dwidenoise2` command
over and above the capabilities of the `dwidenoise` command in *MRtrix3*:

### Bidirectional Divide and Conquer Singular Value Decomposition (BDC-SVD)

Both *MRtrix3* `dwidenoise` and `dwidenoise2` here use the Eigen C++ library
for linear algebra calculations, including singular value decomposition for PCA denoising.
Where *MRtrix3* `dwidenoise` uses the `SelfAdjointEigenSolver` class,
`dwidenoise2` uses the newer `BDCSVD` class made available in Eigen 3.4.0,
which is slower but more numerically precise.

### Complex data demodulation

Retaining complex data exported by the scanner sequence for utilisation in complex denoising
can yield substantial improvements in noise floor rectification.
The strong dephasing that arises from the interaction between strong diffusion sensitisation gradients
and microscopic subject motion can however introduce phase decoherence between volumes.
This can be detrimental to denoising efficacy as it makes the signal less sparse.
In `dwidenoise2` complex input data can be explicitly demodulated prior to PCA.
By default a smooth *nonlinear* phase map for demodulation is derived
through *k*-space filtering with a Hann window.

### Optimal shrinkage

*MRtrix3* `dwidenoise` achieves denoising through a hard truncation of singular values.
`dwidenoise2` instead uses optimal shrinkage of singular values based on minimisation of the Frobenius norm.

### Overcomplete local PCA
For each output image voxel, 
*MRtrix3* `dwidenoise` computes the denoised version of the data for that voxel
through truncation of the PCA where that voxel was at the centre of the kernel.
`dwidenoise2` instead reconstructs the denoised data for each output voxel
through a weighted combination of the denoised versions of all PCA patches
of which that voxel was a member.
By default the contribution of each PCA patch to that output image voxel
is weighted based on a Gaussian distribution on the distance between the voxel
and the centre of the patch.

### Sliding window kernel shape

By default, a *spherical* rather than *cuboid* kernel is used.
This provides better guarantees on equal noise level of all samples within each patch as,
compared to a cuboid kernel with the same number of voxels,
the maximal distance of any voxel to the centre of the patch is reduced.
The kernel is isotropic in realspace, and therefore suitably accounts for anisotropic voxels.

For patches near the edge of the image FoV,
the patch is dynamically increased in radius in order to have approximately
the same number of voxels within that patch as a patch in the middle of the image.

### Demeaning

-   For multi-shell DWI data, the mean intensity per *b*-value shell is regressed from the data
    prior to PCA.

-   For multi-echo fMRI data, where echoes are concatenated across the fifth image axis,
    the mean intensity per echo is regressed from the data prior to PCA.
        This reduces the rank of the signal and better exposes the distribution of noise components.

### Subsampling

The number of PCAs performed can be smaller than the number of image voxels.
By default, in the final step of denoising, all spatial axes are subsampled by a factor of two,
such that the number of PCAs is approximately 1/8 the number of voxels.
Where subsampling is performed by an even factor,
the PCA kernel is centred in between input image voxels
in order to reduce biases in denoising arising from different voxels having different
distances to the kernels to which it contributes.

### Variance-stabilising transform

PCA decomposition of any given patch of voxels assumes that the noise level
is equivalent for all voxels within that patch.
This may not be precisely correct in some circumstances,
for instance if B1- bias field correction is applied by the scanning hardware
to data acquired with a high-density receive array.
Where a pre-determined noise level map is available,
the voxel data are explicitly scaled to unit variance prior to PCA.
The noise level map can come from a pre-estimated noise level image
provided to the `dwidenoise` command by the user,
or by that estimated from a previous iteration (see below).

### Multi-resolution iterative noise map refinement

Where an input a priori noise map estimate is not provided,
`dwidenoise2` uses an iterative approach to derive the estimated noise level prior to denoising.
Initially, a low-resolution noise map is estimated assuming homoscedasticity (equal noise level everywhere).
The noise map is subsequently re-estimated at a higher spatial resolution,
with the noise map estimate from the previous iteration utilised by the variance-stabilising transform.
In the final iteration, when denoising of the input data is finally performed,
the noise map estimate from the last iteration is utilised without re-estimation.

The `dwi2noise` command performs this same multi-resolution estimation strategy,
but omits the final data denoising step;
its primary output is instead the final estimated noise map.
