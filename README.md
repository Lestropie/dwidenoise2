# `dwidenoise2`

This is a reworked implementation of Marchenko-Pastur (MP)
Principal Components Analysis (PCA) based denoising of >3D MRI data,
building upon the "`dwidenoise`" command in *MRtrix3*.

It integrates many technical developments in the domain
since the original derivation of this method and its implementation in *MRtrix3*
(see "enhancements" section below).

### Citation

The primary scientific citation for utilising MP-PCA for MRI data denoising is
[Veraart et al. (2016a)](#veraart2016a).
For performing noise level estimation one should also cite
[Veraart et al. (2016b)](#veraart2016b).

Further references relating to specific feature augmentations
are cited by name throughout the "technical enhancements" section below,
with full bibliographic details---and the circumstances under which each applies---
collated in the [References](#references) section at the bottom of this document.
The same conditional citation list is reproduced in the help page of each command.

### Permissions

`dwidenoise2` is distributed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
Commercial utilisation of the MP-PCA method is restricted by the following patent:

<a id="patent"></a>
US10698065B2
System, method and computer accessible medium for noise estimation, noise removal and gibbs ringing removal.
Dmitry Novikov, Jelle Veraart, Els Fieremans.
Contact: https://tov.med.nyu.edu/about/contact-us/

Within `dwidenoise2`, the scope of this patent is understood to encompass
the `exp1` and `exp2` noise level estimators (see "Data-driven noise level estimation" below).

### Demonstration

From top to bottom: Empirical data; MRtrix3 `dwidenoise`; `dwidenoise2`

<img src=images/anim.gif width="592">

Demonstration data:
-   Siemens Prisma Fit 3T
-   1.8mm isotropic, multi-band factor 4, SENSE1+ multi-coil combination (CMRR sequence)
-   *b* = 0 (8), 300 (11), 1600 (26), 5000 (64) (only *b*=5000 volumes shown)
-   Gradient table split between A>>P and P>>A phase encoding directions (only A>>P volumes shown)
-   Denoising applied to complex data
-   Runtimes on Dell Latitude 5531: `dwidenoise` 287s; `dwidenoise2` 181s

## Usage

Currently the simplest way to utilise the software is through a container.

The container itself can be built using eg.:

```ShellSession
docker build . -t dwidenoise2:latest
```

Within this container, the two most relevant commands are `dwidenoise2` and `dwi2noise`;
a limited subset of *MRtrix3* core commands are also compiled in the container
due to their utility in converting the image data that are input / output for these commands.

### Default usage

```ShellSession
docker run -it --rm -v $(pwd):/data dwidenoise2:latest \
    dwidenoise2 ...
```

Note that despite the Docker image being named "`dwidenoise2`",
it is still necessary to specify that it is the command named "`dwidenoise2`"
within the constructed container that is to be executed;
this is because of the container providing several other commands also.

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

### Debugging

If a particular dataset proves to be problematic for the implementation,
a request may be made to re-run the dataset utilising the debugging version of the Docker image.
This is achieved as follows:

```ShellSession
docker build . -f Dockerfile_debug -t dwidenoise2:debug
docker run -it --rm -v $(pwd):/data dwidenoise2:debug \
    ...
```

Unlike the default usage above,
the "`dwidenoise2`" command does not need to be explicitly specified here;
for specifically the debugging container,
that command is the hard-coded entrypoint.
What will appear in the terminal is the interface to the GNU Debugging Tool (`gdb`).
First, hit "`r`" then Enter to commence running the program
(note that command execution will progress more slowly than the standard container).
If the command encounters some problem during execution,
type "`bt`" then Enter to generate the backtrace.
The resulting data can then be provided to the developer.

## Technical enhancements

The following is a list of technological enhancements present in the `dwidenoise2` command
over and above the capabilities of the `dwidenoise` command in *MRtrix3*:

### Bidirectional Divide and Conquer Singular Value Decomposition (BDC-SVD)

Both *MRtrix3* `dwidenoise` and `dwidenoise2` here use the Eigen C++ library
for linear algebra calculations, including singular value decomposition for PCA denoising.
Where *MRtrix3* `dwidenoise` uses the `SelfAdjointEigenSolver` class,
`dwidenoise2` uses the newer `BDCSVD` class made available in Eigen 3.4.0,
which is slower but more numerically precise.
Where the number of volumes is very large,
the increased computational expense of `BDCSVD` can become prohibitive;
the original self-adjoint decomposition can be selected in that case
using the `-decomposition selfadjoint` option,
which is typically around twice as fast but not as numerically precise.

### Complex data demodulation

Retaining complex data exported by the scanner sequence for utilisation in complex denoising
can yield substantial improvements in noise floor rectification.
The strong dephasing that arises from the interaction between strong diffusion sensitisation gradients
and microscopic subject motion can however introduce phase decoherence between volumes.
This can be detrimental to denoising efficacy as it makes the signal less sparse.
In `dwidenoise2` complex input data can be explicitly demodulated prior to PCA;
the method of phase estimation is selected using the `-demodulate` option.
Three approaches are available:

-   `apc` (the default): *noise-adaptive* phase correction.
    The smooth background phase is re-estimated at every noise level estimation iteration
    directly from the empirical complex data
    through a noise-weighted total-variation smoothing,
    the strength of which is driven by the current noise level map estimate.
    The first iteration, which has no noise map as yet,
    self-calibrates from a data-derived global noise level with uniform spatial weighting.

-   `hann`: a fixed smooth *nonlinear* phase map derived once
    through *k*-space filtering with a Hann window.

-   `linear`: regression of a strictly *linear* phase term from each *k*-space.

The default noise-adaptive `apc` approach follows
[Pizzolato et al. (2020)](#pizzolato2020).
The *linear* phase demodulation approach is similar to that shown in
[Cordero-Grande et al. (2019)](#cordero-grande2019).
Inclusion of *non-linear* phase demodulation was motivated by description in
[Patron et al. (2024)](#patron2024).

### Optimal shrinkage

*MRtrix3* `dwidenoise` achieves denoising through a hard truncation of singular values.
`dwidenoise2` instead uses optimal shrinkage of singular values based on minimisation of the Frobenius norm.
This was first demonstrated for denoising of diffusion MRI in
[Cordero-Grande et al. (2019)](#cordero-grande2019).

The manner in which eigenvector contributions are filtered based on the estimated noise level
can be selected using the `-filter` option.
In addition to the default optimal shrinkage (`optshrink`),
the hard truncation of the original `dwidenoise` command is available (`truncate`;
this may be preferable for functional MRI, as it minimises the risk of attenuating
BOLD signal fluctuations near the noise floor),
as is the optimal hard threshold of [Gavish & Donoho (2014)](#gavish2014) (`optthresh`).

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
This was first shown in the denoising of diffusion MRI in
[Manjon et al. (2013)](#manjon2013).

### Sliding window kernel shape

By default, a *spherical* rather than *cuboid* kernel is used.
This provides better guarantees on equal noise level of all samples within each patch as,
compared to a cuboid kernel with the same number of voxels,
the maximal distance of any voxel to the centre of the patch is reduced.
The kernel is isotropic in realspace, and therefore suitably accounts for anisotropic voxels.

The shape and size of the sliding-window kernel are configured entirely through the `kernel`
column of the (per-iteration) schedule; there are no command-line options for the kernel.
A bespoke kernel (e.g. a fixed radius `radius=<mm>`, a fixed voxel count `voxels=<count>`,
or a `cuboid`) is obtained by authoring a schedule file rather than passing command-line flags
(see the schedule file format in the command help).

This was first shown for diffusion MRI denoising in
[Cordero-Grande et al. (2019)](#cordero-grande2019).

### Sliding window dynamic sizing

The spatial size of the PCA kernel patch can be dynamically altered,
driven by the following two mechanisms:

1.  For patches near the edge of the image FoV,
    the patch under default behaviour is dynamically increased in radius
    in order to have approximately the same number of voxels within that patch
    as would a patch in the middle of the image.

2.  The kernel is resized in such a way
    that the number of voxels in the patch
    should be approximately the number of volumes plus the signal rank;
    in this way the Casorati matrix should consist of a concatenation
    of an approximately square noise block
    and those columns constituting the signal of interest.

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

This was first demonstrated in [Cordero-Grande et al. (2019)](#cordero-grande2019).

### Volume partitioning and eigenspectrum aggregation

For a series with a very large number of volumes *m*
(for instance long and/or multi-echo functional MRI acquisitions,
which may comprise thousands of volumes),
the cost of the PCA of each patch grows steeply with *m*,
and the sliding-window patch must contain a commensurately large number of voxels
in order to preserve the Casorati matrix aspect ratio,
compromising the spatial specificity of denoising.

To make computation feasible in this regime,
the *m* volumes of each PCA patch can be split into *P* disjoint partitions,
with an independent eigendecomposition performed for each partition
and the resulting eigenspectra pooled ("aggregated") to yield
a single noise level estimate for that patch.
Because each partition holds only *m*/*P* volumes,
the spatial patch can be shrunk in proportion
(the number of voxels need only preserve the aspect ratio of the *smaller* per-partition matrices);
the two effects together reduce the PCA cost of each patch
by a factor of approximately *P²*.
The assignment of volumes to partitions is balanced across the demeaning groups
(*b*-value shells or volume groups)
so that every partition retains a comparable matrix aspect ratio and group composition.

This behaviour is configured through the `partitions` (or `max_partition_size`) column
of the (per-iteration) schedule;
the bundled "`vlarge`" schedule (see example usages) enables it by default.

### Data-driven noise level estimation

Where a noise level map is not provided a priori,
`dwidenoise2` estimates it from the data
through classification of the PCA eigenspectrum into signal and noise components.
Several estimators are available through the `-estimator` option:

-   `mrm2023` (the default) generalises the Marchenko-Pastur fit
    to the eigenspectrum of multi-dimensional data,
    as introduced in [Olesen et al. (2023)](#olesen2023).

-   `tbme2022` is a multiple-moment generalised-quarter-circle estimator,
    developed with functional MRI in mind,
    as introduced in [Zhu et al. (2022)](#zhu2022).

-   `med` estimates the noise level from the median eigenvalue,
    as in [Gavish & Donoho (2014)](#gavish2014).

-   `exp1` and `exp2` implement the Marchenko-Pastur threshold search
    of, respectively, the original `dwidenoise` command
    ([Veraart et al., 2016a](#veraart2016a))
    and its refinement in [Cordero-Grande et al. (2019)](#cordero-grande2019);
    note that these two estimators are subject to the [patent](#patent) noted above.

### Variance-stabilising transform

PCA-based denoising assumes that the noise is additive, Gaussian-distributed and homoscedastic
(of equal variance) across all samples within each patch.
Both of these assumptions may be violated by real data.

Firstly, the noise level may vary spatially,
for instance where B1- bias field correction is applied by the scanning hardware
to data acquired with a high-density receive array.
Where a noise level map is available,
the voxel data are explicitly scaled by the local noise level prior to PCA,
so that the stabilised data are of approximately unit variance everywhere.
The noise level map may be one provided a priori by the user
(via the `-noise_in` option),
or one estimated at a previous iteration (see below).

Secondly, magnitude-reconstructed MRI data do not have Gaussian noise:
the noise follows a Rician distribution for a single receive channel,
or a non-central chi distribution for multiple channels combined by sum-of-squares reconstruction.
This deviation from the Gaussian model is most pronounced for data close to the noise floor.
For magnitude input data `dwidenoise2` therefore applies a *non-linear*, noise-model-aware
variance-stabilising transform (VST) prior to PCA,
which renders the data approximately Gaussian and homoscedastic across the full intensity range.
The number of receive channels *N* (such that the noise has 2*N* degrees of freedom)
is specified using the `-noise_dof` option (default 1, i.e. Rician),
and the form of the transform is selected using the `-vst_method` option.

The default transform (`-vst_method foi`) uses the exact-unbiased forward transform and inverse of
[Foi (2011)](#foi2011).
Alternative inverses are available based on the analytically exact scheme of
[Koay & Basser (2006)](#koay2006) (`-vst_method koay`),
or a closed-form method of moments (`-vst_method mom`).

The combination of a variance-stabilising transform with singular-value-based denoising
of magnitude diffusion MRI data was demonstrated in [Ma et al. (2020)](#ma2020).

#### Removal of the noise-floor bias

For magnitude data the non-central chi distribution is biased:
its expectation exceeds the true underlying signal level,
increasingly so as that level approaches the noise floor.
By default, `dwidenoise2` removes this bias from the output data.
Rather than simply undoing the forward transform algebraically,
the inverse is evaluated at the *exact-unbiased* operating point,
mapping each denoised sample back to the bias-free underlying signal level.
Because this inverse is evaluated per sample
(rather than being linearised about a per-volume-group mean),
the debiasing is independent of the choice of demeaning grouping.
This suppresses the characteristic residual "haze"
that otherwise appears in data denoised close to the noise floor,
while returning the denoised signal fluctuations on the original intensity scale.
This behaviour can be disabled using the `-preserve_noise_bias` option,
in which case conventional biased-magnitude-scale output is produced.
For complex (or phase-demodulated) input data the noise is Gaussian and there is no such bias,
so the transform reduces to a simple linear scaling and these options have no effect.

### Multi-resolution iterative noise map refinement

Where an a priori noise map estimate is not provided,
`dwidenoise2` uses a novel iterative multi-resolution *pyramid* to derive the estimated noise level
prior to denoising.
The key principle is that robust *estimation* of the noise level
and spatially specific *denoising* of the data
have opposing requirements on the size of the PCA patch,
and are therefore best served by different patch sizes at different stages:

-   The early iterations perform noise level estimation only.
    They use aggressive spatial subsampling
    (so that comparatively few PCAs are performed, keeping them fast)
    together with large, rank-naive patches (by default of the order of twice as many voxels as volumes).
    Such patches yield numerically stable, well-posed decompositions
    with a well-populated Marchenko-Pastur noise bulk,
    from which the noise level can be estimated reliably;
    the result is a smooth, low-resolution noise map.

-   Each subsequent iteration re-estimates the noise map at a progressively finer spatial resolution,
    using the noise map from the previous iteration to drive the variance-stabilising transform.
    The patch size in these iterations is controlled so as to reach a target precision
    of the noise level estimate.
    The map produced by the penultimate iteration is additionally smoothed.

-   In the final iteration, denoising of the input data is performed
    using the noise map from the penultimate iteration without further re-estimation.
    This reconstruction pass instead uses a small, rank-adaptive local patch
    (sized such that the number of voxels approximates the number of volumes plus the signal rank),
    which maximises the spatial specificity of the denoised output.

In this way the well-posed large matrices required for stable noise level estimation
and the small local patches required for spatially specific denoising
are each employed where they are most appropriate.

The `dwi2noise` command performs this same multi-resolution estimation strategy,
but omits the final data denoising step;
its primary output is instead the final estimated noise map.
This permits noise level estimation and denoising to be performed as two separate steps
(the noise map and the associated per-voxel signal-rank density exported by `dwi2noise`
being passed to `dwidenoise2`),
reproducing the result of a single `dwidenoise2` invocation (see example usages).

The entire schedule of iterations---the spatial and temporal subsampling,
the patch sizing rule, the volume partitioning and the noise map smoothing at each iteration---is
fully user-configurable through a schedule file supplied via the `-schedule` option.
Several schedules are bundled with the software,
including "`default`" (the built-in behaviour described above),
"`vlarge`" (a lighter schedule for very large series),
and "`legacy`" (approximating the behaviour of the original `dwidenoise` command).

## Acknowledgments

RS is supported by fellowship funding from the National Imaging Facility (NIF),
an Australian Government National Collaborative Research Infrastructure Strategy (NCRIS) capability.

The Florey Institute of Neuroscience and Mental Health
acknowledges the strong support from the Victorian Government and,
in particular,
the funding from the Operational Infrastructure Support Grant.

## References

Listed alphabetically by first author.
Each entry states the circumstances under which that citation is applicable;
the same conditions govern the reference lists printed by the
`dwidenoise2` and `dwi2noise` command help pages.

<a id="cordero-grande2019"></a>
**Cordero-Grande et al. (2019)**
L. Cordero-Grande, D. Christiaens, J. Hutter, A.N. Price, J.V. Hajnal.
Complex diffusion-weighted image estimation via matrix recovery under general noise models.
NeuroImage 2019:200;391--404.
*Applicable to*: all default usage of both `dwidenoise2` and `dwi2noise`
(spherical [kernel shape](#sliding-window-kernel-shape),
[subsampling](#subsampling),
and, for `dwidenoise2`, [optimal shrinkage](#optimal-shrinkage));
additionally `-demodulate linear` and `-estimator exp2`.

<a id="foi2011"></a>
**Foi (2011)**
A. Foi.
Noise estimation and removal in MR imaging: The variance-stabilization approach.
IEEE International Symposium on Biomedical Imaging (ISBI) 2011;1809--1814.
*Applicable to*: magnitude (non-complex) input data processed with
`-vst_method foi` (the default), in either command.

<a id="gavish2014"></a>
**Gavish & Donoho (2014)**
M. Gavish, D.L. Donoho.
The Optimal Hard Threshold for Singular Values is 4/sqrt(3).
IEEE Transactions on Information Theory 2014:60(8);5040--5053.
*Applicable to*: `-estimator med` (either command),
and `-filter optthresh` (`dwidenoise2` only).

<a id="koay2006"></a>
**Koay & Basser (2006)**
C.G. Koay, P.J. Basser.
Analytically exact correction scheme for signal extraction from noisy magnitude MR signals.
Journal of Magnetic Resonance 2006:179(2);317--322.
*Applicable to*: magnitude (non-complex) input data processed with
`-vst_method koay`, in either command.

<a id="ma2020"></a>
**Ma et al. (2020)**
X. Ma, K. Ugurbil, X. Wu.
Denoise magnitude diffusion magnetic resonance images via variance-stabilizing transformation
and optimal singular-value manipulation.
NeuroImage 2020:215;116852.
*Applicable to*: magnitude (non-complex) input data denoised with any non-linear
variance-stabilising transform (`-vst_method foi` / `koay` / `mom`), in either command.

<a id="manjon2013"></a>
**Manjon et al. (2013)**
J.V. Manjon, P. Coupe, L. Concha, A. Buades, D.L. Collins, M. Robles.
Diffusion Weighted Image Denoising Using Overcomplete Local PCA.
PLoS ONE 2013:8(9);e73021.
*Applicable to*: `dwidenoise2` with any aggregator other than `-aggregator exclusive`
(i.e. all default usage); not applicable to `dwi2noise`, which performs no reconstruction.

<a id="olesen2023"></a>
**Olesen et al. (2023)**
J.L. Olesen, A. Ianus, L. Ostergaard, N. Shemesh, S.N. Jespersen.
Tensor denoising of multidimensional MRI data.
Magnetic Resonance in Medicine 2023:89(3);1160--1172.
*Applicable to*: `-estimator mrm2023` (the default) in either command.

<a id="patron2024"></a>
**Patron et al. (2024)**
J.P.M. Patron, S. Moeller, J.L.R. Andersson, K. Ugurbil, E. Yacoub, S.N. Sotiropoulos.
Denoising diffusion MRI: Considerations and implications for analysis.
Imaging Neuroscience 2024:2;00060.
*Applicable to*: complex input data processed with `-demodulate hann`,
in either command.

<a id="pizzolato2020"></a>
**Pizzolato et al. (2020)**
M. Pizzolato, G. Gilbert, J.-P. Thiran, M. Descoteaux, R. Deriche.
Adaptive phase correction of diffusion-weighted images.
NeuroImage 2020:206;116274.
*Applicable to*: complex input data processed with `-demodulate apc` (the default),
in either command.

<a id="veraart2016a"></a>
**Veraart et al. (2016a)**
J. Veraart, D. Novikov, D. Christiaens, B. Ades-aron, J. Sijbers, E. Fieremans.
Denoising of diffusion MRI using random matrix theory.
NeuroImage 2016:142;394--406.
*Applicable to*: all usage of `dwidenoise2` (the primary MP-PCA denoising citation);
additionally `-estimator exp1` in either command.

<a id="veraart2016b"></a>
**Veraart et al. (2016b)**
J. Veraart, E. Fieremans, D.S. Novikov.
Diffusion MRI noise mapping using random matrix theory.
Magnetic Resonance in Medicine 2016:76(5);1582--1593.
*Applicable to*: all usage of either command in which the noise level is estimated from the data
(i.e. all usage of `dwi2noise`, and all usage of `dwidenoise2` other than with `-noise_in`).

<a id="zhu2022"></a>
**Zhu et al. (2022)**
W. Zhu, X. Ma, X.-H. Zhu, K. Ugurbil, W. Chen, X. Wu.
Denoise Functional Magnetic Resonance Imaging With Random Matrix Theory Based Principal Component Analysis.
IEEE Transactions on Biomedical Engineering 2022:69(11);3377--3388.
*Applicable to*: `-estimator tbme2022` in either command.

The MP-PCA method is additionally encumbered by [patent US10698065B2](#patent)
(see "Permissions" above),
the scope of which is understood to encompass the `exp1` and `exp2` noise level estimators.
