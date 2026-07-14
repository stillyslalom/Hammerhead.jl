# Hammerhead

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/dev/)
[![Build Status](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Particle image velocimetry (PIV) in Julia: planar two-dimensional,
two-component (2D2C) and stereoscopic two-dimensional, three-component (2D3C)
PIV, plus 2D2C particle tracking velocimetry (PTV). Hammerhead is developed
and validated against the
[International PIV Challenge](https://pivchallenge.org/) cases.

PIV measures fluid motion from two images of small tracer particles taken a
known time apart. Hammerhead divides the images into small *interrogation
windows*, finds how far each particle pattern moved, and returns a displacement
vector at every window. Supplying the physical pixel size and time between
frames converts those displacements into velocities. PTV is the sparse-seeding
counterpart: it follows identifiable particles instead of comparing patterns
inside windows.

- Multi-pass window-deformation iterative multigrid (WIDIM) analysis with
  symmetric image deformation; zero-padded, Gaussian-apodized correlation
  reaches ~0.03 px root-mean-square (RMS) error on synthetic benchmarks
- Vector validation (universal outlier detection, peak ratio, correlation
  moment) with secondary-peak substitution and local-median replacement
- Per-vector uncertainty quantification from correlation statistics
  ([Wieneke 2015](https://doi.org/10.1088/0957-0233/26/7/074002))
- Ensemble (sum-of-correlation) analysis for low signal-to-noise ratio (SNR)
  and micro-PIV recordings, plus time-series statistics and temporal validation
- Full stereo chain: dot-grid target detection, camera calibration with
  pinhole direct linear transformation (DLT) and Soloff polynomial models,
  image dewarping, three-component (3C) reconstruction, and
  disparity self-calibration
  ([Wieneke 2005](https://doi.org/10.1007/s00348-005-0962-z))
- Particle tracking: subpixel particle detection, hybrid PIV-guided two-frame
  matching, and multi-frame trajectory linking
- Batch drivers with incremental JLD2-format Julia data output, static masking,
  in-place preprocessing, physical-unit metadata, and Makie plotting

New to velocimetry? Start with the executable
[first-vector-field tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/first_vector_field/),
then analyze a
[real wind-tunnel recording](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/real_data/).
The [documentation](https://stillyslalom.github.io/Hammerhead.jl/dev/) also has
task-oriented how-to guides, explanations of the methods, stereo and PTV
tutorials, and the full API reference.

## Planar PIV

`run_piv` operates on in-memory image pairs — any equally sized real-valued
matrices. Load image files with `load_image` (TIFF including 16-bit, PNG, and
anything else FileIO can dispatch):

```julia
using Hammerhead

imgA = load_image("frame_0001.tif")  # grayscale Matrix{Float64} in [0, 1]
imgB = load_image("frame_0002.tif")

# One-line presets trade speed for accuracy (:low, :medium, :high):
result = run_piv(imgA, imgB; effort = :high)

# Or spell out the multi-pass schedule: each pass uses the previous validated
# field as a predictor and shrinks the window; repeat or iterate the final
# size for convergence.
passes = multipass_parameters([64, 32, 16];
    correlation_method = :cross, # or :phase
    padding = true,              # zero-padded (linear) correlation, overlap-normalized
    apodization = :gauss,        # Gaussian window on each interrogation window
    final = (; max_iterations = 3, uncertainty = true),  # iterate the final pass, estimate σ
)
result = run_piv(imgA, imgB, passes)  # threaded over windows when Julia has threads

result.u, result.v                # displacement field (px), u along x/columns
result.x, result.y                # interrogation grid centers (px)
result.peak_ratio                 # per-window quality metric
result.outliers                   # validation flags
result.uncertainty_u              # per-vector σ (Wieneke 2015), final pass
```

The `u` and `v` arrays are displacements in pixels, not yet physical
velocities. Each entry represents the motion measured by one interrogation
window. See [Physical units](#physical-units) when the pixel size and frame
interval are known.

Sign convention: a particle at `(row, col)` in the first image found at
`(row + v, col + u)` in the second yields positive `(u, v)`.

`padding = true` with `apodization = :gauss` is the accuracy configuration
(unbiased, ~0.03 px RMS on synthetic data) at roughly four times the fast
Fourier transform (FFT) cost per window.
Vectors failing validation are first re-tested against their
secondary/tertiary correlation peaks — a locally consistent alternative is
accepted as measured data — and otherwise replaced with the local median and
marked in `result.outliers`. Peak locking can be diagnosed with
`peak_locking(result.u)`.

The pipeline's numeric precision follows the images: loading with
`load_image(Float32, path)` runs the correlators, deformation, and validation
in single precision and returns a `PIVResult{Float32}`.

## Batch processing

Process a whole recording with `run_piv_sequence`, which pairs up frames,
shows progress, and persists results incrementally to a JLD2 file:

```julia
files = sort(readdir("run42"; join = true))
pairs = image_pairs(files)               # (1,2), (3,4), ... ; :chained for time series
results = run_piv_sequence(pairs, passes;
    preprocess = img -> highpass_filter(img; sigma = 3),
    output = "run42_piv.jld2")

results = load_results("run42_piv.jld2") # reload later
```

For low-SNR recordings of stationary flow (micro-PIV), `run_piv_ensemble`
sums the correlation planes across all pairs before peak detection instead of
averaging noisy vector fields. For time-resolved sequences,
`validate_temporal!` runs a per-point median test across time,
`field_statistics` computes pointwise turbulence statistics (mean, RMS,
Reynolds stress), and `power_spectrum` gives temporal spectra.

## Stereo PIV

Calibrate each camera from dot-grid target images at known plate positions,
dewarp both views onto a shared world-plane grid, and reconstruct
three-component vectors:

```julia
grids1 = [detect_calibration_grid(load_image(f)) for f in plate_files_cam1]
cam1   = calibrate_camera(grids1, zs)            # zs: plate positions (e.g. mm)
cam2   = calibrate_camera(grids2, zs)

grid = common_dewarp_grid([cam1, cam2], size(imgA1))
dw1, dw2 = ImageDewarper(cam1, grid), ImageDewarper(cam2, grid)

# Correct sheet/plate misregistration from the particle images themselves:
dw1, dw2, report = self_calibrate(imgA1, imgA2, dw1, dw2)

result = run_piv_stereo(imgA1, imgB1, imgA2, imgB2, dw1, dw2; effort = :high)
result.u, result.v, result.w    # world units per frame interval
```

Per-camera uncertainties propagate through the reconstruction into
`uncertainty_u`/`v`/`w`. See the
[stereo tutorials](https://stillyslalom.github.io/Hammerhead.jl/dev/) for the
end-to-end walkthrough, including one on real PIV Challenge case-4E data.

## Particle tracking velocimetry (PTV)

For seeding densities too sparse for correlation windows, track individual
particles:

```julia
ptv = run_ptv(imgA, imgB)            # detect + PIV-guided matching -> PTVResult
ptv.x, ptv.y, ptv.u, ptv.v           # scattered frame-A positions + displacements

tracks = track_particles(frames)     # multi-frame trajectory linking
gridded = ptv_to_grid(ptv, size(imgA))  # bin tracks back onto a regular grid
```

Detection (`detect_particles`), scattered outlier flagging, and batch
processing (`run_ptv_sequence`) are included; results serialize alongside PIV
results.

## Masking

Exclude model geometry or reflection regions with an image-sized `Bool` mask
(`true` = excluded):

```julia
mask = load_mask("impeller_mask.png")   # or polygon_mask(size(img), vertices), or any Bool array
result = run_piv(imgA, imgB, passes; mask)
result.mask                             # windows with no measurement (NaN vectors)
```

Windows whose masked-pixel fraction reaches `mask_threshold` (default 0.5)
produce no vector; windows below the threshold are correlated over their
valid pixels only, with no intensity step at the mask edge.

## Preprocessing

Each operation has a mutating form that reuses the image buffer, so a chained
pipeline allocates nothing per step — the pattern for batch loops:

```julia
bg = compute_background(images)            # ensemble :min (or :mean) background
img = load_image("frame_0001.tif")         # fresh Float64 buffer, safe to mutate
subtract_background!(img, bg)
intensity_cap!(img)                        # cap at median + 2σ
highpass_filter!(img; sigma = 3)           # remove sheet inhomogeneity
clahe!(img)                                # contrast-limited adaptive equalization
```

Allocating versions (`subtract_background`, `intensity_cap`, `highpass_filter`,
`clahe`) accept any real-valued matrix and return a new `Matrix{Float64}`.

## Physical units

Result arrays stay in measured units (px/frame, or world units for stereo);
attach a `PhysicalScale` and convert on demand:

```julia
scale = PhysicalScale(pixel_size = 22e-6, dt = 1e-3,
                      length_unit = "m", time_unit = "s")
result = run_piv(imgA, imgB, passes; scale)
phys = physical(result)   # positions in m, velocities in m/s; labels for plotting
```

With Unitful loaded (a weak dependency),
`PhysicalScale(22.0u"µm", 1.0u"ms")` carries the unit names into plot labels.

## Graphics processing unit (GPU) and alternative backends

Correlation can run through portable KernelAbstractions kernels:
`backend = :ka` (CPU, built in) is bitwise-checked against the default
engine, and loading a device package enables the matching GPU backend —
`using AMDGPU` for `backend = :amdgpu`, `using CUDA` for `backend = :cuda`.
The GPU backends batch whole passes on the device, including subpixel peak
analysis, and currently cover cross- and phase correlation with
`:gauss3`/`:gauss9` subpixel fits, multi-pass deformation, ensemble
accumulation, and Float64 uncertainty statistics. The `:gauss2d` fit and
retained correlation planes remain CPU-only. See the
[GPU how-to](docs/src/howto/gpu.md) for installation, feature coverage,
device-memory sizing, validation, and performance guidance.

## Visualization

Plotting is provided as a package extension — load any Makie backend first:

```julia
using GLMakie  # or CairoMakie
fig = plot_vector_field(result)              # outliers highlighted in red
plot_vector_field!(ax, result)               # into an existing Axis
```

Result methods label axes in physical units when a `PhysicalScale` is
attached. A desktop GUI (result explorer, polygon mask editor, batch runner,
calibration review) is developed in this repository as the
[`HammerheadGUI/`](HammerheadGUI/) subdirectory package.

The lower-level building blocks (`CrossCorrelator`, `PhaseCorrelator`,
`correlate`, `correlate_deformable`, `warp_image`, `smoothn`,
`error_statistics`, `universal_outlier_detection`, ...) are also exported for
custom pipelines — see the
[API reference](https://stillyslalom.github.io/Hammerhead.jl/dev/).
