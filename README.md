# Hammerhead

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/dev/)
[![Build Status](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Particle image velocimetry (PIV) in Julia.

## Usage

`run_piv` operates on in-memory image pairs — any equally sized real-valued
matrices. Load image files with `load_image` (TIFF including 16-bit, PNG, and
anything else FileIO can dispatch):

```julia
using Hammerhead

imgA = load_image("frame_0001.tif")  # grayscale Matrix{Float64} in [0, 1]
imgB = load_image("frame_0002.tif")

# Multi-pass with symmetric image deformation: each pass uses the previous
# validated field as a predictor and shrinks the window. Repeat the final
# size for convergence sweeps.
passes = multipass_parameters([64, 32, 16, 16];
    correlation_method = :cross, # or :phase
    padding = true,              # zero-padded (linear) correlation, overlap-normalized
    apodization = :gauss,        # Gaussian window on each interrogation window
)
result = run_piv(imgA, imgB, passes)  # threaded over windows when Julia has threads

# Single-pass: run_piv(imgA, imgB, PIVParameters(window_size = 32, overlap = 16))

result.u, result.v            # displacement field (px), u along x/columns
result.x, result.y            # interrogation grid centers (px)
result.peak_ratio             # per-window quality metric
result.outliers               # universal outlier detection mask
```

Sign convention: a particle at `(row, col)` in the first image found at
`(row + v, col + u)` in the second yields positive `(u, v)`.

The pipeline's numeric precision follows the images: loading with
`load_image(Float32, path)` (or `run_piv_sequence(...; image_type = Float32)`)
runs the correlators, image deformation, and validation in single precision
and returns a `PIVResult{Float32}` — half the memory traffic, and the natural
precision for an eventual GPU port. Integer or `Float64` inputs use `Float64`.

`padding = true` with `apodization = :gauss` is the most accurate configuration
(unbiased, ~0.03 px RMS on synthetic data) at ~4× the FFT cost per window.
Vectors failing validation (universal outlier detection, optional
`min_peak_ratio`) are replaced with the local median and marked in
`result.outliers`. On a synthetic linear shear (±4 px), the multi-pass schedule
above achieves ~0.04 px RMS where direct 16 px correlation gives ~0.14 px.

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

## Visualization

Plotting is provided as a package extension — load any Makie backend first:

```julia
using GLMakie  # or CairoMakie
fig = plot_vector_field(result)              # outliers highlighted in red
plot_vector_field!(ax, result)               # into an existing Axis
plot_vector_field(x, y, u, v; kwargs...)     # raw arrays
```

The lower-level building blocks ([`CrossCorrelator`](https://stillyslalom.github.io/Hammerhead.jl/dev/),
`PhaseCorrelator`, `correlate`, `correlate_deformable`, `warp_image`,
`calculate_manual_registration`, `universal_outlier_detection`, ...) are also
exported for custom pipelines.
