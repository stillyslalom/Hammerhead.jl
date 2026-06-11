# Hammerhead

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/dev/)
[![Build Status](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Particle image velocimetry (PIV) in Julia.

## Usage

`run_piv` operates on in-memory image pairs (any equally sized real-valued
matrices — load image files with your preferred image package):

```julia
using Hammerhead

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

`padding = true` with `apodization = :gauss` is the most accurate configuration
(unbiased, ~0.03 px RMS on synthetic data) at ~4× the FFT cost per window.
Vectors failing validation (universal outlier detection, optional
`min_peak_ratio`) are replaced with the local median and marked in
`result.outliers`. On a synthetic linear shear (±4 px), the multi-pass schedule
above achieves ~0.04 px RMS where direct 16 px correlation gives ~0.14 px.

## Preprocessing

```julia
bg = compute_background(images)            # ensemble :min (or :mean) background
img = subtract_background(raw, bg)
img = intensity_cap(img)                   # cap at median + 2σ
img = highpass_filter(img; sigma = 3)      # remove sheet inhomogeneity
img = clahe(img)                           # contrast-limited adaptive equalization
```

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
