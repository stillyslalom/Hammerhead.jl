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

params = PIVParameters(
    window_size = 32,            # interrogation window (px)
    overlap = 16,                # 50% overlap
    correlation_method = :cross, # or :phase
    padding = true,              # zero-padded (linear) correlation, overlap-normalized
    apodization = :gauss,        # Gaussian window on each interrogation window
    deformation_iterations = 3,  # iterative window deformation
)
result = run_piv(imgA, imgB, params)  # threaded over windows when Julia has threads

result.u, result.v            # displacement field (px), u along x/columns
result.x, result.y            # interrogation grid centers (px)
result.peak_ratio             # per-window quality metric
result.outliers               # universal outlier detection mask
```

Sign convention: a particle at `(row, col)` in the first image found at
`(row + v, col + u)` in the second yields positive `(u, v)`.

`padding = true` with `apodization = :gauss` is the most accurate configuration
(unbiased, ~0.03 px RMS on synthetic data) at ~4× the FFT cost per window.

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
