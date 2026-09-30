```@meta
CurrentModule = Hammerhead
```

# Hammerhead

[Hammerhead](https://github.com/stillyslalom/Hammerhead.jl) measures particle
motion in images using particle image velocimetry (PIV) and particle tracking
velocimetry (PTV). PIV estimates the displacement of particle patterns in
small image windows; PTV follows individual detected particles. A spatial
calibration and the delay between exposures convert displacement to velocity.

Use planar PIV for two in-plane components, or stereo PIV to reconstruct
three components in a light sheet from two calibrated camera views.
**HammerheadGUI** provides desktop tools for running batches, drawing masks,
and inspecting results. Start with a path that matches your task:

| Task | Start here |
|---|---|
| Understand how images produce a vector field | [Your first vector field](tutorials/first_vector_field.md) |
| Process and assess a recorded image pair | [A real recording](tutorials/real_data.md) and [image inspection](howto/image_quality.md) |
| Work through the desktop interface | [GUI tour](tutorials/gui_tour.md) |
| Analyze a sequence and its fluctuations | [From image pairs to flow statistics](tutorials/sequence_statistics.md) |
| Reconstruct three components with two cameras | [Stereo PIV](tutorials/stereo.md), then [a real stereo recording](tutorials/stereo_real.md) |
| Follow individual particles | [Particle tracking](tutorials/ptv.md) |

## Installation

Use Julia 1.10 or later. Press `]` at the Julia prompt to enter package
mode, then install Hammerhead:

```julia
pkg> add Hammerhead
```

For the optional desktop tools, add the GUI to the same environment:

```julia
pkg> add HammerheadGUI
```

## Quick example

Use an *effort* preset to select a multi-pass analysis schedule.
[`run_piv`](@ref) operates on in-memory image pairs (any equally sized
real-valued matrices); [`load_image`](@ref) loads image files as grayscale
`Matrix{Float64}`:

```julia
using Hammerhead

imgA = load_image("frame_0001.tif")
imgB = load_image("frame_0002.tif")

result = run_piv(imgA, imgB; effort = :high)   # or :low / :medium
```

`effort` picks a full multi-pass schedule sized to the images (see
[Choose an effort level](howto/effort.md)). To choose window sizes and
processing options yourself, pass an explicit schedule:

```julia
# Multi-pass with symmetric image deformation: each pass uses the previous
# validated field as a predictor and shrinks the window.
passes = multipass_parameters([64, 32, 16, 16];
    padding = true,         # zero-padded (linear) correlation
    apodization = :gauss,   # Gaussian window on each interrogation window
    uncertainty = true,     # per-vector uncertainty on the final pass
)
result = run_piv(imgA, imgB, passes)

result.u, result.v    # displacement field (px), u along x/columns
result.x, result.y    # interrogation grid centers (px)
result.outliers       # validation flags
```

The returned `u` and `v` values are displacements in pixels. Positive `u`
points right and positive `v` points down the image. A finite value may be a
replacement for a rejected vector; use `result.outliers` and `result.mask`
to identify accepted measurements. See [validation](howto/validation.md)
for those flags and [physical-unit scaling](howto/scaling.md) for velocity
conversion.

Whole recordings are processed with [`run_piv_sequence`](@ref), which loads
frame pairs (see [`image_pairs`](@ref)), applies optional preprocessing, and
persists results incrementally in the JLD2 Julia data format ([`save_results`](@ref) /
[`load_results`](@ref)).

## Work with your recording

Inspect the images before choosing [window sizes and effort](howto/effort.md).
Use [masks](howto/masking.md) for obscured regions and compare
[preprocessing](howto/preprocessing.md) with raw-image results. Review
[validation flags](howto/validation.md),
[uncertainty estimates](explanation/uncertainty.md), and sensitivity to the
chosen window size before interpreting small flow features.

Once settings work on representative pairs, use [batch processing](howto/batch.md)
for the recording. [Ensemble correlation](howto/ensemble.md) can help when
individual pairs have weak signals and a representative displacement field
is sufficient. [GPU execution](howto/gpu.md) provides another processing
option for supported hardware.

The [coordinate conventions](explanation/conventions.md) explain units,
axis signs, and result locations. For arguments and return types, use the
[pipeline reference](reference/pipeline.md) or the
[GUI reference](reference/gui.md). The [feature matrix](reference/feature_matrix.md)
compares the available analysis paths.
