```@meta
CurrentModule = Hammerhead
```

# Hammerhead

Documentation for [Hammerhead](https://github.com/stillyslalom/Hammerhead.jl),
a Julia package for particle image velocimetry (PIV): planar (2D2C) and
stereoscopic (2D3C) analysis with multi-pass image deformation, vector
validation, per-vector uncertainty quantification, ensemble correlation,
camera calibration, and disparity self-calibration.

## Quick example

[`run_piv`](@ref) operates on in-memory image pairs (any equally sized
real-valued matrices); [`load_image`](@ref) loads image files as grayscale
`Matrix{Float64}`:

```julia
using Hammerhead

imgA = load_image("frame_0001.tif")
imgB = load_image("frame_0002.tif")

# Multi-pass with symmetric image deformation: each pass uses the previous
# validated field as a predictor and shrinks the window.
passes = multipass_parameters([64, 32, 16, 16];
    padding = true,         # zero-padded (linear) correlation
    apodization = :gauss,   # Gaussian window on each interrogation window
)
result = run_piv(imgA, imgB, passes)

result.u, result.v    # displacement field (px), u along x/columns
result.x, result.y    # interrogation grid centers (px)
result.outliers       # validation flags
```

Whole recordings are processed with [`run_piv_sequence`](@ref), which loads
frame pairs (see [`image_pairs`](@ref)), applies optional preprocessing, and
persists results incrementally to JLD2 ([`save_results`](@ref) /
[`load_results`](@ref)).

## Where to go next

The documentation follows the [Diátaxis](https://diataxis.fr/) structure:

- **Tutorials** — guided, executable walkthroughs. Start with
  [Your first vector field](tutorials/first_vector_field.md), continue with
  a [real wind-tunnel recording](tutorials/real_data.md), then
  [Stereo PIV end to end](tutorials/stereo.md).
- **How-to guides** — recipes for specific jobs:
  [masking](howto/masking.md), [preprocessing](howto/preprocessing.md),
  [effort selection](howto/effort.md),
  [GPU execution](howto/gpu.md),
  [validation tuning](howto/validation.md),
  [ensemble correlation](howto/ensemble.md),
  [batch processing](howto/batch.md), and
  [calibrating a real stereo rig](howto/stereo_rig.md).
- **Explanation** — how and why Hammerhead works the way it does:
  [coordinate conventions](explanation/conventions.md),
  [correlation accuracy](explanation/correlation.md),
  [multi-pass deformation](explanation/multipass.md),
  [the masking model](explanation/masking.md),
  [uncertainty quantification](explanation/uncertainty.md),
  [stereo geometry and self-calibration](explanation/stereo.md), and the
  [numeric precision policy](explanation/precision.md).
- **Reference** — the API, one page per topic, starting at
  [Core pipeline and parameters](reference/pipeline.md).
