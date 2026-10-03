# Hammerhead.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/stable/)
[![Development docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/dev/)
[![Build status](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

Hammerhead measures motion from images of tracer particles. Use planar
particle image velocimetry (PIV) for two displacement components in a light
sheet, stereo PIV for three components from two calibrated cameras, or
particle tracking velocimetry (PTV) when particles are sparse enough to
follow individually. Results include validation flags and measurement
diagnostics so you can inspect where the images support the field.

## Install

Use Julia 1.10 or later. In Julia's package mode (`]`):

```julia
pkg> add Hammerhead
```

For the optional desktop application, install `HammerheadGUI` in the same
environment:

```julia
pkg> add HammerheadGUI
```

## Measure one image pair

Press Backspace to return from package mode to the Julia prompt.
Replace the filenames with two successive exposures from your recording.
[`load_image`](https://stillyslalom.github.io/Hammerhead.jl/dev/reference/io/)
reads grayscale images into floating-point matrices; `run_piv` returns
displacements in pixels between the exposures.

```julia
using Hammerhead

imgA = load_image("frame_0001.tif")
imgB = load_image("frame_0002.tif")
result = run_piv(imgA, imgB; effort = :high)

accepted = .!(result.mask .| result.outliers)
(accepted_vectors = count(accepted), total_vectors = length(accepted))
```

`result.x` and `result.y` are interrogation-window centers in pixels.
`result.u` points along image columns and `result.v` along rows; positive
`v` points down the image. Masked windows hold no measurement. An outlier
can still hold a finite replacement value, so use `accepted` when computing
statistics from measured vectors. `result.peak_ratio` compares the two
strongest correlation peaks; `result.uncertainty_u` and
`result.uncertainty_v` estimate random correlation error when enabled by the
chosen analysis settings. These diagnostics help locate weak measurements.
Correlation uncertainty does not include errors in calibration or timing.

If you know the physical pixel size and the time between the *two exposures*,
convert the result to velocity. The values below are examples; use the
calibration and timing measured for your setup.

```julia
scale = PhysicalScale(pixel_size = 0.02, dt = 0.001,
                      length_unit = "mm", time_unit = "s")
velocity = physical(result, scale)
velocity.u[accepted]  # accepted horizontal velocities in mm/s
```

`result` remains in pixels. In `velocity`, positions are in millimetres,
and displacements and their uncertainty estimates are in millimetres per
second. The interval between *successive image pairs* is a separate value
used for time-series analysis.

To plot a field, install a Makie backend such as `CairoMakie`
(`pkg> add CairoMakie`), then load it before calling the plotting extension:

```julia
using CairoMakie
plot_vector_field(result; show_replaced = false)
```

The [first-vector-field tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/first_vector_field/)
explains the correlation and validation steps. The
[real-recording tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/real_data/)
shows how to check image quality, window-size sensitivity, and uncertainty
when the displacement field is unknown.

## Choose a workflow

| If you need to… | Start here |
|---|---|
| Process many pairs and save results as they finish | [Batch processing](https://stillyslalom.github.io/Hammerhead.jl/dev/howto/batch/) |
| Calculate means, fluctuations, and accepted-sample counts | [From image pairs to flow statistics](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/sequence_statistics/) |
| Combine weak correlations across a stationary recording | [Ensemble correlation](https://stillyslalom.github.io/Hammerhead.jl/dev/howto/ensemble/) |
| Calibrate two cameras and reconstruct three components | [Stereo tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/stereo/) and [real stereo recording](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/stereo_real/) |
| Follow individual particles instead of window patterns | [PTV tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/ptv/) |
| Draw masks, run batches, and explore results in a desktop app | [GUI tour](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/gui_tour/) and [HammerheadGUI](HammerheadGUI/) |

Hammerhead also provides preprocessing, physical-unit scaling, temporal
validation, derived flow quantities, and optional GPU execution for
supported PIV settings. The [documentation](https://stillyslalom.github.io/Hammerhead.jl/dev/)
has task guides and the [API reference](https://stillyslalom.github.io/Hammerhead.jl/dev/reference/pipeline/).

Processing settings can be saved as a recipe and applied to new recordings;
see [Save settings and reuse them](docs/src/howto/recipes.md). Development
priorities are tracked in [ROADMAP.md](ROADMAP.md) and API changes in
[CHANGELOG.md](CHANGELOG.md).
