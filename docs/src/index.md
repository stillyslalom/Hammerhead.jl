```@meta
CurrentModule = Hammerhead
```

# Hammerhead

Documentation for [Hammerhead](https://github.com/stillyslalom/Hammerhead.jl),
a Julia package for particle image velocimetry (PIV). Hammerhead covers
planar two-dimensional, two-component (2D2C) measurements, stereoscopic
two-dimensional, three-component (2D3C) measurements, and two-frame particle
tracking velocimetry (PTV) with multi-frame trajectory linking. The analysis
core provides multi-pass image deformation, vector validation, per-vector
uncertainty quantification, ensemble correlation, camera calibration with
disparity self-calibration, physical-unit scaling, batch processing with
incremental result files, and optional GPU execution. A companion desktop
package, **HammerheadGUI**, wraps the same API in interactive tools — result
explorer, mask editor, preprocessing preview with a correlation probe, and
batch forms ([take the tour](tutorials/gui_tour.md)).

PIV starts from two images of tracer particles separated by a known time
interval. By finding how the particle pattern moves from the first image to
the second, it estimates a displacement vector at each location in the image.
Supplying the physical pixel size and time interval turns those displacements
into velocities.

## Installation

```julia
pkg> add Hammerhead        # `]` at the julia> prompt opens pkg>
pkg> add HammerheadGUI     # optional: the desktop GUI tools
```

## Quick example

The simplest complete analysis is one call with an *effort* preset.
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
[Choose an effort level](howto/effort.md)). When your data needs specific
knobs, pass an explicit schedule instead:

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

The returned `u` and `v` values are displacements, not yet physical velocities.
See [Scale results to physical units](howto/scaling.md) when your pixel size and
frame interval are known.

Whole recordings are processed with [`run_piv_sequence`](@ref), which loads
frame pairs (see [`image_pairs`](@ref)), applies optional preprocessing, and
persists results incrementally in the JLD2 Julia data format ([`save_results`](@ref) /
[`load_results`](@ref)).

## Where to go next

The documentation follows the [Diátaxis](https://diataxis.fr/) structure:

- **Tutorials** — guided, executable walkthroughs. Start with
  [Your first vector field](tutorials/first_vector_field.md) or the
  point-and-click path, [a tour of the GUI](tutorials/gui_tour.md); continue
  with a [real wind-tunnel recording](tutorials/real_data.md), then
  [stereo PIV end to end](tutorials/stereo.md),
  [stereo on a real recording](tutorials/stereo_real.md), and
  [particle tracking velocimetry](tutorials/ptv.md).
- **How-to guides** — recipes for specific jobs:
  [masking](howto/masking.md), [preprocessing](howto/preprocessing.md),
  [effort selection](howto/effort.md),
  [physical-unit scaling](howto/scaling.md),
  [GPU execution](howto/gpu.md),
  [validation tuning](howto/validation.md),
  [ensemble correlation](howto/ensemble.md),
  [batch processing](howto/batch.md),
  [working with the GUI](howto/gui.md), and
  [calibrating a real stereo rig](howto/stereo_rig.md).
- **Explanation** — how and why Hammerhead works the way it does:
  [coordinate conventions](explanation/conventions.md),
  [correlation accuracy](explanation/correlation.md),
  [multi-pass deformation](explanation/multipass.md),
  [the masking model](explanation/masking.md),
  [uncertainty quantification](explanation/uncertainty.md),
  [stereo geometry and self-calibration](explanation/stereo.md),
  [the GUI's controller–view split](explanation/gui.md),
  the [numeric precision policy](explanation/precision.md), and the
  [compatibility policy](explanation/compatibility.md).
- **Reference** — the API, one page per topic, starting at
  [Core pipeline and parameters](reference/pipeline.md); the GUI's API is
  under [GUI (HammerheadGUI)](reference/gui.md).
