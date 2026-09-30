# HammerheadGUI.jl

**HammerheadGUI** provides desktop tools for [Hammerhead.jl](https://github.com/stillyslalom/Hammerhead.jl)
particle image velocimetry (PIV): run an analysis, inspect vectors, and adjust
masks or preprocessing while viewing the images.

## Open the batch form

Install Hammerhead and the GUI in the same Julia environment. In Julia 1.10
or later, press `]` to enter package mode:

```julia
pkg> add Hammerhead HammerheadGUI
```

Return to the Julia prompt with Backspace, then open the form:

```julia
using HammerheadGUI
display(batch_runner())
```

The desktop views use GLMakie and require a graphical session.

Start with `batch_runner()` to select frames, choose an effort preset or a
multi-pass schedule, and run planar PIV. The form shows progress, supports
cancellation between pairs, and can save results incrementally in JLD2
format. Open "view results" as soon as the first pair completes; the explorer
adds later results as the batch runs. You can also open saved results with
`result_explorer("results.jld2")`.

Choose the pairing mode to match the acquisition: disjoint pairs for
double-frame recordings, adjacent frames for a uniformly sampled sequence.
Check the first few pairs, including dim or fast-moving regions, before
running the whole recording. Set the physical pixel size and paired-exposure
delay when you need velocity units; a visually plausible vector field alone
does not establish measurement quality.

## Tools

- **Result explorer:** Browse planar PIV, stereo PIV, PTV, and particle tracks,
  including mixed sequences. Inspect vectors, components, diagnostics, and
  uncertainty; scrub through frames; and adjust color limits. Planar fields
  also offer derived quantities such as vorticity and interactive profile and
  circulation measurements. A `PhysicalScale` supplies physical-unit labels.
- **Mask editor:** Draw and edit exclusion polygons over an image with
  `mask_editor(image_or_path)`. Export the mask with `polygon_mask(editor)` or
  save an image that `load_mask` can read.
- **Preprocessing preview:** Use `preprocess_preview(image_or_path)` to compare
  raw and processed images. Supply the paired frame, then click a location
  to inspect its single-window displacement and correlation peak ratio as
  you adjust processing steps.
- **Scale tool:** Use `scale_tool(image_or_path)` to measure a feature of known
  length. Enter the paired-exposure delay and use
  `scale_tool(image_or_path; batch = controller)` to enable **apply to batch**.
  From code, `apply_scale!(controller, scale_controller)` applies the same scale.
- **Stereo workflow:** Review dot detection and reprojection errors across
  calibration planes with `calibration_review`. Build a shared dewarping grid
  with `stereo_calibration`, then process synchronized camera frames with
  `stereo_batch_runner()`. Use `selfcal_review(report)` to inspect disparity
  maps and the self-calibration report.

To automate repeated work, use the Julia controllers for the same operations.
They store selections and settings without requiring a window. See the
[GUI tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/gui_tour/)
for a worked example and the [GUI guide](../docs/src/howto/gui.md) for task recipes.

## Development

On Julia ≥ 1.11 the `[sources]` entry in `Project.toml` couples this package
to the sibling core checkout automatically:

```
julia --project=HammerheadGUI -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

On Julia 1.10, explicitly develop the core path from the repository root:

```bash
julia --project=HammerheadGUI -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.test()'
```

Releases go core-first, then a GUI compat bump; registration uses
`subdir=HammerheadGUI` and TagBot tags releases as `HammerheadGUI-v*`.
