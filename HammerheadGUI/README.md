# HammerheadGUI.jl

**HammerheadGUI** provides desktop tools for [Hammerhead.jl](https://github.com/stillyslalom/Hammerhead.jl)
particle image velocimetry (PIV). It uses GLMakie and NativeFileDialog and
lives in the repository's `HammerheadGUI/` subdirectory.

Start with `batch_runner()` to select frames, choose an effort preset or a
multi-pass schedule, and run planar PIV. The form shows progress, supports
cancellation between pairs, and can save results incrementally in JLD2
format. Open "view results" as soon as the first pair completes; the explorer
adds later results as the batch runs. You can also open saved results with
`result_explorer(results_or_path)`.

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
  length and attach its `PhysicalScale` to a batch with `apply_scale!`.
- **Stereo workflow:** Review dot detection and reprojection errors across
  calibration planes with `calibration_review`. Build a shared dewarping grid
  with `stereo_calibration`, then process synchronized camera frames with
  `stereo_batch_runner()`. Use `selfcal_review(report)` to inspect disparity
  maps and the self-calibration report.

Each window has a Julia controller for its state and actions. The view sends
user input to that controller, so the same operations are available from code
without opening a window. See the
[GUI tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/gui_tour/)
for a worked example and the [GUI guide](../docs/src/howto/gui.md) for task recipes.

## Development

On Julia ≥ 1.11 the `[sources]` entry in `Project.toml` couples this package
to the sibling core checkout automatically:

```
julia --project=HammerheadGUI -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

On 1.10, `Pkg.develop(path="..")` into the GUI environment first (CI does the
equivalent). Releases go core-first, then a GUI compat bump; registration
uses `subdir=HammerheadGUI` and TagBot tags releases as `HammerheadGUI-v*`.
