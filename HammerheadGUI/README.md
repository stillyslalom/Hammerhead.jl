# HammerheadGUI.jl

**HammerheadGUI** provides desktop tools for [Hammerhead.jl](https://github.com/stillyslalom/Hammerhead.jl)
particle image velocimetry (PIV): prepare the images, test the settings on one
pair, run a recording, and inspect the vectors, all in one window.

## Open the planar PIV window

Install Hammerhead and the GUI in the same Julia environment. In Julia 1.10
or later, press `]` to enter package mode:

```julia
pkg> add Hammerhead HammerheadGUI
```

Start Julia with several threads (`julia -t auto`), return to the Julia
prompt with Backspace, and open the window:

```julia
using HammerheadGUI
wf = planar_window()
```

The window needs a graphical session. It walks through one analysis in six
steps, with an image viewer that follows the step and can pop out into its own
window:

- **Images:** add frames and choose how they pair (1–2, 3–4 or 1–2, 2–3);
  pick the representative pair the other steps preview.
- **Prepare:** preprocessing with a raw/processed view and a single-window
  correlation probe; mask polygons drawn on the image; an analysis region;
  and the physical scale, typed or measured from two points.
- **Passes:** low/medium/high presets fill an editable pass table, with the
  window sizes outlined on the particles.
- **Test pair:** run the current settings on the representative pair, exactly
  as the batch will, with a summary of valid vectors and peak ratios.
- **Run:** process every pair in the background, writing results as they
  finish; cancelling keeps the finished pairs.
- **Results:** browse fields, inspect vectors, and measure profiles and
  circulation.

**Save settings…** and **Open settings…** store and read the settings as a
core `PIVRecipe`, and every results file carries the recipe that produced it.
`planar_window` returns its workflow when the window closes; the controllers
in `HammerheadGUI.Controllers` run the same steps from a script.

## Other tools

- **Result explorer:** `result_explorer("results.jld2"; lazy = true)` browses
  planar PIV, stereo PIV, PTV, and particle tracks, including mixed sequences:
  vectors, components, diagnostics, uncertainty, derived quantities such as
  vorticity, and profile and circulation measurements, in physical units when
  a `PhysicalScale` is attached.
- **Stereo workflow:** review dot detection and reprojection errors across
  calibration planes with `calibration_review`, build a shared dewarping grid
  with `stereo_calibration`, and process synchronized camera frames with
  `stereo_batch_runner()`. Use `selfcal_review(report)` to inspect disparity
  maps and the self-calibration report.

See the [GUI tutorial](https://stillyslalom.github.io/Hammerhead.jl/dev/tutorials/gui_tour/)
for a worked example and the [GUI guide](../docs/src/howto/gui.md) for each
step's tasks.

## Development

Outstanding GUI work is tracked in the repository [roadmap](../ROADMAP.md).

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
