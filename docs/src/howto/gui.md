# Work interactively with the graphical user interface (GUI)

**Goal:** do the common interactive jobs — browse results, draw a mask,
run a batch, review a calibration — with the HammerheadGUI tools, and get
their output back into scripted analyses. For a guided walkthrough with
figures, start with [the GUI tour](../tutorials/gui_tour.md).

The GUI ships as the separate `HammerheadGUI` package (the core stays free
of GL dependencies); `using HammerheadGUI` loads GLMakie, and each call
below opens a window.

## Explore a results file

[`result_explorer`](@ref) takes anything [`load_results`](@ref) produces —
planar, stereo, PTV, and tracking entries, mixed files included — or
in-memory results:

```julia
using HammerheadGUI

result_explorer("run_042.jld2")
result_explorer(result)              # a PIVResult / StereoPIVResult
result_explorer(ptv_result)          # a PTVResult / TrackingResult
result_explorer([r1, r2, r3])        # a sequence — the slider scrubs frames
```

For gridded results the field menu lists displacement magnitude, components,
the validation diagnostics, and per-vector σ when the analysis ran with
`uncertainty = true`; click any vector to inspect its numbers. A
[`PTVResult`](@ref) is drawn as a colored particle scatter with optional
displacement arrows (flagged particles in red), and a
[`TrackingResult`](@ref) as trajectory polylines colored by mean speed, with
breaks at bridged frame gaps.

When a result carries a [`PhysicalScale`](@ref) — attached at analysis time
or with [`with_scale`](@ref) — the explorer displays in physical units:
axis labels, the colorbar, and the inspection panel all read `mm`, `mm/s`,
and so on instead of `px` / `px/frame`.

## Draw a mask and use it

```julia
mask_editor("frame_0001.tif")
```

Left-click adds vertices (inside an existing polygon it selects instead),
right-click closes the polygon, Backspace undoes a vertex, Delete removes
the selected polygon. "Save mask…" writes the white-=-excluded image that
[`load_mask`](@ref) reads back:

```julia
mask = load_mask("mask.png")
result = run_piv(imgA, imgB, passes; mask)
```

To skip the file, hold the [`MaskEditor`](@ref) controller and export
directly — and seed it with existing polygons to resume editing:

```julia
me = MaskEditor(load_image("frame_0001.tif"))
mask_editor(me)                      # draw…
mask = polygon_mask(me)              # image-sized Bool, true = excluded

me = MaskEditor(img; polygons = [[(120, 40), (480, 90), (450, 300)]])
```

## Run a batch from the form

```julia
batch_runner()
```

"Add frames…" picks the image files, the menu chooses paired (`1-2, 3-4`)
or chained (`1-2, 2-3`) pairing, the windows textbox takes a schedule like
`64, 32, 32`, and "choose output…" sets an incremental JLD2 file (written
pair by pair; read it with [`load_results`](@ref)). "Cancel" stops after
the pair in flight and keeps every finished pair. "Explore results" opens
the batch in the result explorer.

The *effort* menu switches between the manual schedule (`:custom`) and
[`run_piv_sequence`](@ref)'s `:low` / `:medium` / `:high` presets — when a
preset is active the manual schedule is ignored (the summary says so). The
*physical scale* group (pixel size, dt, and unit labels) attaches a
[`PhysicalScale`](@ref) to every output when any field is non-default, so the
batch results carry units straight into the explorer. From code:

```julia
set_effort!(bc, :high)
set_scale!(bc; pixel_size = 50.0, dt = 0.001, length_unit = "mm", time_unit = "s")
```

Seed the form from code with a [`BatchRunner`](@ref) — every form field is
an observable on the controller:

```julia
bc = BatchRunner(files = readdir("run42"; join = true), uncertainty = true)
bc.output_path[] = "run_042.jld2"
batch_runner(bc)
```

[`start!`](@ref)`(bc; async = false)` runs the same batch without the
window at all.

## Review a calibration

[`calibration_review`](@ref) shows each plate image with its detected dots
colored by reprojection error (fiducial markers outlined), a plane slider,
and a camera-model menu that refits on switch. The keyword arguments are
[`detect_calibration_grid`](@ref)'s:

```julia
calibration_review(plate_images, zs; spacing = 15.0, two_level = true,
                   level_separation = 3.0, origin_offset = (30.0, 7.5))
```

For judging the numbers it shows — what residuals a physical plate
produces and when to worry — see
[Calibrate a real stereo rig](stereo_rig.md).

After stereo self-calibration, [`selfcal_review`](@ref) browses the
report; keep the disparity maps to inspect them pass by pass in an
embedded explorer:

```julia
dw1c, dw2c, report = self_calibrate(frames1, frames2, dw1, dw2;
                                    keep_disparity_maps = true)
selfcal_review(report)
```

## Embed a view in your own figure

[`result_explorer!`](@ref) builds the explorer into a `GridPosition`, for
composing dashboards — it is the embeddable form the composite views use
internally:

```julia
using GLMakie

fig = Figure(size = (1400, 700))
result_explorer!(fig[1, 1], ResultExplorer(results))
ax = Axis(fig[1, 2])  # your own plots alongside
```

Because the view renders a controller, several views (or your own code)
can share one `ResultExplorer` and stay in sync through its observables.
