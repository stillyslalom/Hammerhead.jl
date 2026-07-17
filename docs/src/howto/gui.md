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

The colorbar defaults to a robust 2–98% percentile range over the valid
(non-masked, non-flagged) vectors ([`color_limits`](@ref)), so outliers
cannot wash out the display. The "color range" group switches to the full
extrema or pins either bound; manual bounds persist across frame/field
switches until cleared:

```julia
set_color_mode!(ex, :full)             # extrema instead of percentiles
set_color_limits!(ex; min = 0, max = 5)
set_color_limits!(ex; max = "auto")    # clear one bound
```

For planar results the field menu also carries the derived fields —
vorticity, divergence, strain rate `|S|`, swirling strength, and Q — via
[`flow_derivatives`](@ref), computed once per frame and labelled `1/s`
(`1/s²` for Q) when a scale is attached. The *tool* menu adds interactive
analysis on the same results:

```julia
set_tool!(ex, :profile)      # two clicks sample u/v/|V| along a line
set_tool!(ex, :circulation)  # click a contour, right-click to close:
                             # line-integral + vorticity-area circulation
tool_summary(ex)             # the live numbers, with units
set_tool!(ex, :inspect)      # back to click-to-inspect
```

Tool state clears when the frame changes, and the analysis tools revert to
`:inspect` on result types without derived analysis (stereo/PTV/tracking).

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
the pair in flight and keeps every finished pair. "View results" activates
as soon as the first pair completes: it opens the finished prefix in the
result explorer and appends later pairs live, so you can inspect a long
batch while it runs (the mechanism is the `completed` observable plus
[`push_result!`](@ref) — available for your own live consumers too, and
scripted runs get the same stream through `run_piv_sequence`'s `on_result`
callback).

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

Instead of typing the pixel size, derive it from an image with the scale
tool — click the two endpoints of a feature of known physical size (a
ruler, or two dots of a calibration plate) and apply the result to the
form:

```julia
st = ScaleTool("calibration_plate.tif")
scale_tool(st; batch = bc)     # click two points, enter the separation…
apply_scale!(bc, st)           # …or do the hand-off from code
```

"Preprocess…" opens a [`preprocess_preview`](@ref) on the first frame: an
ordered, toggleable pipeline over the core preprocessing set (background
subtraction, intensity cap, highpass, CLAHE, percentile stretch, inversion,
local-variance normalization) with a live raw/processed comparison; "use in
batch" installs it. From code, build the pipeline yourself:

```julia
pp = PreprocessPreview(first_frame; enabled = [:highpass_filter, :clahe])
set_step_param!(pp, :highpass_filter, :sigma, 5)
set_preprocess!(bc, pp)        # snapshot: later edits don't affect the run
```

The exported closure copies each frame before its in-place steps, so
in-memory arrays are never mutated.

Give the preview the frame's correlation partner and it gains a
single-window probe: click the processed image and that window's
displacement and peak ratio recompute live with every pipeline change —
judge a preprocessing choice by what it does to the correlation, before
committing to a batch:

```julia
pp = PreprocessPreview(frameA; pair = frameB)   # the batch pop-out does this
set_probe_window!(pp, 48)
probe_summary(pp)      # "du = …, dv = … px, peak ratio = …" after a click
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

## Run a stereo batch

[`stereo_calibration`](@ref) sets the rig up: both cameras'
[`CalibrationReview`](@ref)s embedded side by side, the dewarp-grid options
(coverage, spacing), and a "build dewarpers" button running
[`build_dewarpers`](@ref) over the two fitted cameras:

```julia
cr1 = CalibrationReview(plates_cam1, zs; spacing = 15.0, ...)
cr2 = CalibrationReview(plates_cam2, zs; spacing = 15.0, ...)
sbc = StereoBatchRunner()
stereo_calibration(cr1, cr2; batch = sbc)   # …or from code:
set_dewarpers!(sbc, cr1, cr2)               # build_dewarpers + install
```

Dewarpers you built at the REPL (e.g. after [`self_calibrate`](@ref)) go
straight in with `set_dewarpers!(sbc, dw1, dw2)`. Then
[`stereo_batch_runner`](@ref) is the stereo form: each camera's frame list,
the effort/schedule form, a dt-only physical scale (stereo results are
already in world units), and incremental output —
[`run_piv_stereo_sequence`](@ref) underneath. Cancellation uses the
driver's native between-acquisition predicate, so the completed prefix
always lands in `results`, and "view results" opens the
[`StereoPIVResult`](@ref)s in the explorer live, exactly like the planar
batch:

```julia
add_files!(sbc, cam1_paths; camera = 1)
add_files!(sbc, cam2_paths; camera = 2)
set_effort!(sbc, :medium)
set_dt!(sbc, 0.001); sbc.time_unit[] = "s"; sbc.length_unit[] = "mm"
stereo_batch_runner(sbc)     # or start!(sbc; async = false) headless
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
