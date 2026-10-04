# HammerheadGUI redesign: workflow-first Qt Quick application

Plan for ROADMAP §2. Status: framework chosen (Qt Quick via QML.jl + QMLMakie,
2026-10-03); slices 0–3 done (planar and stereo windows), slice 4's
ensemble runs done, the slice 2/3 designs reviewed and accepted by the user
(2026-10-04), and the stereo core gaps closed (calibration files,
per-camera preprocessing), and slice 5 (PTV, as particle modes of the planar
window) is done; the lab-user session is open. Delete this file once the redesign has
landed and CLAUDE.md describes the result.

## Goal

One main window per modality (planar first, then stereo; PTV later) organized as
steps a user walks through, with every step showing its effect on a
representative pair before a batch runs:

Images → Prepare → Passes → Test pair → Run → Results

Settings are a core `PIVRecipe`: the GUI saves and opens them with
`save_recipe`/`load_recipe` and runs them with `apply_recipe`. It has no private
settings format and calls only public core API.

## Framework decision and evidence

Probes on Windows 11 (Julia 1.11.4, GLMakie 0.13.15, 2048² image + 16,384
vectors, on-screen, 10 pop-out cycles):

| | Qt Quick (QML 0.13.2, QMLMakie 0.3.3) | GTK4 (Gtk4 0.7.14, Gtk4Makie 0.3.9) |
|---|---|---|
| Embedded pan/zoom | ~120 fps; same under 4-thread compute | ~62 fps |
| Pop-out window | ~119 fps | ~64 fps |
| Look on Windows | native title bar, FluentWinUI3 controls | Adwaita; non-native |
| Teardown | crash at quit after a pop-out (workaround below) | clean with care |

Qt wins on polish and frame rate. The earlier Codex QML evaluation ran Qt with
`QT_QPA_PLATFORM=offscreen`, which cannot create a GL context on Windows; its
"embedding failed" conclusion was an artifact. On-screen embedding works.

## Known Qt/QMLMakie issues and how the app handles them

1. **Teardown crash with destroyed canvases.** jlqml's
   `MakieViewport::setup_buffer` connects a lambda capturing `this` to the
   window's `sceneGraphInvalidated` signal with no context object, so the
   connection outlives a destroyed `MakieArea`. At window teardown it calls
   Julia through a dangling item and crashes on Qt's render thread. Julia's
   crash handler then runs exit finalizers on that thread, which
   self-deadlocks. The backtrace was captured. App: never destroy a
   `MakieArea` (no `Loader` around canvases); keep one in the main window
   and one in a persistent pop-out window, and swap figures into them.
   After `exec()` returns, remove the QML screens from `GLMakie.ALL_SCREENS`
   so GLMakie's atexit cleanup skips the destroyed contexts. Verified: 10
   pop-out cycles, then `exec()` returns cleanly and Julia keeps running, so
   a REPL session survives closing the window. Upstream: one-line jlqml fix
   (pass `this` as the connection context).
2. **`disconnect_screen` sleeps on the render thread.** QMLMakie's
   `Makie.disconnect_screen` calls `sleep(0.3)`, which deadlocks when Qt calls
   it from the render thread at teardown. App: override the method without
   the sleep (it only waits for pending zoom/pan actions). Upstream: PR to
   QMLMakie removing it.
3. **FluentWinUI3 style DLL not found.** The style plugin's
   `Qt6QuickControls2FluentWinUI3StyleImpl.dll` sits in the Qt Declarative
   artifact's `bin/`, off the DLL search path. App: `Libdl.dlopen` it before
   loading QML (Windows only; other platforms use their native-ish style).
   Upstream: report to the Qt6Declarative_jll recipe.
4. **Callbacks must not block.** While `exec()` runs, every QML→Julia call runs
   on the GUI thread; a blocking call freezes the window (and deadlocks
   anything that needs the message loop, e.g. `PrintWindow`). App: all work
   longer than a frame runs on `Threads.@spawn`; see the threading model.
5. **Worker threads must reach GC safepoints.** A worker blocked in a plain
   `ccall` (e.g. `Libc.systemsleep`) delays every collection until it returns.
   In the probe, a 0.5 s sleep loop on a worker thread cut panning from
   120 fps to 15 fps. Workers use Julia `sleep`/`wait`, not blocking ccalls.
6. **Qt 6 `quit()` is cancelled if a window rejects its close.** The pop-out
   window turns close into "dock", so it must accept the close while the app
   is quitting.
7. **No `colorbuffer` on QMLMakie screens.** Render checks in tests use QML
   `grabToImage` instead.

Re-check items 1–3 against new QML.jl/QMLMakie releases before each GUI release.

## Architecture

```
HammerheadGUI/
  src/HammerheadGUI.jl        module, launch functions, Qt setup (style DLL, screen cleanup)
  src/controllers/*.jl        framework-free state + logic (Observables only) — kept
  src/canvas/*.jl             Makie figure builders: image, overlays, gestures — from views/
  src/bridge/*.jl             controller ↔ QML adapters (JuliaPropertyMap, @qmlfunction)
  src/qml/Main.qml            window shell: header, step rail, step pages, canvas, status
  src/qml/steps/*.qml         one page per step
  src/qml/components/*.qml    shared widgets (labelled fields with units/validation, pass table)
```

- **Controllers stay the source of truth** and remain testable without Qt or
  GL. The `Controllers` module boundary keeps Qt and Makie out, as it keeps
  Makie out today; a test asserts it.
- **Bridge:** each step exposes a `JuliaPropertyMap` mirroring the controller
  fields QML binds to, plus `@qmlfunction` entry points. Controller observables
  push into the map; QML edits call the entry points, which validate through
  the controller (inline errors come back as map fields). Equality guards on
  both directions, as today.
- **Canvas:** one persistent `MakieArea` in the main window, plus one persistent
  pop-out window created on first use and then only shown or hidden. Each step
  owns one figure, built once and updated through observables. No figure is
  rebuilt per frame or per edit, because each rebuild retained ~70 MiB in both
  probes. Switching steps assigns that step's figure to the canvas. Overlay
  drawing and gestures (mask polygons, ROI box, scale points, probe marker,
  vectors) move from `views/*.jl` into `canvas/*.jl` as plain Makie code.
- **GLMakie stays** as the Makie backend that QMLMakie renders through;
  NativeFileDialog is replaced by QML `FileDialog`/`FolderDialog`.

## Threading model

- `exec()` owns the main thread. QML→Julia calls do only state changes and
  return.
- Test pair, batch runs, background computation, and image loading run on
  `Threads.@spawn`. They report back through controller observables. Writes
  from a worker task reach QML through QML.jl's queued property updates, and
  the canvas redraws on the next frame tick. A QML `Timer` calls
  `MakieArea.update()` when a dirty flag is set.
- Cancellation uses an atomic flag checked by the core drivers' `progress` /
  `cancel` callbacks; the completed prefix stays in the output file.
- `BatchRunner`'s run half (`@async` cooperative loop) becomes a `RunState`
  controller running on a worker task. Its tested contracts are kept:
  cancellation keeps finished pairs, `on_result` drives the live view, and the
  output stores its recipe.

## Startup

Measured warm on this machine: `using GLMakie` 4.7 s, `using QML, QMLMakie`
4.1 s, first figure build 7.6 s (99 % compilation), `using Hammerhead` 2 s.

- Show the window before building any canvas: the shell appears after package
  load, and the canvas fills in when its figure is ready.
- Extend the `PrecompileTools` workload to every step figure's construction,
  plus controller and bridge calls. GL rendering itself cannot run at
  precompile time.
- Target: window visible in < 10 s, first canvas in < 15 s, warm, on this machine.
  Record the numbers in the PR.
- A PackageCompiler app bundle stays the ROADMAP §2 evaluation item for
  near-instant start.

Measured 2026-10-04 (slice 4, warm, `julia -t 4` launch to the first
`hh_tick`, a window opened on four 256² frames): before, planar 20.6 s and
stereo 21.7 s (10.8 s package load; the slice-1 figure was 16.7 s, the
windows grew since). Of the 10 s after loading, 3.2 s rendered ~220 canvas
glyphs into Makie's atlas on every start and 4.6 s was compilation. After
caching the warmed atlas on disk (0.03 s to load) and re-tracing the
precompile statements over both windows' startup and the Qt test scripts
(merged with the old file; 650 lines, a trace omits what is already
precompiled): planar 16.5 s, stereo 16.8 s. The remaining ~4.6 s of
compilation sits behind CxxWrap/QML methods defined at init and closures,
which traced statements do not cache.

## Steps

The full interaction design is in the 2026-10-03 design discussion; summary:

| Step | Page (QML) | Controller | Canvas |
|---|---|---|---|
| Images | file/folder pickers, pairing, pair count, warnings | new `FrameSet` (from `BatchRunner` files/pairing) | frame A / B / blink |
| Prepare | sub-pages Preprocess · Mask · ROI · Scale | `PreprocessPreview` (holds core `PreprocessStep`s), `MaskEditor`, `ROIEditor`, `ScaleTool` | raw/processed, mask, ROI box, scale points, probe |
| Passes | preset buttons fill an editable pass table; shared options; sequence/ensemble | new `PassesEditor` using public `effort_schedule` | window boxes at the probe point |
| Test pair | run button, summary (valid %, peak ratio, σ, max \|d\| vs window/4, time and batch estimate), stale marker | test state, runs `apply_recipe(recipe, [pair])` | result with outliers |
| Run | output path, start/cancel, progress, ETA | `RunState` | latest finished pair |
| Results | browse, derived fields, profile/circulation, export, "use these settings" | `ResultExplorer` | explorer canvas |

A `PlanarWorkflow` controller owns the step controllers and derives the
recipe with `recipe(wf)`. Loading a recipe and calling `recipe(wf)` without
edits returns an `==` recipe, including fields the GUI does not show. The
stereo window adds a Calibration step (two `CalibrationReview`s, the dewarp
grid, and self-calibration) and has no ROI.

## Core changes first

1. Export `effort_schedule`; add it to the reference docs and the CHANGELOG.
   The GUI's `Hammerhead.effort_schedule` call goes away.
2. Add `recipe_preprocess(steps::AbstractVector{PreprocessStep})`, so the
   preview applies exactly the batch's preprocessing. Today the GUI's own
   `_apply_step!` ignores CLAHE `tiles`/`nbins` and local-variance `epsilon`,
   and a loaded recipe cannot populate the preview.

## Slices

0. ✅ Core changes above; the stash `Shared recipe workbench WIP parked for
   experimental QML integration` was dropped (2026-10-03).
1. ✅ Qt shell (2026-10-04): `planar_window`, step rail, persistent canvas
   with pop-out, recipe open/save, Images → Passes → Test pair → Run →
   Results. Found on the way: canvases must create all plots before display
   and only update inputs (no GL context outside Qt's render), so Results got
   a Qt-safe canvas with native controls instead of the GLMakie explorer
   view; GLFW and Qt GL contexts must not share a process (AMD driver crash).
   Startup is 16.7 s to a live window (10.3 s is package loading) — over the
   10 s / 15 s targets; only a sysimage/app bundle removes the load time.
   Moved to slice 2: Prepare editing, the profile and circulation tools in
   Results, async image loading.
2. ✅ Prepare, Results tools, docs (2026-10-04; design accepted). Prepare sub-pages (Preprocess with probe and background
   estimate, Mask, Region, Scale) edit the workflow through `PrepareState`,
   synced both ways with opened settings (design below). Canvas gestures are
   controller functions (`canvas_click!`/`canvas_alt_click!`/`canvas_key!`).
   Pair loading, previews, probes and background estimates run on workers
   under `wf.spawn` with generation counters. Results: inspect / profile /
   circulation on the results canvas; the collapsed profile row hides its
   Axis and legend through scene `visible` (GLMakie skips hidden scenes)
   inside an `Outside`-aligned nested layout, so it takes no space and
   creates no plots. `request_grab(path)` saves window images; the local
   `docs/gui_screenshots.jl` uses it for the committed doc screenshots.
   Retired: the GLMakie batch runner, mask, ROI, scale and preprocess
   windows and the `BatchRunner` controller. Docs: the how-to, tour,
   reference and explanation pages describe the window. Moved on: ensemble
   runs in the Run step (slice 4; the Test step already tests ensembles);
   the narrow Repeats column in the pass table clips its value (seen in the
   screenshots; cosmetic).
3. ✅ Stereo window (2026-10-04; design accepted). Design: "Slice 3
   design" below. 3a ✅ controllers (2026-10-04, overnight, unreviewed):
   `AbstractWorkflow` (workflow.jl) holds the shared settings/test/run/
   results functions with per-workflow hooks; `StereoWorkflow`
   (stereo_workflow.jl) with linked `frames1`/`frames2`, `camera`, and
   `StereoCalibration` (stereo_calibration.jl: plates, detection, fit,
   grid, self-calibration). Found on the way: the Prepare viewer is on the
   dewarped grid on every page (`wf.dewarped` is the raw pair dewarped,
   the preview's `post` dewarps the processed pair), so the probe, mask, and
   viewer share coordinates; background subtraction is unavailable for
   stereo, because a recipe holds one preprocessing list for both cameras
   (core gap: per-camera preprocessing in `PIVRecipe`). 3b ✅ shell + QML
   (2026-10-04, overnight, unreviewed): one `WorkflowShell` for both windows,
   `WorkflowWindow.qml` chrome shared by `PlanarWindow.qml`/`StereoWindow.qml`,
   `StereoCanvas`, `stereo_window()`; the Test/Run viewer shows vectors on the
   *shown* camera's dewarped frame (the camera switch applies on every step),
   and plate images given as paths appear only after the fit (they load in
   the fit job). 3c ✅ retirement and docs (2026-10-04): the GLMakie
   `stereo_batch_runner`/`stereo_calibration` views and the
   `StereoBatchRunner` controller are removed (with `parse_schedule`);
   `build_dewarpers(cr1, cr2)` moved to `calibration_review.jl` as the
   script route into `stereo_window(; dewarpers)`; `calibration_review`,
   `calibration_review!` and `selfcal_review` stay standalone. New how-to
   `howto/gui_stereo.md` with `stereo_*` window screenshots from
   `docs/gui_screenshots.jl` (the synthetic rig of `test/stereo_fixture.jl`).
   Follow-ups (2026-10-04, after review): (a) calibration files — core
   `save_calibration`/`load_calibration` (cameras incl. an applied
   self-calibration, image sizes, grid; stereo results files carry theirs);
   the Calibration step has Open/Save calibration, and opened cameras
   replace the plate fits; (b) per-camera preprocessing — `PIVRecipe`
   preprocessing may be a per-camera tuple (recipe format v2), the Preprocess
   page has **Separate steps per camera** and edits the shown camera's list,
   and **Estimate background** subtracts each camera's own background;
   (c) the Self-calibration page's viewer shows the disparity map of a
   chosen pass (maps are kept by default) on the dewarped frame.
4. Ensemble mode in both windows ✅ (2026-10-04; overnight, unreviewed);
   session with a lab user (ROADMAP §2) — open. Run executes `:ensemble`
   recipes: `apply_recipe` returns one pooled result (saved with its recipe),
   which Results then shows. Core addition: `run_piv_ensemble` /
   `run_piv_stereo_ensemble` take a `progress(done, total)` function ticked
   per pair per pass (stereo counts both cameras), and throwing from it
   aborts — so the Run page reports "pass k of P · j of N pairs" and Cancel
   stops after the pair in flight, keeping no result (said beside the
   button). An ensemble test goes stale when the frames change (not the
   representative pair); the Passes page disables Repeats for ensembles
   (the driver runs each pass once); a planar ensemble with an ROI is
   refused up front. Polish from slices 2–3: the magnitude reads
   `|velocity|` with a scale attached; Qt windows refuse to open next to a
   GLFW screen (and the GLMakie views warn after a Qt window); startup
   re-traced (numbers under Startup).
5. ✅ PTV (2026-10-04). Core first: `PIVRecipe` gained `mode = :ptv |
   :tracking` with `ptv::PTVParameters`, `ptv_predictor` (`:piv` uses the
   recipe's passes as the PIV predictor, `:none`), `min_track_length` and
   `max_gap`; `apply_recipe` runs `run_ptv_sequence` on pairs or
   `track_particles` (new `preprocess` keyword) on a frame sequence — one
   recipe API, no parallel format. The user chose particle *modes of the
   planar window* over a separate `ptv_window()` (same images, preparation
   and scale; one window to compare PIV and PTV). The Passes step's
   **Analysis** choice lists the four modes; in particle modes the rail
   names it **Particles** (`step_label`), the page edits `ParticleSettings`
   (detection, matching, validation, tracks) with the pass table as the
   optional PIV predictor, and the viewer circles the particles detected on
   the processed shown frame (a worker job, `particles.detected`). Test pair:
   PTV matches the representative pair (arrows); tracking follows up to
   `TRACKING_TEST_FRAMES` = 10 frames from it (track polylines). Run: PTV per
   pair like a sequence; tracking follows every listed frame and keeps one
   `TrackingResult` (progress per frame step; cancel keeps nothing). The Qt
   results canvas draws PTV particles (coloured by field, with arrows) and
   tracks (coloured by mean speed). Particle modes take no ROI: the recipe
   omits it and `workflow_problem` blocks test/run while one is set.

## Slice 2 design: canvas gestures and the Prepare step

**Gestures without creating plots.** QMLMakie forwards Qt mouse buttons
(left/right/middle), positions, scroll, and keys into the figure's Makie
`events`, so the canvas uses ordinary Makie interactions; what changes is
only what a gesture may do to the scene.

- One `register_interaction!(ax, :workflow_gesture)` on the image canvas
  (and the existing one on the results canvas) turns a `leftclick` into
  `canvas_click!(wf, x, y)` and a `rightclick` into `canvas_alt_click!(wf)`,
  in axis data coordinates (x = column, y = row). Drags are left to the
  Axis (left-drag zoom box, right-drag pan, scroll zoom), so a click edits
  and a drag navigates. Keys (focus on the canvas): Backspace undoes the last
  vertex, Escape cancels the polygon or pending corner, Delete removes the
  selected polygon. Every action also has a button on the step page.
- `canvas_click!`/`canvas_alt_click!` are controller functions
  (framework-free, tested without GL). They dispatch on the step and the
  Prepare sub-page — Preprocess: place the correlation probe; Mask:
  `click!`/`alt_click!` on the `MaskEditor`; ROI: `click!` on the
  `ROIEditor`; Scale: `click!` on the `ScaleTool`; other steps: not consumed
  (the Axis keeps the event). Results: `click!`/`alt_click!` on the
  `ResultExplorer`, which dispatches on its tool (inspect/profile/circulation).
- The gesture only changes controller observables. The canvas listens to
  them and `update!`s overlay plots that exist from the start, each fed a
  NaN placeholder while empty: committed mask polygons (one NaN-separated
  `lines`, per-vertex colours for holes and the selected polygon), the
  polygon being drawn (`lines` + vertex `scatter`), the ROI box and its
  pending first corner, the scale line with endpoints and its length label,
  and the probe window outline. The mask raster heatmap shows the resulting
  mask. Results: the profile line or circulation contour (`lines` +
  `scatter`), and a profile `Axis` created with the figure in a second
  layout row that is collapsed (row size 0, hidden) outside the profile tool.
  Editing overlays show only on their sub-page; the mask and ROI show on
  every image step, as in slice 1.

**Prepare state.** `PlanarWorkflow` keeps `preprocessing`, `mask`, `roi`,
and `scale` as the single source for `workflow_recipe`. A `PrepareState`
(`wf.prepare`) holds the sub-page and the four editors, built for the
current frame size (editors keep a size, not an image copy). Editor changes
write the workflow fields; workflow changes from outside (opening settings)
reseed the editors — a loaded mask becomes the editor's raster, with new
polygons drawn on top; equality guards stop loops. `PreprocessPreview`
holds an ordered `Vector{PreprocessStep}` (the core type, every option
including CLAHE `tiles`/`nbins` and `epsilon`) and previews with
`recipe_preprocess(steps)`, so the preview is exactly the batch.

**Off the GUI thread.** In a window (`wf.spawn[] = true`), loading the
representative pair, the processed preview, the probe correlation, and the
background estimate run on worker tasks. A request captures its inputs on
the GUI thread, computes from them alone, and `deliver`s the result; a
generation counter drops stale results (latest edit wins). Until a pair
lands, the canvas keeps the previous frame and the rail says "loading…".
Without a window everything runs inline, as the tests expect.

## Slice 3 design: the stereo window

Proposed 2026-10-04 (overnight, unreviewed). `stereo_window()` follows the
planar window and shares its machinery; only the steps that differ are new.

**Steps:** Images → Calibration → Prepare → Passes → Test pair → Run →
Results.

- **Images:** two synchronized frame lists (camera 1, camera 2), one
  pairing rule, one representative pair index for both cameras. The viewer
  shows camera 1 or 2, frame A or B (the pair bar gains a camera switch).
  Problems are reported per camera (different counts or sizes).
- **Calibration:** per camera, plate images with one z each, the detection
  settings `detect_calibration_grid` takes (spacing, two-level and level
  separation, origin offset, invert, orientation) and the model
  (Soloff/pinhole). Detection and fitting run on a worker and produce a
  `CalibrationReview` (existing controller). The viewer shows the selected
  camera and plane with detected dots and reprojection residual arrows
  (overlays created up front). Below: the dewarp grid (`common_dewarp_grid`:
  coverage, spacing `:auto` or a value, z), built on a worker into the two
  `ImageDewarper`s, and **self-calibration** (`self_calibrate` on the first
  pairs, on a worker; its `SelfCalibrationReport` summary per pass; **Apply**
  replaces the dewarpers). Calibration inputs are session state: the core
  has no file format for cameras or dewarpers, so they are not saved with
  the settings (open question for the user). `stereo_window(;
  dewarpers = (dw1, dw2))` accepts dewarpers built in a script.
- **Prepare:** Preprocess (raw frames, applied before dewarping — the
  probe correlates the dewarped pair of the shown camera), Mask (drawn on
  the dewarped grid; the viewer shows the dewarped frame with the cameras'
  out-of-view union shaded), Scale (dt and time unit only; lengths are the
  calibration's world units). No ROI (`apply_recipe` rejects one for stereo).
- **Passes, Test pair, Run, Results:** the planar pages and controllers,
  sized to the dewarped grid; test and run call the stereo
  `apply_recipe(recipe, pairs1, pairs2, dw1, dw2)`; the viewer shows in-plane
  vectors on the dewarped camera-1 frame; Results browses `StereoPIVResult`s
  (w and its uncertainty are fields; profile/circulation stay planar-only).

**Sharing:** the step-independent parts of `PlanarWorkflow` (passes, test,
run, explorer, preprocessing/mask/scale, settings, status, `deliver`/`spawn`)
and of `PlanarShell` (queue, tick, canvas host, step and pass models, the
Passes/Test/Run/Results callbacks) are factored so `StereoWorkflow` and its
shell reuse them rather than copy them; QML shares the window chrome and the
common pages. The GLMakie `stereo_batch_runner`/`stereo_calibration` views
and the `StereoBatchRunner` controller retire once the window covers them;
`calibration_review` and `selfcal_review` stay as standalone windows.

## Tests (proportionate)

- Controller tests as today, plus `PlanarWorkflow` recipe round trip,
  test-pair parity with `apply_recipe`, stale detection, and preset sizing.
- One QML load-and-render smoke test per window where a GL context exists
  (Linux CI under xvfb if Mesa supports what QMLMakie needs; otherwise local
  only, stated in the test). No lifecycle harnesses.

## Open questions

- Linux behavior of QMLMakie embedding: test in slice 1 under WSL2 (Ubuntu
  is installed on the dev machine; WSLg provides X11/Wayland with Mesa GL).
  This also shows whether a Linux CI job under xvfb can run the render smoke
  test.
- macOS behavior: no hardware yet; don't claim support in the docs until
  someone tests it.
- Whether the standalone `result_explorer` and `calibration_review` stay as
  separate entry points (proposed: yes) once their pages exist in the windows.
