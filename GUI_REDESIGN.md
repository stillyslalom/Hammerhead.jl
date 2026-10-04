# HammerheadGUI redesign: workflow-first Qt Quick application

Plan for ROADMAP §2. Status: framework chosen (Qt Quick via QML.jl + QMLMakie,
2026-10-03); slices 0–2 done (planar window with Prepare and Results tools),
stereo window next. Delete this file once the redesign has
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
2. ✅ Prepare, Results tools, docs (2026-10-04; implemented overnight,
   unreviewed). Prepare sub-pages (Preprocess with probe and background
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
3. Stereo window.
4. Ensemble mode in both windows; session with a lab user (ROADMAP §2).
5. PTV window, after a core PTV recipe design.

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
