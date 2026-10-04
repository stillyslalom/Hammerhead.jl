# HammerheadGUI redesign: workflow-first Qt Quick application

Plan for ROADMAP §2. Status: framework chosen (Qt Quick via QML.jl + QMLMakie,
2026-10-03); implementation not started. Delete this file once the redesign has
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

1. **Teardown crash.** Destroying a window that holds a `MakieViewport` during
   `QGuiApplication` shutdown calls into Julia on the Qt render thread and
   crashes in dispatch; Julia's crash handler then runs exit finalizers on that
   thread and self-deadlocks waiting for it (backtrace captured). App: on main
   window close, save state and terminate the process without Qt/GL teardown
   (`TerminateProcess` on Windows rather than `_exit`, which still runs DLL
   detach and prints harmless Qt mutex warnings). Upstream: report to QML.jl
   with the backtrace.
2. **`disconnect_screen` sleeps on the render thread.** QMLMakie's
   `Makie.disconnect_screen` calls `sleep(0.3)`, which deadlocks when Qt calls
   it from the render thread at teardown. App: no-op once item 1 skips
   teardown; upstream PR to QMLMakie removing the sleep.
3. **FluentWinUI3 style DLL not found.** The style plugin's
   `Qt6QuickControls2FluentWinUI3StyleImpl.dll` sits in the Qt Declarative
   artifact's `bin/`, off the DLL search path. App: `Libdl.dlopen` it before
   loading QML (Windows only; other platforms use their native-ish style).
   Upstream: report to the Qt6Declarative_jll recipe.
4. **Callbacks must not block.** While `exec()` runs, every QML→Julia call runs
   on the GUI thread; a blocking call freezes the window (and deadlocks
   anything that needs the message loop, e.g. `PrintWindow`). App: all work
   longer than a frame runs on `Threads.@spawn`; see the threading model.
5. **No `colorbuffer` on QMLMakie screens.** Render checks in tests use QML
   `grabToImage` instead.

Re-check items 1–3 against new QML.jl/QMLMakie releases before each GUI release.

## Architecture

```
HammerheadGUI/
  src/HammerheadGUI.jl        module, launch functions, Qt setup (style DLL, hard exit)
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
1. Qt shell: launch function, the workarounds, step rail, persistent canvas
   with pop-out, recipe open/save, then Images → Passes → Test pair → Run →
   Results for planar. Startup measured against the targets.
2. Prepare sub-pages; retire the GLMakie tool windows and views they replace;
   rewrite `docs/src/howto/gui.md` around the window.
3. Stereo window.
4. Ensemble mode in both windows; session with a lab user (ROADMAP §2).
5. PTV window, after a core PTV recipe design.

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
