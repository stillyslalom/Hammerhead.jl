# HammerheadGUI.jl

Desktop GUI for [Hammerhead.jl](https://github.com/stillyslalom/Hammerhead.jl)
(particle image velocimetry), built on GLMakie + NativeFileDialog.

Lives in the Hammerhead monorepo as a subdirectory package (Makie-style):
the core package stays at the repo root and never gains GUI dependencies;
this package is where the GLMakie hard dependency lives.

## Status

Phase 7 in the repo [ROADMAP](../ROADMAP.md), in progress.

- **Result explorer** (done) — `result_explorer(results_or_path)` opens a
  read-only viewer for all four persisted result types (`PIVResult`,
  `StereoPIVResult`, `PTVResult`, `TrackingResult`), mixed sequences
  included: gridded scalar-field heatmap (magnitude, components,
  diagnostics, uncertainty) with a fast quiver-style vector overlay, PTV
  particle scatter with displacement arrows, tracking polylines colored by
  mean speed (gap-aware), robust percentile color limits with manual
  override, frame scrubbing, click-to-inspect, live appending
  (`push_result!`) while a batch runs, and physical-unit labels when a
  `PhysicalScale` is attached; planar results add the derived fields
  (vorticity, divergence, strain rate, swirling strength, Q, unit-labelled)
  and interactive profile/circulation tools
- **Mask editor** (done) — `mask_editor(image_or_path)` draws exclusion
  polygons over the image (left-click add/select, right-click close);
  `polygon_mask(editor)` exports the package mask convention and
  "save mask…" writes a mask image `load_mask` reads back
- **Parameter form + batch runner** (done) — `batch_runner()` picks frames,
  edits the multi-pass `PIVParameters` schedule (or an effort preset), sets
  an optional physical scale and preprocessing pipeline, runs
  `run_piv_sequence` with live progress, cancellation, incremental JLD2
  output, and a "view results" hand-off that opens mid-run and follows the
  batch live
- **Preprocessing preview** (done) — `preprocess_preview(image_or_path)`
  composes the core preprocessing set into an ordered, toggleable pipeline
  with a live raw/processed comparison and a single-window correlation
  probe (click a location; du/dv/peak-ratio recompute as steps change);
  `build_preprocess` exports the batch-driver closure (frame-copying,
  snapshot semantics)
- **Scale tool** (done) — `scale_tool(image_or_path)` derives a
  `PhysicalScale` from a two-point calibration line of known separation;
  `apply_scale!` hands it into a batch form
- **Stereo batch** (done) — `stereo_calibration(cr1, cr2)` builds the
  dewarper pair from two fitted calibration reviews (embedded side by side
  via `calibration_review!`); `stereo_batch_runner()` runs
  `run_piv_stereo_sequence` over two synchronized frame lists with native
  between-acquisition cancellation, a dt-only scale, incremental output,
  and the live explorer hand-off
- **Fast startup** (done) — a PrecompileTools workload brings
  time-to-first-window to ~1 s after loading; a PackageCompiler app bundle
  for non-Julia users is still to be evaluated
- **Calibration & self-calibration diagnostics** (done) —
  `calibration_review(images, zs; spacing, …)` reviews grid detection and
  reprojection errors plane by plane; `selfcal_review(report)` summarizes a
  `SelfCalibrationReport` and browses its disparity maps

## Architecture rule

All application state and logic live in a framework-free controller layer
(plain Julia + Observables). Makie code renders controllers and pushes user
input into them, but controllers never depend on Makie, so the logic is
testable without a GL context and the widget shell stays swappable.

## Development

On Julia ≥ 1.11 the `[sources]` entry in `Project.toml` couples this package
to the sibling core checkout automatically:

```
julia --project=HammerheadGUI -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

On 1.10, `Pkg.develop(path="..")` into the GUI environment first (CI does the
equivalent). Releases go core-first, then a GUI compat bump; registration
uses `subdir=HammerheadGUI` and TagBot tags releases as `HammerheadGUI-v*`.
