# Release notes

## Unreleased

### Core

Added:

- `effort_schedule` is exported: it returns the pass schedule behind
  `effort = :low/:medium/:high` for inspection or as a starting point.
- `recipe_preprocess(steps)` builds the preprocessing function from a vector
  of `PreprocessStep`s, so previews apply exactly what `apply_recipe` runs.

### HammerheadGUI

Added:

- `planar_window()`: a Qt Quick window for planar PIV that walks through
  Images → Prepare → Passes → Test pair → Run → Results, with one image
  viewer that follows the step and can pop out into its own window. Settings
  save and open as core `PIVRecipe` files (results files carry theirs); the
  test pair runs exactly the batch's `apply_recipe` call; runs write results
  as they finish and can be cancelled. Controllers (`PlanarWorkflow`,
  `FrameSet`, `PassesEditor`, `PairTest`, `RunState`) work without a window.
- Prepare step in the window, with four pages: **Preprocess** (ordered core
  `PreprocessStep`s with every option editable, a raw/processed view, a
  background estimate, and a single-window correlation probe placed by
  clicking the image), **Mask** (polygons and holes drawn on the image,
  grow/shrink, mask image files), **Region** (two corners or typed bounds),
  and **Scale** (typed, or measured from two points of known separation).
  Opened settings fill the editors, and an unedited recipe round-trips
  unchanged. `PrepareState` (`wf.prepare`) holds the pages; the gestures are
  controller functions (`canvas_click!`, `canvas_alt_click!`, `canvas_key!`),
  so scripts and tests use the same calls as the canvas.
- Results tools in the window: **Inspect**, **Profile** (u, v and |V| along a
  clicked line, plotted under the field) and **Circulation** (line-integral
  and vorticity-area estimates for a clicked contour). New `profile_series`.
- Frames, previews, probes and background estimates load on worker tasks
  in the window; a newer request supersedes an older one, and the viewer
  keeps the previous pair until the next one arrives.
- `StereoWorkflow` controller (the stereo window's state, usable from
  scripts): two synchronized camera frame lists, a Calibration step
  (`StereoCalibration`: plate images per camera, grid detection and camera
  fits, the shared dewarp grid, self-calibration with apply), Prepare on the
  dewarped grid, and test/run through the stereo `apply_recipe`. It shares
  the settings, test, run and results functions with `PlanarWorkflow`
  through `AbstractWorkflow`.
- `stereo_window()`: the stereo PIV window — Images (two camera frame
  lists), Calibration (plates with z per camera, detection settings and
  model, fit, dewarp grid, self-calibration with apply; the viewer shows the
  plate's dots coloured by reprojection error and magnified residual
  arrows), Prepare (Preprocess, Mask on the dewarped grid with the cameras'
  out-of-view area shaded, and a dt-only Scale), and the shared Passes, Test
  pair, Run and Results pages. A camera switch in the pair bar chooses the
  viewed camera; `dewarpers = (dw1, dw2)` starts from script-built
  dewarpers. Results show stereo fields on world axes (+Y up). New
  `plane_residuals` and `background_note`.
- `request_grab(path)` saves an image of the open window, for screenshots
  and render checks.
- New dependencies: QML.jl, QMLMakie, Qt6Declarative_jll. Requires the core
  release that exports `effort_schedule`.

Breaking:

- The GLMakie tool windows `batch_runner`, `mask_editor`, `roi_editor`
  (and `roi_editor!`), `scale_tool` and `preprocess_preview` (and
  `preprocess_preview!`) are removed; their tasks live in `planar_window`.
  The `BatchRunner` controller is removed with them (`set_preprocess!`,
  `set_scale!`, `set_pixel_size!`, `set_dt!`, `batch_recipe`,
  `preprocess_steps`, `apply_roi!`, `apply_scale!`); use `PlanarWorkflow`,
  whose `workflow_recipe`, `save_settings` and `load_settings!` cover the
  saved settings. `StereoBatchRunner` and the stereo views are unchanged.
- `PreprocessPreview` holds a `Vector{PreprocessStep}`: `add_step!`,
  `remove_step!`, `move_step!`, `set_step_option!` and `set_steps!` replace
  `PreprocStep`, `enable_step!` and `set_step_param!` (the `enabled` keyword
  becomes `steps`), and the preview applies `recipe_preprocess`, so it
  matches the batch exactly.
- `MaskEditor`, `ROIEditor` and `ScaleTool` keep an image size instead of an
  image copy: construct them from a size or a matrix. These three and
  `PreprocessPreview` no longer take an image path.

## Hammerhead 0.2.0 and HammerheadGUI 0.2.0 (2026-10-03)

Changes since core v0.1.0 (2026-07-14) and HammerheadGUI v0.1.1 (2026-07-22).

### Core

Added:

- Saved processing settings. `PIVRecipe` holds a pass schedule, ordered
  built-in `PreprocessStep`s (backgrounds embedded), mask, ROI, scale,
  sequence/ensemble mode, and precision. `save_recipe`/`load_recipe` store it
  in JLD2; `apply_recipe` runs it on planar or stereo pairs and stores the
  recipe alongside the results, so `load_recipe(results_path)` recovers the
  settings. `recipe_diff` lists changed settings; `recipe_preprocess` returns
  the preprocessing as a function.
- `search_area_size` enlarges the frame-B search window around a concentric
  frame-A window (CPU single-pair and ensemble).
- `on_result` callback on all sequence drivers, including stereo, for
  consuming each result as it completes.
- `collect_results = false` on the planar, stereo, and PTV sequence drivers:
  results go to callbacks and the output file without accumulating in memory.
- `ResultFile(path)` / `load_results(path; lazy = true)` index a completed
  results file and load one entry per access.
- `FieldStatisticsAccumulator` and `update_statistics!` compute planar/stereo
  field statistics incrementally in grid-sized memory.
- `export_table` accepts `TrackingResult`, appending trajectory/observation
  IDs, frame indices, gaps, and validity columns.
- Planar-grid `export_table`/`export_vtk` accept
  `transform = PlanarTransform(...)` with explicit length units and optional
  pair delay; coordinates and vector components use the transformed basis.
  VTK files record coordinate and component unit labels.
- `resample_planar` and `resample_image` sample planar vectors and scalar
  images onto shared calibrated coordinates (for example PIV with PLIF), with
  per-sample availability.
- Stereo sequence and ensemble drivers check camera exposure timestamps and
  declared pair delays before loading images (`sync_atol`, `sync_rtol`,
  `missing_timestamps`).
- `flow_derivatives(...; stencil = :centered)` uses central differences only.
- Area circulation over a region reports coverage; incomplete coverage errors
  by default, and `coverage = :report` returns the value with valid/requested
  area.

Changed and fixed:

- Wieneke uncertainty estimates are no longer biased low. Covariance sums now
  use the raw smoothed correlation-difference products instead of
  window-mean-centred ones, and the variance is floored at the independent-pixel
  term, so textured windows no longer report σ = 0. On synthetic data with
  16 px final windows, coverage of the random error rises from about 43%/75% to
  54%/90% at 1σ/2σ; σ values typically increase by 5–15%. Coverage of the
  total error is lower where deformation interpolation bias dominates (small,
  clean particle images); see "How accurate are the measurements?" in the docs.
- Flat, nonpositive, or nonfinite correlation planes now give NaN displacement
  and an outlier flag instead of an arbitrary peak, on CPU, KA, and GPU
  backends. Deformed windows also need contrast in the original pixels they
  sample. Exact constant windows are centered before apodization, predictors
  skip nonfinite neighbors, and alternative-peak candidates are reset between
  windows.
- Effort presets fit the selected ROI, and deformation handles predictor grids
  with a single node along an axis.
- `result_spectrum` requires an explicit sampling interval `dt`; the pair
  delay in `PhysicalScale` is no longer used for it.
- `extract_region` returns `included` (`true` = returned node).
- Field statistics use numerically stable running moments.
- PTV matches particles in one precision when frame element types differ;
  tracking loads and detects frames one at a time.
- Empty results files load as an empty vector; unknown `format_version`
  values are rejected.
- Sequence drivers wait for an in-flight frame prefetch before returning,
  including on failure or cancellation, and report the original error.
- Manual affine registration rejects nonfinite, rank-deficient, and singular
  inputs.

### HammerheadGUI

- The batch form saves and opens its settings as a core recipe ("save
  settings…" / "open settings…"), including the recipe stored in a results
  file. A loaded recipe runs its exact pass schedule through the "saved
  settings" effort, and batch output files carry their recipe.
- `ROIEditor` / `roi_editor`: select an analysis region by two corners or
  numeric bounds and apply it to the batch form; results keep original image
  coordinates.
- `ResultExplorer(path; lazy = true)` browses a completed results file one
  entry at a time.
- Circulation reports area coverage; the self-calibration review explains
  non-converged runs and labels disparity medians as magnitudes.
- Preprocessing pipelines exported to a batch copy their background image.

### Compatibility

- Results files keep `format_version = 1`; result structures are unchanged.
  Files written by `apply_recipe` add a `recipe` entry that existing readers
  ignore.
- Recipe files use a separate `recipe_format_version = 1`.
- `TABLE_SCHEMA_VERSION` remains `hammerhead-table-1`. Tracking tables add
  columns; existing columns keep their order and meaning, so read columns by
  name.
- `result_spectrum` calls that relied on an attached `PhysicalScale` for the
  sampling interval must now pass `dt`.
- HammerheadGUI 0.2.0 requires Hammerhead 0.2. (HammerheadGUI 0.1.1 called
  `run_piv_sequence(...; on_result)`, which core 0.1.0 lacks; upgrade both.)
