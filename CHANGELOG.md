# Release notes

## Unreleased

### Core

Added:

- `effort_schedule` is exported: it returns the pass schedule behind
  `effort = :low/:medium/:high` for inspection or as a starting point.
- `recipe_preprocess(steps)` builds the preprocessing function from a vector
  of `PreprocessStep`s, so previews apply exactly what `apply_recipe` runs.
- `run_piv_ensemble` and `run_piv_stereo_ensemble` accept a `progress`
  function, called as `progress(done, total)` after each pair of each pass
  (stereo counts both cameras); throwing from it aborts the run, as in the
  sequence drivers. `progress = true/false` still toggles the meter.
- `save_results(path, results; recipe, calibration, sources)` stores the
  settings, stereo calibration and frame labels a run would store, and
  `load_sources(path)` returns the stored frame labels per result.
  `replace_vectors!` is exported.
- `save_calibration(path, dw1, dw2)` / `load_calibration(path)` save a
  stereo rig (camera models including an applied self-calibration, image
  sizes, and the shared `DewarpGrid`) and rebuild its dewarpers bitwise.
  Stereo `apply_recipe` results files store the calibration as well, so
  `load_calibration(results)` works like `load_recipe(results)`.
- `PIVRecipe(...; preprocessing = (steps1, steps2))` gives each stereo
  camera its own preprocessing (e.g. per-camera background subtraction);
  `recipe_preprocess` then returns a function per camera. Recipe files are
  now format version 2; version 1 files still load.
- Particle recipes: `PIVRecipe(passes; mode = :ptv | :tracking, ptv,
  ptv_predictor = :piv | :none, min_track_length, max_gap)`. `apply_recipe`
  runs `run_ptv_sequence` on pairs (the passes are the PIV predictor) or
  `track_particles` on a frame sequence, and stores the recipe with the
  results. `track_particles` gained a `preprocess` keyword. `self_calibrate` accepts
  a per-camera `preprocess = (f1, f2)` tuple, like the stereo drivers.
- Recipes are TOML text (format version 3). `save_recipe("x.toml", r)` writes
  a readable settings file with the mask (`x.mask.png`) and backgrounds
  (`x.background.tif`, Float64) beside it; JLD2 recipe and results files embed
  the same text plus the arrays. `recipe_toml(r)` returns the text. Version 1
  and 2 files still load. TOML (a standard library) is a new dependency.
- `apply_recipe(recipe, pairs; masks)` takes per-pair masks for a moving
  boundary: one entry per pair (a mask image path, a `Bool` matrix, or a
  tuple of the two frames' masks), or a function `(i, imgA, imgB) -> mask`;
  each is unioned with the recipe's static mask. Mask paths are stored with
  the results: `load_sources(path; masks = true)`.
- `backend_available(backend)` reports whether a backend is loaded and has a
  working device (`CUDA.functional()` / `AMDGPU.functional()` for the GPU
  extensions); `backend_problem(backend, passes)` returns why a schedule
  cannot run on a backend, or `nothing`.
- `PIVParameters(image_interpolation = :cubic | :linear,
  predictor_interpolation = :linear | :cubic)` choose how deforming passes
  resample the images (cubic B-spline by default) and interpolate the
  predictor field (bilinear by default; `:cubic` is a cubic B-spline through
  the vectors). KernelAbstractions and GPU backends support only the defaults.

### HammerheadGUI

Added:

- `planar_window()`: a Qt Quick window for planar PIV that walks through
  Images → Prepare → Passes → Test pair → Run → Results, with one image
  viewer that follows the step and can pop out into its own window. Settings
  save and open as core `PIVRecipe` files (results files carry theirs); the
  test pair runs exactly the batch's `apply_recipe` call; runs write results
  as they finish and can be canceled. Controllers (`PlanarWorkflow`,
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
  plate's dots colored by reprojection error and magnified residual
  arrows), Prepare (Preprocess, Mask on the dewarped grid with the cameras'
  out-of-view area shaded, and a dt-only Scale), and the shared Passes, Test
  pair, Run and Results pages. A camera switch in the pair bar chooses the
  viewed camera; `dewarpers = (dw1, dw2)` starts from script-built
  dewarpers. Results show stereo fields on world axes (+Y up). New
  `plane_residuals` and `background_note`.
- Stereo calibrations save and open: **Save calibration…** /
  **Open calibration…** on the Calibration step (`save_calibration_file`,
  `open_calibration!`; a stereo results file opens the calibration that
  produced it), and `stereo_window(; calibration = path)`. Dewarp grid
  options rebuild the grid for opened cameras too (`can_build_grid`).
- Per-camera preprocessing in the stereo window: **Separate steps per
  camera** (`set_separate_preprocessing!`) gives each camera its own list,
  edited for the shown camera; **Estimate background** subtracts each
  camera's own background (`set_backgrounds!`). Self-calibration uses each
  camera's steps.
- Particle analysis in the planar window: **Analysis** on the Passes step
  adds **PTV** (particle matches per pair) and **Particle tracking**
  (tracks through the frames) to the PIV modes. In these modes the step is
  **Particles**: detection, matching, validation and track settings
  (`ParticleSettings`, `set_particle_option!`), the pass table as the
  optional PIV predictor, and a live preview circling the particles
  detected on the shown frame. Test pair matches the representative pair
  (or tracks up to ten frames from it), Run writes per-pair PTV results or
  one set of tracks, and Results draw particles and tracks. Settings save
  as particle recipes.
- Window usability (test-drive feedback): the Results field menu picks the
  field shown (it mismatched entries once a label contained "|"); Results
  step through the pair bar, independent of the representative pair
  (`pair_position`, `go_to_pair!`, `step_pair!`); ← / → step pairs or
  results, A / B and shift+← / → switch frames; **Auto contrast** in the
  pair bar (display only); ctrl-click resets the zoom in every mode; Passes
  has separate **Zero padding** and **Gaussian weighting** switches and a
  **Validation** section (normalized median test, threshold, neighborhood,
  minimum peak ratio, replacement); the Run step's output path has
  **Clear**. U.S. spelling throughout.
- Results step: percentile or absolute color limits with editable values;
  diverging zero-centered scale for vorticity, divergence and Q; a
  physical-units toggle; flagged vectors optionally included in derived
  fields, profiles and circulation; display-only re-validation with other
  outlier-test settings (`set_revalidation!`); the profile follows the
  shown field; profile and contour points can be dragged and deleted; the
  particle image of the result's pair as a field (frames A/B). Run: results
  kept in memory save afterwards (`save_run_results!`). Images: **Reuse
  settings…**, replacing **Use these settings** on Results.
- Images: add frames by folder and glob pattern (`matching_files`,
  `set_frame_pattern!`, `add_matching!`), in natural order; **From two
  frames…** infers the pattern from two frames (`infer_pattern`). Passes:
  click the viewer for a correlation probe of the final window size.
  Scale: measure the pixel size on a separate ruler image (`load_ruler!`).
  A viewer toolbar sets the view mode (edit, zoom, pan; `set_view_mode!`),
  resets the view, and saves it as an image.
- Prepare › Mask: **Per-frame mask images** for a moving boundary (files, or
  folder and pattern), one per frame; each pair uses both frames' images and
  the static mask, and the viewer shades the representative pair's combined
  mask (`wf.frame_masks`, `representative_mask`, `frame_masks_problem`).
  Passes: **Image interpolation** and **Predictor interpolation**. **Save
  settings…** writes a TOML settings file by default (`.jld2` still
  accepted). `start_test!`/`start_run!` take `options` keyword inputs for
  `apply_recipe`.
- Passes: **Run on the GPU** switch (`use_gpu!`, `set_backend!`,
  `gpu_packages`): loads CUDA or AMDGPU on first use and runs PIV tests and
  batches on it; unavailable when neither package is installed. Settings the
  GPU backends do not implement are reported on the step.
- The Run and Results steps turn to needing attention when settings or inputs
  change after a run (`run_stale`), as Test pair already did; a failed or
  canceled run also needs attention. The window title reads
  `Hammerhead planar PIV | settings.toml`.
- `request_grab(path)` saves an image of the open window, for screenshots
  and render checks.
- Ensemble runs in both windows: with **Ensemble** chosen on Passes, Run
  pools all pairs into one result (written with its recipe), reports
  progress per pass and pair (`run_progress`), and can be canceled after
  the pair in flight (no partial result is kept). Results then shows the
  ensemble result. An ensemble test goes stale when the frames change.
- The Qt windows refuse to open (`ArgumentError`) in a Julia session that
  already has a GLMakie screen, and `result_explorer`, `calibration_review`
  and `selfcal_review` warn after a Qt window was opened: GLFW and Qt GL
  contexts sharing a process crashed the AMD driver.
- New dependencies: QML.jl, QMLMakie, Qt6Declarative_jll. Requires the core
  release that exports `effort_schedule` and takes an ensemble `progress`
  function.

Changed:

- With a `PhysicalScale` attached, the magnitude field reads `|velocity|`
  (e.g. `|velocity| (mm/s)`); unscaled results keep `|displacement| (px)`.
  New `field_name(result, field)`.
- An ensemble's pass summary shows no repeat counts and the Passes page
  disables the Repeats column (the ensemble driver runs each pass once).
- The windows open about 4 s faster (≈16.5 s warm from launch on the
  development machine, from ≈21 s): the canvas glyph atlas is cached on
  disk beside Makie's own atlas cache, and the traced precompile statements
  cover both windows.

Fixed:

- Canvas clicks (Results tools, mask and scale points) stopped working after
  focus moved to a control while Shift or Ctrl was held: the canvas never saw
  the key release and kept treating clicks as zoom gestures. A canvas now
  releases held keys and drags when it loses focus, and takes focus when the
  pointer enters it (unless a text field is being edited), so the first click
  after using a menu counts.
- The circulation tool's line integral now runs counterclockwise in the
  result's x–y frame whatever the click order, so it agrees in sign with the
  vorticity-area estimate (Stokes' theorem) instead of following the
  direction the contour was clicked.

Breaking:

- The GLMakie tool windows `batch_runner`, `mask_editor`, `roi_editor`
  (and `roi_editor!`), `scale_tool` and `preprocess_preview` (and
  `preprocess_preview!`) are removed; their tasks live in `planar_window`.
  The `BatchRunner` controller is removed with them (`set_preprocess!`,
  `set_scale!`, `set_pixel_size!`, `set_dt!`, `batch_recipe`,
  `preprocess_steps`, `apply_roi!`, `apply_scale!`); use `PlanarWorkflow`,
  whose `workflow_recipe`, `save_settings` and `load_settings!` cover the
  saved settings.
- The GLMakie stereo windows `stereo_batch_runner` and `stereo_calibration`
  and the `StereoBatchRunner` controller are removed (with its
  `set_schedule!`, `set_effort!`, `build_parameters`, `build_scale`,
  `validate`, `start!`, `cancel!` and `stereo_pairs`, and the
  `parse_schedule` helper); `stereo_window` covers calibration, dewarping,
  self-calibration and synchronized runs. `build_dewarpers(cr1, cr2)` stays
  and builds a dewarper pair from two `CalibrationReview`s, for
  `stereo_window(; dewarpers)` or `run_piv_stereo`; `set_dewarpers!` now
  applies to a `StereoWorkflow` or `StereoCalibration`. The standalone
  `calibration_review`, `calibration_review!`, `selfcal_review` and
  `result_explorer` views stay.
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
  window-mean-centered ones, and the variance is floored at the independent-pixel
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
