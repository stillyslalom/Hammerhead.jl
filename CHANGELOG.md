# Release notes

## Unreleased

### Core

Added:

- `effort_schedule` is exported: it returns the pass schedule behind
  `effort = :low/:medium/:high` for inspection or as a starting point.
- `recipe_preprocess(steps)` builds the preprocessing function from a vector
  of `PreprocessStep`s, so previews apply exactly what `apply_recipe` runs.

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
