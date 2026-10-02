# Release notes

## Unreleased

These entries describe changes after the registered baseline. No new package
version or release date has been assigned. See [RELEASING.md](RELEASING.md) for
validation and the core-first, GUI-second release sequence.

### Core

- Added opt-in planar execution diagnostics for actual pass sweeps, tolerance
  checks, stopping conditions, and primary residual displacement summaries.
  Sequence files can retain versioned diagnostics beside each result; replay
  associates them with the verified recipe and inputs. Result structures and
  numerical stopping behavior are unchanged. Residuals describe the primary
  correlation measurement before alternatives, replacement, or predictor addition.
- `compare_recipe_pair` evaluates two complete built-in planar recipes on an
  explicitly selected, content-matched pair. Reports compare exact common grid
  nodes and retain native-grid quality populations, units, settings, and input
  provenance. Saved TOML snapshots describe recipe sensitivity, not accuracy;
  unequal grids are not interpolated and unsupported uncertainty is explicit.
- Added separate version-1 planar checkpoints with immutable per-pair native
  results and verified commit records. Built-in CPU/KA recipes resume only with
  matching ordered inputs, recipe, and software identity. Interrupted writers
  require explicit recovery; partial files are not adopted. Lazy committed
  prefixes and native aggregate export preserve earlier committed results.
  Same-directory publication does not promise power-loss durability.
- Non-informative correlation planes (flat, nonpositive, or nonfinite) now
  produce unavailable measurements instead of arbitrary boundary displacements.
  Exact constant windows are centered without roundoff texture before
  apodization. Unmasked missing measurements remain rejected independently of
  optional validators; UOD and deformation predictors exclude nonfinite donors.
  Deformed windows now require exact contrast in both original sampled raw
  stencil unions. Distant B-spline coefficient leakage and virtual boundary
  zeros alone do not establish source information. This convention preserves
  any contrast present in the sampled original pixels, without an amplitude
  threshold; uninformative correlation and UQ contributions are skipped.
- `RunQualityReport` summarizes planar/stereo stored fields with node-weighted
  counts and explicit denominator/availability rules, then saves validated TOML.
  Completed experiment outputs can be associated by verified content identity.
  Current flags are not replacement histories, and numerical uncertainty
  availability does not establish accuracy, coverage, or measurement association.
- `recipe_diff` reports deterministic, readable processing-setting changes
  between verified planar recipe snapshots. Large mask/background payloads use
  shape/precision/content summaries; script paths, inputs, and run environments
  remain separate from scientific recipe settings.
- Added version-1 planar experiment records with content-addressed recipes and
  inputs, explicit built-in preprocessing, embedded backgrounds/masks/ROI,
  environment provenance, and streaming replay with optional run records.
  Replay supports CPU/KA and Float32/Float64; custom preprocessing requires a
  caller-provided function matching a recorded script reference. Saved scripts
  are never automatically executed. Stereo, PTV, tracking, and GPU experiment
  recipes remain unsupported.
- Added a reproducible validation scorecard command covering seeded synthetic
  truth and committed Challenge A/4E smoke data. Reports separate error claims
  from smoke checks and record source/input hashes, full settings, warmed CPU
  timings, and cumulative Julia allocations (not peak memory).
- Stereo sequence and ensemble processing validate available exposure times,
  observed pair delays, and declared `FramePair.dt` before reading images or
  opening output. `sync_atol` and `sync_rtol` control tolerance relative to pair
  delay; `missing_timestamps = :error` requires metadata. The default `:allow`
  preserves path/matrix workflows without asserting synchronization.
- Planar, stereo, and PTV sequence drivers accept `collect_results = false`.
  Results still reach callbacks and persistence, but the driver returns
  `nothing` and does not retain a growing result vector. Existing defaults
  continue to return results. Consumers may still retain their own copies.
- `export_table` accepts `TrackingResult`. Eight columns are appended to the
  existing CSV schema for trajectory/observation IDs, original frame indices,
  derived elapsed time, gaps, and numerical validity. Readers should select
  columns by name. Acquisition timestamps are not inferred from frame indices.
- Effort presets fit the selected ROI. Deformation now handles predictor grids
  with a single node along one or both axes by constant extension.
- `FieldStatisticsAccumulator`, `update_statistics!`, and
  `field_statistics(accumulator)` calculate planar/stereo population moments
  without retaining result histories. Updates validate coordinates, dimensions,
  and scale metadata before changing state; snapshots are independent copies.
- `ResultFile(path)` and `load_results(path; lazy = true)` index completed native
  files and read one entry per access. The index retains keys rather than result
  payloads and rejects detectable file changes. Eager loading stays the default.
  Saving an index or its standard array views over the source file is rejected
  before opening output, including when the destination is a file alias.
- Planar-grid table and VTK exports accept `transform = PlanarTransform(...)`
  with explicit length units and optional pair delay/time units. Coordinates,
  vectors, and component uncertainties use the transformed basis. Mixed-axis
  uncertainty requires an explicit independence assumption or is reported as
  unavailable. Transform export requires raw pixel results without attached
  scale metadata; stereo, PTV, and tracking transforms remain unsupported.

### HammerheadGUI

- Added a checkpoint workflow for creating or reopening complete planar
  experiments, progress, cancellation between committed pairs, and explicit
  recovery after a stopped writer. Browsing retains a fixed lazy prefix and
  exports use the core's fresh-destination checks. Processing within a pair and
  identity verification can still delay UI interaction.
- The saved-experiment workflow generates, saves, and displays the shared core
  quality report after verifying the completed run. Report saves protect known
  inputs, outputs, and experiment records; changed/failed/busy runs are refused.
- Added a dedicated saved-experiment controller and workflow, accessible from
  the batch form. Supported form settings export complete recipes; imported
  recipes retain fields that the ordinary form cannot edit. Replay records run
  status and opens completed results lazily, with explicit environment-change
  consent. This first workflow has no live progress or cancellation.
- Added an opt-in, isolated Qt6/QML prototype and desktop requirements matrix.
  Windows resolution, controller reuse, and native framebuffer capture have
  evidence; native OpenGL teardown fails and broader platform/input checks remain
  open. This adds no toolkit dependencies to the production GUI.
- Built preprocessing pipelines now copy a background used for subtraction,
  so later in-place preview edits cannot change a captured batch pipeline.
- Added `ROIEditor`, `roi_editor`/`roi_editor!`, and batch ROI controls, with
  two-corner selection, numeric bounds, reset, and full-image coordinate
  preservation. Invalid or oversized custom windows are rejected before opening
  batch output. The selected ROI is captured when the run starts.
- `ResultExplorer(path; lazy = true)` and `result_explorer(path; lazy = true)`
  browse completed files with one displayed result and bounded derivative
  caching. Failed reads preserve the previous frame and report an error.
  Lazy explorers do not follow live writes or accept appended results.

### Compatibility

Native JLD2 `format_version` remains **1**; persisted result structures are
unchanged. Experiment records have their own `experiment_format_version = 1`
and do not change the native result schema. Checkpoints and quality reports use
separate version-1 metadata formats. `TABLE_SCHEMA_VERSION` remains **`hammerhead-table-1`** under its
additive-column policy. Fixed-column-count CSV readers need to accommodate the
eight tracking columns. Existing columns retain their order and meaning.

Optional execution companions and standalone recipe-comparison reports also use
independent version-1 schemas. Existing result readers ignore companions;
result-only copies may drop them. No execution history is inferred from older
files' requested settings.

The production GUI framework remains GLMakie. Qt/QML and other toolkit
candidates are evaluations in [ROADMAP.md](ROADMAP.md), not supported
replacement shells. Qt/QML dependencies belong only to the opt-in prototype.
