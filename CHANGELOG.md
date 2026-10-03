# Release notes

## Unreleased

These entries describe changes after the registered baseline. No new package
version or release date has been assigned. See [RELEASING.md](RELEASING.md) for
validation and the core-first, GUI-second release sequence.

### Core

- Added opt-in derivative stencil descriptions and a centered-only policy.
  Contributor indices, signed spans and weights remain separate from input
  eligibility and finite outputs. Invalid coordinate geometry is rejected;
  native integer component overflow produces an unavailable derivative.
- Added separate native stereo sequence timing companions with ordered camera
  metadata, exact observed delays and midpoints, synchronization policy and
  effective scaling-delay provenance. Frozen selections and measurement-field
  binding protect callback delivery and persistence.
- Added opt-in version-3 quality reports with planar and per-camera execution
  coverage, sweep/check counts and primary support. Stereo fields are verified
  when generating the report; residual amplitudes are not pooled across grids.
  Existing default and history-only report formats remain unchanged.
- Added a bounded synthetic spatial-response study comparing two complete
  schedules at matched output spacing, with analytic midpoint truth, guarded
  harmonic fits and original full-error/uncertainty populations.
- Added opt-in stereo execution companions that retain each camera's planar
  pass observations, common dewarp-grid geometry and measurement-field binding.
  A separate reader distinguishes metadata inspection from verified result fields;
  residuals remain in dewarped pixels.
- Added a bounded rendering/interpolation diagnostic for clean synthetic scenes,
  separating particle-support and pixel-area sampling contrasts from known-shift
  image-warp comparisons. Original full-error controls and production uncertainty
  defaults are retained.
- Added explicit sample-time validation for temporal spectra, checking interval
  regularity and accumulated timing drift with exact arithmetic and optional
  tolerances. Result spectra reject incompatible grids, shapes and scales;
  existing explicit-interval calculations retain their numerical behavior.
- Dedicated timed artifacts and calibrated companions accept foreign absolute
  source locators as provenance while protecting consumed local files. Explicit
  local relocation does not rewrite historical source paths or imply verification
  of unavailable source images.
- Experiment replay accepts a completed-pair progress callback. Output guards
  detect prospective aliases through parent links and Windows path spelling
  before processing begins.
- Added a separate calibrated CSV export for particle matches and trajectories,
  with affine point/vector transformation and a versioned TOML companion for
  transform settings, units, diagnostic availability and CSV verification.
  Actual-time trajectories retain their recorded velocity intervals; scalar
  particle residuals remain explicitly in their original pixel basis.
- Added a bench-only repeated-noise study on fixed particle scenes, retaining
  full truth-error metrics alongside conditional variability and disjoint-pair
  diagnostics. A pre-specified covariance weighting is an experimental comparator;
  production uncertainty defaults are unchanged.
- Added `tracking_speed_summary` for validated bulk summaries of actual-time
  trajectories, with explicit observation-mean secants and unavailable-track reasons.
- Added explicit actual-time trajectory linking with exact sample metadata,
  elapsed-time prediction and gap validation, and velocities over recorded
  observation intervals. `TimedTrackingResult` preserves timing through its own
  native artifact and CSV schema; ordinary tracking remains ordinal by default.
- Added a bench-only uncertainty diagnostic that reproduces stored estimates
  from retained final-sweep CPU windows and traces covariance-ring selection,
  negative variance clamps and numerical zero outcomes. Full truth-error
  coverage remains separate from centered and residual-inclusive alternatives;
  production estimator defaults are unchanged.
- Added opt-in planar pair-timing companions with exact timestamp values,
  observed delays/midpoints, source identifiers, and effective scaling-delay
  provenance. All selected metadata is checked before loading or opening output;
  callbacks cannot redirect frozen frame selections or invalidate saved binding.
- Added opt-in version-2 quality reports for verified final-sweep history,
  with explicit missing/unsupported coverage and actual event/origin counts.
  Default reports remain version 1. Public history verification and per-node
  accessors support inspection without reloading a raw result or copying a packet.
- Recheck captured measurement history and timing after function-output callbacks,
  including callback-only capture, before opening their destination.
- Added opt-in final-sweep planar measurement history: primary values and
  residuals, first-observed rejection stages, accepted alternative ranks,
  observed fill/restoration events, final origin and stored uncertainty status.
  Separate native companions bind each history to its result; callback mutation
  cannot silently change that binding. Uncertainty is not re-estimated after
  substitution or filling, and history does not certify uncertainty coverage.
- Added a seeded synthetic uncertainty scorecard with component coverage,
  signed normalized errors, explicit zero/unavailable uncertainty populations,
  and full-population versus uncertainty-subset error metrics. Independent seeds
  and timing repeats remain distinct; pooled moments stream across seeds.
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
  scale metadata. Scattered PTV/tracking calibration uses the separate paired
  table API above; transformed stereo grids remain unsupported.

### HammerheadGUI

- Saved-experiment controls use Files, Replay and Reports sections, with
  persistent cancellation/progress/status and pagination sized to available
  space. Inactive controls retain their settings and are removed from mouse
  hit regions.
- Native lazy explorers can inspect stereo execution companions, verifying raw
  reconstructed and camera fields before physical display. Camera residuals stay
  in dewarped pixels; stereo per-node history remains unavailable.
- Saved-experiment quality reports have separate history and execution options.
  Scaled magnitude fields are labelled as speed. The Qt prototype permits vector
  picking after leaving a demo with mask drawing enabled.
- Extended the isolated Qt software prototype with saved planar experiments,
  complete recipe/history inspection, replay controls and verified lazy result
  inspection with physical units. The native rendering/lifecycle gate remains
  separate from software-shell validation.
- Ordinary saved-experiment replay now reports completed-pair progress and
  supports cooperative cancellation. Requests capture their settings before
  observer notifications; cancellation waits for pair persistence and loader
  cleanup. This workflow restarts from the beginning on a subsequent replay.
- Added explicit loading and inspection of single actual-time trajectory
  artifacts, preserving timing through physical display, selection and export.
  Speed colors use observation-mean secants, with unavailable values identified.
- Added a saved-recipe comparison workflow with explicit pair selection,
  pixel/physical value bases, and shared core verification and report persistence.
  Current choices remain separate from a previous report after edits or failures.
- Added lazy inspection of recorded measurement history and execution diagnostics,
  with raw-result verification before physical display and transactional navigation.
  The experiment workflow can include recorded history in saved quality reports.
- Added an isolated Qt lifecycle harness with owned child processes, bounded
  timeouts, complete logs, source identities, and render/release/exit gates.
  Prototype viewports now dispose application callbacks and use fresh figures.
  The enforced Windows offscreen OpenGL trial cannot create a native context;
  native bridge cleanup and production framework adoption remain unvalidated.
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
and do not change the native result schema. Checkpoints use a separate version-1
format. Quality reports default to version 1; opt-in history-only reports use
version 2 and execution-aware reports use version 3, optionally with history.
Readers accept all three. `TABLE_SCHEMA_VERSION` remains
**`hammerhead-table-1`** under its
additive-column policy. Fixed-column-count CSV readers need to accommodate the
eight tracking columns. Existing columns retain their order and meaning.

Optional execution, measurement-history and pair-timing companions and standalone
recipe-comparison reports also use independent version-1 schemas. Existing
result readers ignore companions; result-only copies may drop them. No execution
history is inferred from older files' requested settings.

Actual-time tracking uses a separate `timed_tracking_format_version = 1`
artifact without the ordinary native `format_version` marker. Generic native
readers reject it rather than silently discarding timing. Its table schema is
`hammerhead-tracking-time-table-1`; the ordinary table schema is unchanged.

The production GUI framework remains GLMakie. Qt/QML and other toolkit
candidates are evaluations in [ROADMAP.md](ROADMAP.md), not supported
replacement shells. Qt/QML dependencies belong only to the opt-in prototype.
