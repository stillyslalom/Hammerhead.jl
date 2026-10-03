# Hammerhead roadmap

This is the single active backlog for Hammerhead and HammerheadGUI, reconciled
on 2026-10-02. It includes outstanding work from the former polish backlog,
maintainer notes, archived plans, and the repository review. Historical design
and implementation records remain under [reference/archive](reference/archive/README.md).
Add new work here; update implementation guidance in `CLAUDE.md` when it lands.

Scope remains planar 2D2C and stereo 2D3C PIV, planar PTV, and their experiment
workflows. Stereo PTV is an exploratory extension within the planar measurement
scope. Tomographic PIV, volumetric reconstruction, acquisition hardware control,
Shake-the-Box, pressure-from-PIV, and modal-analysis suites remain out of scope
unless that scope is explicitly changed.

## Implemented baseline

The core and GUI are registered. Planar and stereo PIV, self-calibration,
ensemble correlation, statistics, masking, physical scaling, derived quantities,
PTV and gap-aware trajectories, JLD2 persistence, table/VTK export, and optional
CUDA/AMDGPU execution are implemented. Executable tutorials and regression tests
cover the baseline workflows; reachability does not establish accuracy on every
Challenge case or experimental recording.

In particular, these are completed rather than open tasks:

- Independent interrogation/search-area sizes on CPU, including ensemble runs.
  KA/CUDA/AMDGPU currently reject enlarged search areas.
- GUI browsing of all four persisted result types, with scaled axes and track gaps.
- The GUI calibration-line tool and its physical-scale hand-off to batch processing.
- The baseline GUI batch, mask, preprocessing, calibration, and result tools.

The [backend feature matrix](docs/src/reference/feature_matrix.md) describes
current execution support. The [archived roadmap](reference/archive/ROADMAP.md)
records the original phases, not current status or release promises.

The saved-ensemble batch passes 12,204 full core and 1,885 full GUI checks on
its final source. The documentation build executes all seven tutorials and the
committed-input saved-ensemble example. These local checks do not close hardware
or cross-platform acceptance gates.

## Delivery order

Start a quantitative validation baseline and the experiment-record design first.
Timing integrity and measurement diagnostics should inform that record. Build
streaming/restart support on stable run identities. A GUI framework prototype
can proceed independently now; integrate the chosen shell with the shared
experiment model after the prototype establishes feasibility. Algorithmic
extensions below require an accuracy case or demonstrated workflow need.

Each slice should ship with the stated acceptance evidence, documentation, and
appropriate regression checks. Candidate evaluations can conclude that the
existing implementation is preferable; they are not commitments to add an API.

## 1. Quantitative validation and representative workloads

- [x] Build a reproducible dataset scorecard distinguishing synthetic ground
  truth, known-motion experiments, and real-data smoke checks. Record dataset
  identity, processing recipe, selection rules, and reference conventions.
  [The scorecard command](docs/src/howto/validation_scorecard.md) covers seeded
  translation/shear, an empty-image failure case, and committed A/4E smoke data.
  Known-motion measurements are explicitly unavailable. Reports record input
  and source hashes, complete settings, environment, warmed CPU timings, and
  cumulative Julia allocations; allocations are not peak memory.
- [x] Reject absent, nonpositive, nonfinite, and flat correlation planes on CPU
  and shared KA paths; center exact constant correlation-input windows without
  artificial apodization texture. [Regressions](test/test_noninformative_windows.jl)
  cover stale peak scratch, masks, tiny real contrast, wholly blank multipass/
  ensemble inputs, finite predictors, and unavailable uncertainty. These guards
  repair the original empty-image scorecard failure.
- [x] Carry original-source contrast evidence through deformation and ensemble
  processing. Both sampled raw stencil unions must contain exact contrast in
  processing precision; virtual boundary zeros and distant B-spline coefficient
  leakage alone do not count. This explicit [stencil convention](docs/src/explanation/noninformative_windows.md)
  preserves genuine weak contrast and gates correlation, alternatives, and UQ
  consistently. [Regressions](test/test_original_source_support.jl) cover the
  formerly fabricated flat-patch vectors, masks, boundaries, precision, CPU/KA,
  and mixed ensemble contributions. New vendor-device hardware evidence remains
  open separately; shared-kernel tests do not establish that coverage.
- [x] Add a bounded [synthetic uncertainty sweep](docs/src/howto/validation_uncertainty.md)
  with independent fixed seeds, controlled primary-only final output, explicit
  component coverage/normalized-error populations, and full versus UQ-subset
  errors. Preserve zero/nonfinite uncertainty and arithmetic-failure counts;
  stream pooled moments and retain only per-seed quantiles. This does not close
  experimental coverage, spatial-resolution transfer or peak-memory evidence.
- [ ] Investigate the synthetic sweep's low baseline coverage of full error by
  reported random uncertainty. Separate signed bias, residual/convergence
  assumptions, estimator behavior, and rendering/interpolation effects with
  controlled comparisons; preserve the original full-error populations and
  report independent-seed evidence before changing estimator defaults. Account
  separately for zero-sigma cases, including the effects of covariance-ring
  truncation and the existing nonnegative variance clamp.
- [x] Add a bounded diagnostic command that reproduces stored uncertainty from
  retained final-sweep CPU windows and traces covariance-ring selection,
  pre-clamp variance and numerical zero outcomes. Compare a few fixed-seed
  baseline/noise/window-size and residual/phase cases without changing estimator
  defaults; keep full truth-error coverage and supplementary alternatives distinct.
  The [diagnostic investigation](docs/src/howto/diagnostic_uncertainty.md) and
  [292 focused checks](test/test_diagnostic_uncertainty.jl) preserve the original
  scorecard metrics. Negative pre-clamp variance explains the observed zero-sigma
  cases; the broader coverage investigation remains open.
- [x] Measure conditional variability with repeated independent noise on fixed
  clean scenes. Retain full truth-error metrics, complete-case selection losses,
  disjoint realization-pair differences and stored zero/unavailable estimates.
  Compare a pre-specified positive covariance weighting as a bench diagnostic,
  with independent algebraic checks, before considering any estimator change.
  The [conditional study](docs/src/howto/conditional_uncertainty.md) includes
  [128 regression checks](test/test_conditional_uncertainty.jl) and a 26-call
  evidence run. Improved paired-noise coverage does not establish calibrated
  total-error coverage; the broader investigation remains open.
- [x] Compare deterministic rendering and deformation-interpolation sensitivity
  on fixed clean scenes. Separate point-support and pixel-area sampling changes,
  preserve original full-error controls, and keep known-translation image-warp
  diagnostics separate from PIV inputs. Do not infer random-uncertainty calibration
  from a clean-image contrast.
  The [rendering study](docs/src/howto/rendering_uncertainty.md) includes
  [219 focused checks](test/test_rendering_uncertainty.jl) and a 12-call evidence
  run with unchanged baseline scientific rows. Wider point support has a small
  effect in these scenes; pixel-area sampling changes the results more, without
  consistently improving uncertainty coverage. The broader investigation remains open.
- [x] Measure spatial response on fixed synthetic transverse-shear scenes with
  analytic midpoint truth, matched output stride and two final window sizes.
  Preserve full-error/yield/UQ populations alongside guarded harmonic fits and
  exact common-coordinate comparisons; do not infer a universal resolution limit.
  The [bounded study](docs/src/howto/spatial_transfer.md) has
  [209 focused checks](test/test_spatial_transfer.jl) and a frozen 16-call run.
  At the shortest 32 px wavelength, common accepted populations shrink to
  43/256 and 59/256 interior nodes; fitted response describes those subsets.
  Full-grid yield/error and uncertainty populations remain visible, and broader
  spatial-resolution validation stays open below.
- [ ] Cover bias/RMS error, valid-vector yield, spatial-resolution sensitivity,
  uncertainty coverage and normalized errors, runtime, and peak host/device
  memory across particle density, diameter, noise, shear, and dropout conditions.
  Include independent Challenge data beyond the committed A pair and 4E slice;
  use an independent implementation comparison where useful and reproducible.
- [ ] Evaluate PTV correspondence precision/recall, trajectory identity switches,
  fragmentation, and gap recovery on independent or annotated recordings.
- [x] Add a reproducible annotated-particle scorer and controlled clip study
  covering detection, correspondence, retained identities, fragmentation and
  gap recovery. Preserve full versus detection-conditioned populations and
  ambiguous identity bounds; verify an independent localization association.
  Use complete ID/frame visibility annotations, guarded input manifests and
  unchanged production tracking. Keep external/real-recording validation open.
  The [scorer guide](docs/src/howto/validation_ptv_tracking.md) and
  [2,197 focused checks](test/test_validation_ptv_tracking.jl) cover independent
  assignment oracles, visibility/identity/gap populations and guarded imports.
  A frozen eight-clip run and independent CSV/source audits preserve all
  predictions: stress accepted recalls are 170/190 and 168/190 despite unit
  accepted precision. This does not establish independent or real-data accuracy.
- [x] Correct the controlled PTV/tracking scorer's recorded PIV threading
  default to reflect `Threads.nthreads() > 1`, add a provenance regression and
  regenerate the affected one-thread study. Historical artifacts retain their
  original recorded values; the sparse VSJ301 study records the actual default.
  The focused suite passes 2,204 parent and 22 one-/four-thread child checks.
  A fresh eight-clip run passes 36,600 audit checks with stable source hashes;
  all 144 scientific CSVs remain byte-identical to the historical run.
- [ ] Add a reproducible larger-data evaluation command with download/cache and
  checksums; keep a small deterministic regression subset in ordinary CI.
- [x] Evaluate a fixed eight-frame VSJ301 independent synthetic clip with guarded
  acquisition/cache hashes, sparse annotations and all predicted associations.
  Preserve unknown positions/visibility between listed rows, report coordinate
  registration hypotheses separately, and distinguish annotated endpoint
  relinking from true absence recovery. Keep real-recording validation open.
  The [fixed study](docs/src/howto/validation_vsj301.md) completed with unchanged
  production defaults and source identities. Independent source/CSV audits cover
  33,250 provided and 7,190 unknown positions, all 12,123 detections and 9,908 raw
  pair correspondences. Association varies sharply across the three fixed origin
  hypotheses; none is declared verified. Two annotation reappearance events do
  not establish true absence or gap-recovery performance. Regression fixtures
  exercise independent assignment oracles, sparse accounting and guarded caches.
- [ ] Add real sequence tutorials for background estimation, ensemble correlation,
  and statistics (for example Challenge 2A/4A), with caching and a bounded docs
  CI budget. Executed tutorials must not depend on local gitignored `cases/`.
- [ ] Establish production benchmarks from the original experiment requirements:
  approximately 29 MP pairs and 200–500 images, including time to a useful rough
  field. Reassess the historical sub-30-second quick-look target on named hardware
  and specified accuracy/settings rather than treating it as an achieved claim.
- [ ] Record repeatable CUDA/AMDGPU hardware validation as release evidence;
  retain the hardware-free KA tier and track performance separately from accuracy.

Acceptance: one documented command regenerates a scorecard with explicit data
provenance, supported claims, failure cases, and comparable timing conditions.

## 2. Reproducible experiments, timing, and coordinates

- [x] Establish a versioned file-based planar experiment/run record in the core:
  input content identities, explicit pass schedule, built-in preprocessing,
  embedded background and original mask/ROI, scale, CPU/KA precision/settings,
  and creation/run environments. [Experiment records](docs/src/howto/experiments.md)
  use a separate version-1 format, reject unknown versions and changed identities,
  and preserve existing native result files and result-only workflows.
- [x] Serialize built-in planar processing recipes and reference user-supplied
  preprocessing scripts by content hash and entrypoint. Replay requires an
  explicit caller-provided function; it never executes a saved script itself.
- [x] Save/reopen and replay a planar recipe without reconstructing settings.
  Replay verifies inputs and environment before output, streams results, and
  optionally records completed/failed runs. This is a rerun-from-start API.
- [x] Connect planar experiment records to GUI batch snapshots, complete-recipe
  inspection/replay, run history, and content-verified lazy result browsing.
  [The GUI workflow](docs/src/howto/gui_experiments.md) preserves exact imported
  settings; ordinary replay progress/cancellation is tracked below, while full
  recipe editing remains open.
- [ ] Extend experiment records to stereo calibration/dewarping/self-calibration,
  PTV/tracking, and supported GPU devices. Define migrations when extending the
  format; keep toolkit dependencies in the GUI package.
- [x] Save, reopen and replay complete planar ensemble experiments, then connect
  them to GUI creation, progress/cancellation, verified inspection and associated
  quality reports. Preserve ordered input pairs, explicit passes, preprocessing,
  mask, scale, backend and precision. Distinguish input-pair contributions from
  the single pooled output; cancellation must not publish a completed pool.
  Preserve existing sequence run-count and report contracts. Validate direct/replay
  parity and exact imported settings; checkpoint resume and estimator applicability
  remain separate requirements.
  [Saved ensembles](docs/src/howto/ensemble_experiments.md) use a separate record
  and associated report format 5, with staged output publication and accurate
  completed/cancelled metadata when optional history saving fails. Core replay
  passes [136 focused checks](test/test_ensemble_experiments.jl); associated
  reports pass [177 checks](test/test_ensemble_experiment_quality.jl). The
  [GUI workflow](docs/src/howto/gui_ensemble_experiments.md) passes 198 focused
  checks with inspected 900/1100-pixel layouts and a 960-pixel batch launch.
  Final full suites pass 12,204 core and 1,885 GUI checks; the docs build executes
  all seven tutorials plus the committed-input saved-ensemble example.
- [x] Add separate replayable stereo sequence records with exact frozen fitted
  builtin camera coefficients, signed dewarp geometry, complete two-camera
  processing and explicit timing/scaling provenance. Verify ordered files,
  environments, rebuilt maps and raw output binding with noncollecting replay.
  Preserve planar schemas; fitted-camera replay does not rerun calibration
  fitting/self-calibration or establish calibration accuracy.
  [Stereo replay](docs/src/howto/stereo_experiments.md) has
  [119 focused checks](test/test_stereo_experiments.jl) for CPU/KA Float32/64
  parity, exact fitted-camera persistence, source/settings corruption, preflight
  and failed-prefix verification. Existing planar schemas remain unchanged.
- [x] Connect saved stereo records to a dedicated GUI workflow with exact batch
  snapshots, immutable imported settings, replay progress/cancellation, historical
  run selection, verified lazy browsing and associated stored-field/execution
  reports. Preserve captured run/report identities across failed actions.
  The [saved stereo workflow](docs/src/howto/gui_stereo_experiments.md) passes
  [194 focused checks](HammerheadGUI/test/test_stereo_experiments.jl), including
  exact imported settings, cancellation boundaries, historical selection and
  retained report identity. Offscreen layouts were inspected at 900/1100 pixels;
  the full GUI suite passes 1,518 checks. The associated core quality overload
  adds [62 checks](test/test_stereo_run_quality.jl) for verified raw output,
  relocation, protected saves and unchanged report schemas.
- [x] Compare experiment processing revisions with a human-readable settings
  diff. `recipe_diff` returns deterministic field paths and before/after values,
  with content summaries for embedded arrays and explicit added/removed items.
  [Comparison tests](test/test_experiment_comparison.jl) cover full settings,
  snapshot integrity, ordered operations, and location-independent script identity.
- [x] Compare representative-pair numerical results across built-in planar
  recipe revisions, including stored quality/UQ and preprocessing/window-size
  sensitivity. [Selected-pair comparisons](docs/src/howto/pair_comparison.md)
  verify ordered input content, rerun complete recipes and compare exact common
  coordinates with explicit populations and units. Detached version-1 TOML
  reports preserve settings/provenance; differences are not accuracy errors.
  [Regressions](test/test_pair_comparison.jl) cover grid/mask/flag populations,
  relocation, incompatible scales, arithmetic overflow and protected outputs.
- [ ] Preserve exposure timestamps, pair delay, sample time, source frame IDs,
  time units, and calibration/coordinate-frame identity through native persistence
  and exports. Define the time assigned to a displacement measurement.
- [x] Add opt-in [planar pair-timing companions](docs/src/howto/pair_timing.md) that preserve source metadata,
  exact timestamp differences/midpoints, and the delay actually used for scaling.
  Preflight all selected pairs before loading or opening output; retain missing
  units/clocks explicitly and verify companion/result binding. Tracking, exports,
  checkpoint/replay and stereo timing persistence remain separate extensions.
  [Timing regressions](test/test_pair_timing.jl) cover exact epochs, metadata
  snapshots, complete preflight, callback integrity and bounded payload lifetime.
- [x] Persist opt-in stereo sequence timing with both ordered camera descriptors,
  exact per-camera exposure times/delays/midpoints, and the reconstructed field's
  effective scaling delay. Preserve four-tuple and paired-list scaling behavior;
  do not invent a common timestamp under tolerated camera skew. Preflight before
  pixels/output and bind metadata to raw stereo fields. Stereo recipes, ensemble
  timing and GUI/export integration remain separate extensions.
  The [stereo timing guide](docs/src/howto/stereo_pair_timing.md) documents
  normalized tolerances and explicit mixed/exact arithmetic. The
  [167 new checks](test/test_stereo_pair_timing.jl) cover numerical parity,
  frozen metadata, native binding, callback mutation, cleanup and lifetime;
  370 existing timing/execution checks also pass.
- [x] Check stereo exposure synchronization when timestamps are available;
  matching pair delays alone does not establish simultaneous acquisition.
  Sequence and ensemble drivers now check exposure times and declared/observed
  delays before loading or opening output; `sync_atol`/`sync_rtol` and
  `missing_timestamps = :allow/:error` define tolerance and missing-data policy.
  Covered by [stereo timing regressions](test/test_stereo_timing.jl).
- [ ] Reject or explicitly handle irregular sampling in analyses that assume a
  fixed interval. Validate unit and coordinate compatibility before combining results.
  Temporal spectra now have the explicit contract below; other temporal analyses
  still require their own timing contracts.
- [x] Validate explicit sample times for temporal spectra, using both interval
  and accumulated grid residuals, exact arithmetic, declared tolerances and
  compatible result geometry/value bases. The [sampling guide](docs/src/howto/spectrum_timing.md)
  and [157 checks](test/test_spectrum_timing.jl) cover large epochs, drift, range
  failures, scale compatibility and preservation of legacy FFT calculations.
- [x] Add explicit actual-time tracking with exact sample metadata, elapsed-time
  prediction/gap validation, secant velocities and table export. Keep legacy
  ordinal tracking unchanged and use a dedicated persisted artifact that cannot
  silently lose timing through an older native-result reader. GUI inspection
  and other irregular-time analysis remain separate extensions.
  [Actual-time tracking](docs/src/howto/tracking_timing.md) has
  [218 focused checks](test/test_tracking_timing.jl) for timing, linking, gaps,
  exact secants, binding integrity, persistence and CSV provenance.
- [x] Apply `PlanarTransform` consistently to planar-grid table and VTK exports,
  including origin, rotation, reflection, anisotropic scaling, and vector basis.
  Raw pixel results require explicit units and optional pair delay; attached
  scales are rejected. Mixed-axis uncertainty is unavailable unless independence
  is explicitly assumed. [Transform export tests](test/test_transformed_export.jl)
  check geometry, uncertainty, default compatibility, and rejection before writes.
- [x] Extend calibrated table export to PTV and ordinal/actual-time tracking.
  Explicit affine transforms retain vector bases, exact interval semantics and
  diagnostic availability in verified CSV/TOML companions. Scalar particle
  residuals remain in pixels because their direction was not recorded.
  The [export workflow](docs/src/howto/calibrated_scattered_export.md) has
  [357 checks](test/test_calibrated_scattered_export.jl), including output aliases
  and refusal before overwriting protected inputs.
- [x] Make recorded source locators portable in dedicated timed artifacts and
  calibrated table companions. Preserve foreign Windows/POSIX paths as provenance,
  protect actual local/relocated artifacts, and avoid interpreting foreign paths
  relative to the current workspace. [115 portability checks](test/test_artifact_paths.jl)
  cover foreign fixtures, relocation, scientific binding and local alias protection.
  These ran on Windows; Linux/macOS and actual UNC-share execution remain separate
  platform evidence. Other persisted experiment/report formats retain their own
  locator restrictions.
- [ ] Provide a documented calibrated registration/resampling workflow for
  simultaneous PIV/PLIF data, preserving vector basis and validity masks; this
  follows the use case recorded in [Design.md](reference/Design.md).

Acceptance: reopen and reproduce an experiment; round-trip its timing and
coordinate metadata; reject deliberately offset stereo acquisitions and
incompatible combined fields with actionable diagnostics.

## 3. Bounded-memory batches, restart, and GPU memory

- [x] Add a result sink or non-collecting sequence mode so saving results does
  not require retaining every result in RAM, for planar, stereo, and PTV drivers.
  `collect_results = false` returns `nothing` while preserving callbacks,
  persistence, cancellation, and failure cleanup. [Sequence sink tests](test/test_sequence_sink.jl)
  check delivery and weak-reference release; lazy browsing and restart remain
  separate items below.
- [x] Add incremental planar/stereo field statistics. `FieldStatisticsAccumulator`
  stores grid-sized online moments, validates coordinate/scale compatibility,
  and returns independent snapshots. [Incremental tests](test/test_incremental_statistics.jl)
  cover population moments, validity, stable accumulation, and fixed retained
  memory; [workflow tests](test/test_streaming_workflow.jl) combine non-collecting
  output, variable pair delays, physical conversion, and lazy replay.
- [x] Add indexed/lazy result loading and GUI browsing without retaining an
  entire recording. `ResultFile` indexes completed files and loads one entry
  per access; a lazy explorer keeps one displayed result and current derivatives.
  [Lazy I/O tests](test/test_lazy_results.jl) cover unselected-entry isolation,
  handle closure, and detectable file changes; GUI tests cover mixed result
  types and navigation failures. Key metadata remains O(number of results);
  live file following, resumability, and production-size memory evidence remain
  separate work.
- [x] Resume built-in file-based planar recipes using exact ordered input,
  recipe, and software identities. [Version-1 checkpoints](docs/src/howto/checkpoints.md)
  retain immutable native per-pair outputs and verified contiguous commit
  descriptors; completed entries are not recomputed or replaced. Lazy prefixes
  and ordinary native aggregate export preserve existing result workflows.
- [x] Define and test per-result publication/recovery separately for handled
  failure, cancellation, and hard process termination. Same-directory rename,
  exclusive writers/export destinations, explicit interrupted-writer recovery,
  and descriptor-authoritative counts cover tested local filesystems. This is
  not power-loss durability or support for concurrent external mutation.
  [Checkpoint tests](test/test_experiment_checkpoint.jl) exercise lock, staging,
  commit, and terminal-status boundaries, changed identities, and source aliases.
- [x] Add [GUI checkpoint controls](docs/src/howto/gui_checkpoints.md) for complete
  built-in planar recipes: creation/open, absolute committed progress, cancellation,
  explicit stopped-writer recovery, fixed lazy browsing and fresh native export.
  Execution captures state before Observable notifications; progress does not
  rescan the store. [GUI regressions](HammerheadGUI/test/test_checkpoints.jl)
  cover state capture, failure/recovery, protected lazy sources and offscreen
  layout. Work within a pair and initial verification can still pause rendering.
- [ ] Extend checkpoints to additional experiment kinds/backends as their recipe
  contracts become available, and integrate execution/quality companions.
  Establish production memory/disk evidence and platform/filesystem recovery beyond this
  Windows validation; do not infer crash durability from ordinary round trips.
- [ ] Port aggregate workspace batch budgeting, fair sharing, LRU eviction, and
  explicit no-workspace cleanup from AMDGPU to CUDA, with CUDA-specific pool/FFT
  behavior. [BATCH_HANDOFF.md](bench/BATCH_HANDOFF.md) supplies implementation and
  hardware-validation details. Verify stable VRAM across changing schedules,
  image sizes, precision, and repeated calls.
- [ ] Reprofile residual B-spline prefilter allocations on production workloads.
  Prefer an upstream/public-API solution; any replacement must preserve documented
  numerical behavior and pass accuracy checks. Historical allocation figures in
  the archived feedback plan are starting evidence, not current measurements.

Acceptance: process and browse a recording larger than RAM; interrupt and resume
without losing committed results or mixing recipes; demonstrate bounded host
and GPU memory in the documented ownership modes.

## 4. Measurement history, uncertainty, and analysis

- [x] Record [final-sweep planar measurement history](docs/src/howto/measurement_history.md):
  raw primary values/residuals, first-observed rejection stages, accepted peak
  ranks, actual fill/restoration events, final origin/flags and numerical UQ
  status. Bind opt-in native companions to result content and reject callback
  mutation; preserve default numerical behavior and bounded sequence memory.
  [Regressions](test/test_measurement_history.jl) cover CPU/KA parity, ROI and
  scale, actual branch events, custom validators, persistence and failure cleanup.
- [ ] Extend measurement history to stereo/ensemble semantics, checkpoint
  companions and exports. Define any earlier-pass/sweep trace
  separately; final-grid events cannot reconstruct correspondence across grids.
  Validate estimator applicability separately from numerical UQ availability.
- [x] Expose actual planar pass sweeps, tolerance checks/stopping conditions and
  primary residual summaries through [execution diagnostics](docs/src/howto/execution_diagnostics.md).
  Opt-in callbacks and native sequence/replay companions preserve result structs
  and numerical behavior. Empty comparison support and unchecked budgeted sweeps
  remain explicit; primary residuals are not attributed to substituted/filled
  vectors. [Tests](test/test_execution_diagnostics.jl) cover CPU/KA equivalence,
  actual loop semantics, persistence, strict replay association and failures.
- [x] Extend execution diagnostics to ensemble pooled sweeps with their distinct
  iteration semantics. [Planar ensemble capture](docs/src/howto/ensemble_execution_diagnostics.md)
  records one sweep per pass, ignored iteration requests, actual source/plane
  contribution populations and pooled primary residuals before predictor addition.
  Separate native companions bind raw result fields without changing arithmetic.
  [205 focused checks](test/test_ensemble_execution_diagnostics.jl) cover CPU/KA
  Float32/64 parity, absent predictors, masks/UQ, tiled tails, callback/path guards,
  exact request snapshots and malformed companions, including unattainable
  contributor extrema and nonfinite-plane capacity. Vendor capture remains refused.
- [x] Integrate ensemble execution companions into GUI inspection and quality
  reports with explicit pooled-sweep populations and verification scope.
  [Format-4 reports](docs/src/howto/ensemble_quality_reports.md) verify raw
  companions, preserve prior planar/stereo report schemas and keep pooled
  contribution counts separate from stored-field and final-node populations.
  [GUI inspection and reports](docs/src/howto/gui_ensemble_companions.md) retain
  transactional navigation, physical-display integrity and prior report identity
  after failed requests. Whole-file requests validate and detach the complete
  native entry mapping before scanning or invoking GUI callbacks.
  [192 core report checks](test/test_ensemble_run_quality.jl) and
  [169 GUI checks](HammerheadGUI/test/test_ensemble_companions.jl) cover mixed
  and missing companions, malformed counts/indexes, captured options, protected
  destinations and bounded retention. Inspector/report captures were reviewed at
  three sizes. Contribution counts do not establish effective independent sample
  size, stationarity or uncertainty coverage; saved ensemble recipes remain open.
- [x] Add bounded per-camera stereo execution companions with dewarped-pixel
  residuals, explicit common-grid geometry, measurement-field binding and a
  separate native reader. Keep ensemble, GUI and report integration separate.
  The [stereo companion guide](docs/src/howto/stereo_execution_diagnostics.md)
  describes verification limits; [157 focused checks](test/test_stereo_execution_diagnostics.jl)
  cover CPU/KA parity, signed geometry, callback failures, persistence and cleanup.
- [x] Add opt-in execution-aware quality reports with explicit planar/stereo
  coverage and per-camera sweep/check/support counts. Preserve existing report
  defaults, avoid pooling pixel residual amplitudes across different grids, and
  distinguish generation-time binding checks from verification when loading a report.
  Version 3 retains per-role entry coverage and execution/support counts;
  existing default and history-only schemas stay unchanged.
  [130 focused checks](test/test_quality_execution.jl) cover mixed/missing
  companions, raw stereo binding, malformed counters, source protection and
  verified experiment associations without inventing planar measurement binding.
- [x] Generate a [saved run-quality report](docs/src/howto/run_quality.md) shared
  by scripts and the GUI for stored planar/stereo fields. Version-1 TOML reports
  preserve explicit node-weighted counts/denominators, numerical uncertainty
  availability, source/recipe/run provenance, and unavailable diagnostic reasons.
  Reports scan bounded result payloads and protect known source/record aliases.
- [x] Add opt-in [history-aware native quality reports](docs/src/howto/run_quality.md) and [lazy GUI companion
  inspection](docs/src/howto/gui_companions.md).
  Verify history against raw results before physical display;
  distinguish recorded/missing/unsupported entries, actual events and final
  origins, and retain explicit history-covered denominators. Keep default
  version-1 reports compatible and navigation failures transactional.
  [Core report tests](test/test_quality_history.jl) and [GUI tests](HammerheadGUI/test/test_companions.jl)
  cover association, missing coverage, physical display, memory release and layout.
- [ ] Extend reports beyond final-sweep planar history with uncertainty
  association, peak locking, and representative window-size/preprocessing
  sensitivity comparisons after the required measurement evidence is available.
  Current flags and finite values must not be presented as replacement counts.
- [ ] Add per-particle position and displacement uncertainty with validated
  detection-fit and match semantics, then add GUI uncertainty overlays.
- [ ] Propagate uncertainty into derived quantities after specifying spatial
  error-correlation assumptions (Wieneke 2015 §3.2). Distinguish correlation
  random error from calibration, timing, and other uncertainty contributions.
- [x] Expose the actual neighboring stencils used by planar derivatives, with an
  explicit policy requiring two-sided support when requested. Separate input-node
  eligibility, stencil geometry and finite derivative output; validate axes and
  preserve existing arithmetic on supported grids. Describe irregular-grid
  secants without implying higher-order accuracy or uncertainty calibration.
  The [support guide](docs/src/howto/derivative_support.md) and
  [362 focused checks](test/test_derivative_support.jl) cover independent legacy
  parity, masks/gaps, descending axes, native overflow, subnormal quotients and
  physical units. Existing analysis/validity checks also pass.
- [x] Show derivative support and unavailable neighborhoods in the result
  explorer after the core stencil contract is established.
  Keep an explicit stencil policy consistent across explorer tools and area
  circulation; component profiles remain independent. Show discrete support
  maps and selected contributors, distinguish current flags from recorded
  replacement history, and retain only
  current-frame metadata without repeating physical conversion.
  The [GUI support guide](docs/src/howto/gui_derivative_support.md) and
  [116 focused GUI checks](HammerheadGUI/test/test_derivative_support.jl) cover
  consistent policies, discrete legends, contributor details, mouse routing,
  valid transition notifications, mutation checks and bounded retention.
  Core derivative/circulation checks pass 369/369; the full GUI suite passes
  1,324/1,324, with refreshed support/scalar captures visually inspected.
- [ ] Evaluate confidence estimates for means and Reynolds stresses with explicit
  assumptions about temporal dependence, finite samples, and measurement noise.
- [x] Export `TrackingResult` in a language-neutral table with trajectory IDs,
  observed frame/time indices, gaps, validity, and units. Eight additive CSV
  columns preserve the existing schema prefix; derived elapsed time is labeled
  explicitly and assumes uniform frame spacing when scaled. Acquisition timestamp
  persistence remains in slice 2. [Tracking export tests](test/test_tracking_export.jl)
  cover gaps, validity, scaling, empty/singleton tracks, and existing result types.

Acceptance: a saved/exported value can be traced to its measurement or replacement;
reports identify unconverged or unsupported uncertainty cases; estimator claims
are checked against known truth or controlled statistical examples.

## 5. Cross-platform GUI framework and complete workflows

Reopen the original pure-GLMakie framework decision. Keep numerical work and
application state in the existing Julia controllers; use Makie for scientific
visualization. Richer forms, tables, experiment navigation, keyboard interaction,
and window management justify evaluating a dedicated application toolkit.

### Framework candidates

This shortlist is based on upstream documentation checked on 2026-10-02. An
[isolated Qt/QML prototype](docs/src/explanation/gui_framework.md) now supplies
initial Windows resolver/controller/framebuffer evidence and a concrete native
shutdown failure. Platform and packaging support remain to be proven for the
assembled application.

| Candidate | Fit and evidence | Main questions for the prototype |
|---|---|---|
| **Qt 6 / Qt Quick via QML.jl, with QMLMakie** | First candidate to prototype. QML.jl exposes Julia item models, callbacks, and Observable-backed property maps; Qt Quick supplies application controls. QMLMakie embeds interactive accelerated Makie plots. | Event-loop integration, cancellation/UI responsiveness, QML resource deployment, accessibility, and bridge stability. Floating/dockable panes are a requirement to evaluate, not an assumed QML feature. |
| **GTK4 via Gtk4.jl and Gtk4Makie** | Alternative desktop shell retaining GLMakie. Upstream reports Windows, macOS, and Linux operation. | Embedded `GtkMakieWidget` is labeled experimental, and the bridge depends on Makie internals; test embedding, upgrades, platform behavior, and distribution. |
| **Bonito with WGLMakie** | Browser-based alternative if remote access becomes a priority; supports interactive Julia-backed applications. | Local file workflows, Julia server/session lifecycle, large-image transfer and rendering, offline desktop packaging, and browser accessibility. Static HTML export alone cannot execute new PIV jobs. |

Sources: [QML.jl](https://juliagraphics.github.io/QML.jl/dev/),
[Qt Quick Controls](https://doc.qt.io/qt-6/qtquickcontrols-index.html),
[QMLMakie](https://github.com/JuliaGraphics/QMLMakie.jl),
[Gtk4Makie](https://github.com/JuliaGtk/Gtk4Makie.jl),
[Bonito deployment](https://bonito.sh/stable/deployment.html), and
[WGLMakie](https://docs.makie.org/stable/explanations/backends/wglmakie).
QMLMakie's inspected [compatibility declarations](https://github.com/JuliaGraphics/QMLMakie.jl/blob/master/Project.toml)
include Julia 1.10, GLMakie 0.13, and Observables 0.5, matching the corresponding
HammerheadGUI requirements. The prototype separately records the exact versions
resolved and exercised on Windows; compatibility ranges alone are not runtime
evidence.

- [x] Define the desktop interaction requirements: experiment/file browser,
  editable parameter tables/forms, errors next to inputs, keyboard navigation,
  HiDPI, menus, and resizable panels. Include separate settings and visualization
  windows, consistent with the original design preference, and compare with an
  integrated layout using the same controllers.
  The [requirements matrix](docs/src/explanation/gui_framework.md) specifies
  acceptance exercises and distinguishes implemented wiring, automated evidence,
  and pending desktop/platform checks.
- [ ] Prototype a Qt/QML batch form and existing-controller result explorer,
  including image pan/zoom, dense vectors, picking, mask gestures, live results,
  cancellation, and window close/reopen. Compare against the current GLMakie
  baseline; assess GTK4 and Bonito against the same requirements if needed.
  The isolated candidate reuses controllers; historical native captures on
  Windows encountered context/teardown errors but did not verify Qt's effective
  platform. The enforced offscreen RHI/OpenGL probe cannot create a context on
  this machine. Complete lifecycle, native input, and responsiveness evidence
  before claiming this prototype requirement complete.
- [x] Add a reproducible native rendering/lifecycle harness with a parent-owned
  child process, complete logs and final exit status, source identity, framebuffer
  evidence, and explicit viewport ownership/release checks. Separate application
  observer disposal from native GL cleanup; gate longer transition/resize trials
  on a clean single-render/release/exit result.
  Ownership and process-harness tests pass; the software shell completes five
  fresh viewport generations/releases with readable controls. The native gate
  remains blocked by this machine's offscreen context-creation failure, so no
  clean native lifecycle or cross-platform support is claimed.
- [x] Connect the isolated Qt software shell to real saved planar experiments:
  intact recipe/history inspection, captured replay and cancellation, verified
  lazy results with physical units, and explicit displayed-run identity after
  failed actions. Validate application ownership and shutdown without relaxing
  the native rendering/lifecycle gate.
  The pre-worker [prototype](HammerheadGUI/prototypes/qml/README.md) passed 250
  focused checks and 24 harness checks. Demo and saved-experiment software
  children exited cleanly after five viewport generations each, with inspected
  physical-unit captures, no remaining subscriptions and no replay at disposal.
  Native input, accelerated Qt rendering and other-platform support remain open.
- [x] Evaluate Qt controls with a separately owned interactive GLMakie window.
  The opt-in `--plot=glfw` mode serializes Qt and GLFW event servicing, preserves
  replay across view closure, and owns each screen through final disposal.
  The pre-worker evaluation passed 87 focused checks and 38 harness checks.
  Demo and saved-experiment children, plus both static-preview children, exited
  cleanly with unchanged source identities and inspected captures. Programmatic picking and
  view changes do not establish native desktop input or responsiveness.
- [x] Move saved-planar replay in the Qt prototype outside the Qt/GLFW event
  thread using an owned core-only Julia subprocess. Capture complete requests,
  bound progress/status messages and preserve ordinary written-pair cancellation,
  failed-prefix metadata, selected/displayed run identities and verified browsing.
  Keep controls and plot servicing active during computation and shutdown; missing
  terminal metadata or child exit alone must not imply completed processing.
  Repair compact sidebar scrolling/layout and verify active-work control handling,
  direct/worker parity, cancellation/error boundaries and child/window disposal.
  Record workload-specific event-pump gaps; desktop input, other platforms and
  embedded native rendering remain separate acceptance gates.
  The 64-bit Windows implementation uses a kill-on-close Job Object and refuses
  unsupported ownership platforms. Direct/worker parity, cancellation, failed
  history saves, malformed messages, startup failure and owner-loss checks pass.
  The hidden active-worker trial verifies retained physical display, control and
  plot servicing, compact scrolling, full error details and confirmed cleanup.
  Both saved-experiment software/GLFW lifecycle children also exit cleanly on
  the same final source; the broader platform and desktop gates remain open.
  The active trial's maximum pump gap was 1.051 s across the whole interval and 0.036 s
  after the first write acknowledgement; the [evidence guide](HammerheadGUI/prototypes/qml/README.md)
  records the small fixture and timing exclusions, without a desktop latency claim.
- [x] Add a manually triggered three-platform prototype evidence workflow.
  The [workflow guide](HammerheadGUI/prototypes/qml/ci_validation.md) describes
  Julia 1.11 on Ubuntu, Windows and macOS, bounded process ownership, retained
  failure artifacts and the separate native rendering prerequisite. Local
  process-owner checks pass; the workflow has not been dispatched, so platform
  results and the full supported Julia-version range remain unverified.
- [x] Add Qt file pickers for saved experiments, native results and replay
  destinations. Accepted selections update draft paths; Open and Replay remain
  explicit actions. Preserve drafts and the current display on cancellation or
  stale dialog events, use Qt local-file URL conversion, and retain protected
  output checks. Verify actual hidden dialog acceptance/cancellation and compact
  layouts; native OS dialogs, accessibility and network shares need separate
  evidence. This does not complete the broader experiment/file browser.
  The Windows offscreen dialog child passed all 30 stages, including four actual
  choices, modal shortcuts, stale callbacks and shutdown. The 129 focused checks
  and saved-planar software regression pass; both children used identical source
  maps. Modal and compact-layout captures were visually inspected. The
  [prototype evidence](HammerheadGUI/prototypes/qml/README.md) records scope and
  retained artifacts; native desktop and broader platform gates remain open.
- [ ] Resolve the candidate environment against supported Julia/Makie versions;
  validate Windows, macOS, and Linux, startup latency, memory, input/HiDPI behavior,
  and responsiveness during CPU/GPU work. Check accessible labels and focus order.
  Establish replay-worker ownership on each platform, including abrupt owner
  loss during processing and confirmed descendant exit before another launch.
  A platform-specific ownership implementation does not close this gate.
- [ ] Record the framework decision, dependency/maintenance cost, distribution
  requirements, and migration sequence. If selected, migrate one tool at a time
  with controller parity and rendering tests; keep toolkit dependencies in the
  GUI package. Evaluate rendering reuse rather than assuming existing GLMakie
  widget layouts transfer unchanged.
- [ ] Connect setup, preprocessing, mask/ROI, calibration, representative-pair
  comparison, batch, saved quality reports, and export through the shared
  experiment record from slice 2.
  The first [saved planar GUI workflow](docs/src/howto/gui_experiments.md)
  snapshots supported batch/preprocessing/mask/ROI/scale settings and reopens
  complete recipes in a separate read-only controller. It replays exact settings,
  retains run history, and verifies completed output content before lazy browsing.
  It also saves and displays the shared core quality report with the same
  provenance checks and metric definitions used by scripts.
  Its separate checkpoint view adds resumable built-in processing with committed
  progress and cancellation. Stereo calibration and broader integration remain open.
- [x] Make scalar-field labels distinguish raw displacement magnitude from scaled
  speed. Keep the existing physical-unit conversion and neutral component labels;
  a quantity labeled displacement must not carry length/time units.
- [x] Make the production saved-experiment workflow usable at smaller window
  sizes. Keep replay cancellation/progress/status reachable, preserve hidden
  section settings and report identity, and adapt text paging to available space.
  Verify long paths/errors and all actions at 1100×800 and 900×600 offscreen;
  retain separate native desktop and accessibility acceptance checks.
  Files/Replay/Reports sections keep cancellation and status visible while
  allocation-sized pages retain full paths, errors and report identities.
  [Layout checks](HammerheadGUI/test/test_workflow_layout.jl) cover actual mouse
  routing and persistent settings; 137 focused GUI assertions and 36 real
  saved/refused-report checks pass, with both window sizes visually inspected.
- [x] Inspect recorded stereo camera execution in the production GUI and expose
  execution-aware saved reports. Verify raw fields before physical conversion,
  retain transactional navigation and label dewarped-pixel diagnostics distinctly
  from world-coordinate fields and unavailable per-node history.
  [Stereo/report tests](HammerheadGUI/test/test_stereo_companions.jl) exercise
  supplied-raw verification, retained-camera display integrity, transactional
  navigation, missing companions, captured report options and protected saves.
  The combined focused GUI checks pass 242 assertions; offscreen stereo and
  saved-report layouts were inspected. Native desktop validation remains separate.
- [x] Keep result picking available when the Qt shell switches from demo mask
  drawing to a saved/native result; the disabled demo-only mode must not intercept
  inspection clicks.
  Native-file and saved-experiment regressions exercise stale drawing mode;
  250 prototype checks, 24 lifecycle-harness checks and both final software
  children pass. Native Qt rendering/input and other-platform gates stay open.
- [x] Add completed-pair progress and cooperative cancellation to ordinary saved
  GUI replay. Capture the full request before notifications, retain original
  errors, wait for loader cleanup, and distinguish native prefixes from resumable
  checkpoints and final-pair completion from cancellation.
  The [replay workflow](docs/src/howto/gui_experiment_replay.md) has
  [28 core checks](test/test_experiment_replay_progress.jl) and
  [77 GUI checks](HammerheadGUI/test/test_experiment_replay_progress.jl), including
  startup/terminal observer failures and default-size offscreen layout review.
- [x] Add a saved-recipe GUI comparison for an explicitly selected ordered image
  pair. Share core recipe/input verification, populations, value bases and report
  persistence; show the saved report's own identities after a failed rerun.
  Keep differences between recipes distinct from accuracy measurements.
  The [comparison workflow](docs/src/howto/gui_comparison.md) has
  [57 focused checks](HammerheadGUI/test/test_recipe_comparison.jl) for captured
  requests, failure recovery, protected outputs, report identity and view controls.
- [x] Inspect dedicated actual-time tracking artifacts in the GUI while retaining
  timing metadata, actual-time units, trajectory selection and explicit gap semantics.
  The [timed explorer](docs/src/howto/gui_tracking_timing.md) has
  [69 GUI checks](HammerheadGUI/test/test_tracking_timing_explorer.jl) and
  [33 core checks](test/test_tracking_speed_summary.jl) for bulk speed summaries,
  unavailable tracks, metadata integrity and dedicated artifact loading.
- [x] Add editable GUI ROI selection using the existing core `ROI` semantics.
  The editor supports two-corner selection, numeric bounds, and full-image reset;
  batches preserve mask and coordinate semantics and snapshot the selected ROI.
  Effort presets fit the ROI; oversized custom schedules fail before output opens.
  Covered by controller and offscreen view tests in the
  [GUI suite](HammerheadGUI/test/runtests.jl).
- [ ] Evaluate a PackageCompiler desktop bundle with the chosen toolkit:
  relocatable assets/libraries, installer size, cold start, clean-machine launch,
  and redistribution requirements. Add platform smoke tests and a documented
  installation path for users without an existing Julia environment.

Acceptance: record a measured framework decision and demonstrate one complete
experiment workflow on Windows, macOS, and Linux before declaring a replacement
shell supported. The prototype and decision can precede the full migration.

## 6. Calibration, ingestion, and evidence-dependent extensions

These remain candidates or bounded follow-ups, rather than prerequisites for
the reproducibility and GUI work above.

- [ ] Harden rolled-target detection with synthetic perspective/noise/marker
  visibility/two-level fixtures and a real rotated-target regression. Preserve
  the world-axis/origin conventions; `orientation = :fiducials` already provides
  roll-invariant indexing when both markers are visible.
- [ ] Add optional video ingestion through a weak dependency or frame-source
  adapter, preserving source order and timing.
- [ ] Evaluate HDF5 or netCDF after the table/VTK contracts have user feedback;
  define a concrete archival/interoperability need before adding a schema.
- [ ] Evaluate variance-normalized cross-correlation on strong illumination and
  contrast changes; compare against phase correlation and CLAHE.
- [ ] Extend independent search footprints to KA/CUDA/AMDGPU if benchmarks justify
  it, preserving CPU geometry, masks, output coordinates, and uncertainty semantics.
- [ ] Consider adaptive/nonuniform interrogation after search-area and output-grid
  semantics stabilize and a resolution benchmark demonstrates the benefit.
- [ ] Revisit ensemble `max_iterations` only if low-SNR accuracy cases justify
  the repeated full-sequence correlation cost; currently it is ignored.
- [ ] Evaluate dynamic per-peak exclusion radii against the existing regional-max
  and fixed-exclusion finders, including peak-ratio and substitution semantics.
- [ ] Evaluate relaxation-method particle matching and Duncan et al.'s
  distance-weighted scattered-UOD variant against the current greedy/plain methods.
- [ ] Evaluate stereo PTV with independently validated correspondence and geometry.
- [ ] Add optional morphological phase separation when a validated two-phase
  image use case is contributed.
- [ ] Estimate light-sheet thickness/overlap from disparity-correlation peak
  widths after validating the Wieneke 2005 §5 model on a stereo fixture.
- [ ] Profile device dewarping and preprocessing (background/highpass first,
  CLAHE later); move validation/replacement/smoothing only if dense-grid costs
  justify it. Preserve masks, signs, precision, and avoid extra host/device copies.
- [ ] Consider additional GPU backends such as Metal or oneAPI only with a user
  need, compatible FFT/UQ capabilities, and available hardware validation.

Acceptance: each extension identifies the failing baseline case, demonstrates
an improvement with stated tradeoffs, and updates the feature matrix and docs.

## Planning and release practice

Use this file for open work; archives preserve past reasoning and measurements.
Completed entries may be summarized in the baseline or release notes instead of
accumulating duplicate checklists.

- [x] Establish [release notes](CHANGELOG.md) for API/native-format changes and
  document [core-first/GUI-second release checks](RELEASING.md), including
  dataset, hardware, persistence, and desktop evidence. Update these records
  with each release; documented checks are not claims of completed validation.
- [ ] Exercise the saved-experiment and desktop workflow with a lab user and
  record concrete friction before promoting optional features into delivery work.
