# CLAUDE.md

Hammerhead.jl — particle image velocimetry (PIV) in Julia. Development is
organized around the International PIV Challenge cases; see [ROADMAP.md](ROADMAP.md)
for the single active backlog, delivery order, and acceptance criteria. Historical
phases live in `reference/archive/ROADMAP.md`. Scope is capped at planar 2D2C + stereo 2D3C (tomographic
PIV is out of scope). All five phases are done: 1 (file I/O & batch),
2 (masking), 3 (ensemble correlation & time-series statistics),
4 (accuracy/UQ), and 5 (stereo: camera calibration, target detection,
dewarping, 3C reconstruction via `run_piv_stereo` → `StereoPIVResult`, and
Wieneke 2005 disparity self-calibration via `self_calibrate`) — every planar
and stereo Challenge case is now reachable. Case 4E data (particle +
calibration images) sits in `cases/` (gitignored); a minimal 4E subset
for the docs and tests is committed at `test/reference_images/E/`. Phase 6 (Diátaxis docs,
July 2026) is also done. Hammerhead and HammerheadGUI are registered in
General; user installation instructions should use `pkg> add Hammerhead`
and `pkg> add HammerheadGUI`. Phase 7 (HammerheadGUI) is underway:
the monorepo conversion and CI/TagBot/CompatHelper subdir wiring are done;
the result explorer, mask editor, batch forms, and calibration diagnostics
are available. Phase 8 (2D2C PTV,
July 2026) is done: per-frame particle detection (`detect_particles`),
hybrid PIV-guided two-frame tracking (`run_ptv` → `PTVResult`, with
`ptv_to_grid` binning and `run_ptv_sequence` batch), scattered validation,
multi-frame trajectory linking (`track_particles` → `TrackingResult`), and
docs — all synthetic-verified, no new deps.

## Commands

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'   # full suite, including subprocess recovery checks
julia --project=docs docs/make.jl                    # docs: executes all seven tutorials ("skipping deployment" warning is normal locally)
julia --project=HammerheadGUI -e 'using Pkg; Pkg.test()'  # GUI tests (needs a GL context; CI wraps in xvfb-run)
julia --project=. --threads=4 bench/validation_scorecard.jl  # provenance + synthetic accuracy / real-data smoke report
```

`PIV sequence failed` error logs from the intentional failure-propagation
tests are expected and do not indicate failed assertions.

Core CI covers single-threaded Ubuntu on LTS/stable/prerelease Julia and
four-threaded stable Julia on Ubuntu and Windows. Lifecycle regressions also
exercise failed/cancelled batches: their prefetched loaders must finish before
the driver returns, while the original failure remains the reported exception.

## Documentation (docs/)

Diátaxis layout under `docs/src/`: `tutorials/` (generated — do not edit),
`howto/`, `explanation/`, `reference/`, plus `index.md` and `references.md`
(bibliography). Rules that keep the build green:

- Organize the public site around reader tasks and a short learning path.
  Tutorials lead with a concrete question, executable example and useful figure,
  then explain the result and invite one change to try. Hide only rendering
  machinery; all inputs needed to run visible example code must be introduced.
  Do not turn implementation batches into new top-level navigation entries.
  Keep detailed pages searchable and linked through the topic hubs in `make.jl`.
  File-format versions, field schemas and edge-case contracts belong in API
  reference; test inventories and framework evaluations belong in development
  material. Page titles describe a reader's task, not an internal delivery slice.
- Explain what a command does, what a result measures, and how the reader uses
  it in direct, affirmative language. Include a limitation when it changes an
  interpretation or next action, and place it with that decision. Avoid habitual
  negative caveats about unrelated claims, guarantees, unsupported scenarios or
  hypothetical misunderstandings. Define populations and artifact identities
  directly instead of repeatedly stating what they are not.
- Tutorials are Literate.jl sources in `docs/lit/*.jl`; `make.jl` converts
  them into `docs/src/tutorials/` (gitignored) with executable `@example`
  blocks, so the docs build runs them end to end — they are integration
  tests. Executed doc code must never reference `cases/` (gitignored);
  synthetic data and committed fixtures are the only inputs — the
  real-data tutorials load committed PIV Challenge subsets via
  `pkgdir(Hammerhead)`: the case-A pair from `test/reference_images/A/`
  and the case-4E stereo slice from `test/reference_images/E/` (cameras
  1 + 3, calibration planes z = −3/0/+3 mm, frames 50–51, losslessly
  re-encoded 16-bit PNG). The 4E calibrations are fit to those images'
  pixel coordinates — never crop or re-encode them independently.
- Reference pages use `@autodocs` filtered by source file (`Pages =
  ["pipeline.jl", ...]`). A new `src/*.jl` file's public docstrings must be
  added to one of the reference pages (and every documented binding must
  appear somewhere) or `makedocs` fails its checkdocs pass.
  `Pages` uses suffix matching; use `"src/quality.jl"` rather than `"quality.jl"`
  when another source such as `run_quality.jl` shares that suffix.
  `reference/internals.md` catches all non-exported docstrings via
  `Public = false`.
- Citations: DocumenterCitations with `docs/src/refs.bib` (authoryear
  style); cite as `[Wieneke2005](@cite)` / `[Wieneke2015](@citet)`. PDFs
  for content-checking live in `reference/`.
- Docs-only deps (Literate, DocumenterCitations, CairoMakie, Unitful for the
  scaling how-to) live in `docs/Project.toml`; the core package must not
  gain doc/GUI deps.
- Explanation pages are the user-facing rewrite of the conventions below —
  when a convention changes, update both.

## Architecture (src/, included in this order)

- `types.jl` — `PhysicalScale` (pixel size + dt + display-only unit labels;
  Float64 factors, validated), `PIVParameters` (immutable, validated in inner
  constructor; `keep_correlation_planes` opts into per-window plane storage),
  `PIVResult{T}` (trailing `correlation_planes` and `scale` fields, `nothing`
  unless supplied; backward-compatible 11-/12-arg constructors keep old call
  sites valid — every result type uses the same trailing-`scale` trick)
- `synthetic_data.jl` — synthetic particle images with ground truth
- `preprocessing.jl` — background subtraction, intensity cap, highpass, CLAHE,
  percentile contrast stretching, inversion, and local-variance normalization
- `correlators.jl` — `CrossCorrelator{T}`/`PhaseCorrelator{T}` cache FFTW
  plans + buffers per window size; subpixel peak fits
- `uncertainty.jl` — Wieneke 2015 correlation-statistics uncertainty
  (per-window `accumulate_uncertainty!` + `finalize_uncertainty`)
- `transforms.jl` — affine transforms, image warping, registration
  (manual fits validate finite Float64 coordinates, affine rank and invertibility;
  fit residual acceptance remains the caller's responsibility)
- `calibration.jl` — `PinholeCamera` (normalized DLT) / `SoloffCamera`
  (19-term polynomial) / `TransformedCamera` (rigid world pre-transform
  wrapper), `calibrate_camera`, `world_to_pixel` / `pixel_to_world`,
  `apply_world_transform`, fit-quality metrics
- `planar_calibration.jl` — `PlanarTransform` + two-point
  `planar_calibration` with explicit rotation/reflection/anisotropic scaling
- `target_detection.jl` — `detect_calibration_grid` (dot-grid plates →
  indexed point pairs), `calibration_points`, `calibrate_camera` /
  `calibration_quality` convenience methods on `(grids, zs)`,
  `render_calibration_target` (synthetic ground-truth fixture);
  `orientation = :fiducials` makes indexing roll-invariant when both square
  and triangle markers are visible (`:image` remains the upright default)
- `dewarp.jl` — `DewarpGrid` (world-plane pixel grid spec, shared per rig) +
  `ImageDewarper` (per-camera precomputed source-coordinate map), `dewarp[!]`
  cubic B-spline resampling onto the common plane, `common_dewarp_grid`
  (auto grid from camera footprints: intersection/union, `:auto` spacing,
  descending `y`)
- `quality.jl` — UOD, peak ratio, correlation moment, validator pipeline,
  `replace_vectors!`, `smooth_field`
- `masking.jl` — `polygon_mask`, intensity/contrast/edge `automatic_mask`,
  and circular `grow_mask`/`shrink_mask`
- `source_support.jl` — lazy original-pixel stencil evidence for deformed
  windows, using exact comparisons in processing precision. Compact UInt8 maps
  identify constant/empty/variable raw stencils; uninformative pair contributions
  are skipped consistently by CPU/shared KA correlation and uncertainty paths.
- `pipeline.jl` — `run_piv`, `piv_pass` (WIDIM multi-pass with symmetric
  image deformation; a pass with `max_iterations > 1` iterates against its
  own validated field until the *q95* per-vector change drops below
  `convergence_tol` — a max-norm never converges, bistable low-signal
  windows flicker between peaks forever; sweeps always force-replace
  internally, with measured values restored at still-flagged cells when
  `replace_outliers = false`; final-pass UQ then runs as a post-loop
  `uncertainty_sweep!` over the last sweep's deformed windows, bitwise equal
  to the fused single-sweep path; an iterated stage with `tol = 0` is
  exactly ≡ repeating the pass — tested; the ensemble path ignores
  `max_iterations`), `process_windows!` (receives its per-chunk correlator),
  `multipass_parameters` (`final = (;)` overrides the last pass only),
  `effort_schedule` (internal builder for `effort = :low/:medium/:high` on
  `run_piv`, `run_piv_sequence`, `run_piv_ensemble`, and `run_piv_stereo`;
  planar presets use the ROI dimensions when supplied; high effort includes
  final-pass UQ, and ensemble high repeats the final
  window because ensemble ignores `max_iterations`),
  `PIVWorkspace`/`piv_workspace()` (optional `workspace` kwarg reusing the
  padded B-spline coefficient buffers via `image_interpolant!`+`interpolate!`,
  the deform buffers, and a per-window-config correlator pool across `run_piv`
  calls — bitwise-identical; the sequence/ensemble drivers hold one).
  Singleton predictor axes extend constantly during deformation and vector
  attribution, so coarse windows may fill an image or ROI dimension.
- `ka_backend.jl` — portable KernelAbstractions correlation/analysis kernels
  + the built-in `backend = :ka` engine that runs them on the KA CPU backend
  (details and GPU kernel conventions under the GPU-extension bullet below)
- `particles.jl` — `Particles` (struct-of-arrays) + `detect_particles`
  (local-maxima + 3-point log-Gaussian fits) and the shared uniform-cell
  neighbor list (`build_cell_list`/`within_radius!`/`knn`) used by dedupe,
  matching, and scattered UOD
- `ptv.jl` — `PTVParameters`, `PTVResult`, `run_ptv` (hybrid PIV-guided
  greedy matching via `greedy_match`, optionally weighted by intensity and
  diameter consistency), `scattered_uod`, `ptv_to_grid` (`bin_to_grid`)
- `tracking.jl` — `Trajectory`, `TrackingResult`, `track_particles`
  (constant-velocity + field predictor linking, bounded gap bridging;
  trajectories retain explicit frame indices; frames are loaded/detected one
  at a time, retaining the first pair only for the initial PIV predictor),
  `trajectory_velocities`
- `stereo.jl` — `StereoPIVResult` + `run_piv_stereo` (per-camera 2C on
  dewarped images → geometric least-squares 3C reconstruction with
  uncertainty propagation), synchronized `run_piv_stereo_sequence`, and
  per-camera-correlation `run_piv_stereo_ensemble`. Sequence/ensemble drivers
  check matching exposure timestamps before loading or opening output;
  `sync_atol`/`sync_rtol` scale tolerance by pair delay, never clock epoch.
  `missing_timestamps = :error` requires metadata; default `:allow` preserves
  path/matrix workflows without claiming synchronization. Declared `FramePair.dt`
  must agree with available source timestamps within the same tolerance.
- `scaling.jl` — `with_scale` (attach/strip `PhysicalScale` metadata,
  arrays shared) + `physical` (same-type conversion to physical units) +
  `plot_axis_labels` (Makie-free label helper) for all four result types
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` 1 + `results/000001`… + optional `sources/…`;
  entries may be `PIVResult`, `StereoPIVResult`, `PTVResult`, or
  `TrackingResult`; `ResultFile(path)` / `load_results(path; lazy = true)`
  indexes sorted keys and loads one entry per access, without a retained
  payload cache or open handle. The index is for completed files; size/mtime
  checks reject detectable changes but do not support concurrent writers. The
  pre-registration dev formats were retired without a load shim when the
  `scale` field landed),
  `run_piv_sequence`/`run_ptv_sequence` batch drivers (shared `_run_sequence`;
  `output` accepts a single path or an `(i, pair) -> path` function for
  per-pair files; `collect_results = false` returns `nothing` while delivering
  each result to `on_result` and output, then releases its reference (also
  supported by the stereo sequence driver); the next pair's load+preprocess is prefetched on a
  `Threads.@spawn` task while the current pair's `process` runs — overlaps
  slow-source IO with compute only under ≥2 threads, results bitwise-identical
  to serial; `run_piv_sequence` also holds one `PIVWorkspace`, reused across
  pairs — the workspace lives only on the serial `process` call, never the
  prefetch task), `frame_index_strings` (differing frame-index substrings
  from a path pair)
- `interoperability.jl` — `ROI`; lazy `AbstractFrameSource` / `FrameSource` /
  `FrameRef` / timestamped `FramePair` / `TIFFStack`; flexible stride, offset,
  multi-delay pairing; dynamic static/per-frame/per-pair/callback masks with
  pair-union semantics; stable long-form `export_table` CSV and structured-grid
  `export_vtk`. Tracking CSV uses additive columns for trajectory/observation
  IDs, original frame indices, derived elapsed time, gaps, and numerical validity;
  no gap rows or acquisition timestamps are invented. Elapsed time assumes
  uniform input-frame spacing when derived from `PhysicalScale.dt`.
  Raw planar-grid exports accept `PlanarTransform` with explicit units and
  optional pair delay. Attached scales are rejected to avoid double conversion;
  mixed-axis uncertainty is unavailable unless independence is explicitly
  assumed. Geometry/topology and vector basis are transformed together.
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor; one
  `PIVWorkspace` reuses the interpolant/deform buffers across pairs)
- `selfcal.jl` — `self_calibrate` (Wieneke 2005 disparity self-calibration:
  ensemble cam1↔cam2 disparity map → triangulation → sheet-plane fit →
  rigid world transform of both cameras) + `SelfCalibrationReport`
- `statistics.jl` — planar/stereo `field_statistics`, 2C/3C
  `validate_temporal!`, `power_spectrum`; `FieldStatisticsAccumulator` with
  `update_statistics!` and independent `field_statistics(acc)` snapshots
  computes population moments with O(grid nodes) retained memory. Updates
  validate grid/component dimensions and scale factors/unit labels before
  mutating; convert inputs through `physical` for velocity statistics.
- `derived.jl` — mask-aware derivatives, vorticity/divergence/strain,
  swirling strength/Q, profile/region extraction, circulation, and
  results-vector spectra with an explicit sampling interval or validated sample
  times (`dt` or `sample_times`, independent of the image-pair delay in
  `PhysicalScale`). Spectra check full grid/shape/scale compatibility first;
  actual times require interval and global-grid agreement within explicit
  period-scaled tolerances. Profile interpolation ignores
  invalid corners with zero weight at exact nodes/edges. `extract_region`
  returns `included` (`true` = returned valid node); its legacy `mask` field
  aliases that grid and has the opposite convention from `PIVResult.mask`.
  `flow_derivatives` keeps its five-field default return; `return_support=true`
  adds contributor indices, spans, weights and separate structural/finite masks.
  `stencil=:centered` refuses one-sided fallbacks. The two-neighbor secant is
  not a general second-order formula on irregular axes. Validate native spans
  before calculation; unavailable reciprocal metadata must not discard a valid
  direct quotient. Support describes stored values, not measurement history/UQ.
  Area `circulation(result; region=...)` now errors on incomplete coverage
  by default; `coverage=:report` returns value, valid/requested area, fraction,
  and authoritative `complete` flag (no valid area gives `NaN` value).
  Its `stencil` keyword uses the same derivative policy as scalar analysis;
  centered-only support can reduce valid integrated area.
- `calibrated_resampling.jl` — CPU bilinear point sampling of raw planar
  vectors and scalar images onto explicit calibrated coordinates. Detached
  outputs retain contributor flags and joint vector availability; masks and
  numerical failures are distinct from measured zeros. Affine geometry and
  vector bases share one applied map. No measurement/UQ records are synthesized.
- `artifact_paths.jl` — portable lexical source-locator classification, explicit
  local-path resolution and prospective alias checks. Foreign provenance is not
  resolved against the receiving workspace; protect actual consumed local files.
- `experiments.jl` — versioned file-based planar `PIVRecipe`/`ExperimentRecord`,
  `PreprocessStep`, hash-verified `ScriptReference`, save/load, and noncollecting
  `replay_experiment` with optional `ExperimentRun` persistence. Recipes embed
  copied backgrounds/masks/ROI and full settings; inputs are content-addressed
  external files. CPU/KA Float32/Float64 only in version 1. Replay checks inputs,
  destination aliases, and creation/run environment compatibility before opening
  output; an environment change requires an explicit override. Scripts are never
  evaluated automatically. The experiment format is separate from result format
  1; unknown versions are rejected. Optional progress runs after each pair is
  written; callback failures retain the completed count and original exception.
  Rerunning is not resuming/checkpoint recovery.
- `execution_diagnostics.jl` — opt-in `PIVExecutionDiagnostics` and
  `PassDiagnostics` store scalar observations in immutable tuples. Preserve the
  existing loop: the final budgeted sweep is unchecked; max_iterations=2 has no
  tolerance comparison, and an empty comparison may satisfy the condition.
  Residuals describe primary correlation corrections before predictor addition,
  alternatives or filling. Planar sequence/replay may persist version-1 native
  companions without changing result structs. Callback delivery precedes result
  delivery/persistence; it is not a commit notification. Stereo has a separate
  companion below; ensemble diagnostics use their own pooled-sweep contract.
  Default calls do not collect diagnostics or hash source files.
- `measurement_history.jl` — opt-in final-pass/final-sweep planar history,
  separate from execution diagnostics and result structs. Record actual first
  rejection, alternative acceptance, fill assignment and restoration events;
  value differences and final flags cannot reconstruct these events. Bind the
  detached snapshot to raw result content including coordinates and scale before
  callbacks. Validate binding before persistence; metadata-only loads do not
  verify payload content unless `verify_result=true`. UQ still describes the
  final deformed windows and is not re-estimated for alternatives or fills.
- `ensemble_execution_diagnostics.jl` — opt-in array-free pooled-sweep packets
  for CPU/KA planar ensembles. Count actual pair/window contributions and
  degenerate planes before accumulation; primary residuals precede predictor
  addition. Every pass performs one sweep with no convergence checks, even if
  its requested iteration budget is larger. Predictor presence is observed,
  not inferred from pass index. Separate native companions bind raw result
  fields and scale; counts do not establish independent sample size or UQ
  coverage. Capture freezes the selected pairs, mask and pass schedule before
  image loading. Ensemble packets are refused by execution-aware quality reports.
- `stereo_execution_diagnostics.jl` — immutable two-camera execution companions,
  with distinct camera IDs, explicit parent/pair association, dewarped-pixel
  residuals and signed common-grid geometry. Measurement-field binding covers
  reconstructed and camera fields, excluding parameters and retained planes.
  Validate structure and binding after callbacks and before output publication.
  A separate native sibling group/reader leaves planar diagnostics unchanged;
  runtime verification status distinguishes metadata-only inspection from field
  checks and is never persisted as a verification claim.
  `execution_diagnostics_data(d; result=raw)` verifies an already loaded raw
  payload and returns detached metadata without retaining or reloading it.
- `pair_timing.jl` — opt-in planar sequence timing companions. Freeze selected
  frame references and O(pairs) scalar metadata before loading/output; exact
  rational encodings preserve integer epochs and timestamp midpoint halves.
  Observed, declared and effective scaling delays remain distinct. Source IDs
  are opaque and missing clock/unit labels remain unknown. Bind the current
  packet to raw numerical result content; generic copies/exports omit it.
- `stereo_pair_timing.jl` — separate opt-in stereo sequence timing companions,
  with ordered camera source descriptors, exact per-camera delays/midpoints,
  synchronization policy and effective reconstructed scaling provenance.
  Freeze all selected scalar metadata before loading/output. Pair-list scaling
  follows camera 1's declared delay; four-tuples retain the supplied scale.
  Capture timing and execution packets before either callback and recheck both
  before publication. Bind raw reconstructed and camera fields plus signed grid
  geometry; source bytes, calibration and hardware synchronization are not verified.
- `stereo_experiments.jl` — independent primitive stereo recipe/record and native
  run-association formats. Snapshot fitted Pinhole/Soloff cameras (or one rigid
  wrapper), signed common grid, raw camera sizes, complete CPU/KA settings and
  two built-in preprocessing pipelines. Decode Pinhole coefficients exactly
  through the private fitted-snapshot constructor; do not renormalize them.
  Rebuild and check maps, preserve exact ordered timing and pair-list versus
  four-tuple scale semantics, and run noncollecting replay. Scientific settings
  identity excludes descriptive calibration provenance and derived-map hashes;
  full snapshot integrity still protects them. Supplied world/frame labels do
  not infer scale factors. Verification streams raw results without rerunning
  PIV; calibration fitting, self-calibration execution, custom cameras/scripts,
  and checkpoints remain separate work. The GUI can inspect and replay the
  exact saved recipe through its dedicated controller.
- `ensemble_experiments.jl` — separate planar `EnsemblePIVRecipe`,
  `EnsembleExperimentRecord` and `EnsembleExperimentRun` schemas for a saved pool.
  Preserve ordered file inputs, exact CPU/KA passes, built-in preprocessing,
  full-image mask and scale. Input pairs, joined pair contributions across passes
  and the single published result have distinct counts. Private ensemble-driver
  hooks deliver immutable progress after joined work; cancellation can stop the
  last contribution before pooled analysis/publication. Snapshot before callbacks
  and run final integrity/path checks after the last user callback. Stage the
  complete artifact beside its destination and use native rename without a
  copy/delete fallback. Output and history are separate publications:
  `EnsembleRunRecordError` retains completed/cancelled run metadata when history
  saving fails; ordinary processing errors keep their original exception.
- `tracking_timing.jl` — explicit `TimedTrackingResult` owning an unchanged
  `TrackingResult` and exact `TrackingTiming` metadata. Default tracking remains
  ordinal. Actual-time prediction and scattered validation normalize elapsed
  intervals to the first transition; returned secants divide by actual time and
  use only the spatial scale. Validate binding before conversion or rebinding.
  Dedicated timed artifacts omit the generic native marker so older readers
  cannot discard timing silently; timed CSV preserves exact stamps and stencil
  support. `tracking_speed_summary` validates once for all trajectories and
  returns detached observation-mean secant speeds with unavailable reasons.
  GUI timed inspection keeps a dedicated singleton container; do not widen native
  result-vector element types or unwrap timing for display/persistence.
- `calibrated_table.jl` — separate CSV/TOML exports for raw PTV, ordinal tracking
  and actual-time tracking. Affine offsets affect positions only; transformed
  scalar match residuals are unavailable because their directions were not stored.
  Metadata retains settings/units even for empty tables and binds CSV content.
  Two closed staged files publish sequentially, without atomic-pair guarantees.
- `experiment_checkpoint.jl` — separate version-1 built-in planar checkpoint
  protocol: disjoint metadata/result directories, immutable singleton native
  payloads and commit descriptors, strict ordered input/recipe/environment
  identity, explicit interrupted-writer recovery, lazy committed-prefix access,
  and native aggregate export. Publish closed/validated files by same-directory
  rename with no copy fallback; do not claim power-loss durability. Cancellation,
  handled failure, and unmatched attempts have distinct metadata. Committed data
  counts come from descriptors; memory counters cannot define recovery state.
- `experiment_comparison.jl` — `recipe_diff` verifies both snapshots and returns
  deterministic field changes, retaining scalar/tuple settings and summarizing
  embedded mask/background payloads by shape, precision, and canonical SHA-256.
  Recipe comparisons exclude script locators, input identities, and run metadata.
- `run_quality.jl` — `RunQualityReport` scans planar/stereo results without
  retaining payloads and defaults to validated version-1 TOML. Opt-in native
  version-2 reports verify final-sweep history, report missing/unsupported entry
  populations and keep event/origin counts on history-covered denominators.
  Associated reports also require matching packet recipe/input/pair identities.
  Opt-in version-3 execution reports require a whole native index and retain
  planar/per-camera entry coverage, pass/check counts and final primary support.
  Verify stereo companions against each raw payload while it is loaded; planar
  execution packets remain entry-key linked. Do not pool residual amplitudes
  across grids or describe report loading as fresh measurement verification.
  Counts are node-weighted
  with explicit denominators; flags/finite values are not replacement history,
  and numerical UQ availability is not coverage or measurement association.
  Record/run reports verify output content and identities; anonymous iterators
  make no association claim. Protect hidden source paths explicitly when saving.
- `ensemble_run_quality.jl` adds explicit format-4 whole-native-file reports via
  `include_ensemble_execution_diagnostics=true`. Preserve v1/v2/v3 contracts and
  their existing ensemble refusal when this flag is false. Classify recorded
  pools, recorded planar iterations and entries without execution metadata;
  absence does not identify an ensemble. Verify packets against the already
  loaded raw payload, reject conflicting companion families on one entry, and
  separate all-pass window-pair observations from final-pass node/UQ populations.
  Repeated observations are not independent samples; do not aggregate residual
  amplitudes. Planar/stereo recipe-associated overloads refuse this option.
  Validate the complete sorted native entry mapping and detach its key vector
  before provenance/iteration. GUI whole-file reports use this guard for all
  formats and capture it before observable notifications or save dialogs.
- `ensemble_experiment_quality.jl` adds associated report format 5. Verify the
  completed pooled artifact, raw fields, recipe geometry and requested companions
  before/after aggregation. Keep input/contribution/publication counts separate
  from sequence `completed_pairs`. Validate v5 provenance independently, then
  reuse existing v1/v4 scientific-counter validators without changing their
  schemas. Saved reports describe past verification; they do not re-open inputs.
- `stereo_run_quality.jl` extends associated reports to completed frozen-camera
  stereo records. Snapshot record/run, verify raw fields/geometry/ordered sources
  and native companions before/after the streaming report; optionally check input
  bytes. Default v1 and execution v3 schemas remain unchanged; stereo history is
  refused. Explicit result relocation preserves foreign historical locators.
  Its execution-aware reports retain v3 semantics and refuse ensemble metadata;
  format-4 ensemble reporting uses a separate unassociated whole-file index.
- `pair_comparison.jl` — `compare_recipe_pair` reruns complete built-in planar
  recipes on explicitly selected content-matched inputs. Exact raw-coordinate
  intersections avoid resampling; native and paired quality populations remain
  separate. Pixel differences ignore attached scales; physical comparisons
  require identical factors and labels. Stable scalar moments and explicit
  arithmetic unavailability preserve denominators. Detached version-1 TOML
  snapshots protect known source aliases and do not claim accuracy/UQ coverage.
- `ext/HammerheadMakieExt.jl` — `plot_vector_field[!]` (weakdep Makie; grid
  methods take `stride`, auto `lengthscale = :auto`, and
  `show_replaced`/`replaced_color`; scale via the core `arrow_lengthscale`
  helper, testable without Makie; result methods route through `physical`
  and label axes from `plot_axis_labels`, so scaled results plot in
  physical units)
- `ext/HammerheadUnitfulExt.jl` — weakdep Unitful; its entire surface is the
  `PhysicalScale(pixel_size::Length, dt::Time)` constructor (ustrip + unit
  labels; no core stub needed — it's a constructor method, not a
  stub-shadowing function)
- `ext/HammerheadAMDGPUExt.jl` (`backend = :amdgpu`, trigger AMDGPU, adds
  rocFFT) + `ext/HammerheadCUDAExt.jl` (`backend = :cuda`, trigger CUDA, adds
  cuFFT; compat "5, 6" — the 6.0 subpackage split keeps the `using CUDA`
  surface) — batched device correlation engines sharing the portable
  kernels in `src/ka_backend.jl` (gather, cross-power,
  shift/gain, and the full `analyze_plane!` port: peak finding, gauss3/gauss9
  subpixel, ratio, moment, alt peaks — only packed per-window scalars return
  to the host). KernelAbstractions and AbstractFFTs are core hard deps
  (AbstractFFTs was already in the graph via FFTW), so a GPU backend needs
  only its device package loaded, and `backend = :ka` — those kernels on the
  KA CPU backend, the hardware-free proving tier — is built into the core
  (`_KABackend`/`_KACorrelationEngine` live in `src/ka_backend.jl` too; no
  ext needed). The exts import the kernels from Hammerhead and `using` the
  core's strong deps directly (works on 1.10 — verified). `test/test_ka.jl`
  guards the kernels; on this box `:ka` matches `:cpu` bitwise for
  non-deforming passes, and to ~1e-12 once a pass deforms — Phase 3 moved
  deformation into the portable `_ka_deform!` kernel (prefilter stays on the
  CPU; the kernel does bilinear predictor eval + cubic B-spline resampling
  from the padded coefficients, verified against Interpolations.jl to ~9e-16;
  seam: `apply_predictor(backend, …; ctx)` with a CPU-delegating default).
  Phase 3b made device deformation zero-copy per sweep: `_deform_context`
  stages the prefiltered padded coefficients once per `run_piv` call (per
  pair in ensemble) into a portable `_KADeformContext` — built with generic
  `KernelAbstractions.allocate` and pooled in the workspace's engine dict, so
  `:ka` proves the exact code path the device exts run — whose warp output
  buffers stay device-resident; per-sweep traffic is only the coarse
  predictor grid, and the engines' `_stage_pair!` consumes device-resident
  warped images in place (host images still upload for non-deforming passes).
  Scope: `:cross`/`:phase` + `:gauss3`/`:gauss9` only (phase uses the CPU
  correlator's Gaussian filter and epsilon-guarded normalized cross-power);
  gauss2d/keep_planes are rejected with a clear error
  (`_ka_scope_check`); `run_piv_stereo` forwards the backend to its
  per-camera `run_piv` calls (dewarp + 3C reconstruction stay CPU), and
  `run_piv_ensemble` runs on all KA-family backends: the summed planes live in
  a device-resident plane-major accumulator (`_KAPlaneAccumulator`, odd
  trailing dim against channel conflicts; `_ka_shiftgain_accum!` adds each
  pair's planes in place, `ensemble_analyze!` returns only packed per-window
  scalars), via the engine-dispatched hooks `_plane_accumulator` /
  `accumulate_planes!` / `ensemble_analyze!`. Phase 4b runs the Wieneke UQ
  statistics kernel in Float64 on all KA-family backends: fused single-pair
  and final post-iteration sweeps return packed scalars only, while ensemble
  statistics remain device-resident and additive across pairs. Single-pair,
  sequence, and stereo runs can instead set `uncertainty_backend = :cpu` for
  a vendor-neutral hybrid path: correlation/deformation stay on the selected
  backend, the final warped pair transfers to the host once, and threaded CPU
  UQ preserves the same Float64 estimator. `benchmark_piv_configurations`
  compares CPU/device/hybrid on a representative production pair; the
  convenience CLI is `bench/gpu_configurations.jl`. Phase 4c cut
  the UQ recompute: `_ka_uq_stats!` originally re-derived the smoothed ΔC
  stencil (12 array reads/pixel) scratch-free, twice per pixel for each of the
  40 covariance offsets — profiling (`bench/gpu_profile_uq.jl`, RTX 2000 Ada)
  put it at 44–62% of a UQ multipass run's device time (half the wall-clock),
  the top opportunity by far (correlation FFTs run every pass but are only
  ~4–8% each; UQ is final-pass-only). `_ka_uq_fill!` now materializes the
  smoothed field once per (component, window) into a batch-major device scratch
  buffer `uqdcs[k, comp, r, c]` (window index leading for coalesced wavefront
  reads, like `Rt`; +1 leading-dim pad) in plane precision T — so the Float64
  read-back is bitwise-identical to the recompute — and fuses in the window
  mean; the stats kernel reads the cache. Result: the stats kernel dropped
  ~5–8×, the whole UQ-multipass pipeline ~2× (e.g. Float64 2048² 1.50 s →
  0.74 s device time), `:ka`↔`:cpu` still ~3e-15 and ensemble bitwise, all on
  hardware. Phase 5 flipped the plane batch to *plane-major* `Rt[i, j, k]`
  (window index trailing, +1 trailing pad) and made `_ka_analyze!`
  *cooperative*: one workgroup of `_KA_TPW` (=128) threads per window, the
  threads splitting the K sequential peak-selection scans (the kernel's whole
  cost — subpixel/ratio/moment already read only a 3×3 via the cached peak).
  Both `:exclusion` and the production-default `:regionalmax` path are cooperative:
  each thread writes its (best, column-major-order) partial to `@localmem`,
  thread 1 reduces + applies the finder-specific rejection/positivity rule +
  records the peak, then does the serial subpixel/moment/alt-peaks. Regional-max
  re-tests the shared 8-neighbor predicate on each of K scans and excludes only
  previously selected exact locations; this is exactly equivalent to the CPU's
  one-scan insertion list, including equal-value ordering and plateau ties.
  (`bench/analyze_ka_coop.jl` was the proving prototype — ordinary locals don't
  survive a barrier on the CPU backend, so all cross-barrier state must be
  `@localmem`.) Plane-major (not
  batch-major) because a block's threads now stride *within* one plane, so
  contiguous plane pixels coalesce; batch-major regressed at high window counts.
  Launch is `ndrange = _KA_TPW * nreal`, groupsize `_KA_TPW`, kernel takes
  `Val{TPW}`. On the RTX 2000 Ada `_ka_analyze!` dropped from ~26% → ~8% of a
  Float32 2048² non-UQ multipass trace, and it's no longer the top kernel.
  Mirroring the same implementation to the RX 6800 XT moved `:regionalmax`
  from 0.8–1.3× CPU to 1.8–3.2× single-pass and 2.5–3.1× multipass; all paths
  still match `:cpu` to ~1e-15 / ensemble bitwise. GPU kernel
  conventions (violations cost 10-50x, found the hard way on the RX 6800 XT):
  no throwing ops in kernels — checked `Int32` conversions and `round(Int, x)`
  compile to malloc hostcalls (use `% Int32` wrapping stores and
  `unsafe_trunc` after guards); the cooperative analyze needs the plane-major
  `Rt[i, j, k]` layout so a block's within-plane strided reads coalesce; the
  `Rt` trailing (window) dimension is padded +1 because a power-of-two
  plane-to-plane byte stride funnels writes into one memory channel; GPU FFT
  in-place plans apply via `p * x` (neither rocFFT nor
  cuFFT implements FFTW's 3-arg `mul!`). `:amdgpu` is hardware-validated +
  benched (`bench/gpu_validate.jl` / `bench/gpu_benchmarks.jl`; ROCm 6.4
  required for RDNA2 on Windows — 7.1 dropped it). Its batch memory is managed
  as one byte-budgeted workspace cache: discovered window configurations get
  fair shares (bounded by their actual job counts), buffers coexist when they
  fit, and cold configurations are released by LRU when they do not; without
  a workspace, each pass releases its batch immediately. The AMDGPU
  `inv(rocFFTPlan)` path is deliberately avoided because it allocates a full
  batch-sized temporary just to compute normalization — Hammerhead builds a
  direct `plan_bfft!` and applies the known scale. On the RX 6800 XT, 4096²
  high-effort Float64 converges to 1024/4608/8192 batches, uses 7.99 GiB live,
  leaves 7.80 GiB free, and runs in 4.99 s steady-state; Float32 runs in 3.29 s
  with 6.62 GiB free. rocFFT work buffers themselves are zero bytes for the
  production power-of-two plans. `:cuda` is also
  hardware-validated (RTX 2000 Ada, CUDA.jl 6.2, driver CUDA 12.8: all paths
  match `:cpu` to ~1e-15 (Float64), 1.4–2.3× threaded-CPU speed)

## HammerheadGUI (HammerheadGUI/)

`experimental_qml_gui` launches the experimental Qt controls in a fresh process
using optional QML/QMLMakie dependencies from the caller's project. Its session
handle owns cooperative shutdown; launch requests capture absolute inputs and
package paths, verified before Qt initialization. Keep every generated artifact
under the writable session directory. The packaged runtime lives in
`HammerheadGUI/prototypes/qml/`, alongside its development harnesses. Its README and
`docs/src/explanation/gui_framework.md` distinguish controller/rendering
evidence from native teardown/input/platform gaps. A successful framebuffer
capture is not a clean application-lifecycle result. Keep generated manifests
and artifacts ignored; retain portable relative source paths in its Project.
The saved-experiment lane uses `ExperimentController` for recipe/history state
and inspection; `worker_client.jl` and `replay_worker.jl` execute captured planar
requests in a core-only subprocess. Only the shell owner updates Observables and
acknowledges written-pair progress. Poll process liveness even while an observer
defers acknowledgement; ordinary shutdown releases deferred acknowledgements
and waits for confirmed exit. An ownership failure is different: preserve the
captured job/request and busy guard, expose the original fault, and refuse
further progress delivery or acknowledgements while cleanup is unconfirmed.
Do not synthesize a terminal outcome from a diagnostic or clear the job to make
the UI appear idle. Startup cleanup retains ownership until reaping finishes.
Windows uses a kill-on-close Job Object. The Linux x86_64 backend uses a core-free
subreaper guardian and transferred self-opened pidfds; its process-group operation
requires kernel/libc capability checks before scientific enrollment. The Linux
client's Process is the guardian; worker PID is zero until enrollment. Retain
kernel references rather than reopening numeric PIDs after asynchronous exit.
Require root reaping, group emptiness, no adopted children, request-bound proof
and normal guardian exit before releasing ownership. Escaped children or guardian
loss keep cleanup unconfirmed. Unsupported hosts refuse explicitly; do not turn
that prerequisite into a successful fallback. See the prototype's
[Linux ownership guide](HammerheadGUI/prototypes/qml/linux_worker.md).
The lane never projects complete recipes into the synthetic demo form. Preserve
the displayed run's own identity after failed actions, verify completed output before lazy inspection,
and use the existing physical-display helpers. Shell ownership of replay survives
viewport close/reopen; shutdown waits for cancellation cleanup before disposing
subscriptions. Software-shell checks do not satisfy the native rendering gate.
Queue loading, replay inspection and viewport transitions outside Qt callbacks;
copy callback arguments to Julia values before enqueueing. Keep diagnostic
hashing and assertions in the owner loop too: Julia exceptions must not unwind
through a QML callback. Shutdown discards
pending actions and waits for active replay cleanup. Software previews render an
explicit hidden GLMakie screen without its background event loop; calling the
figure-level save path can restart that loop through cached-screen configuration.
File pickers stage draft paths only; existing Open and Replay actions retain
controller mutation and protected-output checks. Convert accepted QUrl values
with Qt's local-file APIs on the owner thread before retaining plain Julia
strings. Cancellation, stale dialog tokens and shutdown must not change the
current display or destinations. Each dialog instance owns its request token;
callbacks must not read a newer opening's mutable token. Dispose the instance on
acceptance, rejection or shutdown. Guard shell shortcuts while a picker is open.
Save-file pickers use "Use path" without an overwrite prompt: they do not write
the destination, and explicit replay retains its own output checks.
Hidden picker checks use Qt's non-native dialogs and do not establish native OS
dialog behavior, accessibility or network-share access.
The lifecycle runner owns hidden child processes and records their final exit,
logs, relevant Qt environment, stages and source identities. Set Qt platform/
backend selectors in the parent process environment before launching a child;
Julia `ENV` values alone do not prove Qt's effective C-runtime configuration on
Windows. Application observer release is distinct from native context cleanup.
The opt-in `--plot=glfw` mode gives Qt controls a separate, dedicated GLMakie
screen. Pump Qt and GLFW serially on the owning Julia thread, with no background
renderer; allocate a dedicated screen before attaching the scene, since the
scene constructor can reuse an unrelated singleton screen. Preserve ownership
when destruction fails and report cleanup failures without replacing the
original processing exception. Capture a screen directly; do not start a cached
figure-level renderer. Geometry limits include arrow tips, while ordinary frame
changes preserve manual pan/zoom. This mode still loads the QMLMakie plugin and
does not establish embedded Qt context cleanup or native input behavior.
The manual `.github/workflows/qml-prototype.yml` gathers isolated Julia 1.11
evidence on three operating systems. Failed native prerequisites stay failed;
any unsuccessful child command prevents later launches because descendant cleanup
is unverified, including failures returned by nested owners. Workflow presence is
not platform validation.

`ExperimentController` and `experiment_workflow[!]` provide a separate, read-only
complete-recipe workflow. `experiment_record` / `save_batch_experiment` export
file-based batch settings, exact effective preset schedules, and fingerprinted
built-in preprocessing snapshots. Arbitrary callbacks require `ScriptReference`.
Do not project imported recipes into the narrower batch/preprocessing widgets.
Replay captures state before notification, records completed/failed metadata,
and checks recorded output content before lazy exploration. Progress reports
completed pairs; cancellation is cooperative at pair boundaries and waits for
loader cleanup. Cancelled ordinary runs retain failed core metadata and a native
prefix, without checkpoint resume guarantees. GPU recipes and full recipe
editing remain open. The separate stereo workflow is described below. The batch
form links to this workflow; its API reference is split into `gui_experiments.md`.
Files/Replay/Reports control sections retain their widgets and settings. Hidden
Makie widget scenes still have active mouse regions, so inactive allocations
must also move outside the figure. Cancellation/progress/status stay visible;
full paths and status remain reachable through allocation-sized text pages.
The same workflow saves and displays core quality reports through
`experiment_quality_report` / `save_experiment_quality_report`. These synchronous
scans protect the selected result and run record, and refuse busy/changed runs.
Capture both history/execution options before dialogs or observer notifications;
retain the previous report's own identity after a failed save.

`CheckpointController` / `checkpoint_workflow[!]` provide a separate resumable
built-in recipe path, linked from the saved-experiment view. Capture checkpoint,
recipe, recovery assertion and an independent cancellation token before notifying
Observables or yielding. Progress uses committed counts, not repeated store
scans; cache recipe text separately from progress/status rendering. Opening and
refreshing validate fixed prefixes; explorers retain one displayed result and
never follow live appends. Recovery asserts the former writer has stopped and
resets after use. Preflight and work within a pair can delay UI interaction.

Lazy native explorers can opt into recorded processing details. Verify history
against the raw result before physical conversion and commit navigation state
only after preflight succeeds. Retain one display payload/current packet; a
separate display digest detects later array edits, including retained stereo
camera fields. Planar execution companions v1 bind entry keys, not numerical
result content; stereo companions bind raw reconstructed and camera fields.
Stereo residuals remain dewarped pixels and per-node history stays unavailable.
Missing history is never inferred. Scaled magnitude fields are labelled speed.
Ensemble companions bind raw fields/geometry before physical conversion and
retain scalar pooled observations only; node selection cannot recover unrecorded
contributor histories. Reject incompatible packet families on the same entry.
`explorer_quality_report` and `save_explorer_quality_report` consume a lazy native
explorer's whole raw index, independently of displayed frame and inspection mode.
The separate `result_quality_report` view captures that index on opening and
options before notifications or a picker. Failed/cancelled requests retain the
previous report's own file/SHA identity. Scans are synchronous; eager/bare,
timed and checkpoint explorers have no supported whole-native-file association.

The separate recipe-comparison controller captures complete before/after records,
selected pairs and value basis before notifications or asynchronous scheduling.
It uses the core selected-pair comparison and report contracts. Current choices
and the last report's identities remain distinct after failure. No cancellation
or uninterrupted CPU responsiveness is promised for a single-pair comparison.

`RecipeRevisionController` / `recipe_revision[!]` edit ordered planar passes in
a separate draft. `preprocessing_revision[!]` uses the same controller for ordered
built-in steps, duplicates and exact embedded backgrounds. `RecipeImagePreviewController`
captures an explicit original pair, verifies bytes around decoding/conditioning,
and uses exact replay preprocessing in recipe precision on full images before ROI.
Its detached bundle keeps its own recipe/pair identity; mask/ROI are overlays and
raw/processed views share an explicit intensity range. Scripts remain references.
`recipe_geometry_revision[!]` edits raw ROI/scale drafts on the same controller.
Preserve exact imported bounds, Float64 factors and unit labels. Disabled drafts
retain their raw text but compose `nothing`; enabling never invents a calibration.
Metadata preflight checks each recorded pass and frame. Full-image masks and
backgrounds stay in original coordinates, while stored result pixels/displacements
remain unscaled until physical conversion. Capture all drafts before notifications.
Saved-mask revision retains a detached enabled/full-image raster draft. Disabling
retains bits but composes `nothing`; an enabled all-false mask stays distinct.
Seed `MaskEditor` with copied raster bits, then apply ordered exclusions and holes;
clear-all removes both raster and polygons. A verified raw reference has its own
captured identity and never executes preprocessing scripts. Applying editor pixels
refuses unfinished drawing and changed mask content. Mask imports capture options
before pickers and protect consumed source paths for the controller lifetime.
Preserve all unedited pass fields, ordered validator tuples
and the complete imported recipe options. Raw text remains separate from the
last validated candidate; invalid visible edits must never fall back to stale
parsed values. Metadata previews can work without source files. Creating or
saving a new record verifies available unchanged inputs, retains the exact
ordered input identity, captures the current creation environment and starts
with no runs. Protect source record aliases, inputs, scripts, recorded results
and caller-supplied history/output paths. Capture requests before notifications
or pickers, and queue verification/save work outside native input callbacks.
Cooperative scheduling does not guarantee responsive file I/O or cancellation.
Opening a saved revision uses a separate experiment workflow; do not replace
the original record, history or display. Revision lineage is session metadata,
not an extension to the core version-1 record schema.

Monorepo subdirectory package, Makie-style: own Project.toml (this is where
the GLMakie/NativeFileDialog hard deps live — the core never gains GUI deps),
`[sources]` path coupling to the core for dev (Julia ≥ 1.11; the CI `gui` job
`Pkg.develop`s the core instead so lts/1.10 works, and wraps tests in
`xvfb-run`). The develop step sets `JULIA_PKG_PRECOMPILE_AUTO=0`: GLMakie
precompilation needs a DISPLAY (only the test step runs under xvfb), and
`Pkg.test` recompiles with its own flags (`--check-bounds=yes`) anyway, so
precompiling in the develop step both fails and wastes ~8 CI minutes. Releases go core-first, then GUI compat bump; registration is
`@JuliaRegistrator register subdir=HammerheadGUI`, TagBot tags
`HammerheadGUI-v*` (second TagBot job), CompatHelper covers both packages via
`subdirs`. Architecture rule: all application state/logic lives in a
framework-free controller layer (plain Julia + Observables, testable without
a GL context); Makie code renders controllers and pushes input into them but
controllers never import Makie. The mask editor is the framework proving
ground for pure-GLMakie widget chrome.

Layout: `src/controllers/*.jl` are included into the `Controllers` submodule
(Hammerhead and nonvisual dependencies only — the module boundary enforces
the no-Makie rule, and a test asserts it); `src/views/*.jl` are the
GLMakie shells. Components so far (each = controller + view pair, same
naming): `ResultExplorer`/`result_explorer` (browses all four persisted
result types — `PIVResult`/`StereoPIVResult` grids, `PTVResult` particle
scatter, `TrackingResult` gap-aware polylines colored by mean speed — mixed
sequences included; routes each displayed entry through `physical` so a
`PhysicalScale` gives physical-unit axis/colorbar/inspection labels;
path constructors accept `lazy = true` and `ResultFile` inputs retain only
one converted display payload; all explorers evict derived fields on frame
changes. Lazy navigation failures preserve the prior display and report
status; file snapshots reject live appends. The default remains eager;
selection is a `CartesianIndex` for grids, a linear `Int` for scattered
types; the vector overlay is quiver-style linesegments + rotated-triangle
scatter heads, NOT arrows2d — arrows2d's per-frame pixel-space tip sizing
made pan/zoom crawl at thousands of arrows; colorbar limits default to a
robust 2–98% percentile band over valid vectors (`color_limits`) with
bound-wise manual overrides persisting across frames; `push_result!`
appends live and grows the view's slider via the `count` observable;
planar results add derived fields (:vorticity/:divergence/:strain_rate/
:swirling_strength/:q_criterion via flow_derivatives, cached for the current frame,
unit-labelled 1/time — physical-at-construction keeps the gradients
exactly 1/dt) and a tool mode (:inspect/:profile/:circulation with
`click!`/`alt_click!` gestures, planar-only, state clears on frame
switches; circulation reports both line-integral and vorticity-area
estimators; the profile panel appears as a third layout row);
`set_derivative_stencil!` selects an explorer-wide policy that persists across
tools and frames, including area circulation; component profiles and line
circulation remain independent. The `:derivative_support` tool adds discrete
eligibility, x/y stencil and finite-component-count maps through
`available_fields(ex)`, plus paged contributor details. Rich support metadata
is retained only for the current frame while the tool is active. Inspection
checks displayed-input integrity and requires explicit reselection to refresh
after mutation. Current flags alone never establish measurement replacement.
`MaskEditor`/`mask_editor`
(gesture API `click!`/`alt_click!` holds the editing model; the view only
forwards mouse/key events; `Hammerhead.polygon_mask(::MaskEditor)` exports
the mask, `save_mask` writes the white-=-excluded image `load_mask` reads);
`ROIEditor`/`roi_editor` (two-corner and numeric inclusive pixel bounds,
clear/reset, core `ROI` validation, and `apply_roi!` into `BatchRunner`;
the batch snapshots its ROI, preprocesses full frames, and lets the core crop
images/masks and retain original image coordinates);
`BatchRunner`/`batch_runner` (runs `run_piv_sequence` with its progress
callback inside `@async` — cooperative, so GL renders keep happening off
`run_piv`'s internal thread-spawn yields while observables stay on the
primary thread; cancel = throw `BatchCancelled` from the callback, which
keeps finished pairs in the incremental output; an `effort` menu
(`:custom` manual schedule vs `:low`/`:medium`/`:high` presets) and a
physical-scale form group attach a `PhysicalScale` to the outputs; the
core drivers' `on_result` hook (all sequence drivers incl. stereo: called
`(i, result)` on the caller's task right after storage, before persist and
progress; throwing aborts like progress) feeds the live `completed`
observable, and "view results" opens an explorer mid-run that follows the
batch; `set_preprocess!` attaches a per-frame pipeline);
`PreprocessPreview`/`preprocess_preview` (ordered toggleable pipeline over
the core preprocessing set with live raw/processed preview and a
single-window correlation probe — `set_pair!` + `click!` place it, du/dv/
peak-ratio recompute on every pipeline change via a border-clamped
single-window `run_piv` at the accuracy defaults; `build_preprocess`
exports a frame-copying, snapshot-semantics closure with a copied background for
the batch drivers); `ScaleTool`/`scale_tool` (two clicked points + known
separation → `PhysicalScale`; `apply_scale!` into a batch form);
`StereoBatchRunner`/`stereo_batch_runner` + `stereo_calibration` (two
synchronized frame lists + an `ImageDewarper` pair —
`build_dewarpers(cr1, cr2)` composes `common_dewarp_grid` from two fitted
`CalibrationReview`s, and the workflow view embeds both reviews via the
embeddable `calibration_review!`; runs `run_piv_stereo_sequence` with its
NATIVE zero-arg `cancel` predicate — no exception, completed prefix
returned — and a dt-only stereo scale);
`StereoExperimentController`/`stereo_experiment_workflow` snapshots idle,
file-based stereo batches and preserves rich imported recipes without exposing
unsupported edits. Capture replay settings before observable notifications;
`active_request` carries detached scalar identity/settings, separate from next
choices. Historical run selection, latest attempt, displayed result and retained
report keep their own identities. Verify completed outputs before lazy physical
display; reports use the dedicated stereo core overload. Cancellation waits for
an acquisition boundary and records failure history without implying resumability.
Files/Replay/Reports pages retain reachable cancellation/progress controls;
`EnsembleExperimentController`/`ensemble_experiment_workflow` snapshots supported
file-based batches using ensemble effort presets and preserves full imported
recipes. Backend/precision controls affect the next snapshot only. Keep selected
historical runs, active requests, displayed results and reports distinct; joined
contribution progress is not result publication. Retain a known terminal run if
history save/reopen fails, and show that persistence error separately.
`CalibrationReview`/
`calibration_review` + `selfcal_review` (grid-detection/reprojection review
and the `SelfCalibrationReport` browser — its disparity maps open in an
embedded explorer via `result_explorer!(gridposition, ex)`, the embeddable
form all composite views should use). Shared widget↔controller sync helpers
live in `views/widgets.jl`. View gotchas learned:
recreate heatmap/arrows per refresh instead of updating per-argument
observables (sequential x/y/data updates render transiently mismatched
grids); preserve zoom by capturing/restoring `ax.targetlimits[]` — `limits!`
normalizes the rect and silently undoes `yreversed`; guard every
widget↔controller observable pair with equality checks (Observables notify
on same-value writes, so unguarded two-way wiring loops forever);
`colorbuffer(fig)` may return the screen's reused framebuffer — `copy` it
before comparing renders in tests; `word_wrap` labels need an explicit
`width` (with `tellwidth = false` they wrap at a bogus narrow width).

## Load-bearing conventions

Non-informative correlation planes are unavailable measurements: nonfinite,
nonpositive, or completely flat planes yield NaN diagnostics/displacement and
unmasked outlier flags regardless of optional validators. Exact constant valid
pixels are centered before apodization without averaging roundoff. Predictors
exclude nonfinite donors and use neutral zero only where no finite fill exists;
this does not convert missing results into measurements. Deformed windows also
require exact contrast in both original raw stencil unions sampled by the
predictor. This is an explicit scientific convention: B-spline prefiltering has
nonlocal influence, and remote coefficient leakage alone does not establish
source contrast. Ignore virtual extrapolated zeros and original masked pixels
as contrast evidence; preserve any genuine processing-precision difference.

- **Sign convention (package-wide):** a particle at `(row, col)` in image A
  found at `(row + dv, col + du)` in B yields positive `(du, dv)`; `u` is
  along columns (x), `v` along rows (y). In correlators use `mul!` with the
  cached inverse plan — `ldiv!` with the forward plan silently flips signs.
- **Precision follows the images:** `T = float(promote_type(eltype(imgA),
  eltype(imgB)))` flows through correlators, deformation, and `PIVResult{T}`.
  Deliberate Float64 islands (CPU-side, converted on store): the LsqFit
  `gauss2d` fit, correlation-moment accumulation, replacement medians. Don't
  introduce new Float64 literals/arrays into the hot path.
- **In-place-first preprocessing:** the mutating forms (`highpass_filter!`,
  `clahe!`, …) on `AbstractMatrix{<:AbstractFloat}` are the implementations;
  allocating names are `f(img) = f!(float_copy(img))` wrappers.
- **Masking:** `mask` is image-sized Bool, `true` = excluded, static
  lab-frame geometry — never warped between passes. `result.mask` (dropped
  windows, NaN fields) is distinct from `result.outliers` and masked cells
  are never outliers. Masked pixels load at the valid-pixel mean (zero after
  mean subtraction — no step at the mask edge). UOD takes `exclude`;
  replacement neither draws from nor fills masked cells; the multi-pass
  predictor fills masked cells from valid neighbors before smoothing.
- **Correlation accuracy:** plain circular correlation biases ~0.15 px toward
  zero; `padding = true` requires the overlap-gain normalization (already in
  the correlators) and `padding + apodization = :gauss` is the accuracy
  configuration (~0.03 px RMS). Test tolerances encode this (0.25 px plain,
  0.05–0.1 px deformed/padded) — don't loosen them to make a change pass.
- **Extended search areas:** `search_area_size >= window_size` with an even
  per-axis difference keeps the frame-A interrogation footprint concentric
  with the larger frame-B search footprint. Grid stride remains
  `window_size - overlap`; only the outer centers move inward. CPU single-pair
  and ensemble paths support it (including masks, retained planes, phase,
  padding, and apodization); KA-family backends reject it explicitly until
  their gather kernels carry independent footprints. UQ always uses the
  centered equal-size interrogation pair, not the larger search footprint.
- **UOD defaults matter:** `epsilon = 0.1` px is the physical noise floor
  (near-zero flags everything on uniform fields); `uod_neighborhood = 2`
  (5×5) because 3×3 falsely flags smooth gradients at field edges.
- **`correlate` returns an aliased plane:** `res.correlation` is an internal
  buffer overwritten by the next call — copy before storing.
- **Uncertainty (Wieneke 2015):** computed in `process_windows!` /
  `accumulate_planes!` from the deformed windows, final pass only — the
  method assumes the peak sits at ~zero residual, so it needs a converged
  multipass schedule (`max_iterations` on the final pass, or the equivalent
  explicit repeated final window size). Statistics accumulate
  in Float64 (`2 × UQ_NSTATS` per window) and are additive across pairs
  (that's how the ensemble path pools them). The covariance sums S_δ are
  summed ring by ring until a ring's max drops below `0.05·S00`; inner rings
  are taken whole because their negative members are real signal×noise
  anticorrelation — a per-term positive threshold inflates σ 2–5× at high
  noise. Estimates describe the random error only; near-outlier windows
  legitimately report huge σ, so validation comparisons use medians over
  non-outlier vectors.
- **Physical units:** result arrays always stay in measured units (px/frame
  for planar and PTV, world-per-frame for stereo); a `PhysicalScale`
  (Float64 pixel_size + dt, display-only unit label strings) attached via
  the drivers' `scale` kwarg or `with_scale` is pure metadata. `physical(r)`
  is the single conversion point: positions × pixel_size, displacements + σ
  (and PTV `match_residual` × pixel_size) × pixel_size/dt, factors converted
  to `T` once (Float32 stays Float32); px-native diagnostics (peak_ratio,
  correlation_moment, correlation_planes, particles, stereo cam1/cam2) are
  shared untouched — convert *last*, after validation/peak-locking. The
  converted result carries an identity scale with the labels kept, so
  `physical` is idempotent and the Makie ext (which routes through it) always
  labels what it plots. Two deliberate wrinkles: `physical(::TrackingResult)`
  keeps `dt` in the returned scale (velocities are derived by differencing —
  `trajectory_velocities(t, scale)` applies it), and stereo scales use
  `pixel_size = 1` + dt (a non-1 value is a world-length unit conversion).
  Plumbing gotchas: `run_piv_ensemble` needs the explicit `scale` kwarg in
  BOTH methods (they hard-reject unknown kwargs) and `run_piv_stereo` must
  capture `scale` so it is NOT forwarded to the per-camera `run_piv` calls.
  Unitful is a weakdep (`PhysicalScale(20.0u"µm", 0.5u"ms")` — values
  stripped in their own units, unit names become the labels).
- **Temporal sampling:** `result_spectrum` requires explicit `dt` or
  `sample_times` for successive velocity samples. Exact timing validation uses
  both interval and accumulated grid residuals, with zero tolerance by default;
  positive tolerances explicitly permit approximately uniform sampling. There
  is no resampling. Never infer the sampling interval
  from `PhysicalScale.dt`: that is the image-pair displacement delay, which
  differs for paired/strided recordings and becomes 1 after `physical`.
- **Calibration (Phase 5):** a deliberate Float64 island — offline
  once-per-experiment fits, not the image hot path. World axes are anchored
  to image orientation (+X = lattice direction nearest image-right, +Y
  nearest image-up; assumes roughly upright cameras) with the origin dot
  fixed by the square fiducial + `origin_offset` (4E: `(30.0, 7.5)` mm);
  multi-camera frame consistency depends on the marker. Two-level plates are
  indexed as one 45°-rotated combined lattice; the level is the index parity
  (`indices` are in half-spacing units). Pinhole DLT needs ≥ 2 Z planes,
  Soloff ≥ 3. On the real 4E plates, ~0.5 px per-dot residuals are
  *repeatable* across planes (plate dot-position tolerance, not detection
  error — don't chase them); plane-to-plane detection repeatability is
  0.15–0.3 px.
- **Dewarping (Phase 5, slice 2):** the dewarped image is indexed
  `out[r, c] = world (grid.x[c], grid.y[r], grid.z)` in the order the ranges
  are given (ascending `y` puts +Y *down* the image; pass descending `y` for
  display orientation); displacements convert to world units as
  `du·step(x)`, `dv·step(y)`, signs included. The `ImageDewarper` coordinate
  map is Float64 (built once per camera from `world_to_pixel` — no Newton
  needed, it's the forward map) but `dewarp!` output precision follows the
  image. Out-of-view nodes: zero-filled, flagged in `dw.mask` (`true` =
  excluded, static lab-frame) — feed it to `run_piv(...; mask)`; stereo
  overlap is `dw1.mask .| dw2.mask`. Choose the grid finer than the target
  vector spacing (correlation windows live on dewarped images).
- **Stereo 3C (Phase 5, slice 3):** `run_piv_stereo` takes raw frames + two
  `ImageDewarper`s sharing one `DewarpGrid`; both cameras run with identical
  parameters and the union node mask (`dw1.mask .| dw2.mask .| user`), so
  the per-camera vector grids and masks match exactly. Per point, camera *i*
  measures `uᵢ = dx − dz·tXᵢ`, `vᵢ = dy − dz·tYᵢ` in world units (via
  `u·step(x)`, `v·step(y)`, signs included), where `(tXᵢ, tYᵢ)` =
  `ray_slopes` (in-plane drift per unit Z of the viewing ray, central
  differences of `world_to_pixel`); an unweighted 4×3 LSQ solves
  `(u, v, w)`, and per-camera Wieneke σ propagate through the same
  pseudoinverse (independent-error assumption). Degenerate (parallel-ray)
  points come out NaN. `StereoPIVResult` keeps mask/outliers as unions
  (masked ≠ outlier preserved; flagged vectors were reconstructed from
  replaced 2C data) plus both per-camera `PIVResult`s. Reconstruction is a
  Float64 island (O(vector grid), converted on store); velocities come from
  attaching a dt-only `PhysicalScale` (see the physical-units bullet).
- **Self-calibration (Phase 5, slice 4):** `self_calibrate(frames1, frames2,
  dw1, dw2)` fixes sheet↔plate misregistration. Disparity = ensemble
  cam1-vs-cam2 correlation of same-instant dewarped images (single pass,
  large windows — the outer correct-and-redewarp loop iterates, default ≤ 3
  corrections plus a trailing verification measurement, so `report.passes`
  has one entry per *measurement* with `plane === nothing` on non-correcting
  passes). Triangulation reuses the `ray_slopes` 4×3 system (unknown
  `(X, Y, ζ)`, symmetric ∓d/2 attribution — second-order error the iteration
  absorbs); vectors with > `max_triangulation_error` (0.5 px) residual are
  rejected. The rigid transform maps the fitted plane to `z = grid.z` with
  the paper's anchoring (+Z sheet normal, +X old X projected onto the sheet,
  cam1's view of the old origin fixed) and is applied to BOTH cameras:
  `PinholeCamera` bakes it into `P` exactly, everything else gets the exact
  `TransformedCamera` wrapper (re-wrapping collapses, so iteration doesn't
  nest). Scalar diagnostics are always recorded; per-pass disparity
  `PIVResult`s only with `keep_disparity_maps = true`. If the first
  measurement is already below `tol` the input dewarpers are returned
  `===`-identical.
  task (correlators are mutable state); results must stay bitwise identical
  to serial (tested).
- **PTV (Phase 8):** `PTVResult.x/y` are the **frame-A** particle positions,
  `u/v` the displacement to frame B — this matches the `SyntheticData`
  forward-Euler contract exactly, so ground-truth tests compare directly with
  **no** midpoint correction (unlike PIV's symmetric deformation). Detection
  is in-house (local maxima + 3-point log-Gaussian fits) because PIV-density
  particles overlap and connected-component blobs merge — the sanctioned
  ecosystem-policy exception. Matching is predictor-guided greedy one-to-one
  (`run_ptv` runs a coarse `run_piv` internally by default); scattered UOD
  (Duncan 2010) **flags but never replaces** — a track is a measurement of one
  particle. All neighbor searches (dedupe, match candidates, kNN for UOD) use
  the in-house uniform cell list in `particles.jl`, not NearestNeighbors.jl
  (the no-new-deps rule). `run_ptv` short-circuits to an empty result when
  either frame has no detections (skips the image-size-sensitive predictor).
  Measured in pixels; physical units via the `scale` metadata (see the
  physical-units bullet). `particles.jl` is included before
  `ptv.jl`, so `detect_particles`'s `params` argument is unannotated
  (`PTVParameters` is defined later — a default-value expression is evaluated
  only at call time, unlike a type annotation).
- **Ecosystem policy:** use JuliaImages packages (FileIO/ImageIO,
  ImageFiltering, JLD2) unless they compromise subpixel fidelity — CLAHE
  deliberately stays in-house because ImageContrastAdjustment silently
  `imresize`s images whose dims don't divide into blocks.
- **Makie extension methods** must stay more specifically typed than the
  Vararg stubs in `Hammerhead.jl`, or precompilation hits method overwrites.

## Testing notes

- Windows CI can put the checkout and system temporary directory on different
  volumes and expose temporary paths through an 8.3 short spelling. Compare
  canonical stored file locators against `realpath` expectations; retain plain
  absolute-path expectations where that is the API contract. Exercise relative
  aliases from a directory on the source volume. Hardlink fixtures must live on
  the source volume; guarded benchmark outputs should use temporary directories
  under `bench/profile-output`. Assert the empty output is admitted before adding
  an alias, then verify same-file identity, alias rejection and unchanged source
  bytes. Do not skip protection coverage on Windows or accept an unrelated
  directory-policy error as evidence that alias detection works.
- `test/runtests.jl` defines the `particle_pair`/`add_particle!` helpers used
  by all included test files; new test files can rely on them.
- `SyntheticData` ground truth is a forward-Euler step: each particle's true
  displacement is its launch-point velocity × dt. Symmetric-deformation
  measurements attribute vectors to trajectory *midpoints*, so sub-0.1 px
  accuracy checks against curved flows must evaluate the reference at
  `x − d/2` (see the first tutorial) — comparing against the grid-point
  velocity leaves an O(|d|²·∇V/2) floor that looks like measurement error.
- Adding a result-type field: prefer a trailing field plus back-compat
  positional constructors (the `correlation_planes`/`scale` pattern — every
  result type now has them in both inferred and `{T}` forms), which keeps
  the direct constructor calls in `test_validation.jl`, `test_ensemble.jl`,
  `test_accuracy.jl`, `test_stereo.jl`, `bench/run_benchmarks.jl`, and the
  `StereoPIVResult` fixture in `HammerheadGUI/test/runtests.jl` valid
  without edits. A non-trailing change breaks them all.
- `test_scaling.jl` covers the physical-units feature (PhysicalScale
  validation, with_scale/physical semantics per result type, driver
  plumbing incl. the effort kwarg-split path, JLD2 round-trip, the Unitful
  ext — Unitful is a test-target dep, which is what activates the ext under
  `Pkg.test`).
- `test_stereo_timing.jl` checks exposure synchronization, missing metadata,
  clock-epoch-independent tolerances, and rejection before image loading/output.
  `test_tracking_export.jl` checks CSV compatibility, gaps, numerical validity,
  and physical conversion. `test_sequence_sink.jl` checks non-collecting sequence
  delivery/persistence/cancellation and weak-reference release of old results.
- `test_transformed_export.jl` checks calibrated planar CSV/VTK geometry,
  vector bases, uncertainty assumptions, and refusal before output overwrite.
  `test_incremental_statistics.jl` checks population moments, incompatible
  update rejection, independent snapshots, and fixed retained memory.
  `test_lazy_results.jl` checks indexing, file changes, and lazy payload access;
  `test_streaming_workflow.jl` combines non-collecting output, live statistics,
  variable pair delay, physical conversion, and lazy replay.
- `test_experiments.jl` checks explicit recipe/record round trips, content
  identities, replay and environment guards, alias rejection, and failure records.
  `test_ensemble_experiments.jl` checks separate pooled-run counts, CPU/KA
  Float32/64 direct parity, cancellation before publication, captured settings,
  input/staging mutation guards and accurate outcomes after history-save errors.
  `test_ensemble_experiment_quality.jl` checks associated format-5 reports,
  raw/recipe/companion verification, relocation and protected persistence.
  GUI `test_ensemble_experiments.jl` checks exact snapshots/imports, contribution
  progress, retained terminal/report identities and hidden-window mouse routing.
  `test_validation_scorecard.jl` checks deterministic rendering/hashes, selection
  rules, population error RMS, analytic midpoint shear truth, complete recipes,
  report round trips, and output protection. Real A/4E rows have no displacement
  truth; cumulative Julia allocations are never labeled peak memory.
- `test_experiment_comparison.jl` checks full settings, ordered changes,
  content summaries, scientific versus locator identity, and no retained arrays.
  `test_noninformative_windows.jl` checks degenerate planes, exact constant
  inputs, tiny contrast, masks, nonfinite donors, predictors, and UQ on CPU/KA;
  `test_original_source_support.jl` covers deformed constant patches and the
  original-stencil convention. `test_run_quality.jl` checks stored-field metrics,
  detached bounded-memory summaries, verified run association, primitive schema
  validation, and alias-protected TOML persistence.
  `test_experiment_checkpoint.jl` covers identity/alias guards, exact resumed
  prefixes, handled failure/cancellation, and hidden child-process termination
  at lock/data/commit/terminal boundaries. Source must remain unchanged during
  strict environment-identity tests.
- `test_execution_diagnostics.jl` checks opt-in numerical parity, actual sweep
  and tolerance semantics, primary residuals, native companions, callback
  failures, unsupported-driver refusal before output, and replay association.
  `test_stereo_execution_diagnostics.jl` checks two-camera parity and association,
  signed common-grid geometry, separate native persistence, measurement-field
  verification, mutation guards and non-collecting lifetime. These companions
  do not verify calibration or certify reconstructed uncertainty.
  `test_pair_comparison.jl` checks exact grid/population moments, arithmetic
  overflow, selected-input identities, units and detached TOML snapshots.
  GUI `test_checkpoints.jl` checks captured execution state, cancellation,
  recovery, fixed lazy browsing, protected export and offscreen layouts.
- `test_measurement_history.jl` checks observed final-sweep origin/events,
  first-rejection semantics, numerical parity, mutable-callback guards and
  optional companion/result verification. Keep final-history storage bounded
  across sweeps and non-collecting sequences.
- `test_pair_timing.jl` checks exact timestamp arithmetic, complete preflight,
  frozen source selection/metadata, native companion validation, callback
  integrity and non-collecting lifetime. Timing does not change legacy tracking.
  `test_quality_history.jl` checks report-v2 coverage, event/origin populations,
  packet/run association and v1 compatibility. GUI `test_companions.jl` checks
  transactional loading, physical-display binding, release and panel layouts.
- `test_tracking_timing.jl` checks elapsed-time linking, exact metadata, secant
  time support, scale/delay separation and dedicated artifact/table semantics.
  GUI `test_recipe_comparison.jl` checks captured requests, verified comparisons,
  failed-request report preservation, protected saves and paged inspection.
- GUI `test_recipe_revision.jl` checks complete pass/recipe preservation, ordered
  input identity, fresh creation provenance, empty history, offline metadata
  inspection, invalid drafts and captured alias-protected saves. Revision view
  checks exercise compact layouts, live raw text, rejected busy selections and
  separate workflow launch; hidden controls do not establish desktop acceptance.
- GUI preprocessing revision tests cover exact step order/options/background
  precision, capture and protected saves. Independent parity tests compose public
  core operations in both image precisions, including full-frame-before-ROI and
  changed-input checks. View tests exercise actual controls, shared image ranges,
  original-coordinate overlays and retained preview identities at compact sizes.
- GUI geometry revision tests preserve enabled/disabled settings, invalid text,
  complete-recipe composition and original-coordinate ROI behavior. Independent
  replay checks distinguish attached scale metadata from physical conversion;
  form tests cover compact numeric editing and protected workflow launch.
- `test_tracking_speed_summary.jl` checks bulk actual-time secants, mean semantics,
  unavailable populations and metadata detachment. Calibrated scattered export
  tests check affine bases, units, exact intervals and paired-artifact verification.
- `test_calibrated_resampling.jl` uses independent scalar affine algebra and
  analytic fields to check shared-grid image/vector sampling, contributor
  validity, coordinate order, precision and input preservation. Registration
  validation has separate malformed/rank/scale regressions. Sampled arrays
  remain separate from measured PIV diagnostics and uncertainty.
- `test_artifact_paths.jl` checks foreign source-locator preservation, explicit
  local relocation and protection of consumed artifacts, including prospective
  Windows aliases. Foreign fixtures on Windows do not establish other-OS runtime
  evidence. Ordinary native result and pair-timing contracts remain separate.
- `test_spectrum_timing.jl` checks exact sample-time regularity, accumulated drift,
  large epochs, tolerance/range failures, legacy FFT parity and result value bases.
  Core/GUI `test_experiment_replay_progress.jl` checks completed-pair callback
  ordering, request capture, cancellation and original-error/cleanup semantics.
- `bench/validation_uncertainty.jl` evaluates controlled primary-only synthetic
  outputs across fixed seeds. Component UQ populations include zero sigma;
  normalized errors require positive sigma. Counts retain arithmetic failures,
  pooled moments stream, and quantiles remain per seed. Its regression file
  checks population arithmetic, origin guards, reproducibility and output paths.
- `bench/diagnostic_uncertainty.jl` audits retained final-sweep CPU windows with
  independent covariance tests and explicit numerical zero classifications.
  Keep full truth-error coverage, in-sample centering and residual-inclusive
  sensitivity results distinct; these are diagnostics, not estimator calibration.
- `bench/conditional_uncertainty.jl` holds clean scenes fixed while independently
  perturbing both images. Retain full truth-error metrics, complete-case losses
  and disjoint-realization difference populations. The pre-specified Bartlett
  covariance comparator is bench-only; its nonnegative block identity does not
  establish calibrated coverage or justify a production estimator change.
- `bench/rendering_uncertainty.jl` compares fixed particle placements under
  production point sampling, wider point support and pixel-area integration.
  Preserve the original control exactly and compare common primary populations.
  Its separate known-translation deformation lane never supplies predictors or
  images to the PIV accuracy runs. Quadrature agreement and interior crops are
  numerical checks, not total-error bounds or uncertainty calibration.
- `bench/validation_ptv_tracking.jl` scores complete ID/frame visibility
  annotations using independent maximum-cardinality/minimum-distance localization.
  Operational detection counts do not resolve identity: competing detections,
  nearby targets and target/nuisance overlap retain conservative ambiguity.
  Keep full and both-localized recall denominators, raw/accepted correspondence,
  returned-track identity changes/fragmentation and scheduled-absence recovery
  separate. Preserve all predicted edges, including wrong/unmapped/ambiguous
  cases. Optional unknown intensity must not invent an annotated amplitude.
  The regression suite enumerates tiny assignment oracles and checks explicit
  identity/gap fixtures, manifest guards and cheap production clips. Full study
  evidence requires a fresh process and frozen source; it does not close
  independent or real-recording validation.
- `bench/validation_vsj301.jl` evaluates the fixed independent synthetic VSJ301
  prefix using sparse listed IDs/positions. Unknown rows and visibility remain
  unknown; associated IDs do not certify physical contributors. Report all
  coordinate-origin hypotheses on the same unchanged production objects.
  Independent bounded component assignment must refuse oversized components,
  never drop them. Ambiguous/unknown observations interrupt identity continuity;
  exact adjacent recall and annotated endpoint relinking have separate counts.
  `prepare_vsj301.py` guards acquisition and re-audits selected archive members
  offline on every cache load. Keep source data and annotation-derived ledgers
  private; no redistribution license is asserted. Record actual thread defaults
  and run full studies only with frozen source and a fresh process.
- `test_ptv.jl` ground-truths against `SyntheticData`: knife-edge scenes
  (detection accuracy/dedupe, scattered UOD flagging) use `StableRNGs` and
  fixed geometry; statistical scenes (hybrid-match fraction, tracking recall)
  use `MersenneTwister`. Tracking is verified on an *off-frame*-centered
  vortex (smooth rotation, no near-singular core) with a thick sheet (no
  dropout): a few percent of full tracks legitimately deviate >0.5 px from
  the nearest truth path near the frame edges (detection error, not identity
  switches — asserted via a small per-step displacement bound), so the test
  bounds recall ≥85% and within-0.5 px ≥95% rather than demanding every track.
- `test/reference_images/A/` holds PIV Challenge case A TIFFs for the
  end-to-end reference test; `test/reference_images/E/` holds the case-4E
  stereo subset (16-bit PNGs + readme) shared by the stereo reference
  testset and the real-data stereo tutorial. The 4E bounds are smoke-level
  around the measured numbers (first-pass disparity RMS ≈ 2.8 px, fitted
  plane offset ≈ −0.67 mm, residual RMS ≈ 0.46 px, median σu ≈ 3 µm /
  σw ≈ 13 µm) — real data with no ground truth, so don't tighten them
  into accuracy claims.
- `MersenneTwister` streams changed between Julia 1.10 and 1.11 (seed
  hashing), so seeded-MT test scenarios are not reproducible across the CI
  matrix. Knife-edge scenarios (constructed peak orderings, tight
  acceptance bands) must use `StableRNGs` and keep their critical geometry
  deterministic — see the peak-substitution testset in `test_peaks.jl`
  (fixed dense-particle positions; narrow tripled-amplitude reflection dots
  so the reflection's autocorrelation tail stays inside the `find_peaks`
  exclusion radius). Tolerance-based statistical testsets are fine with MT.
- `test_calibration.jl` builds its own stereo fixture (`make_test_camera` +
  `render_calibration_target`); the real 4E images in `cases/` are *not*
  used by tests (not committed). Rendered marker positions must avoid dot
  lattice sites (including back-level half-sites) or blobs merge and corrupt
  centroids.
- `test_selfcal.jl` renders particles on a displaced/tilted sheet
  `z = a + bX + cY` through cameras calibrated to `z = 0` (`sheet_instant` /
  `sheet_pair_frames`) — ground truth for the recovered plane, the corrected
  frame (`report.R * w + report.t` must land on the sheet), and post-fix
  reconstruction.

## Development planning

All outstanding work is tracked in [ROADMAP.md](ROADMAP.md), including the
cross-platform GUI framework evaluation. GLMakie is the current implementation;
the historical decision to keep all widget chrome in Makie is open for review.
Preserve the framework-free controller boundary when evaluating a new shell.
Keep this file focused on current architecture, commands, and implementation
conventions rather than maintaining a second backlog.
Record user-visible changes in `CHANGELOG.md`; `RELEASING.md` describes the
core-first/GUI-second release procedure and required validation evidence.
