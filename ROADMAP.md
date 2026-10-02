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

- [ ] Build a reproducible dataset scorecard distinguishing synthetic ground
  truth, known-motion experiments, and real-data smoke checks. Record dataset
  identity, processing recipe, selection rules, and reference conventions.
- [ ] Cover bias/RMS error, valid-vector yield, spatial-resolution sensitivity,
  uncertainty coverage and normalized errors, runtime, and peak host/device
  memory across particle density, diameter, noise, shear, and dropout conditions.
  Include independent Challenge data beyond the committed A pair and 4E slice;
  use an independent implementation comparison where useful and reproducible.
- [ ] Evaluate PTV correspondence precision/recall, trajectory identity switches,
  fragmentation, and gap recovery on independent or annotated recordings.
- [ ] Add a reproducible larger-data evaluation command with download/cache and
  checksums; keep a small deterministic regression subset in ordinary CI.
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

- [ ] Define a versioned experiment/run record shared by Julia scripts and the
  GUI: input identities, full pass schedule, preprocessing, original masks/ROI,
  calibration/dewarping/self-calibration inputs, scale, backend/precision, and
  software environment. Define migration and compatibility behavior.
- [ ] Serialize built-in processing recipes and reference user-supplied scripts
  for custom callbacks; make unsupported automatic replay explicit.
- [ ] Save/reopen an experiment, compare processing revisions, and rerun from its
  recipe without reconstructing settings manually. Preserve existing result-only
  workflows and the core/GUI dependency boundary.
- [ ] Preserve exposure timestamps, pair delay, sample time, source frame IDs,
  time units, and calibration/coordinate-frame identity through native persistence
  and exports. Define the time assigned to a displacement measurement.
- [x] Check stereo exposure synchronization when timestamps are available;
  matching pair delays alone does not establish simultaneous acquisition.
  Sequence and ensemble drivers now check exposure times and declared/observed
  delays before loading or opening output; `sync_atol`/`sync_rtol` and
  `missing_timestamps = :allow/:error` define tolerance and missing-data policy.
  Covered by [stereo timing regressions](test/test_stereo_timing.jl).
- [ ] Support actual sample times for tracking and reject or explicitly handle
  irregular sampling in analyses that assume a fixed interval. Validate unit and
  coordinate compatibility before combining results.
- [x] Apply `PlanarTransform` consistently to planar-grid table and VTK exports,
  including origin, rotation, reflection, anisotropic scaling, and vector basis.
  Raw pixel results require explicit units and optional pair delay; attached
  scales are rejected. Mixed-axis uncertainty is unavailable unless independence
  is explicitly assumed. [Transform export tests](test/test_transformed_export.jl)
  check geometry, uncertainty, default compatibility, and rejection before writes.
- [ ] Extend calibrated table export to PTV and tracking after defining how
  anisotropic transforms affect particle diagnostics, trajectory velocities,
  and their units. Current transformed exports accept planar PIV grids only.
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
- [ ] Resume using input and recipe identities, recording completed entries and
  explicit run status. Prevent accidental duplication or mixing of analyses.
- [ ] Define atomic per-result writes/checkpoints and recovery behavior for
  handled failure, cancellation, and process termination; test each separately.
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

- [ ] Preserve per-vector measurement history: primary versus alternative peak,
  rejected/replaced status, validation reasons, and uncertainty applicability.
  Make diagnostic/uncertainty association clear after peak substitution or filling.
- [ ] Expose actual pass iteration counts, convergence outcomes, and residual
  displacement summaries rather than only requested settings.
- [ ] Generate a saved run-quality report shared by scripts and the GUI, including
  rejection/replacement fractions, unavailable uncertainty, peak locking, and
  representative window-size/preprocessing sensitivity comparisons.
- [ ] Add per-particle position and displacement uncertainty with validated
  detection-fit and match semantics, then add GUI uncertainty overlays.
- [ ] Propagate uncertainty into derived quantities after specifying spatial
  error-correlation assumptions (Wieneke 2015 §3.2). Distinguish correlation
  random error from calibration, timing, and other uncertainty contributions.
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

This shortlist is based on upstream documentation checked on 2026-10-02, not on
executed HammerheadGUI prototypes. Platform and packaging support must be proven
for the assembled application.

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
HammerheadGUI requirements. This supports trying it first; it is not a resolved
environment or runtime compatibility test.

- [ ] Define the desktop interaction requirements: experiment/file browser,
  editable parameter tables/forms, errors next to inputs, keyboard navigation,
  HiDPI, menus, and resizable panels. Include separate settings and visualization
  windows, consistent with the original design preference, and compare with an
  integrated layout using the same controllers.
- [ ] Prototype a Qt/QML batch form and existing-controller result explorer,
  including image pan/zoom, dense vectors, picking, mask gestures, live results,
  cancellation, and window close/reopen. Compare against the current GLMakie
  baseline; assess GTK4 and Bonito against the same requirements if needed.
- [ ] Resolve the candidate environment against supported Julia/Makie versions;
  validate Windows, macOS, and Linux, startup latency, memory, input/HiDPI behavior,
  and responsiveness during CPU/GPU work. Check accessible labels and focus order.
- [ ] Record the framework decision, dependency/maintenance cost, distribution
  requirements, and migration sequence. If selected, migrate one tool at a time
  with controller parity and rendering tests; keep toolkit dependencies in the
  GUI package. Evaluate rendering reuse rather than assuming existing GLMakie
  widget layouts transfer unchanged.
- [ ] Connect setup, preprocessing, mask/ROI, calibration, representative-pair
  comparison, batch, saved quality reports, and export through the shared
  experiment record from slice 2.
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
