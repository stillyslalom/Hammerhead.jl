# Hammerhead roadmap

The active backlog for Hammerhead and HammerheadGUI. Historical phases and
design records live under [reference/archive](reference/archive/README.md).
Add new work here; update `CLAUDE.md` when an item lands and `CHANGELOG.md`
when it is user-visible.

Scope: planar 2D2C and stereo 2D3C PIV, planar PTV, and the workflows around
them. Tomographic PIV, volumetric reconstruction, acquisition hardware
control, Shake-the-Box, pressure-from-PIV, and modal-analysis suites are out
of scope.

## Implemented baseline

Both packages are registered (core v0.1.0, HammerheadGUI v0.1.1). The core
provides planar and stereo PIV (calibration, target detection, dewarping, 3C
reconstruction, Wieneke 2005 self-calibration), ensemble correlation,
Wieneke 2015 uncertainty, statistics and derived quantities, masking,
physical scaling, PTV and gap-aware tracking, JLD2 persistence, table/VTK
export, and CPU, KernelAbstractions, CUDA, and AMDGPU execution. Since the
registered release:

- Larger frame-B search areas (CPU single-pair and ensemble).
- Non-informative correlation planes and contrast-free deformed windows are
  reported as unavailable (NaN, flagged) on CPU, KA, and GPU paths.
- Saved processing settings: `PIVRecipe`, `save_recipe`/`load_recipe`,
  `apply_recipe` (planar/stereo, sequence/ensemble; results files carry their
  recipe), `recipe_diff`.
- Streaming and bounded memory: `collect_results = false`, lazy `ResultFile`
  browsing, `FieldStatisticsAccumulator`.
- Stereo exposure-synchronization checks; tracking table export;
  `PlanarTransform`-calibrated grid exports; calibrated PIV/PLIF resampling
  (`resample_planar`/`resample_image`); `stencil = :centered` derivatives;
  ROI-aware effort presets.
- GUI: result explorer (all four result types, lazy browsing, derived fields,
  profile/circulation tools), mask and ROI editors, preprocessing preview,
  scale tool, planar and stereo batch forms with save/open settings,
  calibration and self-calibration review.

The [backend feature matrix](docs/src/reference/feature_matrix.md) lists
execution support per backend.

On 2026-10-03 a cleanup removed an autonomous agent's provenance/companion
records, experiment and checkpoint formats, quality reports, validation bench
studies, and the Qt/QML GUI prototype. They are recoverable from commit
88a4bda if a specific piece is needed again.

## 1. Accuracy and uncertainty

- [ ] **Investigate Wieneke UQ under-coverage.** On synthetic data the
  reported σ covers only ~15–18% of errors at 1σ (68% nominal), and ~12% of
  windows report σ = 0 because a negative pre-clamp covariance sum is clamped
  to zero (ring truncation + clamp in `src/uncertainty.jl`). A
  Bartlett-weighted covariance improved paired 2σ coverage in a bench
  comparison. Separate bias, residual/convergence, estimator, and rendering
  effects; fix the σ = 0 clamp case; check coverage across seeds before
  changing the default estimator.
- [ ] Cover bias/RMS error, valid-vector yield, spatial resolution, UQ
  coverage, runtime, and peak memory across particle density, diameter,
  noise, shear, and dropout. Include Challenge data beyond the committed A pair
  and 4E slice.
- [ ] Add a reproducible larger-data evaluation command with download/cache
  and checksums; keep a small deterministic subset in ordinary CI.
- [ ] Add real-sequence tutorials for background estimation, ensemble
  correlation, and statistics (e.g. Challenge 2A/4A) with caching inside the
  docs CI budget. Executed docs must not read `cases/`.
- [ ] Evaluate PTV correspondence, identity switches, fragmentation, and gap
  recovery on annotated or independent recordings.
- [ ] Add per-particle position/displacement uncertainty, then GUI overlays.
- [ ] Propagate uncertainty into derived quantities once spatial error
  correlation is specified (Wieneke 2015 §3.2).
- [ ] Confidence estimates for means and Reynolds stresses with explicit
  assumptions about temporal dependence and finite samples.

## 2. GUI: workflow-first redesign

The current GUI is a set of separate tool windows. Users have to know which
window to open next and carry settings between them by hand.

- [ ] **One main window per modality** (planar ✅ `planar_window`, stereo ✅
  `stereo_window`; PTV ✅ as the planar window's particle analysis modes,
  2026-10-04, by user decision), organized as
  steps: Images → Prepare (preprocessing, mask, ROI, and scale as embedded
  panels) → Passes → Test pair → Run → Results. Each step shows its effect on
  a representative pair before the batch runs.
  - Settings save and open through the core recipe API (`PIVRecipe`,
    `save_recipe`, `load_recipe`, `apply_recipe`); the GUI holds no
    private settings format.
  - The GUI calls only public core API. Anything it needs (e.g. effort
    presets, now reached through the internal `effort_schedule`) gets a public
    core function first.
  - Keep the framework-free controller layer; reuse the existing controllers
    as the step panels.
  - Supersedes the parked stash `Shared recipe workbench WIP parked for
    experimental QML integration` (built on the removed revision editors);
    drop the stash once the redesign starts.
- [x] Ensemble runs in the GUI windows (GUI_REDESIGN slice 4, 2026-10-04).
  Stereo save/open settings and calibration files landed with
  `stereo_window` (2026-10-04).
- [ ] Merge the planar and stereo windows into one window with a modality
  choice on Images: the controllers, shell, and QML pages are already shared
  through `AbstractWorkflow`/`WorkflowShell`; the remaining differences are
  the Calibration step, the camera switch, and the canvas type.
- [x] One favicon for the docs site and both windows (2026-10-04, H4:
  a hammerhead-arrow hybrid in Julia's colors; `docs/make_icons.jl`).
- [ ] Exercise the redesigned workflow with a lab user and record concrete
  friction before adding further features.

### Test-drive feedback (2026-10-04)

From the user's first session with the windows. Decisions: recipes gain a
TOML form with sidecar images; interpolation choices become CPU-only core
options (GPU backends keep the defaults); per-frame mask images are input
data like frames, not recipe settings.

- [x] Images: add frames by directory + glob pattern; infer the pattern from
  a selected frame pair (nice to have). Move "Use these settings" from
  Results to Images as **Reuse settings…** with a file selector.
- [x] Viewer: contrast toggle; arrow keys step pairs, a/b (and
  shift+arrows) switch frames; ctrl-click zoom reset also in probe mode; a
  toolbar (zoom, pan, reset, screenshot) with shortcut tooltips.
- [x] Prepare: per-frame mask images (core: dynamic masks as an
  `apply_recipe` input); a separate ruler image for **Measure the pixel
  size**.
- [x] Passes: validation and replacement settings; padding and Gaussian
  weighting as separate toggles; correlation probe; interpolation choices.
- [x] Run: save in-memory results to a file after the run; **Clear** for the
  output path.
- [x] Results: fix the field selector (wrong field chosen); use the main pair
  bar instead of a second slider, with the Results frame independent of the
  Images/Prepare pair; absolute/percentile color limits with user values;
  diverging symmetric colormap for vorticity; profile of the selected field;
  draggable profile/circulation points and deleting circulation points;
  physical-units toggle; particle image as a display option (frames A/B);
  re-validation/replacement settings on the shown results.
- [x] U.S. English in the GUI and docs ("color", "center", "millimeters",
  "neighbor").
- [x] Core: TOML recipes with sidecar images (results files embed the same
  TOML); CPU interpolation options for image deformation and the predictor.

### Framework choice

**Decided 2026-10-03: Qt Quick via QML.jl + QMLMakie**, with the Makie
canvas embedded in the main window and an optional pop-out window. On-screen
Windows probes with a 2048² image and 16k vectors measured ~120 fps embedded,
both idle and under background compute. They showed native window decorations
and FluentWinUI3 controls. GTK4 + Gtk4Makie also worked, at ~62 fps, but looks
like GNOME on Windows. The earlier Codex QML prototype ran Qt offscreen, which
has no GL context on Windows, so its embedding failure was an artifact. The
real Qt issues and their workarounds (a teardown crash after a pop-out, a
render-thread `sleep` in QMLMakie, a missing style DLL, callbacks that must not
block) are in the archived
[GUI redesign plan](reference/archive/GUI_REDESIGN.md).

- [ ] Report the teardown crash to QML.jl, and the `disconnect_screen` sleep
  to QMLMakie, with the captured backtrace.
- [ ] Validate the shell on Windows, macOS, and Linux (startup, memory,
  input/HiDPI, responsiveness during CPU/GPU work).
- [ ] Evaluate a PackageCompiler desktop bundle (relocatable assets, installer
  size, cold start, clean-machine launch).

## 3. Performance and GPU

- [ ] Port AMDGPU's aggregate workspace batch budgeting (fair sharing, LRU
  eviction, no-workspace cleanup) to CUDA. See
  [bench/BATCH_HANDOFF.md](bench/BATCH_HANDOFF.md); verify stable VRAM across
  changing schedules, sizes, and precisions.
- [ ] Record repeatable CUDA/AMDGPU validation as release evidence; keep the
  hardware-free `:ka` tier in CI.
- [ ] Production benchmarks from the original requirements: ~29 MP pairs,
  200–500 images, including time to a useful rough field, on named hardware.
- [ ] Reprofile residual B-spline prefilter allocations on production
  workloads; prefer an upstream fix.
- [ ] Extend independent search footprints to KA/CUDA/AMDGPU if benchmarks
  justify it.
- [ ] Profile device dewarping and preprocessing (background/highpass first,
  CLAHE later); move validation/replacement only if dense-grid costs justify it.
- [ ] Additional GPU backends (Metal, oneAPI) only with a user need and
  hardware to validate on.

## 4. Data, timing, and calibration

- [ ] Preserve exposure timestamps, pair delay, and sample time through native
  persistence and exports; define the time assigned to a displacement.
- [ ] Reject or explicitly handle irregular sampling in analyses that assume a
  fixed interval.
- [ ] Harden rolled-target detection with synthetic perspective/noise fixtures
  and a real rotated-target regression.
- [ ] Optional video ingestion through a weak dependency or frame-source
  adapter, preserving order and timing.
- [ ] Evaluate HDF5/netCDF once there is a concrete archival need.

## 5. Evidence-dependent algorithm extensions

Each needs a failing baseline case and a demonstrated improvement.

- [ ] Variance-normalized cross-correlation for strong illumination changes,
  compared against phase correlation and CLAHE.
- [ ] Adaptive/nonuniform interrogation, after a resolution benchmark.
- [ ] Ensemble `max_iterations` (currently ignored), if low-SNR cases justify
  the cost.
- [ ] Dynamic per-peak exclusion radii versus the regional-max and
  fixed-exclusion finders.
- [ ] Relaxation-method particle matching and Duncan et al.'s
  distance-weighted scattered UOD.
- [ ] Stereo PTV with validated correspondence and geometry.
- [ ] Light-sheet thickness/overlap from disparity-peak widths (Wieneke 2005
  §5).
- [ ] Morphological phase separation, when a two-phase use case is
  contributed.

## Release practice

- [ ] Confirm the hosted Windows four-thread CI job passes on this branch.
- [ ] Next release: core first, then the GUI with a raised Hammerhead compat
  bound. Registered HammerheadGUI 0.1.1 already passes `on_result` to
  `run_piv_sequence`, which registered core 0.1.0 lacks, so the GUI batch
  form needs this core release. See [RELEASING.md](RELEASING.md).
