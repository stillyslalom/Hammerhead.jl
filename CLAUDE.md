# CLAUDE.md

Hammerhead.jl — particle image velocimetry (PIV) in Julia. Development is
organized around the International PIV Challenge cases; see [ROADMAP.md](ROADMAP.md)
for the active backlog (historical phases live in `reference/archive/ROADMAP.md`).
Scope is capped at planar 2D2C + stereo 2D3C (tomographic
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
the Qt planar workflow window (`planar_window`: Images → Prepare → Passes →
Test pair → Run → Results, saved settings as recipes) replaced the GLMakie
tool windows; the standalone result explorer and the stereo calibration/batch
views remain until the stereo window. Phase 8 (2D2C PTV,
July 2026) is done: per-frame particle detection (`detect_particles`),
hybrid PIV-guided two-frame tracking (`run_ptv` → `PTVResult`, with
`ptv_to_grid` binning and `run_ptv_sequence` batch), scattered validation,
multi-frame trajectory linking (`track_particles` → `TrackingResult`), and
docs — all synthetic-verified, no new deps.

A 2026-10-03 cleanup removed an autonomous agent's provenance, companion,
experiment-record, checkpoint, quality-report, and Qt/QML layers (recoverable
from commit 88a4bda). Saved settings now go through the single recipe API in
`recipes.jl`; don't reintroduce parallel record/companion formats.

## Commands

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'   # full suite
julia --project=docs docs/make.jl                    # docs: executes all seven tutorials ("skipping deployment" warning is normal locally)
julia --project=HammerheadGUI -e 'using Pkg; Pkg.test()'  # GUI tests (needs a GL context; CI wraps in xvfb-run)
# opt-in Qt window test (own process, needs a display): set HAMMERHEADGUI_QT_TESTS=true
julia --project=docs -t 4 docs/gui_screenshots.jl    # local only: regenerate docs/src/assets/gui_window/*.png, then look at them
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

- Organize the site around reader tasks. Tutorials lead with a concrete
  question, executable example and figure, then explain the result; page
  titles name a reader's task; format/edge-case contracts go in API reference.
  Write directly and affirmatively; place a limitation beside the decision it
  changes and avoid habitual negative caveats.
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
- The GUI tour (`docs/lit/gui_tour.jl`) drives a `PlanarWorkflow` through
  the window's controller calls (no display needed) and embeds window
  screenshots committed under `docs/src/assets/gui_window/`, made by the
  local-only `docs/gui_screenshots.jl` on the tour's own synthetic frames.
  When the window's look or the tour's scene changes, regenerate them, look
  at each image, and keep each under 300 KB.
- Reference pages use `@autodocs` filtered by source file (`Pages =
  ["pipeline.jl", ...]`). A new `src/*.jl` file's public docstrings must be
  added to one of the reference pages (and every documented binding must
  appear somewhere) or `makedocs` fails its checkdocs pass.
  `Pages` suffix-matches: `"calibration.jl"` also catches
  `planar_calibration.jl`, so use `"src/calibration.jl"` there.
  `reference/internals.md` catches all non-exported docstrings via
  `Public = false`. A page's HTML must stay under Documenter's 200 KiB
  `size_threshold` (the build fails above it): the GUI reference is split
  into `reference/gui.md` (planar/shared), `gui_stereo.md`, `gui_results.md`.
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
- `transforms.jl` — affine transforms, image warping, registration (manual
  fits reject nonfinite, rank-deficient, and singular inputs)
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
  descending `y`), `save_calibration`/`load_calibration` (plain-dict JLD2,
  `CALIBRATION_FORMAT_VERSION = 1`: pinhole/Soloff/`TransformedCamera`
  models + image sizes + the shared grid; dewarpers rebuild bitwise — the
  pinhole reload skips renormalization via `PinholeCamera(P, Val(:normalized))`)
- `quality.jl` — UOD, peak ratio, correlation moment, validator pipeline,
  `replace_vectors!`, `smooth_field`
- `masking.jl` — `polygon_mask`, intensity/contrast/edge `automatic_mask`,
  and circular `grow_mask`/`shrink_mask`
- `source_support.jl` — original-pixel contrast gate for deformed windows
  (lazy UInt8 maps of empty/constant/varying raw 4×4 stencils; see the
  non-informative-windows convention)
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
  planar presets size to the ROI when one is given; high effort includes final-pass UQ, and ensemble high repeats the final
  window because ensemble ignores `max_iterations`),
  `PIVWorkspace`/`piv_workspace()` (optional `workspace` kwarg reusing the
  padded B-spline coefficient buffers via `image_interpolant!`+`interpolate!`,
  the deform buffers, and a per-window-config correlator pool across `run_piv`
  calls — bitwise-identical; the sequence/ensemble drivers hold one).
  Singleton predictor axes extend constantly, so a coarse window may span a
  whole image/ROI dimension.
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
  check camera exposure timestamps and declared `FramePair.dt` before loading
  or opening output (`sync_atol`/`sync_rtol` scale with pair delay;
  `missing_timestamps = :allow` default, `:error` to require metadata)
- `scaling.jl` — `with_scale` (attach/strip `PhysicalScale` metadata,
  arrays shared) + `physical` (same-type conversion to physical units) +
  `plot_axis_labels` (Makie-free label helper) for all four result types
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` 1 + `results/000001`… + optional `sources/…`;
  entries may be `PIVResult`, `StereoPIVResult`, `PTVResult`, or
  `TrackingResult`; the
  pre-registration dev formats were retired without a load shim when the
  `scale` field landed; `ResultFile(path)` / `load_results(path; lazy = true)`
  index a completed file and load one entry per access, rejecting detectable
  size/mtime changes),
  `run_piv_sequence`/`run_ptv_sequence` batch drivers (shared `_run_sequence`;
  `output` accepts a single path or an `(i, pair) -> path` function for
  per-pair files; `collect_results = false` delivers each result to
  `on_result`/`output` and returns `nothing` (stereo sequence too); the next pair's load+preprocess is prefetched on a
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
  `export_vtk`. `TrackingResult` tables append trajectory/observation IDs,
  frame indices, gaps, and validity columns (additive-column schema policy).
  Planar-grid exports accept `transform = PlanarTransform(...)` with explicit
  length units (and optional pair delay); coordinates and vector basis
  transform together, and results with an attached scale are rejected to
  avoid double conversion
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor; one
  `PIVWorkspace` reuses the interpolant/deform buffers across pairs;
  `progress` may be a `(done, total)` function ticked per pair per pass —
  stereo ensembles count both cameras — and throwing from it aborts)
- `selfcal.jl` — `self_calibrate` (Wieneke 2005 disparity self-calibration:
  ensemble cam1↔cam2 disparity map → triangulation → sheet-plane fit →
  rigid world transform of both cameras) + `SelfCalibrationReport`
- `statistics.jl` — planar/stereo `field_statistics`, 2C/3C
  `validate_temporal!`, `power_spectrum`; `FieldStatisticsAccumulator` +
  `update_statistics!` stream population moments in O(grid) memory
  (grid/scale compatibility checked before mutating; `field_statistics(acc)`
  returns an independent snapshot)
- `derived.jl` — mask-aware derivatives, vorticity/divergence/strain,
  swirling strength/Q, profile/region extraction, circulation, and
  results-vector spectra with an explicit sampling interval (`dt`, independent
  of the image-pair delay in `PhysicalScale`). Profile interpolation ignores
  invalid corners with zero weight at exact nodes/edges. `extract_region`
  returns `included` (`true` = returned valid node); its legacy `mask` field
  aliases that grid and has the opposite convention from `PIVResult.mask`.
  Area `circulation(result; region=...)` now errors on incomplete coverage
  by default; `coverage=:report` returns value, valid/requested area, fraction,
  and authoritative `complete` flag (no valid area gives `NaN` value).
  `flow_derivatives(...; stencil = :centered)` refuses one-sided fallbacks
  (default `:available`).
- `calibrated_resampling.jl` — `resample_planar` / `resample_image`: CPU
  bilinear sampling of raw planar vectors and scalar images onto explicit
  calibrated coordinates (e.g. PIV/PLIF on one grid). Affine geometry and
  vector basis share one map; outputs carry contributor/availability flags,
  so masked or unsupported samples stay distinct from measured zeros.
- `recipes.jl` — saved processing settings. `PIVRecipe(passes;
  preprocessing, mask, roi, scale, mode = :sequence | :ensemble | :ptv |
  :tracking, image_type, predictor_smoothing, mask_threshold, ptv,
  ptv_predictor, min_track_length, max_gap)` (particle modes: passes = PIV
  predictor, planar/CPU only, no ROI; `:tracking` takes the frame sequence
  and returns one `TrackingResult`) with ordered built-in
  `PreprocessStep`s (backgrounds copied in); `save_recipe`/`load_recipe`
  (JLD2, `RECIPE_FORMAT_VERSION = 2` — v2 added per-camera preprocessing;
  v1 still loads, unknown versions rejected);
  `apply_recipe(recipe, pairs; output, backend, ...)` and the stereo method
  `apply_recipe(recipe, pairs1, pairs2, dw1, dw2; ...)` dispatch to the
  sequence/ensemble drivers and store the recipe in the results file, so
  `load_recipe(results_path)` recovers it (stereo results also store the
  calibration: `load_calibration(results_path)`); `recipe_preprocess`,
  `recipe_diff` (`(; path, before, after)` entries). ROI is planar-sequence
  only; saveable preprocessing is a list of `PreprocessStep`s, or for stereo
  a 2-tuple of lists (per camera; `recipe_preprocess` then returns `(f1, f2)`,
  which the stereo drivers and `self_calibrate` accept as `preprocess`).
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
  read-back is bitwise-identical to the recompute; the stats kernel reads the
  cache (raw products, no window-mean centring, matching the CPU). Result: the stats kernel dropped
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

**Workflow window (Qt, GUI_REDESIGN.md):** `planar_window()` is a Qt Quick
app (QML.jl + QMLMakie) over framework-free controllers. `PlanarWorkflow`
(`controllers/planar_workflow.jl`) owns one controller per step — `FrameSet`
(`frame_set.jl`), `PrepareState` (`prepare.jl`: Prepare sub-page + the four
editors `PreprocessPreview`/`MaskEditor`/`ROIEditor`/`ScaleTool`, built for the
current frame *size*, never an image copy), `PassesEditor`, `PairTest`/`RunState`
(`workflow_jobs.jl`) — plus a `ResultExplorer` for Results. Its
`preprocessing`/`mask`/`roi`/`scale` observables are the single source for
`workflow_recipe`; `prepare_workflow.jl` syncs editors ↔ workflow both ways
under a `syncing` guard (opened settings reseed the editors; a loaded mask
becomes the editor's raster), and an unedited opened recipe round-trips `==`.
Test pair and Run both call `apply_recipe`; an ensemble Run returns one
result (no `on_result`; `RunState.mode/pairs/cameras` + `run_progress` turn
the core's per-pair-per-pass ticks into "pass k of P" text; cancel throws
from `progress` and keeps nothing). Particle analysis is a set of planar
modes, not a window: `passes.mode` ∈ `ANALYSIS_MODES` (`:sequence`,
`:ensemble`, `:ptv`, `:tracking`); `ParticleSettings` (`wf.particles`,
`particle_settings.jl`) holds `PTVParameters`/predictor/track options and
the Passes-step detection preview; the recipe omits the ROI in particle
modes (`workflow_problem` reports one); a tracking test follows
`TRACKING_TEST_FRAMES` frames from the representative pair and a tracking
run all listed frames (one pooled result, like an ensemble).
The settings/test/run/results/step-rail functions are `AbstractWorkflow`
methods (`workflow.jl`) over per-workflow hooks (`workflow_steps`,
`workflow_recipe`, `workflow_problem`, `_test_inputs`/`_run_inputs` = the
`apply_recipe` positional args after the recipe, `_inputs_stale`,
`_step_status`, …); `StereoWorkflow` (`stereo_workflow.jl`) reuses them with
linked `frames1`/`frames2` (pair mode, pair index, shown frame copied both
ways), a `camera` switch, and `StereoCalibration` (`stereo_calibration.jl`:
plates → `CalibrationReview`s → `common_dewarp_grid` dewarpers →
self-calibration, all through the runner with generation counters). Its
Prepare viewer is on the dewarped grid on every page: the preview's `post`
hook dewarps the processed pair (probe coordinates = grid), `wf.dewarped`
is the raw pair dewarped, and the mask editor is sized to the grid. Stereo
has no ROI. `wf.preprocessing` is one list or a per-camera tuple (as in the
recipe); the preview edits the shown camera's list via the
`_edited_steps`/`_set_edited_steps!` hooks, and the background estimate
subtracts each camera's own background (switching to per-camera lists).
The Calibration sub-page lives in the controller (`cal.page`); on
`:selfcal` the canvas shows the dewarped frame under the selected pass's
disparity map, with pass 1's arrow scale for every pass.
`src/qt/shell.jl` bridges to QML (`src/qml/`, one page per step,
`steps/prepare/*Pane.qml`) via one `JuliaPropertyMap` (`app`, written through
an equality-guarded `_set!`) + item models (steps, passes, preprocessing
rows), and `hh_*` callbacks that only change state and return. One
`WorkflowShell{W,C}` serves both windows (`PlanarShell`/`StereoShell`
aliases); per-window hooks are `_connect_window!`, `_refresh_frames!`,
`_refresh_region!`, `_refresh_window!` (stereo: calibration fields + the
`plates1Model`/`plates2Model` lists, `qt/stereo_shell.jl`). QML shares
`WorkflowWindow.qml` (chrome, rail, canvas, pop-out, tick, dialogs; its
children are the page stack) — `PlanarWindow.qml`/`StereoWindow.qml` only
list pages; shared pages take a `stereo` flag where they differ.
`stereo_window()` uses `StereoCanvas` (`canvas/stereo_canvas.jl`): raw frame
(Images), plate + dots + residual arrows ×gain in the title (Calibration),
else the dewarped frame in dewarped px with the out-of-view union shaded;
stereo vectors are mapped to grid coords via `grid_vector_data`. The results
canvas flips `yreversed` off for `StereoPIVResult`s (world axes, +Y up).

Gestures are controller functions: canvases register one Makie interaction
that turns left/right clicks into `canvas_click!(wf, x, y)` /
`canvas_alt_click!(wf)` (and keys into `canvas_key!`), consumed only when the
controller used them, so drags still zoom/pan; dispatch is on step + Prepare
page (probe, mask polygon, ROI corner, scale point). The results canvas sends
clicks to the explorer's tool (`click!`/`alt_click!`; Escape →
`canvas_key!(::ResultExplorer, :escape)`). Tests and the docs tour drive the
same calls.

Background work: `wf.spawn[]` (set by `planar_window`, `false` otherwise)
moves pair loading, preview, probe, and background estimate onto
`Threads.@spawn`; a job captures its inputs on the GUI thread, computes from
them alone, and hands its result to `wf.deliver[]`, which in a window queues
it for `hh_tick` (a 16 ms QML Timer) to apply on the GUI thread. Each request
bumps a generation counter and stale results are dropped (latest edit wins);
nothing on the GUI thread reads an image file. Without a window everything
runs inline (tests rely on it); test/run take an explicit `spawn` kwarg.
Closing a window abandons in-flight jobs (`_abandon_jobs!`) and restores
`deliver`/`spawn`. `request_grab(path)` saves the window body via QML
`grabToImage` on the next tick (works offscreen); `_TICK_HOOK[]` lets a
script drive an open window — see `test/qt_window.jl`,
`test/qt_stereo_window.jl`, and `docs/gui_screenshots.jl` (local-only;
regenerates the committed `docs/src/assets/gui_window/*.png`, planar and
`stereo_*`, from synthetic scenes — the stereo one is
`test/stereo_fixture.jl`'s rig).

Rules learned the hard way:
(1) **never destroy a `MakieArea`** — jlqml connects a context-less
`sceneGraphInvalidated` lambda that dangles and crashes at teardown; the main
and pop-out canvases live for the window's lifetime and figures move between
them with a two-phase handshake (`CanvasHost`: the source renders its own
placeholder so GLMakie releases the figure in that GL context). (2) **Never
add or delete plots on a displayed Qt canvas** — GLMakie builds/frees GPU
objects immediately and there is no current GL context outside Qt's render;
canvases (`src/canvas/`) create all plots up front (empty overlays hold a NaN
placeholder) and change only inputs via `_update!` (`Makie.update!` with
`arg1…` keywords; a lone positional hits the Dict method), so the
plot-rebuilding GLMakie views must not be embedded. The results canvas's
profile panel is an Axis + Legend created with the figure in an
`Outside`-aligned nested layout row; outside the profile tool the row is
`Fixed(0)` and the panel's `blockscene`s are hidden via `scene.visible`. This
leans on Makie internals (blockscene; GLMakie skipping invisible scenes;
verified on GLMakie 0.13) — re-check on Makie upgrades; the offscreen test
asserts figure-wide plot counts never change. (3) GLFW/GLMakie contexts and
Qt canvases must not share a process (AMD driver crash on Qt's render thread
after a GLFW context existed), so the Qt window test (`test/qt_window.jl`,
opt-in via `HAMMERHEADGUI_QT_TESTS=true`) and the screenshot script run in
their own processes; `_run_window` throws if `GLMakie.ALL_SCREENS` holds a
non-QML screen, and the standalone GLMakie views warn once a Qt window was
opened (`_QT_OPENED`). (4) On Windows, Qt reads msvcrt's environment copy:
style selection sets `QT_QUICK_CONTROLS_STYLE` through `_putenv_s`, after
preloading the FluentWinUI3 impl DLL. After `exec()` returns, QML screens are
dropped from `GLMakie.ALL_SCREENS` and the atlas cache, so the REPL survives
and the window can reopen; workers must not block in plain ccalls (they stall
every GC). Startup (warm, to the first `hh_tick`) is ~16.5 s: ~10.8 s package
load + ~5.8 s, of which ~4.6 s is still compilation (CxxWrap/QML methods
are defined at init, so their callers do not cache; closures are not
traced). The canvas glyph atlas (~3 s to render) is cached on disk beside
Makie's atlas (`*.hammerheadgui`) and loaded while no screen uses the
session's atlas. `qt/precompile_statements.jl` holds traced methods
(regenerate with `--trace-compile` around both windows' startup and the Qt
scripts, merged with the old file, after GLMakie/QMLMakie upgrades). CI loads QML with
`QT_QPA_PLATFORM=offscreen`.

Monorepo subdirectory package, Makie-style: own Project.toml (this is where
the GLMakie/QML/NativeFileDialog hard deps live — the core never gains GUI deps),
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
a GL context); Makie and QML code render controllers and push input into them
but controllers never import Makie or Qt.

Layout: `src/controllers/*.jl` are included into the `Controllers` submodule
(Hammerhead + Observables/Printf/LinearAlgebra/FileIO only — the module
boundary enforces the no-Makie rule, and a test asserts it; small shared
helpers such as `_errmsg`, `_try_job`, `display_number` and `BatchCancelled`
live in `shared.jl`); `src/canvas/*.jl` are the Qt-safe workflow canvases;
`src/views/*.jl` are the remaining standalone GLMakie windows. Controllers use
Hammerhead's public API — when the GUI needs something new, add it to the core
first. Standalone views (controller + view pair, same naming):
`ResultExplorer`/`result_explorer`
(browses all four persisted
result types — `PIVResult`/`StereoPIVResult` grids, `PTVResult` particle
scatter, `TrackingResult` gap-aware polylines colored by mean speed — mixed
sequences included; routes each entry through `physical` at construction so a
`PhysicalScale` gives physical-unit axis/colorbar/inspection labels;
`lazy = true` or a `ResultFile` browses a completed file holding one
displayed payload, and refuses `push_result!`; selection is a `CartesianIndex` for grids, a linear `Int` for scattered
types; the vector overlay is quiver-style linesegments + rotated-triangle
scatter heads, NOT arrows2d — arrows2d's per-frame pixel-space tip sizing
made pan/zoom crawl at thousands of arrows; colorbar limits default to a
robust 2–98% percentile band over valid vectors (`color_limits`) with
bound-wise manual overrides persisting across frames; `push_result!`
appends live and grows the view's slider via the `count` observable;
planar results add derived fields (:vorticity/:divergence/:strain_rate/
:swirling_strength/:q_criterion via flow_derivatives, cached per frame,
unit-labelled 1/time — physical-at-construction keeps the gradients
exactly 1/dt) and a tool mode (:inspect/:profile/:circulation with
`click!`/`alt_click!` gestures, planar-only, state clears on frame
switches; circulation reports both line-integral and vorticity-area
estimators; `profile_series`/`tool_summary` feed the window's panel);
`CalibrationReview`/
`calibration_review` (+ embeddable `calibration_review!`) + `selfcal_review`
(grid-detection/reprojection review and the `SelfCalibrationReport` browser —
its disparity maps open in an embedded explorer via
`result_explorer!(gridposition, ex)`, the embeddable form all composite views
should use); `build_dewarpers(cr1, cr2)` (in `calibration_review.jl`) builds a
dewarper pair from two fitted reviews for `stereo_window(; dewarpers)`, sharing
`_dewarper_pair` with the window's grid build. The stereo window replaced the
GLMakie stereo batch/calibration views (2026-10-04). Stereo window
conventions: the calibration saves/opens separately from the recipe
(`save_calibration_file`/`open_calibration!` over the core
`save_calibration`/`load_calibration`; opening clears the plate fits and
grid options then rebuild for the opened cameras); plate images given as
paths load in the fit job. The Prepare editors' controllers also work
alone: `PreprocessPreview` holds core `PreprocessStep`s and previews with
`recipe_preprocess`, so it is exactly the batch; `MaskEditor` exports via
`Hammerhead.polygon_mask(::MaskEditor)` and `save_mask` writes the
white-=-excluded image `load_mask` reads; `ROIEditor` → core `ROI` (results
keep original image coordinates); `ScaleTool` → `PhysicalScale`. Shared
widget↔controller sync helpers live in `views/widgets.jl`. View gotchas learned:
in the GLMakie views, recreate heatmap/arrows per refresh instead of updating
per-argument observables (sequential x/y/data updates render transiently
mismatched grids — the Qt canvases avoid this with atomic `_update!`);
preserve zoom by capturing/restoring `ax.targetlimits[]` — `limits!`
normalizes the rect and silently undoes `yreversed`; guard every
widget↔controller observable pair with equality checks (Observables notify
on same-value writes, so unguarded two-way wiring loops forever);
`colorbuffer(fig)` may return the screen's reused framebuffer — `copy` it
before comparing renders in tests; `word_wrap` labels need an explicit
`width` (with `tellwidth = false` they wrap at a bogus narrow width).

## Load-bearing conventions

- **Non-informative windows are unavailable measurements:** nonfinite,
  nonpositive, or completely flat correlation planes yield NaN displacement/
  diagnostics and an (unmasked) outlier flag regardless of validators; exact
  constant windows are centered before apodization; predictors skip
  nonfinite donors. Deformed windows additionally need exact contrast
  (compared in processing precision) in the original raw pixels their B-spline
  stencils sample (`source_support.jl`) — prefilter leakage from distant
  coefficients doesn't count; non-contributing pairs are skipped by CPU/KA
  correlation and UQ.
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
  (that's how the ensemble path pools them). The S_δ are raw products of the
  (1,2,1)-smoothed ΔC_i field — never centre it on the window mean: that
  forces the all-lag sum to zero and biased the ±4 truncated sum low by
  ~81/N (a third of the variance at 16 px; it caused every σ = 0). The sums
  are taken ring by ring until a ring's max drops below `0.05·S00`; inner
  rings are taken whole because their negative members are real signal×noise
  anticorrelation — a per-term positive threshold inflates σ 2–5× at high
  noise. The variance is floored at S00 (eq 8 independent limit; smoothed
  covariance is non-negative in the paper's model), so σ = 0 only for
  identical deformed windows. Bartlett weighting was evaluated and rejected
  (σ ~18% low). Estimates describe the random error only; near-outlier windows
  legitimately report huge σ, so validation comparisons use medians over
  non-outlier vectors. Measured with repeated noise realizations, σ is ~0.9–1×
  the random error; single-vector coverage is ~52–62% / 89–93% (1σ/2σ)
  because σ̂ itself scatters 30–40% (t-like, few dof) — that is intrinsic,
  not a bug. Clean-image total error with 3 px (4σ) SyntheticData particles is
  ~0.01 px systematic deformation-interpolation bias that UQ cannot see, so
  coverage tests must add noise and use midpoint truth.
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
- **Temporal sampling:** `result_spectrum` requires an explicit `dt` for the
  interval between successive velocity samples. Never infer that interval
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
- **Threading:** each chunk/task owns its correlator (correlators are mutable
  state); results must stay bitwise identical to serial (tested).
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
- `test_performance.jl` bounds `run_piv` allocations (256², three passes,
  Float32/64, with and without a mask). A captured variable that is
  reassigned inside a closure becomes a `Core.Box`; one in `_window_mean`
  once made CPU `run_piv` 3–8× slower with 30–100× the allocation. Keep
  per-window loops type-stable and don't raise the bound to make a change pass.
- `test_noninformative_windows.jl` / `test_original_source_support.jl` cover
  flat/constant/masked windows and the original-stencil gate on CPU and KA;
  `test_recipes.jl` covers recipe round trips, `apply_recipe` parity with the
  sequence/ensemble drivers, and recipes stored in results files.
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

Open work lives in [ROADMAP.md](ROADMAP.md); keep this file to architecture,
commands, and conventions. Record user-visible changes in `CHANGELOG.md`;
`RELEASING.md` describes the core-first/GUI-second release procedure.
