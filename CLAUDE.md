# CLAUDE.md

Hammerhead.jl — particle image velocimetry (PIV) in Julia. Development is
organized around the International PIV Challenge cases; see ROADMAP.md for
phases and status. Scope is capped at planar 2D2C + stereo 2D3C (tomographic
PIV is out of scope). All five phases are done: 1 (file I/O & batch),
2 (masking), 3 (ensemble correlation & time-series statistics),
4 (accuracy/UQ), and 5 (stereo: camera calibration, target detection,
dewarping, 3C reconstruction via `run_piv_stereo` → `StereoPIVResult`, and
Wieneke 2005 disparity self-calibration via `self_calibrate`) — every planar
and stereo Challenge case is now reachable. Case 4E data (particle +
calibration images) sits in `cases/` (gitignored); a minimal 4E subset
for the docs and tests is committed at `test/reference_images/E/`. Phase 6 (Diátaxis docs,
July 2026) is also done; version is 0.1.0, awaiting first General-registry
registration (a maintainer action). Phase 7 (HammerheadGUI) is underway:
the monorepo conversion is done (skeleton package, CI/TagBot/CompatHelper
subdir wiring); the GUI components (result explorer, mask editor, parameter
form, calibration diagnostics, packaging) are next. Phase 8 (2D2C PTV,
July 2026) is done: per-frame particle detection (`detect_particles`),
hybrid PIV-guided two-frame tracking (`run_ptv` → `PTVResult`, with
`ptv_to_grid` binning and `run_ptv_sequence` batch), scattered validation,
multi-frame trajectory linking (`track_particles` → `TrackingResult`), and
docs — all synthetic-verified, no new deps.

## Commands

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'   # full suite, ~1 min after precompile
julia --project=docs docs/make.jl                    # docs, ~7 min: executes all six tutorials ("skipping deployment" warning is normal locally)
julia --project=HammerheadGUI -e 'using Pkg; Pkg.test()'  # GUI tests (needs a GL context; CI wraps in xvfb-run)
```

Two `PIV sequence failed` error logs during tests are intentional
(failure-propagation tests), not failures.

## Documentation (docs/)

Diátaxis layout under `docs/src/`: `tutorials/` (generated — do not edit),
`howto/`, `explanation/`, `reference/`, plus `index.md` and `references.md`
(bibliography). Rules that keep the build green:

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
  high effort includes final-pass UQ, and ensemble high repeats the final
  window because ensemble ignores `max_iterations`),
  `PIVWorkspace`/`piv_workspace()` (optional `workspace` kwarg reusing the
  padded B-spline coefficient buffers via `image_interpolant!`+`interpolate!`,
  the deform buffers, and a per-window-config correlator pool across `run_piv`
  calls — bitwise-identical; the sequence/ensemble drivers hold one)
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
  trajectories retain explicit frame indices), `trajectory_velocities`
- `stereo.jl` — `StereoPIVResult` + `run_piv_stereo` (per-camera 2C on
  dewarped images → geometric least-squares 3C reconstruction with
  uncertainty propagation), synchronized `run_piv_stereo_sequence`, and
  per-camera-correlation `run_piv_stereo_ensemble`
- `scaling.jl` — `with_scale` (attach/strip `PhysicalScale` metadata,
  arrays shared) + `physical` (same-type conversion to physical units) +
  `plot_axis_labels` (Makie-free label helper) for all four result types
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` 1 + `results/000001`… + optional `sources/…`;
  entries may be `PIVResult`, `StereoPIVResult`, `PTVResult`, or
  `TrackingResult`; the
  pre-registration dev formats were retired without a load shim when the
  `scale` field landed),
  `run_piv_sequence`/`run_ptv_sequence` batch drivers (shared `_run_sequence`;
  `output` accepts a single path or an `(i, pair) -> path` function for
  per-pair files; the next pair's load+preprocess is prefetched on a
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
  `export_vtk`
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor; one
  `PIVWorkspace` reuses the interpolant/deform buffers across pairs)
- `selfcal.jl` — `self_calibrate` (Wieneke 2005 disparity self-calibration:
  ensemble cam1↔cam2 disparity map → triangulation → sheet-plane fit →
  rigid world transform of both cameras) + `SelfCalibrationReport`
- `statistics.jl` — planar/stereo `field_statistics`, 2C/3C
  `validate_temporal!`, `power_spectrum`
- `derived.jl` — mask-aware derivatives, vorticity/divergence/strain,
  swirling strength/Q, profile/region extraction, circulation, and
  attached-scale results-vector spectra
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
(only Hammerhead + Observables + Printf in scope — the module boundary
enforces the no-Makie rule, and a test asserts it); `src/views/*.jl` are the
GLMakie shells. Components so far (each = controller + view pair, same
naming): `ResultExplorer`/`result_explorer` (browses all four persisted
result types — `PIVResult`/`StereoPIVResult` grids, `PTVResult` particle
scatter, `TrackingResult` gap-aware polylines colored by mean speed — mixed
sequences included; routes each entry through `physical` at construction so a
`PhysicalScale` gives physical-unit axis/colorbar/inspection labels;
selection is a `CartesianIndex` for grids, a linear `Int` for scattered
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
estimators; the profile panel appears as a third layout row);
`MaskEditor`/`mask_editor`
(gesture API `click!`/`alt_click!` holds the editing model; the view only
forwards mouse/key events; `Hammerhead.polygon_mask(::MaskEditor)` exports
the mask, `save_mask` writes the white-=-excluded image `load_mask` reads);
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
exports a frame-copying, snapshot-semantics closure for
the batch drivers); `ScaleTool`/`scale_tool` (two clicked points + known
separation → `PhysicalScale`; `apply_scale!` into a batch form);
`StereoBatchRunner`/`stereo_batch_runner` + `stereo_calibration` (two
synchronized frame lists + an `ImageDewarper` pair —
`build_dewarpers(cr1, cr2)` composes `common_dewarp_grid` from two fitted
`CalibrationReview`s, and the workflow view embeds both reviews via the
embeddable `calibration_review!`; runs `run_piv_stereo_sequence` with its
NATIVE zero-arg `cancel` predicate — no exception, completed prefix
returned — and a dt-only stereo scale);
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

## Deferred backlog

Rolled-target detection still needs broader synthetic coverage and a real
rotated-target regression fixture; `orientation = :fiducials` already provides
the roll-invariant convention when both markers are visible. Other analysis
deferrals are uncertainty propagation into derived quantities (Wieneke 2015
§3.2: needs spatial error autocorrelation) and light-sheet thickness/overlap
estimation from disparity-correlation peak widths (Wieneke 2005 §5). Multi-frame
real-data doc demos (`compute_background` / `run_piv_ensemble` on Challenge
sequences like 2A/4A) still need build-time download/caching within the docs CI
budget; the committed case-A pair covers only the single-pair tutorial.

PTV deferrals: relaxation-method matching, per-particle position/displacement
uncertainty, stereo PTV, GUI explorer support for persisted
`PTVResult`/`TrackingResult` (including trajectory gaps and unit-labeled axes),
and the Duncan et al. distance-weighted scattered-UOD variant (v1 is the plain
test).
