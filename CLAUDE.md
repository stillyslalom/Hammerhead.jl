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
- Docs-only deps (Literate, DocumenterCitations, CairoMakie) live in
  `docs/Project.toml`; the core package must not gain doc/GUI deps.
- Explanation pages are the user-facing rewrite of the conventions below —
  when a convention changes, update both.

## Architecture (src/, included in this order)

- `types.jl` — `PIVParameters` (immutable, validated in inner constructor;
  `keep_correlation_planes` opts into per-window plane storage),
  `PIVResult{T}` (trailing `correlation_planes` field, `nothing` unless
  opted in; backward-compatible 11-arg constructors keep old call sites valid)
- `synthetic_data.jl` — synthetic particle images with ground truth
- `preprocessing.jl` — background subtraction, intensity cap, highpass, CLAHE
- `correlators.jl` — `CrossCorrelator{T}`/`PhaseCorrelator{T}` cache FFTW
  plans + buffers per window size; subpixel peak fits
- `uncertainty.jl` — Wieneke 2015 correlation-statistics uncertainty
  (per-window `accumulate_uncertainty!` + `finalize_uncertainty`)
- `transforms.jl` — affine transforms, image warping, registration
- `calibration.jl` — `PinholeCamera` (normalized DLT) / `SoloffCamera`
  (19-term polynomial) / `TransformedCamera` (rigid world pre-transform
  wrapper), `calibrate_camera`, `world_to_pixel` / `pixel_to_world`,
  `apply_world_transform`, fit-quality metrics
- `target_detection.jl` — `detect_calibration_grid` (dot-grid plates →
  indexed point pairs), `calibration_points`, `calibrate_camera` /
  `calibration_quality` convenience methods on `(grids, zs)`,
  `render_calibration_target` (synthetic ground-truth fixture)
- `dewarp.jl` — `DewarpGrid` (world-plane pixel grid spec, shared per rig) +
  `ImageDewarper` (per-camera precomputed source-coordinate map), `dewarp[!]`
  cubic B-spline resampling onto the common plane, `common_dewarp_grid`
  (auto grid from camera footprints: intersection/union, `:auto` spacing,
  descending `y`)
- `quality.jl` — UOD, peak ratio, correlation moment, validator pipeline,
  `replace_vectors!`, `smooth_field`
- `masking.jl` — `polygon_mask`
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
- `particles.jl` — `Particles` (struct-of-arrays) + `detect_particles`
  (local-maxima + 3-point log-Gaussian fits) and the shared uniform-cell
  neighbor list (`build_cell_list`/`within_radius!`/`knn`) used by dedupe,
  matching, and scattered UOD
- `ptv.jl` — `PTVParameters`, `PTVResult`, `run_ptv` (hybrid PIV-guided
  greedy matching via `greedy_match`), `scattered_uod`, `ptv_to_grid`
  (`bin_to_grid`)
- `tracking.jl` — `Trajectory`, `TrackingResult`, `track_particles`
  (constant-velocity + field predictor linking), `trajectory_velocities`
- `stereo.jl` — `StereoPIVResult` + `run_piv_stereo` (per-camera 2C on
  dewarped images → geometric least-squares 3C reconstruction with
  uncertainty propagation)
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` 5 + `results/000001`… + optional `sources/…`;
  entries may be `PIVResult`, `StereoPIVResult`, or `PTVResult`),
  `run_piv_sequence`/`run_ptv_sequence` batch drivers (shared `_run_sequence`;
  `output` accepts a single path or an `(i, pair) -> path` function for
  per-pair files; the next pair's load+preprocess is prefetched on a
  `Threads.@spawn` task while the current pair's `process` runs — overlaps
  slow-source IO with compute only under ≥2 threads, results bitwise-identical
  to serial; `run_piv_sequence` also holds one `PIVWorkspace`, reused across
  pairs — the workspace lives only on the serial `process` call, never the
  prefetch task), `frame_index_strings` (differing frame-index substrings
  from a path pair)
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor; one
  `PIVWorkspace` reuses the interpolant/deform buffers across pairs)
- `selfcal.jl` — `self_calibrate` (Wieneke 2005 disparity self-calibration:
  ensemble cam1↔cam2 disparity map → triangulation → sheet-plane fit →
  rigid world transform of both cameras) + `SelfCalibrationReport`
- `statistics.jl` — `field_statistics`, `validate_temporal!`,
  `power_spectrum`
- `ext/HammerheadMakieExt.jl` — `plot_vector_field[!]` (weakdep Makie; grid
  methods take `stride`, auto `lengthscale = :auto`, and
  `show_replaced`/`replaced_color`; scale via the core `arrow_lengthscale`
  helper, testable without Makie)

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
naming): `ResultExplorer`/`result_explorer`; `MaskEditor`/`mask_editor`
(gesture API `click!`/`alt_click!` holds the editing model; the view only
forwards mouse/key events; `Hammerhead.polygon_mask(::MaskEditor)` exports
the mask, `save_mask` writes the white-=-excluded image `load_mask` reads);
`BatchRunner`/`batch_runner` (runs `run_piv_sequence` with its progress
callback inside `@async` — cooperative, so GL renders keep happening off
`run_piv`'s internal thread-spawn yields while observables stay on the
primary thread; cancel = throw `BatchCancelled` from the callback, which
keeps finished pairs in the incremental output); `CalibrationReview`/
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
  Float64 island (O(vector grid), converted on store); no dt/velocity
  scaling (still deferred).
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
  Pixels only, no dt/velocity scaling. `particles.jl` is included before
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
- Adding a `PIVResult` field breaks the direct constructor calls in
  `test_validation.jl`, `test_ensemble.jl`, `test_accuracy.jl`,
  `test_stereo.jl`, and `bench/run_benchmarks.jl` — update them all.
  Adding a `StereoPIVResult` field likewise breaks the fixture in
  `HammerheadGUI/test/runtests.jl`. Adding a `PTVResult` field breaks the
  direct constructor call in `test_ptv.jl` (the `ptv_to_grid` sparse-corner
  fixture).
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

Physical units/scaling (dt/velocity — still absent from `StereoPIVResult`
too), target detection for rolled cameras, multi-frame TIFF in `load_image`,
dynamic (per-frame) masks, temporal spectra beyond the per-point
`power_spectrum` utility, uncertainty propagation into derived quantities
(Wieneke 2015 §3.2: needs spatial error autocorrelation), light-sheet
thickness/overlap estimation from disparity correlation peak widths
(Wieneke 2005 §5), multi-frame real-data doc demos (`compute_background` /
`run_piv_ensemble` on Challenge sequences like 2A/4A — the committed
case-A pair covers the single-pair real-data tutorial, but sequences are
too large to commit and need build-time download from pivchallenge.org,
e.g. DataDeps.jl in `docs/Project.toml`, within the docs CI budget).

PTV (Phase 8) deferrals: gap bridging / track re-acquisition (no bridging in
v1 — an unmatched head just terminates), relaxation-method matching, match
costs beyond distance (intensity/diameter similarity), per-particle position
uncertainty, world-unit PTV results (needs the units/scaling item above),
stereo PTV, GUI explorer support for `PTVResult`/`TrackingResult`, and the
Duncan et al. distance-weighted scattered-UOD variant (v1 is the plain test).
