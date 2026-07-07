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
form, calibration diagnostics, packaging) are next.

## Commands

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'   # full suite, ~1 min after precompile
julia --project=docs docs/make.jl                    # docs, ~6 min: executes all four tutorials ("skipping deployment" warning is normal locally)
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

- `types.jl` — `PIVParameters` (immutable, validated in inner constructor),
  `PIVResult{T}`
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
  cubic B-spline resampling onto the common plane
- `quality.jl` — UOD, peak ratio, correlation moment, validator pipeline,
  `replace_vectors!`, `smooth_field`
- `masking.jl` — `polygon_mask`
- `pipeline.jl` — `run_piv`, `piv_pass` (WIDIM multi-pass with symmetric
  image deformation), `process_windows!`
- `stereo.jl` — `StereoPIVResult` + `run_piv_stereo` (per-camera 2C on
  dewarped images → geometric least-squares 3C reconstruction with
  uncertainty propagation)
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` + `results/000001`… + optional `sources/…`;
  entries may be `PIVResult` or `StereoPIVResult`), `run_piv_sequence`
  batch driver
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor)
- `selfcal.jl` — `self_calibrate` (Wieneke 2005 disparity self-calibration:
  ensemble cam1↔cam2 disparity map → triangulation → sheet-plane fit →
  rigid world transform of both cameras) + `SelfCalibrationReport`
- `statistics.jl` — `field_statistics`, `validate_temporal!`,
  `power_spectrum`
- `ext/HammerheadMakieExt.jl` — `plot_vector_field[!]` (weakdep Makie)

## HammerheadGUI (HammerheadGUI/)

Monorepo subdirectory package, Makie-style: own Project.toml (this is where
the GLMakie/NativeFileDialog hard deps live — the core never gains GUI deps),
`[sources]` path coupling to the core for dev (Julia ≥ 1.11; the CI `gui` job
`Pkg.develop`s the core instead so lts/1.10 works, and wraps tests in
`xvfb-run`). Releases go core-first, then GUI compat bump; registration is
`@JuliaRegistrator register subdir=HammerheadGUI`, TagBot tags
`HammerheadGUI-v*` (second TagBot job), CompatHelper covers both packages via
`subdirs`. Architecture rule: all application state/logic lives in a
framework-free controller layer (plain Julia + Observables, testable without
a GL context); Makie code renders controllers and pushes input into them but
controllers never import Makie. The mask editor is the framework proving
ground for pure-GLMakie widget chrome.

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
  multipass schedule (repeat the final window size). Statistics accumulate
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
