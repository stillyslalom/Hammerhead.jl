# CLAUDE.md

Hammerhead.jl — particle image velocimetry (PIV) in Julia. Development is
organized around the International PIV Challenge cases; see ROADMAP.md for
phases and status. Scope is capped at planar 2D2C + stereo 2D3C (tomographic
PIV is out of scope). Phases 1 (file I/O & batch), 2 (masking), 3 (ensemble
correlation & time-series statistics), and 4 (accuracy/UQ) are done; next is
Phase 5 (stereo).

## Commands

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'   # full suite, ~1 min after precompile
julia --project=docs docs/make.jl                    # docs ("skipping deployment" warning is normal locally)
```

Two `PIV sequence failed` error logs during tests are intentional
(failure-propagation tests), not failures.

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
- `quality.jl` — UOD, peak ratio, correlation moment, validator pipeline,
  `replace_vectors!`, `smooth_field`
- `masking.jl` — `polygon_mask`
- `pipeline.jl` — `run_piv`, `piv_pass` (WIDIM multi-pass with symmetric
  image deformation), `process_windows!`
- `io.jl` — `load_image`/`load_mask` (FileIO), `save_results`/`load_results`
  (JLD2: `format_version` + `results/000001`… + optional `sources/…`),
  `run_piv_sequence` batch driver
- `ensemble.jl` — `run_piv_ensemble` (sum-of-correlation; per-chunk
  correlators reused across pairs; multi-pass via shared predictor)
- `statistics.jl` — `field_statistics`, `validate_temporal!`,
  `power_spectrum`
- `ext/HammerheadMakieExt.jl` — `plot_vector_field[!]` (weakdep Makie)

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
- **Threading:** `piv_pass` chunks windows across tasks, one correlator per
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
- Adding a `PIVResult` field breaks the direct constructor calls in
  `test_validation.jl`, `test_ensemble.jl`, `test_accuracy.jl`, and
  `bench/run_benchmarks.jl` — update them all.
- `test/reference_images/A/` holds PIV Challenge case A TIFFs for the
  end-to-end reference test.

## Deferred backlog

Phase 5 (stereo: calibration, disparity correction, dewarping, 3C
reconstruction). Also: physical units/scaling, multi-frame TIFF in
`load_image`, dynamic (per-frame) masks, temporal spectra beyond the
per-point `power_spectrum` utility, uncertainty propagation into derived
quantities (Wieneke 2015 §3.2: needs spatial error autocorrelation).
