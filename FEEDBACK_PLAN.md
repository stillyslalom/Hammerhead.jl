# feedback.md implementation plan — remaining Fable scope

Source: `feedback.md`. Part A (items 2, 4, 5, 6, 7, 8, 9 — the docs and the
non-hot-path features) was implemented on branch `feedback-batch-1`. What
remains are the two items reserved for a later Fable session because they
touch the correlation hot path, the accuracy-tolerance conventions, or need
benchmarking/design iteration.

---

## B1. Iterative per-stage multipass (feedback item 1) — DONE (branch `feedback-b1`)

Convergence-looped validate → replace → smooth → re-deform → re-correlate
within each stage, optional per-vector convergence tracking. Reserved
because it needs: a benchmark first (each iteration pays a full O(image)
`deform_images` re-warp regardless of how many windows are skipped, which
may gut the value of per-vector tracking), a convergence-criterion design
(px-change threshold vs. correlation-statistics stationarity), and care
around the Wieneke-2015 UQ contract (final-pass-only, assumes converged
predictor) and the `force_replace` semantics in `run_piv`.

**Landed:** `PIVParameters` gains `max_iterations` (default 1 = classic WIDIM,
exact prior code path) and `convergence_tol` (default 0.05 px; `0` disables
the early exit). The loop lives in `piv_pass`: each sweep re-deforms by the
stage's own validated field (always force-replaced internally for predictor
hygiene) and re-correlates; when the pass semantics don't want replacement,
the measured field is stashed per sweep and restored at still-flagged cells
after the loop (exact — peak-substituted cells are unflagged measured data).
Wieneke UQ is estimated in a post-loop sweep over the last sweep's deformed
windows (`uncertainty_sweep!`; the estimator never reads the correlation
plane, so values are bitwise-identical to the fused path, which non-iterating
passes still use). Key property (tested): an iterated final stage with
`convergence_tol = 0` is *exactly* equal to repeating that pass in the
schedule, UQ included.

**Benchmark verdict — no per-vector convergence tracking.** 1024² pair,
serial: `deform_images` 77 ms/iteration fixed vs. all-window correlation
63–75 ms plain (padded+gauss 225 ms, +UQ 347 ms — but UQ left the loop via
the post-sweep). Skipping even every window saves ≤ 50% of a plain-config
iteration, a converged window's measurement still depends on the predictor
through interpolation + smoothing (skipping is approximate, not exact), and
the O(image) re-warp is unavoidable — so whole-field re-correlation it is.

**Convergence-criterion design — 95th-percentile px change.** Measured on
synthetic scenes (clean vortex, noisy shear, garbage sparse): a max-norm
never converges — a few bistable low-signal windows flicker between
correlation peaks for arbitrarily many sweeps (max change ~1 px while the
median falls below 1e-3 px), and excluding flagged cells doesn't fix it
(cells can pass validation on both sides of a flip). q95 decays cleanly and
with tol 0.05 px stops the clean scene at sweep 2 (where RMS accuracy
peaks), noisy shear at sweep 3, and never falsely converges on garbage
(budget caps it). RMS-vs-truth confirmed sweeps beyond 2–3 don't improve
accuracy, so the early exit matters. Constant `CONVERGENCE_QUANTILE = 0.95`
in `pipeline.jl`.

Not extended to `run_piv_ensemble` (an ensemble sweep would re-correlate
every pair — deferred; its passes ignore `max_iterations`).

## B2. Robust peak finding (feedback item 3) -- DONE

Regional-max peak candidates (reusing the in-house local-maxima machinery in
`src/particles.jl`, *not* a new dep) as a `PIVParameters` option, plus the
researchy dynamic per-peak exclusion radius. Reserved because it changes
`find_peaks` semantics on the correlation hot path, interacts with peak-ratio
validation and the knife-edge StableRNGs testsets in `test_peaks.jl` (fixed
geometry, tight acceptance bands, exclusion-radius-dependent construction),
and the accuracy-tolerance conventions forbid "make the test pass" fixes.

**Landed:** `PIVParameters` gains `peak_finder` (`:exclusion` default,
bitwise-prior semantics; `:regionalmax` opt-in). The regional path uses a
shared `is_local_maximum` helper loaded before both `correlators.jl` and
`particles.jl`, scans the correlation plane once, and retains the strongest
`k` regional maxima in the caller's scratch vectors (no per-window
allocation). `find_peaks` and `calculate_peak_ratio` accept the same
`peak_finder` keyword, and `analyze_plane!` routes single-pair and ensemble
PIV through the selected finder, so peak-ratio validation and peak
substitution see the same semantics. Dynamic per-peak exclusion radius was
left deferred: it is a separate modeling choice, not needed for the regional
max blind spot.

**Benchmark verdict -- top-k regional scan, opt-in.** A collect+sort regional
prototype was rejected because it allocates and grows badly with plane size.
Microbenchmarks (Julia 1.11.4, `k = 3`, Gaussian-like peak on random
background) were:
16^2: exclusion 0.98 us, collect+sort regional 1.12 us, top-k regional
0.89 us; 32^2: 4.20 / 5.68 / 3.69 us; 64^2: 13.44 / 33.42 / 19.56 us;
128^2: 46.79 / 190.82 / 121.26 us. The selected implementation avoids the
allocation cliff, but it is slower than fixed exclusion on larger planes, so
the default remains `:exclusion`.

## B3. Cross-pair buffer reuse in the sequence driver — DONE (branch `feedback-b3`)

Reuse the expensive per-pair scratch across a whole `run_piv_sequence` /
`run_piv_ensemble` run via a workspace, instead of re-allocating it per pair.

**Landed:** a public `PIVWorkspace` / `piv_workspace()` reuses the padded cubic
B-spline coefficient buffers (via `image_interpolant!` + `interpolate!` on a
preallocated OffsetArray — bitwise-identical prefilter), the two deform output
buffers, and a correlator pool (keyed by `(T, method, window_size, padding,
apodization)`, one per chunk). Threaded through `run_piv → piv_pass →
process_windows!` (which now *receives* its correlator, mirroring the ensemble
path) as a `workspace` keyword; `run_piv_sequence` and `run_piv_ensemble` each
hold one workspace on their serial pair loop (the prefetch load task never
touches it, preserving the bitwise-identical-to-serial guarantee). 5-pair Case E
profile (128→16, `-t 4`): allocated 596→512 MiB, **GC 36.5%→17.9%**, lock
conflicts 99→0 (pooled FFTW plans removed the `plan_fft!` contention). Tests:
`test_io.jl` "PIVWorkspace reuse" (coefficient equality + reuse + resize +
single-pass pool) plus the existing `seq == direct` guards.

**Remaining lever (deferred):** `interpolate!` still allocates ~16 MiB/call — a
full-image temporary inside its Woodbury prefilter solve
(`Interpolations/src/filter1d.jl`, `_A_ldiv_B_md!(dest, W::Woodbury, …)` →
`_A_mul_B_md(W.U, …)` once per dimension), not reachable via any public API.
Trimming it means either reaching into WoodburyMatrices/AxisAlgorithms internals
with a preallocated `tmp3` (bit-identical but couples to private internals), or
reimplementing the cubic `Line(OnGrid)` prefilter from the math (no coupling but
risks 1-ULP drift that breaks the `==` bit-identity tests and shifts released
numerics — needs accuracy-tolerance re-validation). Left as-is: the current win
is clean and bitwise-safe; the residual is a library-internal floor.

### Original design notes (for reference)

Already landed (localized, bitwise-identical, no API change): `run_piv` now
reuses the two `deform_images` warp buffers across passes, UOD sorts its own
scratch with `median!`, and `image_interpolant` skips the redundant `T.(img)`
copy when the image is already `T`. After those, an allocation profile of a
10-pair × 4-pass (128→16) 1024² sequence dropped 1.60 GiB → 952 MiB and GC
~13% → ~6%, and the **B-spline prefilter in `image_interpolant` is now the
single dominant allocation** — genuine per-pair compute (each pair's images
differ), so the only further win is reusing the coefficient buffer *across*
pairs.

The change: thread a workspace (preallocated padded coefficient arrays reused
by `interpolate!` for the cubic prefilter, plus optionally the correlators —
whose per-window-size FFTW plans are already the ensemble's reuse pattern)
through the `_run_sequence` driver in `src/io.jl` into each `run_piv` call.
Reserved for Fable because it: (a) touches the sequence driver and the
`run_piv` signature (or needs an internal workspace object), (b) must preserve
the driver's prefetch-overlap threading model **and** the bitwise-identical
-to-serial guarantee the memory/CLAUDE.md call out (a shared workspace must
never be touched by the background load task, only by `process`), and
(c) wants a before/after allocation benchmark to confirm the `interpolate!`
in-place prefilter path matches `interpolate`'s numerics (boundary padding
semantics differ) and actually pays off against its added complexity.

---

## Handoff notes

- Fable: read this file + `feedback.md`. **B1, B2, and B3 are done** (branches
  `feedback-b1` / `feedback-b3`, plus the B2 implementation above).
- Reference for what Part A already shipped (final-pass overrides,
  correlation-plane storage, per-pair batch output, `frame_index_strings`,
  `common_dewarp_grid`, the `plot_vector_field` arrow improvements): the
  `feedback-batch-1` commit and the CLAUDE.md architecture section.
