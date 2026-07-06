# Roadmap

Hammerhead's development is organized around the [International PIV
Challenge](https://pivchallenge.org/) cases (2001, 2003, 2005, 2014). Each case
is a published, ground-truthed benchmark, so "address case X" is a concrete,
verifiable milestone.

**Scope:** planar 2D2C and stereoscopic 2D3C PIV. Tomographic PIV (3D volume
reconstruction — MART/FastMART, 3D correlation) is **out of scope**; Challenge
cases 4C and 4D are therefore not targets. The reachable target set is every
planar and stereo case: 1A/B/C/E, 2A/B/C, 3A/B/C, 4A/B/E/F.

## Challenge cases at a glance

| Case | Edition | Content | Primary capability tested |
|------|---------|---------|---------------------------|
| 1A | 2001 | Real strong tip vortex, seeding dropout | Strong gradients, variable particle size |
| 1B | 2001 | Synthetic vortex, density × size sweep | Behavior vs. seeding density / particle diameter |
| 1C | 2001 | Real impeller, blade reflections + mask | Masking, reflection removal |
| 1E | 2001 | Synthetic shear, non-uniform illumination | Gradient accuracy, illumination correction |
| 2A | 2003 | Real turbulent jet, 100 pairs | Batch processing, turbulence statistics |
| 2B | 2003 | Synthetic DNS channel, 100 pairs | Accuracy vs. ground truth |
| 2C | 2003 | Multi-CCD "patchwork" | Non-uniform sensor handling |
| 3A | 2005 | Synthetic, up to 2000² 16-bit | Spatial resolution, bit depth / file I/O |
| 3B | 2005 | Synthetic time-resolved channel, 120 frames | Time-series handling |
| 3C | 2005 | Real high-speed jet, 10 kHz | High-speed sequences, no ground truth |
| 4A | 2014 | Micro-PIV, 600 frames, low SNR | Ensemble correlation, aggressive preprocessing |
| 4B | 2014 | Time-resolved hill flow, near-wall | Dynamic range, masking, out-of-plane |
| 4E | 2014 | Real time-resolved stereo vortex ring | Stereo PIV + calibration |
| 4F | 2014 | Real solid-body rotation, known ω | Bias-error / uncertainty quantification |
| ~~4C~~ | 2014 | Synthetic tomographic 3D vortices | *Out of scope (tomo)* |
| ~~4D~~ | 2014 | Synthetic tomographic DNS turbulence | *Out of scope (tomo)* |

## Current capabilities

The 2D planar core already addresses (or substantially addresses) the
synthetic, single-pair cases:

- Cross- and phase-correlation with FFTW-cached plans
- Multi-pass WIDIM refinement with symmetric image deformation (spatial
  resolution + strong gradients)
- Zero-padded, overlap-normalized correlation with Gaussian apodization
  (~0.03 px RMS on synthetic data)
- Validation: universal outlier detection, peak-ratio, correlation-moment,
  velocity-magnitude; local-median replacement
- Preprocessing: ensemble background subtraction, intensity capping, high-pass
  filtering, CLAHE (covers illumination/SNR aspects of 1A/1E/4A/4B)
- Synthetic image generation with ground truth; threaded execution; Makie
  plotting
- Image-file loading (FileIO/ImageIO), JLD2 result serialization, and a batch
  sequence driver with incremental output (Phase 1, July 2026)

This makes **1B, 1E, 2B, 3A (core), 3B (core)** largely reachable today, plus
file-based single-pair runs on the real cases (1A, 3C frames).

## Phases

Each phase ends at a coherent "can now attempt cases X" milestone. Phases 1–4
extend the existing in-memory matrix engine; Phase 5 adds a second camera.
Phases 6–7 grow the adoption surface (documentation, GUI) now that the
Challenge target set is reachable.

**Ecosystem policy:** lean on the JuliaImages ecosystem instead of from-scratch
implementations where a maintained package exists — FileIO/ImageIO for image
loading, ImageFiltering for convolution/blur, JLD2 for result serialization.
Hand-rolled versions are reserved for PIV-specific algorithms (correlation,
validation, deformation) and for cases where the ecosystem version compromises
subpixel fidelity (ImageContrastAdjustment's CLAHE silently `imresize`s images
whose dimensions don't divide evenly into blocks, so Hammerhead keeps its own
exact-tiling CLAHE).

### Phase 1 — I/O & batch foundation ✅ (July 2026)
Every challenge dataset ships as image files; nothing real can run without this.

- [x] Image-file loading via FileIO/ImageIO (TIFF incl. 16-bit, PNG; BMP through
  the ImageMagick loader): `load_image` → grayscale `Matrix{Float64}`
- [x] Result serialization (JLD2): `save_results` / `load_results`
- [x] Batch / sequence driver: `run_piv_sequence` over `image_pairs` lists with
  a progress meter, per-frame `preprocess` hook, and incremental JLD2 output
  (a crashed batch keeps its finished pairs)
- [x] Retired the hand-rolled Gaussian blur in favor of ImageFiltering

*Milestone:* end-to-end runs on 2A, 2B, 3C, 1A as files.

### Phase 2 — Masking ✅ (July 2026)
Already the flagged next priority; touches every pipeline stage (correlation,
validation, replacement).

- [x] Static binary masks: `run_piv(...; mask, mask_threshold)` with grid-level
  `result.mask`; partially masked windows correlate over valid pixels only
  (masked pixels enter at the valid mean — no intensity step); masked windows
  are excluded from UOD neighborhoods, replacement medians, and the multi-pass
  predictor (filled from valid neighbors before deformation)
- [x] Reflection / geometry exclusion regions: `polygon_mask`, `load_mask`
  (mask image files, e.g. the supplied 1C mask)

*Milestone:* 1C (impeller), near-wall regions of 4B.

### Phase 3 — Ensemble & statistics ✅ (July 2026)
Turbulence and micro-PIV cases.

- [x] Ensemble / sum-of-correlation across a sequence (`run_piv_ensemble`,
  multi-pass with shared deformation predictor; low-SNR micro-PIV)
- [x] Time-series statistics: `field_statistics` (mean, RMS, Reynolds shear
  stress, valid-sample counts) and a `power_spectrum` utility for spectra
- [x] Temporal validation across long sequences (`validate_temporal!`,
  per-point median/MAD test over time)

*Milestone:* 2A, 3B, 3C, 4A.

### Phase 4 — Accuracy, resolution & UQ ✅ (July 2026)
The "how good" cases.

- [x] Multi-peak detection (`find_peaks`, one plane scan per peak; top two are
  free since the peak ratio needs both) + secondary/tertiary peak
  substitution during validation (`n_peaks` parameter, on by default;
  measured ~6% pass-time cost at `n_peaks = 3`)
- [x] Peak-locking diagnostics (`peak_locking`: fractional-part histogram +
  locking index)
- [x] Higher-order subpixel: `:gauss9` closed-form 2D Gaussian regression
  (Nobach & Honkanen — exact for rotated elliptical peaks, less locking);
  resolution sweeps are an analysis task on top of `bench/`
- [x] Bias-error tooling (4F): `error_statistics` against reference arrays or
  `(x, y) -> value` functions (error maps + bias/RMS over valid vectors)
- [x] Garcia `smoothn` (DCT-based penalized least squares, GCV-selected `s`,
  weights/missing data, robust bisquare option)
- [x] Calibrated per-vector uncertainty (Wieneke 2015 correlation statistics,
  implemented from the paper — `reference/Wieneke_2015_*.pdf`): opt-in
  `uncertainty` parameter fills `uncertainty_u`/`uncertainty_v` on the final
  pass, from the deformed windows' correlation-difference statistics;
  pooled across pairs in `run_piv_ensemble`. Validated against measured RMS
  error on synthetic noise sweeps (median σ within ~±25% through 20% noise)

*Milestone:* 3A, 4F, plus quality uplift across all cases.

### Phase 5 — Stereo PIV (2D3C) — final phase ✅ (July 2026)
The largest lift; reuses the 2D correlation engine per camera.

- [x] Camera calibration models + fitting (July 2026): `PinholeCamera`
  (normalized DLT) and `SoloffCamera` (19-term cubic-in-XY, quadratic-in-Z
  polynomial), `calibrate_camera` on (pixel, world) point pairs, forward and
  inverse mappings (`world_to_pixel` / `pixel_to_world`, Newton inversion for
  Soloff), `calibration_quality` / `reprojection_errors` fit metrics
- [x] Target-based calibration (July 2026): `detect_calibration_grid` for
  rectilinear dot grids (LaVision-style, single- and two-level plates) —
  subpixel intensity-weighted centroids, homography-refined lattice indexing,
  square/triangle fiducial classification, marker-anchored origin
  (`origin_offset`, e.g. `(30.0, 7.5)` for case 4E) — plus
  `render_calibration_target`, the ground-truth synthetic fixture. Validated
  on the real 4E calibration images: both cameras, all 7 planes, full grid +
  markers on every image; residuals (~0.6–1 px RMS) are dominated by the
  plate's per-dot position tolerance (repeatable across planes), while
  plane-to-plane detection repeatability is 0.15–0.3 px
- [x] Image dewarping / back-projection to a common measurement plane
  (July 2026): `DewarpGrid` (world extent + resolution + plane `z`, shared by
  all cameras and by the later stereo vector grid) and `ImageDewarper`
  (per-camera map, built once via `world_to_pixel` per grid node and reused
  across a whole sequence) with in-place `dewarp!` (cubic B-spline
  resampling, output precision follows the image). Out-of-view nodes are
  zero-filled and reported in a static lab-frame validity mask
  (`true` = excluded) that plugs straight into `run_piv(...; mask)`
- [x] Three-component (u, v, w) reconstruction from two views (July 2026):
  `run_piv_stereo` dewarps each camera's frame pair onto the shared grid
  (reusing buffers), runs the 2D engine per camera with identical parameters
  and the combined validity mask, and solves each vector's four-equation
  geometric least squares — `uᵢ = dx − dz·tXᵢ`, `vᵢ = dy − dz·tYᵢ` with the
  viewing-ray slopes evaluated from the calibration — for `(u, v, w)` in
  world units. Per-camera Wieneke (2015) uncertainties propagate through the
  same least-squares operator. Results land in the new `StereoPIVResult`
  (world-coordinate vector grid, union outlier/mask flags, both per-camera
  `PIVResult`s retained), handled by `save_results`/`load_results`
  (format v3)
- [x] Disparity / self-calibration correction (July 2026; Wieneke 2005,
  implemented from the paper — `reference/Wieneke_2005_*.pdf`):
  `self_calibrate` cross-correlates the two cameras' dewarped images of the
  same instants (ensemble sum-of-correlation over 1–n instants, single
  large-window pass), triangulates the disparity vectors to world points on
  the true light-sheet plane, fits a plane through them, and rigidly
  transforms both camera models so that plane becomes the measurement plane —
  iterated to convergence (residual disparity < 0.01 px on the synthetic
  fixtures, vs the ~0.1 px the paper reports for real setups). A corrected
  `PinholeCamera` absorbs the transform into its projection matrix; other
  models come back wrapped in the new `TransformedCamera`. Returns
  replacement `ImageDewarper`s (drop-in for `run_piv_stereo`) plus a
  `SelfCalibrationReport` — per-pass disparity/triangulation statistics,
  fitted plane parameters, the cumulative world transform, and (opt-in) the
  disparity vector maps

*Milestone:* 4E (time-resolved stereo vortex ring). The case data (particle
and calibration images) lives in `cases/4th_PIV-Challenge_Case_E/`, which is
gitignored; the instructions PDF is in `reference/`.

### Phase 6 — Documentation (planned)

Structure follows [Diátaxis](https://diataxis.fr/) (tutorials, how-to guides,
reference, explanation). Docs come **before** the GUI: writing the tutorials
is an API audit (warts get fixed before a GUI fossilizes them), and the
how-to inventory defines the GUI's task model.

- [ ] Restructure `docs/src` into the four Diátaxis categories (landing page
  keeps the quick example)
- [ ] Explanation pages ported from CLAUDE.md's conventions section
  (user-facing rewrite; CLAUDE.md keeps the maintainer-only gotchas):
  coordinate & sign conventions, correlation accuracy (padding, apodization,
  bias), multipass & image deformation, the masking model, uncertainty
  methodology (Wieneke 2015), stereo geometry & self-calibration
  (Wieneke 2005), the precision-follows-images policy
- [ ] Split the oversized `function_ref.md` by topic (core pipeline &
  parameters; preprocessing; validation & quality; calibration / dewarping /
  stereo / self-calibration; I/O & batch; ensemble & statistics) — also
  fixes the Documenter size warning
- [ ] Two Literate.jl tutorials, fully synthetic (`cases/` is gitignored),
  executed by the docs build so they double as integration tests:
  (a) *first vector field* — synthetic pair, `run_piv`, multipass,
  outliers, uncertainty; (b) *stereo end-to-end* —
  `render_calibration_target` → `calibrate_camera` → dewarp →
  `self_calibrate` → `run_piv_stereo`
- [ ] How-to guides seeded from workflows the test suite already exercises:
  masking reflections/geometry, preprocessing chains, ensemble correlation
  for low SNR, validation tuning, batch processing + JLD2 round-trips,
  calibrating a real stereo rig (target requirements, `origin_offset`,
  self-calibration on real recordings)
- [ ] Docs infrastructure: Literate.jl + DocumenterCitations (the
  implemented-from-paper methods deserve citations; papers in `reference/`),
  CairoMakie-generated figures at build time
- [ ] API-wart audit driven by the tutorial writing; breaking fixes land now,
  while nothing depends on the API
- [ ] Tag and register Hammerhead 0.1.0

*Milestone:* a newcomer goes from `] add Hammerhead` to a validated stereo
field without reading source; the API is frozen enough to build a GUI on.

### Phase 7 — GUI: HammerheadGUI.jl (planned)

**Structure (settled July 2026):** monorepo subdirectory package,
Makie-style — Hammerhead stays at the repo root, the GUI lives in
`HammerheadGUI/` with its own Project.toml, registered with
`subdir=` and TagBot monorepo tags (copy Makie's CI/Registrator/TagBot
wiring rather than deriving it). Dev-time coupling to the core via
`[sources]` (Julia ≥ 1.11) or `Pkg.develop` in CI; releases go core-first,
then GUI compat bump. The GUI package is where the GLMakie hard dependency
lives — the core package never gains GUI deps.

**Framework (settled July 2026):** pure GLMakie + NativeFileDialog for v1.
Rationale: the PIV-specific hard parts (image pan/zoom, polygon mask
editing, frame scrubbing, vector picking, stereo/disparity views) are canvas
work where GLMakie is strong; the widget chrome Makie lacks (tables,
docking, rich forms) is the cheap part of this app; and the third-party
bridges to richer chrome (Gtk4Makie.jl, QMLMakie.jl) are the
least-maintained joint in any alternative stack — they sit on churning
internals of two frameworks at once. All application state/logic lives in a
framework-free controller layer (plain Julia + Observables) so the widget
shell stays swappable (Gtk4-embedded GLMakie or Bonito/WGLMakie re-skin
later, without rewriting canvas interactions) and testable without a GL
context. The mask editor is the framework proving ground; if pure Makie
handles it plus one parameter form comfortably, the framework question is
closed.

- [ ] Monorepo conversion: `HammerheadGUI/` skeleton, CI matrix
  (`Pkg.develop` of the core before GUI tests, xvfb for GLMakie on Linux
  runners), Registrator subdir + TagBot wiring
- [ ] Result explorer — read-only viewer for `PIVResult`/`StereoPIVResult`
  (field overlays, vector inspection, outlier/uncertainty display); binds
  only to the most stable API surface
- [ ] Mask editor — polygon drawing/editing over the image, exporting the
  package mask convention; biggest usability win over the CLI and the
  framework proving ground
- [ ] Parameter form + batch runner — `PIVParameters` editing, sequence
  processing with progress and incremental JLD2 output
- [ ] Calibration & self-calibration diagnostics — grid-detection review,
  reprojection errors, disparity-map viewer for `SelfCalibrationReport`
- [ ] Packaging pass — PrecompileTools workload; evaluate a PackageCompiler
  app bundle for non-Julia lab users

*Milestone:* a lab user can mask, process, and inspect a recording without
writing Julia.

## Coverage trajectory

| Stage | Cases reachable |
|-------|-----------------|
| Today | ~5 (synthetic-leaning) |
| After Phase 1–2 | ~8 |
| After Phase 3–4 | ~12 |
| After Phase 5 | all planar + stereo (1A/B/C/E, 2A/B/C, 3A/B/C, 4A/B/E/F) |

The only unaddressed cases are the two synthetic tomographic ones (4C, 4D),
which fall outside the planar + stereo scope by design.
