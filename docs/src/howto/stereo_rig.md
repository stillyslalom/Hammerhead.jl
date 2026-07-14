# Calibrate a real stereo rig

**Goal:** go from calibration-plate photographs to corrected camera models
ready for [`run_piv_stereo`](@ref). This guide describes the workflow on
real data, using case E of the 4th International Particle Image Velocimetry
(PIV) Challenge
[Kahler2016](@cite) — a time-resolved stereo vortex ring recorded with a
LaVision two-level dot-grid plate — as the running example. For
executable end-to-end walkthroughs, see the
[stereo tutorial](../tutorials/stereo.md) (synthetic, with ground truth)
and [stereo on a real recording](../tutorials/stereo_real.md) (the 4E
data itself); for the theory, see
[stereo geometry and self-calibration](../explanation/stereo.md).

## What the target must provide

[`detect_calibration_grid`](@ref) handles rectilinear dot grids —
LaVision-style plates, single- or two-level — and needs:

- **Known dot spacing** in world units (the 4E plate: 15 mm per level).
- **Enough Z information.** A pinhole fit needs the plate imaged at ≥ 2 Z
  positions; the (default) Soloff model needs ≥ 3. A *two-level* plate
  contributes two planes per view, so a single photograph of a two-level
  plate suffices for a pinhole fit, and two positions for Soloff — but
  traversing the plate through more positions (4E used seven, −3 mm to
  +3 mm in 1 mm steps) improves conditioning and lets you check
  repeatability.
- **A fiducial marker for multi-camera consistency.** Both cameras must
  agree on which dot is the world origin. That requires the plate's filled
  square marker plus the `origin_offset` that locates the origin dot
  relative to it — for the 4E plate, `(30.0, 7.5)`: the origin dot sits
  30 mm right of and 7.5 mm above the square. Without a marker, each
  detection anchors to the dot nearest the image center, which is *not*
  consistent between cameras.
- **Roughly upright cameras.** World axes are anchored to the image
  orientation (+X nearest image-right, +Y nearest image-up). Standard
  stereo rigs qualify; heavily rolled cameras do not (currently
  unsupported).

## Detect and calibrate

Per camera, detect every plate position and fit one model:

```julia
using Hammerhead

zs = -3.0:1.0:3.0                      # traverse positions, mm
grids = [detect_calibration_grid(load_image(path);
             spacing = 15.0,
             two_level = true, level_separation = 3.0,
             origin_offset = (30.0, 7.5))
         for path in cam1_plate_images]
cam1 = calibrate_camera(grids, collect(zs))          # Soloff (default)
```

Use `invert = true` for dark dots on a bright plate. Check each detection
before fitting: `length(grid.pixels)` should cover the full grid,
`grid.square !== nothing` confirms the marker was found, and on a two-level
plate roughly half the dots should report `level == 1`.

## What residuals to expect

Judge the fit with [`calibration_quality`](@ref) — but know what you're
looking at. On the real 4E plates:

- Per-dot reprojection residuals of **~0.5–1 px root-mean-square (RMS)** that are
  *repeatable across plate positions* reflect the plate's dot-position
  manufacturing tolerance, not detection error. Don't chase them with
  model changes; they are a property of the target.
- Plane-to-plane detection *repeatability* is **0.15–0.3 px** — that is
  the level of the actual measurement noise.

A pinhole fit that trails the Soloff fit badly indicates lens distortion or
refraction along the optical path — expected through windows or liquid;
stay with Soloff.

## Dewarp and self-calibrate on the recordings

Build the shared grid and per-camera dewarpers, then correct the
plate-to-sheet misregistration with [`self_calibrate`](@ref) **on the
actual particle recordings** — the plate never sits exactly in the light
sheet:

```julia
grid = DewarpGrid(x = -40.0:0.1:40.0, y = -30.0:0.1:30.0)   # mm, finer than the vector spacing
dw1 = ImageDewarper(cam1, grid, size_of_camera1_images)
dw2 = ImageDewarper(cam2, grid, size_of_camera2_images)

# 5–50 same-instant frame pairs give a well-shaped disparity correlation.
dw1c, dw2c, report = self_calibrate(frames1, frames2, dw1, dw2)
```

Inspect the report before trusting it:

- Don't judge by `report.converged` alone: on real data the residual
  disparity RMS floors at the sheet-thickness decorrelation level (0.1 px
  on thin-sheet setups [Wieneke2005](@cite); ~0.5 px on the 4E
  recordings), which can sit above the default `tol`. Misalignment is
  *systematic*, so the signed median disparity components — computed from
  the maps with `keep_disparity_maps = true` — are the sharper test; the
  [real-recording tutorial](../tutorials/stereo_real.md) walks through
  this judgment.
- The first pass's `plane` — its offset and tilt tell you how far the
  plate actually was from the sheet; millimeters are normal.
- A large `triangulation_rms` with a small final disparity suggests
  calibration errors rather than sheet misalignment — revisit the plate
  detections.
- Pass `keep_disparity_maps = true` to inspect the raw disparity fields
  when convergence is poor (check the stereo overlap region, seeding
  density, and window size).

The corrected dewarpers then drop into stereo processing of the recording:

```julia
stereo = run_piv_stereo(A1, B1, A2, B2, dw1c, dw2c, passes)
```

For large per-camera analyses, `backend = :amdgpu` or `:cuda` forwards GPU
execution to both two-component (2C) PIV calls. Dewarping the four raw images
and reconstructing the final three-component (3C) field remain CPU operations,
so stereo does not yet form a fully
device-resident pipeline. See [Run PIV on a GPU](gpu.md) for setup and the
supported PIV option matrix.
