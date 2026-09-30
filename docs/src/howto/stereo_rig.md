# Calibrate a real stereo rig

**Goal:** detect a stereo calibration plate, check each camera fit, and
correct the camera models to the particle light sheet before
[`run_piv_stereo`](@ref). The example uses case E of the 4th International
Particle Image Velocimetry (PIV) Challenge [Kahler2016](@cite), a stereo
vortex-ring recording with a two-level dot-grid plate. For
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
- **An orientation convention shared by both cameras.** The default
  `orientation = :image` chooses +X nearest image-right and +Y nearest
  image-up, suitable for roughly upright views. For rolled cameras, use
  `orientation = :fiducials` when both the square and triangle markers are
  visible; the markers then determine the world axes in each view.

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
before fitting: compare the detected dots with the image, confirm the
fiducial markers and lattice indices, and check that both plate levels have
plausible counts. `grid.square !== nothing` only confirms a marker candidate
was found; inspect its position before trusting the shared world origin.

## What residuals to expect

Check the fit with [`calibration_quality`](@ref). On the 4E plates, the
three-plane subset gives roughly 0.7 px RMS for camera 1 and 1.1 px for
camera 3. These are observations for this subset, not acceptance limits for
another rig. Plot per-dot residuals against image position and compare the
same indexed dots across plate positions, as in
[the real-recording tutorial](../tutorials/stereo_real.md).

A pattern tied to image position may point to camera-model limits or
refraction; a pattern tied to particular plate dots may point to target
geometry or dot detection. Neither pattern alone identifies the cause.
Recheck the marked origin and plate-level assignments before changing camera
models.

If a pinhole fit has much larger residuals than a Soloff fit, inspect where
the improvement occurs. Curved position-dependent errors can motivate the
Soloff model, especially with lens distortion or refractive windows; a fit
improvement alone does not prove which effect caused it.

## Dewarp and self-calibrate on the recordings

Build a shared grid over the cameras' overlap and one dewarper per camera.
[`common_dewarp_grid`](@ref) chooses the overlap and a default spacing from
the camera calibrations. Inspect the resulting extent and out-of-view masks
before correlation; reduce the grid area if it includes regions without
usable particle images. Then run [`self_calibrate`](@ref) on **same-instant
particle frames** from both cameras, because the plate and light sheet may
occupy different planes:

```julia
grid = common_dewarp_grid((cam1, cam2),
    (size_of_camera1_images, size_of_camera2_images), 0.0)
dw1 = ImageDewarper(cam1, grid, size_of_camera1_images)
dw2 = ImageDewarper(cam2, grid, size_of_camera2_images)
count(dw1.mask .| dw2.mask)   # grid nodes outside at least one camera's view

dw1c, dw2c, report = self_calibrate(frames1, frames2, dw1, dw2;
                                    keep_disparity_maps = true)
```

Use several instants when available; additional pairs can improve a weak
disparity peak, but cannot repair an incorrect calibration or poor camera
overlap. `frames1[i]` and `frames2[i]` must show the same instant.

Inspect the report before trusting it:

- Check `report.converged` **and** the initial and final disparity maps.
  The scalar RMS can remain above `tol` when the two camera images
  decorrelate. A small signed median may also hide positive and negative
  regions that cancel. Look for broad spatial patterns in both displacement
  components; the [real-recording tutorial](../tutorials/stereo_real.md)
  shows the maps for the 4E subset.
- If the first pass made a correction, check its `plane` for the estimated
  offset and tilt between the calibration plate and light sheet. It is
  `nothing` when the first measurement already meets `tol`.
- A large `triangulation_rms` means the measured disparity is inconsistent
  with the camera geometry. Check the plate detections and disparity maps
  for calibration or correlation errors.
- If the maps contain isolated large vectors or have little valid overlap,
  check the seeding, camera coverage, and disparity-window size before
  accepting the plane fit.

The corrected dewarpers then drop into stereo processing of the recording:

```julia
stereo = run_piv_stereo(A1, B1, A2, B2, dw1c, dw2c, passes)
```

For large per-camera analyses, `backend = :amdgpu` or `:cuda` forwards GPU
execution to both two-component (2C) PIV calls. Dewarping the four raw images
and reconstructing the final three-component (3C) field remain CPU operations.
See [Run PIV on a GPU](gpu.md) for setup and the
supported PIV option matrix.
