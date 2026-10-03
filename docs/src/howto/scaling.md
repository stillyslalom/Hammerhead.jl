# Scale results to physical units

**Goal:** convert measured displacements to physical velocities while
keeping the original measurements and diagnostics available.

Planar PIV and PTV measure displacement in pixels per frame interval and
keep their stored arrays in those measured units. Stereo results use the
dewarp grid's world units, as described below. Physical calibration is *metadata*: a
[`PhysicalScale`](@ref) records the pixel size, the frame interval `dt`, and
the unit names, and [`physical`](@ref) applies it on demand.

## Establish the calibration and pair delay

Measure a known length in the measurement plane with the same camera setup
used for the particle images. For a uniform image scale, divide that length
by its separation in pixels to obtain `pixel_size`. A single scalar scale
does not correct lens distortion or perspective variation across the image;
check calibration features across the field before using one value everywhere.

Set `dt` to the delay between the two exposures or illumination pulses in
each pair. For double-frame recordings this may differ from the camera's
frame period. The interval between successive velocity fields belongs in
time-series calculations; see the
[sequence tutorial](../tutorials/sequence_statistics.md).

For example, 3 px of displacement at 0.02 mm/px over 1 ms gives 60 mm/s.
Check this calculation for a representative vector after conversion. The
[image-quality guide](image_quality.md) explains how displacement, exposure,
and particle loss affect the choice of delay.

## Attach a scale

Pass `scale` to any driver — [`run_piv`](@ref), [`run_piv_sequence`](@ref),
[`run_piv_ensemble`](@ref), [`run_piv_stereo`](@ref), [`run_ptv`](@ref),
[`run_ptv_sequence`](@ref), or [`track_particles`](@ref):

```julia
scale = PhysicalScale(pixel_size = 0.02,   # 0.02 mm per pixel
                      dt = 1e-3,           # 1 ms between frames
                      length_unit = "mm", time_unit = "s")
result = run_piv(imgA, imgB; effort = :medium, scale)
```

The numbers carry the conversion and the unit names are display-only labels,
so results come out in whatever units go in — millimeters and seconds here,
so velocities will be mm/s. `result` itself is unchanged by the attachment:
`result.u` is still pixels, bitwise identical to a run without `scale`.

For a result you already have (e.g. loaded from a file saved without one),
attach after the fact:

```julia
result = with_scale(load_results("run.jld2")[1], scale)
```

## Convert with `physical`

```julia
p = physical(result)
p.u          # velocities, mm/s
p.x, p.y     # positions, mm
```

[`physical`](@ref) returns the same result type with positions multiplied by
`pixel_size` and displacements plus their uncertainties by `pixel_size / dt`.
The converted result carries an *identity* scale with the same unit labels,
so `physical(p) === p` — you cannot double-convert.

Convert **last**. Validation thresholds, [`peak_locking`](@ref), the 0.1 px
noise floor used by universal outlier detection (UOD), and the correlation
diagnostics all speak pixels, and
`physical` deliberately leaves `peak_ratio`/`correlation_moment` untouched.
Run the full pixel-side analysis first and convert at the end, for plotting
and physics.

## Construct from Unitful quantities

Load [Unitful](https://github.com/PainterQubits/Unitful.jl) to activate a
quantity-based constructor (a package extension):

```julia
using Unitful
scale = PhysicalScale(20.0u"µm", 0.5u"ms")   # pixel size, dt
```

The values are stripped in their own units and the unit names become the
labels — this scale produces µm and µm/ms. Convert with `uconvert` *before*
construction if you want different output units.

## Plotting

[`plot_vector_field`](@ref) plots a scaled result in physical units
automatically, labeling the axes with the length unit. To overlay arrows on
the source image in pixel coordinates instead, strip the scale:

```julia
plot_vector_field(result)                      # physical axes (mm)
plot_vector_field!(ax, with_scale(result, nothing))  # pixel axes, e.g. over the image
```

## Stereo: `dt` only

A [`StereoPIVResult`](@ref) is already spatially calibrated — its fields are
in the dewarp grid's world units (typically mm). Attach a scale with `dt`
and labels only, leaving `pixel_size = 1`:

```julia
stereo = run_piv_stereo(A1, B1, A2, B2, dw1, dw2;
                        scale = PhysicalScale(dt = 1e-3,
                                              length_unit = "mm", time_unit = "s"))
physical(stereo).w    # out-of-plane velocity, mm/s
```

The embedded per-camera results (`stereo.cam1`, `stereo.cam2`) always stay
in dewarped pixels — they are diagnostics.

## Particle tracking velocimetry (PTV) and trajectories

[`run_ptv`](@ref) results convert like particle image velocimetry (PIV) results
(the `match_residual` is the distance between a particle's predicted and
observed frame-B positions in pixels, so it scales with `pixel_size`), and
[`ptv_to_grid`](@ref) carries the scale onto the binned grid — bin the raw
result, then convert.

A [`TrackingResult`](@ref) stores only positions; velocities are *derived*
by differencing, so pass the scale to [`trajectory_velocities`](@ref):

```julia
tracks = track_particles(frames; scale)
u, v = trajectory_velocities(tracks.trajectories[1], tracks.scale)   # mm/s
```

This works identically on a raw or a `physical`-converted tracking result:
the converted result's scale keeps `dt` (its positions are already lengths).

## Export in a calibrated planar coordinate frame

A scalar pixel size preserves the image's axis directions and origin.
Use [`PlanarTransform`](@ref) to export a raw planar PIV grid with an origin,
rotation, reflection, or different length scales along the two axes. Both
[`export_table`](@ref) and [`export_vtk`](@ref) apply the same affine map to
coordinates and its linear part to vector components. Supply the length unit
and, for velocities, the exposure delay and its time unit explicitly.

This example builds a small raw grid and calibrates a physical x axis along
the line between two image points, with a reflected perpendicular axis:

```@example transformed_export
using Hammerhead

raw = PIVResult([5.0, 10.0], [5.0, 10.0],
    fill(2.0, 2, 2), fill(1.0, 2, 2), fill(2.0, 2, 2), fill(0.2, 2, 2),
    fill(0.1, 2, 2), fill(0.2, 2, 2), falses(2, 2), falses(2, 2),
    PIVParameters())
calibration = planar_calibration((1.0, 1.0), (11.0, 11.0), 2.0;
    origin=(0.0, 0.0), reflection=true, perpendicular_scale=0.1)

mktempdir() do directory
    options = (transform=calibration, length_unit="mm", dt=0.005,
               time_unit="s", uncertainty_assumption=:independent)
    csv = export_table(joinpath(directory, "calibrated.csv"), raw; options...)
    export_vtk(joinpath(directory, "calibrated.vtk"), raw; options...)
    readlines(csv)[1:2]
end
```

The position uses `A * [x, y] + b`, while the velocity uses
`A * [u, v] / dt`: translation changes the physical origin without changing
vectors. Omitting `dt` exports calibrated displacements per frame interval.
Units are labels, so the transform's numeric factors and the delay must agree
with the supplied labels. No resampling is performed; masks and outlier flags
refer to the same original grid nodes.

The example explicitly assumes independent errors in the original `u` and
`v` components. Under that assumption the exported marginal standard deviation
for row `k` of `A` is `hypot(A[k,1] * σu, A[k,2] * σv) / dt`. Output axes can
still have correlated errors. The default `uncertainty_assumption = :unknown`
instead exports `NaN` for uncertainty when a component mixes both input axes,
because the result does not retain their covariance. A pure axis permutation,
reflection, or diagonal scale needs no covariance assumption. Keep the transform and the chosen assumption with your analysis
notes; the table and VTK file do not store them.

Transform export accepts planar PIV grids with no attached `PhysicalScale`.
For a **raw, unconverted** result that already carries scale metadata, reuse
only its delay and remove its metadata before export:

```julia
delay = result.scale.dt
time_unit = result.scale.time_unit
export_table("calibrated.csv", with_scale(result, nothing);
             transform=calibration, length_unit="mm", dt=delay, time_unit)
```

The affine calibration supplies the spatial scale. Removing metadata does not
undo `physical(result)`, so start from the raw pixel result. Already converted
results and attached-scale combinations are rejected. The `transform` keyword applies
to planar PIV grids only.

To combine a planar PIV field with an image from another camera, use the
[calibrated PIV/PLIF resampling workflow](calibrated_resampling.md). It fits
both cameras to common physical coordinates and samples onto explicit target
axes, retaining separate contributor and availability diagnostics.
