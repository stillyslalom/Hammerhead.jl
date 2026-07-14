# Scale results to physical units

**Goal:** turn pixel displacements into physical velocities — without giving
up the pixel-native diagnostics.

Hammerhead measures in pixels per frame interval and keeps every stored
array in those measured units. Physical calibration is *metadata*: a
[`PhysicalScale`](@ref) records the pixel size, the frame interval `dt`, and
the unit names, and [`physical`](@ref) applies it on demand.

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
(the `match_residual` is a
frame-A distance, so it scales with `pixel_size`), and
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
