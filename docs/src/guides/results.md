# Turn vectors into a flow measurement

A vector field becomes a flow measurement once its units are established and
the locations with usable measurements are known. The analysis of mean
speed, fluctuations, rotation or particle paths follows from there.

## Put the axes and arrows in physical units

For a calibration of 0.02 mm per pixel and exposures 1 ms apart, a raw
planar `result` from `run_piv` is converted as follows:

```julia
using Hammerhead
scale = PhysicalScale(pixel_size=0.02, dt=0.001,
                      length_unit="mm", time_unit="s")
velocity = physical(with_scale(result, scale))
```

Positions are now in millimeters and velocities in millimeters per second.
A displacement of 3 px becomes 60 mm/s. Use the delay **between the two
exposures**, which may differ from the interval between successive pairs.
The [units guide](../howto/scaling.md) covers the other result types.

## Analyses

| Quantity | Example or guide |
|:--|:--|
| Mean flow and fluctuation statistics | [From image pairs to flow statistics](../tutorials/sequence_statistics.md) |
| Rotation and strain | [Gradients, vorticity and circulation](../reference/derived.md) |
| Periodic motion | [`result_spectrum`](@ref) with the interval between successive fields |
| Individual particle velocities and paths | [Particle tracking](../tutorials/ptv.md) |
| Share of usable vectors | [Validation flags and their counts](../howto/validation.md) |

A suspicious region can be checked by repeating the analysis with and
without it. Near a mask or a gap, a velocity may be available while its
derivative is not; plots of vorticity should show those gaps rather than
treat missing values as zeros.
The [measurement concepts](../explanation/index.md) explain these distinctions.

## Share the result

Use [native files, CSV and VTK](../howto/batch.md) to save fields, particles and
trajectories or move them to another analysis tool.
If velocity and an intensity measurement come from different cameras,
[register PIV and PLIF on a common grid](../howto/calibrated_resampling.md)
before comparing them point by point.
