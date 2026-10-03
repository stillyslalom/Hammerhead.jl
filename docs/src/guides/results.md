# Turn vectors into a flow measurement

A vector plot is the beginning of the analysis. First establish its units and
which locations have usable measurements, then ask about the flow feature:
mean speed, fluctuations, rotation, or particle paths.

## Put the axes and arrows in physical units

Suppose one pixel represents 0.02 mm and the exposures are 1 ms apart.
For a raw planar `result` from `run_piv`, attach that calibration and convert:

```julia
using Hammerhead
scale = PhysicalScale(pixel_size=0.02, dt=0.001,
                      length_unit="mm", time_unit="s")
velocity = physical(with_scale(result, scale))
```

Positions are now in millimetres and velocities in millimetres per second.
A displacement of 3 px becomes 60 mm/s. Use the delay **between the two
exposures**, which may differ from the interval between successive pairs.
The [units guide](../howto/scaling.md) covers the other result types.

## Choose the analysis

| Question | Example or guide |
|:--|:--|
| What is the mean flow, and how much does it fluctuate? | [From image pairs to flow statistics](../tutorials/sequence_statistics.md) |
| Where does the flow rotate or stretch? | [Gradients, vorticity and circulation](../reference/derived.md) |
| Is there a periodic motion? | [`result_spectrum`](@ref) with the interval between successive fields |
| How fast do individual particles move? | [Particle tracking](../tutorials/ptv.md) |
| How much of the recording produced usable vectors? | [Validation flags and their counts](../howto/validation.md) |

Try the same analysis with and without a suspicious region. Near a mask or a
gap, a velocity may be available while its derivative is not. Keep those gaps
visible when plotting vorticity instead of treating missing values as zeros.
The [measurement concepts](../explanation/index.md) explain these distinctions.

## Share the result

Use [native files, CSV and VTK](../howto/batch.md) to save fields, particles and
trajectories or move them to another analysis tool.
If velocity and an intensity measurement come from different cameras,
[register PIV and PLIF on a common grid](../howto/calibrated_resampling.md)
before comparing them point by point.
