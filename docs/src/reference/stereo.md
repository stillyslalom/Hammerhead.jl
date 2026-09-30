```@meta
CurrentModule = Hammerhead
```

# Calibration, dewarping, and stereo

Calibrate each camera, dewarp its images onto a shared world-plane grid,
then use [`run_piv_stereo`](@ref) to reconstruct three displacement components.
The result grid and `(u, v, w)` are in world length units per image pair;
attach an exposure interval to convert displacements to velocities. This
page also lists sequence and ensemble drivers, target detection, and
[`self_calibrate`](@ref) for plate-to-sheet alignment. The
[stereo tutorial](../tutorials/stereo.md) follows the steps, and
[Stereo geometry and self-calibration](../explanation/stereo.md) explains
the coordinates. GPU selection applies to the per-camera PIV calculations;
dewarping and reconstruction run on the CPU. See
[Run PIV on a GPU](../howto/gpu.md).

```@index
Pages = ["stereo.md"]
```

## Camera models and calibration

This section also includes [`PlanarTransform`](@ref) and
[`planar_calibration`](@ref), the lightweight two-point calibration for
planar 2D2C work.

```@autodocs
Modules = [Hammerhead]
Pages = ["calibration.jl"]
Private = false
```

## Calibration-target detection

```@autodocs
Modules = [Hammerhead]
Pages = ["target_detection.jl"]
Private = false
```

## Image dewarping

```@autodocs
Modules = [Hammerhead]
Pages = ["dewarp.jl"]
Private = false
```

## Stereo reconstruction

```@autodocs
Modules = [Hammerhead]
Pages = ["stereo.jl"]
Private = false
```

## Self-calibration

```@autodocs
Modules = [Hammerhead]
Pages = ["selfcal.jl"]
Private = false
```
