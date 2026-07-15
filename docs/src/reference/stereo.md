```@meta
CurrentModule = Hammerhead
```

# Calibration, dewarping, and stereo

Everything stereoscopic: two-dimensional, three-component (2D3C) particle
image velocimetry (PIV), including camera calibration models and fitting,
dot-grid target detection, image dewarping onto a common world plane,
three-component reconstruction, synchronized sequence processing, stereo
ensemble correlation, and disparity self-calibration. The
[stereo tutorial](../tutorials/stereo.md) walks the whole chain;
[Stereo geometry and self-calibration](../explanation/stereo.md) explains
the method. [`run_piv_stereo`](@ref), [`run_piv_stereo_sequence`](@ref), and
[`run_piv_stereo_ensemble`](@ref) can forward a GPU backend to their two
per-camera PIV calls; see [Run PIV on a GPU](../howto/gpu.md) for the CPU/GPU
boundary.

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
