```@meta
CurrentModule = Hammerhead
```

# Calibration, dewarping, and stereo

Everything stereoscopic (2D3C): camera calibration models and fitting,
dot-grid target detection, image dewarping onto a common world plane,
three-component reconstruction, and disparity self-calibration. The
[stereo tutorial](../tutorials/stereo.md) walks the whole chain;
[Stereo geometry and self-calibration](../explanation/stereo.md) explains
the method. [`run_piv_stereo`](@ref) can forward a GPU backend to its two
per-camera PIV calls; see [Run PIV on a GPU](../howto/gpu.md) for the CPU/GPU
boundary.

```@index
Pages = ["stereo.md"]
```

## Camera models and calibration

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
