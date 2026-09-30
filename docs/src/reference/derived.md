```@meta
CurrentModule = Hammerhead
```

# Derived flow analysis

Use the functions below for profiles, regions, circulation, spatial
derivatives, and time spectra. Derivatives use immediate valid neighbors;
they do not cross a mask or excluded outlier. The 2D swirling-strength and
Q values describe the measured in-plane gradient tensor and do not include
unmeasured 3D terms.

Area-form circulation integrates vorticity over the portion of each grid cell
inside a requested rectangle or polygon, including fractional boundary cells.
Cells with a masked, excluded outlier, or non-finite vorticity corner are
omitted; `include_invalid=true` admits outliers but still excludes masks.

For [`result_spectrum`](@ref), pass `dt` as the interval between successive
results. `PhysicalScale.dt` is the delay between images within each pair;
it converts displacement to velocity and may differ from the sequence
cadence.

```@index
Pages = ["derived.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["derived.jl"]
Private = false
```
