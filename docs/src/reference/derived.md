```@meta
CurrentModule = Hammerhead
```

# Derived flow analysis

Spatial derivatives use only immediate valid neighbours and never cross a
mask or outlier. The 2D swirling-strength and Q definitions describe the
measured in-plane gradient tensor; they do not assume unmeasured 3D terms.

Area-form circulation integrates vorticity over the portion of each grid cell
inside the requested rectangle or polygon. This preserves fractional cell
boundaries and clips regions to the measured grid.
Cells with a masked, excluded outlier, or non-finite vorticity corner are
omitted; `include_invalid=true` admits outliers but still excludes masks.

For [`result_spectrum`](@ref), pass `dt` as the interval between successive
results. The delay between images within a pair, stored in
`PhysicalScale.dt`, is a velocity conversion factor and may differ from the
sequence cadence.

```@index
Pages = ["derived.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["derived.jl"]
Private = false
```
