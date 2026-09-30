```@meta
CurrentModule = Hammerhead
```

# Derived flow analysis

Use the functions below for profiles, regions, circulation, spatial
derivatives, and time spectra. Derivatives use immediate valid neighbors;
they do not cross a mask or excluded outlier. The 2D swirling-strength and
Q values describe the measured in-plane gradient tensor and do not include
unmeasured 3D terms.

`extract_profile` interpolates only from corners with positive weight. A
masked or invalid corner does not invalidate an exact node or edge sample
when its weight is zero; a contributing invalid corner yields `NaN`.
`extract_region` returns `included`, a Boolean grid where `true` marks a
returned node. Its legacy `mask` field aliases the same inclusion grid;
`result.mask` uses the opposite convention (`true` means excluded).

Area-form circulation integrates vorticity over the portion of each grid cell
inside a requested rectangle or polygon, including fractional boundary cells.
Cells with a masked, excluded outlier, or non-finite vorticity corner are
omitted; `include_invalid=true` admits outliers but still excludes masks.
By default, an incomplete requested region raises an error rather than
returning an unmarked partial integral. Use `coverage=:report` to receive
`(; value, valid_area, requested_area, coverage_fraction, complete)`. When
some area is missing, `value` integrates only the valid area; when none is
valid, it is `NaN`. Check `complete` to decide whether the full region was
integrated; a displayed fraction can round to 1 even with a small gap.

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
