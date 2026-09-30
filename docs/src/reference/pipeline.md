```@meta
CurrentModule = Hammerhead
```

# Core pipeline and parameters

Use [`run_piv`](@ref) for one image pair. Pass [`PIVParameters`](@ref) for
one interrogation pass or [`multipass_parameters`](@ref) for a sequence of
window sizes. The returned [`PIVResult`](@ref) stores window centers, pixel
displacements, validation flags, and correlation diagnostics on the same
grid. See the [first tutorial](../tutorials/first_vector_field.md) for a
worked measurement.

Exclude `result.mask` and `result.outliers` when summarizing measured vectors.
Flagged `u` and `v` entries may hold replacement values; masked entries are
`NaN`. The [validation guide](../howto/validation.md) explains the checks.
For backend selection and supported options, see
[Run PIV on a GPU](../howto/gpu.md).

```@index
Pages = ["pipeline.md"]
```

## Running an analysis

```@autodocs
Modules = [Hammerhead]
Pages = ["types.jl", "pipeline.jl"]
Private = false
```

## Physical units

Attach a [`PhysicalScale`](@ref) with the exposure separation and spatial
calibration for planar PIV. The stored arrays stay in pixels until
[`physical`](@ref) converts positions to length and displacements to
velocity. Stereo arrays start in world length units, so they need the
exposure separation for velocity conversion. See the
[scaling how-to](../howto/scaling.md) and
[the conventions page](../explanation/conventions.md).

```@autodocs
Modules = [Hammerhead]
Pages = ["scaling.jl"]
Private = false
```

## Masks

Analysis masks are image-sized `Bool` matrices with `true` marking excluded
pixels. See the [masking how-to](../howto/masking.md) and
[the masking model](../explanation/masking.md). Masks built from image files
use [`load_mask`](@ref).

```@autodocs
Modules = [Hammerhead]
Pages = ["masking.jl"]
Private = false
```

## Correlators

Lower-level access to the fast Fourier transform (FFT) correlation engine used
by [`run_piv`](@ref):
correlator objects cache FFTW plans and buffers per window size.

```@autodocs
Modules = [Hammerhead]
Pages = ["correlators.jl"]
Private = false
```

## Plotting

Load a Makie backend (e.g. `using GLMakie` or `using CairoMakie`) to enable
the plotting methods.

```@autodocs
Modules = [Hammerhead]
Pages = ["Hammerhead.jl"]
Private = false
```
