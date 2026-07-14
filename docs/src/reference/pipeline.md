```@meta
CurrentModule = Hammerhead
```

# Core pipeline and parameters

The single-pair and multi-pass particle image velocimetry (PIV) engine:
[`run_piv`](@ref) and its
configuration ([`PIVParameters`](@ref), [`multipass_parameters`](@ref)),
the [`PIVResult`](@ref) container, analysis masks, correlators, and
plotting. See the [first tutorial](../tutorials/first_vector_field.md) for a
guided walkthrough.

All public PIV drivers accept an execution-backend selector. See
[Run PIV on a GPU](../howto/gpu.md) for optional package setup, the supported
feature matrix, device-memory behavior, and validation commands.

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

Results carry an optional [`PhysicalScale`](@ref) as metadata; the stored
arrays stay in measured units (pixels, or world units for stereo) until
[`physical`](@ref) converts them. See the
[scaling how-to](../howto/scaling.md) and
[the conventions page](../explanation/conventions.md).

```@autodocs
Modules = [Hammerhead]
Pages = ["scaling.jl"]
Private = false
```

## Masks

Analysis masks are image-sized `Bool` matrices, `true` marking excluded
pixels — see the [masking how-to](../howto/masking.md) and
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

Provided by a package extension — load a Makie backend (e.g. `using GLMakie`
or `using CairoMakie`) to activate the methods.

```@autodocs
Modules = [Hammerhead]
Pages = ["Hammerhead.jl"]
Private = false
```
