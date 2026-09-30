```@meta
CurrentModule = Hammerhead
```

# Validation and quality

Vector validation (universal outlier detection, peak-ratio and other
criteria), outlier replacement, and field smoothing. Validation runs
automatically inside [`run_piv`](@ref). See [`PIVParameters`](@ref) for the
settings and the [validation how-to](../howto/validation.md) for tuning
guidance. You can also call the functions below individually.

```@index
Pages = ["validation.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["quality.jl"]
Private = false
```
