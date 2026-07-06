```@meta
CurrentModule = Hammerhead
```

# Validation and quality

Vector validation (universal outlier detection, peak-ratio and other
criteria), outlier replacement, and field smoothing. Validation runs
automatically inside [`run_piv`](@ref) — see [`PIVParameters`](@ref) for the
knobs and the [validation how-to](../howto/validation.md) for tuning
guidance; the functions below are the building blocks, usable standalone.

```@index
Pages = ["validation.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["quality.jl"]
Private = false
```
