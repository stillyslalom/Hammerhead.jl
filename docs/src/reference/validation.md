```@meta
CurrentModule = Hammerhead
```

# Validation and quality

[`run_piv`](@ref) applies universal outlier detection by default. A
peak-ratio threshold and other checks can be added through
[`PIVParameters`](@ref). Flagged vectors can retain numerical `u` and `v`
values after local-median replacement, so keep their `outliers` flags when
filtering a field. This page lists the validators and smoothing functions;
see [Tune validation](../howto/validation.md) for settings and examples.

```@index
Pages = ["validation.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["src/quality.jl"]
Private = false
```
