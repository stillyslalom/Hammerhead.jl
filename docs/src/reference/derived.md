```@meta
CurrentModule = Hammerhead
```

# Derived flow analysis

Spatial derivatives use only immediate valid neighbours and never cross a
mask or outlier. The 2D swirling-strength and Q definitions describe the
measured in-plane gradient tensor; they do not assume unmeasured 3D terms.

```@index
Pages = ["derived.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["derived.jl"]
Private = false
```
