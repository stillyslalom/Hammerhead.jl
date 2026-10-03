```@meta
CurrentModule = Hammerhead
```

# Actual-time tracking

Use the [actual-time tracking how-to](../howto/tracking_timing.md) for explicit
sample coordinates, predictor/UOD normalization, secant support, units and the
dedicated persistence/export contract. These wrappers preserve the existing
registered `TrackingResult` and `Trajectory` payload layouts.

```@autodocs
Modules = [Hammerhead]
Pages = ["tracking_timing.jl"]
Private = false
```
