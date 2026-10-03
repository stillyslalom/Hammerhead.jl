```@meta
CurrentModule = Hammerhead
```

# Actual-time tracking

Use the [actual-time tracking how-to](../howto/tracking_timing.md) for explicit
sample coordinates, predictor/UOD normalization, secant support, units and the
dedicated persistence/export contract. These wrappers preserve the existing
registered `TrackingResult` and `Trajectory` payload layouts.
The bulk `tracking_speed_summary` validates once and returns detached scalar
observation-mean secant speeds, unavailable reasons and exact track time ranges.
For bounded GUI selection and dedicated artifact loading, see
[actual-time trajectory exploration](../howto/gui_tracking_timing.md).

```@autodocs
Modules = [Hammerhead]
Pages = ["tracking_timing.jl"]
Private = false
```
