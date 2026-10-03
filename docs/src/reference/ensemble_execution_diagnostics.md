# Ensemble execution diagnostics

Ensemble observations use `ensemble_execution_diagnostics_format_version = 1`
and sibling entries such as `ensemble_execution_diagnostics/000001`, with
explicit `result_key="results/000001"` linkage.
They retain scalar pass populations and explicit pooled-sweep semantics, rather
than adapting planar iteration diagnostics or inventing per-pair measurements.
Metadata-only inspection, supplied-result checks and read-time measurement-field
verification have distinct runtime states; those states are not persisted.

The native packet digest binds its primitive metadata. Measurement binding covers
raw axes, displacement, quality metrics, uncertainty, masks/flags and scale,
excluding parameters and retained correlation planes. Independent geometry and
count validation still applies when hashes have deliberately been recomputed.
Stationarity, independence, source authenticity and estimator applicability are
not established by either hash or structural checks.

```@autodocs
Modules = [Hammerhead]
Pages = ["src/ensemble_execution_diagnostics.jl"]
Public = true
```
