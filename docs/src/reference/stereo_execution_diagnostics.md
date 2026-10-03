```@meta
CurrentModule = Hammerhead
```

# Stereo execution diagnostics

The [how-to](../howto/stereo_execution_diagnostics.md) covers actual per-camera
sweep observations, dewarped-pixel versus world coordinates, callbacks,
measurement-field binding and native persistence.

The stereo packet composes two existing planar companions. Its version-1
`stereo_execution_diagnostics` sibling group is separate from planar metadata;
result format 1 and registered result layouts remain unchanged. The metadata
checksum and measurement-field bindings do not authenticate calibration or source
inputs. Runtime-only verification distinguishes capture, metadata-only loading and
loading with an independently checked selected result. Parameters and correlation
planes are explicitly excluded from binding.

`execution_diagnostics_data(packet; result=raw)` verifies already loaded raw
measurement fields and geometry without rereading a native payload. Its detached
data has `inspection_state="supplied_measurement_fields_verified"`; the immutable
packet's original state is unchanged and no result is retained. Omitting `result`
preserves existing capture/metadata/read status. Verification concerns supplied
fields, not the source file or calibration. This is the shared path used by
[version-3 quality reports](run_quality.md#Opt-in-format-3) and lazy GUI inspection;
report loading preserves past generation checks and does not reverify results.

```@autodocs
Modules = [Hammerhead]
Pages = ["stereo_execution_diagnostics.jl"]
Private = false
```
