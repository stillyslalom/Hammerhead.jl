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

```@autodocs
Modules = [Hammerhead]
Pages = ["stereo_execution_diagnostics.jl"]
Private = false
```
