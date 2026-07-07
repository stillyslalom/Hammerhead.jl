# HammerheadGUI.jl

Desktop GUI for [Hammerhead.jl](https://github.com/stillyslalom/Hammerhead.jl)
(particle image velocimetry), built on GLMakie + NativeFileDialog.

Lives in the Hammerhead monorepo as a subdirectory package (Makie-style):
the core package stays at the repo root and never gains GUI dependencies;
this package is where the GLMakie hard dependency lives.

## Status

Phase 7 in the repo [ROADMAP](../ROADMAP.md), in progress.

- **Result explorer** (done) — `result_explorer(results_or_path)` opens a
  read-only viewer for `PIVResult` / `StereoPIVResult` sequences: scalar
  field heatmap (magnitude, components, diagnostics, uncertainty), vector
  arrows with outliers flagged, frame scrubbing, click-to-inspect
- Mask editor — polygon drawing/editing, exporting the package mask convention
- Parameter form + batch runner
- Calibration & self-calibration diagnostics

## Architecture rule

All application state and logic live in a framework-free controller layer
(plain Julia + Observables). Makie code renders controllers and pushes user
input into them, but controllers never depend on Makie, so the logic is
testable without a GL context and the widget shell stays swappable.

## Development

On Julia ≥ 1.11 the `[sources]` entry in `Project.toml` couples this package
to the sibling core checkout automatically:

```
julia --project=HammerheadGUI -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

On 1.10, `Pkg.develop(path="..")` into the GUI environment first (CI does the
equivalent). Releases go core-first, then a GUI compat bump; registration
uses `subdir=HammerheadGUI` and TagBot tags releases as `HammerheadGUI-v*`.
