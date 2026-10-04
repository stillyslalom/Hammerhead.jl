```@meta
CurrentModule = HammerheadGUI
```

# GUI result explorer

The result explorer combines result browsing, field display, vector selection
and planar interactive analysis. The Hammerhead window's Results step shows a
[`ResultExplorer`](@ref Controllers.ResultExplorer) on its own canvas;
[`result_explorer`](@ref) opens one in a separate GLMakie window. For an
example, see the [GUI tour](../tutorials/gui_tour.md).

`ResultExplorer(path; lazy = true)` or `ResultExplorer(ResultFile(path))`
browses a completed results file while holding one display result in memory.
Only the current frame's derived fields are retained; a failed read sets
`status` and keeps the prior frame on screen. Field labels carry the result's
units: with a `PhysicalScale`, displacement fields are velocities and read in
length per time (for example `|velocity| (mm/s)`), otherwise they are
displacements in the measured units (for example `|displacement| (px)`, pixels
per frame for planar PIV).

The tools (`set_tool!`: `:inspect`, `:profile`, `:circulation`) take their
input from `click!` and `alt_click!` in the field's data coordinates, as the
canvases send them; `tool_summary` and `profile_series` give what the window
shows.

```@index
Pages = ["gui_results.md"]
```

## Views

```@autodocs
Modules = [HammerheadGUI]
Order = [:type, :function]
Pages = ["result_explorer.jl"]
```

## Controller and analysis

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:type, :function, :constant]
Pages = ["result_explorer.jl"]
```
