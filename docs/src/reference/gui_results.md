```@meta
CurrentModule = HammerheadGUI
```

# GUI result explorer

The result explorer combines result browsing, field display, vector selection
and planar interactive analysis. For an example, see the
[GUI tour](../tutorials/gui_tour.md).

`ResultExplorer(path; lazy = true)` or `ResultExplorer(ResultFile(path))`
browses a completed results file while holding one display result in memory.
Only the current frame's derived fields are retained; a failed read sets
`status` and keeps the prior frame on screen. Scaled magnitude fields are
labelled speed in length/time units; unscaled magnitude keeps the displacement
label.

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
Order = [:type, :function]
Pages = ["result_explorer.jl"]
```
