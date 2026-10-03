```@meta
CurrentModule = HammerheadGUI
```

# GUI result explorer

The result explorer combines result browsing, field display, vector selection
and planar interactive analysis. For an example, see the
[GUI tour](../tutorials/gui_tour.md). Recorded final-sweep history and execution
counts have a separate [inspection guide](../howto/gui_companions.md).

`ResultExplorer(path; lazy = true)` or `ResultExplorer(ResultFile(path))`
browses a closed results file with one cached display result. Only the current
frame's derived fields are retained; lazy navigation errors set `status` and
preserve the prior frame. The key index has O(number of entries) metadata and
does not follow live writes. Each complete selected result must fit memory.

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
