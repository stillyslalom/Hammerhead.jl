```@meta
CurrentModule = HammerheadGUI
```

# GUI result explorer

The result explorer combines result browsing, field display, vector selection
and planar interactive analysis. For an example, see the
[GUI tour](../tutorials/gui_tour.md). Recorded final-sweep history and execution
counts have a separate [inspection guide](../howto/gui_companions.md).
Dedicated actual-time trajectory bundles have an
[actual-time exploration guide](../howto/gui_tracking_timing.md). Pass a
`TimedTrackingResult` directly or choose `format=:timed_tracking` explicitly;
the wrapper survives physical conversion and selection. This is one complete
bundle, not native lazy per-sample browsing.

`ResultExplorer(path; lazy = true)` or `ResultExplorer(ResultFile(path))`
browses a closed results file with one cached display result. Only the current
frame's derived fields are retained; lazy navigation errors set `status` and
preserve the prior frame. The key index has O(number of entries) metadata and
does not follow live writes. Each complete selected result must fit memory.

Recorded processing details support planar history/execution and stereo camera
execution companions. Stereo raw measurement binding is verified before
physical conversion; the physical display has a separate mutation digest.
Camera residuals stay in dewarped pixels, with no per-node stereo history or
reconstructed world 3C residual. Inspection retains only the current display
and packets. Scaled magnitude fields are labelled speed in length/time units;
unscaled magnitude retains the displacement label.

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
