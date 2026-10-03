```@meta
CurrentModule = HammerheadGUI
```

# GUI experiment workflows

Use the [saved-experiment workflow](../howto/gui_experiments.md) to snapshot
current batch settings or reopen a complete planar recipe. Imported recipes
retain their exact settings in a dedicated controller, including settings that
the ordinary batch form cannot edit.
Use the separate [recipe-comparison workflow](gui_comparison.md) to rerun two
saved recipes on an explicitly selected pair without changing either run history.

```@index
Pages = ["gui_experiments.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["experiment_controller.jl", "experiment_workflow.jl"]
Order = [:type, :function]
```
