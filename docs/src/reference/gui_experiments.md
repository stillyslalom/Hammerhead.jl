```@meta
CurrentModule = HammerheadGUI
```

# GUI experiment workflows

Use the [saved-experiment workflow](../howto/gui_experiments.md) to snapshot
current batch settings or reopen a complete planar recipe. Imported recipes
retain their exact settings in a dedicated controller, including settings that
the ordinary batch form cannot edit.
Pass edits use the separate [recipe revision API](gui_recipe_revision.md).
Opening a record in this workflow preserves its complete saved settings.
Use the separate [recipe-comparison workflow](gui_comparison.md) to rerun two
saved recipes on an explicitly selected pair without changing either run history.
Ordinary replay has scalar written-pair progress and boundary cancellation;
see [monitor and cancel replay](../howto/gui_experiment_replay.md) for the
distinction between GUI cancellation, failed core history and checkpoint resume.
Quality reports default to format 1; independent history/execution toggles opt
into formats 2/3. Their verification is at generation time and their summaries
identify the reported run. `report_path_picker` optionally supplies a destination
function for embedded applications; the default remains the TOML save dialog.

```@index
Pages = ["gui_experiments.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/experiment_controller.jl", "views/experiment_workflow.jl"]
Order = [:type, :function]
```
