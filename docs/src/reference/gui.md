```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The desktop GUI uses GLMakie. The application and view functions below open
interactive tools; `HammerheadGUI.Controllers` holds their state and actions
for scripted use without a display. Start with the
[GUI tour](../tutorials/gui_tour.md) for a worked session.
Result browsing and interactive analysis have a separate
[result explorer reference](gui_results.md).

```@index
Pages = ["gui.md"]
```

## Application and views

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["HammerheadGUI.jl", "widgets.jl", "mask_editor.jl",
         "preprocess_preview.jl", "roi_editor.jl", "batch_runner.jl", "scale_tool.jl",
         "calibration_review.jl", "stereo_batch.jl"]
```

## Controllers

Controllers hold application state and logic. You can use them without
opening a window or creating a GL context.

For planar image selection, `ROIEditor` edits core `ROI` bounds and
`apply_roi!` copies them into `BatchRunner`. `set_roi!` and `clear_roi!`
also operate directly on a batch controller. The `roi_editor!` view can
be embedded alongside an image or result comparison.

The batch form's settings convert to a core [`PIVRecipe`](@ref Hammerhead.PIVRecipe)
with `batch_recipe`; `save_settings` writes it, and `load_settings!` reads a
recipe file, a results file from an earlier run, or a `PIVRecipe` back into the
form. `preprocess_steps` turns the preprocessing preview's enabled operations
into recipe steps.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["mask_editor.jl", "preprocess_preview.jl", "roi_editor.jl",
         "batch_runner.jl", "scale_tool.jl", "calibration_review.jl", "stereo_batch.jl"]
```
