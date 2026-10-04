```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The planar PIV window ([`planar_window`](@ref)) is a Qt Quick application
(QML.jl) with Makie canvases. Every button calls a function in
`HammerheadGUI.Controllers`, so the same steps run in a script, without a
display. Start with the [GUI tour](../tutorials/gui_tour.md) for a worked
session, or [Analyze an image pair in the GUI](../howto/gui.md) for the
window's tasks. Result browsing and interactive analysis have a separate
[result explorer reference](gui_results.md).

```@index
Pages = ["gui.md"]
```

## Workflow window

`planar_window` shows a [`PlanarWorkflow`](@ref Controllers.PlanarWorkflow);
the same controller runs without a window. Its canvases create every plot
before they are first shown and afterwards only update plot data, because a
Qt canvas has a current OpenGL context only while Qt renders it.
`request_grab` saves an image of the open window for screenshots and render
checks.

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["shell.jl", "planar_canvas.jl", "results_canvas.jl"]
```

## Workflow controllers

[`PlanarWorkflow`](@ref Controllers.PlanarWorkflow) holds one controller per
step and the settings. [`workflow_recipe`](@ref Controllers.workflow_recipe)
turns the settings into a core [`PIVRecipe`](@ref Hammerhead.PIVRecipe);
[`save_settings`](@ref Controllers.save_settings) writes it, and
[`load_settings!`](@ref Controllers.load_settings!) reads a recipe file, a
results file from an earlier run, or a `PIVRecipe`. Test pairs and runs go
through `apply_recipe`, so a test predicts the batch.

Canvas gestures are controller functions:
[`canvas_click!`](@ref Controllers.canvas_click!),
[`canvas_alt_click!`](@ref Controllers.canvas_alt_click!) and
[`canvas_key!`](@ref Controllers.canvas_key!) route a click or key to the
open Prepare page's editor, as the window does. With `wf.spawn[]` set (as a
window sets it), loading frames, previews, probes and background estimates
run on worker tasks and hand their results to `wf.deliver[]`.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["planar_workflow.jl", "frame_set.jl", "controllers/prepare.jl", "prepare_workflow.jl",
         "passes_editor.jl", "workflow_jobs.jl"]
```

## Prepare editors

The Prepare step's pages edit the workflow through these controllers
(`wf.prepare.preview`, `wf.prepare.mask[]`, `wf.prepare.roi[]`,
`wf.prepare.scale[]`). [`PreprocessPreview`](@ref Controllers.PreprocessPreview)
holds core `PreprocessStep`s and previews them with `recipe_preprocess`;
[`MaskEditor`](@ref Controllers.MaskEditor) exports its polygons with
`polygon_mask`; [`ROIEditor`](@ref Controllers.ROIEditor) edits core `ROI`
bounds; [`ScaleTool`](@ref Controllers.ScaleTool) measures a pixel size from
two points of known separation. Each editor also works on its own, given an
image or an image size.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["controllers/preprocess_preview.jl", "controllers/mask_editor.jl",
         "controllers/roi_editor.jl", "controllers/scale_tool.jl", "controllers/shared.jl"]
```

## Calibration review and stereo batches

These GLMakie views cover stereo until the stereo workflow window exists:
[`calibration_review`](@ref) checks dot detection and reprojection errors per
calibration plane, [`stereo_calibration`](@ref) builds the shared dewarping
grid from two reviews, and [`stereo_batch_runner`](@ref) runs synchronized
camera frames. [`selfcal_review`](@ref) browses a self-calibration report.

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["HammerheadGUI.jl", "views/calibration_review.jl", "views/stereo_batch.jl"]
```

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["controllers/calibration_review.jl", "controllers/stereo_batch.jl"]
```
