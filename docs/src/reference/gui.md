```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The planar PIV window ([`planar_window`](@ref)) and the stereo PIV window
([`stereo_window`](@ref)) are Qt Quick applications (QML.jl) with Makie
canvases. Every button calls a function in `HammerheadGUI.Controllers`, so
the same steps run in a script, without a display. Start with the
[GUI tour](../tutorials/gui_tour.md) for a worked session,
[Analyze an image pair in the GUI](../howto/gui.md) for the planar window's
tasks, or [Run stereo PIV in the GUI](../howto/gui_stereo.md) for the stereo
window. The stereo window's calibration and workflow controllers have their
own [stereo reference](gui_stereo.md), and result browsing and interactive
analysis a separate [result explorer reference](gui_results.md); the Prepare
step's editors are in the [Prepare editors reference](gui_prepare.md).

```@index
Pages = ["gui.md"]
```

## Workflow window

`planar_window` shows a [`PlanarWorkflow`](@ref Controllers.PlanarWorkflow)
and [`stereo_window`](@ref) a [`StereoWorkflow`](@ref Controllers.StereoWorkflow);
the same controllers run without a window. The canvases create every plot
before they are first shown and afterwards only update plot data, because a
Qt canvas has a current OpenGL context only while Qt renders it.
`request_grab` saves an image of the open window for screenshots and render
checks.

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["qt/shell.jl", "planar_canvas.jl", "results_canvas.jl"]
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
Pages = ["controllers/workflow.jl", "planar_workflow.jl", "frame_set.jl",
         "controllers/prepare.jl", "prepare_workflow.jl", "passes_editor.jl",
         "particle_settings.jl", "workflow_jobs.jl", "compute.jl"]
```
