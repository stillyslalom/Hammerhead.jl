```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The desktop GUI uses GLMakie. The application and view functions below open
interactive tools; `HammerheadGUI.Controllers` holds their state and actions
for scripted use without a display. Start with the
[GUI tour](../tutorials/gui_tour.md) for a worked session.
Result browsing, interactive analysis and recorded processing details have a
separate [result explorer reference](gui_results.md).
Saved-recipe controls and views have their own
[experiment workflow reference](gui_experiments.md).
Resumable processing has a separate [checkpoint workflow reference](gui_checkpoints.md).

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

## [Experimental Qt session](@id experimental-qt-session)

`experimental_qml_gui` opens Qt controls and a separate interactive GLMakie
plot in a fresh child process. Install the optional `QML` and `QMLMakie`
packages in the project supplied to the launcher; `project=Base.active_project()`
uses the caller's environment.

```julia
session = experimental_qml_gui(; experiment="recipe.jld2")
isopen(session)
close(session)
```

`experiment` opens a saved planar recipe; `result` opens an unscaled planar
results file. Choose one initial file. With both omitted, the shell starts its demonstration.
The default `visible=true, plot=:glfw` displays Qt controls and a separate
scientific plot. Session files record readiness, logs and shutdown details;
`session.directory` holds these files and `session.log_path` identifies the log.
`session_dir` names a new directory to create within an existing parent;
omitting it creates a persistent temporary directory. `wait=false` returns the
session handle after spawning so the caller can continue working in Julia.

The settings window owns replay and plot lifetime. Closing just the plot
releases its screen and allows reopening. Closing the session requests
shutdown, joins its worker, and releases GUI resources. `close(session;
timeout=120)` waits for that shutdown; after closing through the window, use
`wait(session)`. Both report an unsuccessful child exit with its log path.
A shutdown timeout leaves the session handle available for further inspection.
`plot=:preview` places a static scientific preview in the controls window;
`visible=false` runs hidden windows for automation.

See [Try the experimental Qt interface](@ref experimental-qt-interface)
for the user workflow and the [framework evaluation](../explanation/gui_framework.md)
for platform evidence and development probes.

```@autodocs
Modules = [HammerheadGUI]
Order = [:type, :function]
Pages = ["qml_gui.jl"]
```

## Controllers

Controllers hold application state and logic. You can use them without
opening a window or creating a GL context.

For planar image selection, `ROIEditor` edits core `ROI` bounds and
`apply_roi!` copies them into `BatchRunner`. `set_roi!` and `clear_roi!`
also operate directly on a batch controller. The `roi_editor!` view can
be embedded alongside an image or result comparison.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["mask_editor.jl", "preprocess_preview.jl", "roi_editor.jl",
         "batch_runner.jl", "scale_tool.jl", "calibration_review.jl", "stereo_batch.jl"]
```
