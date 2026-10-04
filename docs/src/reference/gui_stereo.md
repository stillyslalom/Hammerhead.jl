```@meta
CurrentModule = HammerheadGUI
```

# GUI stereo sessions (HammerheadGUI)

A two-camera session of the Hammerhead window ([`hammerhead`](@ref),
[`stereo_window`](@ref)) shows a
[`StereoWorkflow`](@ref Controllers.StereoWorkflow); the same controllers
run without a window. For its tasks, see
[Run stereo PIV in the GUI](../howto/gui_stereo.md). The settings, test,
run, and results functions it shares with planar sessions are in the
[GUI reference](gui.md).

```@index
Pages = ["gui_stereo.md"]
```

## Stereo canvas

In a stereo session the canvas shows the raw frame on
Images, the selected calibration plate with its reprojection residuals on
Calibration, and the shown camera's dewarped frame on the later steps, where
[`grid_vector_data`](@ref) places stereo vectors on the dewarped grid.

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["qt/stereo_shell.jl", "stereo_canvas.jl"]
```

## Stereo workflow controllers

[`StereoWorkflow`](@ref Controllers.StereoWorkflow) is the stereo
counterpart of `PlanarWorkflow`; both are
[`AbstractWorkflow`](@ref Controllers.AbstractWorkflow)s and share the
settings, test, run, and results functions. It adds two synchronized
camera frame sets and the Calibration step,
[`StereoCalibration`](@ref Controllers.StereoCalibration): plate images
per camera, grid detection and camera fits, the shared dewarp grid, and
self-calibration. Its Prepare step works on the dewarped grid, and its
test and run call the stereo `apply_recipe(recipe, pairs1, pairs2, dw1, dw2)`.
A recipe holds processing settings only; the calibration is session state,
and [`set_dewarpers!`](@ref Controllers.set_dewarpers!) or
`hammerhead(; dewarpers)` start from dewarpers built elsewhere.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["controllers/stereo_workflow.jl", "controllers/stereo_calibration.jl"]
```

## Calibration review

[`calibration_review`](@ref) is a standalone GLMakie window that checks dot
detection and reprojection errors per calibration plane for one camera;
[`calibration_review!`](@ref) embeds the same review in a larger figure.
[`build_dewarpers`](@ref Controllers.build_dewarpers) turns two fitted
reviews into the dewarper pair that `hammerhead(; dewarpers)` and
[`run_piv_stereo`](@ref Hammerhead.run_piv_stereo) take.
[`selfcal_review`](@ref) browses a self-calibration report and its
disparity maps.

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["HammerheadGUI.jl", "views/calibration_review.jl"]
```

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["controllers/calibration_review.jl"]
```
