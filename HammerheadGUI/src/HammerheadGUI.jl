"""
    HammerheadGUI

GLMakie views and controllers for inspecting and running Hammerhead analyses.
Use the views for interactive work, or use `Controllers` to configure and
inspect an analysis without opening a window. Controller state is exposed
through `Observables`.
"""
module HammerheadGUI

using Hammerhead
using GLMakie
using NativeFileDialog
using QML: QML, JuliaPropertyMap, JuliaItemModel, loadqml, exec, @qmlfunction
import QMLMakie                       # registers the MakieArea QML type
import Qt6Declarative_jll
import Libdl

# Framework-free controller layer: the submodule boundary keeps Makie names
# out of scope, so controller code cannot grow GL dependencies by accident.
module Controllers

using Hammerhead
using Observables
using Printf
using LinearAlgebra: LinearAlgebra
using FileIO: FileIO
using ImageCore: Gray

include("controllers/result_explorer.jl")
include("controllers/mask_editor.jl")
include("controllers/preprocess_preview.jl")   # before batch_runner (set_preprocess! signature)
include("controllers/roi_editor.jl")
include("controllers/batch_runner.jl")
include("controllers/scale_tool.jl")           # after batch_runner (apply_scale! signature)
include("controllers/calibration_review.jl")
include("controllers/stereo_batch.jl")         # after calibration_review (build_dewarpers signature)
include("controllers/frame_set.jl")            # workflow window controllers (after batch_runner: _errmsg, BatchCancelled)
include("controllers/passes_editor.jl")
include("controllers/workflow_jobs.jl")
include("controllers/planar_workflow.jl")

export ResultExplorer, nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale, selection_point,
       trajectory_points, trajectory_gap_count,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       current_field_values, set_tool!, clear_tool!, tool_summary
export MaskEditor, add_vertex!, undo_vertex!, close_active!,
       click!, alt_click!, polygon_at, delete_selected!, clear_polygons!,
       begin_hole!, grow_mask!, shrink_mask!, save_mask, status_text
export PreprocessPreview, PreprocStep, set_image!, set_background!,
       enable_step!, set_step_param!, move_step!, apply_pipeline,
       build_preprocess, pipeline_summary,
       set_pair!, set_probe_window!, clear_probe!, probe_summary
export BatchRunner, BatchCancelled, add_files!, clear_files!, frame_pairs,
       parse_schedule, set_schedule!, set_effort!, set_pixel_size!, set_dt!,
       set_scale!, set_preprocess!, build_parameters, build_scale, validate,
       start!, cancel!, batch_recipe, save_settings, load_settings!, preprocess_steps
export ROIEditor, set_roi!, clear_roi!, apply_roi!, roi_summary
export ScaleTool, clear_points!, set_separation!, pixel_distance,
       pixel_size, physical_scale, apply_scale!, scale_summary
export CalibrationReview, nplanes, set_plane!, refit!, plane_errors,
       plane_summary, fit_summary, selfcal_summary
export StereoBatchRunner, set_dewarpers!, build_dewarpers, stereo_pairs
export FrameSet, set_pair_mode!, npairs, select_pair!, show_frame!, current_pair,
       pair_images, shown_image, frames_problem, frames_summary, frame_size
export PassesEditor, fill_preset!, set_analysis_size!, set_mode!, set_image_type!,
       load_passes!, set_pass!, set_option!, add_pass!, remove_pass!, pass_rows,
       option_value, passes_summary
export PairTest, start_test!, test_summary, summary_lines, RunState, start_run!,
       cancel_run!, run_eta
export PlanarWorkflow, WORKFLOW_STEPS, workflow_recipe, settings_modified, set_step!,
       test_pair!, test_stale, open_results!, step_status

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, result_explorer!,
       nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       set_tool!, clear_tool!, tool_summary
export MaskEditor, mask_editor, add_vertex!, undo_vertex!, close_active!,
       begin_hole!, grow_mask!, shrink_mask!, delete_selected!, clear_polygons!, save_mask
export PreprocessPreview, preprocess_preview, preprocess_preview!,
       set_image!, set_background!, enable_step!, set_step_param!,
       move_step!, apply_pipeline, build_preprocess,
       set_pair!, set_probe_window!, clear_probe!, probe_summary
export BatchRunner, batch_runner, add_files!, clear_files!, set_schedule!,
       set_effort!, set_scale!, set_pixel_size!, set_dt!, set_preprocess!,
       start!, cancel!, batch_recipe, save_settings, load_settings!
export ROIEditor, roi_editor, roi_editor!, set_roi!, clear_roi!, apply_roi!
export ScaleTool, scale_tool, clear_points!, set_separation!,
       pixel_size, physical_scale, apply_scale!
export CalibrationReview, calibration_review, calibration_review!,
       selfcal_review, nplanes, set_plane!
export StereoBatchRunner, stereo_batch_runner, stereo_calibration,
       set_dewarpers!, build_dewarpers
export PlanarWorkflow, planar_window, workflow_recipe, test_pair!, start_run!, cancel_run!,
       open_results!, set_step!

include("views/widgets.jl")
include("views/result_explorer.jl")
include("views/mask_editor.jl")
include("views/preprocess_preview.jl")
include("views/roi_editor.jl")
include("views/batch_runner.jl")
include("views/scale_tool.jl")
include("views/calibration_review.jl")
include("views/stereo_batch.jl")
include("canvas/planar_canvas.jl")
include("canvas/results_canvas.jl")
include("qt/shell.jl")

using PrecompileTools: @setup_workload, @compile_workload

# Evaluates traced `precompile(...)` statements (see qt/precompile_statements.jl)
# with every loaded package's name in scope; failing lines are skipped.
module _TracedPrecompiles end
function _precompile_traced(path::AbstractString)
    M = _TracedPrecompiles
    for m in values(Base.loaded_modules)
        name = nameof(m)
        isdefined(M, name) || Core.eval(M, :(const $name = $m))
    end
    for line in eachline(path)
        startswith(line, "precompile(") || continue
        try
            Core.eval(M, Meta.parse(line))
        catch
        end
    end
    return
end
include_dependency(joinpath(@__DIR__, "qt", "precompile_statements.jl"))

# Time-to-first-window workload: run the pipeline once and build each view
# (Figure construction only — no GL context at precompile time, so no
# colorbuffer/display).
@setup_workload begin
    imgA = rand(64, 64)
    imgB = circshift(imgA, (2, 3))
    @compile_workload begin
        r = run_piv(imgA, imgB, PIVParameters(window_size = 32))
        ex = ResultExplorer(r)
        result_explorer(ex)
        set_field!(ex, :peak_ratio)
        select_nearest!(ex, r.x[1], r.y[1])
        describe_selection(ex)

        # Scattered-result explorer paths (PTV particles + trajectories),
        # built from tiny constructed results so the first window on those
        # paths is warm too. Figure construction only — no GL context here.
        pts = Particles([10.0, 20.0, 30.0], [10.0, 20.0, 30.0],
                        [1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
        ptv = PTVResult([10.0, 20.0, 30.0], [10.0, 20.0, 30.0],
                        [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1],
                        falses(3), [1, 2, 3], [1, 2, 3], pts, pts, PTVParameters())
        exp = ResultExplorer(ptv)
        result_explorer(exp)
        select_nearest!(exp, 10.0, 10.0)
        describe_selection(exp)

        tr = TrackingResult([Trajectory(1, [10.0, 11.0, 12.0], [10.0, 10.5, 11.0])],
                            3, PTVParameters())
        result_explorer(ResultExplorer(tr))

        me = MaskEditor(imgA)
        mask_editor(me)
        Controllers.click!(me, 5.0, 5.0)
        Controllers.click!(me, 20.0, 5.0)
        Controllers.click!(me, 20.0, 20.0)
        close_active!(me)
        polygon_mask(me)

        pp = PreprocessPreview(imgA; enabled = [:percentile_stretch])
        preprocess_preview(pp)
        enable_step!(pp, :invert_image)
        build_preprocess(pp)

        bc = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                         padding = false, apodization = :none)
        batch_runner(bc)
        start!(bc; async = false)

        # Workflow window: controllers and canvases (the Qt window itself
        # needs a display and is not part of the workload).
        wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
        fill_preset!(wf.passes, :low)
        pc = planar_canvas(wf)
        set_step!(wf, :passes)
        test_pair!(wf; spawn = false)
        set_step!(wf, :test)
        start_run!(wf; spawn = false)
        set_step!(wf, :results)
        for st in WORKFLOW_STEPS
            step_status(wf, st)
        end
        rc = results_canvas()
        set_explorer!(rc, wf.explorer[])
        set_field!(wf.explorer[], :vorticity)
        set_frame!(wf.explorer[], 2)
        summary_lines(test_summary(wf.test))

        # First render of a Qt canvas (needs a GL context, so traced instead).
        _precompile_traced(joinpath(@__DIR__, "qt", "precompile_statements.jl"))
    end
end

end # module HammerheadGUI
