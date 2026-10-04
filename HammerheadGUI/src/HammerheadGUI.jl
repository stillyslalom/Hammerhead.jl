"""
    HammerheadGUI

Windows and controllers for setting up, running, and inspecting Hammerhead
analyses: the planar workflow window (`planar_window`), plus GLMakie views
for results, calibration, and stereo batches. Use `Controllers` to configure
and inspect an analysis without opening a window. Controller state is
exposed through `Observables`.
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

include("controllers/shared.jl")
include("controllers/result_explorer.jl")
include("controllers/mask_editor.jl")
include("controllers/preprocess_preview.jl")
include("controllers/roi_editor.jl")
include("controllers/scale_tool.jl")
include("controllers/calibration_review.jl")
include("controllers/stereo_batch.jl")         # after calibration_review (build_dewarpers signature)
include("controllers/frame_set.jl")            # workflow window controllers
include("controllers/passes_editor.jl")
include("controllers/workflow_jobs.jl")
include("controllers/prepare.jl")              # before the workflows (field type)
include("controllers/workflow.jl")             # AbstractWorkflow + shared steps
include("controllers/planar_workflow.jl")
include("controllers/stereo_calibration.jl")
include("controllers/stereo_workflow.jl")
include("controllers/prepare_workflow.jl")

export ResultExplorer, nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale, selection_point,
       trajectory_points, trajectory_gap_count,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       current_field_values, set_tool!, clear_tool!, tool_summary, profile_series
export MaskEditor, add_vertex!, undo_vertex!, close_active!, cancel_active!,
       click!, alt_click!, polygon_at, delete_selected!, clear_polygons!,
       begin_hole!, grow_mask!, shrink_mask!, set_raster!, has_mask, save_mask, status_text
export PreprocessPreview, PREPROCESS_OPERATIONS, preprocess_label, add_step!, remove_step!,
       move_step!, set_step_option!, set_steps!, step_options, set_background!,
       estimate_background, set_image!, set_pair!, set_frames!, preview_frames,
       apply_pipeline, build_preprocess, pipeline_summary,
       set_probe_window!, clear_probe!, probe_rect, probe_correlation, probe_summary
export BatchCancelled, parse_schedule, add_files!, clear_files!, frame_pairs,
       set_schedule!, set_effort!, build_parameters, build_scale, validate,
       start!, cancel!, save_settings, load_settings!
export ROIEditor, set_roi!, clear_roi!, cancel_corner!, roi_summary
export ScaleTool, clear_points!, undo_point!, set_separation!, pixel_distance,
       pixel_size, physical_scale, scale_summary, scale_description
export CalibrationReview, nplanes, set_plane!, refit!, plane_errors,
       plane_summary, fit_summary, selfcal_summary
export StereoBatchRunner, set_dewarpers!, build_dewarpers, stereo_pairs
export FrameSet, set_pair_mode!, npairs, select_pair!, show_frame!, current_pair,
       pair_images, shown_image, frames_problem, frames_summary, frame_size, pair_loading
export PassesEditor, fill_preset!, set_analysis_size!, set_mode!, set_image_type!,
       load_passes!, set_pass!, set_option!, add_pass!, remove_pass!, pass_rows,
       option_value, passes_summary
export PairTest, start_test!, test_summary, summary_lines, RunState, start_run!,
       cancel_run!, run_eta
export PrepareState, PREPARE_PAGES, set_prepare_page!, canvas_click!, canvas_alt_click!,
       canvas_key!, edit_step_option!, estimate_background!, edit_roi!, edit_scale!,
       set_scale_field!, clear_scale!, load_mask_file!, save_mask_file
export AbstractWorkflow, workflow_steps, prepare_pages, workflow_problem
export PlanarWorkflow, WORKFLOW_STEPS, workflow_recipe, settings_modified, set_step!,
       test_pair!, test_stale, open_results!, step_status
export StereoCalibration, CALIBRATION_OPTIONS, add_plate!, remove_plate!, set_plate_z!,
       clear_plates!, detect_options, set_calibration_option!, edit_calibration_option!,
       fit_calibration!, fit_stale, build_dewarpers!, grid_summary, calibration_summary,
       apply_selfcal!
export StereoWorkflow, STEREO_WORKFLOW_STEPS, STEREO_PREPARE_PAGES, camera_frames,
       shown_frames, set_camera!, out_of_view, grid_size, start_selfcal!

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, result_explorer!,
       nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       set_tool!, clear_tool!, tool_summary, profile_series
export MaskEditor, add_vertex!, undo_vertex!, close_active!, cancel_active!,
       begin_hole!, grow_mask!, shrink_mask!, delete_selected!, clear_polygons!,
       set_raster!, save_mask
export PreprocessPreview, add_step!, remove_step!, move_step!, set_step_option!,
       set_steps!, set_background!, set_image!, set_pair!, apply_pipeline,
       build_preprocess, set_probe_window!, clear_probe!, probe_summary
export add_files!, clear_files!, set_schedule!, set_effort!, start!, cancel!,
       save_settings, load_settings!
export ROIEditor, set_roi!, clear_roi!
export ScaleTool, clear_points!, set_separation!, pixel_size, physical_scale
export CalibrationReview, calibration_review, calibration_review!,
       selfcal_review, nplanes, set_plane!
export StereoBatchRunner, stereo_batch_runner, stereo_calibration,
       set_dewarpers!, build_dewarpers
export PlanarWorkflow, planar_window, workflow_recipe, test_pair!, start_run!, cancel_run!,
       open_results!, set_step!, set_prepare_page!
export StereoWorkflow, StereoCalibration, fit_calibration!, start_selfcal!, apply_selfcal!

include("views/widgets.jl")
include("views/result_explorer.jl")
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

        # Workflow window: controllers and canvases (the Qt window itself
        # needs a display and is not part of the workload).
        wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
        fill_preset!(wf.passes, :low)
        pc = planar_canvas(wf)
        set_step!(wf, :prepare)
        pp = wf.prepare.preview
        add_step!(pp, :percentile_stretch)
        edit_step_option!(wf, 1, "low", "2")
        add_step!(pp, :clahe)
        edit_step_option!(wf, 2, "tiles", "4, 4")
        canvas_click!(wf, 32.0, 32.0)                       # probe
        wf.prepare.show_processed[] = true
        set_prepare_page!(wf, :mask)
        for (x, y) in ((5.0, 5.0), (20.0, 5.0), (20.0, 20.0))
            canvas_click!(wf, x, y)
        end
        canvas_alt_click!(wf)
        canvas_click!(wf, 10.0, 8.0)                        # select
        canvas_key!(wf, :delete)
        set_prepare_page!(wf, :roi)
        canvas_click!(wf, 4.0, 4.0); canvas_click!(wf, 60.0, 60.0)
        edit_roi!(wf, "2", "63", "2", "63")
        set_prepare_page!(wf, :scale)
        canvas_click!(wf, 4.0, 4.0); canvas_click!(wf, 40.0, 4.0)
        edit_scale!(wf, :separation, "2")
        edit_scale!(wf, :dt, "0.001")
        clear_scale!(wf)
        wf.roi[] = nothing
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
