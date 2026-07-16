"""
    HammerheadGUI

Desktop GUI for Hammerhead.jl (particle image velocimetry) built on GLMakie.

Application state and logic live in the framework-free `Controllers`
submodule (plain Julia + Observables — no Makie in scope); the view layer
renders controllers and pushes user input into them, so the widget shell
stays swappable and the logic testable without a GL context.
"""
module HammerheadGUI

using Hammerhead
using GLMakie
using NativeFileDialog

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
include("controllers/batch_runner.jl")
include("controllers/calibration_review.jl")

export ResultExplorer, nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale, selection_point,
       trajectory_points, trajectory_gap_count,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits
export MaskEditor, add_vertex!, undo_vertex!, close_active!,
       click!, alt_click!, polygon_at, delete_selected!, clear_polygons!,
       begin_hole!, grow_mask!, shrink_mask!, save_mask, status_text
export BatchRunner, BatchCancelled, add_files!, clear_files!, frame_pairs,
       parse_schedule, set_schedule!, set_effort!, set_pixel_size!, set_dt!,
       set_scale!, build_parameters, build_scale, validate,
       start!, cancel!
export CalibrationReview, nplanes, set_plane!, refit!, plane_errors,
       plane_summary, fit_summary, selfcal_summary

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, result_explorer!,
       nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits
export MaskEditor, mask_editor, add_vertex!, undo_vertex!, close_active!,
       begin_hole!, grow_mask!, shrink_mask!, delete_selected!, clear_polygons!, save_mask
export BatchRunner, batch_runner, add_files!, clear_files!, set_schedule!,
       set_effort!, set_scale!, start!, cancel!
export CalibrationReview, calibration_review, selfcal_review,
       nplanes, set_plane!

include("views/widgets.jl")
include("views/result_explorer.jl")
include("views/mask_editor.jl")
include("views/batch_runner.jl")
include("views/calibration_review.jl")

using PrecompileTools: @setup_workload, @compile_workload

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

        bc = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                         padding = false, apodization = :none)
        batch_runner(bc)
        start!(bc; async = false)
    end
end

end # module HammerheadGUI
