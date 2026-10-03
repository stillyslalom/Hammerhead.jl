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

# Framework-free controller layer: the submodule boundary keeps Makie names
# out of scope, so controller code cannot grow GL dependencies by accident.
module Controllers

using Hammerhead
using Observables
using Printf
using LinearAlgebra: LinearAlgebra
using FileIO: FileIO
using ImageCore: Gray
import SHA

include("controllers/result_explorer.jl")
include("controllers/result_quality_report.jl")
include("controllers/derivative_inspection.jl")
include("controllers/mask_editor.jl")
include("controllers/preprocess_preview.jl")   # before batch_runner (set_preprocess! signature)
include("controllers/roi_editor.jl")
include("controllers/batch_runner.jl")
include("controllers/experiment_controller.jl")
include("controllers/recipe_comparison.jl")
include("controllers/recipe_revision.jl")
include("controllers/recipe_image_preview.jl")
include("controllers/recipe_mask_reference.jl")
include("controllers/checkpoint_controller.jl")
include("controllers/scale_tool.jl")           # after batch_runner (apply_scale! signature)
include("controllers/calibration_review.jl")
include("controllers/stereo_batch.jl")         # after calibration_review (build_dewarpers signature)
include("controllers/stereo_experiment_controller.jl")
include("controllers/ensemble_experiments.jl")

export ResultExplorer, nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale, selection_point,
       trajectory_points, trajectory_gap_count,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       current_field_values, set_tool!, clear_tool!, tool_summary,
       set_companion_inspection!, companion_summary, describe_companion_selection
export set_derivative_stencil!, derivative_support_summary, describe_derivative_selection
export explorer_quality_report, save_explorer_quality_report
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
       start!, cancel!
export ROIEditor, set_roi!, clear_roi!, apply_roi!, roi_summary
export preprocess_steps, experiment_record, save_batch_experiment,
       ExperimentController, open_experiment!, save_experiment_record!,
       experiment_results, experiment_summary, experiment_run_history,
       experiment_quality_report, save_experiment_quality_report
export CheckpointController, create_checkpoint!, open_checkpoint!, refresh_checkpoint!,
       checkpoint_explorer, export_checkpoint_results!
export RecipeComparisonController, open_comparison_record!, set_comparison_pairs!,
       compare!, open_comparison_report!, save_comparison_report!, comparison_summary
export RecipeRevisionController, revision_fields, set_revision_pass!, insert_revision_pass!,
       move_revision_pass!, delete_revision_pass!, revision_recipe, revision_diff,
       revision_record, apply_recipe_revision!, save_recipe_revision!
export preprocessing_fields, set_revision_preprocess!, insert_revision_preprocess!,
       move_revision_preprocess!, delete_revision_preprocess!, set_revision_background!,
       load_revision_background!, RecipeImagePreviewController, preview_recipe_images!
export revision_roi_fields, revision_scale_fields, set_revision_roi!, set_revision_scale!
export set_revision_mask!, reset_revision_mask!, load_revision_mask!,
       RecipeMaskReferenceController, load_recipe_mask_reference!, apply_revision_mask!
export ScaleTool, clear_points!, set_separation!, pixel_distance,
       pixel_size, physical_scale, apply_scale!, scale_summary
export CalibrationReview, nplanes, set_plane!, refit!, plane_errors,
       plane_summary, fit_summary, selfcal_summary
export StereoBatchRunner, set_dewarpers!, build_dewarpers, stereo_pairs
export StereoExperimentController, select_experiment_run!
export EnsembleExperimentController, ensemble_experiment_record, save_ensemble_batch_experiment

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, result_explorer!,
       nframes, current_result, set_frame!, push_result!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       color_limits, set_color_mode!, set_color_limits!, current_color_limits,
       set_tool!, clear_tool!, tool_summary,
       set_companion_inspection!, companion_summary, describe_companion_selection
export set_derivative_stencil!, derivative_support_summary, describe_derivative_selection
export explorer_quality_report, save_explorer_quality_report, result_quality_report
export MaskEditor, mask_editor, add_vertex!, undo_vertex!, close_active!,
       begin_hole!, grow_mask!, shrink_mask!, delete_selected!, clear_polygons!, save_mask
export PreprocessPreview, preprocess_preview, preprocess_preview!,
       set_image!, set_background!, enable_step!, set_step_param!,
       move_step!, apply_pipeline, build_preprocess,
       set_pair!, set_probe_window!, clear_probe!, probe_summary
export BatchRunner, batch_runner, add_files!, clear_files!, set_schedule!,
       set_effort!, set_scale!, set_pixel_size!, set_dt!, set_preprocess!,
       start!, cancel!
export ROIEditor, roi_editor, roi_editor!, set_roi!, clear_roi!, apply_roi!
export preprocess_steps, experiment_record, save_batch_experiment,
       ExperimentController, open_experiment!, save_experiment_record!,
       experiment_results, experiment_summary, experiment_run_history,
       experiment_quality_report, save_experiment_quality_report,
       experiment_workflow, experiment_workflow!
export CheckpointController, create_checkpoint!, open_checkpoint!, refresh_checkpoint!,
       checkpoint_explorer, export_checkpoint_results!, checkpoint_workflow, checkpoint_workflow!
export RecipeComparisonController, open_comparison_record!, set_comparison_pairs!,
       compare!, open_comparison_report!, save_comparison_report!, comparison_summary,
       recipe_comparison, recipe_comparison!
export RecipeRevisionController, revision_fields, set_revision_pass!, insert_revision_pass!,
       move_revision_pass!, delete_revision_pass!, revision_recipe, revision_diff,
       revision_record, apply_recipe_revision!, save_recipe_revision!, recipe_revision, recipe_revision!
export preprocessing_fields, set_revision_preprocess!, insert_revision_preprocess!,
       move_revision_preprocess!, delete_revision_preprocess!, set_revision_background!,
       load_revision_background!, RecipeImagePreviewController, preview_recipe_images!
export preprocessing_revision, preprocessing_revision!
export revision_roi_fields, revision_scale_fields, set_revision_roi!, set_revision_scale!,
       recipe_geometry_revision, recipe_geometry_revision!
export set_revision_mask!, reset_revision_mask!, load_revision_mask!,
       RecipeMaskReferenceController, load_recipe_mask_reference!, apply_revision_mask!,
       recipe_mask_revision, recipe_mask_revision!
export ScaleTool, scale_tool, clear_points!, set_separation!,
       pixel_size, physical_scale, apply_scale!
export CalibrationReview, calibration_review, calibration_review!,
       selfcal_review, nplanes, set_plane!
export StereoBatchRunner, stereo_batch_runner, stereo_calibration,
       set_dewarpers!, build_dewarpers
export StereoExperimentController, select_experiment_run!, stereo_experiment_workflow, stereo_experiment_workflow!
export EnsembleExperimentController, ensemble_experiment_record, save_ensemble_batch_experiment
export ensemble_experiment_workflow, ensemble_experiment_workflow!

include("views/widgets.jl")
include("views/result_explorer.jl")
include("views/result_quality_report.jl")
include("views/recipe_comparison.jl")
include("views/recipe_revision.jl")
include("views/preprocessing_revision.jl")
include("views/recipe_geometry_revision.jl")
include("views/recipe_mask_revision.jl")
include("views/experiment_workflow.jl")
include("views/checkpoint_workflow.jl")
include("views/mask_editor.jl")
include("views/preprocess_preview.jl")
include("views/roi_editor.jl")
include("views/batch_runner.jl")
include("views/scale_tool.jl")
include("views/calibration_review.jl")
include("views/stereo_batch.jl")
include("views/stereo_experiment_workflow.jl")
include("views/ensemble_experiments.jl")

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

        pp = PreprocessPreview(imgA; enabled = [:percentile_stretch])
        preprocess_preview(pp)
        enable_step!(pp, :invert_image)
        build_preprocess(pp)

        bc = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                         padding = false, apodization = :none)
        batch_runner(bc)
        start!(bc; async = false)
    end
end

end # module HammerheadGUI
