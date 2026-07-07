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
using FileIO: FileIO
using ImageCore: Gray

include("controllers/result_explorer.jl")
include("controllers/mask_editor.jl")
include("controllers/batch_runner.jl")

export ResultExplorer, nframes, current_result, set_frame!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale
export MaskEditor, add_vertex!, undo_vertex!, close_active!,
       click!, alt_click!, polygon_at, delete_selected!, clear_polygons!,
       save_mask, status_text
export BatchRunner, BatchCancelled, add_files!, clear_files!, frame_pairs,
       parse_schedule, set_schedule!, build_parameters, validate,
       start!, cancel!

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, nframes, current_result, set_frame!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection
export MaskEditor, mask_editor, add_vertex!, undo_vertex!, close_active!,
       delete_selected!, clear_polygons!, save_mask
export BatchRunner, batch_runner, add_files!, clear_files!, set_schedule!,
       start!, cancel!

include("views/widgets.jl")
include("views/result_explorer.jl")
include("views/mask_editor.jl")
include("views/batch_runner.jl")

end # module HammerheadGUI
