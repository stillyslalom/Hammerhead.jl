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

include("controllers/result_explorer.jl")

export ResultExplorer, nframes, current_result, set_frame!,
       available_fields, field_values, field_name, field_label, set_field!,
       select_nearest!, clear_selection!, describe_selection,
       vector_data, auto_lengthscale

end # module Controllers

using .Controllers

export ResultExplorer, result_explorer, nframes, current_result, set_frame!,
       available_fields, field_values, set_field!,
       select_nearest!, clear_selection!, describe_selection

include("views/result_explorer.jl")

end # module HammerheadGUI
