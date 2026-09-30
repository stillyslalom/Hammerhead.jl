# The graphical user interface (GUI) controller–view split

Each HammerheadGUI tool has a **controller** that holds its state and a
**view** that displays it. Controller functions let you perform the same
actions from Julia code as from the GUI.

## Controllers own the state

A controller such as [`ResultExplorer`](@ref), [`MaskEditor`](@ref),
[`BatchRunner`](@ref), or [`CalibrationReview`](@ref) stores its state in
[`Observables`](https://juliagizmos.github.io/Observables.jl/stable/):
the current frame, the displayed field, the polygons drawn so far, or the
batch progress. Tool actions are ordinary functions on the
controller: `set_field!`, `close_active!`, `start!`. The controllers live
in `HammerheadGUI.Controllers`, which does not import Makie. You can create
and use a controller on a server without opening a window.

User gestures are controller methods too. When you left-click in the mask
editor, the view calls `click!(me, x, y)`; the decision of whether that
click adds a vertex, selects a polygon, or starts a new one lives in the
controller. To call gesture methods directly, import them from the
submodule: `using HammerheadGUI.Controllers: click!, alt_click!`.

## Views render and forward

The view functions ([`result_explorer`](@ref), [`mask_editor`](@ref), …)
build GLMakie figures whose widgets use the controller's observables.
Moving the frame slider updates the controller; calling `set_frame!` from
code updates an open view. The controller remains available after its
window closes.

Three practical consequences:

- **Drive an open window from the REPL.** Update an observable or call a
  controller function and the figure follows. See
  [the GUI tour](../tutorials/gui_tour.md) for examples.
- **Script a set of actions.** Use controller calls to reproduce an analysis
  setup or drive the same tool without interacting with widgets.
- **Views compose.** [`result_explorer!`](@ref) builds into a
  `GridPosition` of a larger figure; the self-calibration review embeds a
  full result explorer for its disparity maps the same way.

## The boundary to the core package

Controllers use the core API. The mask editor exports a mask made with
[`polygon_mask`](@ref), using the same
`true` = excluded convention described in
[the masking model](masking.md); its "save" writes the image
[`load_mask`](@ref) reads. The batch runner calls
[`run_piv_sequence`](@ref) with its documented progress callback; its
output file is an ordinary JLD2-format Julia data file written by
[`save_results`](@ref). The calibration
review calls [`detect_calibration_grid`](@ref) and
[`calibrate_camera`](@ref). GUI output can therefore be read and processed
with the same functions you use in a script.

HammerheadGUI is a separate package that depends on Hammerhead. The core
package does not require GLMakie, so it can run without a display.
