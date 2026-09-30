# The graphical user interface (GUI) controller–view split

Each HammerheadGUI tool has a **controller** that holds its state and a
**view** that displays it. You can control a view from Julia code or embed it
in a larger figure.

## Controllers own the state

A controller — [`ResultExplorer`](@ref), [`MaskEditor`](@ref),
[`BatchRunner`](@ref), [`CalibrationReview`](@ref) — is a plain Julia
object whose fields are
[`Observables`](https://juliagizmos.github.io/Observables.jl/stable/):
the current frame, the displayed field, the polygons drawn so far, the
batch progress. Everything the tool *does* is an ordinary function on the
controller: `set_field!`, `close_active!`, `start!`. The controllers live
in `HammerheadGUI.Controllers`, which does not import Makie. You can create
and use a controller on a server without opening a window.

User gestures are controller methods too. When you left-click in the mask
editor, the view calls `click!(me, x, y)`; the decision of whether that
click adds a vertex, selects a polygon, or starts a new one lives in the
controller, not in the widget code. (The gesture API is not re-exported at
top level — `using HammerheadGUI.Controllers: click!, alt_click!` when you
want it.)

## Views render and forward

The view functions ([`result_explorer`](@ref), [`mask_editor`](@ref), …)
build GLMakie figures whose widgets are wired to the controller's
observables in both directions: moving the frame slider calls
`set_frame!`, and calling `set_frame!` moves the slider. The view holds no
state of its own — delete the window and the controller is intact;
open two views on one controller and they stay in sync.

Three practical consequences:

- **Drive an open window from the REPL.** Update an observable or call a
  controller function and the figure follows. See
  [the GUI tour](../tutorials/gui_tour.md) for examples.
- **GUI sessions are reproducible.** A sequence of clicks is a sequence of
  controller calls, so an interactive session can be replayed as a script.
- **Views compose.** [`result_explorer!`](@ref) builds into a
  `GridPosition` of a larger figure; the self-calibration review embeds a
  full result explorer for its disparity maps the same way.

## The boundary to the core package

Controllers use the core API. The mask editor exports a mask made with
[`polygon_mask`](@ref), using the same
`true` = excluded convention described in
[the masking model](masking.md); its "save" writes the image
[`load_mask`](@ref) reads. The batch runner *is*
[`run_piv_sequence`](@ref) with its documented progress callback; its
output file is an ordinary JLD2-format Julia data file written by
[`save_results`](@ref). The calibration
review calls [`detect_calibration_grid`](@ref) and
[`calibrate_camera`](@ref). GUI output can therefore be read and processed
with the same functions you use in a script.

HammerheadGUI is a separate package that depends on Hammerhead. The core
package does not require GLMakie, so it can run without a display.
