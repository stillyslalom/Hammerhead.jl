# The GUI's controller–view split

Every HammerheadGUI tool is two pieces with a strict boundary between
them: a **controller** that owns all state and logic, and a **view** that
renders it. Knowing the boundary is what turns the GUI from a set of
windows into something you can script, test, and embed.

## Controllers own the state

A controller — [`ResultExplorer`](@ref), [`MaskEditor`](@ref),
[`BatchRunner`](@ref), [`CalibrationReview`](@ref) — is a plain Julia
object whose fields are
[`Observables`](https://juliagizmos.github.io/Observables.jl/stable/):
the current frame, the displayed field, the polygons drawn so far, the
batch progress. Everything the tool *does* is an ordinary function on the
controller: `set_field!`, `close_active!`, `start!`. The controllers live
in a `HammerheadGUI.Controllers` submodule that never imports Makie, so
none of this needs a GL context — constructing a `MaskEditor`, feeding it
click gestures, and exporting the mask works headless, on a server, or in
a test suite.

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

- **Any open window can be driven from the REPL.** Update an observable or
  call a controller function and the figure follows —
  [the GUI tour](../tutorials/gui_tour.md) is rendered exactly this way,
  and the same technique automates screenshots or demo recordings.
- **GUI sessions are reproducible.** A sequence of clicks is a sequence of
  controller calls, so an interactive session can be replayed as a script.
- **Views compose.** [`result_explorer!`](@ref) builds into a
  `GridPosition` of a larger figure; the self-calibration review embeds a
  full result explorer for its disparity maps the same way.

## The boundary to the core package

The controllers speak the core API and nothing else. The mask editor's
export *is* [`polygon_mask`](@ref) — the same rasterization, the same
`true` = excluded convention described in
[the masking model](masking.md); its "save" writes the image
[`load_mask`](@ref) reads. The batch runner *is*
[`run_piv_sequence`](@ref) with its documented progress callback; its
output file is an ordinary [`save_results`](@ref) JLD2. The calibration
review calls [`detect_calibration_grid`](@ref) and
[`calibrate_camera`](@ref) verbatim. Nothing you produce in the GUI is in
a GUI-only format, and any GUI workflow has a line-for-line scripted
equivalent.

The dependency boundary mirrors the conceptual one: Hammerhead never
depends on the GUI or on GLMakie — HammerheadGUI is a separate package
that depends on Hammerhead, so headless installations (clusters, CI)
never pay for it.
