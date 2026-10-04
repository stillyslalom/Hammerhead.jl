# The graphical user interface (GUI) controller–view split

HammerheadGUI keeps everything the window knows and does in plain Julia
**controllers**. The Qt window and its Makie **canvases** only display
controller state and forward the user's input. Every button and every click on
the image is a controller function, so a script can perform the same steps,
and the tests check them without a display.

## Three layers

- **Controllers** (`HammerheadGUI.Controllers`) hold the state in
  [`Observables`](https://juliagizmos.github.io/Observables.jl/stable/) and
  implement the actions. [`PlanarWorkflow`](@ref) owns one controller per
  step: `FrameSet` (frames and pairing), `PrepareState` (the Prepare pages and
  their editors: [`PreprocessPreview`](@ref), [`MaskEditor`](@ref),
  [`ROIEditor`](@ref), [`ScaleTool`](@ref)), `PassesEditor`, `PairTest`,
  `RunState`, and a [`ResultExplorer`](@ref) for the results.
  [`StereoWorkflow`](@ref) shares these and adds a frame set per camera,
  linked so both cameras keep one pairing rule and one representative pair,
  and the Calibration step's [`StereoCalibration`](@ref). The module does
  not import Makie or Qt; a test checks this boundary.
- **Canvases** are Makie figures that draw the controllers: the frame, mask,
  region, window outlines, vectors, editing overlays, and the result field
  with its profile panel. The stereo canvas also draws calibration plates
  with their reprojection residuals, and shows the later steps on the
  dewarped grid, so a click lands in the grid coordinates the mask and probe
  use.
- **The Qt shell** lays out the step pages in QML. It mirrors the controller
  values QML displays into one property map and calls controller functions
  when a control changes.

The workflow is the single source of the settings. Its `preprocessing`,
`mask`, `roi` and `scale` fields are what the Prepare editors edit, and
`workflow_recipe(wf)` turns them into a core [`PIVRecipe`](@ref). Opening
settings writes those fields, and the editors follow. Opening a recipe and
asking for it back without edits returns an equal recipe, including settings
the window does not show.

## Gestures are controller functions

A click on the image becomes `canvas_click!(wf, x, y)` in image coordinates;
a right-click becomes `canvas_alt_click!(wf)`, and keys such as Backspace,
Escape and Delete become `canvas_key!(wf, key)`. The controller decides what
the gesture means from the step and the open Prepare page: place the
correlation probe, add a mask vertex or select a polygon, set a region
corner, or set a scale point. On the Results step, clicks go to the
explorer's tool (inspect, profile, or circulation). A gesture the
controller does not use stays with the viewer, so dragging still zooms and
pans. This is why the [GUI tour](../tutorials/gui_tour.md) can show the code
for each click: it is the same call.

## Canvases never create plots after display

A Qt canvas has an OpenGL context only while Qt renders it. GLMakie creates or
frees GPU objects as soon as a plot is added to or removed from a displayed
figure. Outside Qt's render pass that would happen without a context. Each
canvas therefore creates all of its plots when it is built: overlays that are
empty hold a placeholder of `NaN` points. Afterwards only the plots' data,
colours and visibility change. The profile panel under the result field is
an axis that exists from the start; outside the profile tool its layout row
collapses and it is hidden.

## Work happens off the window's thread

Qt's event loop runs on Julia's main thread while the window is open, so a
callback from the window must return quickly. Loading the representative
pair, the preprocessing preview, the correlation probe, the background
estimate, calibration fits, dewarp-grid builds, self-calibration, the test
pair, and the batch run all run on worker tasks. A job
captures its inputs when it starts and hands its result back to the window,
which applies it between frames. A newer request supersedes an older one,
so after rapid edits the viewer shows the latest settings. This is why the
window needs Julia started with several threads (`julia -t auto`). Without a
window the same controllers run each job before returning, which is what
scripts and tests expect.

## The boundary to the core package

Controllers use the core API. A test pair and a run both call
[`apply_recipe`](@ref) with the current recipe, so the test predicts the
batch. A run's output is an ordinary results file written by
[`save_results`](@ref), with the recipe stored beside the results, so
[`load_recipe`](@ref) recovers the settings from either a settings file or
a results file. The preprocessing preview applies `recipe_preprocess` to the
same `PreprocessStep`s the batch uses. The mask editor exports its polygons
with [`polygon_mask`](@ref), using the `true` = excluded convention of
[the masking model](masking.md), and writes mask images that
[`load_mask`](@ref) reads. The stereo calibration calls
[`detect_calibration_grid`](@ref), [`calibrate_camera`](@ref),
[`common_dewarp_grid`](@ref) and [`self_calibrate`](@ref), and a stereo test
or run calls the stereo `apply_recipe` with the two cameras' dewarpers. A
recipe holds processing settings only, so the calibration stays with the
window session.

HammerheadGUI is a separate package that depends on Hammerhead. The core
package does not depend on Qt or GLMakie, so it runs without a display.
