# # A tour of the graphical user interface (GUI)
#
# Use the batch form to analyze a pair of particle images, then add a
# physical scale, define a mask, adjust preprocessing, and inspect
# the resulting field. The final section shows where to find calibration,
# stereo, and particle-tracking tools.
#
# Install the companion **HammerheadGUI** package and open its batch form:
#
# ```julia
# pkg> add HammerheadGUI        # `]` at the julia> prompt opens pkg>
#
# julia> using HammerheadGUI
# julia> batch_runner()         # open an empty batch form
# ```
#
# The examples call each window's Julia *controller*, which performs the
# same actions as its widgets. At the REPL, the figures open as interactive
# windows, so you can follow the steps with the mouse or run the code.
# See [the controller–view split](../explanation/gui.md) for details.
#
# ## First vectors
#
# Add your image files to the batch form with "add frames…". For this
# walkthrough, generate a pair depicting a Lamb–Oseen vortex:

using Hammerhead
using Hammerhead.SyntheticData
using Random

center, rc, Γ = (128.0, 128.0), 40.0, 1200.0
function flow(x, y, z, t)
    dx, dy = x - center[1], y - center[2]
    r² = dx^2 + dy^2
    k = r² < 1e-9 ? Γ / (2π * rc^2) : Γ / (2π * r²) * (1 - exp(-r² / rc^2))
    return (-k * dy, k * dx, 0.0)
end

rng = MersenneTwister(42)
imgA, imgB, _, _ = generate_synthetic_piv_pair(flow, (256, 256), 1.0;
    particle_density = 0.05, background_noise = 0.03,
    z_range = (-1.0, 1.0), rng)

# On the [`BatchRunner`](@ref) form, add the frames, choose an effort preset
# (`:low`/`:medium`/`:high`; see
# [Choose an effort level](../howto/effort.md)), then press "run":

using HammerheadGUI

bc = BatchRunner(files = Any[imgA, imgB])   # "add frames…"
set_effort!(bc, :medium)                    # the "effort" menu
start!(bc; async = false)                   # the "run" button
bc.status[]

# The GUI runs the batch in the background. A progress bar counts pairs;
# "cancel" stops after the current pair and keeps completed results. "View
# results" becomes available after the first pair and adds later results
# as they finish. Open the same explorer from code with
# [`result_explorer`](@ref):

ex = ResultExplorer(bc.results[])
fig = result_explorer(ex)

# The explorer shows displacement magnitude under the vector arrows, a
# field menu with components and diagnostics, a
# frame slider for longer recordings, and click-to-inspect on every
# vector.
#
# This result uses pixel coordinates and displacement per frame. Add a
# physical scale to display velocity units, mask an excluded region, and
# choose preprocessing that improves the correlation at a test location.
#
# ## Improve it: physical units
#
# [`scale_tool`](@ref) turns a calibration image into a
# [`PhysicalScale`](@ref): click the two endpoints of a feature of known
# physical size and enter its separation. You can open a photographed target
# with `scale_tool("plate.png")`. This example renders a plate with dots
# 15 mm apart ([`render_calibration_target`](@ref)):

θ = deg2rad(10.0)
R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
camC = R' * [0.0, 0.0, -500.0]
K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
cam = PinholeCamera(K, R, -R * camC)
plate = render_calibration_target(cam, (512, 512); spacing = 15.0)

# In the window: click the centres of two neighbouring dots, enter the
# separation and units, and read the derived pixel size off the status
# line. The code below uses `Controllers.click!` for those clicks:

st = ScaleTool(plate)
dot1 = world_to_pixel(cam, (0.0, 0.0, 0.0))     # two neighbouring dots,
dot2 = world_to_pixel(cam, (15.0, 0.0, 0.0))    # 15 mm apart in the world
HammerheadGUI.Controllers.click!(st, dot1[1], dot1[2])
HammerheadGUI.Controllers.click!(st, dot2[1], dot2[2])
set_separation!(st, 15.0)                        # mm between the clicks
st.dt[] = 0.001                                  # 1 ms between frames
st.time_unit[] = "s"
scale_tool(st)

# "Apply to batch" copies the pixel size, dt, and unit labels into the batch
# form ([`apply_scale!`](@ref)). Every result the batch produces from now on
# will carry this scale:

apply_scale!(bc, st)
physical_scale(st)

# ## Improve it: mask what should not correlate
#
# [`mask_editor`](@ref) draws exclusion polygons over a frame: left-click
# adds vertices (a click inside an existing polygon selects it instead),
# right-click closes the polygon, and buttons undo, delete, and grow or
# shrink the mask. In code those clicks are [`add_vertex!`](@ref) and
# [`close_active!`](@ref):

me = MaskEditor(imgA)
fig = mask_editor(me)
for (x, y) in ((10, 10), (60, 10), (60, 60), (10, 60))
    add_vertex!(me, x, y)
end
close_active!(me)
me.show_mask[] = true   # the "show mask" toggle
fig

# The editor exports an image-sized `Bool` mask, where `true` means excluded
# ([the masking model](../explanation/masking.md)).
# The batch form takes the mask directly ("load mask…" reads a saved mask image
# instead):

bc.mask[] = polygon_mask(me)
count(bc.mask[])

# ## Improve it: preprocessing, tuned with a correlation probe
#
# [`preprocess_preview`](@ref) composes the
# [core preprocessing set](../howto/preprocessing.md) into an ordered,
# toggleable pipeline with a live raw/processed comparison. Give the
# [`PreprocessPreview`](@ref) controller the frame's pair partner and it
# also runs a *single-window correlation probe*: click a location on the
# processed image and that window's displacement and peak ratio update
# live as you toggle and tune steps. Use those values to assess how a
# preprocessing change affects the measured displacement:

pp = PreprocessPreview(imgA; pair = imgB, enabled = [:highpass_filter])
set_step_param!(pp, :highpass_filter, :sigma, 5)
HammerheadGUI.Controllers.click!(pp, 190.0, 128.0)   # probe off the vortex core
preprocess_preview(pp)

#-

probe_summary(pp)

# Toggle a step to recompute the probe. Here the displacement changes little,
# while the peak ratio shifts:

enable_step!(pp, :percentile_stretch)
probe_summary(pp)

#-

enable_step!(pp, :percentile_stretch, false)

# "Use in batch" installs the pipeline; from code that is
# [`set_preprocess!`](@ref), which snapshots the steps so later preview
# edits cannot affect a running batch:

set_preprocess!(bc, pp)

# ## Re-run with the full setup
#
# Back on the batch form: scale, mask, and preprocessing are now set, and
# the "custom" effort setting exposes the manual multi-pass window
# schedule and accuracy options. Use the default
# 64/32/32 px schedule with the uncertainty estimator switched on:

set_effort!(bc, :custom)     # back to the manual parameter form
bc.uncertainty[] = true      # the "uncertainty" toggle
fig = batch_runner(bc)
start!(bc; async = false)
fig

# The batch returns a `Vector` of [`PIVResult`](@ref)s, each carrying the
# physical scale:

bc.results[]

# ## Explore and analyze
#
# Open the explorer again. Because the results now carry a scale, every
# axis, colorbar, and inspection panel reads in physical units:

ex = ResultExplorer(bc.results[])
fig = result_explorer(ex)

# The field menu holds the components and diagnostics *plus the derived
# fields*: vorticity, divergence, strain rate, swirling strength, and Q,
# computed via [`flow_derivatives`](@ref) and labelled `1/s` here. The
# colorbar range defaults to a robust 2–98% percentile band over the valid
# vectors ([`color_limits`](@ref)) so outliers cannot wash it out, with
# manual overrides in the "color range" group. Switch to vorticity and
# inspect a vector near the core (the axes are in millimetres now):

set_field!(ex, :vorticity)
w = last(current_result(ex).x)      # field extent, mm
select_nearest!(ex, w / 2, w / 2)
describe_selection(ex)

#-

fig

# The *tool* menu adds analysis on planar results. `profile`
# samples u, v, and |V| along a two-click line ([`extract_profile`](@ref)),
# drawn in a side panel; `circulation` accumulates a contour (right-click
# closes it) and evaluates [`circulation`](@ref) with both the
# line-integral and vorticity-area estimators:

set_tool!(ex, :profile)
HammerheadGUI.Controllers.click!(ex, 0.1w, 0.5w)
HammerheadGUI.Controllers.click!(ex, 0.9w, 0.5w)
fig

#-

set_tool!(ex, :circulation)
for (px, py) in ((0.35, 0.35), (0.75, 0.35), (0.75, 0.75), (0.35, 0.75))
    HammerheadGUI.Controllers.click!(ex, px * w, py * w)   # clear of the mask
end
HammerheadGUI.Controllers.alt_click!(ex)
tool_summary(ex)

# ## Beyond the planar workflow
#
# **Calibration review.** [`calibration_review`](@ref) runs the dot-grid
# detection on real plate images, fits a camera, and shows each plate with
# its dots colored by reprojection error. On the Particle Image Velocimetry
# (PIV) Challenge case-4E plates from the
# [real stereo tutorial](stereo_real.md):

dir = joinpath(pkgdir(Hammerhead), "test", "reference_images", "E")
plates = [load_image(joinpath(dir, "E_camera_1_z_$k.png")) for k in (1, 4, 7)]

calibration_review(plates, [-3.0, 0.0, 3.0]; spacing = 15.0,
    two_level = true, level_separation = 3.0, origin_offset = (30.0, 7.5))

# The fiducial markers appear in cyan. Use the slider to inspect each plane
# and the camera-model menu to refit. For guidance on physical-plate
# residuals, see the [stereo-rig how-to](../howto/stereo_rig.md).
# Its sibling [`selfcal_review`](@ref) browses a
# [`SelfCalibrationReport`](@ref) with the per-pass disparity maps in an
# embedded explorer.
#
# **Stereo batch.** [`stereo_calibration`](@ref) embeds two of these
# reviews side by side and builds the shared-grid [`ImageDewarper`](@ref)
# pair ([`build_dewarpers`](@ref)); [`stereo_batch_runner`](@ref) then
# drives [`run_piv_stereo_sequence`](@ref) over two synchronized frame
# lists. It shows live progress, results, and incremental output like the
# planar form. See [the GUI how-to](../howto/gui.md) for the workflow.
#
# **Scattered results.** The explorer browses all four persisted result
# types, mixed sequences included. A [`PTVResult`](@ref) draws as a colored
# particle scatter with its scattered field menu:

ptv = run_ptv(imgA, imgB)
result_explorer(ptv)

# ## Where to go next
#
# - Task recipes — masks from files, batch output, live viewing, stereo:
#   [Work interactively with the GUI](../howto/gui.md).
# - For controller functions behind the widgets, see
#   [The GUI's controller–view split](../explanation/gui.md).
# - The full API: [GUI reference](../reference/gui.md).
