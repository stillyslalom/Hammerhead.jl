# # A tour of the graphical user interface (GUI)
#
# This tutorial takes a pair of particle images to an explorable vector
# field in three form actions, then improves that first result the way a
# real analysis grows: attach a physical scale, mask what should not
# correlate, tune preprocessing against a live correlation probe, re-run,
# and analyze the flow. Calibration review, stereo, and particle tracking
# make a brief appearance at the end.
#
# The GUI lives in a companion package, **HammerheadGUI**, so the core
# package never carries plotting dependencies. Install and launch it like
# any Julia package:
#
# ```julia
# pkg> add HammerheadGUI        # `]` at the julia> prompt opens pkg>
#
# julia> using HammerheadGUI
# julia> batch_runner()         # the front door: an empty batch form
# ```
#
# One reading note. Every window is a thin shell over a plain-Julia
# *controller*, and clicking a widget calls the same function the code
# below calls — the docs build cannot click, so where you would use the
# mouse this tutorial drives the controller instead. At your REPL every
# figure in this page opens as a live interactive window, and you can
# work with the mouse, the code, or both at once. (Why it is built this
# way, and what that lets you automate:
# [the controller–view split](../explanation/gui.md).)
#
# ## First vectors
#
# Your data is a folder of image files, added to the batch form with its
# "add frames…" file picker. The docs build fabricates a recording
# instead — a Lamb–Oseen vortex imaged as realistic particle images. Skip
# over this block; it merely stands in for your camera:

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

# The whole first analysis is three actions on the [`BatchRunner`](@ref)
# form, a front end to [`run_piv_sequence`](@ref): add the frames, pick an
# effort preset from the "effort" menu (`:low`/`:medium`/`:high` — see
# [Choose an effort level](../howto/effort.md)), press "run":

using HammerheadGUI

bc = BatchRunner(files = Any[imgA, imgB])   # "add frames…"
set_effort!(bc, :medium)                    # the "effort" menu
start!(bc; async = false)                   # the "run" button
bc.status[]

# In the GUI the run executes as a background task, so the window stays
# live: a progress bar counts pairs, "cancel" stops after the pair in
# flight (keeping the finished ones), and "view results" lights up as soon
# as the first pair completes, appending later pairs to the open explorer
# as they finish. The docs build runs synchronously and opens
# [`result_explorer`](@ref) on the output — exactly what "view results"
# does:

ex = ResultExplorer(bc.results[])
fig = result_explorer(ex)

# That is a working PIV measurement: the displacement-magnitude field
# under the vector arrows, a field menu of components and diagnostics, a
# frame slider for longer recordings, and click-to-inspect on every
# vector.
#
# It can be better, though. The axes read pixels and the displacements
# pixels-per-frame; the frame edges correlate content that belongs to no
# flow; and the defaults know nothing about your imaging conditions. The
# rest of the walkthrough upgrades the same batch form one step at a
# time — which is how a real analysis usually iterates.
#
# ## Improve it: physical units
#
# [`scale_tool`](@ref) turns a calibration image into a
# [`PhysicalScale`](@ref): click the two endpoints of a feature of known
# physical size and type in the separation. Normally you would open a
# photographed target with `scale_tool("plate.png")`; the docs build
# renders a synthetic plate — dots at exactly 15 mm spacing seen through a
# known camera ([`render_calibration_target`](@ref)). Skip this block too,
# it only fabricates that photo:

θ = deg2rad(10.0)
R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
camC = R' * [0.0, 0.0, -500.0]
K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
cam = PinholeCamera(K, R, -R * camC)
plate = render_calibration_target(cam, (512, 512); spacing = 15.0)

# In the window: click the centres of two neighbouring dots, enter the
# separation and units, and read the derived pixel size off the status
# line. `Controllers.click!` is literally the function a mouse click
# invokes:

st = ScaleTool(plate)
dot1 = world_to_pixel(cam, (0.0, 0.0, 0.0))     # two neighbouring dots,
dot2 = world_to_pixel(cam, (15.0, 0.0, 0.0))    # 15 mm apart in the world
HammerheadGUI.Controllers.click!(st, dot1[1], dot1[2])
HammerheadGUI.Controllers.click!(st, dot2[1], dot2[2])
set_separation!(st, 15.0)                        # mm between the clicks
st.dt[] = 0.001                                  # 1 ms between frames
st.time_unit[] = "s"
scale_tool(st)

# "Apply to batch" copies the whole scale — pixel size, dt, and unit
# labels — into the batch form ([`apply_scale!`](@ref)); every result the
# batch produces from now on will carry it:

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

# The editor exports the package's mask convention — image-sized `Bool`,
# `true` = excluded ([the masking model](../explanation/masking.md)) — and
# the batch form takes it directly ("load mask…" reads a saved mask image
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
# live as you toggle and tune steps — a direct readout of what each
# preprocessing choice does to the measurement itself, not just to how
# the image looks:

pp = PreprocessPreview(imgA; pair = imgB, enabled = [:highpass_filter])
set_step_param!(pp, :highpass_filter, :sigma, 5)
HammerheadGUI.Controllers.click!(pp, 190.0, 128.0)   # probe off the vortex core
preprocess_preview(pp)

#-

probe_summary(pp)

# Toggle a step and the probe recomputes — here the displacement barely
# moves but the peak ratio shifts, which is exactly the comparison the
# probe exists for:

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
# schedule and the accuracy options in their place — here the default
# 64/32/32 px schedule with the uncertainty estimator switched on:

set_effort!(bc, :custom)     # back to the manual parameter form
bc.uncertainty[] = true      # the "uncertainty" toggle
fig = batch_runner(bc)
start!(bc; async = false)
fig

# The results are the plain `Vector` of [`PIVResult`](@ref)s any script
# would produce, each carrying the physical scale:

bc.results[]

# ## Explore and analyze
#
# Open the explorer again — because the results now carry a scale, every
# axis, colorbar, and inspection panel reads in physical units:

ex = ResultExplorer(bc.results[])
fig = result_explorer(ex)

# The field menu holds the components and diagnostics *plus the derived
# fields* — vorticity, divergence, strain rate, swirling strength, and Q,
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

# The *tool* menu adds interactive analysis on planar results. `profile`
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

# The fiducial markers are outlined in cyan, the slider steps through the
# planes, and the camera-model menu refits on the spot; what residuals to
# expect from a physical plate (these ~0.5 px are the plate, not the
# detection) is covered in the [stereo-rig how-to](../howto/stereo_rig.md).
# Its sibling [`selfcal_review`](@ref) browses a
# [`SelfCalibrationReport`](@ref) with the per-pass disparity maps in an
# embedded explorer.
#
# **Stereo batch.** [`stereo_calibration`](@ref) embeds two of these
# reviews side by side and builds the shared-grid [`ImageDewarper`](@ref)
# pair ([`build_dewarpers`](@ref)); [`stereo_batch_runner`](@ref) then
# drives [`run_piv_stereo_sequence`](@ref) over two synchronized frame
# lists — same live progress, live results, and incremental output as the
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
# - Why every tool is a controller + view pair, and what that lets you
#   automate: [The GUI's controller–view split](../explanation/gui.md).
# - The full API: [GUI reference](../reference/gui.md).
