# # A tour of the graphical user interface (GUI)
#
# Hammerhead's interactive tools live in a companion package,
# **HammerheadGUI**. Two design facts shape how you use them:
#
# - Every tool is a thin shell over the same API this manual documents.
#   What you click is what you would have typed: the mask editor produces
#   the `Bool` matrices [`run_piv`](@ref) takes, the batch runner writes
#   the JLD2-format Julia data files [`load_results`](@ref) reads — the GUI and
#   your scripts mix freely.
# - Every window is driven by a *controller*: a plain-Julia object holding
#   the tool's state as `Observables`. You can construct it yourself,
#   drive it from the REPL, and any open view follows live. (Why it is
#   built this way: [the controller–view split](../explanation/gui.md).)
#
# This tutorial walks one planar workflow end to end — set a scale, mask,
# choose preprocessing, run, explore — driving each tool through its
# controller, which is how the docs build renders real figures; at your
# REPL each `Figure` below opens as an interactive window instead. The
# remaining tools (calibration review, stereo, PTV browsing) follow after
# the walkthrough.
#
# ## A recording to process
#
# The same Lamb–Oseen vortex as [the first tutorial](first_vector_field.md):

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

# The workflow revolves around the batch runner — a parameter form around
# [`run_piv_sequence`](@ref). Seed the [`BatchRunner`](@ref) controller with
# the frames (normally file paths picked with "add frames…"; here our
# in-memory pair) and keep a handle on it — each of the following steps
# hands its output to this form:

using HammerheadGUI

bc = BatchRunner(files = Any[imgA, imgB], uncertainty = true)

# ## Step 1: set the physical scale
#
# [`scale_tool`](@ref) turns a calibration image into a
# [`PhysicalScale`](@ref): click the two endpoints of a feature of known
# physical size and enter the separation. Here the calibration image is a
# synthetic dot grid rendered with [`render_calibration_target`](@ref) — a
# plate of dots at exactly 15 mm spacing seen through a known camera — and
# we "click" the centres of two neighbouring dots through the
# [`ScaleTool`](@ref) controller, exactly what a mouse click does:

θ = deg2rad(10.0)
R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
camC = R' * [0.0, 0.0, -500.0]
K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
cam = PinholeCamera(K, R, -R * camC)
plate = render_calibration_target(cam, (512, 512); spacing = 15.0)

st = ScaleTool(plate)
dot1 = world_to_pixel(cam, (0.0, 0.0, 0.0))     # two neighbouring dots,
dot2 = world_to_pixel(cam, (15.0, 0.0, 0.0))    # 15 mm apart in the world
HammerheadGUI.Controllers.click!(st, dot1[1], dot1[2])
HammerheadGUI.Controllers.click!(st, dot2[1], dot2[2])
set_separation!(st, 15.0)                        # mm between the clicks
st.dt[] = 0.001                                  # 1 ms between frames
st.time_unit[] = "s"
scale_tool(st)

# The status line reports the derived pixel size, and "apply to batch"
# copies the whole scale — pixel size, dt, and unit labels — into the batch
# form ([`apply_scale!`](@ref)); every result the batch produces will carry
# it:

apply_scale!(bc, st)
physical_scale(st)

# ## Step 2: mask what should not correlate
#
# [`mask_editor`](@ref) draws exclusion polygons over a frame: left-click
# adds vertices (inside a polygon it selects instead), right-click closes,
# Backspace/Delete undo and remove. [`add_vertex!`](@ref) and
# [`close_active!`](@ref) are exactly what the view calls when you click:

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

# ## Step 3: choose preprocessing, with a correlation probe
#
# [`preprocess_preview`](@ref) composes the
# [core preprocessing set](../howto/preprocessing.md) into an ordered,
# toggleable pipeline with a live raw/processed comparison. Give the
# [`PreprocessPreview`](@ref) controller the frame's pair partner and it
# also runs a *single-window correlation probe*: click a location on the
# processed image and that window's displacement and peak ratio update live
# as you toggle and tune steps — a direct readout of what each
# preprocessing choice does to the correlation:

pp = PreprocessPreview(imgA; pair = imgB, enabled = [:highpass_filter])
set_step_param!(pp, :highpass_filter, :sigma, 5)
HammerheadGUI.Controllers.click!(pp, 190.0, 128.0)   # probe off the vortex core
preprocess_preview(pp)

#-

probe_summary(pp)

# Toggle a step and the probe recomputes — the displacement barely moves
# but the peak ratio shifts, which is exactly the comparison the probe
# exists for:

enable_step!(pp, :percentile_stretch)
probe_summary(pp)

#-

enable_step!(pp, :percentile_stretch, false)

# "Use in batch" installs the pipeline; from code that is
# [`set_preprocess!`](@ref), which snapshots the steps so later form edits
# cannot affect a running batch:

set_preprocess!(bc, pp)

# ## Step 4: run
#
# The parameter form holds the multi-pass window schedule and the accuracy
# options, or an *effort* preset (`:low`/`:medium`/`:high` — see
# [Choose an effort level](../howto/effort.md)) that replaces the manual
# schedule. [`start!`](@ref) is the run button; in the GUI it runs as a
# cooperative background task so the window stays live, "cancel" stops
# after the pair in flight, and "view results" opens the explorer as soon
# as the first pair finishes, appending later pairs live. Here we run
# synchronously:

fig = batch_runner(bc)
start!(bc; async = false)
fig

# The results are the plain `Vector` of [`PIVResult`](@ref)s any script
# would produce, each carrying the scale from step 1:

bc.results[]

# ## Step 5: explore the results
#
# [`result_explorer`](@ref) browses the batch output — the "view results"
# button opens exactly this. Because the results carry a scale, every
# label, colorbar, and inspection panel reads in physical units:

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
