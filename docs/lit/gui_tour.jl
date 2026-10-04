# # Your first PIV session in the GUI
#
# Turn a short recording into a vector field in the planar PIV window: exclude
# a reflection, attach millimetres and seconds, test one pair, run the batch,
# and measure a profile and a circulation in the result. The recording is a
# synthetic vortex, so you can follow along without downloading data.
#
# Install **HammerheadGUI** with `pkg> add HammerheadGUI`, then start Julia
# with several threads (`julia -t auto`) and open the window:
#
# ```julia
# using HammerheadGUI
# wf = planar_window()
# ```
#
# The window walks through six steps, listed on its left: **Images → Prepare →
# Passes → Test pair → Run → Results**. The viewer on the right follows the
# step you are on. The screenshots below come from the window; each code block
# is the equivalent of the clicks it describes, written against the window's
# workflow controller. `planar_window()` returns that `PlanarWorkflow` when
# the window closes, and `planar_window(wf)` opens one prepared in code.

using Hammerhead
using HammerheadGUI
using HammerheadGUI.Controllers                 # the functions behind the buttons
using HammerheadGUI.Controllers: click!, alt_click!

# ## 1. Add the frames
#
# Particles circulate around the centre of these images. The bright square in
# the upper-left corner stays put: it stands for a reflection, which would
# give misleading displacements. Two exposure pairs make a short recording:

using Hammerhead.SyntheticData, Random
flow(x, y, z, t) = (-0.015 * (y - 128), 0.015 * (x - 128), 0.0)
work = mkpath("gui-example")             # use your own folder if you prefer
paths = String[]
for (k, seed) in enumerate((42, 43))
    a, b, _, _ = generate_synthetic_piv_pair(flow, (256, 256), 1.0;
        particle_density = 0.05, background_noise = 0.01, rng = MersenneTwister(seed))
    peak = max(maximum(a), maximum(b))   # one intensity scale for the pair
    for (j, img) in enumerate((a, b))
        img = img ./ peak
        img[12:60, 12:60] .= 1.0         # the reflection
        path = joinpath(work, "frame_$(lpad(2k - 2 + j, 4, '0')).png")
        Hammerhead.FileIO.save(path, Hammerhead.Gray.(img))
        push!(paths, path)
    end
end

# On the **Images** step, click **Add frames…** and select the four files.
# **Pairing** forms pairs from frames in acquisition order: **Paired** for
# separate A/B exposures (1–2, 3–4), **Chained** for a uniformly sampled
# sequence (1–2, 2–3). The bar below the viewer steps through the pairs and
# switches between frame A and frame B. Particles should shift slightly
# between the two frames, not jump.

wf = PlanarWorkflow(files = paths)                   # Add frames…
frames_summary(wf.frames)

# ![The Images step: four frames form two pairs; the viewer shows frame A of the first pair.](../assets/gui_window/images.png)
#
# The pair shown is the *representative pair*: every preview, probe and test
# uses it, so pick one with the flow features you care about.

# ## 2. Check the preprocessing with the probe
#
# Open **Prepare**. On its **Preprocess** page, add a **Highpass filter** with
# **Add step**, switch the viewer to **Processed**, and click a place in the
# flow. The probe correlates one window of the processed pair at that point:

set_step!(wf, :prepare)
add_step!(wf.prepare.preview, :highpass_filter)      # Add step
wf.prepare.show_processed[] = true                   # Viewer: Processed
canvas_click!(wf, 190, 150)                          # click the image
print(probe_summary(wf.prepare.preview))

# ![Prepare, Preprocess page: a highpass filter, the processed frame, and the probe window in yellow.](../assets/gui_window/prepare_preprocess.png)
#
# A peak ratio well above 1.5 means the window finds one clear match. Probe a
# few places, especially dim or fast regions; the preview runs exactly the
# preprocessing the batch will run.

# ## 3. Exclude the reflection
#
# On the **Mask** page, click just outside the square's four corners, then
# right-click to close the polygon. The shaded area is excluded:

set_prepare_page!(wf, :mask)
for (x, y) in ((8, 8), (64, 8), (64, 64), (8, 64))
    canvas_click!(wf, x, y)                          # click a vertex
end
canvas_alt_click!(wf)                                # right-click closes
count(wf.mask[])                                     # excluded pixels

# ![Prepare, Mask page: the polygon around the reflection, shaded as excluded.](../assets/gui_window/prepare_mask.png)
#
# Backspace undoes a vertex and Escape cancels the polygon. A click inside a
# finished polygon selects it, and Delete removes it. A mask only removes
# regions; whether the remaining vectors are accurate is the job of the test
# below.

# ## 4. Attach millimetres and seconds
#
# On the **Scale** page, type the pixel size and the time between the paired
# exposures. Here that is 0.02 mm per pixel and 0.001 s; on your recording, use
# your measured values. With a ruler or calibration target in view, click two
# points of known separation instead and type their **Distance**.

set_prepare_page!(wf, :scale)
edit_scale!(wf, :pixel_size, "0.02")
edit_scale!(wf, :length_unit, "mm")
edit_scale!(wf, :dt, "0.001")
edit_scale!(wf, :time_unit, "s")
wf.scale[]

# The vectors are still computed in pixels; the scale converts what you see in
# the results, as described in [Scale results to physical units](../howto/scaling.md).

# ## 5. Choose the passes and test one pair
#
# On **Passes**, click **Medium**. The preset fills the pass table for the
# frame size, and the viewer outlines each window size against the particles.
# A first window at least four times the largest displacement keeps most
# particle pairs inside the window.

fill_preset!(wf.passes, :medium)                     # Preset: Medium
passes_summary(wf.passes)

# ![Passes: the medium preset's pass table and its window sizes outlined on the particles.](../assets/gui_window/passes.png)
#
# On **Test pair**, click **Test pair 1**. The test runs the same call as the
# batch on the representative pair, so its summary predicts the run:

test_pair!(wf; spawn = false)                        # Test pair 1
print(join(summary_lines(test_summary(wf.test)), "\n"))

# ![Test pair: the summary beside the vectors, valid in blue; the masked corner has none.](../assets/gui_window/test_pair.png)
#
# Change a setting and the step rail marks the test as out of date until you
# test again.

# ## 6. Run the recording
#
# On **Run**, choose an output file with **Browse…** and click **Run 2 pairs**.
# Results are written as each pair finishes, together with the settings that
# produced them. **Cancel** stops after the pair in progress and keeps the
# finished ones. In the window the run happens in the background (the viewer
# shows the latest pair); here it runs before the next line:

wf.run.output_path[] = joinpath(work, "vectors.jld2")
start_run!(wf; spawn = false)                        # Run 2 pairs
wf.run.status[]

# ## 7. Measure a profile and a circulation
#
# **Results** opens the output file. Choose a field and step through the
# pairs; with the **Inspect** tool, a click on a vector reads its components and
# status. Axes and colours use millimetres and seconds now.
#
# Choose the **Profile** tool and click two points across the vortex. The
# panel under the field plots u, v and |V| along the line: v changes sign
# through the centre and grows linearly, as in solid-body rotation.

ex = wf.explorer[]
r = current_result(ex)
xmid, ymid = (first(r.x) + last(r.x)) / 2, (first(r.y) + last(r.y)) / 2
set_tool!(ex, :profile)                              # Tool: Profile
click!(ex, xmid - 2, ymid)                           # two clicks on the viewer
click!(ex, xmid + 2, ymid)
tool_summary(ex)

# ![Results with the Profile tool: the line across the vortex and the velocity along it.](../assets/gui_window/results_profile.png)
#
# Choose **Circulation**, click the corners of a contour around the centre,
# and right-click to close it. The summary gives Γ from the line integral of
# the velocity and from the vorticity enclosed:

set_tool!(ex, :circulation)                          # Tool: Circulation
for (x, y) in ((xmid - 1, ymid - 1), (xmid + 1, ymid - 1), (xmid + 1, ymid + 1), (xmid - 1, ymid + 1))
    click!(ex, x, y)
end
alt_click!(ex)                                       # right-click closes
print(tool_summary(ex))

# This vortex rotates at 0.015 rad per frame, so its vorticity is 0.03 per
# frame: 30 s⁻¹ with the scale above. Around this 2 mm square, both estimates
# come close to 30 s⁻¹ × 4 mm² = 120 mm²/s. Check flagged vectors and the mask
# before trusting derived quantities, because derivatives amplify local errors.

# ## 8. Keep the result and the settings
#
# The results file already holds the settings that produced it. **Save
# settings…** writes them to a separate recipe file as well:

settings_path = joinpath(work, "vortex-settings.jld2")
save_settings(wf, settings_path)                     # Save settings…
load_recipe(settings_path) == load_recipe(wf.run.output_path[])

# In a later session, **Open settings…** reads either file back, and **Use
# these settings** on the Results step does the same for the open results. Add
# the next recording's frames, test a pair, and run:

next_wf = PlanarWorkflow()
load_settings!(next_wf, wf.run.output_path[])        # Open settings…
next_wf.status[]

# To browse a results file without the workflow, open the standalone explorer
# with `result_explorer("gui-example/vectors.jld2"; lazy = true)`.
#
# You have made a masked vector field in physical units, tested it before the
# run, measured a profile and a circulation, and kept the settings for the
# next recording. [Analyze an image pair in the GUI](../howto/gui.md) covers
# the rest of the window, and [Save settings and reuse them](../howto/recipes.md)
# runs the same settings from a script.
