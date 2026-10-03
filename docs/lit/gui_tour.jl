# # Your first PIV session in the GUI
#
# Turn two particle images into a vector field, exclude a reflection, and save
# both the result and the settings needed to repeat it. This example uses a
# synthetic vortex, so you can follow along without downloading a recording.
#
# Install **HammerheadGUI** with `pkg> add HammerheadGUI`, then load it:

using HammerheadGUI
using Hammerhead

# The batch form, mask editor, and result explorer are **separate windows**.
# The screenshots below show the real tools. The short code blocks make the
# same selections for this worked example; you can use their buttons instead.

const GUI = HammerheadGUI.GLMakie #hide
tour_assets = joinpath(pkgdir(Hammerhead), "docs", "build", "assets", "gui") #hide
mkpath(tour_assets) #hide
function tour_picture(fig, name) #hide
    screen = GUI.Screen(fig.scene; visible=false, start_renderloop=false) #hide
    try #hide
        for _ in 1:(length(filter(b -> b isa GUI.Toggle, fig.content)) + 1) #hide
            tick = GUI.events(fig).tick[] #hide
            GUI.events(fig).tick[] = GUI.Makie.Tick(GUI.Makie.UnknownTickState, tick.count+1, tick.time+1., 1.) #hide
        end #hide
        Hammerhead.FileIO.save(joinpath(tour_assets, name*".png"), copy(GUI.colorbuffer(screen))) #hide
    finally #hide
        GUI.destroy!(screen) #hide
    end #hide
    nothing #hide
end #hide
nothing #hide

# ## 1. Start with two exposures
#
# Particles move around the center of this pair. The bright square stays put:
# it represents a reflection that would give misleading displacement measurements.

using Hammerhead.SyntheticData, Random
flow(x,y,z,t) = (-0.015*(y-128), 0.015*(x-128), 0.0)
imgA, imgB, _, _ = generate_synthetic_piv_pair(flow, (256,256), 1.0;
    particle_density=0.05, background_noise=0.03, rng=MersenneTwister(42))
peak = max(maximum(imgA), maximum(imgB))   # same PNG intensity scale for both
imgA ./= peak
imgB ./= peak
imgA[12:60,12:60] .= 1.0
imgB[12:60,12:60] .= 1.0

pair_figure = GUI.Figure(size=(900,400)) #hide
for (i, image) in enumerate((imgA, imgB)) #hide
    axis = GUI.Axis(pair_figure[1,i]; title=i==1 ? "Exposure A" : "Exposure B", #hide
        xlabel="x (px)", ylabel="y (px)", aspect=GUI.DataAspect(), yreversed=true) #hide
    GUI.heatmap!(axis, 1:256, 1:256, permutedims(image); colormap=:grays, colorrange=(0,1)) #hide
end #hide
tour_picture(pair_figure, "pair") #hide

# ![Two particle exposures with a stationary bright reflection in the upper-left corner.](../assets/gui/pair.png)
#
# Save the example images so the file-based workflow can use them:

work = mkpath("gui-example")   # use your own output folder if you prefer
paths = [joinpath(work, "frame-A.png"), joinpath(work, "frame-B.png")]
for (path, image) in zip(paths, (imgA, imgB))
    Hammerhead.FileIO.save(path, Hammerhead.Gray.(image))
end
batch = BatchRunner(files=paths)
nothing #hide

# With your own images, open `batch_runner()` and use **add frames…** to select
# them in A/B order. For a longer recording, choose paired exposures
# (`1–2, 3–4`) or consecutive frames (`1–2, 2–3`).

# ## 2. Exclude the reflection
#
# Open `mask_editor(paths[1])`. Left-click just outside the square's four corners,
# then right-click to close the polygon. Turn on **show mask**: red means excluded.
# These calls draw the same boundary:

mask = MaskEditor(paths[1])
for (x, y) in ((9,9), (63,9), (63,63), (9,63))
    add_vertex!(mask, x, y)
end
close_active!(mask)
mask.show_mask[] = true
tour_picture(mask_editor(mask; size=(900,600)), "mask") #hide

# ![The mask editor excludes the bright reflection with a red polygon.](../assets/gui/mask.png)
#
# Click **save mask…**, then **load mask…** in the batch window. Here we make
# that hand-off through the same mask image format:

mask_path = joinpath(work, "reflection-mask.png")
save_mask(mask, mask_path)
batch.mask[] = load_mask(mask_path)
nothing #hide

# A mask excludes image regions. It does not decide whether the remaining
# vectors are accurate; we will inspect those after processing.

# ## 3. Choose the settings and units
#
# Choose **medium** effort for this first run. The preset selects a multi-pass
# window schedule for you. Choose an output file so the result survives closing
# the window.
#
# For this example, declare a pixel size of **0.02 mm** and an exposure delay of
# **0.001 s**. On your recording, use your measured scale and the delay between
# the paired exposures. Typing a unit label does not convert the numeric factor.

set_effort!(batch, :medium)
set_scale!(batch; pixel_size=0.02, dt=0.001, length_unit="mm", time_unit="s")
batch.output_path[] = joinpath(work, "vectors.jld2")
tour_picture(batch_runner(batch; size=(960,760)), "batch") #hide

# ![The batch form is ready to process one masked image pair with physical units.](../assets/gui/batch.png)
#
# Need a smaller region? **edit ROI…** opens a rectangle editor. Need to condition
# the images? **preprocess…** opens a raw/processed comparison. Both have an
# explicit **apply/use in batch** action. Leave them unchanged for this first run.

# ## 4. Run, then inspect the vectors
#
# Press **run**. The progress count reaches `1 / 1`; **view results** opens
# another window. For this tutorial, wait for the same run through code:

start!(batch; async=false)
batch.status[]

# The arrows should circulate around the center, with the reflection region
# omitted. The axes and colorbar now use the scale entered above. Click a vector
# to read its components and status; red flagged vectors deserve a closer look.

explorer = ResultExplorer(batch.results[])
select_nearest!(explorer, 3.5, 2.5)
tour_picture(result_explorer(explorer; size=(1100,750)), "explorer") #hide

# ![The result explorer shows the vortex in millimetres, with a selected vector's measurements beside it.](../assets/gui/explorer.png)
#
# Try **u** or **v** in the field menu. To look for rotation, choose **vorticity**;
# check masks and flagged vectors before interpreting small features, because
# derivatives amplify local errors. For a longer recording, the frame slider
# moves through the results.
#
# Use **cancel** during a batch to stop after the current pair. Finished pairs
# remain available. For a resumable long run, use the separate
# [checkpoint workflow](../howto/gui_checkpoints.md).

# ## 5. Keep the result and the recipe
#
# The result is already in `vectors.jld2`. Reopen it with:

reopened = ResultExplorer(batch.output_path[]; lazy=true)
nframes(reopened)

# Saving a **recipe** is a different task: it keeps the image pairs and settings
# for another run. Click **saved experiments…** in the batch window, then
# **snapshot batch** and **save experiment…**. The equivalent code is:

recipe_path = joinpath(work, "vortex-experiment.jld2")
save_batch_experiment(recipe_path, batch)
saved = ExperimentController(recipe_path)
saved_figure = experiment_workflow(saved; size=(1100,800)) #hide
for (option, value) in (("Files", :files), ("run history", :history)) #hide
    menu = only(filter(b -> b isa GUI.Menu && (option,value) in b.options[], saved_figure.content)) #hide
    menu.i_selected[] = findfirst(==((option,value)), menu.options[]) #hide
end #hide
tour_picture(saved_figure, "saved-experiment") #hide

# ![The saved-experiment window shows file actions and the new recipe's empty run history.](../assets/gui/saved-experiment.png)
#
# This new recipe has no run history yet; saving settings does not attach the
# earlier batch run retroactively. On another session, open it, choose a result
# destination, and use **replay exact recipe**. See
# [Save your settings and run them again](../howto/gui_experiments.md).
#
# You have now made a masked vector field, inspected it in physical units,
# saved its native result, and saved a recipe for repeating the analysis.
#
# For your next task, use [the GUI task guide](../howto/gui.md). It points to
# preprocessing, recipe revisions, profiles, stereo, and tracking without
# requiring you to learn every tool at once.
