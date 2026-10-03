```@meta
CurrentModule = HammerheadGUI
```

# Crop a saved experiment and display velocity

Your recording covers more than the region you want to measure. Select that
region, enter a spatial calibration and exposure delay, then save a revised
experiment. The original images and experiment remain available.

## Make the change in the GUI

1. Open your saved planar experiment and choose **revise ROI / scale...**.
2. Enable **ROI enabled** and enter the first/last row and column. These are
   inclusive coordinates in the original image: rows are y, columns are x.
3. Enable **Physical scale enabled**. Enter the length per pixel, the delay
   between the two exposures, and matching length/time labels.
4. Choose **validate / metadata diff**, then **save distinct revision...**.
   Open the saved revision and replay it to produce a new field.

Every interrogation window must fit inside your crop. If validation refuses a
small crop, enlarge it or deliberately [revise the passes](gui_recipe_revision.md).
For an image overlay before replay, use the
[preprocessing preview](gui_preprocessing_revision.md) with the same controller.

## Try it on a supplied recording

This executable example selects a 192 × 192 px region of the bundled tip-vortex
pair. The scale of **0.025 mm/px** and exposure delay of **1 ms** are illustrative
values for practicing conversion. Use your measured calibration and exposure
delay when applying the example to your own recording.

```@example saved_geometry_lesson
using Hammerhead, HammerheadGUI, CairoMakie
CairoMakie.activate!()

mktempdir() do directory
    images = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
    a = joinpath(images, "A001_1.tif")
    b = joinpath(images, "A001_2.tif")
    original = ExperimentRecord([(a, b)],
        PIVRecipe(multipass_parameters([32, 16]; padding=true, apodization=:gauss);
            image_type=Float32))
    source = save_experiment(joinpath(directory, "original.jld2"), original)
    editor = RecipeRevisionController(source)
    set_revision_roi!(editor,
        Dict(:row_first=>"33", :row_last=>"224",
             :col_first=>"49", :col_last=>"240"); enabled=true)
    set_revision_scale!(editor,
        Dict(:pixel_size=>"0.025", :dt=>"0.001",
             :length_unit=>"mm", :time_unit=>"s"); enabled=true)
    save_recipe_revision!(editor, joinpath(directory, "cropped.jld2"); async=false)
    @assert editor.state[] == :completed
    run = replay_experiment(editor.saved_record[];
        output=joinpath(directory, "vectors.jld2"))
    native = load_results(run.output)[1]
    velocity = physical(native)

    raw = load_image(Float32, a)
    nr, nc = size(raw)
    fig = Figure(size=(850, 370))
    image_axis = Axis(fig[1, 1]; title="Selected region in the original image",
        xlabel="x (px)", ylabel="y (px)", yreversed=true, aspect=DataAspect())
    image!(image_axis, (0.5, nc+0.5), (0.5, nr+0.5), raw'; colormap=:grays)
    lines!(image_axis, [48.5, 240.5, 240.5, 48.5, 48.5],
        [32.5, 32.5, 224.5, 224.5, 32.5]; color=:cyan, linewidth=2)
    field_axis = Axis(fig[1, 2]; title="Replayed crop with illustrative scale",
        xlabel="x (mm)", ylabel="y (mm)", yreversed=true, aspect=DataAspect())
    plot_vector_field!(field_axis, velocity; stride=2, color=:dodgerblue,
        lengthscale=0.004)
    fig
end
```

The cyan box shows the requested region. The field keeps its position within
the original image, using the original coordinate origin. Arrows on the right
represent velocity, drawn with a 0.004 s length
factor for visibility. Orange marks vectors flagged during validation.

Check one conversion by hand: a displacement of 2 px would become
`2 × 0.025 / 0.001 = 50 mm/s`. Native results retain displacement in pixels;
[`physical`](@ref) converts them for display. Convert the calibration number
when changing units: 0.025 mm/px equals 0.000025 m/px.

**Try it:** halve `dt` and replay. Velocities should double while vector
positions and pixel displacements stay the same. Then move the crop without
changing its size and inspect which part of the recording you measure.

Use a measured calibration and the actual exposure delay for your own data;
the [scaling guide](scaling.md) explains how to obtain them. Draft validation,
complete-recipe preservation and save behavior are described in the
[geometry reference](../reference/gui_recipe_geometry_revision.md).
