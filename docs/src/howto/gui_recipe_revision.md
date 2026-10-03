```@meta
CurrentModule = HammerheadGUI
```

# Add a finer final pass to a saved experiment

Your first pass finds the broad motion. What changes when a second pass uses
smaller windows? Keep the recording and other settings fixed, save a revision,
and inspect the two fields side by side.

The example uses the bundled tip-vortex recording and a small region around its
core. No download is needed. The [real-recording lesson](../tutorials/real_data.md)
shows the full images.

## 1. Start with a saved coarse pass

```@example finer_pass
using Hammerhead, HammerheadGUI, CairoMakie
CairoMakie.activate!()

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
a = joinpath(directory, "A001_1.tif")
b = joinpath(directory, "A001_2.tif")
rows, cols = 385:640, 449:704
coarse = PIVParameters(window_size=64, overlap=32,
    padding=true, apodization=:gauss, replace_outliers=false)
original = ExperimentRecord([(a, b)],
    PIVRecipe(coarse; roi=ROI(rows, cols), image_type=Float32))
work = mktempdir()  # use your analysis directory to keep the files
original_path = save_experiment(joinpath(work, "coarse-experiment.jld2"), original)
nothing # hide
```

In the GUI, open this record in the
[saved-experiment workflow](gui_experiments.md), then choose
**revise pass schedule...**. The revision opens separately; your original stays
available for comparison.

## 2. Duplicate the pass and reduce its window

Choose **duplicate pass**. Select the new final pass and the **Geometry** group.
Set **Window size** and **Search area size** to `32, 32`, and **Overlap** to
`16, 16`. These pairs mean rows, columns. Keep the first pass at 64 px.

The same edits through the controller are:

```@example finer_pass
editor = RecipeRevisionController(original_path)
insert_revision_pass!(editor, 2; source=1)
set_revision_pass!(editor, 2, Dict(
    :window_size => "32, 32",
    :search_area_size => "32, 32",
    :overlap => "16, 16"))
apply_recipe_revision!(editor; async=false)
@assert editor.state[] == :completed
[(pass.window_size, pass.overlap) for pass in revision_recipe(editor).passes]
```

Choose **validate / preview changes** and inspect the settings difference.
This shows the processing plan that the next replay will use.
The smaller final window uses fewer pixels per sample, and the smaller spacing
places more samples inside the same region.

## 3. Save, then inspect the measured fields

Choose **save distinct revision...** and a new filename, then
**open saved revision...** to replay it.
Here we replay each saved recipe on the same pair:

```@example finer_pass
save_recipe_revision!(editor, joinpath(work, "finer-experiment.jld2"); async=false)
@assert editor.state[] == :completed
revised = editor.saved_record[]
@assert revised.input_id == original.input_id && isempty(revised.runs)
before_run = replay_experiment(original; output=joinpath(work, "coarse-vectors.jld2"))
after_run = replay_experiment(revised; output=joinpath(work, "finer-vectors.jld2"))
before = only(load_results(before_run.output))
after = only(load_results(after_run.output))
nothing # hide
```

```@example finer_pass
first_image = load_image(Float32, a)
fig = Figure(size=(900, 390))
for (column, result, name) in ((1, before, "One 64 px pass"),
                               (2, after, "64 px then 32 px"))
    ax = Axis(fig[1, column];
        title="$name\n$(length(result.x)) × $(length(result.y)) grid samples",
        xlabel="original x (px)", ylabel="original y (px)",
        yreversed=true, aspect=DataAspect())
    image!(ax, (first(cols)-0.5, last(cols)+0.5),
        (first(rows)-0.5, last(rows)+0.5), first_image[rows, cols]';
        colormap=:grays, colorrange=(0, 0.6))
    plot_vector_field!(ax, result; stride=1, lengthscale=2,
        color=:cyan, replaced_color=:orangered)
    limits!(ax, first(cols)-0.5, last(cols)+0.5, last(rows)+0.5, first(rows)-0.5)
end
fig
```

Both panels show displacement in pixels, with arrow lengths enlarged by the
same factor of two. Orange arrows carry current outlier flags; this example
leaves those flagged values unreplaced. The denser final grid shows how the
sampling changed. Judge the vectors against the particle patterns, paying
particular attention to the sparse signal inside the vortex core.

For a numerical comparison, use
[compare a representative pair](gui_comparison.md) with these two saved records
and pair 1 in each. That workflow compares exact common grid centers rather
than interpolating the fields to look alike.

**Try it:** keep the final window at 32 px and change only its overlap from
16 to 24 px. Save to another new filename. How does sample spacing change?
Which image information stays the same despite the extra arrows?

```@example finer_pass
rm(work; recursive=true) # hide
nothing # hide
```

To change image conditioning, bounds or exclusions, continue with the
[preprocessing](gui_preprocessing_revision.md),
[ROI and scale](gui_recipe_geometry_revision.md), or
[mask](gui_recipe_mask_revision.md) form. The
[revision reference](../reference/gui_recipe_revision.md) documents editable
fields, retained settings and save rules.
