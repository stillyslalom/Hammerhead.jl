```@meta
CurrentModule = HammerheadGUI
```

# Try high-pass filtering before replay

High-pass filtering subtracts a smooth background and clamps negative values to
zero. Preview it on an original image pair to check particle visibility, then
save and replay the revised recipe to compare vector fields.

## Compare an original and filtered image

This example saves a recipe for the bundled Challenge A pair, adds one
high-pass step with `sigma=2` pixels, and previews the captured pair.

```@example preprocessing_revision_preview
using Hammerhead, HammerheadGUI, CairoMakie
CairoMakie.activate!()

mktempdir() do directory
    images = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
    pairs = [(joinpath(images, "A001_1.tif"), joinpath(images, "A001_2.tif"))]
    original = ExperimentRecord(pairs,
        PIVRecipe(PIVParameters(window_size=64, overlap=32); image_type=Float32))
    source = joinpath(directory, "original.jld2")
    save_experiment(source, original)

    revision = RecipeRevisionController(source)
    insert_revision_preprocess!(revision, 1, :highpass_filter)
    set_revision_preprocess!(revision, 1, Dict(:sigma=>"2.0"))
    preview = RecipeImagePreviewController()
    preview_recipe_images!(preview, revision; pair_index=1, async=false)
    @assert preview.state[] == :completed
    bundle = preview.bundle[]
    @assert bundle.processed_a == highpass_filter(bundle.raw_a; sigma=2.0)
    @assert bundle.recipe_id == revision_recipe(revision).recipe_id
    save_recipe_revision!(revision, joinpath(directory, "filtered.jld2"); async=false)
    @assert revision.state[] == :completed && isempty(revision.saved_record[].runs)
    @assert isempty(original.recipe.preprocessing)

    intensity_range = (min(minimum(bundle.raw_a), minimum(bundle.processed_a)),
                       max(maximum(bundle.raw_a), maximum(bundle.processed_a)))
    fig = Figure(size=(850, 360))
    for (column, title, image) in ((1, "Original frame A", bundle.raw_a),
                                  (2, "High-pass, sigma = 2 px", bundle.processed_a))
        ax = Axis(fig[1, column]; title, xlabel="x (px)", ylabel="y (px)",
            yreversed=true, aspect=DataAspect())
        heatmap!(ax, image'; colormap=:grays, colorrange=intensity_range)
    end
    Colorbar(fig[1, 3]; colormap=:grays, limits=intensity_range, label="intensity")
    fig
end
```

Both images use the **same intensity range**. Background should darken after
filtering; separate automatic ranges could conceal that change. Inspect the
particles too: an aggressive filter can remove signal you wanted to retain.
The saved revision keeps the same ordered inputs and starts with empty history.
Replay it explicitly when you are ready to compare vector fields.

## Try it in the GUI

1. Choose **revise preprocessing...** in the saved planar workflow.
2. Select `highpass_filter` in the catalogue, choose **add**, then enter sigma.
3. Choose **preview verified pair** and compare original/processed frame A or B.
4. Use **validate / metadata diff**, then **save distinct revision...**.

**Try it:** compare sigma values of 1, 2 and 6 pixels on the same pair. Refresh
the preview after each edit and compare particle visibility on the shared
range. Keep the original record so you can compare replays later.

Steps run in order; a duplicated filter runs twice. The preview shows each
full image with its ROI and mask overlaid. In the GUI, cyan outlines the
ROI and red marks excluded pixels. For supported operations, background
precision, file verification and script restrictions, see the
[preprocessing revision reference](../reference/gui_preprocessing_revision.md).
