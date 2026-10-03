```@meta
CurrentModule = HammerheadGUI
```

# Exclude another reflection from a saved recipe

Open a saved planar experiment and choose **revise mask...**. Load one original
pair/frame as a raw reference, draw around the new reflection, then choose
**Apply raster (enable mask)** and **save distinct revision...**. The existing
mask is copied exactly; the original record stays unchanged.

## Try one added exclusion

This example uses a bundled particle image and an illustrative starting mask.
Both polygons are chosen by hand for this example. Red shows pixels excluded
from PIV.

```@example saved_mask_revision
using Hammerhead, HammerheadGUI, CairoMakie
CairoMakie.activate!()

mktempdir() do directory
    images = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
    a = joinpath(images, "A001_1.tif")
    b = joinpath(images, "A001_2.tif")
    raw = load_image(Float32, a)
    nr, nc = size(raw)
    seed = falses(nr, nc)
    seed[1:round(Int, nr/5), 1:round(Int, nc/4)] .= true
    original = ExperimentRecord([(a, b)],
        PIVRecipe(PIVParameters(window_size=64, overlap=32); mask=seed, image_type=Float32))
    original_path = joinpath(directory, "original.jld2")
    save_experiment(original_path, original)

    revision = RecipeRevisionController(original_path)
    reference = RecipeMaskReferenceController()
    load_recipe_mask_reference!(reference, revision; async=false)
    raw = reference.bundle[].raw_image
    editor = reference.bundle[].editor
    for (x, y) in ((.6nc, .3nr), (.8nc, .3nr), (.8nc, .6nr), (.6nc, .6nr))
        add_vertex!(editor, x, y)
    end
    close_active!(editor)
    apply_revision_mask!(revision, reference; async=false)
    save_recipe_revision!(revision, joinpath(directory, "revised.jld2"); async=false)
    @assert revision.state[] == :completed
    revised_mask = revision.saved_record[].recipe.mask
    @assert all(revised_mask[seed]) && count(revised_mask) > count(seed)
    @assert original.recipe.mask == seed

    fig = Figure(size=(820, 360))
    for (column, title, mask) in ((1, "Imported mask", seed), (2, "Added exclusion", revised_mask))
        ax = Axis(fig[1, column]; title, xlabel="x (px)", ylabel="y (px)",
            yreversed=true, aspect=DataAspect())
        heatmap!(ax, raw'; colormap=:grays, colorrange=extrema(raw))
        heatmap!(ax, Float32.(mask)'; colormap=[(:red, 0.0), (:red, 0.45)], colorrange=(0, 1))
    end
    fig
end
```

The revised record has the same ordered inputs and empty run history. Replay
it to measure a new field with the revised mask.

**Try it:** move the polygon, then draw a hole over part of the imported red
region. Inspect which pixels are restored before saving and replaying.

## Use the canvas

Draw in full-image pixel coordinates. Close the polygon before applying it.
**Draw hole** removes exclusions, including imported ones; later polygons act
in drawing order. Grow/shrink adjusts the combined raster. **Clear all editor
pixels** clears the editor only; **Apply raster (enable mask)** puts its
all-false raster into the recipe. **Reset to imported mask** restores the
original optional recipe mask instead.

The reference shows the complete original image. Cyan outlines a saved ROI;
red marks excluded pixels. To compare conditioning,
use the separate [preprocessing preview](gui_preprocessing_revision.md).

You can also **import mask image...**: choose the threshold and polarity, then
validate and save. Replacement bits must match every original full-frame size.
For capture, stale-edit, file verification and save rules, see the
[mask revision reference](../reference/gui_recipe_mask_revision.md).
