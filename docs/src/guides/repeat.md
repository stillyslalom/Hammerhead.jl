# From one pair to a repeatable recording

Once a representative pair looks useful, apply the same settings to the rest
of the recording. Save both the results and the choices that produced them so
you can return to an interesting frame or compare another processing method.

## Process the frames in acquisition order

For filenames padded as `frame_0001.tif`, `frame_0002.tif`, and so on:

```julia
using Hammerhead
files = filter(endswith(".tif"), readdir("recording"; join=true))
pairs = image_pairs(files)  # (1,2), (3,4), … for double-frame acquisition
passes = multipass_parameters([64, 32]; padding=true, apodization=:gauss)
run_piv_sequence(pairs, passes; output="vectors.jld2", collect_results=false)
results = load_results("vectors.jld2"; lazy=true)
```

Check a few pairs before running. For a time-resolved sequence where adjacent
frames form each pair, use `image_pairs(files; mode=:chained)` instead.
The [batch guide](../howto/batch.md) covers pairing, memory use and exports.

## Save the settings as well

A result file stores the measurements; a recipe stores the passes,
preprocessing, mask, ROI and scale that produced them. Run the recording
through [`apply_recipe`](@ref) and the recipe is written into the results file:

```julia
recipe = PIVRecipe(passes)
apply_recipe(recipe, pairs; output="vectors.jld2")
same_settings = load_recipe("vectors.jld2")
```

[Save settings and reuse them](../howto/recipes.md) covers saving a recipe on
its own, comparing two recipes, and pooled ensembles. The GUI batch form saves
and opens the same recipe files.

## Return to a questionable result

Open the results file in the [result explorer](../howto/gui.md) and inspect the
images for the pairs in question. Check the rejected vectors with the
[validation guide](../howto/validation.md), and change one setting at a time
on that pair before running the recording again.
