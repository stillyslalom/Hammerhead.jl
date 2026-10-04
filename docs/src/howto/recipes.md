# Save settings and reuse them

Once a pass schedule, preprocessing chain, mask and scale work on a
representative pair, store them as a [`PIVRecipe`](@ref). A recipe holds the
settings only, so the same file processes this recording, the next one, or a
rerun next month.

The examples use four small synthetic pairs with a uniform displacement of
(1.5, −1.0) px; substitute your own image pairs or file paths.

```@example recipes
using Hammerhead, Random
using Hammerhead.SyntheticData

flow(x, y, z, t) = (1.5, -1.0, 0.0)
pairs = map(1:4) do seed
    imgA, imgB, _, _ = generate_synthetic_piv_pair(flow, (128, 128), 1.0;
        particle_density = 0.05, rng = MersenneTwister(seed))
    (imgA, imgB)
end
nothing # hide
```

## Build a recipe

Pass a schedule to `PIVRecipe`, plus any of the optional settings:
preprocessing steps, a full-image mask, an [`ROI`](@ref), a
[`PhysicalScale`](@ref) and the processing precision.

```@example recipes
passes = multipass_parameters([32, 16]; padding = true, apodization = :gauss)
mask = falses(128, 128)
mask[1:20, 1:20] .= true                     # true = excluded
recipe = PIVRecipe(passes;
    preprocessing = [PreprocessStep(:highpass_filter; sigma = 4)],
    mask,
    scale = PhysicalScale(pixel_size = 0.02, dt = 0.001,
                          length_unit = "mm", time_unit = "s"))
nothing # hide
```

Each [`PreprocessStep`](@ref) names a built-in operation and its options, so
the chain can be saved. A background for `:subtract_background` is copied into
the step. Validation settings inside `PIVParameters` are saved too, as long as
they use the built-in validators.

## Save it and load it again

```@example recipes
dir = mktempdir()
save_recipe(joinpath(dir, "settings.jld2"), recipe)
load_recipe(joinpath(dir, "settings.jld2")) == recipe
```

## Apply it to a recording

[`apply_recipe`](@ref) runs the sequence driver with the recipe's settings.
With an `output` path, it writes each result as it completes and stores the
recipe in the same file:

```@example recipes
results = apply_recipe(recipe, pairs;
                       output = joinpath(dir, "vectors.jld2"), progress = false)
length(results), results[1].scale.length_unit
```

Pairs can be file paths, as from [`image_pairs`](@ref), or in-memory images.
Other keywords, such as `backend = :cuda`, `collect_results = false` or
`on_result`, go to the driver; see [Batch processing](batch.md).

## Recover the settings from a results file

A results file written by `apply_recipe` (or by a run in the GUI window) knows how
it was made. Load its recipe to inspect it or to process new images the same way:

```@example recipes
from_results = load_recipe(joinpath(dir, "vectors.jld2"))
from_results == recipe
```

## Compare two recipes

[`recipe_diff`](@ref) lists the settings that differ, with a path to each one:

```@example recipes
finer = PIVRecipe(multipass_parameters([32, 12]; padding = true, apodization = :gauss);
    preprocessing = [PreprocessStep(:highpass_filter; sigma = 6)],
    mask = recipe.mask, scale = recipe.scale)
for change in recipe_diff(recipe, finer)
    println(change.path, ": ", change.before, " → ", change.after)
end
```

Masks and backgrounds appear by size, so a changed mask shows up as one entry;
an added or removed pass is listed with all of its settings.

## Pool the pairs into one field

Set `mode = :ensemble` to sum correlations over all pairs and return one
result, as [`run_piv_ensemble`](@ref) does. Ensemble recipes cover the full
image; use the mask to exclude regions.

```@example recipes
pooled = PIVRecipe(passes; mask, mode = :ensemble)
field = apply_recipe(pooled, pairs; progress = false)
u = filter(isfinite, field.u)
round(sum(u) / length(u); digits = 2)   # mean u in px
```

For two cameras, pass both cameras' pairs and their dewarpers:
`apply_recipe(recipe, pairs1, pairs2, dw1, dw2)`. A stereo recipe's mask is on
the dewarped grid.

## In the GUI

The planar window's **Save settings…** button writes the same recipe file, and
**Open settings…** loads a recipe file or an earlier run's results file into
the window. A run in the window stores its recipe in the results file, as
`apply_recipe` does. See [Analyze an image pair in the GUI](gui.md).
