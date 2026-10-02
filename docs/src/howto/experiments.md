```@meta
CurrentModule = Hammerhead
```

# Save and replay a planar experiment

**Goal:** reopen a file-based planar PIV recipe and reproduce its processing
without reconstructing the pass schedule, preprocessing, mask, ROI, or units.
Version 1 deliberately covers static planar recipes; it does not yet save
stereo calibration, dynamic callbacks, acquisition timestamps, or resume state.

Build an explicit schedule and ordered built-in preprocessing steps. The
recipe copies its arrays/settings, including original full-image masks and
backgrounds. Keep pixel-side settings in their measured units; attaching a
`PhysicalScale` records calibration without converting the saved results.

This executable example uses the committed PIV Challenge A pair. It analyzes
a small ROI so the example remains inexpensive, while recording the original
image-file identities and complete mask:

```@example experiments
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(path -> endswith(lowercase(path), ".tif"),
                    readdir(directory; join=true)))
pairs = [(files[1], files[2])]
image = load_image(files[1])
mask = falses(size(image))
mask[1:12, 1:12] .= true

passes = multipass_parameters([32, 16]; padding=true, apodization=:gauss)
recipe = PIVRecipe(passes;
    preprocessing=[PreprocessStep(:highpass_filter; sigma=3)],
    mask, roi=ROI(1:64, 1:64), image_type=Float32,
    scale=PhysicalScale(pixel_size=0.02, dt=0.001,
                        length_unit="mm", time_unit="s"))
experiment = ExperimentRecord(pairs, recipe)

mktempdir() do work
    record_path = save_experiment(joinpath(work, "experiment.jld2"), experiment)
    reopened = load_experiment(record_path)
    run = replay_experiment(reopened;
        output=joinpath(work, "vectors.jld2"), run_record=record_path)
    results = load_results(run.output; lazy=true)
    (status=run.status, completed_pairs=run.completed_pairs,
     precision=eltype(results[1].u), saved_runs=length(load_experiment(record_path).runs))
end
```

The experiment file embeds settings, masks/backgrounds, input content hashes,
software provenance, and optional run metadata. Image files remain external.
Results use the existing native result format. `run_record` appends execution
metadata to the reopened record and records failures after processing begins;
preflight rejection does not alter either destination.

## Record processing revisions

Treat recipes as snapshots. To change the schedule or preprocessing, create
a new `PIVRecipe` and `ExperimentRecord` from the same input pairs. Use
`recipe_diff` to inspect the processing changes before running either revision:

```@example experiments
revised = PIVRecipe(recipe.passes;
    preprocessing=[PreprocessStep(:highpass_filter; sigma=5)],
    mask=recipe.mask, roi=recipe.roi, scale=recipe.scale, image_type=Float64)
changes = recipe_diff(recipe, revised)
[(change.path, change.before, change.after) for change in changes]
```

The change paths identify image precision and the first preprocessing step's
sigma. Display `changes` directly for a compact readable report, or use its
`changes`, `before_id`, and `after_id` fields in scripts or GUI forms. Comparison
checks snapshot integrity first and rejects recipes whose contained settings
were mutated; it does not execute PIV or read input/script files.

Passes, preprocessing, and validators are compared by their ordered positions.
Window/search sizes, ROI bounds, and CLAHE tiles remain readable tuples. Changed
embedded masks and backgrounds have `RecipeArraySummary` values containing
size, element type, and full content digest, rather than pixel dumps. Added or
removed sequence items use `missing`; an explicitly disabled optional mask,
ROI, scale, or script remains `nothing`. Script content digests and entrypoints
are compared, while relocated locators alone do not create changes. Recreate
a `ScriptReference` to snapshot changed script content; comparison does not
rehash files on disk.

Compare experiment `input_id` values separately: `recipe_diff(first.recipe,
second.recipe)` describes processing settings, not input, output, or environment
changes. Identical byte inputs at different paths have the same `input_id`,
while altered pair order, content, or dimensions change it. A configuration
report does not predict the numerical effect on representative pairs. Use a
distinct result path for each run if earlier outputs must remain available for
comparison.

Replay checks input and referenced-script content before opening result output.
It also rejects overlapping input/script/record destinations, including same-file
aliases. Keep files unchanged during processing. File hashes detect changes;
they do not provide an atomic snapshot against concurrent writes.

## Handle a custom preprocessor explicitly

Reference a script and its intended entrypoint without storing executable
functions in the record:

```julia
reference = ScriptReference("my_preprocessing.jl"; entrypoint="prepare_image(image)")
recipe = PIVRecipe(passes; external_preprocess=reference, image_type=Float32)
experiment = ExperimentRecord(pairs, recipe)
save_experiment("custom-experiment.jld2", experiment)

# The caller loads/reviews its own implementation and supplies the function.
replay_experiment(load_experiment("custom-experiment.jld2");
    output="custom-vectors.jld2", custom_preprocess=prepare_image,
    run_record="custom-experiment.jld2")
```

The library verifies the referenced bytes and never includes/evaluates the script
or looks up its entrypoint. Your function must match the intended script, preserve
the full-image dimensions and saved Float32/Float64 precision, and return finite
values. It runs after any built-in steps. External state used by that function
is not automatically recorded. Supplying a custom function without a script
reference, or omitting the function when a reference is present, is rejected.

## Inspect software differences before rerunning

`creation_environment` records Julia/core/package versions, source and package
hashes, platform/thread settings, and Project/Manifest text. Replay compares the
tracked software identity by default. When intentionally changing environments,
pass `allow_environment_change=true`; the run records its actual environment
separately. This is an explicit rerun, not a bitwise reproducibility guarantee.
Inspect those records when comparing revisions. Project/Manifest snapshots are
provenance and are never installed or activated automatically.

Unknown experiment versions, incomplete or malformed recipes, changed identities,
incompatible dimensions/ROI/masks, and unsupported backends are rejected. Every
replay starts from the first pair. Interrupted/failed output may retain a
completed native prefix, but version 1 does not resume from that prefix. See the
[experiment reference](../reference/experiments.md) for the schema, numerical
assumptions, callback/failure behavior, and scope limits.
