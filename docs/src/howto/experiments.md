```@meta
CurrentModule = Hammerhead
```

# Save an experiment and run it again

You have found useful settings for a recording. Now save them so that tomorrow
you can reopen the experiment and recover the same image pairs, processing
choices and calibration without reconstructing your session.

This example uses the supplied tip-vortex images and a small region to keep
the run short. The [real-recording lesson](../tutorials/real_data.md) shows the
full images and flow.

## 1. Describe the measurement

```@example experiments
using Hammerhead
directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(p -> endswith(lowercase(p), ".tif"),
                    readdir(directory; join=true)))
pairs = [(files[1], files[2])]
passes = multipass_parameters([32, 16]; padding=true, apodization=:gauss)
recipe = PIVRecipe(passes;
    roi=ROI(1:64, 1:64),
    preprocessing=[PreprocessStep(:highpass_filter; sigma=3)],
    image_type=Float32)
experiment = ExperimentRecord(pairs, recipe)
nothing # hide
```

The recipe is the processing plan. The experiment connects that plan to these
particular input images. A mask and a `PhysicalScale` can also be included in
the recipe when you have them; this example keeps displacements in pixels.

## 2. Save, reopen and run

```@example experiments
work = mktempdir()  # replace with your analysis directory for a lasting result
record_path = save_experiment(joinpath(work, "experiment.jld2"), experiment)
reopened = load_experiment(record_path)
run = replay_experiment(reopened;
    output=joinpath(work, "vectors.jld2"), run_record=record_path)
results = load_results(run.output; lazy=true)
(status=run.status, fields_written=length(results),
 saved_runs=length(load_experiment(record_path).runs))
```

There are two files with different jobs: `experiment.jld2` retains the recipe
and run history; `vectors.jld2` contains the measured fields. The image files
remain where they were. Replay checks them against the saved record before
processing, so replacing an image does not silently change the experiment.

## 3. Change one setting deliberately

Try a different high-pass width while keeping the other settings fixed:

```@example experiments
revised = PIVRecipe(recipe.passes;
    roi=recipe.roi, image_type=recipe.image_type,
    preprocessing=[PreprocessStep(:highpass_filter; sigma=5)])
changes = recipe_diff(recipe, revised)
[(change.path, change.before, change.after) for change in changes]
```

The difference should identify the filter's `sigma`. It describes a settings
change, not whether the resulting vectors improve. Next,
[compare the recipes on the same pair](pair_comparison.md), and use a distinct
output filename for each run you want to keep.

```@example experiments
rm(work; recursive=true) # hide
nothing # hide
```

## Continue with your recording

- Replace the example pair with your ordered pairs from the
  [batch guide](batch.md).
- Use [checkpoints](checkpoints.md) if the job needs to resume after interruption.
- Use the dedicated [stereo](stereo_experiments.md) or
  [ensemble](ensemble_experiments.md) workflow for those processing methods.

For custom preprocessing, software-environment changes, progress callbacks and
saved-file details, consult the [experiment reference](../reference/experiments.md).
Saved scripts are references: replay never executes a script automatically.
