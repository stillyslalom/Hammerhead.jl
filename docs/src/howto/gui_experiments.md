# Save your settings and run them again

Once a representative pair looks useful, save its recipe before processing
the rest of the recording. A **results file** holds the measured vectors; an
**experiment file** holds the input pairs and settings for another run.

Start from the batch form used in [Your first PIV session in the GUI](../tutorials/gui_tour.md).
The saved-experiment workflow opens in a **separate window**.

![The saved-experiment window from the GUI tutorial.](../assets/gui/saved-experiment.png)

## Save the working setup

1. Click **saved experiments…** in the batch window.
2. In **Files**, choose **snapshot batch**, then **save experiment…**.
3. Give the experiment a useful name, such as `vortex-medium.jld2`.

The snapshot keeps the ordered file pairs, passes, preprocessing, mask, ROI,
and scale. It saves the effective preset as explicit passes; use the recipe
pages to check them. Save array-based frames as image files before snapshotting.
The new experiment starts with an empty run history; replay adds its first run.

## Run the saved recipe

1. Use **open experiment…** to reopen the record.
2. In **Files**, choose a new result output. Choose a run record if needed;
   an opened experiment is normally where its run history is appended.
3. Switch to **Replay** and press **replay exact recipe**.
4. When it finishes, use **view completed results** to inspect the field.

Keep distinct result outputs if you want earlier runs to remain inspectable.
**cancel after current pair** waits for the current pair to finish writing;
a cancel request at the final write can still mean completion. Ordinary replay
starts from the beginning. For restart between committed pairs, open the
separate [checkpoint workflow](gui_checkpoints.md).

### Try it on a small recording

This example uses a committed image pair, saves the settings, and replays them
through the controller. The buttons above perform the same actions.

```@example gui_experiments
using HammerheadGUI
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(p -> endswith(lowercase(p), ".tif"), readdir(directory; join=true)))
batch = BatchRunner(files=files[1:2], window_schedule=[32,16],
                    roi=ROI(1:64,1:64), pixel_size=0.02, dt=0.001,
                    length_unit="mm", time_unit="s")

mktempdir() do work
    path = joinpath(work, "experiment.jld2")
    save_batch_experiment(path, batch)
    saved = ExperimentController(path)
    saved.output_path[] = joinpath(work, "vectors.jld2")
    start!(saved; async=false)
    (status=saved.state[], result_frames=nframes(experiment_results(saved)),
     recorded_runs=length(saved.record[].runs))
end
```

## Save a variation

Use **revise pass schedule…**, **revise preprocessing…**, or **revise ROI / scale…**
to open a dedicated editor. Validate the changes, save a **distinct revision**,
then open that revision for replay. The source recipe and its history stay separate.

For a worked task, choose [pass settings](gui_recipe_revision.md),
[ordered preprocessing and image previews](gui_preprocessing_revision.md), or
[ROI and physical scale](gui_recipe_geometry_revision.md). To exclude another
reflection while keeping the old mask, [change the saved mask](gui_recipe_mask_revision.md).
[Compare saved recipes](gui_comparison.md) when you want to inspect the difference.

## Check a completed run

In **Reports**, **save quality report…** writes a report and shows its summary.
Use its available, masked, flagged, and uncertainty counts to inspect the run.
[Saved run-quality reports](run_quality.md) explains the populations.

If replay reports changed inputs or another software environment, check the
message before enabling **allow environment changes**. For recipes with custom
scripts, follow the callback setup in the
[experiment workflow reference](../reference/gui_experiments.md).
[Replay and cancellation](gui_experiment_replay.md) covers partial outputs.