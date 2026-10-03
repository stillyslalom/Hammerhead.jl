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
and scale. It uses the effective preset schedule, so **medium** is saved as
explicit passes. Use the recipe pages to check the setup before saving.
The record format needs image files; an array-only batch must first be saved
as images. Saving a snapshot does not attach earlier batch results as run history.

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
without opening desktop windows. The same actions are available through the
buttons above.

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

## Make a variation without losing the original

Use **revise pass schedule…**, **revise preprocessing…**, or **revise ROI / scale…**
to open a dedicated editor. Validate the changes, save a **distinct revision**,
then open that revision for replay. The source recipe and its history stay separate.

For a worked task, choose [pass settings](gui_recipe_revision.md),
[ordered preprocessing and image previews](gui_preprocessing_revision.md), or
[ROI and physical scale](gui_recipe_geometry_revision.md).
[Compare saved recipes](gui_comparison.md) when you want to inspect the difference.

## Check a completed run

In **Reports**, **save quality report…** writes a report and shows its summary.
The report states which vectors were available, masked, or flagged; uncertainty
availability alone does not establish accuracy. See [saved run-quality reports](run_quality.md)
for interpreting those counts.

If replay refuses changed inputs or another software environment, check the
message before enabling **allow environment changes**. Referenced custom scripts
are never executed automatically. [Replay and cancellation](gui_experiment_replay.md)
covers failures and partial outputs; the [experiment workflow reference](../reference/gui_experiments.md)
covers full settings, verification, and custom callbacks.
