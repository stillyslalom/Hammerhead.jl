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

A result file stores the measurements. An experiment record also stores the
recipe and identifies the input images. Follow the worked
[save-and-replay example](../howto/experiments.md) to save a record, reopen it,
and produce another result without rebuilding the settings by hand.

The corresponding workflows for
[stereo recordings](../howto/stereo_experiments.md) and
[pooled ensembles](../howto/ensemble_experiments.md) preserve the settings
specific to those methods. The [GUI](../howto/gui_experiments.md) provides a
saved-experiment window for the same planar workflow.

For long work that must survive interruption, use
[checkpoints](../howto/checkpoints.md), which store the progress needed to resume.
Ordinary batch output retains the results completed before a failure.

## Return to a questionable result

Start with a [quality summary](../howto/run_quality.md) and inspect the relevant
images. For a closer investigation, you can record extra information at run time:

- [Pass progress](../howto/execution_diagnostics.md) shows what each planar pass
  actually did; [stereo](../howto/stereo_execution_diagnostics.md) and
  [ensemble](../howto/ensemble_execution_diagnostics.md) runs have their own views.
- [Vector history](../howto/measurement_history.md) distinguishes measured,
  rejected and filled values.
- [Frame-pair timing](../howto/pair_timing.md) and
  [stereo timing](../howto/stereo_pair_timing.md) retain the available acquisition
  times alongside the measurements.
- [Ensemble quality summaries](../howto/ensemble_quality_reports.md) keep pooled
  image contributions separate from the single field they produce.

These are optional tools for answering a specific question. They are not
prerequisites for processing your first recording.
