```@meta
CurrentModule = Hammerhead
```

# Save and replay an ensemble experiment

Save the ordered image pairs and processing settings together so you can
recreate a pooled field and check which inputs produced it.

## Save, reopen, run

This example uses the committed PIV Challenge A pair. It exercises the saved
workflow with one pair; it does not demonstrate the benefit of averaging.
All output files live in a temporary directory.

```@example saved_ensemble
using Hammerhead

mktempdir() do directory
    images = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
    pairs = [(joinpath(images, "A001_1.tif"), joinpath(images, "A001_2.tif"))]
    passes = PIVParameters(window_size=64, overlap=32, padding=true,
        uod_enable=false, validation=(), replace_outliers=false)
    recipe = EnsemblePIVRecipe(passes; threaded=false)
    record = EnsembleExperimentRecord(pairs, recipe)

    record_path = joinpath(directory, "experiment.jld2")
    save_experiment(record_path, record)
    reopened = load_ensemble_experiment(record_path)
    run = replay_experiment(reopened;
        output=joinpath(directory, "pooled.jld2"), run_record=record_path,
        record_diagnostics=true)

    report = quality_report(reopened, run; verify_inputs=true)
    report_path = save_quality_report(joinpath(directory, "quality.toml"), report)
    data = quality_report_data(load_quality_report(report_path))
    p = data["provenance"]
    @assert run.status == :completed && p["published_results"] == 1
    (; status=run.status, input_pairs=p["input_pairs"],
       pooling_passes=p["scheduled_passes"],
       pair_contributions=p["completed_contributions"],
       pooled_results=p["published_results"])
end
```

The result is a completed run with **one input pair, one contribution and one
pooled result**. With four pairs and two passes, there would be eight pair
contributions but still one result. Contributions count processing work,
including masked or skipped windows, not independent measurements.

## Capture your own settings

Use only image files, ordered by acquisition, and include the complete schedule:

```julia
files = sort(filter(f -> endswith(lowercase(f), ".tif"),
                    readdir("recording"; join=true)))
pairs = image_pairs(files; mode=:paired)
passes = multipass_parameters([64, 32, 16, 16]; padding=true,
    final=(uncertainty=true,))
recipe = EnsemblePIVRecipe(passes;
    preprocessing=[PreprocessStep(:highpass_filter; sigma=3)],
    backend=:cpu, image_type=Float64, threaded=false)
record = EnsembleExperimentRecord(pairs, recipe)
save_experiment("ensemble-record.jld2", record)
```

Saved recipes support CPU/KA with Float32/Float64, built-in preprocessing,
embedded backgrounds, a full-image static mask and an optional physical scale.
ROI, custom scripts and vendor-device recipes are not supported here.
The direct [ensemble driver](ensemble.md) supports a broader set of workflows.

## Monitor or cancel a replay

```julia
record = load_ensemble_experiment("ensemble-record.jld2")
run = replay_experiment(record;
    output="ensemble-results.jld2", run_record="ensemble-record.jld2",
    progress=event -> println(event.completed_contributions, "/",
                              event.total_contributions),
    cancel_requested=() -> false,
    record_diagnostics=true)
```

Progress follows each joined pair computation. Even the last progress event
precedes peak analysis and publication. Cancellation is checked between pairs
and before publication; a cancelled run has no published pool. Replaying starts
from the beginning, rather than resuming an accumulator.

Processing failure or cancellation preserves a previous output destination.
Successful replay may replace it, so choose a new output name to keep older
runs verifiable. Input and record aliases are refused. If the environment has
changed, replay requires an explicit `allow_environment_change=true`.

## Check the saved output

```julia
verify_ensemble_experiment_run(record, run; verify_results=true)
report = quality_report(record, run; verify_inputs=true)
display(report)
save_quality_report("ensemble-quality.toml", report)
```

The check binds the completed output to the saved settings, ordered inputs and
run. The report adds stored-field counts and, when recorded, contribution
observations. See [Checking an ensemble result](ensemble_quality_reports.md)
for interpretation. Neither check establishes stationarity or accuracy.

For a moved native result, supply `output="local/moved.jld2"` to the verifier
or report call. Loading a saved report checks its contents, not the current
result file. Input files are rechecked only with `verify_inputs=true` during
inspection, and always before replay.

Output and run history are saved separately. If history saving fails after a
completed or cancelled attempt, `EnsembleRunRecordError.run` retains its true
status and `.cause` explains the write failure. See the
[saved-ensemble reference](../reference/ensemble_experiments.md) for failure,
publication and exact verification rules.
