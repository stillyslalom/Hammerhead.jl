# Save, replay and inspect a planar ensemble experiment

Use a saved ensemble experiment to retain the exact ordered image pairs,
processing schedule, preprocessing, mask and physical scale used for one pooled
field. Ensemble processing sums correlation planes before finding the vector
field; it does not produce one independent measurement per pair. Choose and
assess the recording interval separately: saved provenance does not demonstrate
stationarity or a common displacement.

The first format supports file-based planar ensembles, CPU/KA, Float32/Float64,
explicit passes, embedded built-in preprocessing and a full-image static mask.
It refuses ROI, custom preprocessing scripts, independent uncertainty backends
and vendor devices. Existing planar/stereo experiment formats remain unchanged.
Input bytes, settings and order define identities; path locations are provenance.

## Capture complete processing settings

```julia
using Hammerhead

files = sort(readdir("recording"; join=true))
pairs = image_pairs(files; mode=:paired)
passes = multipass_parameters([64, 32, 16]; padding=true,
    final=(uncertainty=true,))
recipe = EnsemblePIVRecipe(passes;
    preprocessing=[PreprocessStep(:highpass_filter; sigma=3)],
    backend=:cpu, image_type=Float64, threaded=false)
record = EnsembleExperimentRecord(pairs, recipe)
save_experiment("ensemble-record.jld2", record)
```

Keep the full pass schedule. Requested iteration budgets and convergence
tolerances remain part of the settings even though the ensemble driver executes
one pooled sweep per pass and ignores those iteration controls. Resolve effort
presets using their ensemble semantics before constructing an explicit recipe;
do not substitute the ordinary sequence preset.

## Replay and observe progress

```julia
record = load_ensemble_experiment("ensemble-record.jld2")
run = replay_experiment(record;
    output="ensemble-results.jld2", run_record="ensemble-record.jld2",
    record_diagnostics=true)
```

Replay validates settings, all input identities, environment and destinations
before producing output. The output may replace an existing destination after
successful staged processing, but may not alias an input or protected record.
Computation failure or cancellation leaves a pre-existing destination untouched;
this does not promise preservation through every publication or filesystem error.
Replacing a historical result invalidates that older run's saved output hash.
`allow_environment_change=true` explicitly
permits a different execution environment and records the actual environment;
it does not promise identical numerical output across environments.

Progress describes completed pair accumulation work across all passes, including
masked or source-gated contributions. It is not result persistence, informative
sample count or effective sample size. Cooperative cancellation is checked
between joined pair computations and before result publication; current
computation/preflight/I/O can pause the caller or GUI. A cancelled attempt has no
published pooled result. Callback or processing failures retain their original
exception and attempt to save failed-run metadata when a run-record destination
was supplied. Secondary history-save failures are logged.
This workflow reruns from the beginning and does not resume partial accumulators.

Native output and optional run history are separate publications. If history
saving fails after an otherwise completed or cancelled replay,
`EnsembleRunRecordError` carries the accurate terminal metadata in `error.run`
and the writing exception in `error.cause`. A completed native result remains
completed even if its history could not be saved. Ordinary processing failures
retain their original exception instead of this wrapper. Save or inspect the
carried run explicitly; do not relabel it as failed processing.

## Verify and report the completed output

```julia
verify_ensemble_experiment_run(record, run; verify_results=true)
report = quality_report(record, run; verify_inputs=true)
save_quality_report("ensemble-quality.toml", report)
display(report)
```

The new associated overload emits [report format 5](../reference/ensemble_experiment_quality.md).
It verifies the native output hash, exact one-result mapping, recipe grid/mask/
scale, raw measurement association and requested execution packet before and
after aggregation. The report retains separate ordered input-pair, processed
contribution and published-result counts. Stored uncertainty availability is
finite/nonnegative array availability, not estimator applicability or coverage.

Use `include_ensemble_execution_diagnostics=false` for a stored-field-only
format-5 report. Run association and requested-companion verification still
occur; omitting the display section does not skip integrity checks. Failed or
cancelled runs cannot be reported as completed pooled results.

```julia
reopened = load_quality_report("ensemble-quality.toml")
data = quality_report_data(reopened)
@assert data["provenance"]["published_results"] == 1
```

Loading a saved report validates historical metadata; it does not freshly verify
the source. To locate a moved native file, pass `output="local/moved.jld2"` to
the verifier/report overload. Replay uses the recorded input locators; moved
images require capturing a new record with their local paths. Unchanged ordered
input bytes and settings can retain their scientific identities, but the new
record has a different full creation-record binding. Known local dependency paths are protected on report
saves; add hidden dependencies through `protected_paths` if needed.

## Run a complete local example

This executable example uses the committed Challenge A pair only to exercise
save/reopen/replay/report. One pair is not ensemble averaging evidence or a
stationarity test.

```@example saved_ensemble
using Hammerhead

mktempdir() do directory
    root = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
    pairs = [(joinpath(root, "A001_1.tif"), joinpath(root, "A001_2.tif"))]
    parameters = PIVParameters(window_size=64, overlap=32, padding=true,
        uod_enable=false, validation=(), replace_outliers=false)
    record = EnsembleExperimentRecord(pairs,
        EnsemblePIVRecipe(parameters; threaded=false))
    record_path = joinpath(directory, "experiment.jld2")
    save_experiment(record_path, record)
    reopened = load_ensemble_experiment(record_path)
    run = replay_experiment(reopened;
        output=joinpath(directory, "pooled.jld2"), run_record=record_path,
        record_diagnostics=true)
    report = quality_report(reopened, run; verify_inputs=true)
    report_path = joinpath(directory, "quality.toml")
    save_quality_report(report_path, report)
    data = quality_report_data(load_quality_report(report_path))
    @assert data["provenance"]["published_results"] == 1
    @assert data["provenance"]["input_pairs"] == 1
    (; report_format=data["quality_report_format_version"],
       input_pairs=data["provenance"]["input_pairs"],
       published_results=data["provenance"]["published_results"])
end
```
