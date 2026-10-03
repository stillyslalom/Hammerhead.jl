```@meta
CurrentModule = HammerheadGUI
```

# Save and reopen experiments in the GUI

**Goal:** save supported planar batch settings, reopen the complete recipe,
replay it, and browse its completed results without rebuilding the settings.
The saved-experiment workflow uses the core version-1 format described in
[Save and replay a planar experiment](experiments.md).

Open the batch form's **saved experiments…** button, or create the workflow
directly:

```julia
using HammerheadGUI

batch = BatchRunner()
display(experiment_workflow(; batch))
```

You can open a saved experiment even when the batch has no files. **Snapshot
batch** captures the current file pairs, exact effective pass schedule,
built-in preprocessing, original full-image mask, ROI, and scale. **Save
experiment…** writes that intact record. Effort presets are expanded using
the full image or ROI dimensions; pairs that need different preset schedules
are refused. In-memory images are outside the file-based record format.

Preprocessing attached with `set_preprocess!(batch, preview)` retains an ordered
built-in recipe and a copied background alongside its processing function.
Later preview edits do not alter that snapshot. The existing batch workflow
uses CPU/Float64 and the current core thread default; those effective settings
are saved explicitly. An arbitrary function cannot be inferred from a closure
and requires an explicit script reference through the controller API.

## Replay an intact record

**Open experiment…** loads the full recipe into a separate read-only workflow.
The recipe pages show every pass field and preprocessing option, and the
history pages show recorded run identities, statuses, outputs and failures.
Use **previous**/**next** to inspect long recipes. Embedded backgrounds/masks
are summarized by shape/type or excluded-pixel count; their full values remain
in the saved record. Opening does not project settings onto the narrower batch
or preprocessing forms, so nondefault fields and repeated operations survive.

Choose a new result output and a run-record destination, then **replay exact
recipe**. An opened record is the default destination for appended run history.
The status reports busy, cancellation requested, cancelled, completed or failed.
Written-pair progress updates after each pair's native writes. **Cancel after
current pair** waits for a written-pair boundary and loading/output/history
cleanup; cancellation after the final write means completion. Replay starts
from pair 1 and does not resume a partial output. See [monitor and cancel
replay](gui_experiment_replay.md) for failure-history semantics. Custom scripts
are never loaded or evaluated automatically. The GUI remains a cooperative
Julia task; preflight and CPU/I/O work can delay rendering.

This executable controller example uses committed image fixtures without
opening a window:

```@example gui_experiments
using HammerheadGUI
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(path -> endswith(lowercase(path), ".tif"),
                    readdir(directory; join=true)))
batch = BatchRunner(files=files[1:2], window_schedule=[32, 16],
                    roi=ROI(1:64, 1:64), pixel_size=0.02, dt=0.001,
                    length_unit="mm", time_unit="s")
preview = PreprocessPreview(files[1]; enabled=[:invert_image])
set_preprocess!(batch, preview)

mktempdir() do work
    path = joinpath(work, "experiment.jld2")
    save_batch_experiment(path, batch)
    controller = ExperimentController(path)
    controller.output_path[] = joinpath(work, "vectors.jld2")
    start!(controller; async=false)
    explorer = experiment_results(controller)
    report_path = joinpath(work, "quality.toml")
    save_experiment_quality_report(report_path, controller)
    (state=controller.state[], pairs=nframes(explorer),
     saved_runs=length(controller.record[].runs),
     quality_report_saved=isfile(report_path))
end
```

**View completed results** checks the recorded result-file content hash and
opens a lazy `ResultExplorer`. It retains one display result plus key metadata,
rather than collecting the whole sequence. Reused or overwritten output is
refused because it no longer represents that recorded run. Keep result files
unchanged while browsing and use distinct outputs to retain earlier runs.

**Save quality report…** verifies the completed run and saves the same TOML
report available to scripts through `quality_report`. Its readable summary
appears in the **quality report** pages. The report scans one result at a time
and records explicit denominators for mask, current-flag, finite-vector, and
stored-uncertainty availability fractions. A flagged finite value is not evidence
of replacement, and finite uncertainty is not an accuracy or coverage claim.
Unavailable measurement history and sensitivity metrics are identified explicitly.
See [saved run-quality reports](run_quality.md) for the complete contract.
The controller equivalents are `experiment_quality_report(controller)` and
`save_experiment_quality_report(path, controller)`; saving protects the experiment
record and known input/result paths. The scan is synchronous and can delay the UI
on large recordings. Opening another experiment or running again clears the
displayed summary.

Unchecked report toggles preserve the stored-field format-1 default. **Include
recorded history in report** selects format 2; **include recorded execution in
report** selects format 3, optionally with history. Missing entries remain
explicit. A report's checks describe generation time; its displayed summary
names the reported run, recipe and inputs. Failed scans/saves keep the prior
summary with that identity. See [recorded processing details](gui_companions.md)
for camera residual units and the separation from per-node history.

## Handle refusals and custom processing

**Checkpoint / resume…** opens the [checkpoint workflow](gui_checkpoints.md).
It takes a copy of the complete current recipe for creating a resumable store,
or opens an existing store. This separate path supports built-in preprocessing,
strict software identity, progress and cancellation between committed pairs.
Ordinary replay output is not adopted as a checkpoint.

Changed inputs/scripts, incompatible software, malformed recipes, and output
aliases are rejected before result output is opened. A preflight failure
preserves existing destinations. Processing failures can leave a completed
native prefix and, when a run-record destination is supplied, failed-run
history. The results and history files are not an atomic transaction; version 1
does not resume from that prefix.

The **allow environment changes** toggle explicitly permits another software
environment. It starts off and resets when opening another record. Each run
records its actual environment separately from recipe creation; matching or
overridden metadata is not a cross-hardware bitwise reproducibility guarantee.

For a custom preprocessor, establish its implementation yourself and reference
the reviewed script before saving:

```julia
using Hammerhead: ScriptReference

reference = ScriptReference("prepare.jl"; entrypoint="prepare_image(image)")
set_preprocess!(batch, prepare_image)
save_batch_experiment("custom-experiment.jld2", batch; script_reference=reference)

controller = ExperimentController("custom-experiment.jld2")
controller.custom_preprocess[] = prepare_image
controller.output_path[] = "custom-vectors.jld2"
start!(controller)
```

The supplied function must correspond to the referenced bytes, preserve the
saved precision/full-image dimensions, and return finite values. External
callback state remains the caller's responsibility. The graphical snapshot
button refuses an unreferenced callback instead of saving an incomplete recipe.
Stereo calibration, PTV/tracking, GPU recipes, acquisition timing, and complete
recipe editing remain outside this saved-planar-recipe workflow.
