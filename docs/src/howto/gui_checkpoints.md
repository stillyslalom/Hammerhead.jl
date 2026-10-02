```@meta
CurrentModule = HammerheadGUI
```

# Resume planar checkpoints in the GUI

**Goal:** stop at a committed pair boundary and resume the remaining planar
recording, retaining an exact recipe and the previously committed results.
This workflow uses the separate [core checkpoint store](checkpoints.md).
Ordinary experiment replay still starts from pair 1.

Open **checkpoint / resume…** from the saved-experiment workflow, or construct
an empty workflow that can open an existing store:

```julia
using HammerheadGUI

display(checkpoint_workflow())
```

The experiment link copies its complete record as a creation candidate.
**Choose complete recipe…** loads another saved experiment without changing
its passes or projecting it onto the batch form. With
`checkpoint_workflow(; batch=runner)`, **snapshot batch** uses the same exact
effective-pass, preprocessing, mask/ROI and scale snapshot as the saved-experiment
bridge. A script-based candidate remains inspectable with a creation refusal;
**open checkpoint…** stays available. Checkpoint restart supports built-in
preprocessing only and never evaluates or reconstructs arbitrary callbacks.

## Create, cancel and resume

Select separate new or empty **metadata directory…** and **per-pair result
directory…**, then **create checkpoint**. Existing user data, aliases, changed
inputs and incompatible software are refused by the core. Opening/creating a
store verifies it before replacing the controller's previous valid snapshot.
Its recipe pages show every saved pass and option; long paths and descriptions
remain reachable with **previous** and **next**.

**Resume committed prefix** validates the exact recipe, ordered input content,
software environment and committed payloads, then starts at the first missing
absolute pair. There is no environment override. The counter reports published
commit descriptors, including earlier pairs. The controller keeps metadata and
no growing result vector.

**Cancel after current pair** sets an independent cancellation request. The
pair in flight commits before cancellation is acknowledged. Cancelling after
the final pair records completion. Native cancellation is shown as `:cancelled`
and does not become an error. A progress callback exception or processing error
is a failure; the original exception and verified committed prefix remain
available. **Refresh checkpoint state** is an explicit idle action that scans
committed bytes, rather than a constant-time or per-pair status poll.

The GUI uses a cooperative Julia task and yields between committed pairs.
Initial verification and current-pair computation can pause rendering; this
slice has no in-pass progress, immediate interruption or background worker that
mutates GUI Observables.

This executable controller example uses committed fixtures without a window:

```@example gui_checkpoints
using HammerheadGUI
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(p -> endswith(lowercase(p), ".tif"),
                    readdir(directory; join=true)))
pair = (files[1], files[2])
record = ExperimentRecord([pair, pair],
    PIVRecipe(multipass_parameters([32, 16]; padding=true);
              roi=ROI(1:64, 1:64), image_type=Float32))

mktempdir() do work
    controller = CheckpointController(record;
        checkpoint_dir=joinpath(work, "metadata"),
        output_dir=joinpath(work, "per-pair-results"))
    create_checkpoint!(controller)
    start!(controller; async=false,
           progress=(committed, total) -> cancel!(controller))
    stopped = controller.checkpoint_status[]
    prefix = checkpoint_explorer(controller)

    start!(controller; async=false)
    output = export_checkpoint_results!(controller, joinpath(work, "vectors.jld2"))
    (stopped=stopped, fixed_prefix_pairs=nframes(prefix),
     final_progress=controller.progress[], data_complete=controller.data_complete[],
     exported_pairs=length(load_results(output; lazy=true)))
end
```

## Recover a stopped writer

After process termination, a writer lock or unfinished attempt may remain.
Stop the former writer before checking **former writer has stopped**, then
resume. This is your explicit assertion, not an automatic stale-PID check.
It resets after every open/create/start and after execution; each later recovery
requires a new assertion. Known live writers in the same process are refused,
and concurrent recovery is unsupported. Recognized old lock publication states
are archived; unfamiliar files are preserved and refused.

The checkpoint-state page separates **data complete** from native attempt status.
All pairs can be committed while an attempt is unfinished or failed. Recovery
records interruption without inventing a finish time and can finalize a complete
prefix without recomputing it. This view shows current verified state and the
last returned attempt in the current session; the detailed saved history stays
in the core store. Local rename publication and process recovery do not establish
power-loss durability or network-filesystem guarantees.

For relocated input files, supply an equivalent core record explicitly through
the controller API: `start!(controller; record=relocated_record)` or assign
`controller.resume_record[]`. Reopening with an explicitly relocated result
directory uses `open_checkpoint!(controller, path; output_dir=relocated_output)`.
Those optional relocation choices are API operations in this first view; exact
content identities and the strict software environment still apply.

## Browse a fixed prefix or export

**Browse fixed prefix** opens a nonempty verified `CheckpointResults` index in
the result explorer. It keeps one physical display frame and the current
frame's derived fields. The index length is fixed: after resume, explicitly
open another explorer to see later commits. Empty prefixes and busy execution
disable browsing. Changed/unreadable entries report an error while retaining
the prior displayed frame.

**Export complete native file…** is enabled when all pairs have committed,
independent of the last attempt status. Select a fresh destination outside both
owned directories. Core source/record/payload alias guards and an exclusive
destination lock protect publication. Failed export leaves the checkpoint
intact; after a terminated export leaves its lock, use another fresh path.
Concurrent outside writers are unsupported. Export is optional and uses
additional disk space; it does not replace the authoritative restart store.

See the [GUI checkpoint API](../reference/gui_checkpoints.md) and
[core checkpoint reference](../reference/checkpoints.md) for the schema and
publication boundaries. Checkpoint controls do not add execution diagnostics or
quality-report integration to per-pair checkpoint results in this slice.
