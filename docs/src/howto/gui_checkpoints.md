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
**Choose complete recipe…** loads another saved experiment with its saved
passes intact. With `checkpoint_workflow(; batch=runner)`, **snapshot batch**
captures effective passes, preprocessing, mask/ROI, and scale. Use built-in
preprocessing for checkpoint restart. A script-based recipe remains inspectable,
and **open checkpoint…** lets you open an existing supported store.

## Create, cancel and resume

Select separate new or empty **metadata directory…** and **per-pair result
directory…**, then **create checkpoint**. The core checks destinations, input
content, and software compatibility before replacing the current store snapshot.
Use **previous** and **next** to inspect the complete recipe and paths.

**Resume committed prefix** checks the exact recipe, ordered inputs, software
environment, and committed payloads, then starts at the first missing pair.
Use the recorded software environment for restart. The counter includes all
published commits, including pairs completed in earlier attempts.

**Cancel after current pair** lets the pair in flight commit before stopping.
At the final pair, the attempt completes. The view shows cancellation as
`:cancelled`; processing or callback errors appear as failures with the original
exception and verified prefix. Use **Refresh checkpoint state** while idle to
verify the currently committed bytes.

The GUI yields between committed pairs. Initial verification and current-pair
computation finish before the next cancellation boundary.

This controller example uses the supplied image pair:

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

After process termination, stop the former writer, check **former writer has
stopped**, then resume. This check is required anew after opening or running a
store. Perform recovery with one owner at a time; the core archives recognized
old lock states and preserves unfamiliar files for inspection.

The checkpoint-state page separates **data complete** from attempt status.
Every pair can be committed while the last attempt is unfinished or failed.
Recovery records the interruption and finalizes a complete prefix using the
existing committed results. The view shows current verified state and the last
returned attempt; the core store retains detailed history.

For moved input files, create an equivalent core record and pass
`start!(controller; record=relocated_record)` or assign
`controller.resume_record[]`. To open moved results, use
`open_checkpoint!(controller, path; output_dir=relocated_output)`. These API
operations retain the exact content and software checks.

## Browse a fixed prefix or export

**Browse fixed prefix** opens the currently verified, nonempty prefix in a
separate result explorer. After resuming, open another explorer to see new
commits. Browse while execution is idle. An unreadable or changed entry leaves
the previously displayed frame available and shows its error.

When **data complete** is true, use **Export complete native file…** and choose
a fresh destination outside both checkpoint directories. The export adds an
ordinary native results file alongside the restart store. If export fails, retain
the checkpoint and retry at a fresh path; after termination, a destination lock
may still occupy the old path.

See the [GUI checkpoint API](../reference/gui_checkpoints.md) and
[core checkpoint reference](../reference/checkpoints.md) for recovery and
publication details. Open the exported native file for ordinary result browsing.