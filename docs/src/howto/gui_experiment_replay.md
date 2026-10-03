```@meta
CurrentModule = HammerheadGUI
```

# Monitor and cancel a saved-experiment replay

Open the [saved-experiment workflow](gui_experiments.md), choose a result output
and optional run-record destination in **Files**, then select **replay exact
recipe** in **Replay**. Cancel, written-pair progress and status stay visible
across sections. Text pages show full messages and paths. The controller captures
your recipe and choices at the start; edits during execution apply to the next run.

**Written pairs** counts pairs whose result, requested processing details, and
source labels have reached the native output. The counter starts at zero and
updates after each pair write. For resumable committed progress, use the
[checkpoint workflow](gui_checkpoints.md).

Select **cancel after current pair** to finish the pair in progress and stop
at its write boundary. The controller stays busy while loading and output/history
cleanup finish. A request at the final write completes the recording. A request
before scheduled execution starts retains the existing output and history.

Earlier cancellation can leave a readable native output prefix. When a
run-record destination is supplied, the core history records `:failed`, the
written-pair count, and the cancellation message; the GUI shows `:cancelled`.
Choose a completed run for result exploration and quality reports. Another
ordinary replay starts at pair 1; checkpoints resume a committed prefix.

```julia
using HammerheadGUI

controller = ExperimentController("experiment.jld2")
controller.output_path[] = "vectors.jld2"
start!(controller; progress=(written, total) -> @info "Written pairs" written total)
# Request from an application event or observer:
cancel!(controller)
# Read controller.progress[], controller.state[], and controller.error[].
```

`async=true` schedules a cooperative Julia task that yields between pair writes.
Cancel is checked at those boundaries; current input verification, correlation,
file I/O, and cleanup finish first.

A startup or progress callback error produces a failed attempt. The controller
releases its busy state and keeps the original exception in `error[]`. A throwing
progress callback takes priority over a cancellation request. The
[core replay guide](experiments.md) describes callback ordering and partial outputs.