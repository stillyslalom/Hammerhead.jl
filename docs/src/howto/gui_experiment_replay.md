```@meta
CurrentModule = HammerheadGUI
```

# Monitor and cancel a saved-experiment replay

Open the [saved-experiment workflow](gui_experiments.md), choose a result output
and optional run-record destination in **Files**, then select **replay exact
recipe** in **Replay**. Cancel, written-pair progress and the status preview stay
visible when switching sections. Full status and destination strings remain in
the text pages, which adapt to the window size. The
controller captures the complete recipe, destinations, environment policy and
custom preprocessor before notifying observers or scheduling execution. Editing
those choices while it runs affects a later request.

**Written pairs** counts pairs whose result, requested processing details and
source labels have been written to the native output. It starts at zero and
updates after each write. It is not a durability guarantee, a per-pass progress
estimate, or a resumable checkpoint. Input verification and work inside a pair
have no intermediate progress notification.

Select **cancel after current pair** to request cancellation. The controller stays
busy until the next completed-pair boundary and through prefetched loading,
output closure, hashing and run-history handling. If cancellation arrives after
the final pair write, the attempt completes successfully. A request made before
scheduled execution starts leaves output and history untouched. Idle requests
have no effect.

An earlier cancellation can leave a readable native output prefix. With a
run-record destination, its core version-1 run entry has status `:failed`, the
written-pair count, and a cancellation exception message. The GUI labels this
attempt `:cancelled`; it does not create a new persisted status. Without a
run-record destination, failed run metadata is unavailable. Result exploration
and quality-report actions require a completed latest run. Replaying again
starts from pair 1; use the separate checkpoint workflow when resume is needed.

```julia
using HammerheadGUI

controller = ExperimentController("experiment.jld2")
controller.output_path[] = "vectors.jld2"
start!(controller; progress=(written, total) -> @info "Written pairs" written total)
# Request from an application event or observer:
cancel!(controller)
# Read controller.progress[], controller.state[], and controller.error[].
```

`async=true` schedules a cooperative Julia task, with a yield between pair
writes. It does not move Observable updates to a worker thread. Preflight,
correlation, file I/O and cleanup can pause rendering; hidden rendering tests do
not establish interactive responsiveness. Cancellation is not an interruption of
those operations.

Startup or progress observer exceptions produce a failed attempt. Terminal
notifications assign final values while suppressing observer exceptions so busy
state is released and the original replay exception remains available. A
captured optional progress callback that throws is an ordinary replay failure,
even if it also requested cancellation. See the [core replay guide](experiments.md)
for writer callback ordering and partial-output semantics.
