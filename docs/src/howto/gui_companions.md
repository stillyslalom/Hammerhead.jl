# Inspect recorded processing details

Open a completed native file lazily, then enable **recorded processing details**
in the result explorer. The details panel shows the selected frame's execution
counts and the selected grid node's recorded measurement history. **Previous
details** and **next details** expose every line when the report needs several
pages. Turning the toggle off releases the packet and returns the panel's
vertical space to the plot.
While recorded details are open, a profile's line remains overlaid but its
separate graph is hidden. Close details to show that graph; this prevents two
lower panels from crowding the plot and controls.

```julia
using HammerheadGUI
explorer = ResultExplorer("recorded-results.jld2"; lazy=true)
set_companion_inspection!(explorer)
explorer.selection[] = CartesianIndex(2, 3)
println(companion_summary(explorer))
println(describe_companion_selection(explorer))
figure = result_explorer(explorer)
```

The file must have recorded optional
[measurement history](measurement_history.md) and/or
[execution diagnostics](execution_diagnostics.md) during processing. Missing
entries say **not recorded**; current outlier flags never substitute for absent
history. Stereo entries show optional per-camera execution observations; PTV
and tracking entries say unsupported. Eager/in-memory inputs
have no checked native association merely because their title names a file.
Checkpoint companion inspection is not supported.

Before changing the visible frame, the controller loads the raw result and its
recorded data, verifies history against that raw result, then converts the result
for physical display. Selection uses the same grid `[row, column]` index through
conversion. Primary displacement, primary residual and raw coordinates are
labeled **px**; the ordinary selection panel labels the returned values in
their displayed displacement/velocity units. ROI coordinates remain in the
original image frame.

History shows the first observed rejection stage, actual accepted alternative
rank, median attempt/assignment, primary restoration, final origin/flag and
stored uncertainty numerical status. Median assignment can leave a value
unchanged or produce a nonfinite value. Internal fills can later be undone.
The uncertainty statistics use the final deformed windows; alternative peaks
are not reestimated and filled uncertainty is not propagated. Neither finite
sigma nor primary origin establishes applicability, accuracy or coverage.

Recorded execution counts include actual sweeps, tolerance checks and primary
residual summaries. They are not verified against displayed vector values by
the execution format-1 companion. History separately verifies numerical content;
the two packets have independent UUIDs. Tolerance outcomes do not establish
measurement validity, and an empty eligible comparison can meet tolerance.

Stereo execution companions are verified against the raw reconstructed and both
retained camera measurement fields before physical display conversion. The
panel labels **Camera 1** and **Camera 2**, with actual pass sweeps, checks,
stop reasons and primary residual summaries in **dewarped px**. Those residuals
are not converted to world units or reconstructed into a 3C residual. There is
no stereo per-node measurement history: selecting a node shows the ordinary
displayed vector values and explicitly states that recorded node history is
unavailable. Calibration, input images, synchronization, uncertainty applicability
and measurement accuracy are not verified by this inspection.

A load or verification error leaves the previous frame, selection, display and
recorded-data bundle visible and reports the error. A failed attempt to enable
inspection preserves the previous mode too. The setter and direct
`frame[]`/`companion_enabled[]` writes use the same preflight rules. Fix/reopen a
completed source after an external change; these indexes do not follow writers.

The controller retains one physical display result, one current history packet,
and the current execution metadata. It releases the raw result after conversion,
with retained camera fields remaining part of the ordinary stereo result,
evicts previous packets on navigation and clears them when disabled. Inspection
refreshes scan/hash the current history and physical display arrays to detect
edits, so inspection costs O(grid nodes). They do not reload results or copy a
full history packet per click. This is not continuous mutation monitoring;
after editing display arrays, disable and reenable inspection to reload verified
data. Caller-retained objects/copies have their own memory cost.

For aggregate event counts, the experiment workflow has an independent
**include recorded history in report** toggle. It opts into
[quality report format 2](run_quality.md); its unchecked default remains format 1.
The separate **include recorded execution in report** toggle selects format 3;
history can also be included. Counts have explicit recorded/missing/unsupported
coverage, with primary-support counts kept separate for planar and each stereo
camera; report v3 does not aggregate residual amplitudes. Verification describes the files at report generation time, not future
edits. The displayed report identifies its own saved run, recipe and inputs;
a failed scan/save retains that previous report with its identity.

```julia
controller = ExperimentController("experiment-with-run-history.jld2")
report = save_experiment_quality_report("quality-execution.toml", controller;
    include_execution_diagnostics=true, include_measurement_history=true)
```

The saved experiment workflow currently records/replays planar recipes. It can
report their recorded planar execution entries; a generic native stereo file's
format-3 report is generated through the core `quality_report(ResultFile(path);
include_execution_diagnostics=true)` API. No stereo experiment replay is implied.
