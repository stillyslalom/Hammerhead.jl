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

Record optional [measurement history](measurement_history.md) or
[execution diagnostics](execution_diagnostics.md) during processing, then open
the completed native file lazily. Missing entries show **not recorded**. Planar
entries can expose node history; stereo entries expose recorded camera execution.
The panel labels PTV and tracking entries as unsupported.

Recorded ensemble packets show one pooled sweep per pass, requested iteration
settings, and contribution populations. They are checked against the raw result
before physical display. Node selection shows the displayed vector values;
pooled contributor counts are pass summaries. See the
[ensemble inspection reference](../reference/gui_ensemble_companions.md).

Before changing the visible frame, the controller loads the raw result and its
recorded data, verifies history against that raw result, then converts the result
for physical display. Selection uses the same grid `[row, column]` index through
conversion. Primary displacement, primary residual and raw coordinates are
labeled **px**; the ordinary selection panel labels the returned values in
their displayed displacement/velocity units. ROI coordinates remain in the
original image frame.

History shows the first rejection stage, accepted alternative rank, median
attempt/assignment, primary restoration, final origin/flag, and stored uncertainty
status. Read these together to follow how the returned value was produced.
Uncertainty statistics come from final deformed windows; use the
[measurement history guide](measurement_history.md) for their interpretation.

Execution details show actual sweeps, tolerance checks, and primary residual
summaries. History and execution are separate recorded packets with their own
identities. The [execution diagnostics guide](execution_diagnostics.md) explains
the stopping conditions and eligible populations.

Stereo details label **Camera 1** and **Camera 2**, with actual sweeps, checks,
stop reasons, and primary residuals in **dewarped px**. The packets are checked
against raw reconstructed and camera measurement fields before physical display.
Selecting a node shows its ordinary displayed vector values. Read the camera
execution summaries for recorded stereo processing details.

A load or verification error keeps the previous frame, selection, and details
visible. Fix or reopen the completed source after an external change. The public
setter and direct frame/toggle writes use the same preflight checks.

The explorer keeps the current display and packets, releasing previous ones on
navigation or when details are closed. After editing display arrays, disable and
reenable inspection to reload the verified source.

For aggregate counts, use **include recorded history in report** or
**include recorded execution in report** in the experiment workflow. The report
lists recorded, missing, and unsupported coverage, with primary-support counts
separate for planar entries and each stereo camera. The displayed report retains
its saved run, recipe, and input identities after a failed scan or save.

```julia
controller = ExperimentController("experiment-with-run-history.jld2")
report = save_experiment_quality_report("quality-execution.toml", controller;
    include_execution_diagnostics=true, include_measurement_history=true)
```

The planar workflow reports recorded planar execution. The
[saved-stereo workflow](gui_stereo_experiments.md) reports verified camera
observations. For a general native file, use
`quality_report(ResultFile(path); include_execution_diagnostics=true)`.

The lazy native explorer's **quality report** action opens a separate whole-file
report window. Choose history, ordinary execution, or ensemble observations, then
generate the report. It scans every entry and keeps its own source identity.
Use the [ensemble report guide](gui_ensemble_companions.md) for the options.