# Inspect ensemble pooling and report a native file

Open a completed native results file lazily:

```julia
using HammerheadGUI
explorer = ResultExplorer("ensemble-results.jld2"; lazy=true)
figure = result_explorer(explorer)
```

Enable **recorded processing details**. The ensemble packet is checked against
the selected raw measurement and geometry before conversion to display units.
After editing display arrays, disable and reenable details to reload the source.

The drawer distinguishes ensemble pooling from ordinary pair iterations. Each
ensemble pass performs one pooled sweep. It shows requested iteration settings,
masked/source-gated/accumulated contributions, and correlation-plane categories.
Contributor summaries count eligible nodes with zero, some, or all finite nonzero
planes. Primary residuals use processing pixels before predictor addition;
pooled uncertainty counts are recorded before validation and availability cleanup.

Selecting a node shows its ordinary displayed values; ensemble contributor
summaries apply to the pooled pass. Entries are identified by their recorded
packet family. If navigation or inspection fails, the previous frame and details
stay visible. Choose a readable entry or disable details to recover.

The explorer keeps the current display and packets, releasing previous payloads
on navigation. Refresh checks current display content.

Click **whole-file quality report** to open its separate window. Choose
**Recorded ensemble pooling observations** for files containing pooled execution;
ordinary execution and final-sweep history can be included too. Then choose
**generate report** or **generate / save TOML**. The report scans every native
entry, including frames you have yet to visit.

```julia
report = explorer_quality_report(explorer;
    include_ensemble_execution_diagnostics=true)
save_explorer_quality_report("quality.toml", explorer;
    include_execution_diagnostics=true,
    include_ensemble_execution_diagnostics=true)
```

The report separates recorded ensembles, recorded ordinary iterations, and
entries with unknown execution metadata. Current-field metrics use node-weighted
populations; pooled counters use their labeled window/pair and final-node counts.
The default report contains current-field metrics; pooled observations select
format 4.

The report window checks and snapshots the complete native entry mapping and
your options. Reopen the explorer if its key list has been edited. Failed scans,
saves, or cancelled dialogs retain the prior report, its source path, and digest.
Use text pages for complete messages and summaries. For a report tied to a saved
run, use the [saved ensemble workflow](gui_ensemble_experiments.md). Generation
checks file content before and after scanning; a loaded report describes that
past verification.

Use a lazy explorer opened on a completed native results file for this report.
Scanning and hashing finish as one action. Choose a fresh report destination;
known source aliases are protected. If a write fails, inspect a possible partial
TOML and retry at a new path. The
[ensemble report reference](../reference/gui_ensemble_companions.md) documents
coverage and file checks.