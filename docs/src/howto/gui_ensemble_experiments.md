# Save and replay a planar ensemble in the GUI

Open **saved ensemble…** in the planar batch form, or create the separate
workflow directly:

```julia
using HammerheadGUI
batch = BatchRunner(files=["A1.png", "B1.png", "A2.png", "B2.png"],
                    window_schedule=[32, 16])
fig = ensemble_experiment_workflow(; batch)
```

In **Files**, choose the next snapshot's CPU or KA backend and Float32 or
Float64 precision, then select **snapshot ensemble batch**. KA uses the core's
CPU device here. The snapshot captures the ordered file pairs, effective ensemble
passes, built-in preprocessing, static mask and optional physical scale. Effort
presets expand using the ensemble schedule and full image dimensions. ROI,
custom callbacks/scripts and in-memory frames refuse with a readable error;
clear the ROI or use supported file inputs before snapshotting. They are never
silently discarded.

**Open experiment** preserves a complete saved `EnsemblePIVRecipe` without
reducing it to the batch form. The backend and precision menus affect only a
future form snapshot. Complete recipe pages retain all requested pass settings,
including iterations and tolerances that ensemble processing ignores. Large
embedded masks/backgrounds appear as shape/type/content summaries. Use the page
buttons to read full paths, identities and errors at 900×600 and larger sizes.

Choose result and optional run-record destinations, then switch to **Replay**.
An environment difference requires an explicit override. Recorded environments
are inspected, never activated. **Record pooled execution** saves the existing
pooled diagnostics companion with the output. Replay captures the full recipe,
paths and options before notifying observers or scheduling its task; subsequent
choices apply to a later request.

Joined contributions count one input pair per scheduled pass. For two input
pairs and three passes the budget is six contributions, followed by **one**
pooled result. Contribution progress is not a result-publication count.
**Cancel between contributions** requests cooperative cancellation before an
input load, after joined contribution work, or before pool publication. Even a
request after the last contribution can prevent publication. Cancellation stays
busy through cleanup and optional history saving. It produces a cancelled run,
not a resumable checkpoint. Preflight, numerical work and I/O can pause the GUI;
task scheduling and hidden render tests do not establish desktop responsiveness.

The latest attempt, selected historical run, active captured request, independently
displayed result and retained report have separate identities. Select an older
completed run, then choose **view completed results** to open its verified lazy
result rather than the latest destination. Selection alone reads metadata.
Failure/cancellation leaves an earlier selection and independent display intact.
If the pool was published successfully but run-history saving or reopening fails,
the status names that history error and retains the known completed run and its
publication counts. Its verified result and report remain available; saving a
new record can retain this in-memory history.

In **Reports**, select whether to verify current input bytes and include pooled
observations, then **save quality report**. The request is captured before the
path dialog. Associated ensemble reports use format 5 even when pooled observations
are omitted; they verify the recorded output/run association at generation time.
Known input, history and output paths are protected against replacement. Failed
report actions preserve the previously displayed report and its own run/recipe/input
IDs. Verification is not a continuing guarantee against file edits.

The result explorer converts raw measurement fields to physical display once and
can inspect recorded pooled sweeps/support. Neither contributions nor pooled
support establish stationarity, an effective independent sample size, convergence,
measurement accuracy or uncertainty coverage. There is no per-input vector history
or saved-ensemble checkpoint/resume workflow.
