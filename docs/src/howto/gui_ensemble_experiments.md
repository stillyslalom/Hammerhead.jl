# Save and replay a planar ensemble in the GUI

Open **saved ensemble…** in the planar batch form, or create the separate
workflow directly:

```julia
using HammerheadGUI
batch = BatchRunner(files=["A1.png", "B1.png", "A2.png", "B2.png"],
                    window_schedule=[32, 16])
fig = ensemble_experiment_workflow(; batch)
```

In **Files**, choose CPU or KA and Float32 or Float64, then **snapshot ensemble
batch**. KA uses the core CPU device. The snapshot captures ordered image-file
pairs, effective ensemble passes, built-in preprocessing, static mask, and scale.
Use full-frame acquisitions: clear an ROI and save in-memory frames as files
before snapshotting. Validation gives a readable error for unsupported settings.

**Open experiment** keeps the complete saved `EnsemblePIVRecipe`. Backend and
precision menus configure the next batch snapshot. Recipe pages show every pass,
including iteration/tolerance settings recorded in the recipe. Embedded arrays
appear as compact shape/type/content summaries. Page controls expose full paths,
identities, and messages.

Choose result and optional run-record destinations, then switch to **Replay**.
Review an environment difference before enabling its override. **Record pooled
execution** saves pooled diagnostics alongside the result. Replay captures the
recipe, paths, and options at the start; later choices apply to the next request.

Joined contributions count one input pair per scheduled pass: two pairs and
three passes produce six contributions followed by one pooled result.
**Cancel between contributions** stops at an input/contribution boundary or
before publication. A request after the last contribution can still prevent
publication. The workflow stays busy through cleanup and optional history saving.
Verification, numerical work, and I/O finish before the next boundary.

Choose a completed historical run and **view completed results** to open its
verified lazy result. Failed or cancelled attempts retain your earlier selection
and independent display. If publication succeeded but history saving/reopening
fails, the status shows the history error and keeps the known completed run.
Its result and report remain available; saving a new record can retain this history.

In **Reports**, choose input-byte verification and pooled observations, then
**save quality report**. The captured request checks the output/run association
and writes an associated ensemble report. Known input, history, and output paths
are protected. A failed action keeps the previous report and its own identities.

The explorer displays measurements in the attached physical units and shows
recorded pooled sweeps/support. Read its contribution and eligible-node counts
as descriptions of the pooling operation. The
[ensemble workflow reference](../reference/gui_ensemble_experiments.md) explains
the counters, supported recipes, and report verification.