# Compare saved recipes on one representative pair

Open **compare a representative pair** from the saved-experiment workflow. The
new window captures the current complete record as **before**. Open a separate
saved experiment as **after**, then enter one pair index for each record. These
are explicit choices: filenames, list positions and timestamps do not prove
that two selected pairs have the same ordered image content.

Choose **raw displacement (px)** or **physical velocity**, then run **compare
selected pair**. Physical comparison requires identical attached scale factors
and unit labels. The unchecked environment override is separate from replay's
override; enabling it reruns both recipes under one recorded current environment.
Custom preprocessing scripts/callbacks are unsupported and never executed here.

The core verifies selected ordered bytes, sizes and decoded dimensions before
either computation, retains complete recipe settings, and checks inputs and
software again afterward. It pairs only numerically exact common grid centers
in raw image coordinates, including ROI offsets. It performs no interpolation
to make different grids match. Differences are **after minus before** and
describe recipe sensitivity; they do not establish accuracy or uncertainty
coverage. Empty common/eligible populations are explicitly unavailable.

```julia
using HammerheadGUI
controller = RecipeComparisonController("before.jld2", "after.jld2")
set_comparison_pairs!(controller, 2, 5)
controller.basis[] = :pixels
compare!(controller; async=false)
println(comparison_summary(controller))
save_comparison_report!(controller, "pair-comparison.toml")
figure = recipe_comparison(controller)
```

The **current request** page shows current record IDs, chosen pair indices and
complete selected paths/content identities. Other sections describe only the
**last report**, with its own before/after pair indices, recipe/input IDs and
basis. Settings, populations and provenance pages expose every line. Embedded
mask/background changes appear as shape/type/content summaries rather than
large array dumps. Native-grid populations remain separate; common differences
use finite, unmasked, currently unflagged vectors in both results. Stored UQ
comparisons further require finite nonnegative components, without certifying
applicability or coverage. Current flags do not reconstruct replacement history.

Changing choices or a failed comparison preserves the prior report with its own
historical labels. **Save last report** exports that report without recomputing
current choices. **Open report** loads a validated past TOML report for read-only
inspection; it does not reopen/reverify input files or reconstruct runnable
recipes from digest summaries. Loading failures preserve the previous report.
Saving protects known inputs, records, scripts, result files and captured GUI
destinations, including filesystem aliases. Writes are not atomic publication.

The controller freezes both record snapshots, indices, basis, environment
override and protected paths before notifications or task scheduling. Busy
record/report actions, pair edits and duplicate runs are refused. Basis and
environment controls can change the next attempt, while the active request
remains frozen. Invalid visible pair text is checked again on Run, so it cannot
silently reuse old indices. Asynchronous scheduling does not promise responsive
CPU preflight/computation, live progress or cancellation.

Only two recipe snapshots and metadata from the last report remain after a run.
No comparison image or numerical result arrays are retained. Copies of embedded
recipe masks/backgrounds still have their normal storage cost. This independent
comparison does not append experiment run history or replace production output.
