# Compare saved recipes on one representative pair

Open **compare a representative pair** from the saved-experiment workflow. The
new window captures the current record as **before**. Open a second saved
experiment as **after**, then choose its representative pair. Select pairs with
the same ordered image content so the comparison measures your settings change.

Choose **raw displacement (px)** or **physical velocity**, then **compare
selected pair**. Physical comparison requires matching scale factors and unit
labels. An explicit environment override reruns both recipes under the recorded
current environment. Use recipes with built-in preprocessing for this workflow.

The core checks the selected image content, dimensions, and settings before
computation and checks input/software identities afterward. It compares exact
common grid centers in original-image coordinates, including ROI offsets.
Differences are **after minus before**: use them to inspect recipe sensitivity.
The report lists each field's population and marks empty comparisons unavailable.

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

The **current request** page shows your selected records, pairs, and paths.
Other pages show the **last report** with its own choices and identities.
Settings changes include compact shape/type/content summaries for masks and
backgrounds. Common differences use finite, unmasked, unflagged vectors in both
results; stored uncertainty comparisons additionally use finite nonnegative
components. Each population has its own count.

Changing choices or a failed comparison keeps the previous report available.
**Save last report** exports that captured report; **Open report** displays a
saved TOML report. A failed load keeps the current report. Choose a distinct
destination: inputs, records, scripts, results, and known workflow paths are
protected. If a write fails, check for a partial TOML and retry at a fresh path.

The active request keeps the choices captured at its start. Wait for it to
finish before opening records, editing pair indices, or starting another
comparison. Basis/environment changes apply to the next request. Run parses the
visible pair text again. Verification and computation complete as one action.

The comparison keeps its metadata report separately from experiment run history
and result outputs. See the [comparison reference](../reference/gui_comparison.md)
for matching populations, captured requests, and save behavior.