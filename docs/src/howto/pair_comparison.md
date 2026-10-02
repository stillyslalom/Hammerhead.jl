```@meta
CurrentModule = Hammerhead
```

# Compare recipes on a representative pair

Use [`compare_recipe_pair`](@ref) to rerun two saved planar recipes on the same
explicitly selected ordered image pair. The report combines settings changes,
current flags and stored uncertainty with numerical differences at exactly
shared vector centers. These differences describe processing sensitivity;
they do not establish which recipe is more accurate.

Choose the pair based on the behavior you want to examine. The library does
not select a representative frame automatically. The two recordings may have
different pair lists; the selected pair must have identical ordered file bytes,
sizes and decoded dimensions. Relocated copies are accepted. Swapping the two
frames or changing their encoding changes that identity.

This executable example uses the committed Challenge A pair, a small ROI,
and two explicit window/preprocessing recipes:

```@example pair_comparison
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(path -> endswith(lowercase(path), ".tif"),
                    readdir(directory; join=true)))
pair = (files[1], files[2])
before_recipe = PIVRecipe(PIVParameters(window_size=32, overlap=16, padding=true);
                          roi=ROI(1:64, 1:64), image_type=Float32)
after_recipe = PIVRecipe(PIVParameters(window_size=16, overlap=8, padding=true);
    roi=ROI(1:64, 1:64), image_type=Float32,
    preprocessing=[PreprocessStep(:highpass_filter; sigma=3)])
before = ExperimentRecord([pair], before_recipe)
after = ExperimentRecord([pair], after_recipe)
report = compare_recipe_pair(before, after; pair_indices=(1, 1))
data = pair_comparison_data(report)
(settings_changes=length(data["settings_changes"]),
 common_nodes=data["common"]["counts"]["nodes"],
 jointly_valid=data["common"]["counts"]["valid_both"],
 differences_available=data["common"]["velocity_difference"]["available"])
```

Each complete recipe runs with its saved precision, pass schedule, preprocessing,
mask, ROI, backend and deformation options. Only selected input files are
opened. Unrelated files need not still exist, although the full record's
metadata and recipe must remain intact. Comparison supports the built-in
planar recipe scope; external scripts and custom callbacks are refused.
It writes no result file and does not append execution history to either record.

## Read the populations before the differences

`native.before` and `native.after` summarize each recipe's complete output
grid using the [run-quality counting conventions](../reference/run_quality.md).
Their fractions can describe different spatial populations and are not a paired
yield difference. `common` first intersects raw x/y coordinates by exact value
in the original image frame, including ROI offsets. It then counts mask states
and the four before/after validity combinations on nodes unmasked in both.
Validity requires finite u/v and no current outlier flag.

Velocity differences use only nodes valid in both results. A finer grid can
share centers with a coarser grid while adding many unmatched nodes. A shifted
ROI or differently phased grid can have no common centers. The report then
marks differences unavailable with `no_common_nodes`, instead of interpolating
or comparing different populations. All unavailable metrics omit numerical
values; an empty selection is not zero difference.

Available component metrics contain the mean signed difference and RMS
difference, with **after minus before** as the direction. RMS includes the
mean difference. Vector RMS combines the two component RMS values. These are
differences between processing outputs, not errors against a truth field.

Stored uncertainty comparisons use a further restriction: both components in
both results must be finite and nonnegative at the jointly valid nodes. Zero
uncertainty is numerically available. Read that subgroup's count and the saved
request settings before interpreting its summaries. Stored uncertainty does
not establish measurement association after filling or alternative peaks,
calibrated coverage, or independence between the two recipe errors. The report
therefore leaves normalized differences by uncertainty unavailable.

## Choose the value basis explicitly

The default `basis=:pixels` compares raw displacements in pixels. Differing
attached scales remain visible in the settings/provenance; pixel agreement
does not imply agreement in physical velocity.

For `basis=:physical`, both recipes must have identical `PhysicalScale`
factors and unit labels. Node matching still uses original image pixels;
displacements and uncertainties convert in Float64 using `pixel_size / dt`.
Opaque unit labels are not converted automatically. Missing or incompatible
scales are rejected before loading images. Conversion/subtraction/norm overflow
marks the metric unavailable with `nonfinite_arithmetic`; its raw jointly valid
count remains intact instead of silently dropping overflowing samples.

Comparison requires the recorded creation software environment by default.
To intentionally rerun both recipes under the current environment, use
`allow_environment_change=true`. The report records the actual common
environment separately and checks its identity before and after execution.
Keep input and software files unchanged throughout comparison; hashes detect
changes but do not provide a concurrent-writer snapshot.

## Save a detached comparison

```@example pair_comparison
mktempdir() do work
    path = save_pair_comparison(joinpath(work, "comparison.toml"), report)
    restored = load_pair_comparison(path)
    pair_comparison_data(restored)["provenance"]["ordered_pair_id"] ==
        data["provenance"]["ordered_pair_id"]
end
```

The separate version-1 TOML report contains primitive metadata and scalar
summaries. `pair_comparison_data` returns a copy suitable for scripts or GUI
displays; text/plain display provides a short human-readable summary.
Reports retain no image, field, mask, background or correlation-plane arrays.
Computation still requires the selected images, workspace and two result
grids; configured retained planes can substantially increase transient memory.

Loading validates a past snapshot without reopening input images or rerunning
PIV. Saving protects both records' known input, record and result paths against
aliases, including after loading; use `protected_paths` for additional
dependencies. Ordinary report files may be overwritten. Report writing is not
atomic publication or restart state. See the
[comparison schema](../reference/pair_comparison.md) for exact metric and
population definitions.
