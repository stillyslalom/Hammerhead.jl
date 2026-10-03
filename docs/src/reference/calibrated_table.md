```@meta
CurrentModule = Hammerhead
```

# Calibrated scattered tables

`export_calibrated_table` writes a paired CSV and version-1 TOML companion for raw
PTV, ordinal tracking or timed tracking. It is a separate entry point from
`export_table`. Use the [how-to](../howto/calibrated_scattered_export.md) for affine
vector semantics, time choices, diagnostic limits and publication recovery.

The companion records the applied Float64 matrix and offset, original coefficient
precision, source/output conventions, opaque coordinate-frame label, units,
timing policy, diagnostic availability, CSV schema/hash/row count and protected
locators. Timed exports also preserve the original exact acquisition snapshot.
The CSV keeps the existing table columns as a prefix and adds timed context,
`vector_quantity`, `vector_unit`, `match_residual_pixel` and
`match_residual_pixel_unit`. Its schema marker is
`hammerhead-calibrated-scattered-table-1`; these paired artifacts are not native
result files accepted by `load_results`.

The metadata verifier checks structural integrity and optional CSV structure/hash.
It does not verify numerical result semantics, source bytes, physical calibration
accuracy or authenticity. A metadata object contains no result or CSV row payload;
`calibrated_table_data` returns detached data with verification-at-load status.
Recorded `protected_locators` remain verbatim provenance, including foreign
absolute locators. Computed `local_protected_paths` selects locally interpretable
sources and the local consumed/associated artifacts for protection by later
writers. Source relocation is explicit; incompatible relative CSV separator
syntax requires receiving-host `csv_path`. Formats and integrity encoding remain
version 1. See the how-to for the conservative UNC ambiguity rule and test scope.
Pair publication is sequential, not atomic.

```@autodocs
Modules = [Hammerhead]
Pages = ["calibrated_table.jl"]
Private = false
```
