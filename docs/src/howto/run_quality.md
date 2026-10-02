```@meta
CurrentModule = Hammerhead
```

# Save a run-quality report

Use [`quality_report`](@ref) to summarize the fields and flags actually stored
in completed planar/stereo result files. The report contains aggregate counts,
dimensionless fractions and provenance, with explicit reasons for unavailable
diagnostics. It does not establish measurement accuracy or uncertainty coverage.

For a completed experiment run, use the record and run together:

```julia
record = load_experiment("experiment.jld2")
run = last(record.runs)
report = quality_report(record, run)
show(stdout, MIME"text/plain"(), report)
save_quality_report("run-quality.toml", report)
```

This verifies the recipe/input/run metadata, completed result count and recorded
output SHA-256. It reads the output lazily and records the recipe, input, run and
run-environment identities. It does not reopen input images or evaluate a
preprocessing script. The association describes the verified recorded output;
the report cannot recover measurement events that were never persisted.

For an independent result file, use `quality_report(ResultFile(path))`. This
records the source file's identity but labels experiment association
`unassociated`. A finite iterator or vector also works; anonymous iterators do
not expose their hidden source files. Supply their known dependencies as
`protected_paths` when saving.
`CheckpointResults` and its standard array views expose their payload paths for
overwrite protection, but a generic report over that index still has
`unassociated` experiment provenance.

The following executable example constructs a small stored field, writes a
native result file, summarizes it lazily and round-trips the TOML report:

```@example run_quality
using Hammerhead

parameters = PIVParameters(window_size=4, overlap=2, uncertainty=true)
result = PIVResult([1., 2.], [1., 2.],
    [1. 2.; 3. NaN], ones(2, 2), ones(2, 2), ones(2, 2),
    [0.1 0.2; -0.1 NaN], fill(0.2, 2, 2),
    BitMatrix([false true; false false]),
    BitMatrix([false false; false true]), parameters)

mktempdir() do directory
    source = save_results(joinpath(directory, "vectors.jld2"), [result])
    report = quality_report(ResultFile(source))
    destination = save_quality_report(joinpath(directory, "quality.toml"), report)
    restored = load_quality_report(destination)
    data = quality_report_data(restored)
    (association=data["provenance"]["association"],
     counts=data["groups"]["planar"]["counts"],
     masked_fraction=data["groups"]["planar"]["fractions"]["masked_fraction"])
end
```

Read `quality_report_data(report)` for a detached dictionary suitable for
scripts and GUI displays. A fraction with a zero denominator has
`available=false` and no `value`; avoid displaying it as zero percent. Stored
uncertainty is counted as numerically available only when finite and
nonnegative. Separate counters expose finite negative values and nonfinite
values. Numerical availability says nothing about which measurement a stored
uncertainty belongs to after alternative-peak substitution or vector filling.

Planar and stereo groups remain separate. Fractions sum node counts across
entries, so larger grids carry proportionally more weight. Stereo counts use
the reconstructed fields and their union mask/outlier flags, not the two
camera histories. Attached scales do not change these dimensionless counts.
See [the metric definitions](../reference/run_quality.md) for each denominator.

Reporting retains aggregate counters and one result payload at a time, with a
transient next entry during iteration. Native entries may themselves contain
large correlation planes or camera fields. A `ResultFile` retains an index of
O(entries) keys, and associated reports retain O(inputs/runs) locator strings;
they do not retain the result sequence or embedded image/recipe arrays.

Keep native result files unchanged throughout reporting. File stamps and
before/after hashes detect changes, but this is not a concurrent-writer
snapshot. Loading a TOML report validates its schema and consistency without
reopening any recorded files; it records a past verification rather than
performing a new one.

Saving validates and serializes before opening the destination, and rejects
same-file aliases of known result/input/script/record paths, including hard
links where the filesystem supports them. For an anonymous iterator, use
`save_quality_report(path, report; protected_paths=[source_path])`. Ordinary
report files may be overwritten. Writes are not atomic publication or resume
state, and an I/O failure can leave a partial destination. Keep recipe
comparison, peak-locking analysis and sensitivity experiments separate from
this stored-field summary.
