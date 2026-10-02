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

Opt into verified recorded events when the native file contains
[measurement history](measurement_history.md):

```@example quality_history
using Hammerhead, Random
a = rand(MersenneTwister(237), Float32, 48, 48)
p = PIVParameters(window_size=16, overlap=8, n_peaks=3)
mktempdir() do directory
    path = joinpath(directory, "recorded.jld2")
    run_piv_sequence([(a,a), (a,a)], p; output=path,
        record_measurement_history=true, collect_results=false, progress=false)
    report = quality_report(ResultFile(path); include_measurement_history=true)
    saved = save_quality_report(joinpath(directory, "quality-v2.toml"), report)
    data = quality_report_data(load_quality_report(saved))
    @assert data["quality_report_format_version"] == 2
    data["measurement_history"]["counts"]
end
```

This emits report format 2; the default still emits format 1 and its loader
remains compatible. `quality_report(record, run; include_measurement_history=true)`
also verifies that every present packet has the selected recipe/input IDs and
absolute pair index. A present but unassociated or mismatched packet is refused;
a missing packet instead reduces coverage. A generic `ResultFile` report remains
experiment-unassociated even when packet metadata names a recipe.

History counts cover only the final pass's final executed sweep. They distinguish
first observed rejection, flags before alternatives, accepted alternative peaks,
attempted and assigned medians, restored primaries, and final origin. A median
assignment can be nonfinite, numerically unchanged, or undone by restoration.
Missing entries are coverage gaps, not zero-event observations. Event fractions
use only history-covered unmasked nodes; entry coverage uses planar entry counts.
Unknown rejection labels remain unclassified. Earlier-sweep history and
uncertainty applicability/coverage remain unavailable.

Opt-in reports require a direct whole-file `ResultFile` or verified record/run.
Bare iterators, array views, converted GUI result wrappers and checkpoint indexes
have no supported checked companion mapping. Mixed native files explicitly count
unsupported stereo/PTV/tracking history; PTV/tracking entries have no numerical
quality group in format 2. Existing malformed packets or a planar packet attached
to an unsupported result kind are refused. Each selected raw result is loaded
once and its companion verified before aggregation; neither payload is retained
in the report. Source identities and overwrite protections apply to both formats.

The GUI experiment workflow's unchecked **include recorded history in report**
toggle selects format 2. Its default remains format 1. The GUI controller methods
`experiment_quality_report` and `save_experiment_quality_report` expose the same
`include_measurement_history` keyword. The result explorer offers separate
[recorded processing details](gui_companions.md) for one selected frame/node.

Saving validates and serializes before opening the destination, and rejects
same-file aliases of known result/input/script/record paths, including hard
links where the filesystem supports them. For an anonymous iterator, use
`save_quality_report(path, report; protected_paths=[source_path])`. Ordinary
report files may be overwritten. Writes are not atomic publication or resume
state, and an I/O failure can leave a partial destination. Keep recipe
comparison, peak-locking analysis and sensitivity experiments separate from
this stored-field summary.
