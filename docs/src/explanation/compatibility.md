# Compatibility policy

Hammerhead follows semantic versioning. Before 1.0, exported Julia APIs and
the package-native JLD2 representation may change in a minor release; changes
are called out in release notes and backward-compatible constructors are used
where practical. A stable release will not make an incompatible exported-API
or native-format change without a major version bump.

User-visible changes are recorded in the repository's
[release notes](https://github.com/stillyslalom/Hammerhead.jl/blob/main/CHANGELOG.md).
The [release procedure](https://github.com/stillyslalom/Hammerhead.jl/blob/main/RELEASING.md)
describes core-first/GUI-second validation and the dataset, device, and workflow
evidence associated with a candidate release.

JLD2 is the lossless Julia round-trip format. Files carry `format_version`;
readers reject unknown versions rather than silently misinterpreting data.
An empty result vector (or a batch stopped before its first result) is a valid
versioned file and loads as an empty vector.
Users who need long-lived, language-neutral archives should also export the
table or VTK form.

Experiment records use a separate `experiment_format_version = 1`, with explicit
primitive settings and identities rather than persisted recipe objects. They do
not change the result-file schema. Unknown versions, unexpected fields, and
changed recipe/input identities are rejected. There is no implicit migration
from historical planning files or result-only files: those files do not contain
the complete processing recipe. See [experiment replay](../howto/experiments.md)
for the supported planar scope and environment compatibility checks.

Checkpoint metadata has its own `checkpoint_format_version = 1`, separate from
experiment records and native result files. It binds immutable per-pair native
outputs to an ordered recipe/input identity and execution environment. Unknown
versions are rejected; existing result-only prefixes are not adopted as
checkpoints. See [checkpoint recovery](../howto/checkpoints.md) for supported
interruptions and filesystem limits.

Quality reports default to `quality_report_format_version = 1` in language-neutral
TOML. Opt-in native-file history reports use version 2, with explicit recorded,
missing and unsupported populations and verified final-sweep event counts.
Readers accept both versions and validate counters, denominators, provenance,
and unavailable diagnostic reasons. These summaries do not change native result
structures or reconstruct missing history. See [the report schema](../reference/run_quality.md).

Execution diagnostics are optional native-file companions with
`execution_diagnostics_format_version = 1`. Each entry binds scalar pass
observations to a result key; existing result readers ignore the companion.
The diagnostics reader rejects malformed or unknown versions, while absence
means not recorded. Generic result-only copies do not preserve companions.
See [execution diagnostics](../reference/execution_diagnostics.md).

Final-sweep measurement history uses a separate optional native companion,
`measurement_history_format_version = 1`. It records observed validation,
alternative-peak and filling events without changing result structures or the
execution-diagnostics schema. Snapshot integrity and result-key binding are
checked by the history reader; `verify_result=true` additionally loads the
selected payload and verifies its numerical content binding. Metadata-only
loading does not perform that payload check. Result-only copies drop this
companion. See [measurement history](../reference/measurement_history.md).

Planar sequence timing uses optional `pair_timing_format_version = 1`
companions. Exact rational encodings retain provided timestamp values and
derived differences/midpoints, with separate observed and effective scaling
delays. The reader checks schema, integrity and result-key linkage; payload
verification is explicit through `verify_result=true`. Result-only copies and
current table exports omit timing companions. Existing `FrameSource` positional
construction and default processing remain compatible; new source/clock/unit
labels are optional metadata. See [pair timing](../reference/pair_timing.md).

Actual-time tracking is explicit through `TimedTrackingResult`; the registered
`Trajectory` and `TrackingResult` layouts and default ordinal behavior remain
unchanged. Dedicated artifacts use `timed_tracking_format_version = 1` and omit
the ordinary native marker, so generic native readers reject them. Use
`save_timed_tracking` and `load_timed_tracking` to retain essential timing.
The separate `hammerhead-tracking-time-table-1` CSV schema preserves exact
timestamps and velocity time support. Explicitly extracting the legacy payload
discards timing semantics. See [tracking timing](../reference/tracking_timing.md).

Calibrated scattered exports use a separate
`hammerhead-calibrated-scattered-table-1` CSV and a TOML companion whose metadata
contains `calibrated_table_format_version = 1`. The companion retains affine
settings, coordinate conventions, unit assumptions and diagnostic availability,
including for empty CSVs. Its reader validates metadata integrity and can verify
the paired CSV's content and structure; it does not verify the original images
or numerical result. The two files publish sequentially, so readers must detect
an incomplete or mismatched pair. Ordinary table/native formats are unchanged.
See [calibrated tables](../reference/calibrated_table.md).

Dedicated timed artifacts and calibrated companions accept recorded absolute
Windows/POSIX source locators independently of the reading host. These strings
remain provenance; foreign paths are not resolved against the receiving workspace.
The consumed local artifacts and explicitly supplied local inputs remain protected
against output aliases. Supply a local `csv_path` when the recorded relative CSV
locator uses a foreign separator dialect. This broadens version-1 reading without
changing recorded locator strings, result layouts or scientific binding rules.
It does not relocate or verify unavailable source images, and does not extend
foreign-locator support to experiment, comparison or quality-report formats.

Stereo execution observations use a separate version-1 sibling group alongside
unchanged native stereo results. Existing planar diagnostics and result-only
readers retain their contracts. The stereo reader distinguishes metadata-only
inspection from optional measurement-field verification, including both retained
cameras. This binding excludes parameter objects and correlation planes, and
does not verify calibration, source images or synchronization. Its verification
status describes the current read/capture and is not stored as an attestation.

GUI timed-trajectory inspection explicitly selects the dedicated artifact
format and retains its wrapper through physical display. Ordinary eager and
lazy native result vectors retain their element types and persistence behavior.
Timed bundles have one explorer entry and cannot be appended to native sequences.

Representative-pair comparisons use independent version-1 TOML snapshots,
identified by `pair_comparison_format_version`. They preserve selected-input
provenance, settings differences, units and comparison populations. Loading
validates the saved snapshot without rerunning or reopening its image inputs.
See [recipe comparisons](../reference/pair_comparison.md).

The long-form table contract is identified by `TABLE_SCHEMA_VERSION` and the
ordered `TABLE_COLUMNS` constant. Columns are a backward-compatible superset
across planar, stereo, PTV, and tracking results: unavailable values are empty rather
than changing shape. Existing columns will not be renamed or change meaning
within a schema version. An incompatible contract change increments the schema
version; additive columns may be introduced without invalidating readers that
select columns by name.

Tracking adds eight columns after the original columns, preserving the original
column order and meaning. Each row is an **observed** trajectory position; gaps
are metadata, not synthesized positions. Trajectory IDs follow the result's
trajectory-vector order and are local to that result. Empty trajectories emit no
rows, so IDs can have gaps. Observation IDs count points within a trajectory;
the existing `point_id` counts rows across the entire result. Header-only tables
represent results with no observations. They cannot retain empty trajectories,
the total input frame count, or scale metadata; use JLD2 when that information
must round-trip.

`frame_index` records the input frame's one-based index, while `frame_id` remains
a caller-supplied label for the whole result. `elapsed_time` is derived from
input frame 1, using frame intervals without a scale or the attached scale's
`dt` with one. `time_provenance` identifies which convention was used. Physical
elapsed times assume uniformly spaced input frames; a tracking result does not
retain acquisition timestamps or original frame-source indices. Supplying a
scale after subsampling therefore requires the interval between the frames
actually passed to tracking. This export cannot recover irregular acquisition
timing from frame indices alone.

Tracking velocity columns contain frame-aware differences of observed
positions, using the same endpoint and central-difference convention as
[`trajectory_velocities`](@ref). Physical conversion retains the interval and
is idempotent. `position_valid` and `velocity_valid` report numerical finiteness;
they make no claim about particle identity, uncertainty, or detection quality.
Tracking does not retain mask/outlier flags, so those existing columns are
empty rather than implying that all observations passed a validator. The
[I/O reference](../reference/io.md) defines the added columns and edge cases.

VTK export uses the legacy structured-grid contract documented by
[`export_vtk`](@ref). `FIELD FieldData` stores `coordinate_unit` and
`component_unit` as UTF-8 byte arrays. They follow the attached
[`PhysicalScale`](@ref) when present. Unscaled planar coordinates are pixels
and components are pixels per frame; unscaled stereo coordinates and
components are in calibration-grid world units and world units per frame. The
grid does not carry a unit name, so these labels read `world_unit` and
`world_unit/frame` until a scale supplies explicit names.
