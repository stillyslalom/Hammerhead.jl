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
