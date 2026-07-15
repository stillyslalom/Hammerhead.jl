# Compatibility policy

Hammerhead follows semantic versioning. Before 1.0, exported Julia APIs and
the package-native JLD2 representation may change in a minor release; changes
are called out in release notes and backward-compatible constructors are used
where practical. A stable release will not make an incompatible exported-API
or native-format change without a major version bump.

JLD2 is the lossless Julia round-trip format. Files carry `format_version`;
readers reject unknown versions rather than silently misinterpreting data.
Users who need long-lived, language-neutral archives should also export the
table or VTK form.

The long-form table contract is identified by `TABLE_SCHEMA_VERSION` and the
ordered `TABLE_COLUMNS` constant. Columns are a backward-compatible superset
across planar, stereo, and PTV results: unavailable values are empty rather
than changing shape. Existing columns will not be renamed or change meaning
within a schema version. An incompatible contract change increments the schema
version; additive columns may be introduced without invalidating readers that
select columns by name.

VTK export uses the legacy structured-grid contract documented by
[`export_vtk`](@ref). Coordinate/component units are written in field metadata
and follow the attached [`PhysicalScale`](@ref) when present.
