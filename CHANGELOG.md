# Release notes

## Unreleased

These entries describe changes after the registered baseline. No new package
version or release date has been assigned. See [RELEASING.md](RELEASING.md) for
validation and the core-first, GUI-second release sequence.

### Core

- Stereo sequence and ensemble processing validate available exposure times,
  observed pair delays, and declared `FramePair.dt` before reading images or
  opening output. `sync_atol` and `sync_rtol` control tolerance relative to pair
  delay; `missing_timestamps = :error` requires metadata. The default `:allow`
  preserves path/matrix workflows without asserting synchronization.
- Planar, stereo, and PTV sequence drivers accept `collect_results = false`.
  Results still reach callbacks and persistence, but the driver returns
  `nothing` and does not retain a growing result vector. Existing defaults
  continue to return results. Consumers may still retain their own copies.
- `export_table` accepts `TrackingResult`. Eight columns are appended to the
  existing CSV schema for trajectory/observation IDs, original frame indices,
  derived elapsed time, gaps, and numerical validity. Readers should select
  columns by name. Acquisition timestamps are not inferred from frame indices.
- Effort presets fit the selected ROI. Deformation now handles predictor grids
  with a single node along one or both axes by constant extension.
- `FieldStatisticsAccumulator`, `update_statistics!`, and
  `field_statistics(accumulator)` calculate planar/stereo population moments
  without retaining result histories. Updates validate coordinates, dimensions,
  and scale metadata before changing state; snapshots are independent copies.
- `ResultFile(path)` and `load_results(path; lazy = true)` index completed native
  files and read one entry per access. The index retains keys rather than result
  payloads and rejects detectable file changes. Eager loading stays the default.
  Saving an index or its standard array views over the source file is rejected
  before opening output, including when the destination is a file alias.
- Planar-grid table and VTK exports accept `transform = PlanarTransform(...)`
  with explicit length units and optional pair delay/time units. Coordinates,
  vectors, and component uncertainties use the transformed basis. Mixed-axis
  uncertainty requires an explicit independence assumption or is reported as
  unavailable. Transform export requires raw pixel results without attached
  scale metadata; stereo, PTV, and tracking transforms remain unsupported.

### HammerheadGUI

- Added `ROIEditor`, `roi_editor`/`roi_editor!`, and batch ROI controls, with
  two-corner selection, numeric bounds, reset, and full-image coordinate
  preservation. Invalid or oversized custom windows are rejected before opening
  batch output. The selected ROI is captured when the run starts.
- `ResultExplorer(path; lazy = true)` and `result_explorer(path; lazy = true)`
  browse completed files with one displayed result and bounded derivative
  caching. Failed reads preserve the previous frame and report an error.
  Lazy explorers do not follow live writes or accept appended results.

### Compatibility

Native JLD2 `format_version` remains **1**; persisted result structures are
unchanged. `TABLE_SCHEMA_VERSION` remains **`hammerhead-table-1`** under its
additive-column policy. Fixed-column-count CSV readers need to accommodate the
eight tracking columns. Existing columns retain their order and meaning.

The GUI framework remains GLMakie. Qt/QML and other toolkit candidates are
evaluations in [ROADMAP.md](ROADMAP.md), not added dependencies or supported
replacement shells.
