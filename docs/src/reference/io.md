```@meta
CurrentModule = Hammerhead
```

# Input/output (I/O) and batch processing

Load images with [`load_image`](@ref), build path pairs with
[`image_pairs`](@ref), and process them with [`run_piv_sequence`](@ref).
[`save_results`](@ref) and [`load_results`](@ref) write and read JLD2 result
files. The sequence driver accepts file paths or in-memory arrays and can
write results as pairs finish. See [Batch processing](../howto/batch.md)
for an end-to-end workflow. The time between images in a pair and the time
between successive pairs serve different purposes; see the
[sequence tutorial](../tutorials/sequence_statistics.md).

[`export_table`](@ref) writes planar, stereo, PTV, and tracking results as UTF-8
CSV. Readers should select columns by name from `TABLE_COLUMNS` and check
`schema_version` against `TABLE_SCHEMA_VERSION`. Strings are quoted, embedded
quotes are doubled, and labels may contain commas or newlines. Empty fields
mean unavailable values; nonfinite numerical values use `NaN`, `Inf`, or
`-Inf`. [`export_vtk`](@ref) writes structured planar/stereo grids.

For a `TrackingResult`, the original table columns retain their meanings:
`x`/`y` are observed positions, `u`/`v` are derived velocities, `point_id`
counts written observations, and `length_unit`/`time_unit`/`velocity_unit`
label the exported values. `frame_id`, `source_a`, and `source_b` are labels
supplied for the whole result. Grid indices, the third coordinate/component,
mask/outlier flags, and quality/uncertainty/match fields are empty. The added
columns are empty for planar, stereo, and PTV rows.

| Added column | Meaning for tracking rows |
|:-------------|:--------------------------|
| `trajectory_id` | One-based index in `result.trajectories`; local to this result, preserved across exports. |
| `observation_id` | One-based index within that trajectory. |
| `frame_index` | Observed one-based input frame index stored in `Trajectory.frames`. |
| `elapsed_time` | Time since input frame 1: `frame_index - 1`, or `(frame_index - 1) * scale.dt` when scaled. |
| `gap_before` | Number of unobserved frames since the preceding observation; zero at a trajectory's first observation. |
| `position_valid` | `true` exactly when both exported coordinates are finite. |
| `velocity_valid` | `true` exactly when the current position and both derived components are finite; always `false` for a singleton. |
| `time_provenance` | `frame_index` for elapsed frame intervals, or `physical_scale` for elapsed time derived from the attached scale. |

Elapsed time is derived, not an absolute acquisition timestamp. With a scale,
`dt` must represent a uniform interval between the input frames, including any
subsampling; irregular timestamp information is not available in a
`TrackingResult`. An unscaled result exports positions in `px`, time in `frame`,
and velocities in `px/frame`. A scaled result converts positions once via
[`physical`](@ref), derives velocities via [`trajectory_velocities`](@ref)
using the retained interval, and exports the scale's unit labels. Already
converted results export the same values as their scaled raw counterparts.
The velocity estimate uses one-sided differences at endpoints and central
differences at interior points, dividing by the actual frame-index difference
so gaps are accounted for.

Only observations are written. Empty trajectories emit no rows and retain
their slots in trajectory numbering; singleton trajectories emit one row with
empty `u`/`v`. A result without observations emits only the header. Nonfinite
observations are retained with validity flags. Validity flags check numerical
finiteness, not identity or tracking quality. Malformed trajectory arrays,
non-increasing/out-of-range frames, inconsistent `start_frame`, and negative
frame counts are rejected before the destination file is replaced. Tables do
not preserve empty tracks, the input frame count, or parameters; use
[`save_results`](@ref) for a lossless round-trip.

```julia
tracks = TrackingResult([
    Trajectory{Float64}(2, [10.0, 14.0, 26.0], [20.0, 18.0, 12.0], [2, 3, 6])
], 6, PTVParameters())
tracks = with_scale(tracks, PhysicalScale(pixel_size=0.25, dt=0.5,
                                         length_unit="mm", time_unit="s"))
export_table("tracks.csv", tracks; frame_id="run-1")
# Three rows: frames 2, 3, 6; elapsed times 0.5, 1.0, 2.5 s;
# gap_before 0, 0, 2. The missing frames do not become rows.
```

```@index
Pages = ["io.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["io.jl", "interoperability.jl"]
Private = false
```
