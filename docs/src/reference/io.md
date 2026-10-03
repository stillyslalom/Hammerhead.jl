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

For a completed native file, `ResultFile(path)` or `load_results(path; lazy=true)`
indexes the available result keys and loads an entry only when accessed, such
as `results[3]`. Construction retains O(N) key metadata without caching payloads
or keeping a file handle open. Open the index after the batch has finished
writing; it describes the entries present at that time.

[`export_table`](@ref) writes planar, stereo, PTV, and tracking results as UTF-8
CSV. Readers should select columns by name from `TABLE_COLUMNS` and check
`schema_version` against `TABLE_SCHEMA_VERSION`. Strings are quoted, embedded
quotes are doubled, and labels may contain commas or newlines. Empty fields
mean unavailable values; nonfinite numerical values use `NaN`, `Inf`, or
`-Inf`. [`export_vtk`](@ref) writes structured planar/stereo grids.

Both writers accept a `transform = PlanarTransform(...)` for a **raw planar
PIV grid**. The affine map exports each coordinate as `A * [x, y] + b` and
each vector in the same physical basis as `A * [u, v]`. It supports an origin
offset, rotation, reflection, anisotropic scale, or a general nonsingular
affine map. Grid indices/topology, mask/outlier flags, and pixel-native quality
values retain their meaning; rotated VTK coordinates need not form separable
axes. `length_unit` is required because a `PlanarTransform` contains no unit
labels. Components use that length unit per frame interval unless positive
finite `dt` and an explicit `time_unit` are also supplied, in which case
components and their uncertainties divide by `dt`.

Transformed uncertainty defaults to `uncertainty_assumption = :unknown`:
a component depending on just one original axis preserves its marginal
standard deviation with the absolute scale factor, while a component mixing
both axes reports `NaN`. The input does not store the cross-component
covariance needed for an exact mixed-axis standard deviation. Setting
`:independent` explicitly assumes zero input cross-component covariance and
uses `hypot(A[k,1] * uncertainty_u, A[k,2] * uncertainty_v) / dt`. A missing
or negative uncertainty on a contributing axis remains unavailable. A zero
coefficient does not require that axis's uncertainty. This reports marginal
standard deviations only: the output axes can be correlated even under the
independence assumption, and their induced covariance is not exported.
Calibration and timing uncertainty are not added.

A transform requires `result.scale === nothing`. Attached `PhysicalScale`
metadata, including the identity metadata on an already converted result,
is rejected to prevent ambiguous or duplicate spatial conversion. For an
unconverted pixel result, remove the metadata with `with_scale(raw, nothing)`
and pass its exposure delay explicitly; removing metadata from `physical(raw)`
does **not** recover pixels. The `transform` keyword on these writers rejects
stereo, PTV and tracking results. Invalid transform factors, unit/time options, or
unsupported combinations are rejected before replacing the destination.
Keep the affine map and the covariance assumption with your analysis notes;
neither export format stores them. See
[Scale results to physical units](@ref) for an executable example.

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

Elapsed time is derived from frame indices. With a scale, `dt` must be the
uniform interval between the input frames, including any subsampling. An unscaled result exports positions in `px`, time in `frame`,
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
observations are retained with validity flags. Malformed trajectory arrays,
non-increasing/out-of-range frames, inconsistent `start_frame`, and negative
frame counts are rejected before the destination file is replaced. Use
[`save_results`](@ref) for a lossless round-trip that also keeps empty tracks,
the input frame count and parameters.

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
