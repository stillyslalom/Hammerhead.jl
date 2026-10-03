# Track particles at actual sample times

Use `sample_times` when the selected frames have irregular acquisition intervals.
The default `track_particles` call continues to use input-frame intervals and
returns `TrackingResult`, even when a source provides timestamps. Opting in
returns `TimedTrackingResult`, which keeps the original registered result layout
inside `result` and binds it to a detached `TrackingTiming` snapshot.

```@example tracking_timing
using Hammerhead
using Hammerhead.SyntheticData: generate_gaussian_particle!

elapsed = [0, 1, 4, 5]
images = [zeros(96, 96) for _ in elapsed]
for (k, time) in enumerate(elapsed), (x, y) in ((20., 20.), (30., 45.), (50., 65.))
    generate_gaussian_particle!(images[k], (x + 2time, y), 3., 1.)
end
epoch = big(typemax(Int64)) + 10
source = FrameSource(4, i -> images[i]; timestamps=epoch .+ elapsed,
    source_id="camera-run", frame_ids=["frame-$i" for i in 1:4],
    time_unit="s", clock_id="camera-clock")
refs = [FrameRef(source, i) for i in 1:4]
initial = (x=[1., 96.], y=[1., 96.], u=fill(2., 2, 2), v=zeros(2, 2))
tracks = track_particles(refs, PTVParameters(search_radius=0.7, uod_enable=false);
    sample_times=:source, predictor=initial, min_track_length=4, progress=false)
u, v = trajectory_velocities(tracks, 1)
@assert all(isapprox.(u, 2.; atol=1e-6))
tracking_timing_data(tracks)["sample_times"][1]
```

`sample_times=:source` requires timestamped `FrameRef` inputs. Alternatively,
provide one numeric value per selected frame:

```julia
tracks = track_particles(images; sample_times=[0, 0.01, 0.04, 0.05],
    time_unit="s", clock_id="provided-coordinate", progress=false)
```

An explicit vector defines its own common time coordinate. Original source
timestamps, units and clocks remain provenance; they need not equal that chosen
coordinate. In source mode, known source units/clocks must agree, including with
explicit keyword labels. Mixed source objects require a common known unit and
clock, supplied through metadata or explicit keywords. Labels do not convert
units, authenticate inputs or establish synchronization.

Every selected frame reference, source descriptor and numeric time is frozen
before the first load. Callback changes to the input list or mutable source
metadata cannot change later selections or the recorded context. Pixel contents
and source loaders are not frozen or authenticated. Preflight retains O(selected
frames) scalar metadata; detection still loads frames one at a time. The result
owns the observed trajectories, as ordinary tracking already does.

Times must be complete, finite and strictly increasing. Supported values are
standard signed/unsigned integers including BigInt, Float16/32/64, and rationals
over standard integer types. Bool, BigFloat and custom numeric types are refused.
Exact numerator/denominator encodings preserve integer epochs and fractional
intervals. No missing timestamp is invented from an index or `PhysicalScale.dt`.
Conservative interval-ratio checks reject unrepresentable temporal ranges before
loading; later prediction/normalization overflow or underflow raises an error.

## Predictors, gaps and validation

The first selected transition defines a fixed reference interval. Initial PIV or
named-tuple predictors represent pixel displacement over that transition. A
scaled `PIVResult` predictor is rejected because it can contain converted velocity
components; supply a raw pixel-displacement field. The default `predictor=:piv`
still calculates that first field from the first two selected images.

Established heads use the last observed displacement multiplied by actual elapsed
time divided by the last link's duration. Fresh heads use the prior accepted grid
field scaled to actual elapsed time. Accepted links, including gap-spanning links,
are normalized to pixels per reference interval before scattered UOD and before
building the next grid predictor. Thus `uod_epsilon` is a noise floor in pixels per
reference interval. Changing time labels and consistently rescaling all timestamps
does not change these linking predictions or UOD populations. Changing the first
selected interval changes that reference and can change the validator's noise
floor; this is recorded, not an estimated localization uncertainty.

`max_gap` remains a count of missed **selected input frames**, not seconds or
source-frame indices. No predicted observations are inserted. Source indices and
IDs are preserved separately from `Trajectory.frames`, which indexes the selected
input list. These predictors assume locally constant motion; they do not establish
particle identity or remove acceleration bias.

## Velocity support and physical units

`trajectory_velocities(tracks, trajectory_id)` returns Float64 secants. Endpoints
use their adjacent observation; an interior row uses the observations immediately
before and after it, divided by their actual time separation. For `x=t²` at times
`[0, 1, 4]`, the interior outer secant is 4 while the derivative at the current time
is 2. The returned value is the outer time window's average slope, associated with
that observation row; it is not an instantaneous derivative on an irregular grid.
An arithmetic mean of these magnitudes would weight observations, not durations.
Each public call validates the whole timing packet and trajectory payload, costing
O(selected frames + total observations), in addition to the requested secants.
Timed CSV validates once and streams one trajectory's velocity vectors at a time;
repeated public per-trajectory calls are not a cheap bulk calculation path.

An attached `PhysicalScale` supplies the position factor and length label.
Actual-time velocities **never divide by its `dt`**. Explicit time units must match
the scale's time label. If sample units are unknown, the existing FrameSource
same-unit contract can supply an effective scale time label with
`effective_time_unit_provenance="legacy_scale_same_unit"`; original acquisition
metadata remains unknown. Without either label, velocity has pixels per provided
time coordinate and its physical unit is unavailable.

```@example tracking_timing
scaled = with_scale(tracks, PhysicalScale(pixel_size=0.25, dt=99.,
    length_unit="mm", time_unit="s"))
converted = physical(scaled)
@assert trajectory_velocities(scaled, 1) == trajectory_velocities(converted, 1)
@assert physical(converted) === converted
```

Both transformations validate the old numerical binding before producing a new
one; edited trajectories cannot be legitimized by conversion. Converted positions
keep an identity position factor and remain attached to the same actual times.
Their scale cannot subsequently be stripped or given another position factor or
length label. Float32 positions retain their processing precision during physical
conversion, so rounding can cause small differences from raw-position Float64
secants. Conversion overflow or nonzero-to-zero underflow is refused.

## Persist and export the wrapper

```@example tracking_timing
mktempdir() do directory
    path = joinpath(directory, "timed-tracks.jld2")
    save_timed_tracking(path, scaled)
    restored = load_timed_tracking(path)
    @assert trajectory_velocities(restored, 1) == trajectory_velocities(scaled, 1)
    export_table(joinpath(directory, "timed-tracks.csv"), restored)
end
```

The dedicated artifact stores one unchanged `TrackingResult` payload, exact timing
metadata and integrity digests. It lacks ordinary native `format_version`, so
older `load_results` and `ResultFile` fail clearly instead of discarding essential
timing. `save_results` rejects the wrapper; use the dedicated writer. Existing
native-v1 result files and registered result layouts remain unchanged. Loading
validates the whole trajectory/parameter/scale binding and retains no file handle.
Digests detect inconsistency, not malicious modification or source-content changes.
Known input paths and loaded artifact paths, including file aliases, are protected
against export/save overwrite. Completed files only; concurrent writers are not
supported.

Dedicated version-1 artifacts can retain absolute Windows/POSIX source locators
from another platform. These strings and their integrity encoding remain verbatim
provenance: the reader never resolves foreign locators on the receiving host or
infers source-image relocation. It always protects the actual local loaded
artifact and locally interpretable known sources. Protect relocated inputs
explicitly when saving or exporting:

```julia
restored = load_timed_tracking("relocated-timed-tracks.jld2")
save_timed_tracking("another-timed-tracks.jld2", restored;
    protected_paths=["relocated-input-a.tif", "relocated-input-b.tif"])
export_table("tracks.csv", restored;
    protected_paths=["relocated-input-a.tif", "relocated-input-b.tif"])
```

Saving records those extra local protection paths in the new artifact without
changing the supplied wrapper. Explicit reader/writer/protection arguments refer
to the receiving host; foreign-looking absolute paths are refused before
`abspath`. Prospective alias guards retain existing parent-link, hardlink and
conservative Windows case checks, and reject Windows trailing-dot/space
components. Leading `//server/share` is conservatively Windows UNC under the
untagged version-1 contract; POSIX provenance should use a single leading slash.
Drive-relative/current-drive locators and malformed UNC roots are refused.
Regression tests use foreign fixtures on Windows, not actual Linux/macOS or UNC
share execution.

Timed CSV uses `hammerhead-tracking-time-table-1`, with the legacy columns followed
by exact timestamp/elapsed numerators and denominators, original numeric types,
selected/acquisition clocks and units, source IDs/indices/labels, effective-unit
provenance and velocity start/end frame support. Elapsed time starts at the first
selected input, not the first observed point of each trajectory. Unrepresentable
decimal elapsed values are empty while exact rational fields remain. No gap rows
or unavailable uncertainty estimates are invented. Singleton observations have
no velocity; empty results produce only the header.

Pass the wrapper to supported methods. Explicitly extracting `tracks.result`
discards timing and restores ordinal semantics. Use the
[actual-time GUI guide](gui_tracking_timing.md) for supported dedicated artifact
loading and mean-speed displays, and
[calibrated scattered export](calibrated_scattered_export.md) for explicit spatial
transforms. Combined timed native sequences, checkpoint/replay, and time-aware
spectral analyses remain separate extensions.
