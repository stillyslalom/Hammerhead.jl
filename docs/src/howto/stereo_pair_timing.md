# Preserve stereo sequence timing

Opt in to timing capture on `run_piv_stereo_sequence` with
`on_pair_timing(i, packet)` for live inspection, or `record_pair_timing=true`
with native `output` for persistence. The pair-list and four-tuple overloads
support explicit passes and effort presets. Capture is off by default.

```julia
# dw1 and dw2 are the calibrated dewarpers for the two cameras.
camera1 = FrameSource(2, i -> images1[i]; timestamps=[1000, 1003],
    source_id="camera-1", frame_ids=["A", "B"], time_unit="ns", clock_id="rig-clock")
camera2 = FrameSource(2, i -> images2[i]; timestamps=[1001, 1004],
    source_id="camera-2", frame_ids=["A", "B"], time_unit="ns", clock_id="rig-clock")
run_piv_stereo_sequence(image_pairs(camera1), image_pairs(camera2), dw1, dw2;
    effort=:low, sync_atol=1, output="stereo.jld2", record_pair_timing=true,
    collect_results=false,
    on_pair_timing=(i, p) -> println(pair_timing_data(p)["reconstructed_time_reference"]))
```

Each camera retains its own original timestamps, declared `FramePair.dt`, exact
observed delay, and exact midpoint of its provided timestamps. In this example
the camera midpoints are 1001.5 and 1002.5. They are **not** combined into a
common midpoint. `reconstructed_time_reference` explicitly uses camera 1's
midpoint; this convention does not establish an exposure center or a common
acquisition time when camera skew is tolerated. Numeric values use decimal
numerator/denominator strings, preserving integer epochs and midpoint halves.

The existing synchronization policy remains separate: `sync_atol` and
`sync_rtol` bound matching A/B exposures and camera delays by the available
pair delay, never the epoch. Opt-in capture normalizes both synchronization
tolerances to finite Float64 values. Subtraction of two floating values uses
native floating arithmetic; subtraction involving an integer/rational value is
exact. When all available camera delays are floating, the bound uses native
arithmetic with those normalized tolerances. If any delay is exact, the entire
bound is evaluated from exact encodings, including the tolerances.
This can change near-boundary admission relative to a legacy call supplying
Float32 or exact-rational tolerances. The default no-capture gate is unchanged.
The recorded exposure offsets and
observed delays are exact differences of the supplied values, so those exact
descriptors can differ slightly from a rounded comparison observation.

`timing_atol=0` and `timing_rtol=sqrt(eps(Float64))` separately check agreement
between exact observed delays and declared delays in the encoding. The bound is
`timing_atol + timing_rtol * max(abs(observed), abs(declared))`.
The small relative default accommodates usual Float64 subtraction rounding in
an existing `FramePair.dt`. Lower-precision timestamp arithmetic can require an
explicit larger tolerance; zero tolerances demand exact encoded agreement.
Both timing and synchronization checks must pass. These tolerances do not
convert units or correct clock offsets.

Known time-unit and clock labels must agree across both cameras and both
exposures. Explicit units must agree with an attached `PhysicalScale.time_unit`.
Missing or partial labels remain unknown; no label is inferred from the scale.
`missing_timestamps=:allow` preserves absent/partial timestamp metadata while
checking available comparisons; `:error` requires all four timestamps. Opaque
source/frame IDs are caller identifiers, not verified byte hashes.

Scaling retains its legacy behavior. Pair lists with `FramePair.dt` and an
attached scale use **camera 1's declared delay** to override the scale delay.
Without a scale, that declared delay is metadata only: no velocity conversion
is performed. Four-tuples use the supplied scale delay even when timestamps
imply a different delay. `effective_delay_provenance`, `scale_applied`, and
`effective_agrees_with_camera1_observed` describe these distinctions. Timing
capture never quietly changes the numerical time scale.

All ordered frame selections and scalar metadata are frozen before the first
load or output open. This uses O(number of acquisitions) metadata, independent
of pixel/result array sizes. Mutable two-frame vectors become immutable tuples
for processing; output callbacks receive the frozen four-frame acquisition.
This freezes selected frame identities and descriptors, not mutable pixel
content returned by arbitrary loaders. Known selected input paths (including
TIFF stacks) are protected against output aliases; arbitrary source labels
are not interpreted as verified file paths.

Timing and optional execution packets are both bound after reconstruction and
scale attachment, before the first public callback. The order is diagnostics,
timing, result, output, progress. Both bindings are checked after each callback;
function-output paths are resolved and checked before their destinations open,
even when capture is callback-only. A shared output opens after complete metadata
preflight and is checked before each write. Progress runs after persistence:
an error there can leave the already completed entry. Failure/cancellation joins
the outstanding loader before returning. With `collect_results=false`, the driver
releases each completed result and packet; retaining them in callbacks is optional.

```julia
index = ResultFile("stereo.jld2")
packet = load_stereo_pair_timing(index, 1) # metadata only
data = pair_timing_data(packet)
raw = index[1]
checked = pair_timing_data(packet; result=raw) # no second payload read
# Or load/discard exactly one payload while reading the companion:
packet = load_stereo_pair_timing(index, 1; verify_result=true)
```

Verification covers raw reconstructed/camera measurement fields, flags, UQ,
coordinates and scale, plus independent grid/shape checks. It excludes parameter
objects and retained correlation planes; it does not verify calibration, source
bytes or hardware synchronization. Runtime inspection status is not stored as a
fresh verification claim. Completed-file lazy reads retain no result or open handle.

This slice supports stereo sequences only. Single-pair and ensemble capture,
checkpoint/replay, GUI/report timing display, and timing-aware CSV/VTK exports
remain unsupported. Ordinary result copying/saving omits timing companions.
See [the schema reference](@ref stereo-pair-timing-reference) and
[planar timing](pair_timing.md) for the existing planar companion.
