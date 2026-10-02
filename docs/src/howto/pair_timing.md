# Preserve pair timestamps without retaining result arrays

Opt in to planar sequence timing with `record_pair_timing=true` and/or
`on_pair_timing(i,packet)`. These companions preserve provided acquisition
metadata independently of the result struct. The default behavior and the
existing `FrameSource` timestamp/`PhysicalScale` same-unit contract remain
unchanged. New source, frame, clock and unit labels are additive:

```@example pair_timing
using Hammerhead
image = [Float64(mod(17i + 31j + 7i*j, 251))/250 for i in 1:32, j in 1:32]
epoch = Int64(1_000_000_000_000_000_003)
source = FrameSource(2, i -> i == 1 ? image : circshift(image, (1,2));
    timestamps=[epoch, epoch+101], labels=["frame-A", "frame-B"],
    source_id="tutorial-source", frame_ids=["capture-1", "capture-2"],
    time_unit="ns", clock_id="tutorial-clock")
packets = PairTiming[]
path = joinpath(mktempdir(), "timed.jld2")
params = PIVParameters(window_size=16, overlap=8, padding=true)
run_piv_sequence(image_pairs(source), params;
    output=path, record_pair_timing=true, collect_results=false, progress=false,
    on_pair_timing=(i,packet)->push!(packets,packet))
data = pair_timing_data(only(packets))
midpoint = data["sample_time"]
exact_midpoint = parse(BigInt, midpoint["numerator"]) // parse(BigInt, midpoint["denominator"])
@assert exact_midpoint == BigInt(epoch) + 101//2
@assert data["timestamp_status"] == "complete"
data["sample_time_convention"]
```

`source_id` and `frame_ids` are opaque caller identifiers. Their presence does
not verify image bytes or make a label a content hash. Each referenced frame
also records its source index and copied label. Path pairs preserve labels but
do not invent acquisition timestamps or content IDs; matrix pairs have no
provided source identity. Source loaders and image pixels are never stored in
the packet. Callbacks that retain packets do retain their scalar metadata, as
the example does explicitly.

## Exact values and the measurement-time convention

Original timestamps/delays are represented by decimal integer numerator and
denominator strings plus their original numeric type. This preserves large
integer epochs and exact numerical values without cancellation from conversion
to Float64. Derived observed delay and midpoint use exact rational arithmetic
with BigInt. Signed zero has numerical value zero in this representation.
Supported inputs are standard signed/unsigned integers (including BigInt),
Float16/32/64, and rationals over those integer types. Bool, BigFloat, custom
Real implementations and nonfinite values are rejected when capture is enabled.

The assigned sample time is **the midpoint of the two provided timestamps**.
It is not assumed to be the center of an exposure. Timestamp conventions
(exposure start, center or another source convention) remain the provider's
responsibility. Missing or partial timestamps produce explicit statuses and
no midpoint. `scale.dt` and sequence indices never synthesize acquisition time.
Available timestamps may be negative; their observed delay must be positive.
Machine-integer subtraction overflow cannot silently become recorded time:
exact arithmetic is used for observed delay/midpoint, and a wrapped declared
`FramePair.dt` from prior subtraction is rejected.

`observed_delay`, `declared_delay` and `effective_delay` are distinct. A supplied
physical scale still uses `FramePair.dt` as its delay override, exactly as
before. Without a scale, declared pair delay is metadata and arrays remain
pixel displacements. Tuple pairs with a scale preserve that scale's delay;
observed timestamps do not quietly override it. A legacy tuple/scale delay
that differs from the observed delay is recorded explicitly by
`effective_agrees_with_observed=false`. Missing declared/effective values are
not inferred from observed timestamps. The result's actual attached scale
and effective delay provenance are retained.

Declared pair delay and observed delay must agree within
`abs(a-b) <= timing_atol + timing_rtol*max(abs(a),abs(b))`, with defaults
`timing_atol=0.0` and `timing_rtol=sqrt(eps(Float64))`. Tolerances apply to
**delays**, not large absolute epochs. This accommodates ordinary floating
subtraction rounding; both tolerances zero require exact agreement. Delays and
tolerances must be finite, and delays positive. No clock-offset correction or
time-unit conversion is performed.

Explicit unit or clock labels must agree across frames when both are supplied.
Explicit units must also agree with a supplied scale's time-unit label. Missing
or partial clock/unit metadata remains unknown, even when timestamps exist.
Unknown units are not silently relabeled as seconds: a scale separately records
its existing same-unit assumption. Matching caller-provided clock labels does
not certify hardware synchronization.

## Preflight, callback boundaries and memory

All pairs' scalar descriptors are validated and copied **before the first
image is loaded or output is opened**. A bad later pair therefore cannot
truncate an existing destination or partially process the first pair. This
preflight uses O(number of pairs) scalar metadata, while only the current
result arrays and bound packet are retained during processing. Source timestamp,
ID and label edits by a loader or earlier callback do not rewrite the snapshot.

Capture also freezes both selected frame references and copies declared delay.
Mutable two-element pair vectors are normalized to immutable tuples; the
function-output callback receives that frozen tuple. FramePair inputs remain
FramePairs. Mask selection follows the frozen frames; mask callbacks keep their
existing `(i,imgA,imgB)` arguments. Custom pair containers with custom `dt`
properties are rejected by opt-in capture. Default sequence behavior is
unchanged. Pixels and loader behavior from mutable sources are not frozen or
authenticated.

The timing packet binds to raw coordinates, components, UQ, flags and scale.
Timing/result mutation through callbacks is checked before persistence,
including function-output callbacks after they choose a path. A failure aborts
that pair before its result is written; earlier completed results remain saved.
Pending prefetch loading still finishes before the sequence returns.
Correlation planes and parameter objects are outside the numerical binding.

## Read metadata independently

```@example pair_timing
index = ResultFile(path)
metadata_only = load_pair_timing(index, 1)
verified = load_pair_timing(index, 1; verify_result=true)
@assert pair_timing_data(metadata_only) == pair_timing_data(verified)
length(index)
```

The default reader validates schema, packet integrity and result-key linkage
without deserializing result arrays. `verify_result=true` loads only the selected
raw result and checks its numerical binding. Verification does not establish
source identity, pixel-content correctness, time calibration or authentication.
The reader retains neither a result nor an open file handle. Completed-file
size/mtime checks can detect some changes; they do not provide live-writer
safety. Per-pair output files retain the original input-sequence index while
their native result key is `results/000001`.

A missing packet returns `nothing`; old results have no recoverable timing.
Ordinary `save_results` copies and bare results omit companions. There is no
timing export, checkpoint/replay integration, stereo, PTV, ensemble or irregular
tracking support in this slice. Those paths reject the new capture options;
do not interpret this companion as implementing their temporal semantics.
