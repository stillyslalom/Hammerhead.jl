const PAIR_TIMING_FORMAT_VERSION = 1
const _TimingInteger = Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}
const _TimingNumber = Union{_TimingInteger,Float16,Float32,Float64,Rational{<:_TimingInteger}}
const _TIMING_TYPES = Dict(string(T) => T for T in
    (Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt,Float16,Float32,Float64,
     Rational{Int8},Rational{Int16},Rational{Int32},Rational{Int64},Rational{Int128},
     Rational{UInt8},Rational{UInt16},Rational{UInt32},Rational{UInt64},Rational{UInt128},Rational{BigInt}))
_timing_error(message) = throw(ArgumentError(message))

function _reject_pair_timing(kwargs, workflow)
    any(k -> haskey(kwargs, k), (:on_pair_timing, :record_pair_timing, :timing_atol, :timing_rtol,
        :scale_pairs, :timing_pairs2, :_timing_snapshots)) &&
        _timing_error("pair timing supports planar and stereo PIV sequences only; $workflow timing is not implemented by this driver")
    nothing
end

"""
    PairTiming

Detached, integrity-checked timing/source metadata for one completed planar PIV
sequence pair. No image, loader, source object or result arrays are retained.
Use [`pair_timing_data`](@ref) for a copied primitive snapshot. Sequence capture
preflights O(number of pairs) scalar descriptors before pixels/output, then
binds only the current packet to its raw result. Caller-retained packets consume
memory at the caller's request. This is provided metadata, not authentication,
exposure-center attribution or synchronization certification.
"""
struct PairTiming
    _data::Dict{String,Any}
    _sha256::String
end

function _timing_ratio(value)
    value isa _TimingNumber && !(value isa Bool) && isfinite(value) ||
        _timing_error("timing numbers must be finite standard integers, Float16/32/64, or integer rationals; Bool, BigFloat and custom Real types are unsupported")
    Rational{BigInt}(value)
end
function _timing_number(value; derived = false)
    value === nothing && return nothing
    r = _timing_ratio(value)
    Dict{String,Any}("numerator" => string(numerator(r)), "denominator" => string(denominator(r)),
        "numeric_type" => derived ? "derived_exact_rational" : string(typeof(value)))
end
function _timing_decode(data; derived = false)
    data === nothing && return nothing
    _experiment_keys(data, ["numerator", "denominator", "numeric_type"], "exact timing number")
    all(k -> data[k] isa String, ("numerator", "denominator", "numeric_type")) || _timing_error("invalid timing number strings")
    n, d = tryparse(BigInt, data["numerator"]), tryparse(BigInt, data["denominator"])
    n !== nothing && d !== nothing && d > 0 && string(n) == data["numerator"] && string(d) == data["denominator"] ||
        _timing_error("invalid canonical timing rational")
    r = n // d
    numerator(r) == n && denominator(r) == d || _timing_error("timing rational is not reduced")
    if derived
        data["numeric_type"] == "derived_exact_rational" || _timing_error("invalid derived timing type")
    else
        T = get(_TIMING_TYPES, data["numeric_type"], nothing)
        T === nothing && _timing_error("unsupported original timing type")
        value = try convert(T, r) catch; _timing_error("timing rational is not representable in original type") end
        isfinite(value) && _timing_ratio(value) == r || _timing_error("original timing value changed precision")
    end
    r
end
function _timing_positive(data; derived = false)
    r = _timing_decode(data; derived)
    r === nothing || r > 0 || _timing_error("pair delays must be positive and finite")
    r
end
function _timing_tolerance(atol, rtol)
    all(x -> x isa Real && !(x isa Bool) && isfinite(x) && x >= 0, (atol, rtol)) ||
        _timing_error("timing tolerances must be finite nonnegative numbers")
    a, r = Float64(atol), Float64(rtol)
    isfinite(a) && isfinite(r) || _timing_error("timing tolerances must be representable in Float64")
    Dict{String,Any}("atol" => a, "rtol" => r)
end
function _timing_agree(a, b, tolerance)
    abs(a - b) <= _timing_ratio(tolerance["atol"]) +
        _timing_ratio(tolerance["rtol"]) * max(abs(a), abs(b))
end
_timing_scale(s) = s === nothing ? nothing : Dict{String,Any}(String(k) => getfield(s,k) for k in fieldnames(PhysicalScale))
_timing_status(a, b) = a === nothing && b === nothing ? "missing" : a === nothing || b === nothing ? "partial" : "complete"
function _timing_frame(frame; timestamp_value=Val(:read))
    data = Dict{String,Any}("kind" => "matrix", "source_id" => nothing,
        "source_identity" => "unavailable", "frame_id" => nothing, "frame_index" => nothing,
        "label" => nothing, "timestamp" => nothing, "time_unit" => nothing, "clock_id" => nothing)
    if frame isa FrameRef
        source, i = frame.source, frame.index
        checkbounds(1:length(source), i)
        data["kind"] = "frame_ref"; data["frame_index"] = i
        data["label"] = String(frame_source_label(source, i))
        data["timestamp"] = _timing_number(timestamp_value isa Val{:read} ? frame_timestamp(source,i) : timestamp_value)
        if source isa FrameSource
            data["source_id"] = _source_metadata_string(source.source_id, "source ID")
            data["source_identity"] = source.source_id === nothing ? "unavailable" : "opaque_provided"
            data["frame_id"] = source.frame_ids === nothing ? nothing : _source_metadata_string(source.frame_ids[i], "frame ID")
            data["time_unit"] = _source_metadata_string(source.time_unit, "time unit")
            data["clock_id"] = _source_metadata_string(source.clock_id, "clock ID")
        end
    elseif frame isa AbstractString
        data["kind"] = "file_path"; data["label"] = String(frame)
    elseif !(frame isa AbstractMatrix{<:Real})
        _timing_error("pair timing requires file, matrix or FrameRef entries")
    end
    data
end
function _timing_derive(frames, declared, scale, tolerance)
    a, b = frames
    ta, tb = _timing_decode(a["timestamp"]), _timing_decode(b["timestamp"])
    for key in ("time_unit", "clock_id")
        a[key] === nothing || b[key] === nothing || a[key] == b[key] || _timing_error("paired $key metadata disagrees")
    end
    observed = ta === nothing || tb === nothing ? nothing : tb - ta
    observed === nothing || observed > 0 || _timing_error("observed timestamp delay must be positive")
    dt = _timing_positive(declared)
    dt === nothing || observed === nothing || _timing_agree(dt, observed, tolerance) ||
        _timing_error("declared FramePair.dt disagrees with observed timestamp delay")
    if scale !== nothing
        all(k -> scale[k] isa Float64 && isfinite(scale[k]) && scale[k] > 0, ("pixel_size", "dt")) ||
            _timing_error("effective scale factors must be positive finite Float64 values")
        if dt !== nothing
            converted = Float64(dt)
            isfinite(converted) && scale["dt"] == converted ||
                _timing_error("pair-dt scale override differs from converted declared delay")
        end
        for frame in frames
            frame["time_unit"] === nothing || frame["time_unit"] == scale["time_unit"] ||
                _timing_error("explicit timestamp unit disagrees with PhysicalScale.time_unit; no conversion is performed")
        end
    end
    effective = scale === nothing ? declared : _timing_number(scale["dt"])
    provenance = scale === nothing ? (dt === nothing ? "unavailable" : "declared_pair_dt_metadata_only") :
        dt === nothing ? "supplied_scale_dt" : "pair_dt_scale_override"
    er = _timing_positive(effective)
    Dict{String,Any}("timestamp_status" => _timing_status(ta, tb),
        "time_unit_status" => _timing_status(a["time_unit"], b["time_unit"]),
        "clock_status" => _timing_status(a["clock_id"], b["clock_id"]),
        "pair_time_unit" => a["time_unit"] === nothing || b["time_unit"] === nothing ? nothing : a["time_unit"],
        "pair_clock_id" => a["clock_id"] === nothing || b["clock_id"] === nothing ? nothing : a["clock_id"],
        "observed_delay" => _timing_number(observed; derived = true), "declared_delay" => declared,
        "effective_delay" => effective, "effective_delay_provenance" => provenance,
        "effective_agrees_with_observed" => observed === nothing || er === nothing ? nothing : _timing_agree(er, observed, tolerance),
        "scale_applied" => scale !== nothing,
        "scale_time_unit_assumption" => scale === nothing ? "not_applied" : "legacy_timestamps_and_scale_same_unit",
        "sample_time" => observed === nothing ? nothing : _timing_number(ta + observed / 2; derived = true),
        "sample_time_convention" => "midpoint_of_provided_timestamps_not_exposure_center")
end
function _timing_freeze_pairs(pairs)
    [begin
        length(pair) == 2 || _timing_error("pair timing requires exactly two frames")
        if pair isa FramePair
            FramePair(pair[1], pair[2], deepcopy(pair.dt))
        elseif pair isa Union{Tuple,AbstractVector} && !hasproperty(pair, :dt)
            (pair[1], pair[2])
        else
            _timing_error("timing capture supports FramePair, tuple, or two-element vector containers without custom dt properties")
        end
    end for pair in pairs]
end
function _timing_preflight(pairs, scale, atol, rtol)
    tolerance = _timing_tolerance(atol, rtol)
    snapshots = Dict{String,Any}[]
    for (i, pair) in enumerate(pairs)
        length(pair) == 2 || _timing_error("pair timing requires exactly two frames")
        frames = [_timing_frame(pair[1]), _timing_frame(pair[2])]
        declared = _timing_number(pair isa FramePair ? pair.dt : nothing)
        _timing_positive(declared)
        actual_scale = pair_scale(scale, pair)
        s = _timing_scale(actual_scale)
        data = Dict{String,Any}("pair_timing_format_version" => PAIR_TIMING_FORMAT_VERSION,
            "input_sequence_index" => i, "frames" => frames, "scale" => s, "delay_tolerance" => copy(tolerance))
        merge!(data, _timing_derive(frames, declared, s, tolerance))
        push!(snapshots, deepcopy(data))
    end
    snapshots
end

function _timing_validate_frames(frames)
    frames isa AbstractVector && length(frames) == 2 || _timing_error("invalid timing frames")
    for frame in frames
        _experiment_keys(frame, ["kind", "source_id", "source_identity", "frame_id", "frame_index", "label", "timestamp", "time_unit", "clock_id"], "timing source frame")
        frame["kind"] in ("frame_ref", "file_path", "matrix") || _timing_error("invalid frame kind")
        for key in ("source_id", "frame_id", "time_unit", "clock_id")
            _source_metadata_string(frame[key], key)
        end
        frame["label"] === nothing || frame["label"] isa String || _timing_error("invalid frame label")
        frame["source_identity"] == (frame["source_id"] === nothing ? "unavailable" : "opaque_provided") || _timing_error("unsupported source identity claim")
        if frame["kind"] == "frame_ref"
            frame["frame_index"] isa Int && frame["frame_index"] > 0 || _timing_error("invalid source frame index")
            frame["label"] isa String || _timing_error("missing source frame label")
        else
            all(k -> frame[k] === nothing, ("frame_index", "source_id", "frame_id", "timestamp", "time_unit", "clock_id")) || _timing_error("non-reference frame contains unavailable acquisition metadata")
            (frame["kind"] == "file_path" ? frame["label"] isa String : frame["label"] === nothing) || _timing_error("invalid frame label")
        end
        _timing_decode(frame["timestamp"])
    end
    nothing
end

function _timing_validate(data)
    fields = ["pair_timing_format_version", "input_sequence_index", "frames", "scale", "delay_tolerance", "raw_result_sha256",
        "timestamp_status", "time_unit_status", "clock_status", "pair_time_unit", "pair_clock_id", "observed_delay",
        "declared_delay", "effective_delay", "effective_delay_provenance", "effective_agrees_with_observed", "scale_applied",
        "scale_time_unit_assumption", "sample_time", "sample_time_convention"]
    _experiment_keys(data, fields, "pair timing")
    data["pair_timing_format_version"] isa Int && data["pair_timing_format_version"] == PAIR_TIMING_FORMAT_VERSION || _timing_error("unsupported pair timing format")
    data["input_sequence_index"] isa Int && data["input_sequence_index"] > 0 || _timing_error("invalid timing sequence index")
    frames = data["frames"]
    _timing_validate_frames(frames)
    s = data["scale"]
    if s !== nothing
        _experiment_keys(s, String.(fieldnames(PhysicalScale)), "timing scale")
        all(k -> s[k] isa Float64 && isfinite(s[k]) && s[k] > 0, ("pixel_size", "dt")) || _timing_error("invalid timing scale factors")
        all(k -> s[k] isa String, ("length_unit", "time_unit")) || _timing_error("invalid timing scale units")
    end
    t = data["delay_tolerance"]
    _experiment_keys(t, ["atol", "rtol"], "timing tolerances")
    all(k -> t[k] isa Float64, ("atol", "rtol")) || _timing_error("invalid tolerance types")
    _timing_tolerance(t["atol"], t["rtol"])
    expected = _timing_derive(frames, data["declared_delay"], s, t)
    for (key, value) in expected
        isequal(data[key], value) || _timing_error("inconsistent derived pair timing field: $key")
        value isa Bool && !(data[key] isa Bool) && _timing_error("invalid Boolean timing field")
    end
    _experiment_hash(data["raw_result_sha256"]) || _timing_error("invalid raw-result timing binding")
    nothing
end
function _timing_checked_data(packet::PairTiming)
    _experiment_digest(packet._data) == packet._sha256 || _timing_error("pair timing packet changed after capture")
    _timing_validate(packet._data)
    packet._data
end
function _timing_bind(snapshot, result::PIVResult)
    data = deepcopy(snapshot)
    isequal(data["scale"], _timing_scale(result.scale)) || _timing_error("result scale differs from preflight timing snapshot")
    data["raw_result_sha256"] = _history_result_digest(result)
    _timing_validate(data)
    PairTiming(data, _experiment_digest(data))
end
function _pair_timing_check_result(packet::PairTiming, result)
    data = _timing_checked_data(packet)
    result isa PIVResult && _history_result_digest(result) == data["raw_result_sha256"] ||
        _timing_error("raw numerical result differs from pair timing binding")
    nothing
end

"""
    pair_timing_data(packet::PairTiming) -> Dict{String,Any}

Return a detached primitive timing snapshot after integrity/schema validation.
Exact values use decimal numerator/denominator strings, with original numeric
types for input timestamps/delays and `derived_exact_rational` for observed
delay/midpoint. Supported inputs are standard signed/unsigned integers including
BigInt, Float16/32/64, and standard-integer rationals; Bool, BigFloat, custom Real
and nonfinite values are refused. This getter neither loads nor verifies a result.
"""
pair_timing_data(packet::PairTiming) = deepcopy(_timing_checked_data(packet))

function _timing_source_labels(snapshot)
    labels = [frame["label"] for frame in snapshot["frames"]]
    any(isnothing, labels) ? nothing : String[labels...]
end

_pair_timing_key(key) = "pair_timing/" * last(split(key, '/'))
function _write_pair_timing(file, key, packet)
    data = _timing_checked_data(packet)
    haskey(file, "pair_timing_format_version") || (file["pair_timing_format_version"] = PAIR_TIMING_FORMAT_VERSION)
    file[_pair_timing_key(key)] = Dict{String,Any}("result_key" => key, "timing" => data, "timing_sha256" => packet._sha256)
    nothing
end

"""
    load_pair_timing(index::ResultFile, i; verify_result=false)
    load_pair_timing(path::AbstractString, i=1; verify_result=false)

Read one optional timing companion from a completed native results file, or
return `nothing` when not recorded. The default validates the detached schema,
integrity and result-key association without deserializing numerical results.
`verify_result=true` loads exactly that one raw result and verifies its binding
to coordinates/components/UQ/flags/scale. No payload or file handle is retained.
File size/mtime checks detect some changes but do not provide live-writer safety,
authentication, source-content verification or synchronization certification.
Ordinary `save_results` copies retain result objects only and omit this companion.
"""
function load_pair_timing(index::ResultFile, i::Integer; verify_result::Bool = false)
    checkbounds(index, i)
    _check_result_file(index)
    packet = jldopen(index.path, "r") do file
        _check_results_format(file, index.path)
        if !haskey(file, "pair_timing_format_version")
            haskey(file, "pair_timing") && _timing_error("pair timing group lacks format marker")
            return nothing
        end
        version = file["pair_timing_format_version"]
        version isa Int && version == PAIR_TIMING_FORMAT_VERSION || _timing_error("unsupported pair timing companion format")
        key = "results/" * index.entry_keys[i]
        timing_key = _pair_timing_key(key)
        haskey(file, timing_key) || return nothing
        saved = file[timing_key]
        _experiment_keys(saved, ["result_key", "timing", "timing_sha256"], "saved pair timing")
        saved["result_key"] == key || _timing_error("pair timing result key mismatch")
        saved["timing"] isa Dict{String,Any} && _experiment_hash(saved["timing_sha256"]) || _timing_error("invalid timing packet storage")
        result = PairTiming(saved["timing"], saved["timing_sha256"])
        _timing_checked_data(result)
        verify_result && _pair_timing_check_result(result, file[key])
        result
    end
    _check_result_file(index)
    packet
end
load_pair_timing(path::AbstractString, i::Integer = 1; kwargs...) = load_pair_timing(ResultFile(path), i; kwargs...)

function Base.show(io::IO, packet::PairTiming)
    data = _timing_checked_data(packet)
    print(io, "PairTiming(pair=", data["input_sequence_index"], ", timestamps=", data["timestamp_status"], ")")
end
