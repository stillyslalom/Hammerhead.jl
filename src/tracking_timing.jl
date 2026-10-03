const TRACKING_TIMING_FORMAT_VERSION = 1

"""
    TrackingTiming

Detached, integrity-checked sample times and selected-frame descriptors for an
actual-time tracking result. Times retain exact rational values. Metadata is
provided context, not exposure-center attribution, authentication or clock
synchronization evidence. Use [`tracking_timing_data`](@ref) for a copied snapshot.
"""
struct TrackingTiming
    _data::Dict{String,Any}
    _sha256::String
end

"""
    TimedTrackingResult

Opt-in result from [`track_particles`](@ref) with `sample_times`. Owns an unchanged
registered [`TrackingResult`](@ref) payload and a [`TrackingTiming`](@ref) companion.
Pass this wrapper to actual-time velocity, conversion, export and persistence
methods. Its `result` field is explicitly the legacy payload; extracting it
discards essential timing semantics. No implicit unwrapping is performed.
"""
struct TimedTrackingResult{T<:AbstractFloat}
    result::TrackingResult{T}
    timing::TrackingTiming
end

function _tracking_product(::Type{T}, value, ratio) where T
    ratio > 0 || _timing_error("tracking time ratio must be positive")
    r = Float64(ratio)
    isfinite(r) && r > 0 || _timing_error("tracking time ratio overflows or underflows Float64")
    isfinite(value) || _timing_error("nonfinite tracking predictor or displacement")
    product = T(Float64(value) * r)
    isfinite(product) && (value == 0 || product != 0) ||
        _timing_error("tracking time-normalized displacement overflows or underflows processing precision")
    product
end

function _tracking_ratio_bounds(times)
    extent = times[end] - times[1]
    shortest = minimum(times[k] - times[k-1] for k in 2:length(times))
    reference = times[2]-times[1]
    bounds = (extent/shortest,shortest/extent,extent/reference,reference/extent)
    all(r -> isfinite(Float64(r)) && Float64(r)>0,bounds) ||
        _timing_error("tracking interval ratios cannot be represented safely in Float64")
    nothing
end
function _tracking_predict(::Type{T}, position, value, ratio) where T
    prediction = T(Float64(position) + Float64(_tracking_product(T, value, ratio)))
    isfinite(prediction) || _timing_error("nonfinite time-aware predicted position")
    prediction
end

function _tracking_units(unit, scale)
    scale === nothing && return (unit, unit === nothing ? "unknown" : "provided_sample_unit")
    all(x -> isfinite(x) && x > 0, (scale.pixel_size, scale.dt)) || _timing_error("invalid tracking scale")
    isempty(scale.time_unit) && _timing_error("actual-time scale needs a nonempty time unit label")
    unit === nothing || unit == scale.time_unit || _timing_error("sample time unit conflicts with scale time unit")
    (unit === nothing ? scale.time_unit : unit,
        unit === nothing ? "legacy_scale_same_unit" : "provided_sample_unit")
end

function _tracking_preflight(frames, sample_times, time_unit, clock_id, scale, predictor)
    predictor isa PIVResult && predictor.scale !== nothing &&
        _timing_error("actual-time tracking requires an unscaled pixel-displacement PIV predictor")
    unit = _source_metadata_string(time_unit, "time_unit")
    clock = _source_metadata_string(clock_id, "clock_id")
    frozen = [f isa AbstractString ? String(f) : f for f in frames]
    all(f -> f isa Union{FrameRef,AbstractString,AbstractMatrix{<:Real}}, frozen) ||
        _timing_error("actual-time tracking supports FrameRef, path or real matrix inputs")
    descriptors = [_timing_frame(f) for f in frozen]
    source_mode = sample_times === :source
    source_scope = "provided_vector"
    sample_times isa Symbol && !source_mode && _timing_error("sample_times symbol must be :source")
    if source_mode
        all(f -> f isa FrameRef, frozen) || _timing_error("sample_times=:source requires timestamped FrameRef inputs")
        all(d -> d["timestamp"] !== nothing, descriptors) || _timing_error("all selected tracking frames need timestamps")
        units = unique(d["time_unit"] for d in descriptors if d["time_unit"] !== nothing)
        clocks = unique(d["clock_id"] for d in descriptors if d["clock_id"] !== nothing)
        length(units) <= 1 && length(clocks) <= 1 || _timing_error("conflicting source time units or clocks")
        unit === nothing || isempty(units) || unit == only(units) || _timing_error("explicit time unit conflicts with source")
        clock === nothing || isempty(clocks) || clock == only(clocks) || _timing_error("explicit clock conflicts with source")
        all_unit = all(d -> d["time_unit"] !== nothing, descriptors)
        all_clock = all(d -> d["clock_id"] !== nothing, descriptors)
        unit === nothing && all_unit && (unit = only(units))
        clock === nothing && all_clock && (clock = only(clocks))
        same_source = all(f -> f.source === frozen[1].source, frozen)
        source_scope = same_source ? "single_source" : "mixed_sources"
        same_source || (clock !== nothing && unit !== nothing) ||
            _timing_error("mixed sources require a common explicit clock and time unit")
        encoded = [deepcopy(d["timestamp"]) for d in descriptors]
    else
        sample_times isa AbstractVector && length(sample_times) == length(frozen) ||
            _timing_error("sample_times must contain one numeric value per selected frame")
        encoded = [_timing_number(t) for t in sample_times]
        all(!isnothing, encoded) || _timing_error("tracking sample times cannot be missing")
    end
    times = [_timing_decode(t) for t in encoded]
    all(k -> times[k] > times[k - 1], 2:length(times)) || _timing_error("tracking sample times must be strictly increasing")
    reference = times[2] - times[1]
    # O(frames) conservative ratio bound covers every possible observed span,
    # including bridged gaps, before loading any pixel data.
    _tracking_ratio_bounds(times)
    effective, provenance = _tracking_units(unit, scale)
    paths = String[abspath(String(f)) for f in frozen if f isa AbstractString]
    append!(paths,[abspath(f.source.path) for f in frozen if f isa FrameRef && f.source isa TIFFStack])
    data = Dict{String,Any}("tracking_timing_format_version"=>TRACKING_TIMING_FORMAT_VERSION,
        "n_frames"=>length(frozen), "frames"=>descriptors, "sample_times"=>encoded,
        "sample_time_origin"=>source_mode ? "frame_source" : "explicit_vector",
        "source_scope"=>source_scope,
        "time_unit"=>unit, "clock_id"=>clock,
        "effective_time_unit"=>effective, "effective_time_unit_provenance"=>provenance,
        "reference_interval"=>_timing_number(reference; derived=true),
        "predictor_basis"=>"pixels_per_first_transition_reference_interval",
        "velocity_convention"=>"endpoint_one_sided_interior_outer_secant",
        "position_basis"=>"pixels", "scale"=>_timing_scale(scale),
        "protected_paths"=>sort!(unique(paths)))
    (frames=frozen, times=times, reference=reference, data=data)
end

function _tracking_result_digest(r::TrackingResult)
    # Canonical hashing streams trajectory vectors without an observation-sized
    # encoded byte buffer. Descriptors reference existing vectors temporarily.
    tracks = [Dict{String,Any}("start_frame"=>t.start_frame,"x"=>t.x,"y"=>t.y,"frames"=>t.frames) for t in r.trajectories]
    data = Dict{String,Any}("trajectories"=>tracks,"n_frames"=>r.n_frames,
        "parameters"=>Dict{String,Any}(String(k)=>getfield(r.parameters,k) isa Symbol ? String(getfield(r.parameters,k)) : getfield(r.parameters,k) for k in fieldnames(PTVParameters)),
        "scale"=>_timing_scale(r.scale))
    # Coordinate precision must survive even for an empty tracking result.
    data["coordinate_type"] = string(typeof(r).parameters[1])
    _history_digest(data)
end

function _tracking_validate(data)
    _experiment_keys(data,["tracking_timing_format_version","n_frames","frames","sample_times",
        "sample_time_origin","source_scope","time_unit","clock_id","effective_time_unit","effective_time_unit_provenance",
        "reference_interval","predictor_basis","velocity_convention","position_basis","scale",
        "protected_paths","result_sha256"],"tracking timing")
    data["tracking_timing_format_version"] isa Int && data["tracking_timing_format_version"] == 1 || _timing_error("unsupported tracking timing version")
    n = data["n_frames"]
    n isa Int && n >= 2 || _timing_error("invalid tracking frame count")
    data["frames"] isa AbstractVector && data["sample_times"] isa AbstractVector &&
        length(data["frames"]) == length(data["sample_times"]) == n || _timing_error("tracking frame/time lengths disagree")
    data["sample_time_origin"] in ("frame_source","explicit_vector") || _timing_error("invalid sample time origin")
    for key in ("time_unit","clock_id","effective_time_unit")
        value = data[key]
        value === nothing || value isa String && !isempty(value) || _timing_error("invalid tracking unit or clock")
    end
    times = [_timing_decode(t) for t in data["sample_times"]]
    all(!isnothing,times) && all(k -> times[k] > times[k-1],2:n) || _timing_error("invalid tracking time order")
    _tracking_ratio_bounds(times)
    _timing_decode(data["reference_interval"];derived=true) == times[2]-times[1] || _timing_error("invalid tracking reference interval")
    for (i, frame) in enumerate(data["frames"])
        _tracking_validate_frame(frame)
        data["sample_time_origin"] == "frame_source" && frame["timestamp"] != data["sample_times"][i] && _timing_error("source sample time differs from frame timestamp")
    end
    if data["sample_time_origin"] == "frame_source"
        data["source_scope"] in ("single_source","mixed_sources") || _timing_error("invalid source time scope")
        data["source_scope"] == "mixed_sources" && (data["time_unit"] === nothing || data["clock_id"] === nothing) &&
            _timing_error("mixed source times need a common known unit and clock")
        all(f -> f["kind"] == "frame_ref",data["frames"]) || _timing_error("source times require source frame descriptors")
        for (descriptor_key,aggregate_key) in (("time_unit","time_unit"),("clock_id","clock_id"))
            known = unique(f[descriptor_key] for f in data["frames"] if f[descriptor_key] !== nothing)
            length(known)<=1 || _timing_error("conflicting tracking source units or clocks")
            aggregate = data[aggregate_key]
            aggregate === nothing || isempty(known) || aggregate == only(known) || _timing_error("sample metadata conflicts with source descriptors")
            all(f -> f[descriptor_key] !== nothing,data["frames"]) && aggregate === nothing &&
                _timing_error("complete source metadata cannot be discarded")
        end
    else
        data["source_scope"] == "provided_vector" || _timing_error("invalid explicit-vector time scope")
    end
    s = data["scale"]
    scale = if s === nothing
        nothing
    else
        _experiment_keys(s,String.(fieldnames(PhysicalScale)),"tracking scale")
        all(k -> s[k] isa Float64 && isfinite(s[k]) && s[k]>0,("pixel_size","dt")) || _timing_error("invalid tracking scale factors")
        all(k -> s[k] isa String,("length_unit","time_unit")) || _timing_error("invalid tracking scale labels")
        PhysicalScale(s["pixel_size"],s["dt"],s["length_unit"],s["time_unit"])
    end
    unit, provenance = _tracking_units(data["time_unit"],scale)
    isequal(data["effective_time_unit"],unit) && data["effective_time_unit_provenance"] == provenance || _timing_error("inconsistent effective tracking units")
    data["predictor_basis"] == "pixels_per_first_transition_reference_interval" || _timing_error("invalid predictor basis")
    data["velocity_convention"] == "endpoint_one_sided_interior_outer_secant" || _timing_error("invalid velocity convention")
    data["position_basis"] in ("pixels","scaled_length") || _timing_error("invalid position basis")
    data["position_basis"] == "scaled_length" && (scale === nothing || scale.pixel_size != 1.0) && _timing_error("scaled positions require an identity position scale")
    paths = data["protected_paths"]
    paths isa AbstractVector && all(p -> p isa String && isabspath(p),paths) && paths == sort(unique(paths)) || _timing_error("invalid protected tracking paths")
    _experiment_hash(data["result_sha256"]) || _timing_error("invalid tracking result binding")
    nothing
end

function _tracking_validate_frame(frame)
    _experiment_keys(frame,["kind","source_id","source_identity","frame_id","frame_index","label","timestamp","time_unit","clock_id"],"tracking frame")
    frame["kind"] in ("frame_ref","file_path","matrix") || _timing_error("invalid tracking frame kind")
    for key in ("source_id","frame_id","time_unit","clock_id")
        value = frame[key]
        value === nothing || value isa String && !isempty(value) || _timing_error("invalid tracking frame metadata")
    end
    frame["label"] === nothing || frame["label"] isa String || _timing_error("invalid tracking frame label")
    if frame["kind"] == "frame_ref"
        frame["frame_index"] isa Int && frame["frame_index"] > 0 || _timing_error("invalid source frame index")
        frame["label"] isa String || _timing_error("missing source frame label")
        frame["source_identity"] == (frame["source_id"] === nothing ? "unavailable" : "opaque_provided") || _timing_error("invalid source identity claim")
    else
        frame["frame_index"] === nothing && frame["source_id"] === nothing && frame["frame_id"] === nothing &&
            frame["time_unit"] === nothing && frame["clock_id"] === nothing && frame["timestamp"] === nothing &&
            frame["source_identity"] == "unavailable" || _timing_error("invalid matrix/path source metadata")
        (frame["kind"] == "file_path" ? frame["label"] isa String : frame["label"] === nothing) || _timing_error("invalid matrix/path label")
    end
    _timing_decode(frame["timestamp"])
    nothing
end

function _tracking_checked(packet::TrackingTiming)
    _history_digest(packet._data) == packet._sha256 || _timing_error("tracking timing packet changed")
    _tracking_validate(packet._data)
    packet._data
end
function _tracking_check(timed::TimedTrackingResult)
    data = _tracking_checked(timed.timing)
    _check_tracking_table(timed.result)
    timed.result.n_frames == data["n_frames"] && isequal(_timing_scale(timed.result.scale),data["scale"]) &&
        _tracking_result_digest(timed.result) == data["result_sha256"] || _timing_error("tracking payload differs from timing binding")
    data
end
function _tracking_bind(result::TrackingResult, snapshot)
    _check_tracking_table(result)
    data = deepcopy(snapshot)
    data["scale"] = _timing_scale(result.scale)
    data["effective_time_unit"], data["effective_time_unit_provenance"] = _tracking_units(data["time_unit"],result.scale)
    data["result_sha256"] = _tracking_result_digest(result)
    _tracking_validate(data)
    TimedTrackingResult(result,TrackingTiming(data,_history_digest(data)))
end

"""
    tracking_timing_data(timed::TimedTrackingResult)
    tracking_timing_data(timing::TrackingTiming)

Return a detached primitive snapshot with exact sample times, source descriptors,
units, normalization and position conventions. The wrapper form also verifies
its numerical trajectory binding; the companion-only form verifies metadata.
"""
tracking_timing_data(timed::TimedTrackingResult) = deepcopy(_tracking_check(timed))
tracking_timing_data(packet::TrackingTiming) = deepcopy(_tracking_checked(packet))

"""
    trajectory_velocities(timed::TimedTrackingResult, trajectory_id) -> (u, v)

Return Float64 actual-time secants for one trajectory after binding validation.
Endpoints use adjacent observations; interior entries use the outer observations
`i-1` and `i+1`. Each value is that time window's mean slope, associated with the
row's observation; on irregular sampling it is not a derivative evaluated at the
current observation time. No gap observations are invented. At least two points
are required. Raw positions use `pixel_size` when a scale is attached; `scale.dt`
never divides actual-time velocities. Unlabelled times have unknown physical units.
"""
function trajectory_velocities(timed::TimedTrackingResult, id::Integer)
    data = _tracking_check(timed)
    checkbounds(timed.result.trajectories,id)
    t = timed.result.trajectories[id]
    length(t) >= 2 || _timing_error("trajectory_velocities needs at least two observations")
    _tracking_velocities(timed.result,t,data)
end
function _tracking_velocities(result,t,data,times = [_timing_decode(value) for value in data["sample_times"]])
    factor = result.scale === nothing ? 1.0 : result.scale.pixel_size
    u, v = Vector{Float64}(undef,length(t)), Vector{Float64}(undef,length(t))
    for i in eachindex(t.x)
        a, b = i == 1 ? (1,2) : i == length(t) ? (i-1,i) : (i-1,i+1)
        elapsed = times[t.frames[b]] - times[t.frames[a]]
        for (out,positions) in ((u,t.x),(v,t.y))
            if !isfinite(positions[a]) || !isfinite(positions[b])
                out[i] = NaN
                continue
            end
            # Preserve the exact difference and duration until the final quotient,
            # even when separately converting an epoch/span would overflow.
            exact = (_timing_ratio(positions[b]) - _timing_ratio(positions[a])) * _timing_ratio(factor) / elapsed
            value = Float64(exact)
            isfinite(value) && (exact == 0 || value != 0) || _timing_error("actual-time velocity overflows or underflows Float64")
            out[i] = value
        end
    end
    u,v
end

"""
    tracking_speed_summary(timed::TimedTrackingResult) -> NamedTuple

Validate the complete timing/result binding once, decode the timeline once, and
return detached per-trajectory scalar summaries. `speeds` are arithmetic means
of the magnitudes of **observation-associated secants**: adjacent observations
at endpoints, outer observations at interior rows. They are not instantaneous
speeds, elapsed-time-weighted means, or path length divided by elapsed time.

`available` and `reasons` accompany `speeds`. Empty/singleton tracks, any
nonfinite position/secant/magnitude, and Float64 arithmetic overflow/underflow
make the whole track's speed unavailable (`NaN`); observations are never omitted.
`tracks` contains observation counts, selected-input ordinal bounds/gap counts,
and exact first/last/elapsed time strings (no invented gap observations).
Aggregate units, unit provenance, clock/source scope, and conventions are also
returned. Unknown sample-time units remain unknown; a scale's same-unit
assumption is explicitly labeled. Nominal `scale.dt` never divides these speeds.

Cost is O(all observations + selected input frames); retained output is
O(trajectories), with temporary velocity arrays for one trajectory. This is a
captured scalar snapshot, not an ongoing mutation-proof view or trusted context.
Calling it again validates current mutable payloads. No frame descriptors or
trajectory position arrays are retained in the output.
"""
function tracking_speed_summary(timed::TimedTrackingResult)
    data = _tracking_check(timed)
    times = [_timing_decode(value) for value in data["sample_times"]]
    trajectories = timed.result.trajectories
    speeds = fill(NaN,length(trajectories))
    available = falses(length(trajectories))
    reasons = fill(:none,length(trajectories))
    tracks = NamedTuple[]
    exact_string(value) = denominator(value)==1 ? string(numerator(value)) : string(numerator(value),"/",denominator(value))
    for (id,t) in pairs(trajectories)
        n = length(t)
        a,b = n==0 ? (nothing,nothing) : (first(t.frames),last(t.frames))
        push!(tracks,(observations=n,first_frame=a,last_frame=b,
            gaps=count(>(1),diff(t.frames)),
            first_time=a===nothing ? nothing : exact_string(times[a]),
            last_time=b===nothing ? nothing : exact_string(times[b]),
            elapsed=a===nothing ? nothing : exact_string(times[b]-times[a])))
        if n<2
            reasons[id] = n==0 ? :empty : :singleton
            continue
        elseif !all(isfinite,t.x) || !all(isfinite,t.y)
            reasons[id] = :nonfinite_position
            continue
        end
        velocities = try
            _tracking_velocities(timed.result,t,data,times)
        catch err
            if err isa ArgumentError && err.msg=="actual-time velocity overflows or underflows Float64"
                reasons[id] = :arithmetic_range
                continue
            end
            rethrow()
        end
        u,v = velocities
        maximum_speed = 0.0
        for i in eachindex(u)
            if !isfinite(u[i]) || !isfinite(v[i])
                reasons[id] = :nonfinite_secant
                break
            end
            speed = hypot(u[i],v[i])
            if !isfinite(speed)
                reasons[id] = :nonfinite_magnitude
                break
            end
            maximum_speed = max(maximum_speed,speed)
        end
        reasons[id]===:none || continue
        # Scale before accumulating, then use compensated summation. This
        # avoids overflow for large finite speeds and detects a nonzero mean
        # that would round to zero (including subnormal speed fixtures).
        total,correction = 0.0,0.0
        if maximum_speed>0
            for i in eachindex(u)
                term = hypot(u[i],v[i])/maximum_speed-correction
                next = total+term
                correction = (next-total)-term
                total = next
            end
        end
        mean = maximum_speed==0 ? 0.0 : maximum_speed*(total/length(u))
        isfinite(mean) && (maximum_speed==0 || mean!=0) || (reasons[id]=:arithmetic_range; continue)
        speeds[id],available[id] = mean,true
    end
    scale = timed.result.scale
    (speeds=speeds,available=available,reasons=reasons,tracks=tracks,
        length_unit=scale===nothing ? "px" : scale.length_unit,
        time_unit=data["effective_time_unit"],
        time_unit_provenance=data["effective_time_unit_provenance"],
        sample_time_unit=data["time_unit"],clock_id=data["clock_id"],
        source_scope=data["source_scope"],
        velocity_convention=data["velocity_convention"],
        mean_convention="observation_mean_secant_magnitude")
end

function with_scale(timed::TimedTrackingResult,scale::Union{Nothing,PhysicalScale})
    data = _tracking_check(timed)
    if data["position_basis"] == "scaled_length"
        old = timed.result.scale
        scale !== nothing && scale.pixel_size == 1.0 && scale.length_unit == old.length_unit ||
            _timing_error("cannot strip or rescale already converted time-aware positions")
    end
    _tracking_bind(with_scale(timed.result,scale),data)
end
function physical(timed::TimedTrackingResult)
    data = _tracking_check(timed)
    result = timed.result
    result.scale === nothing && return timed
    data["position_basis"] == "scaled_length" && return timed
    converted = physical(result)
    for (old,new) in zip(result.trajectories,converted.trajectories), (a,b) in ((old.x,new.x),(old.y,new.y)), i in eachindex(a)
        isfinite(a[i]) && (!isfinite(b[i]) || (a[i]!=0 && b[i]==0)) &&
            _timing_error("physical tracking positions overflow or underflow processing precision")
    end
    # The legacy position conversion preserves dt; actual-time differencing
    # deliberately ignores it on both raw and converted payloads.
    snapshot = deepcopy(data)
    snapshot["position_basis"] = "scaled_length"
    _tracking_bind(converted,snapshot)
end
physical(timed::TimedTrackingResult,scale::PhysicalScale) = physical(with_scale(timed,scale))

function Base.show(io::IO,timed::TimedTrackingResult)
    data = _tracking_check(timed)
    print(io,"TimedTrackingResult(",length(timed.result.trajectories)," tracks, ",data["n_frames"],
        " actual sample times, unit=",something(data["effective_time_unit"],"unknown"),")")
end

save_results(path::AbstractString,timed::TimedTrackingResult) =
    _timing_error("actual-time tracking requires save_timed_tracking; native-v1 storage would discard timing semantics")
save_results(path::AbstractString,results::AbstractVector{<:TimedTrackingResult}) =
    _timing_error("actual-time tracking requires save_timed_tracking; native-v1 storage would discard timing semantics")

function _tracking_output_guard(path,data)
    full = abspath(path)
    ispath(full) && !isfile(full) && _timing_error("timed tracking output must be a file path")
    islink(full) && !ispath(full) && _timing_error("cannot write timed tracking through a dangling symlink")
    for source in data["protected_paths"]
        full == source && _timing_error("cannot overwrite a timed-tracking artifact or input source")
        ispath(full) && ispath(source) && Base.samefile(full,source) &&
            _timing_error("cannot overwrite a timed-tracking artifact or input alias")
    end
    full
end

"""
    save_timed_tracking(path, timed::TimedTrackingResult) -> path

Save one time-aware tracking result to its dedicated version-1 artifact, replacing
an existing unrelated destination. The artifact intentionally lacks native
`format_version`: ordinary/older `load_results` and `ResultFile` refuse it instead
of dropping essential timing. Binding and known input/artifact aliases are checked
before output. A same-directory temporary file is closed before publication;
concurrent writers and external filesystem mutation are unsupported.
"""
function save_timed_tracking(path::AbstractString,timed::TimedTrackingResult)
    data = _tracking_check(timed)
    destination = _tracking_output_guard(path,data)
    parent = dirname(destination)
    mkpath(parent)
    temp, handle = mktemp(parent)
    close(handle)
    try
        jldopen(temp,"w") do file
            file["timed_tracking_format_version"] = TRACKING_TIMING_FORMAT_VERSION
            file["tracking_result"] = timed.result
            file["tracking_timing"] = data
            file["tracking_timing_sha256"] = timed.timing._sha256
        end
        _tracking_check(timed)
        _tracking_output_guard(destination,data)
        mv(temp,destination;force=true)
    finally
        ispath(temp) && rm(temp;force=true)
    end
    path
end

"""
    load_timed_tracking(path) -> TimedTrackingResult

Load and validate one completed dedicated timed-tracking artifact, including its
exact metadata and whole trajectory/scale/parameter binding. One result payload is
read, with no retained file handle. Native-v1 files are not timed artifacts. File
size/mtime checks detect some concurrent changes but do not provide writer safety.
The loaded artifact and recorded input paths remain protected output destinations.
"""
function load_timed_tracking(path::AbstractString)
    full = abspath(path)
    stamp = stat(full)
    timed = jldopen(full,"r") do file
        haskey(file,"format_version") && _timing_error("native results are not dedicated timed-tracking artifacts")
        haskey(file,"timed_tracking_format_version") || _timing_error("missing timed tracking format marker")
        version = file["timed_tracking_format_version"]
        version isa Int && version == TRACKING_TIMING_FORMAT_VERSION || _timing_error("unsupported timed tracking artifact version")
        all(k -> haskey(file,k),("tracking_result","tracking_timing","tracking_timing_sha256")) || _timing_error("incomplete timed tracking artifact")
        data, digest = file["tracking_timing"],file["tracking_timing_sha256"]
        data isa Dict{String,Any} && _experiment_hash(digest) || _timing_error("invalid tracking timing storage")
        packet = TrackingTiming(data,digest)
        _tracking_checked(packet)
        result = file["tracking_result"]
        result isa TrackingResult || _timing_error("timed tracking payload is not TrackingResult")
        wrapped = TimedTrackingResult(result,packet)
        _tracking_check(wrapped)
        snapshot = deepcopy(data)
        push!(snapshot["protected_paths"],full)
        sort!(unique!(snapshot["protected_paths"]))
        _tracking_bind(result,snapshot)
    end
    after = stat(full)
    after.size == stamp.size && after.mtime == stamp.mtime || _timing_error("timed tracking artifact changed during reading")
    timed
end

# A distinct CSV schema keeps the legacy table contract unchanged. The legacy
# columns retain their order; actual-time rows add exact provided/elapsed values,
# acquisition metadata and the velocity stencil's explicit selected-frame support.
const TIMED_TRACKING_TABLE_COLUMNS = (TABLE_COLUMNS...,"sample_time_numerator","sample_time_denominator",
    "sample_time_numeric_type","elapsed_time_numerator","elapsed_time_denominator","clock_id",
    "source_id","source_frame_id","source_frame_index","source_label","acquisition_time_unit","acquisition_clock_id",
    "effective_time_unit_provenance","velocity_start_frame","velocity_end_frame")

"""
    export_table(path, timed::TimedTrackingResult; frame_id="", source_a="", source_b="")

Export observed trajectory positions with actual-time secants and exact provided
timestamps/source metadata. The dedicated `hammerhead-tracking-time-table-1`
schema retains legacy columns and appends exact sample/elapsed rationals and
velocity stencil support. Elapsed time is relative to the first selected input,
not the first trajectory observation. Unknown units remain empty. Numeric elapsed
values that cannot be represented in Float64 are empty; exact elapsed values remain.
No gap rows, instantaneous derivative claims or uncertainty estimates are invented.
"""
function export_table(path::AbstractString,timed::TimedTrackingResult;
                      frame_id="",source_a="",source_b="")
    data = _tracking_check(timed)
    _tracking_output_guard(path,data)
    converted = physical(timed)
    q = converted.result
    times = [_timing_decode(v) for v in data["sample_times"]]
    lu = q.scale === nothing ? "px" : q.scale.length_unit
    tu = data["effective_time_unit"]
    vu = tu === nothing ? missing : lu*"/"*tu
    # Validate arithmetic before opening output; recompute one track's secants
    # during streaming publication rather than retaining every velocity vector.
    for t in q.trajectories
        length(t)>=2 && _tracking_velocities(q,t,data,times)
    end
    open(path,"w") do io
        println(io,join(TIMED_TRACKING_TABLE_COLUMNS,','))
        point_id = 0
        for (id,t) in enumerate(q.trajectories)
            speed = length(t)>=2 ? _tracking_velocities(q,t,data,times) : nothing
            for k in eachindex(t.x)
                point_id += 1
                f = t.frames[k]
                a,b = speed === nothing ? (nothing,nothing) : k==1 ? (t.frames[1],t.frames[2]) :
                    k==length(t) ? (t.frames[k-1],t.frames[k]) : (t.frames[k-1],t.frames[k+1])
                u,v = speed === nothing ? (missing,missing) : (speed[1][k],speed[2][k])
                position_valid = isfinite(t.x[k]) && isfinite(t.y[k])
                velocity_valid = speed !== nothing && position_valid && isfinite(u) && isfinite(v)
                elapsed = times[f]-times[1]
                ef = Float64(elapsed)
                numeric_elapsed = isfinite(ef) && (elapsed==0 || ef!=0) ? ef : missing
                stamp = data["sample_times"][f]
                source = data["frames"][f]
                gap = k==1 ? 0 : t.frames[k]-t.frames[k-1]-1
                vals = ("hammerhead-tracking-time-table-1","tracking",frame_id,source_a,source_b,
                    point_id,missing,missing,t.x[k],t.y[k],missing,u,v,missing,
                    missing,missing,missing,missing,missing,missing,missing,missing,missing,missing,
                    lu,something(tu,missing),vu,id,k,f,numeric_elapsed,gap,position_valid,velocity_valid,"actual_sample_times",
                    stamp["numerator"],stamp["denominator"],stamp["numeric_type"],string(numerator(elapsed)),string(denominator(elapsed)),
                    something(data["clock_id"],missing),something(source["source_id"],missing),something(source["frame_id"],missing),
                    something(source["frame_index"],missing),something(source["label"],missing),something(source["time_unit"],missing),something(source["clock_id"],missing),
                    data["effective_time_unit_provenance"],something(a,missing),something(b,missing))
                println(io,join(_csv.(vals),','))
            end
        end
    end
    path
end
