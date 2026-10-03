const STEREO_PAIR_TIMING_FORMAT_VERSION = 1

"""
    StereoPairTiming

Detached timing/source companion for one completed stereo sequence acquisition.
Each camera retains its own exact timestamps, delay and provided-timestamp
midpoint. A reconstructed time reference, when available, is explicitly camera 1;
there is no common exposure midpoint or synchronization certification. Measurement
binding includes raw reconstructed and retained camera fields, excluding parameters
and correlation planes. No result, source object or image is retained.
"""
struct StereoPairTiming
    _data::Dict{String,Any}
    _sha256::String
    _verification::Symbol
end

_stereo_timing_capture(kwargs) = get(kwargs,:record_pair_timing,false) || get(kwargs,:on_pair_timing,nothing)!==nothing
function _stereo_timing_guard(kwargs)
    any(k->haskey(kwargs,k),(:scale_pairs,:timing_pairs2,:_timing_snapshots)) &&
        _timing_error("private stereo timing/scale plumbing is not a public keyword")
    filtered=(; (k=>v for (k,v) in pairs(kwargs) if k ∉ (:on_pair_timing,:record_pair_timing,:timing_atol,:timing_rtol))...)
    _reject_execution_diagnostics(filtered;stereo_supported=true)
end

# Unlike the planar companion, stereo's existing missing-timestamp policy accepts
# both nothing and missing. Normalize just this adapter, without changing planar.
function _stereo_timing_frame(frame)
    if frame isa FrameRef
        t=frame_timestamp(frame.source,frame.index)
        return _timing_frame(frame;timestamp_value=ismissing(t) ? nothing : t)
    end
    _timing_frame(frame)
end

const _STEREO_CAMERA_TIME_FIELDS=("timestamp_status","time_unit_status","clock_status","pair_time_unit","pair_clock_id",
    "observed_delay","declared_delay","sample_time","sample_time_convention")
function _stereo_timing_camera(frames,declared,tolerance,role)
    derived=_timing_derive(frames,declared,nothing,tolerance)
    data=Dict{String,Any}("camera_role"=>role,"frames"=>frames)
    for k in _STEREO_CAMERA_TIME_FIELDS
        data[k]=derived[k]
    end
    data
end

# Preserve the legacy floating subtraction gate. Integer/rational arithmetic is
# safely promoted; exact encodings and delay agreement use the separate tolerance.
function _stereo_timing_original(data)
    r=_timing_decode(data)
    r===nothing ? nothing : convert(_TIMING_TYPES[data["numeric_type"]],r)
end
_stereo_timing_native_difference(a,b) = a isa AbstractFloat && b isa AbstractFloat ? a-b : _timing_ratio(a)-_timing_ratio(b)
function _stereo_timing_sync_bound(available,t)
    known=filter(!isnothing,available)
    exact_delay=isempty(known) ? 0//big(1) : maximum(_timing_ratio(x) for x in known)
    floating=!isempty(known) && all(x->x isa AbstractFloat,known)
    native_bound=floating ? t["atol"]+t["rtol"]*maximum(known) :
        _timing_ratio(t["atol"])+_timing_ratio(t["rtol"])*exact_delay
    _timing_ratio(native_bound)
end
function _stereo_timing_sync(cameras,policy)
    _experiment_keys(policy,["atol","rtol","missing_timestamps"],"stereo synchronization policy")
    policy["atol"] isa Float64 && policy["rtol"] isa Float64 || _timing_error("invalid synchronization tolerance types")
    t=_timing_tolerance(policy["atol"],policy["rtol"])
    policy["missing_timestamps"] in ("allow","error") || _timing_error("invalid missing timestamp policy")
    times=[Any[_stereo_timing_original(f["timestamp"]) for f in c["frames"]] for c in cameras]
    policy["missing_timestamps"]=="error" && any(isnothing,Iterators.flatten(times)) &&
        _timing_error("stereo timing is missing required exposure timestamps")
    delays=map(eachindex(cameras)) do role
        a,b=times[role]
        observed=a===nothing || b===nothing ? nothing : _stereo_timing_native_difference(b,a)
        declared=_stereo_timing_original(cameras[role]["declared_delay"])
        observed===nothing || observed>0 || _timing_error("stereo exposure timestamps must increase")
        (observed,declared)
    end
    available=Any[o===nothing ? d : o for (o,d) in delays]
    bound=_stereo_timing_sync_bound(available,t)
    for (o,d) in delays
        o===nothing || d===nothing || abs(_timing_ratio(_stereo_timing_native_difference(o,d)))<=bound || _timing_error("declared stereo delay disagrees with exposure timestamps")
    end
    any(isnothing,available) || abs(_timing_ratio(_stereo_timing_native_difference(available[1],available[2])))<=bound || _timing_error("stereo camera delays disagree")
    offsets=map(1:2) do k
        a,b=times[1][k],times[2][k]
        offset=a===nothing || b===nothing ? nothing : _timing_ratio(b)-_timing_ratio(a)
        # Native floating subtraction preserves the admission rule; stored offsets
        # remain exact differences of the supplied numeric values.
        a===nothing || b===nothing || abs(_timing_ratio(_stereo_timing_native_difference(b,a)))<=bound || _timing_error("stereo exposures are unsynchronized")
        _timing_number(offset;derived=true)
    end
    complete=all(f->f["timestamp"]!==nothing,Iterators.flatten(c["frames"] for c in cameras))
    labelled=all(f->f["time_unit"]!==nothing && f["clock_id"]!==nothing,Iterators.flatten(c["frames"] for c in cameras))
    Dict{String,Any}("exposure_offsets_camera2_minus_camera1"=>offsets,
        "delay_scaled_bound"=>_timing_number(bound;derived=true),
        "comparison_status"=>complete ? "all_numeric_comparisons_checked" : "available_numeric_comparisons_only",
        "clock_unit_status"=>labelled ? "provided_labels_consistent" : "unknown_or_partial_labels",
        "comparison_arithmetic"=>"native_float_pairs_and_all_float_delay_bound_otherwise_exact",
        "certification"=>"none_provided_metadata_only")
end

function _stereo_timing_derive(cameras,scale,tolerance,policy)
    frames=collect(Iterators.flatten(c["frames"] for c in cameras))
    for key in ("time_unit","clock_id")
        known=unique(f[key] for f in frames if f[key]!==nothing)
        length(known)<=1 || _timing_error("stereo $key labels disagree; no clock/unit conversion is performed")
    end
    if scale!==nothing
        all(f->f["time_unit"]===nothing || f["time_unit"]==scale["time_unit"],frames) ||
            _timing_error("explicit stereo timestamp unit disagrees with PhysicalScale.time_unit")
    end
    # Only camera 1's declared dt controls legacy pair-list scale overrides.
    d=_timing_derive(cameras[1]["frames"],cameras[1]["declared_delay"],scale,tolerance)
    Dict{String,Any}("synchronization"=>_stereo_timing_sync(cameras,policy),
        "effective_delay"=>d["effective_delay"],"effective_delay_provenance"=>d["effective_delay_provenance"],
        "effective_agrees_with_camera1_observed"=>d["effective_agrees_with_observed"],
        "scale_applied"=>d["scale_applied"],"scale_time_unit_assumption"=>d["scale_time_unit_assumption"],
        "reconstructed_time_reference"=>cameras[1]["sample_time"],
        "reconstructed_time_reference_convention"=>"camera1_provided_timestamp_midpoint_not_common_exposure_time")
end

function _stereo_timing_preflight(acquisitions,pairs1,pairs2,scale,atol,rtol,sync_atol,sync_rtol,missing_policy)
    tolerance=_timing_tolerance(atol,rtol)
    sync=_timing_tolerance(sync_atol,sync_rtol)
    policy=Dict{String,Any}("atol"=>sync["atol"],"rtol"=>sync["rtol"],"missing_timestamps"=>String(missing_policy))
    snapshots=Dict{String,Any}[]
    for (i,a) in enumerate(acquisitions)
        pairs=pairs1===nothing ? ((a[1],a[2]),(a[3],a[4])) : (pairs1[i],pairs2[i])
        cameras=[_stereo_timing_camera([_stereo_timing_frame(p[1]),_stereo_timing_frame(p[2])],
            _timing_number(p isa FramePair ? p.dt : nothing),tolerance,role) for (role,p) in enumerate(pairs)]
        s=_timing_scale(pairs1===nothing ? scale : pair_scale(scale,pairs[1]))
        data=Dict{String,Any}("stereo_pair_timing_format_version"=>STEREO_PAIR_TIMING_FORMAT_VERSION,
            "input_sequence_index"=>i,"cameras"=>cameras,"scale"=>s,"delay_tolerance"=>copy(tolerance),
            "synchronization_policy"=>copy(policy))
        merge!(data,_stereo_timing_derive(cameras,s,tolerance,policy))
        push!(snapshots,deepcopy(data))
    end
    snapshots
end

function _stereo_timing_result_scale(snapshot)
    s=snapshot["scale"]
    s===nothing ? nothing : PhysicalScale(s["pixel_size"],s["dt"],s["length_unit"],s["time_unit"])
end
function _stereo_timing_protected_inputs(acquisitions)
    paths=String[]
    for f in Iterators.flatten(acquisitions)
        f isa AbstractString && push!(paths,_artifact_local_path(f))
        f isa FrameRef && f.source isa TIFFStack && push!(paths,_artifact_local_path(f.source.path))
    end
    unique(paths)
end
function _stereo_timing_output_guard(path,inputs)
    destination=_artifact_local_path(path)
    any(p->_artifact_alias(destination,p),inputs) && _timing_error("stereo output aliases a selected input file")
    nothing
end
function _stereo_timing_write_sources(file,i,snapshot)
    labels=[f["label"] for c in snapshot["cameras"] for f in c["frames"]]
    any(isnothing,labels) || (file[source_key(i)]=String[labels...])
end

function _stereo_timing_validate(data)
    _experiment_keys(data,["stereo_pair_timing_format_version","input_sequence_index","cameras","scale","delay_tolerance",
        "synchronization_policy","synchronization","effective_delay","effective_delay_provenance",
        "effective_agrees_with_camera1_observed","scale_applied","scale_time_unit_assumption","reconstructed_time_reference",
        "reconstructed_time_reference_convention","geometry","raw_result_sha256","binding_basis"],"stereo pair timing")
    data["stereo_pair_timing_format_version"]===STEREO_PAIR_TIMING_FORMAT_VERSION || _timing_error("unsupported stereo timing version")
    data["input_sequence_index"] isa Int && data["input_sequence_index"]>0 || _timing_error("invalid stereo sequence index")
    tolerance=data["delay_tolerance"]
    _experiment_keys(tolerance,["atol","rtol"],"stereo delay tolerances")
    all(k->tolerance[k] isa Float64,("atol","rtol")) || _timing_error("invalid delay tolerance types")
    _timing_tolerance(tolerance["atol"],tolerance["rtol"])
    cameras=data["cameras"]
    cameras isa AbstractVector && length(cameras)==2 || _timing_error("stereo timing requires two cameras")
    for (role,c) in enumerate(cameras)
        _experiment_keys(c,["camera_role","frames",_STEREO_CAMERA_TIME_FIELDS...],"stereo camera timing")
        c["camera_role"]===role || _timing_error("stereo camera role/order mismatch")
        _timing_validate_frames(c["frames"])
        # Reuse the existing independently validated source/exact-number schema.
        derived=_timing_derive(c["frames"],c["declared_delay"],nothing,tolerance)
        virtual=Dict{String,Any}("pair_timing_format_version"=>PAIR_TIMING_FORMAT_VERSION,
            "input_sequence_index"=>data["input_sequence_index"],"frames"=>c["frames"],"scale"=>nothing,
            "delay_tolerance"=>tolerance,"raw_result_sha256"=>data["raw_result_sha256"])
        merge!(virtual,derived);_timing_validate(virtual)
        expected=_stereo_timing_camera(c["frames"],c["declared_delay"],tolerance,role)
        isequal(c,expected) || _timing_error("inconsistent derived stereo camera timing")
    end
    s=data["scale"]
    if s!==nothing
        _experiment_keys(s,String.(fieldnames(PhysicalScale)),"stereo timing scale")
        all(k->s[k] isa Float64 && isfinite(s[k]) && s[k]>0,("pixel_size","dt")) &&
            all(k->s[k] isa String,("length_unit","time_unit")) || _timing_error("invalid stereo timing scale")
    end
    for (k,v) in _stereo_timing_derive(cameras,s,tolerance,data["synchronization_policy"])
        isequal(data[k],v) || _timing_error("inconsistent reconstructed stereo timing: $k")
        v isa Bool && !(data[k] isa Bool) && _timing_error("invalid Boolean stereo timing field: $k")
    end
    _experiment_hash(data["raw_result_sha256"]) && data["binding_basis"]=="stereo_and_camera_measurement_fields_excluding_parameters_and_planes" ||
        _timing_error("invalid stereo timing measurement binding")
    _stereo_timing_validate_geometry(data["geometry"])
    data
end

function _stereo_timing_validate_geometry(g)
    _experiment_keys(g,["x","y","z","measurement_shape","camera_basis","world_basis"],"stereo timing geometry")
    g["camera_basis"]=="dewarped_pixels_x_columns_y_rows" && g["world_basis"]=="provided_dewarp_grid_coordinates" || _timing_error("invalid coordinate basis")
    g["z"] isa Float64 && isfinite(g["z"]) || _timing_error("invalid measurement plane")
    shape=g["measurement_shape"]
    shape isa AbstractVector && length(shape)==2 && all(n->n isa Int && n>0,shape) || _timing_error("invalid measurement shape")
    try Base.Checked.checked_mul(shape...) catch;_timing_error("measurement size overflows");end
    for a in (g["x"],g["y"])
        _experiment_keys(a,["first","last","count","step"],"stereo world axis")
        a["count"] isa Int && a["count"]>=2 && all(k->a[k] isa Float64 && isfinite(a[k]),("first","last","step")) &&
            a["step"]!=0 && a["step"]==(a["last"]-a["first"])/(a["count"]-1) || _timing_error("invalid world-grid axis")
    end
    shape[1]<=g["y"]["count"] && shape[2]<=g["x"]["count"] || _timing_error("measurement shape exceeds world grid")
end

function _stereo_timing_checked(packet::StereoPairTiming)
    packet._verification in (:captured_measurement_fields,:metadata_only,:loaded_measurement_fields_verified) || _timing_error("invalid timing inspection state")
    _experiment_digest(packet._data)==packet._sha256 || _timing_error("stereo timing packet changed after capture")
    _stereo_timing_validate(packet._data)
end
function _stereo_timing_bind(snapshot,result,grid)
    data=deepcopy(snapshot)
    isequal(data["scale"],_timing_scale(result.scale)) || _timing_error("result scale differs from frozen stereo timing")
    geometry=merge(_stereo_execution_geometry(grid),(measurement_shape=size(result.u),))
    data["geometry"]=_execution_primitive(geometry)
    data["raw_result_sha256"]=_stereo_execution_measurement_digest(result)
    data["binding_basis"]="stereo_and_camera_measurement_fields_excluding_parameters_and_planes"
    packet=StereoPairTiming(data,_experiment_digest(data),:captured_measurement_fields)
    _stereo_pair_timing_check_result(packet,result)
    packet
end
function _stereo_pair_timing_check_result(packet,result)
    data=_stereo_timing_checked(packet)
    result isa StereoPIVResult || _timing_error("stereo timing requires a raw StereoPIVResult")
    g=data["geometry"];dims=_stereo_execution_shape(result)
    collect(dims)==g["measurement_shape"] && result.z==g["z"] || _timing_error("stereo timing shape/plane mismatch")
    for (axis,world,a) in ((result.cam1.x,result.x,g["x"]),(result.cam1.y,result.y,g["y"]))
        all(isfinite,axis) && all(v->1<=v<=a["count"],axis) && all(>(0),diff(axis)) || _timing_error("invalid camera measurement coordinates")
        T=eltype(world)
        world==T[a["first"]+(Float64(v)-1)*a["step"] for v in axis] || _timing_error("world/camera coordinate mismatch")
    end
    result.cam1.scale===nothing && result.cam2.scale===nothing || _timing_error("stereo camera fields must remain raw")
    isequal(data["scale"],_timing_scale(result.scale)) && _stereo_execution_measurement_digest(result)==data["raw_result_sha256"] ||
        _timing_error("raw stereo/camera measurement fields differ from timing binding")
    nothing
end

"""
    pair_timing_data(packet::StereoPairTiming; result=nothing)

Return detached exact timing/source metadata. Optionally verify an already loaded
RAW stereo result's measurement fields and geometry without rereading or retaining
it. Runtime verification status is inspection-only: metadata integrity is not
source-byte, calibration, full-serialization or hardware synchronization verification.
Exact numbers and delay agreement use the planar timing encoding conventions.
"""
function pair_timing_data(packet::StereoPairTiming;result=nothing)
    data=deepcopy(_stereo_timing_checked(packet))
    result===nothing || _stereo_pair_timing_check_result(packet,result)
    state=result===nothing ? packet._verification : :supplied_measurement_fields_verified
    data["verification"]=Dict{String,Any}("inspection_state"=>String(state),"measurement_field_binding_checked"=>state!==:metadata_only,
        "source_inputs"=>false,"calibration"=>false,"full_result_serialization"=>false,"synchronization_certified"=>false)
    data
end

_stereo_pair_timing_key(key)="stereo_pair_timing/"*last(split(key,'/'))
function _write_stereo_pair_timing(file,key,packet,result)
    _stereo_pair_timing_check_result(packet,result)
    if haskey(file,"stereo_pair_timing_format_version")
        file["stereo_pair_timing_format_version"]===STEREO_PAIR_TIMING_FORMAT_VERSION || _timing_error("unsupported existing stereo timing marker")
    else
        file["stereo_pair_timing_format_version"]=STEREO_PAIR_TIMING_FORMAT_VERSION
    end
    file[_stereo_pair_timing_key(key)]=Dict{String,Any}("result_key"=>key,"timing"=>_stereo_timing_checked(packet),"timing_sha256"=>packet._sha256)
    nothing
end

"""
    load_stereo_pair_timing(index::ResultFile, i; verify_result=false)
    load_stereo_pair_timing(path, i=1; verify_result=false)

Read one optional native stereo timing companion. Metadata-only reads validate the
version, schema, exact derived times, ordered camera/source association, digest and
result-key linkage. `verify_result=true` additionally loads/discards one selected
raw result and checks measurement fields and geometry. No payload/handle is retained.
Native result format 1 and planar timing are unchanged; ordinary saves omit timing.
Completed files only: size/mtime checks are not concurrent-writer protection.
"""
function load_stereo_pair_timing(index::ResultFile,i::Integer;verify_result::Bool=false)
    checkbounds(index,i);_check_result_file(index)
    key="results/"*index.entry_keys[i]
    packet=jldopen(index.path,"r") do file
        _check_results_format(file,index.path)
        if !haskey(file,"stereo_pair_timing_format_version")
            haskey(file,"stereo_pair_timing") && _timing_error("stereo timing group lacks version")
            return nothing
        end
        file["stereo_pair_timing_format_version"]===STEREO_PAIR_TIMING_FORMAT_VERSION || _timing_error("unsupported stereo timing version")
        haskey(file,_stereo_pair_timing_key(key)) || return nothing
        saved=file[_stereo_pair_timing_key(key)]
        _experiment_keys(saved,["result_key","timing","timing_sha256"],"stereo timing entry")
        saved["result_key"]==key || _timing_error("stereo timing result-key mismatch")
        saved["timing"] isa Dict{String,Any} && _experiment_hash(saved["timing_sha256"]) || _timing_error("invalid stereo timing storage")
        p=StereoPairTiming(saved["timing"],saved["timing_sha256"],:metadata_only)
        _stereo_timing_checked(p)
        if verify_result
            _stereo_pair_timing_check_result(p,file[key])
            p=StereoPairTiming(p._data,p._sha256,:loaded_measurement_fields_verified)
        end
        p
    end
    _check_result_file(index);packet
end
load_stereo_pair_timing(path::AbstractString,i::Integer=1;kwargs...)=load_stereo_pair_timing(ResultFile(_artifact_local_path(path)),i;kwargs...)
Base.show(io::IO,p::StereoPairTiming)=print(io,"StereoPairTiming(pair=",_stereo_timing_checked(p)["input_sequence_index"],", ",p._verification,")")
