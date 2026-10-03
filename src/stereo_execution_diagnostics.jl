const STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION = 1
const _StereoExecutionAxis=NamedTuple{(:first,:last,:count,:step),Tuple{Float64,Float64,Int,Float64}}
const _StereoExecutionGeometry=NamedTuple{(:x,:y,:z,:measurement_shape,:camera_basis,:world_basis),
    Tuple{_StereoExecutionAxis,_StereoExecutionAxis,Float64,Tuple{Int,Int},String,String}}

"""
    StereoPIVExecutionDiagnostics

Immutable per-camera execution observations for one completed stereo analysis.
`cam1` and `cam2` retain existing planar pass semantics, in dewarped processing
pixels. Their distinct execution IDs share this packet's parent execution ID and
pair index through explicit camera-role associations in the primitive data.
`geometry` describes the provided common world grid, including signed spacing.
No images, fields, calibration objects or sweep traces are retained.

Measurement-field binding covers reconstructed coordinates/components/UQ/flags/
scale and both cameras' planar measurement fields. Parameter objects and retained
correlation planes are excluded; this is not full result serialization identity.
The inspection verification state is runtime-only, never a stored assertion of
result, calibration or source verification. Use [`execution_diagnostics_data`](@ref).
"""
struct StereoPIVExecutionDiagnostics
    execution_id::String
    pair_index::Union{Nothing,Int}
    cam1::PIVExecutionDiagnostics
    cam2::PIVExecutionDiagnostics
    geometry::_StereoExecutionGeometry
    measurement_sha256::String
    camera_measurement_sha256::Tuple{String,String}
    _metadata_sha256::String
    _verification::Symbol
end

function _stereo_execution_geometry(grid)
    axis(r)=(first=Float64(first(r)),last=Float64(last(r)),count=length(r),step=Float64(step(r)))
    (x=axis(grid.x),y=axis(grid.y),z=grid.z,
     camera_basis="dewarped_pixels_x_columns_y_rows",world_basis="provided_dewarp_grid_coordinates")
end
function _stereo_execution_shape(result)
    dims=(length(result.y),length(result.x))
    all(a->size(a)==dims,(result.u,result.v,result.w,result.uncertainty_u,
        result.uncertainty_v,result.uncertainty_w,result.mask,result.outliers)) ||
        _execution_error("stereo measurement field dimensions disagree")
    all(isfinite,result.x) && all(isfinite,result.y) && isfinite(result.z) ||
        _execution_error("stereo measurement coordinates must be finite")
    result.cam1.x==result.cam2.x && result.cam1.y==result.cam2.y ||
        _execution_error("camera measurement grids disagree")
    for camera in (result.cam1,result.cam2)
        (length(camera.y),length(camera.x))==dims && all(a->size(a)==dims,
            (camera.u,camera.v,camera.peak_ratio,camera.correlation_moment,camera.uncertainty_u,
             camera.uncertainty_v,camera.mask,camera.outliers)) ||
            _execution_error("camera measurement dimensions disagree with stereo field")
    end
    dims
end
function _stereo_execution_measurement_digest(result::StereoPIVResult;
    camera_hashes=(_history_result_digest(result.cam1),_history_result_digest(result.cam2)))
    _stereo_execution_shape(result)
    data=Dict{String,Any}(String(k)=>getfield(result,k) for k in
        (:x,:y,:z,:u,:v,:w,:uncertainty_u,:uncertainty_v,:uncertainty_w,:mask,:outliers))
    data["scale"]=_timing_scale(result.scale)
    data["cam1_measurement_sha256"],data["cam2_measurement_sha256"]=camera_hashes
    _history_digest(data)
end

function _stereo_execution_payload(d::StereoPIVExecutionDiagnostics)
    Dict{String,Any}("stereo_diagnostics_format_version"=>STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION,
        "execution_id"=>d.execution_id,"pair_index"=>d.pair_index,
        "geometry"=>_execution_primitive(d.geometry),"measurement_sha256"=>d.measurement_sha256,
        "binding_basis"=>"stereo_and_camera_measurement_fields_excluding_parameters_and_planes",
        "cameras"=>[Dict{String,Any}("camera_role"=>role,"parent_execution_id"=>d.execution_id,
            "measurement_sha256"=>d.camera_measurement_sha256[role],
            "diagnostics"=>execution_diagnostics_data(camera)) for (role,camera) in enumerate((d.cam1,d.cam2))])
end
function _stereo_execution_validate(data)
    _experiment_keys(data,["stereo_diagnostics_format_version","execution_id","pair_index","geometry",
        "measurement_sha256","binding_basis","cameras"],"stereo execution diagnostics")
    data["stereo_diagnostics_format_version"]===STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION ||
        _execution_error("unsupported stereo diagnostics version")
    id=data["execution_id"]
    id isa String && try UUIDs.UUID(id);true catch;false end || _execution_error("invalid stereo execution ID")
    pair=data["pair_index"]
    pair===nothing || pair isa Int && pair>0 || _execution_error("invalid stereo pair index")
    _experiment_hash(data["measurement_sha256"]) &&
        data["binding_basis"]=="stereo_and_camera_measurement_fields_excluding_parameters_and_planes" ||
        _execution_error("invalid stereo measurement binding")
    geometry=data["geometry"]
    _experiment_keys(geometry,["x","y","z","measurement_shape","camera_basis","world_basis"],"stereo diagnostics geometry")
    geometry["camera_basis"]=="dewarped_pixels_x_columns_y_rows" &&
        geometry["world_basis"]=="provided_dewarp_grid_coordinates" || _execution_error("unsupported stereo coordinate basis")
    geometry["z"] isa Float64 && isfinite(geometry["z"]) || _execution_error("invalid world plane")
    measurement_shape=geometry["measurement_shape"]
    measurement_shape isa AbstractVector && length(measurement_shape)==2 &&
        all(v->v isa Int && v>0,measurement_shape) || _execution_error("invalid stereo measurement shape")
    nodes=try Base.Checked.checked_mul(measurement_shape...) catch;_execution_error("stereo measurement shape overflows");end
    for axis in (geometry["x"],geometry["y"])
        _experiment_keys(axis,["first","last","count","step"],"world-grid axis")
        axis["count"] isa Int && axis["count"]>=2 &&
            all(k->axis[k] isa Float64 && isfinite(axis[k]),("first","last","step")) &&
            axis["step"]!=0 && axis["step"]==(axis["last"]-axis["first"])/(axis["count"]-1) ||
            _execution_error("invalid finite monotonic world-grid axis")
    end
    cameras=data["cameras"]
    cameras isa AbstractVector && length(cameras)==2 || _execution_error("stereo diagnostics require two camera roles")
    children=Dict{String,Any}[]
    for (role,camera) in enumerate(cameras)
        _experiment_keys(camera,["camera_role","parent_execution_id","measurement_sha256","diagnostics"],"stereo camera association")
        camera["camera_role"]===role && camera["parent_execution_id"]==id || _execution_error("invalid camera/parent association")
        _experiment_hash(camera["measurement_sha256"]) || _execution_error("invalid camera measurement binding")
        child=_execution_validate(camera["diagnostics"])
        child["pair_index"]===pair && child["association"]===nothing || _execution_error("invalid camera pair/recipe association")
        push!(children,child)
        for pass in child["passes"]
            all(k->pass["residual"][k]===nothing || pass["residual"][k] isa Float64,
                ("mean_magnitude","rms_magnitude","maximum_magnitude")) || _execution_error("stereo residual observations must use immutable Float64 scalars")
            check=pass["last_check"]
            check===nothing || check["value"]===nothing || check["value"] isa Float64 ||
                _execution_error("stereo convergence observations must use immutable Float64 scalars")
        end
        residual=last(child["passes"])["residual"]
        _execution_sum(_execution_sum(residual["finite_count"],residual["nonfinite_count"]),residual["masked_count"])==nodes ||
            _execution_error("camera final support does not match captured measurement shape")
    end
    length(unique([id;getindex.(children,"execution_id")]))==3 || _execution_error("camera execution IDs must be distinct")
    children[1]["processing_size"]==children[2]["processing_size"] &&
        children[1]["backend"]==children[2]["backend"] && children[1]["image_type"]==children[2]["image_type"] &&
        children[1]["core_source_sha256"]==children[2]["core_source_sha256"] ||
        _execution_error("camera processing shape/precision/backend/source identities disagree")
    length(children[1]["passes"])==length(children[2]["passes"]) &&
        all(p->p[1]["requested_iterations"]==p[2]["requested_iterations"] &&
            p[1]["requested_tolerance"]==p[2]["requested_tolerance"],zip(children[1]["passes"],children[2]["passes"])) ||
        _execution_error("camera requested pass schedules disagree")
    shape=children[1]["processing_size"]
    shape[1]<=geometry["y"]["count"] && shape[2]<=geometry["x"]["count"] ||
        _execution_error("camera processing shape exceeds common world grid")
    data
end

function _stereo_execution_decode(data,digest,verification)
    _stereo_execution_validate(data)
    _experiment_hash(digest) && _history_digest(data)==digest || _execution_error("stereo diagnostics metadata digest mismatch")
    axis(a)=(first=a["first"],last=a["last"],count=a["count"],step=a["step"])
    g=data["geometry"]
    geometry=(x=axis(g["x"]),y=axis(g["y"]),z=g["z"],measurement_shape=Tuple(g["measurement_shape"]),
        camera_basis=g["camera_basis"],world_basis=g["world_basis"])
    StereoPIVExecutionDiagnostics(data["execution_id"],data["pair_index"],
        _execution_decode(data["cameras"][1]["diagnostics"]),_execution_decode(data["cameras"][2]["diagnostics"]),
        geometry,data["measurement_sha256"],Tuple(c["measurement_sha256"] for c in data["cameras"]),digest,verification)
end
function _stereo_execution_finish(cameras,result,geometry,pair_index,source_before)
    all(d->d isa PIVExecutionDiagnostics,cameras) || _execution_error("missing camera execution diagnostics")
    children=map(d->_execution_context(d,pair_index,nothing),cameras)
    all(d->d.core_source_sha256==source_before,children) &&
        _experiment_software()["core_source_sha256"]==source_before || _execution_error("core source changed during stereo execution")
    geometry=(x=geometry.x,y=geometry.y,z=geometry.z,measurement_shape=size(result.u),
        camera_basis=geometry.camera_basis,world_basis=geometry.world_basis)
    camera_hashes=(_history_result_digest(result.cam1),_history_result_digest(result.cam2))
    d=StereoPIVExecutionDiagnostics(string(UUIDs.uuid4()),pair_index,children...,geometry,
        _stereo_execution_measurement_digest(result;camera_hashes),camera_hashes,"",:captured_measurement_fields)
    payload=_stereo_execution_payload(d)
    packet=_stereo_execution_decode(payload,_history_digest(payload),:captured_measurement_fields)
    _stereo_execution_check_result(packet,result)
    packet
end
function _stereo_execution_checked(d)
    d._verification in (:captured_measurement_fields,:metadata_only,:loaded_measurement_fields_verified) ||
        _execution_error("invalid stereo inspection state")
    data=_stereo_execution_payload(d)
    _stereo_execution_validate(data)
    _history_digest(data)==d._metadata_sha256 || _execution_error("stereo diagnostics packet changed")
    data
end
function _stereo_execution_check_result(d,result)
    data=_stereo_execution_checked(d)
    result isa StereoPIVResult || _execution_error("stereo diagnostics require StereoPIVResult")
    dims=_stereo_execution_shape(result)
    dims==d.geometry.measurement_shape && result.z==d.geometry.z || _execution_error("stereo result shape/plane differs from captured geometry")
    for (axis,world_axis,grid_axis) in ((result.cam1.x,result.x,d.geometry.x),(result.cam1.y,result.y,d.geometry.y))
        all(isfinite,axis) && all(v->1<=v<=grid_axis.count,axis) && all(>(0),diff(axis)) ||
            _execution_error("camera measurement axes must be finite increasing dewarped coordinates")
        T=eltype(world_axis)
        expected=T[grid_axis.first+(Float64(v)-1)*grid_axis.step for v in axis]
        world_axis==expected || _execution_error("stereo coordinates disagree with provided world grid")
    end
    result.cam1.scale===nothing && result.cam2.scale===nothing || _execution_error("camera fields must retain raw dewarped pixels")
    camera_hashes=(_history_result_digest(result.cam1),_history_result_digest(result.cam2))
    camera_hashes==d.camera_measurement_sha256 || _execution_error("camera measurement fields changed or camera roles were swapped")
    _stereo_execution_measurement_digest(result;camera_hashes)==d.measurement_sha256 ||
        _execution_error("stereo measurement fields changed after diagnostics capture")
    for camera in data["cameras"]
        residual=last(camera["diagnostics"]["passes"])["residual"]
        residual["finite_count"]+residual["nonfinite_count"]+residual["masked_count"]==prod(dims) ||
            _execution_error("camera diagnostic support does not match measurement grid")
    end
    nothing
end

"""
    execution_diagnostics_data(diagnostics::StereoPIVExecutionDiagnostics)

Return detached stereo metadata and inspection-only verification status. Camera
pass observations retain planar processing-pixel semantics, including empty
tolerance comparisons and primary residuals before validation/filling. World
grid spacing is signed; scalar pixel residual magnitudes cannot be transformed
into an anisotropic world norm or reconstructed 3C residual. No convergence,
accuracy, calibration, synchronization or final-vector association is certified.
The runtime verification-at-capture/read status is not persisted.
"""
function execution_diagnostics_data(d::StereoPIVExecutionDiagnostics)
    data=_stereo_execution_checked(d)
    data["verification"]=Dict{String,Any}("inspection_state"=>String(d._verification),
        "measurement_field_binding_checked"=>d._verification!==:metadata_only,
        "calibration"=>false,"source_inputs"=>false,"full_result_serialization"=>false)
    data
end

_stereo_execution_key(key)="stereo_execution_diagnostics/"*last(split(key,'/'))
function _write_stereo_execution_diagnostics(file,key,d,result)
    _stereo_execution_check_result(d,result)
    if haskey(file,"stereo_execution_diagnostics_format_version")
        file["stereo_execution_diagnostics_format_version"]===STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION ||
            _execution_error("unsupported existing stereo diagnostics version")
    else
        file["stereo_execution_diagnostics_format_version"]=STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION
    end
    file[_stereo_execution_key(key)]=Dict("result_key"=>key,"diagnostics"=>_stereo_execution_checked(d),
        "diagnostics_sha256"=>d._metadata_sha256)
    nothing
end

"""
    load_stereo_execution_diagnostics(index::ResultFile, i; verify_result=false)
    load_stereo_execution_diagnostics(path, i=1; verify_result=false)

Read one optional native stereo companion, returning `nothing` when not recorded.
The default reads only metadata: schema, scalar counts, parent/camera association,
metadata digest and exact result-key linkage. It does not check result fields or
calibration. `verify_result=true` loads/discards exactly one selected stereo result
and verifies its measurement-field binding, including both retained cameras.
The returned packet retains no result/handle and marks verification at read time.
Native result format 1 and planar companion schemas/readers remain unchanged.
File size/mtime checks detect some changes; concurrent writers are unsupported.
Bare/older result saves cannot recover execution diagnostics from settings.
"""
function load_stereo_execution_diagnostics(index::ResultFile,i::Integer;verify_result::Bool=false)
    checkbounds(index,i);_check_result_file(index)
    key="results/"*index.entry_keys[i]
    packet=jldopen(index.path,"r") do file
        _check_results_format(file,index.path)
        if !haskey(file,"stereo_execution_diagnostics_format_version")
            haskey(file,"stereo_execution_diagnostics") && _execution_error("stereo metadata lacks version")
            return nothing
        end
        file["stereo_execution_diagnostics_format_version"]===STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION ||
            _execution_error("unsupported stereo diagnostics version")
        haskey(file,_stereo_execution_key(key)) || return nothing
        entry=file[_stereo_execution_key(key)]
        _experiment_keys(entry,["result_key","diagnostics","diagnostics_sha256"],"stereo execution entry")
        entry["result_key"]==key || _execution_error("stereo diagnostics/result key mismatch")
        d=_stereo_execution_decode(entry["diagnostics"],entry["diagnostics_sha256"],:metadata_only)
        if verify_result
            _stereo_execution_check_result(d,file[key])
            d=_stereo_execution_decode(entry["diagnostics"],entry["diagnostics_sha256"],:loaded_measurement_fields_verified)
        end
        d
    end
    _check_result_file(index)
    packet
end
load_stereo_execution_diagnostics(path::AbstractString,i::Integer=1;kwargs...)=
    load_stereo_execution_diagnostics(ResultFile(_artifact_local_path(path)),i;kwargs...)

Base.show(io::IO,d::StereoPIVExecutionDiagnostics)=print(io,"StereoPIVExecutionDiagnostics(two cameras, ",d._verification,")")
function Base.show(io::IO,::MIME"text/plain",d::StereoPIVExecutionDiagnostics)
    execution_diagnostics_data(d)
    print(io,"Stereo execution: ",d.execution_id,"; ",d._verification)
    for (role,camera) in enumerate((d.cam1,d.cam2))
        print(io,"\nCamera ",role," (dewarped pixels):\n")
        show(io,MIME"text/plain"(),camera)
    end
    print(io,"\nWorld-grid steps: ",d.geometry.x.step,", ",d.geometry.y.step,
        "; calibration/source/synchronization are not verified.")
end
