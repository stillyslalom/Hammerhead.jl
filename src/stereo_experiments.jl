const STEREO_EXPERIMENT_FORMAT_VERSION=1
const STEREO_EXPERIMENT_RUN_FORMAT_VERSION=1

"""
    StereoPIVRecipe(passes, dw1, dw2; preprocessing=(PreprocessStep[],PreprocessStep[]),
        mask=nothing, roi=nothing, scale=nothing, backend=:cpu, image_type=Float64,
        threaded=false, predictor_smoothing=true, mask_threshold=.5,
        uncertainty_backend=:same, sync_atol=0, sync_rtol=0,
        missing_timestamps=:allow, timing_atol=0, timing_rtol=sqrt(eps(Float64)),
        world_unit=nothing, coordinate_frame=nothing, calibration_note=nothing,
        self_calibration_report=nothing)

Snapshot stereo PIV from frozen fitted builtin cameras, a common signed world
grid and complete processing settings. CPU/KA Float32/64, builtin validation,
two builtin raw-image preprocessing pipelines, static grid mask and ROI are
supported. Supplied dewarpers must match rebuilt camera/grid maps (NaN-aware).
No camera fitting, self-calibration execution, scripts or custom camera/map
implementation is replayed. Optional self-calibration scalar summaries and notes
are supplied descriptive provenance, not verified calibration transformations.
World/frame labels never infer or override PhysicalScale conversion.
"""
struct StereoPIVRecipe
    _data::Dict{String,Any}
    recipe_id::String
    _sha256::String
end

function _stereo_camera_data(camera; transformed=false)
    if camera isa PinholeCamera
        data=Dict{String,Any}("model"=>"pinhole","P"=>Matrix(camera.P))
    elseif camera isa SoloffCamera
        data=Dict{String,Any}("model"=>"soloff","ax"=>collect(camera.ax),"ay"=>collect(camera.ay),
            "center"=>collect(camera.center),"scale"=>collect(camera.scale))
    elseif camera isa TransformedCamera && !transformed
        data=Dict{String,Any}("model"=>"transformed","camera"=>_stereo_camera_data(camera.cam;transformed=true),
            "R"=>Matrix(camera.R),"t"=>collect(camera.t))
    else
        _experiment_error("stereo recipes require fitted Pinhole/Soloff or one rigid transformed builtin camera")
    end
    _stereo_camera_decode(data)
    data
end
function _stereo_float_array(a,shape,what)
    a isa Array{Float64} && size(a)==shape && all(isfinite,a) || _experiment_error("invalid finite Float64 $what")
    a
end
function _stereo_camera_decode(data;transformed=false)
    data isa AbstractDict && get(data,"model",nothing) isa String || _experiment_error("invalid camera model descriptor")
    model=data["model"]
    if model=="pinhole"
        _experiment_keys(data,["model","P"],"pinhole camera")
        return _pinhole_fitted_snapshot(_stereo_float_array(data["P"],(3,4),"projection matrix"))
    elseif model=="soloff"
        _experiment_keys(data,["model","ax","ay","center","scale"],"Soloff camera")
        ax=_stereo_float_array(data["ax"],(19,),"Soloff x coefficients")
        ay=_stereo_float_array(data["ay"],(19,),"Soloff y coefficients")
        center=_stereo_float_array(data["center"],(3,),"Soloff normalization center")
        scale=_stereo_float_array(data["scale"],(3,),"Soloff normalization scale")
        all(>(0),scale) || _experiment_error("Soloff normalization scales must be positive")
        return SoloffCamera(SVector{19,Float64}(ax),SVector{19,Float64}(ay),SVector{3,Float64}(center),SVector{3,Float64}(scale))
    elseif model=="transformed" && !transformed
        _experiment_keys(data,["model","camera","R","t"],"transformed camera")
        base=_stereo_camera_decode(data["camera"];transformed=true)
        R=_stereo_float_array(data["R"],(3,3),"rigid rotation")
        t=_stereo_float_array(data["t"],(3,),"rigid translation")
        return TransformedCamera(base,R,t)
    end
    _experiment_error("unsupported or nested stereo camera model")
end
_stereo_grid_data(g)=Dict{String,Any}("x"=>Dict("first"=>first(g.x),"last"=>last(g.x),"count"=>length(g.x)),
    "y"=>Dict("first"=>first(g.y),"last"=>last(g.y),"count"=>length(g.y)),"z"=>g.z)
function _stereo_grid_decode(data)
    _experiment_keys(data,["x","y","z"],"common grid")
    for a in (data["x"],data["y"])
        _experiment_keys(a,["first","last","count"],"grid axis")
        a["count"] isa Int && a["count"]>=2 && all(k->a[k] isa Float64 && isfinite(a[k]),("first","last")) && a["first"]!=a["last"] ||
            _experiment_error("grid axes must have finite distinct Float64 endpoints and Int counts")
    end
    data["z"] isa Float64 && isfinite(data["z"]) || _experiment_error("invalid world plane")
    try Base.Checked.checked_mul(data["x"]["count"],data["y"]["count"]) catch;_experiment_error("grid dimensions overflow");end
    DewarpGrid(x=LinRange(data["x"]["first"],data["x"]["last"],data["x"]["count"]),
        y=LinRange(data["y"]["first"],data["y"]["last"],data["y"]["count"]),z=data["z"])
end
_stereo_map_digest(dw)=_experiment_digest(Dict{String,Any}("rows"=>dw.rows,"cols"=>dw.cols,"mask"=>dw.mask))

function _stereo_selfcal_data(report)
    report===nothing && return nothing
    report isa SelfCalibrationReport || _experiment_error("self_calibration_report must be supplied SelfCalibrationReport")
    data=Dict{String,Any}("status"=>"supplied_descriptive_unverified","converged"=>report.converged,"tol"=>report.tol,
        "R"=>Matrix(report.R),"t"=>collect(report.t),"passes"=>[Dict{String,Any}("disparity_rms"=>p.disparity_rms,
        "disparity_median"=>p.disparity_median,"n_vectors"=>p.n_vectors,"triangulation_rms"=>p.triangulation_rms,
        "plane"=>p.plane===nothing ? nothing : Dict{String,Any}(String(k)=>v for (k,v) in pairs(p.plane))) for p in report.passes],
        "disparity_maps"=>"not_embedded","execution_replayed"=>false,"initial_calibration_inputs_verified"=>false)
    _stereo_selfcal_validate(data);data
end
function _stereo_selfcal_validate(data)
    data===nothing && return nothing
    _experiment_keys(data,["status","converged","tol","R","t","passes","disparity_maps","execution_replayed","initial_calibration_inputs_verified"],"supplied self-calibration summary")
    data["status"]=="supplied_descriptive_unverified" && data["converged"] isa Bool && data["disparity_maps"]=="not_embedded" &&
        data["execution_replayed"]===false && data["initial_calibration_inputs_verified"]===false || _experiment_error("unsupported self-calibration provenance claim")
    data["tol"] isa Float64 && isfinite(data["tol"]) && data["tol"]>=0 || _experiment_error("invalid supplied self-calibration tolerance")
    _check_rigid(_stereo_float_array(data["R"],(3,3),"self-calibration rotation"),_stereo_float_array(data["t"],(3,),"self-calibration translation"))
    data["passes"] isa AbstractVector || _experiment_error("invalid self-calibration passes")
    for pass in data["passes"]
        _experiment_keys(pass,["disparity_rms","disparity_median","n_vectors","triangulation_rms","plane"],"self-calibration pass")
        pass["n_vectors"] isa Int && pass["n_vectors"]>=0 || _experiment_error("invalid supplied disparity count")
        all(k->pass[k] isa Float64 && (isnan(pass[k]) || isfinite(pass[k]) && pass[k]>=0),
            ("disparity_rms","disparity_median","triangulation_rms")) || _experiment_error("invalid supplied disparity statistic")
        if pass["plane"]!==nothing
            _experiment_keys(pass["plane"],["a","b","c"],"supplied fitted plane")
            all(v->v isa Float64 && isfinite(v),values(pass["plane"])) || _experiment_error("invalid supplied fitted plane")
        end
    end
    nothing
end

function _stereo_recipe_config(data)
    _experiment_keys(data,["kind","cameras","grid","raw_image_sizes","processing","preprocessing","mask","scale","synchronization_policy",
        "delay_tolerance","world_unit","coordinate_frame","calibration_provenance","reference_map_sha256"],"stereo recipe")
    data["kind"]=="frozen_fitted_stereo_piv" || _experiment_error("unsupported stereo recipe kind")
    data["cameras"] isa AbstractVector && length(data["cameras"])==2 || _experiment_error("recipe requires two cameras")
    cams=Tuple(_stereo_camera_decode(c) for c in data["cameras"])
    grid=_stereo_grid_decode(data["grid"])
    sizes=data["raw_image_sizes"]
    sizes isa AbstractVector && length(sizes)==2 && all(s->s isa Vector{Int} && length(s)==2 && all(>(0),s),sizes) || _experiment_error("invalid raw camera image sizes")
    processing=_experiment_recipe_from_data(data["processing"],nothing)
    _experiment_digest(_experiment_recipe_data(processing))==_experiment_digest(data["processing"]) || _experiment_error("noncanonical camera processing settings")
    processing.mask===nothing && processing.scale===nothing && processing.external_preprocess===nothing && isempty(processing.preprocessing) ||
        _experiment_error("camera processing must not invent mask/scale/script preprocessing")
    selected=processing.roi===nothing ? size(grid) : (length(processing.roi.rows),length(processing.roi.cols))
    processing.roi===nothing || (last(processing.roi.rows)<=size(grid)[1] && last(processing.roi.cols)<=size(grid)[2]) || _experiment_error("ROI exceeds world grid")
    all(p->all(p.search_area_size .<= selected),processing.passes) || _experiment_error("pass search area exceeds dewarped image/ROI")
    steps=data["preprocessing"]
    steps isa AbstractVector && length(steps)==2 || _experiment_error("two preprocessing pipelines required")
    pipelines=map(1:2) do role
        steps[role] isa AbstractVector || _experiment_error("invalid camera preprocessing pipeline")
        parsed=PreprocessStep[]
        for step in steps[role]
            _experiment_keys(step,["operation","options"],"camera preprocessing step")
            step["operation"] isa String && step["options"] isa AbstractDict || _experiment_error("invalid preprocessing descriptor")
            s=PreprocessStep(Symbol(step["operation"]);(Symbol(k)=>v for (k,v) in step["options"])...)
            _experiment_digest(Dict{String,Any}("operation"=>String(s.operation),"options"=>s.options))==_experiment_digest(step) || _experiment_error("noncanonical preprocessing options")
            s.operation===:subtract_background && size(s.options["background"])!=Tuple(sizes[role]) && _experiment_error("camera background shape differs from raw images")
            push!(parsed,s)
        end
        parsed
    end
    mask=data["mask"]
    mask===nothing || mask isa BitMatrix && size(mask)==size(grid) || _experiment_error("static mask must match the common full world grid")
    scale=data["scale"]
    if scale!==nothing
        _experiment_keys(scale,String.(fieldnames(PhysicalScale)),"stereo scale")
        all(k->scale[k] isa Float64 && isfinite(scale[k]) && scale[k]>0,("pixel_size","dt")) &&
            all(k->scale[k] isa String,("length_unit","time_unit")) || _experiment_error("invalid stereo scale")
        scale=PhysicalScale(scale["pixel_size"],scale["dt"],scale["length_unit"],scale["time_unit"])
    end
    for k in ("world_unit","coordinate_frame")
        _source_metadata_string(data[k],k)
    end
    policy=data["synchronization_policy"]
    _experiment_keys(policy,["atol","rtol","missing_timestamps"],"stereo synchronization settings")
    all(k->policy[k] isa Float64,("atol","rtol")) || _experiment_error("invalid synchronization tolerance types")
    _timing_tolerance(policy["atol"],policy["rtol"])
    policy["missing_timestamps"] in ("allow","error") || _experiment_error("invalid missing timestamp policy")
    tolerance=data["delay_tolerance"]
    _experiment_keys(tolerance,["atol","rtol"],"stereo timing settings")
    all(k->tolerance[k] isa Float64,("atol","rtol")) || _experiment_error("invalid delay tolerance types")
    _timing_tolerance(tolerance["atol"],tolerance["rtol"])
    provenance=data["calibration_provenance"]
    _experiment_keys(provenance,["status","note","self_calibration"],"calibration provenance")
    _source_metadata_string(provenance["note"],"calibration note")
    provenance["status"]==(provenance["note"]===nothing && provenance["self_calibration"]===nothing ? "unknown" : "supplied_unverified") || _experiment_error("unsupported calibration provenance claim")
    _stereo_selfcal_validate(provenance["self_calibration"])
    maps=data["reference_map_sha256"]
    maps isa Vector{String} && length(maps)==2 && all(_experiment_hash,maps) || _experiment_error("invalid reference dewarp map identities")
    (;cams,grid,sizes,processing,pipelines,mask,scale,policy,tolerance)
end
function _stereo_recipe_science(data)
    # Rebuilt maps are environment-dependent derived provenance, not settings.
    Dict{String,Any}(k=>v for (k,v) in data if k ∉ ("reference_map_sha256","calibration_provenance"))
end

function StereoPIVRecipe(passes,dw1::ImageDewarper,dw2::ImageDewarper;
    preprocessing=(PreprocessStep[],PreprocessStep[]),mask=nothing,roi=nothing,scale=nothing,
    backend::Symbol=:cpu,image_type::DataType=Float64,threaded::Bool=false,predictor_smoothing::Bool=true,
    mask_threshold::Real=.5,uncertainty_backend::Symbol=:same,sync_atol::Real=0.,sync_rtol::Real=0.,
    missing_timestamps::Symbol=:allow,timing_atol::Real=0.,timing_rtol::Real=sqrt(eps(Float64)),
    world_unit=nothing,coordinate_frame=nothing,calibration_note=nothing,self_calibration_report=nothing)
    dw1.grid==dw2.grid || _experiment_error("stereo recipe requires one shared grid")
    preprocessing isa Tuple && length(preprocessing)==2 && all(s->s isa AbstractVector{PreprocessStep},preprocessing) || _experiment_error("preprocessing must be two vectors of builtin PreprocessStep")
    mask===nothing || mask isa AbstractMatrix{Bool} || _experiment_error("static grid mask must contain Bool values")
    scale===nothing || scale isa PhysicalScale || _experiment_error("scale must be PhysicalScale or nothing")
    processing=PIVRecipe(passes;roi,backend,image_type,threaded,predictor_smoothing,mask_threshold,uncertainty_backend)
    sync=_timing_tolerance(sync_atol,sync_rtol)
    selfcal=_stereo_selfcal_data(self_calibration_report)
    data=Dict{String,Any}("kind"=>"frozen_fitted_stereo_piv","cameras"=>[_stereo_camera_data(dw1.cam),_stereo_camera_data(dw2.cam)],
        "grid"=>_stereo_grid_data(dw1.grid),"raw_image_sizes"=>[collect(dw1.image_size),collect(dw2.image_size)],
        "processing"=>_experiment_recipe_data(processing),
        "preprocessing"=>[[Dict{String,Any}("operation"=>String(s.operation),"options"=>deepcopy(s.options)) for s in steps] for steps in preprocessing],
        "mask"=>mask===nothing ? nothing : BitMatrix(mask),"scale"=>_timing_scale(scale),
        "synchronization_policy"=>Dict{String,Any}("atol"=>sync["atol"],"rtol"=>sync["rtol"],"missing_timestamps"=>String(missing_timestamps)),
        "delay_tolerance"=>_timing_tolerance(timing_atol,timing_rtol),"world_unit"=>_source_metadata_string(world_unit,"world unit"),
        "coordinate_frame"=>_source_metadata_string(coordinate_frame,"coordinate frame"),
        "calibration_provenance"=>Dict{String,Any}("status"=>calibration_note===nothing && selfcal===nothing ? "unknown" : "supplied_unverified",
            "note"=>_source_metadata_string(calibration_note,"calibration note"),"self_calibration"=>selfcal),
        "reference_map_sha256"=>[_stereo_map_digest(dw1),_stereo_map_digest(dw2)])
    config=_stereo_recipe_config(data)
    for (role,provided) in enumerate((dw1,dw2))
        rebuilt=ImageDewarper(config.cams[role],config.grid,Tuple(config.sizes[role]))
        isequal(provided.rows,rebuilt.rows) && isequal(provided.cols,rebuilt.cols) && isequal(provided.mask,rebuilt.mask) || _experiment_error("supplied dewarper contains undocumented map edits")
    end
    StereoPIVRecipe(deepcopy(data),_experiment_digest(_stereo_recipe_science(data)),_experiment_digest(data))
end

"""
    recipe_identity(recipe::StereoPIVRecipe)

Return the location-independent frozen stereo settings identity after independent
schema and integrity checks. Rebuilt reference-map digests are checked provenance
but excluded from settings identity. No pixel payload or dewarp map is retained.
"""
function recipe_identity(recipe::StereoPIVRecipe)
    _experiment_digest(recipe._data)==recipe._sha256 && _experiment_digest(_stereo_recipe_science(recipe._data))==recipe.recipe_id || _experiment_error("stereo recipe changed after snapshot")
    _stereo_recipe_config(recipe._data);recipe.recipe_id
end

"""
    StereoExperimentRecord(acquisitions, recipe::StereoPIVRecipe; timestamps=(nothing,nothing),
        time_units=(nothing,nothing), clock_ids=(nothing,nothing), source_ids=(nothing,nothing),
        frame_ids=(nothing,nothing))
    StereoExperimentRecord(pairs1, pairs2, recipe::StereoPIVRecipe; kwargs...)

Capture ordered four-file acquisitions and their byte identities/dimensions.
Camera metadata vectors follow each supplied camera's A/B stream (length 2N);
frame indices in replay companions refer to that ordered stream, not an inferred
original recording index. IDs are opaque, optional provided timestamps are exact,
and missing unit/clock labels stay unknown. Four-tuples retain supplied-scale
delay semantics; pair lists also preserve each file FramePair's declared delay
and camera 1's scale override. Arbitrary loaders/FrameRefs are unsupported here.
Calibration fitting and self-calibration execution are not part of replay.
"""
struct StereoExperimentRecord
    recipe::StereoPIVRecipe
    input_files::Vector{Dict{String,Any}}
    pairs::Vector{Vector{Int}}
    timing_metadata::Vector{Dict{String,Any}}
    scaling_mode::Symbol
    input_id::String
    creation_environment::Dict{String,Any}
    runs::Vector{ExperimentRun}
    record_paths::Vector{String}
    _sha256::String
end

function _stereo_record_core(r)
    Dict{String,Any}("recipe"=>r.recipe._data,"recipe_id"=>r.recipe.recipe_id,"recipe_sha256"=>r.recipe._sha256,
        "input_files"=>r.input_files,"pairs"=>r.pairs,"timing_metadata"=>r.timing_metadata,
        "scaling_mode"=>String(r.scaling_mode),"input_id"=>r.input_id,"creation_environment"=>r.creation_environment)
end
function _stereo_input_data(files,pairs,metadata,mode)
    descriptor(f)=Dict{String,Any}(k=>f[k] for k in ("sha256","size_bytes","image_size"))
    acquisitions=[Dict{String,Any}("cameras"=>[Dict{String,Any}("camera_role"=>role,
        "frames"=>[Dict{String,Any}("input"=>descriptor(files[pair[2role-2+k]]),
            "timestamp"=>metadata[role]["timestamps"][2i-2+k]) for k in 1:2],
        "declared_delay"=>metadata[role]["declared_delays"][i],"time_unit"=>metadata[role]["time_unit"],
        "clock_id"=>metadata[role]["clock_id"]) for role in 1:2]) for (i,pair) in enumerate(pairs)]
    Dict{String,Any}("scaling_mode"=>String(mode),"acquisitions"=>acquisitions)
end
function _stereo_two_metadata(value,name)
    value isa Tuple && length(value)==2 || _experiment_error("$name must contain two camera values")
    value
end
function _stereo_frame_id(value)
    id=_source_metadata_string(value,"frame ID")
    id===nothing && _experiment_error("frame IDs cannot contain nothing")
    id
end
function _stereo_capture_record(acquisitions,recipe,declared,mode;
    timestamps=(nothing,nothing),time_units=(nothing,nothing),clock_ids=(nothing,nothing),source_ids=(nothing,nothing),frame_ids=(nothing,nothing))
    recipe_identity(recipe)
    isempty(acquisitions) && _experiment_error("stereo experiment requires acquisitions")
    all(a->a isa Tuple && length(a)==4 && all(f->f isa AbstractString,a),acquisitions) || _experiment_error("stereo experiment acquisitions must be four-file tuples")
    for (name,v) in (("timestamps",timestamps),("time_units",time_units),("clock_ids",clock_ids),("source_ids",source_ids),("frame_ids",frame_ids))
        _stereo_two_metadata(v,name)
    end
    n=length(acquisitions);files=Dict{String,Any}[];pairs=Vector{Int}[]
    config=_stereo_recipe_config(recipe._data)
    for a in acquisitions
        indices=Int[]
        for path in a
            localpath=realpath(_artifact_local_path(path))
            index=findfirst(f->f["path"]==localpath,files)
            if index===nothing
                sha=_experiment_file_digest(localpath);bytes=filesize(localpath)
                image=load_image(config.processing.image_type,localpath)
                _experiment_file_digest(localpath)==sha && filesize(localpath)==bytes || _experiment_error("input changed during capture")
                push!(files,Dict{String,Any}("path"=>localpath,"sha256"=>sha,"size_bytes"=>bytes,"image_size"=>collect(size(image))))
                index=length(files)
            end
            push!(indices,index)
        end
        push!(pairs,indices)
    end
    metadata=map(1:2) do role
        ts=timestamps[role];ids=frame_ids[role]
        ts===nothing || ts isa AbstractVector && length(ts)==2n || _experiment_error("camera timestamps require 2N ordered entries")
        ids===nothing || ids isa AbstractVector && length(ids)==2n || _experiment_error("camera frame IDs require 2N ordered entries")
        encoded=Any[_timing_number(ts===nothing || ismissing(ts[i]) ? nothing : ts[i]) for i in 1:2n]
        frameid=ids===nothing ? nothing : String[_stereo_frame_id(v) for v in ids]
        Dict{String,Any}("camera_role"=>role,"frame_index_scope"=>"provided_ordered_file_stream",
            "timestamps"=>encoded,"declared_delays"=>Any[_timing_number(v) for v in declared[role]],
            "time_unit"=>_source_metadata_string(time_units[role],"time unit"),"clock_id"=>_source_metadata_string(clock_ids[role],"clock ID"),
            "source_id"=>_source_metadata_string(source_ids[role],"source ID"),"frame_ids"=>frameid)
    end
    input_id=_experiment_digest(_stereo_input_data(files,pairs,metadata,mode))
    environment=_experiment_software()
    provisional=StereoExperimentRecord(deepcopy(recipe),files,pairs,metadata,mode,input_id,environment,ExperimentRun[],String[],"")
    record=StereoExperimentRecord(provisional.recipe,files,pairs,metadata,mode,input_id,environment,provisional.runs,provisional.record_paths,
        _experiment_digest(_stereo_record_core(provisional)))
    _stereo_record_preflight(record);record
end
function StereoExperimentRecord(acquisitions::AbstractVector,recipe::StereoPIVRecipe;kwargs...)
    n=length(acquisitions)
    _stereo_capture_record(acquisitions,recipe,(fill(nothing,n),fill(nothing,n)),:four_tuple;kwargs...)
end
function StereoExperimentRecord(pairs1::AbstractVector,pairs2::AbstractVector,recipe::StereoPIVRecipe;kwargs...)
    length(pairs1)==length(pairs2) || _experiment_error("camera pair lists must have equal length")
    frozen=(_timing_freeze_pairs(pairs1),_timing_freeze_pairs(pairs2))
    acquisitions=[(a[1],a[2],b[1],b[2]) for (a,b) in zip(frozen...)]
    declared=Tuple(Any[p isa FramePair ? p.dt : nothing for p in pairs] for pairs in frozen)
    _stereo_capture_record(acquisitions,recipe,declared,:pair_list;kwargs...)
end

function _stereo_record_frames(record,config;load_pixels=false)
    n=length(record.pairs)
    sources=map(1:2) do role
        metadata=record.timing_metadata[role]
        indices=Int[pair[k] for pair in record.pairs for k in (2role-1,2role)]
        FrameSource(2n,i->load_pixels ? _experiment_load_image(record.input_files[indices[i]],config.processing.image_type) : _experiment_error("metadata preflight must not load pixels");
            timestamps=Any[_stereo_timing_original(t) for t in metadata["timestamps"]],
            labels=String[record.input_files[j]["path"] for j in indices],source_id=metadata["source_id"],frame_ids=metadata["frame_ids"],
            time_unit=metadata["time_unit"],clock_id=metadata["clock_id"])
    end
    camera_pairs=map(1:2) do role
        [begin
            a,b=FrameRef(sources[role],2i-1),FrameRef(sources[role],2i)
            dt=_stereo_timing_original(record.timing_metadata[role]["declared_delays"][i])
            dt===nothing ? (a,b) : FramePair(a,b,dt)
        end for i in 1:n]
    end
    acquisitions=[(a[1],a[2],b[1],b[2]) for (a,b) in zip(camera_pairs...)]
    (;camera_pairs,acquisitions)
end
function _stereo_record_timing(record,config)
    frames=_stereo_record_frames(record,config)
    p1,p2=record.scaling_mode===:pair_list ? Tuple(frames.camera_pairs) : (nothing,nothing)
    _stereo_timing_preflight(frames.acquisitions,p1,p2,config.scale,config.tolerance["atol"],config.tolerance["rtol"],
        config.policy["atol"],config.policy["rtol"],Symbol(config.policy["missing_timestamps"]))
end

function _stereo_record_preflight(record;verify_files=false)
    recipe_identity(record.recipe)
    _experiment_digest(_stereo_record_core(record))==record._sha256 || _experiment_error("stereo record snapshot changed")
    config=_stereo_recipe_config(record.recipe._data)
    record.scaling_mode in (:four_tuple,:pair_list) || _experiment_error("invalid stereo scaling mode")
    isempty(record.pairs) && _experiment_error("stereo experiment has no acquisitions")
    isempty(record.input_files) && _experiment_error("stereo experiment has no files")
    for file in record.input_files
        _experiment_keys(file,["path","sha256","size_bytes","image_size"],"stereo input file")
        _artifact_absolute_locator(file["path"]) && _experiment_hash(file["sha256"]) && file["size_bytes"] isa Int && file["size_bytes"]>=0 || _experiment_error("invalid stereo input identity")
        file["image_size"] isa Vector{Int} && length(file["image_size"])==2 && all(>(0),file["image_size"]) || _experiment_error("invalid input image dimensions")
    end
    for pair in record.pairs
        pair isa Vector{Int} && length(pair)==4 && all(i->1<=i<=length(record.input_files),pair) || _experiment_error("invalid stereo ordered acquisition")
        for role in 1:2,k in (2role-1,2role)
            record.input_files[pair[k]]["image_size"]==config.sizes[role] || _experiment_error("raw input dimensions disagree with calibrated camera")
        end
    end
    sort!(unique!(vcat(record.pairs...)))==collect(1:length(record.input_files)) || _experiment_error("unreferenced stereo input file")
    n=length(record.pairs)
    length(record.timing_metadata)==2 || _experiment_error("two ordered camera metadata entries required")
    for (role,m) in enumerate(record.timing_metadata)
        _experiment_keys(m,["camera_role","frame_index_scope","timestamps","declared_delays","time_unit","clock_id","source_id","frame_ids"],"ordered camera timing metadata")
        m["camera_role"]===role && m["frame_index_scope"]=="provided_ordered_file_stream" || _experiment_error("invalid camera source association")
        m["timestamps"] isa AbstractVector && length(m["timestamps"])==2n && m["declared_delays"] isa AbstractVector && length(m["declared_delays"])==n || _experiment_error("invalid timing vector dimensions")
        foreach(_timing_decode,m["timestamps"]);foreach(_timing_positive,m["declared_delays"])
        record.scaling_mode===:four_tuple && any(!isnothing,m["declared_delays"]) && _experiment_error("four-tuples cannot imply a pair-dt scaling override")
        for k in ("time_unit","clock_id","source_id")
            _source_metadata_string(m[k],k)
        end
        ids=m["frame_ids"]
        ids===nothing || ids isa Vector{String} && length(ids)==2n && all(!isempty,ids) || _experiment_error("invalid provided frame IDs")
    end
    _experiment_digest(_stereo_input_data(record.input_files,record.pairs,record.timing_metadata,record.scaling_mode))==record.input_id || _experiment_error("stereo scientific input identity mismatch")
    _stereo_record_timing(record,config)
    _experiment_validate_environment(record.creation_environment)
    _check_backend_params(_resolve_backend(config.processing.backend),config.processing.passes)
    if verify_files
        for file in record.input_files
            _artifact_local_path(file["path"])
            _experiment_load_image(file,config.processing.image_type)
        end
    end
    config
end

function _stereo_run_validate(data,record)
    _experiment_keys(data,String.(fieldnames(ExperimentRun)),"stereo run")
    data["run_id"] isa String && try UUIDs.UUID(data["run_id"]);true catch;false end || _experiment_error("invalid stereo run UUID")
    data["recipe_id"]==record.recipe.recipe_id && data["input_id"]==record.input_id || _experiment_error("stereo run identities disagree")
    n=data["completed_pairs"]
    n isa Int && 0<=n<=length(record.pairs) && data["status"] in ("completed","failed") || _experiment_error("invalid stereo run count/status")
    data["status"]=="completed" && n!=length(record.pairs) && _experiment_error("incomplete completed stereo run")
    all(k->data[k] isa Float64 && isfinite(data[k]),("started_at","finished_at")) && data["finished_at"]>=data["started_at"] || _experiment_error("invalid stereo run times")
    _artifact_absolute_locator(data["output"]) || _experiment_error("invalid output locator")
    data["output_sha256"]===nothing || _experiment_hash(data["output_sha256"]) || _experiment_error("invalid output digest")
    (data["status"]=="completed" ? data["error"]===nothing : data["error"] isa String) || _experiment_error("invalid stereo failure summary")
    _experiment_validate_environment(data["environment"])
    ExperimentRun(data["run_id"],data["recipe_id"],data["input_id"],data["started_at"],data["finished_at"],Symbol(data["status"]),n,
        data["output"],data["output_sha256"],deepcopy(data["environment"]),data["error"])
end
function _stereo_record_protect(path,record;runs=record.runs,records=false,results=false)
    locators=String[f["path"] for f in record.input_files]
    records && append!(locators,record.record_paths)
    results && append!(locators,[r.output for r in runs])
    destination=_artifact_local_path(path)
    any(p->_artifact_alias(destination,p),_artifact_local_protected_paths(locators)) && _experiment_error("destination aliases a protected stereo input, record or result")
    nothing
end

"""
    save_experiment(path, record::StereoExperimentRecord; runs=record.runs)

Save the independent primitive version-1 stereo experiment schema. Planar/native
markers and registered layouts are unchanged; legacy readers refuse this artifact.
Known input/output/record aliases and changed snapshots are rejected before writing.
"""
function save_experiment(path::AbstractString,record::StereoExperimentRecord;runs=record.runs)
    _stereo_record_preflight(record)
    validated=[_stereo_run_validate(_experiment_run_data(r),record) for r in runs]
    _stereo_record_protect(path,record;runs=validated,results=true)
    payload=deepcopy(_stereo_record_core(record))
    payload["runs"]=[_experiment_run_data(r) for r in validated];payload["record_sha256"]=record._sha256
    _experiment_digest(payload)
    jldopen(_artifact_local_path(path),"w") do file
        file["stereo_experiment_format_version"]=STEREO_EXPERIMENT_FORMAT_VERSION
        file["stereo_experiment"]=payload
    end
    locator=realpath(path);locator in record.record_paths || push!(record.record_paths,locator)
    path
end

"""
    load_stereo_experiment(path) -> StereoExperimentRecord

Validate a primitive stereo experiment without reading external input images or
running calibration/processing. Unknown versions/schema, changed settings/input
identities and invalid supplied provenance are refused. External paths remain
locators; replay checks actual local bytes and environment separately.
"""
function load_stereo_experiment(path::AbstractString)
    localpath=_artifact_local_path(path)
    data=jldopen(localpath,"r") do file
        haskey(file,"stereo_experiment_format_version") && file["stereo_experiment_format_version"]===STEREO_EXPERIMENT_FORMAT_VERSION || _experiment_error("missing/unsupported stereo experiment version")
        haskey(file,"stereo_experiment") || _experiment_error("missing stereo experiment payload")
        file["stereo_experiment"]
    end
    _experiment_keys(data,["recipe","recipe_id","recipe_sha256","input_files","pairs","timing_metadata","scaling_mode","input_id","creation_environment","runs","record_sha256"],"stereo experiment")
    data["recipe"] isa Dict{String,Any} && _experiment_hash(data["recipe_id"]) && _experiment_hash(data["recipe_sha256"]) || _experiment_error("invalid stereo recipe encoding")
    recipe=StereoPIVRecipe(data["recipe"],data["recipe_id"],data["recipe_sha256"])
    data["input_files"] isa Vector{Dict{String,Any}} && data["pairs"] isa Vector{Vector{Int}} && data["timing_metadata"] isa Vector{Dict{String,Any}} || _experiment_error("invalid stereo input collections")
    data["scaling_mode"] in ("four_tuple","pair_list") && _experiment_hash(data["input_id"]) && _experiment_hash(data["record_sha256"]) || _experiment_error("invalid stereo input identity/mode")
    data["creation_environment"] isa Dict{String,Any} && data["runs"] isa AbstractVector || _experiment_error("invalid stereo environment/run collection")
    record=StereoExperimentRecord(recipe,data["input_files"],data["pairs"],data["timing_metadata"],Symbol(data["scaling_mode"]),data["input_id"],data["creation_environment"],ExperimentRun[],[realpath(localpath)],data["record_sha256"])
    _stereo_record_preflight(record)
    append!(record.runs,[_stereo_run_validate(r,record) for r in data["runs"]]);record
end

function _stereo_rebuild_maps(record,config;allow_environment_change=false)
    dewarpers=Tuple(ImageDewarper(config.cams[r],config.grid,Tuple(config.sizes[r])) for r in 1:2)
    actual=String[_stereo_map_digest(dw) for dw in dewarpers]
    allow_environment_change || actual==record.recipe._data["reference_map_sha256"] || _experiment_error("rebuilt dewarp maps differ from captured environment")
    dewarpers,actual
end
function _stereo_verify_input_bytes(record)
    for f in record.input_files
        path=_artifact_local_path(f["path"])
        isfile(path) && filesize(path)==f["size_bytes"] && _experiment_file_digest(path)==f["sha256"] || _experiment_error("stereo input changed or is missing")
    end
end
function _stereo_run_association(record,run_id,status,n,maps,hashes,record_diagnostics,record_pair_timing)
    Dict{String,Any}("run_id"=>run_id,"recipe_id"=>record.recipe.recipe_id,"input_id"=>record.input_id,
        "record_snapshot_sha256"=>record._sha256,"scaling_mode"=>String(record.scaling_mode),
        "status"=>String(status),"completed_pairs"=>n,"reference_map_sha256"=>record.recipe._data["reference_map_sha256"],
        "actual_map_sha256"=>maps,"entry_measurement_sha256"=>hashes[1:n],
        "diagnostics_recorded"=>record_diagnostics,"timing_recorded"=>record_pair_timing,
        "binding_basis"=>"stereo_and_camera_measurement_fields_excluding_parameters_and_planes")
end
function _stereo_write_run_association(path,data)
    jldopen(path,"r+") do file
        _check_results_format(file,path)
        haskey(file,"stereo_experiment_run") && _experiment_error("stereo run association already present")
        file["stereo_experiment_run_format_version"]=STEREO_EXPERIMENT_RUN_FORMAT_VERSION
        file["stereo_experiment_run"]=Dict{String,Any}("association"=>data,"association_sha256"=>_experiment_digest(data))
    end
end
function _stereo_finish_run(record,path,environment,started,run_id,status,n,error)
    digest=isfile(path) ? try _experiment_file_digest(path) catch;nothing end : nothing
    ExperimentRun(run_id,record.recipe.recipe_id,record.input_id,started,time(),status,n,path,digest,deepcopy(environment),error)
end

"""
    replay_experiment(record::StereoExperimentRecord; output, allow_environment_change=false,
        run_record=nothing, progress=nothing, on_diagnostics=nothing, record_diagnostics=false,
        on_pair_timing=nothing, record_pair_timing=false) -> ExperimentRun

Noncollecting stereo replay from frozen fitted cameras. Validate all metadata,
input bytes/dimensions, destinations and environment before output; check bytes
around each load and at completion. No fitting, scripts, settings overrides or
resume operation occurs. Companions keep their existing schemas; a separate
native sibling records run/recipe/input association and measurement hashes.
Progress runs after persistence and counts completed acquisitions, not cameras.
Failures join outstanding loading, preserve the completed prefix/original error,
and optionally save failed-run metadata. Two-file publication is not atomic.
An explicit environment override permits changed rebuilt maps, whose actual
digests and environment are recorded; it promises no numerical equivalence.
"""
function replay_experiment(record::StereoExperimentRecord;output::AbstractString,
    allow_environment_change::Bool=false,run_record::Union{Nothing,AbstractString}=nothing,
    progress::Union{Nothing,Function}=nothing,on_diagnostics::Union{Nothing,Function}=nothing,record_diagnostics::Bool=false,
    on_pair_timing::Union{Nothing,Function}=nothing,record_pair_timing::Bool=false)
    snapshot=deepcopy(record)
    config=_stereo_record_preflight(snapshot)
    for r in snapshot.runs
        _stereo_run_validate(_experiment_run_data(r),snapshot)
    end
    path=_artifact_local_path(output)
    _stereo_record_protect(path,snapshot;records=true)
    if run_record!==nothing
        _stereo_record_protect(run_record,snapshot;results=true)
        _artifact_alias(path,run_record) && _experiment_error("output and run record must differ")
    end
    if isfile(path)
        saved_record=try jldopen(f->haskey(f,"experiment_format_version") || haskey(f,"stereo_experiment_format_version"),path,"r") catch;false end
        saved_record && _experiment_error("result output must not replace a saved experiment")
    end
    environment=_experiment_software()
    allow_environment_change || _experiment_environment_signature(environment)==_experiment_environment_signature(snapshot.creation_environment) ||
        _experiment_error("software environment differs; explicitly allow_environment_change to rerun")
    # Metadata/alias/backend errors precede decoding any input pixels.
    _stereo_record_preflight(snapshot;verify_files=true)
    dw,maps=_stereo_rebuild_maps(snapshot,config;allow_environment_change)
    _experiment_environment_signature(_experiment_software())==_experiment_environment_signature(environment) || _experiment_error("actual software environment changed during preflight")
    frames=_stereo_record_frames(snapshot,config;load_pixels=true)
    preprocess=Tuple(_experiment_preprocess(PIVRecipe(config.processing.passes;preprocessing=steps,image_type=config.processing.image_type),nothing) for steps in config.pipelines)
    started=time();run_id=string(UUIDs.uuid4());completed=Ref(0);hashes=String[]
    result_delivery=(i,r)->begin
        i==length(hashes)+1 || _experiment_error("unexpected stereo result order")
        push!(hashes,_stereo_execution_measurement_digest(r));nothing
    end
    delivery=(i,n)->begin
        completed[]=i;progress===nothing || progress(i,n);nothing
    end
    settings=(;backend=config.processing.backend,image_type=config.processing.image_type,
        threaded=config.processing.threaded,predictor_smoothing=config.processing.predictor_smoothing,
        mask_threshold=config.processing.mask_threshold,uncertainty_backend=config.processing.uncertainty_backend,
        mask=config.mask,roi=config.processing.roi,scale=config.scale,preprocess,output=path,collect_results=false,
        progress=delivery,on_result=result_delivery,on_diagnostics,record_diagnostics,on_pair_timing,record_pair_timing,
        sync_atol=config.policy["atol"],sync_rtol=config.policy["rtol"],missing_timestamps=Symbol(config.policy["missing_timestamps"]),
        timing_atol=config.tolerance["atol"],timing_rtol=config.tolerance["rtol"])
    # Force callback-only timing validation even when no companion is requested:
    # this preserves the record's safely promoted exact/mixed admission semantics.
    timing_delivery=on_pair_timing===nothing ? (i,p)->nothing : on_pair_timing
    settings=merge(settings,(;on_pair_timing=timing_delivery))
    run=try
        if snapshot.scaling_mode===:pair_list
            run_piv_stereo_sequence(frames.camera_pairs...,dw...,config.processing.passes;settings...)
        else
            run_piv_stereo_sequence(frames.acquisitions,dw...,config.processing.passes;settings...)
        end
        _stereo_verify_input_bytes(snapshot)
        _experiment_environment_signature(_experiment_software())==_experiment_environment_signature(environment) || _experiment_error("actual software environment changed during stereo replay")
        association=_stereo_run_association(snapshot,run_id,:completed,completed[],maps,hashes,record_diagnostics,record_pair_timing)
        _stereo_write_run_association(path,association)
        _stereo_finish_run(snapshot,path,environment,started,run_id,:completed,completed[],nothing)
    catch err
        if isfile(path)
            try
                _stereo_write_run_association(path,_stereo_run_association(snapshot,run_id,:failed,completed[],maps,hashes,record_diagnostics,record_pair_timing))
            catch secondary
                @error "Failed to publish stereo failure association" exception=secondary
            end
        end
        failed=_stereo_finish_run(snapshot,path,environment,started,run_id,:failed,completed[],sprint(showerror,err))
        if run_record!==nothing
            try save_experiment(run_record,snapshot;runs=[snapshot.runs;failed]) catch secondary
                @error "Failed to save stereo failure run metadata" exception=secondary
            end
        end
        rethrow()
    end
    run_record===nothing || save_experiment(run_record,snapshot;runs=[snapshot.runs;run])
    run
end

function _stereo_association_validate(data,record,run)
    _experiment_keys(data,["run_id","recipe_id","input_id","record_snapshot_sha256","scaling_mode","status","completed_pairs",
        "reference_map_sha256","actual_map_sha256","entry_measurement_sha256","diagnostics_recorded","timing_recorded","binding_basis"],"stereo run association")
    for (k,v) in (("run_id",run.run_id),("recipe_id",record.recipe.recipe_id),("input_id",record.input_id),
        ("record_snapshot_sha256",record._sha256),("scaling_mode",String(record.scaling_mode)),("status",String(run.status)),("completed_pairs",run.completed_pairs),
        ("reference_map_sha256",record.recipe._data["reference_map_sha256"]))
        _experiment_digest(Dict{String,Any}("v"=>data[k]))==_experiment_digest(Dict{String,Any}("v"=>v)) || _experiment_error("stereo run association mismatch: $k")
    end
    for k in ("reference_map_sha256","actual_map_sha256")
        data[k] isa Vector{String} && length(data[k])==2 && all(_experiment_hash,data[k]) || _experiment_error("invalid dewarp map association")
    end
    data["entry_measurement_sha256"] isa Vector{String} && length(data["entry_measurement_sha256"])==run.completed_pairs && all(_experiment_hash,data["entry_measurement_sha256"]) || _experiment_error("invalid measurement association count")
    all(k->data[k] isa Bool,("diagnostics_recorded","timing_recorded")) &&
        data["binding_basis"]=="stereo_and_camera_measurement_fields_excluding_parameters_and_planes" || _experiment_error("invalid stereo measurement association contract")
    data
end
function _stereo_run_companion_markers(file,association)
    for (marker,group,requested) in (("stereo_execution_diagnostics_format_version","stereo_execution_diagnostics",association["diagnostics_recorded"]),
        ("stereo_pair_timing_format_version","stereo_pair_timing",association["timing_recorded"]))
        if haskey(file,marker)
            file[marker]===1 || _experiment_error("unsupported stereo run companion marker")
            requested || _experiment_error("unrequested stereo companion marker")
        else
            haskey(file,group) && _experiment_error("stereo companion group lacks its marker")
            requested && association["completed_pairs"]>0 && _experiment_error("requested stereo companion marker missing")
        end
    end
    any(k->haskey(file,k),("execution_diagnostics","execution_diagnostics_format_version","measurement_history","measurement_history_format_version","pair_timing","pair_timing_format_version",
        "ensemble_execution_diagnostics","ensemble_execution_diagnostics_format_version")) &&
        _experiment_error("stereo experiment output cannot contain crossed planar or ensemble companion metadata")
end
function _stereo_verify_result_fields(result,snapshot,config,dewarpers)
    result isa StereoPIVResult{config.processing.image_type} || _experiment_error("wrong stereo payload kind/precision")
    mask=dewarpers[1].mask .| dewarpers[2].mask
    config.mask===nothing || (mask .|= config.mask)
    roi=config.processing.roi
    selected=roi===nothing ? size(config.grid) : (length(roi.rows),length(roi.cols))
    roi===nothing || (mask=view(mask,roi.rows,roi.cols))
    final=last(config.processing.passes);T=config.processing.image_type
    grid=pass_grid(T,selected,final,mask,config.processing.mask_threshold)
    x=roi===nothing ? grid.x : grid.x .+ T(first(roi.cols)-1)
    y=roi===nothing ? grid.y : grid.y .+ T(first(roi.rows)-1)
    for camera in (result.cam1,result.cam2)
        camera.x==x && camera.y==y && camera.mask==grid.grid_mask && camera.scale===nothing &&
            _experiment_digest(_experiment_pass_data(camera.parameters))==_experiment_digest(_experiment_pass_data(final)) ||
            _experiment_error("camera result grid/mask/final settings disagree with frozen recipe")
    end
    result.mask==grid.grid_mask || _experiment_error("reconstructed mask disagrees with frozen recipe")
    _experiment_digest(_experiment_pass_data(result.parameters))==_experiment_digest(_experiment_pass_data(final)) ||
        _experiment_error("reconstructed final settings disagree with frozen recipe")
    _stereo_timing_bind(snapshot,result,config.grid)
end

"""
    verify_stereo_experiment_run(record, run; verify_results=false, verify_inputs=false,
        output=run.output)

Verify run/recipe/input association and the native artifact SHA without recomputing
PIV. `verify_results=true` streams one raw result at a time, checking precision,
expected grid/mask/last-pass settings, measurement-field hashes and any requested
execution/timing companion. `verify_inputs=true` additionally checks current local
input byte identities. Return detached scalar inspection status; retain no payload.
A failed run may contain one trailing, incompletely published native entry. Only
its completed prefix is inspected; `measurement_fields_checked_acquisitions` and
`unverified_trailing_entries` report these scopes explicitly. Completed runs may
not contain trailing entries.
Metadata/output integrity is not source authenticity, calibration accuracy or
proof of numerically rerunning the recipe. Recorded run environment is historical;
verification does not claim execution in the current reader environment.
"""
function verify_stereo_experiment_run(record::StereoExperimentRecord,run::ExperimentRun;
    verify_results::Bool=false,verify_inputs::Bool=false,output::AbstractString=run.output)
    snapshot=deepcopy(record);config=_stereo_record_preflight(snapshot)
    checked=_stereo_run_validate(_experiment_run_data(run),snapshot)
    verify_inputs && _stereo_verify_input_bytes(snapshot)
    path=_artifact_local_path(output)
    checked.output_sha256!==nothing && isfile(path) && _experiment_file_digest(path)==checked.output_sha256 || _experiment_error("stereo run output changed or is missing")
    index=ResultFile(path)
    count_ok=checked.status===:completed ? length(index)==checked.completed_pairs :
        checked.completed_pairs<=length(index)<=checked.completed_pairs+1
    count_ok || _experiment_error("native stereo completed prefix count disagrees")
    association=jldopen(path,"r") do file
        haskey(file,"stereo_experiment_run_format_version") && file["stereo_experiment_run_format_version"]===STEREO_EXPERIMENT_RUN_FORMAT_VERSION || _experiment_error("missing/unsupported stereo run association marker")
        haskey(file,"stereo_experiment_run") || _experiment_error("missing stereo run association")
        saved=file["stereo_experiment_run"]
        _experiment_keys(saved,["association","association_sha256"],"stereo run association storage")
        _experiment_hash(saved["association_sha256"]) && _experiment_digest(saved["association"])==saved["association_sha256"] || _experiment_error("stereo run association digest mismatch")
        data=_stereo_association_validate(saved["association"],snapshot,checked)
        _stereo_run_companion_markers(file,data)
        deepcopy(data)
    end
    if verify_results
        dewarpers,_=_stereo_rebuild_maps(snapshot,config;allow_environment_change=true)
        currentmaps=String[_stereo_map_digest(dw) for dw in dewarpers]
        currentmaps==association["actual_map_sha256"] || _experiment_error("current decoder's rebuilt maps differ; raw recipe geometry cannot be verified in this environment")
        timing=_stereo_record_timing(snapshot,config)
        for i in 1:checked.completed_pairs
            raw=index[i]
            labels=jldopen(path,"r") do file
                haskey(file,source_key(i)) || _experiment_error("ordered stereo source labels missing")
                file[source_key(i)]
            end
            expected_labels=String[snapshot.input_files[j]["path"] for j in snapshot.pairs[i]]
            labels isa Vector{String} && labels==expected_labels || _experiment_error("native ordered camera/source labels disagree with record")
            expected=_stereo_verify_result_fields(raw,timing[i],config,dewarpers)
            _stereo_execution_measurement_digest(raw)==association["entry_measurement_sha256"][i] || _experiment_error("stereo measurement hash differs from recorded run")
            d=load_stereo_execution_diagnostics(index,i)
            if association["diagnostics_recorded"]
                d===nothing && _experiment_error("requested stereo execution companion missing")
                d.pair_index===i || _experiment_error("diagnostic acquisition association differs")
                execution_diagnostics_data(d;result=raw)
            else
                d===nothing || _experiment_error("unexpected stereo execution companion")
            end
            p=load_stereo_pair_timing(index,i)
            if association["timing_recorded"]
                p===nothing && _experiment_error("requested stereo timing companion missing")
                pair_timing_data(p;result=raw)
                _experiment_digest(p._data)==_experiment_digest(expected._data) || _experiment_error("timing packet source/acquisition context disagrees with record")
            else
                p===nothing || _experiment_error("unexpected stereo timing companion")
            end
            raw=nothing;expected=nothing;d=nothing;p=nothing
        end
    end
    _check_result_file(index)
    _experiment_file_digest(path)==checked.output_sha256 || _experiment_error("stereo output changed during verification")
    verify_inputs && _stereo_verify_input_bytes(snapshot)
    Dict{String,Any}("run_id"=>checked.run_id,"recipe_id"=>snapshot.recipe.recipe_id,"input_id"=>snapshot.input_id,
        "completed_acquisitions"=>checked.completed_pairs,"output_integrity_checked"=>true,"run_association_checked"=>true,
        "unverified_trailing_entries"=>length(index)-checked.completed_pairs,
        "measurement_fields_checked"=>verify_results,"current_input_bytes_checked"=>verify_inputs,
        "measurement_fields_checked_acquisitions"=>verify_results ? checked.completed_pairs : 0,
        "calibration_accuracy_verified"=>false,"source_authenticity_verified"=>false,"piv_recomputed"=>false)
end
