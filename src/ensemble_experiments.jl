import JLD2: Group

const ENSEMBLE_EXPERIMENT_FORMAT_VERSION = 1
const ENSEMBLE_EXPERIMENT_RUN_FORMAT_VERSION = 1

"""
    EnsemblePIVRecipe(passes; preprocessing=PreprocessStep[], mask=nothing,
        scale=nothing, backend=:cpu, image_type=Float64, threaded=false,
        predictor_smoothing=true, mask_threshold=.5)

Snapshot a complete builtin planar-ensemble schedule. CPU/KA Float32/64,
builtin preprocessing/backgrounds, static full-image masks and scale are
supported. ROI, scripts, custom validators, vendor devices and split uncertainty
backends are refused. Requested iteration/tolerance settings are retained,
including infinite tolerance, although ensemble executes one sweep per pass.
Inputs are detached; mutation is detected by [`recipe_identity`](@ref).
"""
struct EnsemblePIVRecipe
    passes::Vector{PIVParameters}
    preprocessing::Vector{PreprocessStep}
    mask::Union{Nothing,BitMatrix}
    scale::Union{Nothing,PhysicalScale}
    backend::Symbol
    image_type::DataType
    threaded::Bool
    predictor_smoothing::Bool
    mask_threshold::Float64
    recipe_id::String
end

function _ensemble_experiment_pass(data)
    _experiment_keys(data,String.(fieldnames(PIVParameters)),"ensemble pass")
    tolerance=data["convergence_tol"]
    tolerance isa Float64 && !isnan(tolerance) && tolerance>=0 || _experiment_error("invalid ignored ensemble tolerance")
    ordinary=copy(data);ordinary["convergence_tol"]=isinf(tolerance) ? 0. : tolerance
    p=_experiment_pass(ordinary)
    args=NamedTuple{fieldnames(PIVParameters)}(Tuple(k===:convergence_tol ? tolerance : getfield(p,k) for k in fieldnames(PIVParameters)))
    result=PIVParameters(;args...)
    _experiment_digest(_experiment_pass_data(result))==_experiment_digest(data) || _experiment_error("noncanonical ensemble pass settings")
    result
end
function _ensemble_recipe_data(recipe::EnsemblePIVRecipe)
    Dict{String,Any}("kind"=>"planar_ensemble_piv","passes"=>[_experiment_pass_data(p) for p in recipe.passes],
        "preprocessing"=>[Dict{String,Any}("operation"=>String(s.operation),"options"=>deepcopy(s.options)) for s in recipe.preprocessing],
        "mask"=>recipe.mask,"scale"=>recipe.scale===nothing ? nothing : Dict{String,Any}(String(k)=>getfield(recipe.scale,k) for k in fieldnames(PhysicalScale)),
        "backend"=>String(recipe.backend),"image_type"=>string(recipe.image_type),"threaded"=>recipe.threaded,
        "predictor_smoothing"=>recipe.predictor_smoothing,"mask_threshold"=>recipe.mask_threshold)
end
function EnsemblePIVRecipe(passes::Union{PIVParameters,AbstractVector{PIVParameters}};
        preprocessing::AbstractVector{PreprocessStep}=PreprocessStep[],mask=nothing,
        scale::Union{Nothing,PhysicalScale}=nothing,backend::Symbol=:cpu,image_type::DataType=Float64,
        threaded::Bool=false,predictor_smoothing::Bool=true,mask_threshold::Real=.5,
        roi=nothing,external_preprocess=nothing,uncertainty_backend::Symbol=:same)
    roi===nothing || _experiment_error("ensemble experiment ROI is unsupported")
    external_preprocess===nothing || _experiment_error("ensemble experiment scripts/custom preprocessing are unsupported")
    uncertainty_backend===:same || _experiment_error("ensemble experiment split uncertainty backend is unsupported")
    backend in (:cpu,:ka) || _experiment_error("ensemble recipes support CPU/KA only")
    image_type in (Float32,Float64) || _experiment_error("ensemble recipe precision must be Float32/64")
    mask===nothing || mask isa AbstractMatrix{Bool} || _experiment_error("ensemble mask must be a static full-image Bool matrix")
    isfinite(mask_threshold) && 0<mask_threshold<=1 || _experiment_error("invalid ensemble mask threshold")
    schedule=passes isa PIVParameters ? [passes] : collect(passes)
    isempty(schedule) && _experiment_error("ensemble recipe needs at least one explicit pass")
    schedule=[_ensemble_experiment_pass(_experiment_pass_data(p)) for p in schedule]
    _check_backend_params(_resolve_backend(backend),schedule)
    steps=[PreprocessStep(s.operation;(Symbol(k)=>v for (k,v) in deepcopy(s.options))...) for s in preprocessing]
    args=(schedule,steps,mask===nothing ? nothing : BitMatrix(mask),scale,backend,image_type,threaded,predictor_smoothing,Float64(mask_threshold))
    provisional=EnsemblePIVRecipe(args...,"")
    EnsemblePIVRecipe(args...,_experiment_digest(_ensemble_recipe_data(provisional)))
end
function _ensemble_recipe_decode(data)
    _experiment_keys(data,["kind","passes","preprocessing","mask","scale","backend","image_type","threaded","predictor_smoothing","mask_threshold"],"ensemble recipe")
    data["kind"]=="planar_ensemble_piv" && data["passes"] isa AbstractVector && data["preprocessing"] isa AbstractVector || _experiment_error("invalid ensemble recipe kind/schedule")
    steps=PreprocessStep[]
    for s in data["preprocessing"]
        _experiment_keys(s,["operation","options"],"ensemble preprocessing step")
        s["operation"] isa String && s["options"] isa AbstractDict || _experiment_error("invalid preprocessing descriptor")
        step=PreprocessStep(Symbol(s["operation"]);(Symbol(k)=>v for (k,v) in s["options"])...)
        _experiment_digest(Dict{String,Any}("operation"=>String(step.operation),"options"=>step.options))==_experiment_digest(s) || _experiment_error("noncanonical preprocessing descriptor")
        push!(steps,step)
    end
    data["mask"]===nothing || data["mask"] isa BitMatrix || _experiment_error("invalid static ensemble mask")
    scale=data["scale"]
    if scale!==nothing
        _experiment_keys(scale,String.(fieldnames(PhysicalScale)),"ensemble scale")
        all(k->scale[k] isa Float64 && isfinite(scale[k]) && scale[k]>0,("pixel_size","dt")) &&
            all(k->scale[k] isa String,("length_unit","time_unit")) || _experiment_error("invalid ensemble scale metadata")
        scale=PhysicalScale(scale["pixel_size"],scale["dt"],scale["length_unit"],scale["time_unit"])
    end
    all(k->data[k] isa Bool,("threaded","predictor_smoothing")) && data["mask_threshold"] isa Float64 || _experiment_error("invalid ensemble driver settings")
    data["backend"] in ("cpu","ka") && data["image_type"] in ("Float32","Float64") || _experiment_error("unsupported ensemble backend/precision")
    recipe=EnsemblePIVRecipe([_ensemble_experiment_pass(p) for p in data["passes"]];preprocessing=steps,mask=data["mask"],scale,
        backend=Symbol(data["backend"]),image_type=data["image_type"]=="Float32" ? Float32 : Float64,
        threaded=data["threaded"],predictor_smoothing=data["predictor_smoothing"],mask_threshold=data["mask_threshold"])
    _experiment_digest(_ensemble_recipe_data(recipe))==_experiment_digest(data) || _experiment_error("noncanonical ensemble recipe")
    recipe
end
function recipe_identity(recipe::EnsemblePIVRecipe)
    _experiment_hash(recipe.recipe_id) && _experiment_digest(_ensemble_recipe_data(recipe))==recipe.recipe_id || _experiment_error("ensemble recipe changed after snapshot")
    _ensemble_recipe_decode(_ensemble_recipe_data(recipe));recipe.recipe_id
end

"""
    EnsembleExperimentRun

Array-free execution metadata, separate from sequence runs. `input_pairs` and
`scheduled_passes` determine `total_contributions`; `completed_contributions`
counts joined pair work across passes, including masked/gated opportunities.
`completed_pools` and `published_results` are zero until publication, then one.
Statuses are `:completed`, `:cancelled`, or `:failed`. Only completed runs bind
an output/measurement digest. `record_diagnostics` records the requested policy.
This is not a checkpoint or an independent-sample/convergence claim.
"""
struct EnsembleExperimentRun
    run_id::String
    recipe_id::String
    input_id::String
    started_at::Float64
    finished_at::Float64
    status::Symbol
    input_pairs::Int
    scheduled_passes::Int
    total_contributions::Int
    completed_contributions::Int
    completed_pools::Int
    published_results::Int
    output::String
    output_sha256::Union{Nothing,String}
    measurement_sha256::Union{Nothing,String}
    record_diagnostics::Bool
    environment::Dict{String,Any}
    error::Union{Nothing,String}
end

"""
    EnsembleExperimentRecord(file_pairs, recipe::EnsemblePIVRecipe)

Capture ordered two-file pairs, byte hashes/dimensions, complete pooled settings
and creation environment. Paths are locators, not scientific input identity.
Every image has the same raw dimensions. Frames/timestamps and independent
sample assumptions are not inferred. Builtin settings/backgrounds/mask are
embedded; source pixels are not. Requests are detached before replay callbacks.
"""
struct EnsembleExperimentRecord
    recipe::EnsemblePIVRecipe
    input_files::Vector{Dict{String,Any}}
    pairs::Vector{Vector{Int}}
    input_id::String
    creation_environment::Dict{String,Any}
    runs::Vector{EnsembleExperimentRun}
    record_paths::Vector{String}
    _sha256::String
end
_ensemble_record_core(record)=Dict{String,Any}("recipe"=>_ensemble_recipe_data(record.recipe),"recipe_id"=>record.recipe.recipe_id,
    "input_files"=>record.input_files,"pairs"=>record.pairs,"input_id"=>record.input_id,"creation_environment"=>record.creation_environment)
_ensemble_run_data(run::EnsembleExperimentRun)=Dict{String,Any}(String(k)=>getfield(run,k) isa Symbol ? String(getfield(run,k)) : getfield(run,k) for k in fieldnames(EnsembleExperimentRun))
function _ensemble_record_preflight(record;verify_files=false)
    recipe_identity(record.recipe)
    _experiment_hash(record._sha256) && _experiment_digest(_ensemble_record_core(record))==record._sha256 || _experiment_error("ensemble record snapshot changed")
    recipe=record.recipe
    isempty(record.input_files) && _experiment_error("ensemble record has no files")
    isempty(record.pairs) && _experiment_error("ensemble record has no pairs")
    expected=nothing
    for file in record.input_files
        _experiment_keys(file,["path","sha256","size_bytes","image_size"],"ensemble input file")
        _artifact_absolute_locator(file["path"]) && _experiment_hash(file["sha256"]) && file["size_bytes"] isa Int && file["size_bytes"]>=0 || _experiment_error("invalid ensemble input identity")
        dims=file["image_size"]
        dims isa Vector{Int} && length(dims)==2 && all(>(0),dims) || _experiment_error("invalid raw input dimensions")
        if expected===nothing
            expected=dims
        else
            dims==expected || _experiment_error("every ensemble input must have the same dimensions")
        end
        recipe.mask===nothing || size(recipe.mask)==Tuple(dims) || _experiment_error("ensemble mask differs from full-image dimensions")
        all(p->all(collect(p.search_area_size).<=dims),recipe.passes) || _experiment_error("ensemble pass exceeds full-image dimensions")
        for step in recipe.preprocessing
            step.operation===:subtract_background && size(step.options["background"])!=Tuple(dims) && _experiment_error("ensemble background differs from raw image dimensions")
        end
        verify_files && _experiment_load_image(merge(file,Dict("path"=>_artifact_local_path(file["path"]))),recipe.image_type)
    end
    for pair in record.pairs
        pair isa Vector{Int} && length(pair)==2 && all(i->1<=i<=length(record.input_files),pair) || _experiment_error("invalid ordered ensemble pair")
    end
    sort!(unique!(vcat(record.pairs...)))==collect(1:length(record.input_files)) || _experiment_error("unreferenced ensemble input file")
    _experiment_hash(record.input_id) && _experiment_digest(_experiment_input_data(record.input_files,record.pairs))==record.input_id || _experiment_error("ensemble input identity changed")
    _ensemble_mul(length(record.pairs),length(recipe.passes))
    _experiment_validate_environment(record.creation_environment)
    record
end
function EnsembleExperimentRecord(file_pairs::AbstractVector,recipe::EnsemblePIVRecipe)
    recipe_identity(recipe);isempty(file_pairs) && _experiment_error("ensemble record requires file pairs")
    recipe=deepcopy(recipe)
    selected=Tuple{String,String}[]
    for pair in file_pairs
        pair isa Tuple && length(pair)==2 && all(p->p isa AbstractString,pair) || _experiment_error("ensemble inputs must be tuples of two file paths")
        push!(selected,(String(pair[1]),String(pair[2])))
    end
    files=Dict{String,Any}[];pairs=Vector{Int}[]
    for pair in selected
        ids=Int[]
        for name in pair
            path=realpath(_artifact_local_path(name));index=findfirst(f->f["path"]==path,files)
            if index===nothing
                file=Dict{String,Any}("path"=>path,"sha256"=>_experiment_file_digest(path),"size_bytes"=>filesize(path),"image_size"=>collect(size(load_image(recipe.image_type,path))))
                _experiment_load_image(file,recipe.image_type);push!(files,file);index=length(files)
            end
            push!(ids,index)
        end
        push!(pairs,ids)
    end
    args=(deepcopy(recipe),files,pairs,_experiment_digest(_experiment_input_data(files,pairs)),_experiment_software(),EnsembleExperimentRun[],String[])
    provisional=EnsembleExperimentRecord(args...,"")
    record=EnsembleExperimentRecord(args...,_experiment_digest(_ensemble_record_core(provisional)))
    _ensemble_record_preflight(record;verify_files=true);record
end
function _ensemble_run_validate(data,record)
    _experiment_keys(data,String.(fieldnames(EnsembleExperimentRun)),"ensemble run")
    data["run_id"] isa String && try UUIDs.UUID(data["run_id"]);true catch;false end || _experiment_error("invalid ensemble run UUID")
    data["recipe_id"]==record.recipe.recipe_id && data["input_id"]==record.input_id || _experiment_error("ensemble run identities disagree")
    data["status"] in ("completed","cancelled","failed") || _experiment_error("unsupported ensemble run status")
    all(k->data[k] isa Int && data[k]>=0,("input_pairs","scheduled_passes","total_contributions","completed_contributions","completed_pools","published_results")) || _experiment_error("invalid ensemble run counts")
    n,s=length(record.pairs),length(record.recipe.passes)
    data["input_pairs"]==n && data["scheduled_passes"]==s && data["total_contributions"]==_ensemble_mul(n,s) &&
        data["completed_contributions"]<=data["total_contributions"] || _experiment_error("ensemble run contribution counts disagree")
    completed=data["status"]=="completed"
    data["completed_pools"]==data["published_results"]==Int(completed) &&
        (!completed || data["completed_contributions"]==data["total_contributions"]) || _experiment_error("ensemble run publication/counts disagree")
    all(k->data[k] isa Float64 && isfinite(data[k]),("started_at","finished_at")) && data["finished_at"]>=data["started_at"] || _experiment_error("invalid ensemble run times")
    _artifact_absolute_locator(data["output"]) && data["record_diagnostics"] isa Bool || _experiment_error("invalid ensemble output/policy")
    if completed
        _experiment_hash(data["output_sha256"]) && _experiment_hash(data["measurement_sha256"]) && data["error"]===nothing || _experiment_error("completed ensemble run lacks binding")
    else
        data["output_sha256"]===data["measurement_sha256"]===nothing && data["error"] isa String || _experiment_error("unpublished ensemble run must not bind an existing destination")
    end
    _experiment_validate_environment(data["environment"])
    EnsembleExperimentRun((k===:status ? Symbol(data[String(k)]) : deepcopy(data[String(k)]) for k in fieldnames(EnsembleExperimentRun))...)
end
function _ensemble_record_protect(path,record;runs=record.runs,records=false,results=false)
    locators=String[f["path"] for f in record.input_files]
    records && append!(locators,record.record_paths)
    results && append!(locators,[r.output for r in runs])
    destination=_artifact_local_path(path)
    any(p->_artifact_alias(destination,p),_artifact_local_protected_paths(locators)) && _experiment_error("destination aliases protected ensemble input, record or result")
    destination
end

"""
    save_experiment(path, record::EnsembleExperimentRecord; runs=record.runs)

Save the independent primitive ensemble record version 1. Existing sequence,
stereo and native-result schemas are unchanged. Unknown settings, changed
snapshots and input/result aliases reject before writing. Saved locations are
remembered to protect them from subsequent output replacement.
"""
function save_experiment(path::AbstractString,record::EnsembleExperimentRecord;runs=record.runs)
    _ensemble_record_preflight(record)
    validated=[_ensemble_run_validate(_ensemble_run_data(run),record) for run in runs]
    localpath=_ensemble_record_protect(path,record;runs=validated,results=true)
    payload=deepcopy(_ensemble_record_core(record));payload["runs"]=[_ensemble_run_data(run) for run in validated];payload["record_sha256"]=record._sha256
    _experiment_digest(payload)
    jldopen(localpath,"w") do file
        file["ensemble_experiment_format_version"]=ENSEMBLE_EXPERIMENT_FORMAT_VERSION
        file["ensemble_experiment"]=payload
    end
    locator=realpath(localpath);locator in record.record_paths || push!(record.record_paths,locator);path
end

"""
    load_ensemble_experiment(path) -> EnsembleExperimentRecord

Validate the dedicated primitive record/settings/input identities and saved
runs without loading external images or activating its recorded environment.
Foreign locators remain provenance and cannot silently become local paths.
Replay separately verifies local input bytes and environment compatibility.
"""
function load_ensemble_experiment(path::AbstractString)
    localpath=_artifact_local_path(path)
    data=jldopen(localpath,"r") do file
        haskey(file,"ensemble_experiment_format_version") && file["ensemble_experiment_format_version"]===1 || _experiment_error("missing/unsupported ensemble experiment version")
        haskey(file,"ensemble_experiment") || _experiment_error("missing ensemble experiment payload")
        file["ensemble_experiment"]
    end
    _experiment_keys(data,["recipe","recipe_id","input_files","pairs","input_id","creation_environment","runs","record_sha256"],"ensemble experiment")
    recipe=_ensemble_recipe_decode(data["recipe"]);recipe.recipe_id==data["recipe_id"] || _experiment_error("saved ensemble recipe identity mismatch")
    _experiment_hash(data["input_id"]) && _experiment_hash(data["record_sha256"]) || _experiment_error("invalid saved ensemble identities")
    data["input_files"] isa Vector{Dict{String,Any}} && data["pairs"] isa Vector{Vector{Int}} && data["creation_environment"] isa Dict{String,Any} && data["runs"] isa AbstractVector || _experiment_error("invalid ensemble record collections")
    record=EnsembleExperimentRecord(recipe,data["input_files"],data["pairs"],data["input_id"],data["creation_environment"],EnsembleExperimentRun[],[realpath(localpath)],data["record_sha256"])
    _ensemble_record_preflight(record)
    append!(record.runs,[_ensemble_run_validate(run,record) for run in data["runs"]]);record
end

"""
    EnsembleRunRecordError

Requested run-history publication failed after an otherwise completed or
cancelled replay. `run` preserves the accurate terminal counts/status and
`cause` is the history-writing error. Completed pooled output remains completed;
output and history are not an atomic transaction. Processing errors before
publication retain their original exception instead of this wrapper.
"""
struct EnsembleRunRecordError <: Exception
    run::EnsembleExperimentRun
    cause::Any
end
function Base.showerror(io::IO,error::EnsembleRunRecordError)
    print(io,error.run.status===:completed ? "pooled output completed, but requested run-history save failed: " :
        "ensemble replay cancelled, but requested run-history save failed: ")
    showerror(io,error.cause)
end
function _ensemble_verify_input_bytes(record)
    for file in record.input_files
        path=_artifact_local_path(file["path"])
        isfile(path) && filesize(path)==file["size_bytes"] && _experiment_file_digest(path)==file["sha256"] ||
            _experiment_error("ensemble input changed or is missing: $path")
    end
    nothing
end
function _ensemble_replay_destinations(path,run_record,record)
    _ensemble_record_protect(path,record;records=true)
    isdir(path) && _experiment_error("ensemble output must be a file destination")
    if isfile(path)
        saved=try jldopen(path,"r") do file
            any(k->haskey(file,k),("experiment_format_version","stereo_experiment_format_version","ensemble_experiment_format_version"))
        end catch;false end
        saved && _experiment_error("ensemble output must not replace a saved experiment")
    end
    if run_record!==nothing
        _ensemble_record_protect(run_record,record;results=true)
        _artifact_alias(path,run_record) && _experiment_error("ensemble output and run record must differ")
    end
    nothing
end
function _ensemble_output_association(record,run_id,measurement,record_diagnostics,environment)
    n,s=length(record.pairs),length(record.recipe.passes);total=_ensemble_mul(n,s)
    Dict{String,Any}("run_id"=>run_id,"recipe_id"=>record.recipe.recipe_id,"input_id"=>record.input_id,
        "record_snapshot_sha256"=>record._sha256,"run_environment_id"=>_experiment_digest(_experiment_environment_signature(environment)),
        "status"=>"completed","input_pairs"=>n,"scheduled_passes"=>s,
        "total_contributions"=>total,"completed_contributions"=>total,"completed_pools"=>1,"published_results"=>1,
        "result_key"=>result_key(1),"measurement_sha256"=>measurement,"record_diagnostics"=>record_diagnostics,
        "binding_basis"=>"raw_planar_measurement_fields_excluding_parameters_and_planes")
end
function _ensemble_native_replace(staging,final)
    dirname(abspath(staging))==dirname(abspath(final)) || _experiment_error("ensemble publication requires same-directory staging")
    # No copy/delete fallback or prior explicit destination removal. Native
    # replacement still depends on filesystem/platform behavior; no durability
    # or concurrent-writer guarantee is made.
    code=ccall(:jl_fs_rename,Int32,(Cstring,Cstring),staging,final)
    code<0 && Base.uv_error("ensemble result publication",code)
    final
end
function _ensemble_terminal_run(record,path,environment,started,run_id,status,completed,policy;
        output_sha256=nothing,measurement_sha256=nothing,error=nothing)
    n,s=length(record.pairs),length(record.recipe.passes);published=Int(status===:completed)
    run=EnsembleExperimentRun(run_id,record.recipe.recipe_id,record.input_id,started,max(started,time()),status,
        n,s,_ensemble_mul(n,s),completed,published,published,path,output_sha256,measurement_sha256,policy,deepcopy(environment),error)
    _ensemble_run_validate(_ensemble_run_data(run),record)
end
function _ensemble_persist_terminal(run_record,record,run)
    run_record===nothing && return run
    try
        save_experiment(run_record,record;runs=[record.runs;run])
    catch cause
        throw(EnsembleRunRecordError(run,cause))
    end
    run
end

"""
    replay_experiment(record::EnsembleExperimentRecord; output, run_record=nothing,
        progress=nothing, cancel_requested=nothing, allow_environment_change=false,
        on_diagnostics=nothing, record_diagnostics=false) -> EnsembleExperimentRun

Replay one complete pool without collecting per-pair results. Preflight checks
all metadata, bytes, backend/environment and known aliases before processing.
`progress(event)` runs after each pair's threaded contribution work joins;
the immutable event contains pass/pair indices and cumulative/total contribution
counts, input-pair/pass counts, and zero completed pools/published results.
The last contribution precedes peak analysis/publication, not convergence.

`cancel_requested()` must return Bool. It is checked before pixel loads,
after contribution callbacks and before publication. Cancellation returns a
`:cancelled` run with no output binding. Ordinary processing/callback failures
optionally save failed metadata and rethrow the original error. Source selection,
settings and metadata are detached before callbacks; bytes/dimensions are checked
around loads and bytes/environment again before publication. No timestamps,
stationarity, independent samples or uncertainty applicability are inferred.

A complete native result, separate run association and optional unchanged
ensemble diagnostics are prepared in a sibling temporary file. Computation or
cancellation does not replace the prior destination. Final native replacement
has no copy/delete fallback; filesystem failures, concurrent mutation, process
kill and durability are not covered by a transaction guarantee. Only completed
publication sets `completed_pools=published_results=1`. There is no user callback
after publication. Output and optional run history are published separately;
[`EnsembleRunRecordError`](@ref) carries an accurate completed/cancelled run if
the requested history write fails. Recorded environments are never activated.
"""
function replay_experiment(record::EnsembleExperimentRecord;output::AbstractString,
        run_record::Union{Nothing,AbstractString}=nothing,progress::Union{Nothing,Function}=nothing,
        cancel_requested::Union{Nothing,Function}=nothing,allow_environment_change::Bool=false,
        on_diagnostics::Union{Nothing,Function}=nothing,record_diagnostics::Bool=false)
    snapshot=deepcopy(record);_ensemble_record_preflight(snapshot)
    foreach(run->_ensemble_run_validate(_ensemble_run_data(run),snapshot),snapshot.runs)
    path=_artifact_local_path(output);record_path=run_record===nothing ? nothing : _artifact_local_path(run_record)
    _ensemble_replay_destinations(path,record_path,snapshot)
    environment=_experiment_software()
    allow_environment_change || _experiment_environment_signature(environment)==_experiment_environment_signature(snapshot.creation_environment) ||
        _experiment_error("software environment differs from ensemble creation; explicitly allow_environment_change to rerun")
    _ensemble_verify_input_bytes(snapshot)
    recipe=snapshot.recipe;started=time();run_id=string(UUIDs.uuid4());completed=Ref(0);packet=Ref{Any}(nothing)
    # FrameRef carries frozen selection; the loader refuses changed bytes/size.
    source=FrameSource(length(snapshot.input_files),i->_experiment_load_image(snapshot.input_files[i],recipe.image_type))
    pairs=[(FrameRef(source,p[1]),FrameRef(source,p[2])) for p in snapshot.pairs]
    preprocess=_experiment_preprocess(recipe,nothing)
    delivery=event->begin
        event.completed_contributions==completed[]+1 || _experiment_error("unexpected ensemble contribution order")
        completed[]=event.completed_contributions;progress===nothing || progress(event);nothing
    end
    diagnostics_delivery=record_diagnostics || on_diagnostics!==nothing ? d->begin
        packet[]=d;on_diagnostics===nothing || on_diagnostics(d);nothing
    end : nothing
    staging=nothing
    run=try
        _ensemble_cancel_check(cancel_requested)
        raw=run_piv_ensemble(pairs,recipe.passes;preprocess,backend=recipe.backend,image_type=recipe.image_type,
            threaded=recipe.threaded,predictor_smoothing=recipe.predictor_smoothing,mask=recipe.mask,mask_threshold=recipe.mask_threshold,
            scale=recipe.scale,progress=false,on_diagnostics=diagnostics_delivery,
            _on_contribution=delivery,_cancel_requested=cancel_requested)
        completed[]==_ensemble_mul(length(pairs),length(recipe.passes)) || _experiment_error("incomplete ensemble contribution work")
        _ensemble_verify_raw(raw,snapshot)
        packet[]===nothing || execution_diagnostics_data(packet[];result=raw)
        measurement=_history_result_digest(raw)
        association=_ensemble_output_association(snapshot,run_id,measurement,record_diagnostics,environment)
        _ensemble_replay_destinations(path,record_path,snapshot)
        _ensemble_cancel_check(cancel_requested)
        staging=joinpath(dirname(path),".$(basename(path)).$(UUIDs.uuid4()).partial")
        jldopen(staging,"w") do file
            file["format_version"]=RESULTS_FORMAT_VERSION;file[result_key(1)]=raw
            file["ensemble_sources"]=String[String(snapshot.input_files[i]["path"]) for pair in snapshot.pairs for i in pair]
            file["ensemble_experiment_run_format_version"]=ENSEMBLE_EXPERIMENT_RUN_FORMAT_VERSION
            file["ensemble_experiment_run"]=Dict{String,Any}("association"=>association,"association_sha256"=>_experiment_digest(association))
            record_diagnostics && _write_ensemble_execution_diagnostics(file,result_key(1),packet[],raw)
        end
        prepared_hash=_experiment_file_digest(staging)
        # Rechecks include callback side effects and staging time, before replacing
        # any destination. No callback runs after the native publication succeeds.
        _ensemble_cancel_check(cancel_requested)
        _ensemble_verify_input_bytes(snapshot)
        _experiment_environment_signature(_experiment_software())==_experiment_environment_signature(environment) || _experiment_error("actual software environment changed during ensemble replay")
        _ensemble_replay_destinations(path,record_path,snapshot)
        hash=_experiment_file_digest(staging)
        hash==prepared_hash || _experiment_error("prepared ensemble output changed before publication")
        terminal=_ensemble_terminal_run(snapshot,path,environment,started,run_id,:completed,completed[],record_diagnostics;
            output_sha256=hash,measurement_sha256=measurement)
        _ensemble_native_replace(staging,path);staging=nothing
        terminal
    catch error
        if error isa _EnsembleCancelled
            _ensemble_terminal_run(snapshot,path,environment,started,run_id,:cancelled,completed[],record_diagnostics;error=sprint(showerror,error))
        else
            failed=_ensemble_terminal_run(snapshot,path,environment,started,run_id,:failed,completed[],record_diagnostics;error=sprint(showerror,error))
            if record_path!==nothing
                try save_experiment(record_path,snapshot;runs=[snapshot.runs;failed]) catch secondary
                    @error "Failed to save ensemble failure history" exception=secondary
                end
            end
            rethrow()
        end
    finally
        if staging!==nothing
            try
                isfile(staging) && rm(staging;force=true)
            catch secondary
                @error "Failed to remove ensemble staging file" exception=secondary
            end
        end
    end
    _ensemble_persist_terminal(record_path,snapshot,run)
end

function _ensemble_verify_raw(raw,record)
    recipe=record.recipe;T=recipe.image_type
    raw isa PIVResult{T} || _experiment_error("ensemble output has wrong kind/precision")
    final=last(recipe.passes);dims=Tuple(first(record.input_files)["image_size"])
    grid=pass_grid(T,dims,final,recipe.mask,recipe.mask_threshold)
    expected=(length(grid.y),length(grid.x))
    all(a->size(a)==expected,(raw.u,raw.v,raw.peak_ratio,raw.correlation_moment,raw.uncertainty_u,raw.uncertainty_v,raw.mask,raw.outliers)) || _experiment_error("ensemble raw fields/flags disagree with final grid")
    raw.x==grid.x && raw.y==grid.y && raw.mask==grid.grid_mask || _experiment_error("ensemble axes/mask disagree with recipe")
    _experiment_digest(_experiment_pass_data(raw.parameters))==_experiment_digest(_experiment_pass_data(final)) || _experiment_error("ensemble final result parameters disagree with recipe")
    scale_data(scale)=scale===nothing ? nothing : Dict{String,Any}(String(k)=>getfield(scale,k) for k in fieldnames(PhysicalScale))
    _experiment_digest(scale_data(raw.scale))==_experiment_digest(scale_data(recipe.scale)) || _experiment_error("ensemble output scale differs from recipe")
    (raw.correlation_planes!==nothing)==final.keep_correlation_planes || _experiment_error("ensemble retained-plane policy differs from recipe")
    if raw.correlation_planes!==nothing
        size(raw.correlation_planes)==expected || _experiment_error("ensemble retained-plane grid shape differs")
        plane_size=final.padding ? 2 .* final.search_area_size : final.search_area_size
        all(p->p===nothing || p isa Matrix{T} && size(p)==plane_size,raw.correlation_planes) || _experiment_error("ensemble retained plane has invalid type/shape")
    end
    raw
end
function _ensemble_verify_output_metadata(file,index,record,run)
    _check_results_format(file,index.path)
    index.entry_keys==[last(split(result_key(1),'/'))] || _experiment_error("ensemble output must contain exactly one expected pooled result")
    any(k->haskey(file,k),("execution_diagnostics_format_version","execution_diagnostics","measurement_history_format_version","measurement_history",
        "stereo_execution_diagnostics_format_version","stereo_execution_diagnostics","pair_timing_format_version","pair_timing","stereo_pair_timing_format_version","stereo_pair_timing",
        "stereo_experiment_run_format_version","stereo_experiment_run")) && _experiment_error("crossed companion/run workflow in ensemble output")
    haskey(file,"ensemble_experiment_run_format_version") && file["ensemble_experiment_run_format_version"]===1 && haskey(file,"ensemble_experiment_run") || _experiment_error("missing/unsupported ensemble run association")
    saved=file["ensemble_experiment_run"]
    _experiment_keys(saved,["association","association_sha256"],"ensemble run association storage")
    _experiment_hash(saved["association_sha256"]) && _experiment_digest(saved["association"])==saved["association_sha256"] || _experiment_error("ensemble association digest mismatch")
    expected=_ensemble_output_association(record,run.run_id,run.measurement_sha256,run.record_diagnostics,run.environment)
    _experiment_keys(saved["association"],collect(keys(expected)),"ensemble run association")
    # Bool and integer equality are not interchangeable in an independent schema.
    a=saved["association"]
    all(k->a[k] isa Int,("input_pairs","scheduled_passes","total_contributions","completed_contributions","completed_pools","published_results")) &&
        a["record_diagnostics"] isa Bool && isequal(a,expected) || _experiment_error("ensemble run association disagrees with requested record/run")
    sources=String[String(record.input_files[i]["path"]) for pair in record.pairs for i in pair]
    haskey(file,"ensemble_sources") && file["ensemble_sources"] isa Vector{String} && file["ensemble_sources"]==sources || _experiment_error("ordered ensemble sources disagree with record")
    present=haskey(file,"ensemble_execution_diagnostics_format_version") || haskey(file,"ensemble_execution_diagnostics")
    if run.record_diagnostics
        haskey(file,"ensemble_execution_diagnostics_format_version") && file["ensemble_execution_diagnostics_format_version"]===1 && haskey(file,"ensemble_execution_diagnostics") || _experiment_error("requested ensemble diagnostics missing/unsupported")
        file["ensemble_execution_diagnostics"] isa Group && sort!(collect(String,keys(file["ensemble_execution_diagnostics"])))==index.entry_keys || _experiment_error("ensemble companion entry mapping disagrees")
    else
        present && _experiment_error("unexpected ensemble diagnostics contrary to run policy")
    end
    deepcopy(a)
end
function _ensemble_verify_packet(packet,record,run;raw=nothing)
    packet===nothing && _experiment_error("requested ensemble diagnostics packet missing")
    data=execution_diagnostics_data(packet;result=raw)
    recipe=record.recipe
    packet.pair_count==run.input_pairs && length(packet.passes)==run.scheduled_passes && packet.backend===recipe.backend &&
        packet.requested_image_type==string(recipe.image_type) && packet.core_source_sha256==run.environment["core_source_sha256"] &&
        packet.measurement_sha256==run.measurement_sha256 || _experiment_error("ensemble packet execution/input/binding disagrees with run")
    dims=Tuple(first(record.input_files)["image_size"])
    for (d,p) in zip(packet.passes,recipe.passes)
        grid=pass_grid(recipe.image_type,dims,p,recipe.mask,recipe.mask_threshold)
        d.requested_iterations==p.max_iterations && isequal(d.requested_tolerance,p.convergence_tol) && d.processing_size==dims && d.image_type==string(recipe.image_type) &&
            _experiment_digest(data["passes"][d.pass_index]["grid"])==_experiment_digest(_execution_primitive(_ensemble_grid_data(grid,p))) || _experiment_error("ensemble diagnostic schedule/grid differs from recipe")
    end
    nothing
end

"""
    verify_ensemble_experiment_run(record, run; output=run.output,
        verify_results=false, verify_inputs=false)

Inspect a dedicated ensemble run without rerunning PIV. A completed artifact
must match its byte hash, exact single-result mapping, separate run association,
ordered source labels and diagnostic recording policy. Metadata-only inspection
checks requested diagnostic schema/settings/linkage without loading raw result
pixels. `verify_results=true` additionally checks one raw payload's recipe
geometry, final parameters, mask/scale and measurement digest, and verifies its
recorded diagnostics against those raw fields. Parameters/plane contents are not
part of the measurement digest; plane type/shape/presence are checked separately.
`verify_inputs=true` checks current input bytes before/after inspection.

Unpublished failed/cancelled runs have no asserted output identity; metadata-only
inspection does not read their destination and raw verification is refused.
An explicit local `output` can locate a completed relocated artifact without
rewriting historical provenance. This is integrity/association, not source
authentication, stationarity, estimator applicability or accuracy verification.
No raw result or open handle is retained by the returned detached scalar data.
"""
function verify_ensemble_experiment_run(record::EnsembleExperimentRecord,run::EnsembleExperimentRun;
        output::AbstractString=run.output,verify_results::Bool=false,verify_inputs::Bool=false)
    snapshot=deepcopy(record);_ensemble_record_preflight(snapshot)
    checked=_ensemble_run_validate(_ensemble_run_data(deepcopy(run)),snapshot)
    verify_inputs && _ensemble_verify_input_bytes(snapshot)
    published=checked.status===:completed
    if !published
        verify_results && _experiment_error("unpublished ensemble runs have no raw output to verify")
    else
        path=_artifact_local_path(output)
        isfile(path) && _experiment_file_digest(path)==checked.output_sha256 || _experiment_error("ensemble output changed or is missing")
        index=_quality_whole_file_index(ResultFile(path))
        jldopen(file->_ensemble_verify_output_metadata(file,index,snapshot,checked),path,"r")
        packet=checked.record_diagnostics ? load_ensemble_execution_diagnostics(index,1) : nothing
        raw=verify_results ? index[1] : nothing
        if verify_results
            _ensemble_verify_raw(raw,snapshot)
            _history_result_digest(raw)==checked.measurement_sha256 || _experiment_error("ensemble raw measurement fields differ from run")
        end
        checked.record_diagnostics && _ensemble_verify_packet(packet,snapshot,checked;raw)
        raw=nothing;packet=nothing
        _check_result_file(index)
        _experiment_file_digest(path)==checked.output_sha256 || _experiment_error("ensemble output changed during inspection")
    end
    verify_inputs && _ensemble_verify_input_bytes(snapshot)
    Dict{String,Any}("run_id"=>checked.run_id,"recipe_id"=>checked.recipe_id,"input_id"=>checked.input_id,
        "input_pairs"=>checked.input_pairs,"scheduled_passes"=>checked.scheduled_passes,"total_contributions"=>checked.total_contributions,
        "completed_contributions"=>checked.completed_contributions,"completed_pools"=>checked.completed_pools,"published_results"=>checked.published_results,
        "output_integrity_checked"=>published,"run_association_checked"=>published,"measurement_fields_checked"=>verify_results,
        "current_input_bytes_checked"=>verify_inputs,"piv_recomputed"=>false,"source_authenticity_verified"=>false)
end
