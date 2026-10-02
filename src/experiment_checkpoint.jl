# Independent append-only checkpoint protocol for built-in planar recipes.
const EXPERIMENT_CHECKPOINT_VERSION = 1
const _CHECKPOINT_ACTIVE_WRITERS = Set{String}()
const _CHECKPOINT_WRITER_MUTEX = ReentrantLock()
const _CHECKPOINT_PAIR_DIGITS = 12

"""
    ExperimentCheckpoint

Location and identities of a version-1 planar checkpoint. Create with
[`create_checkpoint`](@ref) or reopen with [`load_checkpoint`](@ref). Immutable
native singleton result files live in the user-selected `output_dir`; separate
metadata records bind their bytes to the complete recipe and ordered inputs.
Use [`checkpoint_state`](@ref) to inspect current progress. No payload cache,
open writer, or result arrays are retained by this handle.
"""
struct ExperimentCheckpoint
    path::String
    output_dir::String
    checkpoint_id::String
    record::ExperimentRecord
    environment::Dict{String,Any}
end

"""
    CheckpointAttempt

Metadata for one checkpoint execution: attempt/checkpoint IDs, status
(`:completed`, `:cancelled`, `:failed`, or `:interrupted`), initial/total
committed counts, total pairs, start/finish times, and optional error summary.
Interrupted attempts have no inferred finish time. Processing failures persist
metadata when possible and rethrow their original exception instead of returning.
"""
struct CheckpointAttempt
    attempt_id::String
    checkpoint_id::String
    status::Symbol
    start_committed::Int
    committed::Int
    total_pairs::Int
    started_at::Float64
    finished_at::Union{Nothing,Float64}
    error::Union{Nothing,String}
end

_checkpoint_uuid(value) = value isa String && try UUIDs.UUID(value); true catch; false end
_checkpoint_index(i) = lpad(string(i),_CHECKPOINT_PAIR_DIGITS,'0')
_checkpoint_payload(i,id) = "pair-$(_checkpoint_index(i))-$id.jld2"
_checkpoint_descriptor(i) = "pair-$(_checkpoint_index(i)).jld2"
_checkpoint_attempt_name(i,id,phase) = "attempt-$(_checkpoint_index(i))-$id.$phase.jld2"
_checkpoint_pair_id(record,i) = _experiment_digest(_experiment_input_data(record.input_files,[record.pairs[i]]))
function _checkpoint_boundary(path)
    sep=Sys.iswindows() ? "\\" : "/"
    endswith(path,sep) ? path : path*sep
end

function _checkpoint_record(record; verify_files=false)
    _experiment_preflight(record;verify_files)
    _experiment_validate_environment(record.creation_environment)
    for run in record.runs
        _experiment_run(_experiment_run_data(run),record)
    end
    record.recipe.external_preprocess===nothing ||
        _experiment_error("checkpoint version 1 supports built-in preprocessing only; custom script/callback state cannot be resumed")
    length(record.pairs)<10^_CHECKPOINT_PAIR_DIGITS || _experiment_error("too many checkpoint pairs")
    record
end

function _checkpoint_normal_path(path)
    full=abspath(path)
    ispath(full) && return realpath(full)
    suffix=String[]
    while !ispath(full)
        pushfirst!(suffix,basename(full))
        parent=dirname(full)
        parent==full && _experiment_error("cannot resolve checkpoint path")
        full=parent
    end
    normpath(joinpath(realpath(full),suffix...))
end

function _checkpoint_disjoint(a,b)
    aa,bb=_checkpoint_normal_path(a),_checkpoint_normal_path(b)
    Sys.iswindows() && ((aa,bb)=(lowercase(aa),lowercase(bb)))
    aa!=bb && !startswith(aa,_checkpoint_boundary(bb)) && !startswith(bb,_checkpoint_boundary(aa)) ||
        _experiment_error("checkpoint metadata and result directories must be disjoint")
    nothing
end

function _checkpoint_empty_directory(path)
    islink(path) && _experiment_error("checkpoint creation refuses symlink destinations")
    ispath(path) && (!isdir(path) || !isempty(readdir(path))) &&
        _experiment_error("checkpoint creation requires a new or empty directory: $path")
end

function _checkpoint_write(path,kind,data;during_write=nothing)
    (ispath(path) || islink(path)) && _experiment_error("checkpoint metadata must not overwrite an existing destination")
    _experiment_digest(data) # No arbitrary objects in checkpoint metadata.
    jldopen(path,"w") do f
        f["checkpoint_format_version"]=EXPERIMENT_CHECKPOINT_VERSION
        f["checkpoint_kind"]=kind
        during_write===nothing || during_write()
        f["checkpoint_data"]=data
    end
    _checkpoint_read(path,kind)==data || _experiment_error("checkpoint metadata failed write verification")
    path
end

function _checkpoint_read(path,kind)
    jldopen(path,"r") do f
        Set(keys(f))==Set(["checkpoint_format_version","checkpoint_kind","checkpoint_data"]) ||
            _experiment_error("malformed checkpoint file: $path")
        f["checkpoint_format_version"]===EXPERIMENT_CHECKPOINT_VERSION ||
            _experiment_error("unknown checkpoint format version")
        f["checkpoint_kind"]==kind || _experiment_error("unexpected checkpoint record kind")
        data=f["checkpoint_data"]
        data isa Dict{String,Any} || _experiment_error("checkpoint metadata must be a primitive mapping")
        _experiment_digest(data)
        data
    end
end

# Base.Filesystem.rename falls back to copy+delete on errors; that fallback
# would expose a partial destination. Surface the native rename error instead.
# Publication is into a previously absent filename, under an exclusive writer.
function _checkpoint_publish(staging,final)
    dirname(abspath(staging))==dirname(abspath(final)) ||
        _experiment_error("checkpoint publication requires same-directory staging")
    (ispath(final) || islink(final)) && _experiment_error("checkpoint publication must not overwrite an existing file")
    code=ccall(:jl_fs_rename,Int32,(Cstring,Cstring),staging,final)
    code<0 && Base.uv_error("checkpoint rename",code)
    final
end

function _checkpoint_publish_metadata(final,kind,data)
    staging=joinpath(dirname(final),".$(basename(final)).$(UUIDs.uuid4()).partial")
    _checkpoint_write(staging,kind,data)
    _checkpoint_publish(staging,final)
end

function _checkpoint_check_result(path,record)
    index=ResultFile(path)
    length(index)==1 || _experiment_error("checkpoint payload must contain exactly one native result")
    result=index[1]
    result isa PIVResult{record.recipe.image_type} || _experiment_error("checkpoint payload type/precision mismatch")
    _experiment_pass_data(result.parameters)==_experiment_pass_data(last(record.recipe.passes)) ||
        _experiment_error("checkpoint payload pass settings mismatch")
    scale=record.recipe.scale
    (scale===nothing ? result.scale===nothing : result.scale!==nothing &&
        all(k->getfield(scale,k)==getfield(result.scale,k),fieldnames(PhysicalScale))) ||
        _experiment_error("checkpoint payload scale mismatch")
    result
end

function _checkpoint_identity(data,cp)
    get(data,"checkpoint_id",nothing)==cp.checkpoint_id &&
    get(data,"recipe_id",nothing)==recipe_identity(cp.record.recipe) &&
    get(data,"input_id",nothing)==cp.record.input_id || _experiment_error("checkpoint record identity mismatch")
end

"""
    create_checkpoint(checkpoint_dir, record::ExperimentRecord; output_dir) -> ExperimentCheckpoint

Create a separate version-1 checkpoint and per-pair native result directory.
Both directories must be disjoint and new/empty. Validate full recipe/input
content, built-in-only preprocessing and strict creation/execution software
identity before creating files. No existing user data are deleted/replaced.
CPU/KA, Float32/Float64, static masks/ROI and scalar scale are supported.
"""
function create_checkpoint(path::AbstractString,record::ExperimentRecord;output_dir::AbstractString)
    snapshot=deepcopy(record)
    _checkpoint_record(snapshot;verify_files=true)
    environment=_experiment_software()
    _experiment_environment_signature(environment)==_experiment_environment_signature(snapshot.creation_environment) ||
        _experiment_error("checkpoint creation requires the experiment creation software environment")
    metadata,output=_checkpoint_normal_path(path),_checkpoint_normal_path(output_dir)
    _checkpoint_disjoint(metadata,output)
    for directory in (metadata,output)
        _experiment_protect(directory,snapshot;records=true,results=true)
        _checkpoint_empty_directory(directory)
    end
    mkpath(metadata);mkpath(output)
    mkdir(joinpath(metadata,"commits"));mkdir(joinpath(metadata,"attempts"))
    recipe_path=joinpath(metadata,"experiment.jld2")
    save_experiment(recipe_path,snapshot)
    header=Dict{String,Any}("checkpoint_id"=>string(UUIDs.uuid4()),"recipe_id"=>recipe_identity(snapshot.recipe),
        "input_id"=>snapshot.input_id,"total_pairs"=>length(snapshot.pairs),"output_dir"=>output,
        "experiment_sha256"=>_experiment_file_digest(recipe_path),"environment"=>environment,
        "record_paths"=>copy(snapshot.record_paths),"created_at"=>time())
    _checkpoint_publish_metadata(joinpath(metadata,"header.jld2"),"header",header)
    load_checkpoint(metadata)
end

"""
    load_checkpoint(checkpoint_dir; output_dir=nothing) -> ExperimentCheckpoint

Reopen and validate checkpoint metadata and its complete committed prefix.
An explicitly relocated output directory is accepted after verifying every
committed payload's SHA-256 and native schema. Original input files are not
needed just to inspect; resume verifies them. Uncommitted staging/orphan files
are ignored, never adopted or deleted. Corrupt/missing commits are rejected.
"""
load_checkpoint(path::AbstractString;output_dir=nothing) = first(_checkpoint_load(path;output_dir))

function _checkpoint_load(path::AbstractString;output_dir=nothing)
    root=realpath(path)
    header=_checkpoint_read(joinpath(root,"header.jld2"),"header")
    _experiment_keys(header,["checkpoint_id","recipe_id","input_id","total_pairs","output_dir",
        "experiment_sha256","environment","record_paths","created_at"],"checkpoint header")
    _checkpoint_uuid(header["checkpoint_id"]) && _experiment_hash(header["experiment_sha256"]) ||
        _experiment_error("invalid checkpoint header identities")
    isfile(joinpath(root,"experiment.jld2")) && _experiment_file_digest(joinpath(root,"experiment.jld2"))==header["experiment_sha256"] ||
        _experiment_error("checkpoint experiment snapshot changed or is missing")
    record=load_experiment(joinpath(root,"experiment.jld2"))
    _checkpoint_record(record)
    header["recipe_id"]==recipe_identity(record.recipe) && header["input_id"]==record.input_id &&
        header["total_pairs"] isa Int && header["total_pairs"]==length(record.pairs) ||
        _experiment_error("checkpoint experiment identities/count mismatch")
    _experiment_validate_environment(header["environment"])
    _experiment_environment_signature(header["environment"])==_experiment_environment_signature(record.creation_environment) ||
        _experiment_error("checkpoint environment differs from recipe creation")
    header["created_at"] isa Float64 && isfinite(header["created_at"]) || _experiment_error("invalid checkpoint creation time")
    header["record_paths"] isa Vector{String} && all(isabspath,header["record_paths"]) || _experiment_error("invalid protected record paths")
    append!(record.record_paths,header["record_paths"])
    header["output_dir"] isa String && isabspath(header["output_dir"]) || _experiment_error("invalid checkpoint output directory")
    output=realpath(output_dir===nothing ? header["output_dir"] : output_dir)
    isdir(output) || _experiment_error("checkpoint output directory is missing")
    _checkpoint_disjoint(root,output)
    cp=ExperimentCheckpoint(root,output,header["checkpoint_id"],record,deepcopy(header["environment"]))
    cp,_checkpoint_snapshot(cp)
end

function _checkpoint_snapshot(cp)
    commits_dir,attempts_dir=joinpath(cp.path,"commits"),joinpath(cp.path,"attempts")
    for path in (commits_dir,attempts_dir,joinpath(cp.path,"header.jld2"),joinpath(cp.path,"experiment.jld2"))
        islink(path) && _experiment_error("checkpoint artifacts/directories must not be symlinks")
    end
    isdir(commits_dir) && isdir(attempts_dir) || _experiment_error("checkpoint metadata directories are missing")
    begins=Dict{String,Any}[]
    endings=Dict{String,Dict{String,Any}}()
    for name in sort!(filter(n->endswith(n,".jld2"),readdir(attempts_dir)))
        matched=match(r"^attempt-([0-9]{12})-([0-9a-f-]{36})\.(begin|end)\.jld2$",name)
        matched===nothing && _experiment_error("unknown final checkpoint attempt filename")
        ordinal,id,phase=parse(Int,matched[1]),String(matched[2]),String(matched[3])
        path=joinpath(attempts_dir,name)
        islink(path) && _experiment_error("checkpoint attempt must not be a symlink")
        data=_checkpoint_read(path,"attempt_"*phase)
        expected=phase=="begin" ? ["checkpoint_id","recipe_id","input_id","attempt_id","attempt_index",
            "start_committed","started_at","environment","input_pairs"] :
            ["checkpoint_id","recipe_id","input_id","attempt_id","attempt_index",
             "status","committed","ended_at","recorded_at","error"]
        _experiment_keys(data,expected,"attempt "*phase)
        _checkpoint_identity(data,cp)
        _checkpoint_uuid(id) && data["attempt_id"]==id && data["attempt_index"]===ordinal ||
            _experiment_error("invalid checkpoint attempt identity/index")
        if phase=="begin"
            ordinal==length(begins)+1 || _experiment_error("checkpoint attempts must be contiguous")
            data["start_committed"] isa Int && 0<=data["start_committed"]<=length(cp.record.pairs) || _experiment_error("invalid attempt starting count")
            data["started_at"] isa Float64 && isfinite(data["started_at"]) || _experiment_error("invalid attempt start time")
            _experiment_validate_environment(data["environment"])
            _experiment_environment_signature(data["environment"])==_experiment_environment_signature(cp.environment) || _experiment_error("mixed checkpoint software environments")
            data["input_pairs"] isa Vector && length(data["input_pairs"])==length(cp.record.pairs) &&
                all(p->p isa Vector{String} && length(p)==2 && all(isabspath,p),data["input_pairs"]) || _experiment_error("invalid attempt input locators")
            push!(begins,data)
        else
            haskey(endings,id) && _experiment_error("duplicate checkpoint attempt ending")
            status=data["status"]
            status isa String && status in ("completed","cancelled","failed","interrupted") || _experiment_error("invalid checkpoint attempt status")
            count=data["committed"]
            count isa Int && 0<=count<=length(cp.record.pairs) || _experiment_error("invalid committed count")
            data["recorded_at"] isa Float64 && isfinite(data["recorded_at"]) || _experiment_error("invalid attempt record time")
            if status=="interrupted"
                data["ended_at"]===nothing || _experiment_error("interrupted attempt must not infer finish time")
            else
                data["ended_at"] isa Float64 && isfinite(data["ended_at"]) || _experiment_error("invalid attempt end time")
            end
            status=="completed" && count!=length(cp.record.pairs) && _experiment_error("completed attempt has incomplete data")
            status=="cancelled" && count==length(cp.record.pairs) && _experiment_error("final commit counts as completion")
            (status=="failed" ? data["error"] isa String : data["error"]===nothing) || _experiment_error("invalid checkpoint attempt error")
            endings[id]=data
        end
    end
    commits=Dict{String,Any}[]
    for name in sort!(filter(n->endswith(n,".jld2"),readdir(commits_dir)))
        name==_checkpoint_descriptor(length(commits)+1) || _experiment_error("checkpoint commits must be an exact contiguous prefix")
        path=joinpath(commits_dir,name)
        islink(path) && _experiment_error("checkpoint commit must not be a symlink")
        data=_checkpoint_read(path,"commit")
        _experiment_keys(data,["checkpoint_id","recipe_id","input_id","pair_index","pair_id","attempt_id",
            "result_file","result_sha256","committed_at"],"checkpoint commit")
        _checkpoint_identity(data,cp)
        i=length(commits)+1
        i<=length(cp.record.pairs) && data["pair_index"]===i && data["pair_id"]==_checkpoint_pair_id(cp.record,i) || _experiment_error("checkpoint pair identity/index mismatch")
        id=data["attempt_id"]
        _checkpoint_uuid(id) && any(b->b["attempt_id"]==id,begins) || _experiment_error("checkpoint commit has no known attempt")
        data["result_file"]==_checkpoint_payload(i,id) && _experiment_hash(data["result_sha256"]) || _experiment_error("invalid checkpoint payload locator/digest")
        data["committed_at"] isa Float64 && isfinite(data["committed_at"]) || _experiment_error("invalid commit time")
        payload=joinpath(cp.output_dir,data["result_file"])
        !islink(payload) && isfile(payload) && _experiment_file_digest(payload)==data["result_sha256"] || _experiment_error("committed checkpoint result changed or is missing")
        _checkpoint_check_result(payload,cp.record)
        beginning=only(b for b in begins if b["attempt_id"]==id)
        sources=jldopen(f->f[source_key(1)],payload,"r")
        sources==beginning["input_pairs"][i] ||
            _experiment_error("checkpoint payload source locators differ from its attempt")
        push!(commits,data)
    end
    previous=0
    ids=Set(b["attempt_id"] for b in begins)
    all(id->id in ids,keys(endings)) || _experiment_error("attempt ending has no begin record")
    for (n,beginning) in enumerate(begins)
        id=beginning["attempt_id"]
        beginning["start_committed"]==previous || _experiment_error("checkpoint attempt starting count mismatch")
        own=findall(c->c["attempt_id"]==id,commits)
        own==collect(previous+1:previous+length(own)) || _experiment_error("checkpoint commits mix execution attempts")
        current=previous+length(own)
        if haskey(endings,id)
            ending=endings[id]
            ending["attempt_index"]===n && ending["committed"]==current || _experiment_error("checkpoint attempt ending count/index mismatch")
            ending["ended_at"]===nothing || ending["ended_at"]>=beginning["started_at"] || _experiment_error("attempt times are reversed")
        elseif n!=length(begins)
            _experiment_error("unfinished checkpoint attempt precedes another attempt")
        end
        previous=current
    end
    previous==length(commits) || _experiment_error("unowned checkpoint commits")
    (;commits,begins,endings)
end

function _checkpoint_fresh(cp)
    current,state=_checkpoint_load(cp.path;output_dir=cp.output_dir)
    current.checkpoint_id==cp.checkpoint_id && recipe_identity(current.record.recipe)==recipe_identity(cp.record.recipe) &&
        current.record.input_id==cp.record.input_id || _experiment_error("checkpoint handle identities changed")
    current,state
end

"""
    checkpoint_state(checkpoint) -> NamedTuple

Verify committed data and report `committed`, `total_pairs`, `data_complete`,
`status` and `writer_lock_present`. Status is `:ready` before execution;
an unmatched begin is `:unfinished`, never automatically inferred cancellation
or failure. Complete data can coexist with an unfinished/failed attempt.
"""
function checkpoint_state(cp::ExperimentCheckpoint)
    current,state=_checkpoint_fresh(cp)
    status=isempty(state.begins) ? :ready : haskey(state.endings,last(state.begins)["attempt_id"]) ?
        Symbol(state.endings[last(state.begins)["attempt_id"]]["status"]) : :unfinished
    (;checkpoint_id=current.checkpoint_id,committed=length(state.commits),total_pairs=length(current.record.pairs),
        data_complete=length(state.commits)==length(current.record.pairs),status,
        writer_lock_present=isdir(joinpath(current.path,"writer-lock")))
end

function _checkpoint_end(cp,beginning,status,count,error=nothing)
    data=Dict{String,Any}("checkpoint_id"=>cp.checkpoint_id,"recipe_id"=>recipe_identity(cp.record.recipe),
        "input_id"=>cp.record.input_id,"attempt_id"=>beginning["attempt_id"],"attempt_index"=>beginning["attempt_index"],
        "status"=>String(status),"committed"=>count,"ended_at"=>status===:interrupted ? nothing : time(),
        "recorded_at"=>time(),"error"=>error)
    path=joinpath(cp.path,"attempts",_checkpoint_attempt_name(beginning["attempt_index"],beginning["attempt_id"],"end"))
    _checkpoint_publish_metadata(path,"attempt_end",data)
    CheckpointAttempt(beginning["attempt_id"],cp.checkpoint_id,status,beginning["start_committed"],count,
        length(cp.record.pairs),beginning["started_at"],data["ended_at"],error)
end

# Reliable in-process live-writer detection; PIDs of other processes are
# provenance only. Explicit recovery asserts they have stopped. No auto expiry.
function _checkpoint_lock(cp,recover,hook,acquired)
    lock(_CHECKPOINT_WRITER_MUTEX) do
        _checkpoint_lock_locked(cp,recover,hook,acquired)
    end
end

_checkpoint_hook(hook,phase,i=0) = hook===nothing ? nothing : hook(phase,i)

function _checkpoint_lock_locked(cp,recover,hook,acquired)
    path=joinpath(cp.path,"writer-lock")
    cp.path in _CHECKPOINT_ACTIVE_WRITERS && _experiment_error("checkpoint already has a live writer in this process")
    if ispath(path) || islink(path)
        recover || _experiment_error("checkpoint writer lock remains; stop the former writer and explicitly recover_interrupted")
        isdir(path) && !islink(path) || _experiment_error("invalid checkpoint writer lock")
        names=readdir(path)
        all(n->n=="owner.jld2" || occursin(r"^\.owner\.jld2\.[0-9a-f-]{36}\.partial$",n),names) &&
            all(n->!islink(joinpath(path,n)) && isfile(joinpath(path,n)),names) ||
            _experiment_error("writer lock contains unknown files; refusing recovery")
        if "owner.jld2" in names
            owner=_checkpoint_read(joinpath(path,"owner.jld2"),"writer")
            _experiment_keys(owner,["checkpoint_id","pid"],"writer lock")
            owner["checkpoint_id"]==cp.checkpoint_id && owner["pid"] isa Int && owner["pid"]>0 || _experiment_error("invalid writer ownership")
        end
        # Preserve recognized empty/partial/complete locks; never delete their
        # content or guess whether another process is still running.
        archive=joinpath(cp.path,"recovered-writer-lock-$(UUIDs.uuid4())")
        _checkpoint_publish(path,archive)
    end
    try
        mkdir(path) # Exclusive creation; competing writers fail without overwriting.
        push!(_CHECKPOINT_ACTIVE_WRITERS,cp.path)
        acquired[]=true # Caller cleanup remains armed across return/assignment.
        _checkpoint_hook(hook,:lock_directory_created)
        final=joinpath(path,"owner.jld2")
        staging=joinpath(path,".owner.jld2.$(UUIDs.uuid4()).partial")
        _checkpoint_write(staging,"writer",Dict{String,Any}("checkpoint_id"=>cp.checkpoint_id,"pid"=>Int(getpid()));
            during_write=()->_checkpoint_hook(hook,:lock_owner_staging))
        _checkpoint_hook(hook,:lock_owner_staged)
        _checkpoint_publish(staging,final)
        _checkpoint_hook(hook,:lock_owner_published)
    catch
        delete!(_CHECKPOINT_ACTIVE_WRITERS,cp.path)
        rethrow()
    end
    path
end

function _checkpoint_unlock(cp,path)
    lock(_CHECKPOINT_WRITER_MUTEX) do
        try
            if isdir(path) && !islink(path) && readdir(path)==["owner.jld2"]
                owner=_checkpoint_read(joinpath(path,"owner.jld2"),"writer")
                if owner["checkpoint_id"]==cp.checkpoint_id && owner["pid"]==getpid()
                    rm(joinpath(path,"owner.jld2"));rm(path)
                end
            end
        finally
            delete!(_CHECKPOINT_ACTIVE_WRITERS,cp.path)
        end
    end
end

struct _CheckpointCancelled <: Exception end

function _checkpoint_begin(cp,record,state,environment)
    ordinal=length(state.begins)+1
    id=string(UUIDs.uuid4())
    data=Dict{String,Any}("checkpoint_id"=>cp.checkpoint_id,"recipe_id"=>recipe_identity(record.recipe),
        "input_id"=>record.input_id,"attempt_id"=>id,"attempt_index"=>ordinal,"start_committed"=>length(state.commits),
        "started_at"=>time(),"environment"=>environment,
        "input_pairs"=>[[record.input_files[i]["path"] for i in pair] for pair in record.pairs])
    _checkpoint_publish_metadata(joinpath(cp.path,"attempts",_checkpoint_attempt_name(ordinal,id,"begin")),"attempt_begin",data)
    data
end

function _checkpoint_commit(cp,record,beginning,i,result,hook)
    filename=_checkpoint_payload(i,beginning["attempt_id"])
    final=joinpath(cp.output_dir,filename)
    staging=joinpath(cp.output_dir,".$filename.$(UUIDs.uuid4()).partial")
    (ispath(staging) || islink(staging)) && _experiment_error("checkpoint staging file already exists")
    jldopen(staging,"w") do f
        f["format_version"]=RESULTS_FORMAT_VERSION
        _checkpoint_hook(hook,:during_staging,i)
        f[result_key(1)]=result
        f[source_key(1)]=[record.input_files[j]["path"] for j in record.pairs[i]]
    end
    _checkpoint_hook(hook,:payload_closed,i)
    _checkpoint_check_result(staging,record)
    digest=_experiment_file_digest(staging)
    _checkpoint_publish(staging,final)
    _checkpoint_hook(hook,:payload_published,i)
    data=Dict{String,Any}("checkpoint_id"=>cp.checkpoint_id,"recipe_id"=>recipe_identity(record.recipe),
        "input_id"=>record.input_id,"pair_index"=>i,"pair_id"=>_checkpoint_pair_id(record,i),
        "attempt_id"=>beginning["attempt_id"],"result_file"=>filename,"result_sha256"=>digest,"committed_at"=>time())
    descriptor=joinpath(cp.path,"commits",_checkpoint_descriptor(i))
    temporary=joinpath(dirname(descriptor),".$(basename(descriptor)).$(UUIDs.uuid4()).partial")
    _checkpoint_write(temporary,"commit",data)
    _checkpoint_hook(hook,:descriptor_staged,i)
    _checkpoint_publish(temporary,descriptor)
    _checkpoint_hook(hook,:descriptor_published,i)
    nothing
end

"""
    resume_checkpoint!(checkpoint, record=checkpoint.record;
                       cancel=()->false, progress=nothing, recover_interrupted=false)

Process only the remaining pairs after validating the exact ordered input/recipe
identities, all input bytes, all committed payloads and strict software identity.
Only built-in preprocessing is supported. Relocated input locators may be
supplied through an equivalent `ExperimentRecord`. Progress receives absolute
`(committed,total)` counts only after descriptor publication; callback errors
are failures. Cancellation is checked before computation and between committed
pairs, after the pair in flight; cancellation after the final pair is completion.

An unfinished attempt or remaining writer lock requires explicit
`recover_interrupted=true`: the caller asserts the former writer has stopped.
Known live writers in this process are refused; other PIDs are provenance only.
Concurrent recovery/writer mutation is unsupported. Recognized old lock states
are archived, including empty/partial publication gaps; unknown files are refused.
Interrupted attempts receive no guessed finish time. Failures rethrow their
original exception; secondary metadata/cleanup failures are logged.

Native singleton data and descriptors are closed/validated then published with
same-directory renames, without copy fallback or replacing committed names.
Uncommitted staging/orphan files are ignored and never adopted/deleted. This
protocol is for tested local filesystems, not a power-loss/fsync durability claim.
"""
function resume_checkpoint!(cp::ExperimentCheckpoint,record::ExperimentRecord=cp.record;
                            cancel::Function=()->false,progress::Union{Nothing,Function}=nothing,
                            recover_interrupted::Bool=false,_phase_hook=nothing)
    snapshot=deepcopy(record)
    _checkpoint_record(snapshot;verify_files=true)
    current,state=_checkpoint_fresh(cp)
    recipe_identity(snapshot.recipe)==recipe_identity(current.record.recipe) && snapshot.input_id==current.record.input_id ||
        _experiment_error("resume recipe or ordered input identity differs from checkpoint")
    environment=_experiment_software()
    _experiment_environment_signature(environment)==_experiment_environment_signature(current.environment) ||
        _experiment_error("checkpoint resume requires the original software environment; mixed environments are unsupported")
    unfinished=!isempty(state.begins) && !haskey(state.endings,last(state.begins)["attempt_id"])
    unfinished && !recover_interrupted && _experiment_error("unfinished attempt requires explicit interruption recovery")
    lock_path=joinpath(current.path,"writer-lock")
    acquired=Ref(false)
    primary_failed=false
    try
        _checkpoint_lock(current,recover_interrupted,_phase_hook,acquired)
        # Recheck under the writer lock before publishing attempt metadata.
        current,state=_checkpoint_fresh(current)
        if !isempty(state.begins) && !haskey(state.endings,last(state.begins)["attempt_id"])
            recover_interrupted || _experiment_error("unfinished attempt requires explicit interruption recovery")
            _checkpoint_end(current,last(state.begins),:interrupted,length(state.commits))
            state=_checkpoint_snapshot(current)
        end
        if length(state.commits)==length(snapshot.pairs) && !isempty(state.begins) &&
           state.endings[last(state.begins)["attempt_id"]]["status"]=="completed"
            b=last(state.begins);e=state.endings[b["attempt_id"]]
            return CheckpointAttempt(b["attempt_id"],current.checkpoint_id,:completed,b["start_committed"],length(state.commits),
                length(snapshot.pairs),b["started_at"],e["ended_at"],nothing)
        end
        beginning=_checkpoint_begin(current,snapshot,state,environment)
        completed=Ref(length(state.commits))
        try
            _checkpoint_hook(_phase_hook,:attempt_started)
            if completed[]<length(snapshot.pairs)
                cancel() && throw(_CheckpointCancelled())
                recipe=snapshot.recipe
                source=FrameSource(length(snapshot.input_files),j->_experiment_load_image(snapshot.input_files[j],recipe.image_type))
                first_index=completed[]+1
                pairs=[(FrameRef(source,p[1]),FrameRef(source,p[2])) for p in snapshot.pairs[first_index:end]]
                delivery=(relative,result)->begin
                    absolute=first_index+relative-1
                    _checkpoint_commit(current,snapshot,beginning,absolute,result,_phase_hook)
                    completed[]=absolute
                    _checkpoint_hook(_phase_hook,:committed,absolute)
                    progress===nothing || progress(absolute,length(snapshot.pairs))
                    absolute<length(snapshot.pairs) && cancel() && throw(_CheckpointCancelled())
                end
                run_piv_sequence(pairs,recipe.passes;collect_results=false,progress=false,on_result=delivery,
                    preprocess=_experiment_preprocess(recipe,nothing),backend=recipe.backend,image_type=recipe.image_type,
                    mask=recipe.mask,roi=recipe.roi,scale=recipe.scale,threaded=recipe.threaded,
                    predictor_smoothing=recipe.predictor_smoothing,mask_threshold=recipe.mask_threshold,
                    uncertainty_backend=recipe.uncertainty_backend)
            end
            _checkpoint_hook(_phase_hook,:before_terminal,completed[])
            _checkpoint_end(current,beginning,:completed,completed[])
        catch err
            if err isa _CheckpointCancelled
                completed[]=length(_checkpoint_snapshot(current).commits)
                _checkpoint_end(current,beginning,completed[]==length(snapshot.pairs) ? :completed : :cancelled,completed[])
            else
                primary_failed=true
                try
                    completed[]=length(_checkpoint_snapshot(current).commits)
                    _checkpoint_end(current,beginning,:failed,completed[],sprint(showerror,err))
                catch metadata_error
                    @error "Failed to record checkpoint attempt failure" exception=metadata_error
                end
                rethrow()
            end
        end
    catch
        primary_failed=true
        rethrow()
    finally
        try acquired[] && _checkpoint_unlock(current,lock_path)
        catch cleanup_error
            primary_failed ? (@error "Failed checkpoint writer cleanup" exception=cleanup_error) : rethrow()
        end
    end
end

"""
    CheckpointResults <: AbstractVector

Fixed lazy index of a checkpoint's validated committed prefix. Each access
rechecks content identity and loads one raw `PIVResult`; no payload or file
handle is cached. The index does not follow subsequent commits. `collect`
retains payloads at the caller's request. Construct with [`checkpoint_results`](@ref).
"""
struct CheckpointResults <: AbstractVector{PIVResult}
    paths::Vector{String}
    digests::Vector{String}
end
Base.size(index::CheckpointResults)=(length(index.paths),)
Base.IndexStyle(::Type{CheckpointResults})=IndexLinear()
_result_protected_paths(index::CheckpointResults)=copy(index.paths)
function Base.getindex(index::CheckpointResults,i::Int)
    checkbounds(index,i)
    path=index.paths[i]
    !islink(path) && isfile(path) && _experiment_file_digest(path)==index.digests[i] || _experiment_error("checkpoint result changed since indexing")
    result=only(ResultFile(path))
    _experiment_file_digest(path)==index.digests[i] || _experiment_error("checkpoint result changed during reading")
    result isa PIVResult || _experiment_error("checkpoint result is not planar PIV")
    result
end

"""
    checkpoint_results(checkpoint) -> CheckpointResults

Verify metadata/payloads and index the current committed prefix with O(number
of pairs) key/digest metadata. Payload verification reads one result at a time.
"""
function checkpoint_results(cp::ExperimentCheckpoint)
    current,state=_checkpoint_fresh(cp)
    CheckpointResults([joinpath(current.output_dir,c["result_file"]) for c in state.commits],
        [c["result_sha256"] for c in state.commits])
end

"""
    save_checkpoint_results(path, checkpoint) -> path

Stream the completed checkpoint data into a fresh ordinary native results file,
one payload at a time. Existing destinations, source/record/checkpoint aliases
and paths inside either owned directory are refused. Export is a derivative
operation: a failed/terminated export cannot alter committed checkpoint data.
Close/validate a unique same-directory staging file before publication. This
uses an exclusive per-destination export lock, preventing concurrent library
exports from overwriting one another. A terminated export may leave its lock;
use another fresh destination. Concurrent outside writers are unsupported.
Source labels retain each committed attempt's actual input locators, including
relocated inputs. No power-loss or network-filesystem durability is guaranteed.
"""
function save_checkpoint_results(path::AbstractString,cp::ExperimentCheckpoint;_phase_hook=nothing)
    current,state=_checkpoint_fresh(cp)
    length(state.commits)==length(current.record.pairs) || _experiment_error("native export requires all checkpoint pairs committed")
    output=_checkpoint_normal_path(path)
    _experiment_protect(output,current.record;records=true,results=true)
    for directory in (current.path,current.output_dir)
        root=Sys.iswindows() ? lowercase(directory) : directory
        candidate=Sys.iswindows() ? lowercase(output) : output
        (candidate==root || startswith(candidate,_checkpoint_boundary(root))) && _experiment_error("aggregate export must be outside checkpoint/result directories")
    end
    (ispath(output) || islink(output)) && _experiment_error("checkpoint export requires a fresh destination")
    isdir(dirname(output)) || _experiment_error("export parent directory must exist")
    export_lock=joinpath(dirname(output),".$(basename(output)).checkpoint-export-lock")
    (ispath(export_lock) || islink(export_lock)) && _experiment_error("checkpoint export destination is locked; use a fresh destination after interruption")
    mkdir(export_lock)
    token=string(UUIDs.uuid4())
    failed=false
    try
        write(joinpath(export_lock,"owner"),token)
        (ispath(output) || islink(output)) && _experiment_error("checkpoint export requires a fresh destination")
        _checkpoint_export(output,current,state,_phase_hook)
    catch
        failed=true
        rethrow()
    finally
        try
            names=readdir(export_lock)
            if names==["owner"] && !islink(joinpath(export_lock,"owner")) && read(joinpath(export_lock,"owner"),String)==token
                rm(joinpath(export_lock,"owner"));rm(export_lock)
            elseif isempty(names)
                rm(export_lock)
            else
                _experiment_error("checkpoint export lock changed; refusing cleanup")
            end
        catch cleanup_error
            failed ? (@error "Failed checkpoint export cleanup" exception=cleanup_error) : rethrow()
        end
    end
    path
end

function _checkpoint_export(output,current,state,hook)
    staging=joinpath(dirname(output),".$(basename(output)).$(UUIDs.uuid4()).partial")
    (ispath(staging) || islink(staging)) && _experiment_error("checkpoint export staging destination already exists")
    index=CheckpointResults([joinpath(current.output_dir,c["result_file"]) for c in state.commits],[c["result_sha256"] for c in state.commits])
    inputs=Dict(b["attempt_id"]=>b["input_pairs"] for b in state.begins)
    jldopen(staging,"w") do f
        f["format_version"]=RESULTS_FORMAT_VERSION
        f["checkpoint_id"]=current.checkpoint_id
        f["checkpoint_recipe_id"]=recipe_identity(current.record.recipe)
        f["checkpoint_input_id"]=current.record.input_id
        for i in eachindex(index)
            f[result_key(i)]=index[i]
            f[source_key(i)]=inputs[state.commits[i]["attempt_id"]][i]
        end
    end
    length(ResultFile(staging))==length(index) || _experiment_error("checkpoint aggregate verification failed")
    _checkpoint_hook(hook,:aggregate_staged,length(index))
    _checkpoint_publish(staging,output)
    output
end
