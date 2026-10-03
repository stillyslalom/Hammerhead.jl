module WorkerProtocol
using TOML, SHA, UUIDs

const VERSION = 1
const MAX_CONTROL_BYTES = 131072
const MAX_TERMINAL_BYTES = 16777216
const MAX_ERROR_CHARS = 4000
const REQUEST_KEYS = ("version","job_id","owner_pid","record_file","record_sha256","recipe_id","input_id",
    "total","output","run_record","allow_environment_change","ack_timeout","project","threads",
    "fftw_threads","protected_record_paths","worker_sources")
error(message)=throw(ArgumentError("replay worker protocol: $message"))
digest(path)=open(io->bytes2hex(sha256(io)),path,"r")
is_hash(x)=x isa String && occursin(r"^[0-9a-f]{64}$",x)
function check_keys(data,keys)
    data isa AbstractDict && Set(Base.keys(data))==Set(keys) || error("unexpected fields")
    data
end
function uuid(value)
    value isa String || error("job ID must be a string")
    try UUID(value) catch;error("invalid job ID") end
    value
end
function request_data(data)
    check_keys(data,REQUEST_KEYS);data["version"]===VERSION || error("unsupported request version")
    uuid(data["job_id"])
    all(k->data[k] isa Int && data[k]>0,("owner_pid","total","threads","fftw_threads")) || error("invalid request counts")
    all(k->data[k] isa String && !isempty(data[k]),("record_file","output","project")) || error("invalid request paths")
    data["run_record"] isa String && data["allow_environment_change"] isa Bool || error("invalid request options")
    all(k->is_hash(data[k]),("record_sha256","recipe_id","input_id")) || error("invalid request identities")
    data["ack_timeout"] isa Float64 && isfinite(data["ack_timeout"]) && data["ack_timeout"]>0 || error("invalid acknowledgement deadline")
    data["protected_record_paths"] isa AbstractVector && all(p->p isa String,data["protected_record_paths"]) || error("invalid protected record paths")
    data["worker_sources"] isa AbstractDict && !isempty(data["worker_sources"]) &&
        all(p->first(p) isa String && is_hash(last(p)),pairs(data["worker_sources"])) || error("invalid worker source identity")
    data
end
function progress_data(data,job_id,total)
    check_keys(data,("version","job_id","sequence","written","total"))
    data["version"]===VERSION && data["job_id"]==job_id && data["total"]===total || error("crossed progress identity/count")
    data["sequence"] isa Int && data["written"]===data["sequence"] && 1<=data["written"]<=total || error("invalid written progress")
    (kind=:progress,job_id=job_id,sequence=data["sequence"],written=data["written"],total=total)
end
function acknowledgement_data(data,job_id,sequence)
    check_keys(data,("version","job_id","sequence","action","error_type","message","backtrace"))
    data["version"]===VERSION && data["job_id"]==job_id && data["sequence"]===sequence || error("crossed acknowledgement")
    data["action"] in ("continue","abort") && all(k->data[k] isa String,("error_type","message","backtrace")) || error("invalid acknowledgement")
    data["action"]=="continue" && any(k->!isempty(data[k]),("error_type","message","backtrace")) && error("continue acknowledgement carries an error")
    data["action"]=="abort" && isempty(data["error_type"]) && error("abort acknowledgement lacks error type")
    data
end
function cancellation_data(data,job_id)
    check_keys(data,("version","job_id","cancel"))
    data["version"]===VERSION && data["job_id"]==job_id && data["cancel"]===true || error("invalid cancellation request")
    true
end
function read_control(path)
    isfile(path) && !islink(path) || error("control file missing or symbolic")
    filesize(path)<=MAX_CONTROL_BYTES || error("control file exceeds bounded size")
    TOML.parsefile(path)
end
function native_replace(source,destination)
    dirname(abspath(source))==dirname(abspath(destination)) || error("control publication must stay in its directory")
    result=ccall(:jl_fs_rename,Int32,(Cstring,Cstring),source,destination)
    result<0 && Base.uv_error("replay worker control publication",result)
    destination
end
function write_control(path,data)
    io=IOBuffer();TOML.print(io,data;sorted=true);bytes=take!(io)
    length(bytes)<=MAX_CONTROL_BYTES || error("control packet exceeds bounded size")
    staging=path*"."*string(uuid4())*".partial"
    try
        open(staging,"w") do stream;write(stream,bytes);flush(stream);end
        native_replace(staging,path)
    finally
        try
            isfile(staging) && rm(staging;force=true)
        catch secondary
            @error "Failed to remove replay control staging" exception=secondary
        end
    end
    path
end
function bounded_text(value)
    text=String(value)
    length(text)<=MAX_ERROR_CHARS ? text : first(text,MAX_ERROR_CHARS)*"\n[worker protocol text truncated]"
end
function error_data(exception,trace=nothing)
    original=exception isa CapturedException ? exception.ex : exception
    rendered=exception isa CapturedException ? sprint(showerror,exception) :
        trace===nothing ? sprint(showerror,exception) : sprint(showerror,exception,trace)
    (error_type=bounded_text(string(typeof(original))),message=bounded_text(sprint(showerror,original)),backtrace=bounded_text(rendered))
end
function acknowledgement(job_id,sequence;abort=nothing)
    info=abort===nothing ? (error_type="",message="",backtrace="") : error_data(abort)
    Dict{String,Any}("version"=>VERSION,"job_id"=>job_id,"sequence"=>sequence,
        "action"=>abort===nothing ? "continue" : "abort",(String(k)=>v for (k,v) in pairs(info))...)
end
function terminal_data(data,job_id,total)
    check_keys(data,("version","job_id","status","written","total","run","error_type","message","backtrace",
        "history_error","core_cleanup_returned","worker_pid","loaded_modules"))
    data["version"]===VERSION && data["job_id"]==job_id && data["total"]===total || error("crossed terminal identity")
    data["status"] in ("completed","cancelled","failed") && data["written"] isa Int && 0<=data["written"]<=total || error("invalid terminal status/count")
    data["core_cleanup_returned"]===true && data["worker_pid"] isa Int && data["worker_pid"]>0 || error("terminal lacks core cleanup/PID")
    all(k->data[k] isa String,("error_type","message","backtrace","history_error")) || error("invalid terminal errors")
    data["loaded_modules"] isa AbstractVector && all(m->m isa String,data["loaded_modules"]) || error("invalid module evidence")
    any(m->m in ("QML","QMLMakie","GLMakie","HammerheadGUI"),data["loaded_modules"]) && error("worker imported a GUI module")
    if data["status"]=="completed"
        data["written"]===total && data["run"] isa AbstractDict && all(k->isempty(data[k]),("error_type","message","backtrace")) || error("completed terminal lacks returned run")
    else
        data["run"]===nothing || data["run"] isa AbstractDict || error("invalid recovered run")
        !isempty(data["error_type"]) || error("failed/cancelled terminal lacks error")
        data["status"]=="cancelled" && data["written"]==total && error("last-written cancellation must complete")
    end
    data
end
end
