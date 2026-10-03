module ReplayWorkerClient
using Hammerhead, UUIDs
using Base.Threads: threadid
export ReplayJob, ReplayOutcome, start_replay, poll!, acknowledge_progress!, request_cancel!,
    active, outcome, job_id, pid, shutdown!, startup_cleanup_pending, poll_startup_cleanup!
include("worker_protocol.jl")
using .WorkerProtocol
const _owned_job=Ref{Any}(nothing)
const _incomplete_startup=Ref{Any}(nothing)
startup_cleanup_pending()=_incomplete_startup[]!==nothing
function poll_startup_cleanup!()
    pending=_incomplete_startup[]
    pending===nothing && return true
    threadid()==pending.owner_thread || throw(ArgumentError("startup cleanup requires its owning thread"))
    process_exited(pending.process) && _lease_count(pending.lease)==0 || return false
    try wait(pending.process) catch end
    pending.stdout_io===nothing || close(pending.stdout_io)
    pending.stderr_io===nothing || close(pending.stderr_io)
    _close_lease!(pending.lease)
    _incomplete_startup[]=nothing
    true
end

struct ReplayOutcome
    status::Symbol
    written::Int
    total::Int
    run::Union{Nothing,ExperimentRun}
    error_type::String
    message::String
    backtrace::String
    history_error::String
    exit_code::Int
    cleanup_confirmed::Bool
    job_id::String
    directory::String
end

# The handle is non-inheritable. Windows closes it when its owning Qt process
# exits, including abrupt termination. No PID enumeration/lookup is used to kill.
# https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
mutable struct _WindowsLease
    handle::Ptr{Cvoid}
end
function _winerror(name)
    code=ccall((:GetLastError,"kernel32"),UInt32,())
    throw(ErrorException("$name failed (Windows error $code)"))
end
function _close_lease!(lease::_WindowsLease)
    lease.handle==C_NULL && return
    ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),lease.handle)!=0 || _winerror("CloseHandle(job)")
    lease.handle=C_NULL
end
function _new_lease()
    Sys.iswindows() && Sys.WORD_SIZE==64 || throw(ArgumentError("subprocess replay requires validated 64-bit Windows Job Object ownership; this prototype's saved replay is unavailable on other hosts pending equivalent ownership validation"))
    handle=ccall((:CreateJobObjectW,"kernel32"),Ptr{Cvoid},(Ptr{Cvoid},Ptr{UInt16}),C_NULL,C_NULL)
    handle!=C_NULL || _winerror("CreateJobObjectW")
    lease=_WindowsLease(handle)
    try
        # Win64 JOBOBJECT_EXTENDED_LIMIT_INFORMATION: BasicLimitInformation's
        # DWORD LimitFlags is at offset16; total structure size144 bytes.
        info=zeros(UInt8,144)
        GC.@preserve info begin
            unsafe_store!(Ptr{UInt32}(pointer(info)+16),UInt32(0x2000)) # KILL_ON_JOB_CLOSE
            ccall((:SetInformationJobObject,"kernel32"),Int32,(Ptr{Cvoid},Int32,Ptr{Cvoid},UInt32),handle,9,pointer(info),144)!=0 || _winerror("SetInformationJobObject")
        end
        finalizer(lease) do owned
            try _close_lease!(owned) catch end
        end
        lease
    catch
        _close_lease!(lease);rethrow()
    end
end
function _assign!(lease,process)
    # Assign only the Process object just created by this owner. SET_QUOTA |
    # TERMINATE are required by AssignProcessToJobObject.
    handle=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x0101,0,UInt32(getpid(process)))
    handle!=C_NULL || _winerror("OpenProcess(owned child)")
    try
        ccall((:AssignProcessToJobObject,"kernel32"),Int32,(Ptr{Cvoid},Ptr{Cvoid}),lease.handle,handle)!=0 || _winerror("AssignProcessToJobObject")
    finally
        ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),handle)
    end
end
function _lease_count(lease)
    lease.handle==C_NULL && return 0
    info=zeros(UInt8,48)
    GC.@preserve info begin
        ccall((:QueryInformationJobObject,"kernel32"),Int32,(Ptr{Cvoid},Int32,Ptr{Cvoid},UInt32,Ptr{UInt32}),lease.handle,1,pointer(info),48,C_NULL)!=0 || _winerror("QueryInformationJobObject")
        Int(unsafe_load(Ptr{UInt32}(pointer(info)+40)))
    end
end
function _terminate!(lease)
    lease.handle==C_NULL && return
    ccall((:TerminateJobObject,"kernel32"),Int32,(Ptr{Cvoid},UInt32),lease.handle,1)!=0 || _winerror("TerminateJobObject(owned child)")
end

mutable struct ReplayJob
    job_id::String
    directory::String
    process::Base.Process
    worker_pid::Int
    snapshot::ExperimentRecord
    request::Dict{String,Any}
    stdout::IO
    stderr::IO
    lease::_WindowsLease
    owner_thread::Int
    pending::Any
    last_sequence::Int
    acknowledged_sequence::Int
    cancelled::Bool
    reaped::Bool
    delivered_terminal::Bool
    terminal::Union{Nothing,ReplayOutcome}
    fault::Union{Nothing,Symbol}
    fault_message::String
    observer_failure::Any
    shutdown_timeout::Float64
end
job_id(job)=job.job_id
pid(job)=job.worker_pid
outcome(job)=job.terminal
active(job)=!job.reaped
function _owner(job)
    threadid()==job.owner_thread || throw(ArgumentError("replay client must be serviced by its owning thread"))
end
function _guard_paths(snapshot,output,history)
    paths=String[f["path"] for f in snapshot.input_files];append!(paths,snapshot.record_paths)
    protected=Hammerhead._artifact_local_protected_paths(paths)
    any(p->Hammerhead._artifact_alias(output,p),protected) && throw(ArgumentError("worker output aliases input/record"))
    if history!==nothing
        append!(paths,[r.output for r in snapshot.runs]);protected=Hammerhead._artifact_local_protected_paths(paths)
        any(p->Hammerhead._artifact_alias(history,p),protected) && throw(ArgumentError("worker history aliases input/result"))
        Hammerhead._artifact_alias(output,history) && throw(ArgumentError("worker output/history must differ"))
    end
    nothing
end

"""Capture a builtin saved-planar request and start one owned, hidden core worker.
No application Observable or Qt/GL object is touched. `worker_script` is a test
seam, not a script-preprocessing option. Unsupported ownership platforms refuse
before spawning. Parent applies each progress event then explicitly acknowledges
it; one event is outstanding. Evidence/control files are retained.
"""
function start_replay(record::ExperimentRecord;output::AbstractString,run_record=nothing,
        allow_environment_change::Bool=false,initial_cancel::Bool=false,
        artifact_root=joinpath(@__DIR__,"artifacts"),ack_timeout::Real=60.,shutdown_timeout::Real=60.,
        worker_script::AbstractString=joinpath(@__DIR__,"replay_worker.jl"))
    _incomplete_startup[]===nothing || throw(ErrorException("previous worker startup cleanup is unverified; refuse further launches"))
    _owned_job[]===nothing || throw(ArgumentError("one owned replay is already active; poll/join it before another launch"))
    isfinite(ack_timeout) && ack_timeout>0 && isfinite(shutdown_timeout) && shutdown_timeout>0 || throw(ArgumentError("worker deadlines must be finite positive"))
    ack_seconds,shutdown_seconds=try
        Float64(ack_timeout),Float64(shutdown_timeout)
    catch
        throw(ArgumentError("worker deadlines must fit finite positive Float64 seconds"))
    end
    isfinite(ack_seconds) && ack_seconds>0 && isfinite(shutdown_seconds) && shutdown_seconds>0 ||
        throw(ArgumentError("worker deadlines must fit finite positive Float64 seconds"))
    snapshot=deepcopy(record);Hammerhead._experiment_preflight(snapshot)
    snapshot.recipe.external_preprocess===nothing || throw(ArgumentError("Qt worker does not execute referenced scripts"))
    snapshot.recipe.backend in (:cpu,:ka) || throw(ArgumentError("Qt worker supports builtin CPU/KA only"))
    path=Hammerhead._artifact_local_path(output)
    history=run_record===nothing || isempty(run_record) ? nothing : Hammerhead._artifact_local_path(run_record)
    _guard_paths(snapshot,path,history)
    project=Base.active_project();project===nothing && throw(ArgumentError("worker requires an active project"))
    script=realpath(worker_script);lease=_new_lease()
    directory="";process=nothing;stdout_io=nothing;stderr_io=nothing
    try
        mkpath(artifact_root);directory=mktempdir(artifact_root;prefix="worker-",cleanup=false)
        captured_paths=copy(snapshot.record_paths)
        record_file=joinpath(directory,"request-record.jld2");save_experiment(record_file,snapshot)
        request=Dict{String,Any}("version"=>WorkerProtocol.VERSION,"job_id"=>string(uuid4()),"owner_pid"=>Int(getpid()),
            "record_file"=>abspath(record_file),"record_sha256"=>WorkerProtocol.digest(record_file),
            "recipe_id"=>recipe_identity(snapshot.recipe),"input_id"=>snapshot.input_id,"total"=>length(snapshot.pairs),
            "output"=>path,"run_record"=>something(history,""),"allow_environment_change"=>allow_environment_change,
            "ack_timeout"=>ack_seconds,"project"=>abspath(project),"threads"=>Threads.nthreads(),
            "fftw_threads"=>Int(Hammerhead.FFTW.get_num_threads()),"protected_record_paths"=>captured_paths,
            "worker_sources"=>Dict(realpath(p)=>WorkerProtocol.digest(p) for p in (script,@__FILE__,joinpath(@__DIR__,"worker_protocol.jl"),joinpath(@__DIR__,"replay_worker.jl"))))
        WorkerProtocol.request_data(request);request_file=joinpath(directory,"request.toml")
        WorkerProtocol.write_control(request_file,request)
        initial_cancel && WorkerProtocol.write_control(joinpath(directory,"cancel.toml"),Dict("version"=>1,"job_id"=>request["job_id"],"cancel"=>true))
        stdout_io=open(joinpath(directory,"stdout.log"),"w");stderr_io=open(joinpath(directory,"stderr.log"),"w")
        # Inherit the same numerical thread pools/project; no environment activation.
        interactive=Threads.nthreads(:interactive)
        pools=interactive==0 ? string(Threads.nthreads(:default)) : "$(Threads.nthreads(:default)),$interactive"
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(project)) --threads=$pools $script $request_file`
        cmd=Cmd(cmd;windows_hide=true)
        process=run(pipeline(cmd;stdout=stdout_io,stderr=stderr_io);wait=false)
        _assign!(lease,process)
        worker_pid=Int(getpid(process))
        owner=Dict("version"=>1,"job_id"=>request["job_id"],"owner_pid"=>Int(getpid()),"worker_pid"=>worker_pid,
            "ownership"=>"windows_kill_on_close_job","directory"=>abspath(directory),"request_sha256"=>WorkerProtocol.digest(request_file))
        WorkerProtocol.write_control(joinpath(directory,"owner.toml"),owner)
        # Child waits for this enrollment before importing Hammerhead/processing.
        WorkerProtocol.write_control(joinpath(directory,"enrolled.toml"),owner)
        job=ReplayJob(request["job_id"],abspath(directory),process,worker_pid,snapshot,request,stdout_io,stderr_io,lease,threadid(),
            nothing,0,0,initial_cancel,false,false,nothing,nothing,"",nothing,shutdown_seconds)
        _owned_job[]=job
        job
    catch primary
        try _terminate!(lease) catch end
        if process!==nothing
            # Startup failure owns this child exclusively; no subsequent launch
            # happens while its native job still has members.
            process_exited(process) || try kill(process) catch end
            confirmed=timedwait(()->process_exited(process) && _lease_count(lease)==0,10.;pollint=.02)==:ok
            if !confirmed
                _incomplete_startup[]=(;process,lease,stdout_io,stderr_io,directory,owner_thread=threadid())
                try WorkerProtocol.write_control(joinpath(directory,"incomplete_owner.toml"),Dict("version"=>1,"owner_pid"=>Int(getpid()),"message"=>"startup child exit remains unverified")) catch end
                throw(ErrorException("worker startup failed and owned child cleanup remains unverified: $(sprint(showerror,primary))"))
            end
            process_exited(process) && try wait(process) catch end
        end
        stdout_io===nothing || close(stdout_io);stderr_io===nothing || close(stderr_io)
        _close_lease!(lease);rethrow()
    end
end

function request_cancel!(job)
    _owner(job);job.reaped && return job
    if !job.cancelled
        WorkerProtocol.write_control(joinpath(job.directory,"cancel.toml"),Dict("version"=>1,"job_id"=>job.job_id,"cancel"=>true))
        job.cancelled=true
    end
    job
end
function acknowledge_progress!(job,event;abort=nothing)
    _owner(job)
    job.pending!==nothing && isequal(event,job.pending) || throw(ArgumentError("no matching pending worker progress event"))
    abort===nothing || abort isa Exception || abort isa CapturedException || throw(ArgumentError("abort must be an exception"))
    packet=WorkerProtocol.acknowledgement(job.job_id,event.sequence;abort)
    WorkerProtocol.write_control(joinpath(job.directory,"ack-$(event.sequence).toml"),packet)
    abort===nothing || (job.observer_failure=abort)
    job.acknowledged_sequence=event.sequence
    job.pending=nothing
    job
end
function _fault!(job,status,message)
    job.fault===nothing || return
    job.fault=status;job.fault_message=String(message)
    _terminate!(job.lease)
end
function _finish!(job)
    process_exited(job.process) || return false
    _lease_count(job.lease)==0 || return false
    try wait(job.process) catch end
    job.reaped=true;close(job.stdout);close(job.stderr)
    code=Int(job.process.exitcode)
    data=nothing;run=nothing;status=something(job.fault,:crashed);message=job.fault_message
    type="WorkerProcessFailure";trace="";history_error="";cleanup=false
    if job.fault===nothing
        try
            terminal=joinpath(job.directory,"terminal.jld2")
            isfile(terminal) && !islink(terminal) && filesize(terminal)<=WorkerProtocol.MAX_TERMINAL_BYTES || throw(ArgumentError("missing/oversized worker terminal metadata"))
            data=Hammerhead.jldopen(f->f["terminal"],terminal,"r")
            WorkerProtocol.terminal_data(data,job.job_id,job.request["total"])
            data["worker_pid"]===job.worker_pid || throw(ArgumentError("crossed worker PID"))
            job.acknowledged_sequence<=data["written"]<=job.acknowledged_sequence+1 ||
                throw(ArgumentError("terminal prefix exceeds acknowledged progress"))
            data["status"]=="completed" && (job.acknowledged_sequence!=job.request["total"] || job.pending!==nothing) &&
                throw(ArgumentError("completed worker lacks all progress acknowledgements"))
            expected=data["status"]=="failed" ? 1 : 0
            code==expected || throw(ArgumentError("worker terminal disagrees with OS exit"))
            run=data["run"]===nothing ? nothing : Hammerhead._experiment_run(data["run"],job.snapshot)
            if run!==nothing
                run.output==job.request["output"] && run.completed_pairs==data["written"] &&
                    !(run.run_id in (r.run_id for r in job.snapshot.runs)) || throw(ArgumentError("recovered run disagrees with captured job"))
                (run.status===:completed)==(data["status"]=="completed") || throw(ArgumentError("core run and worker terminal status disagree"))
            end
            status=Symbol(data["status"]);type=data["error_type"];message=data["message"];trace=data["backtrace"]
            history_error=data["history_error"];cleanup=true
        catch error
            run=nothing
            status=isfile(joinpath(job.directory,"terminal.jld2")) ? :protocol_failed : :crashed
            message=sprint(showerror,error);type=string(typeof(error));trace=sprint(showerror,error,catch_backtrace())
        end
    end
    _close_lease!(job.lease)
    _owned_job[]===job && (_owned_job[]=nothing)
    job.terminal=ReplayOutcome(status,cleanup ? data["written"] : job.last_sequence,job.request["total"],run,
        type,message,trace,history_error,code,cleanup,job.job_id,job.directory)
    true
end

"""Read at most one bounded progress packet. Terminal is delivered only after
OS exit, wait/reap and zero native job members. File reads do not wait for child
work. Missing/corrupt terminal or unexpected exit never implies completion.
"""
function poll!(job)
    _owner(job)
    if !job.reaped
        _finish!(job)
        if !job.reaped && job.fault===nothing && job.pending===nothing
            path=joinpath(job.directory,"progress.toml")
            if isfile(path)
                try
                    event=WorkerProtocol.progress_data(WorkerProtocol.read_control(path),job.job_id,job.request["total"])
                    if event.sequence>job.last_sequence
                        event.sequence==job.last_sequence+1 || throw(ArgumentError("worker progress skipped a written boundary"))
                        job.last_sequence=event.sequence;job.pending=event;return event
                    elseif event.sequence<job.last_sequence
                        throw(ArgumentError("worker progress moved backwards"))
                    end
                catch error
                    _fault!(job,:protocol_failed,sprint(showerror,error))
                end
            end
        end
    end
    if job.reaped && !job.delivered_terminal
        job.delivered_terminal=true
        return (kind=:terminal,job_id=job.job_id,outcome=job.terminal)
    end
    nothing
end

"""Cancel and join the owned child with a finite deadline. This explicit teardown
helper acknowledges remaining written boundaries; ordinary UI polling must apply
its observer before acknowledgement. Deadline termination yields :timeout with
cleanup_confirmed=false, never a valid completed output claim.
"""
function shutdown!(job;timeout=job.shutdown_timeout)
    _owner(job);isfinite(timeout) && timeout>=0 || throw(ArgumentError("invalid shutdown deadline"))
    request_cancel!(job);deadline=time()+timeout
    while active(job) && time()<deadline
        event=poll!(job)
        job.pending===nothing || acknowledge_progress!(job,job.pending)
        sleep(.01)
    end
    if active(job)
        _fault!(job,:timeout,"worker exceeded cooperative shutdown deadline; terminated owned job")
        timedwait(()->begin _finish!(job);!active(job) end,10.;pollint=.02)==:ok || throw(ErrorException("owned worker termination remains unverified; refuse further launches"))
    end
    outcome(job)
end
end
