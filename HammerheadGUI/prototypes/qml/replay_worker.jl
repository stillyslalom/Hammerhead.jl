# Core-only entry point. Enrollment precedes importing Hammerhead, so an
# assignment/startup failure cannot load pixels or open numerical destinations.
isdefined(@__MODULE__,:WorkerProtocol) || include("worker_protocol.jl")
using .WorkerProtocol
length(ARGS)==1 || error("replay worker expects one captured request file")
const request_file=abspath(only(ARGS))
const directory=dirname(request_file)
const request=WorkerProtocol.request_data(WorkerProtocol.read_control(request_file))
if Sys.islinux()
    ccall(:setpgid,Cint,(Cint,Cint),0,0)==0 || error("worker cannot create private Linux group")
    WorkerProtocol.write_control(joinpath(directory,"linux_worker_group.toml"),Dict("version"=>1,"job_id"=>request["job_id"],
        "worker_pid"=>Int(getpid()),"pgid"=>Int(ccall(:getpgrp,Cint,()))))
end
const enrollment=joinpath(directory,"enrolled.toml")
const enrollment_deadline=time()+10.
while !isfile(enrollment)
    time()<enrollment_deadline || error("owner failed to enroll worker; no replay started")
    sleep(.01)
end
const enrolled=WorkerProtocol.read_control(enrollment)
WorkerProtocol.check_keys(enrolled,("version","job_id","owner_pid","worker_pid","ownership","directory","request_sha256"))
enrolled["version"]===1 && enrolled["job_id"]==request["job_id"] && enrolled["owner_pid"]===request["owner_pid"] &&
    enrolled["worker_pid"]===Int(getpid()) && enrolled["directory"]==directory &&
    enrolled["ownership"]==(Sys.islinux() ? "linux_pidfd_subreaper_guardian" : "windows_kill_on_close_job") && enrolled["request_sha256"]==WorkerProtocol.digest(request_file) || error("worker enrollment identity disagrees")
(Sys.iswindows() || Sys.islinux()) && Sys.WORD_SIZE==64 || error("worker ownership unavailable on this host")
for (path,hash) in request["worker_sources"]
    WorkerProtocol.digest(path)==hash || error("worker source changed before execution")
end
WorkerProtocol.digest(request["record_file"])==request["record_sha256"] || error("captured record file changed")

using Hammerhead
Hammerhead.FFTW.set_num_threads(request["fftw_threads"])
Threads.nthreads()==request["threads"] && Base.active_project()==request["project"] || error("worker project/numerical threads disagree with captured policy")

struct WorkerCancelled <: Exception end
Base.showerror(io::IO,::WorkerCancelled)=print(io,"saved replay cancelled at a written-pair boundary")
struct WorkerProgressObserverError <: Exception
    error_type::String
    message::String
    backtrace::String
end
Base.showerror(io::IO,e::WorkerProgressObserverError)=print(io,"parent progress observer failed (",e.error_type,"): ",e.message)
struct WorkerAcknowledgementTimeout <: Exception end
Base.showerror(io::IO,::WorkerAcknowledgementTimeout)=print(io,"owner acknowledgement deadline expired; replay stopped before next write")

function cancellation_requested()
    path=joinpath(directory,"cancel.toml")
    isfile(path) && WorkerProtocol.cancellation_data(WorkerProtocol.read_control(path),request["job_id"])
end
function await_acknowledgement(sequence)
    path=joinpath(directory,"ack-$sequence.toml");deadline=time()+request["ack_timeout"]
    while !isfile(path)
        time()<deadline || throw(WorkerAcknowledgementTimeout())
        sleep(.01)
    end
    ack=WorkerProtocol.acknowledgement_data(WorkerProtocol.read_control(path),request["job_id"],sequence)
    ack["action"]=="abort" && throw(WorkerProgressObserverError(ack["error_type"],ack["message"],ack["backtrace"]))
    nothing
end
function recovered_run(record,path,written)
    isempty(path) && return nothing,""
    try
        saved=load_experiment(path)
        recipe_identity(saved.recipe)==recipe_identity(record.recipe) && saved.input_id==record.input_id || error("history belongs to another request")
        length(saved.runs)==length(record.runs)+1 || error("history does not contain exactly one new core run")
        all(i->Hammerhead._experiment_digest(Hammerhead._experiment_run_data(saved.runs[i]))==
            Hammerhead._experiment_digest(Hammerhead._experiment_run_data(record.runs[i])),eachindex(record.runs)) || error("previous history changed")
        run=last(saved.runs)
        !(run.run_id in (r.run_id for r in record.runs)) && run.output==request["output"] && run.completed_pairs==written || error("new history run disagrees with progress/output")
        run.status===:failed || error("failure recovery cannot invent completed run")
        run,nothing
    catch secondary
        nothing,WorkerProtocol.bounded_text(sprint(showerror,secondary))
    end
end
function write_terminal(data)
    WorkerProtocol.terminal_data(data,request["job_id"],request["total"])
    staging=joinpath(directory,"terminal.$(WorkerProtocol.UUIDs.uuid4()).partial")
    try
        Hammerhead.jldopen(staging,"w") do file;file["terminal"]=data;end
        filesize(staging)<=WorkerProtocol.MAX_TERMINAL_BYTES || error("terminal metadata exceeds bounded size")
        WorkerProtocol.native_replace(staging,joinpath(directory,"terminal.jld2"))
    finally
        try isfile(staging) && rm(staging;force=true) catch secondary
            @error "Failed to remove worker terminal staging" exception=secondary
        end
    end
end

function main()
    record=load_experiment(request["record_file"])
    append!(record.record_paths,request["protected_record_paths"]);unique!(record.record_paths)
    recipe_identity(record.recipe)==request["recipe_id"] && record.input_id==request["input_id"] &&
        length(record.pairs)==request["total"] || error("captured request/record identity disagrees")
    record.recipe.external_preprocess===nothing && record.recipe.backend in (:cpu,:ka) || error("unsupported worker recipe")
    history=isempty(request["run_record"]) ? nothing : request["run_record"]
    written=Ref(0);run=nothing;status=:completed;failure=(error_type="",message="",backtrace="");history_error=""
    try
        cancellation_requested() && throw(WorkerCancelled())
        progress=(i,n)->begin
            i==written[]+1 && n==request["total"] || error("unexpected core progress order")
            written[]=i
            WorkerProtocol.write_control(joinpath(directory,"progress.toml"),Dict("version"=>1,"job_id"=>request["job_id"],"sequence"=>i,"written"=>i,"total"=>n))
            await_acknowledgement(i)
            # Observer abort was handled first, including after the final write.
            cancellation_requested() && i<n && throw(WorkerCancelled())
            nothing
        end
        run=replay_experiment(record;output=request["output"],run_record=history,
            allow_environment_change=request["allow_environment_change"],progress)
        if history!==nothing
            try
                saved=load_experiment(history)
                last(saved.runs).run_id==run.run_id || error("returned completed run not present in requested history")
            catch secondary
                history_error=WorkerProtocol.bounded_text(sprint(showerror,secondary))
            end
        end
    catch exception
        status=exception isa WorkerCancelled ? :cancelled : :failed
        failure=exception isa WorkerProgressObserverError ?
            (error_type=exception.error_type,message=exception.message,backtrace=exception.backtrace) : WorkerProtocol.error_data(exception,catch_backtrace())
        showerror(stderr,exception,catch_backtrace());println(stderr);flush(stderr)
        if written[]>0 || status===:failed
            recovered,secondary=recovered_run(record,something(history,""),written[])
            run=recovered;history_error=something(secondary,"")
        end
    end
    # Core has returned or rethrown through its joined-loader/output cleanup.
    for (path,hash) in request["worker_sources"]
        WorkerProtocol.digest(path)==hash || error("worker source changed during replay")
    end
    terminal=Dict{String,Any}("version"=>1,"job_id"=>request["job_id"],"status"=>String(status),
        "written"=>written[],"total"=>request["total"],"run"=>run===nothing ? nothing : Hammerhead._experiment_run_data(run),
        (String(k)=>v for (k,v) in pairs(failure))...,"history_error"=>history_error,
        "core_cleanup_returned"=>true,"worker_pid"=>Int(getpid()),
        "loaded_modules"=>sort!(unique!(String[string(nameof(m)) for m in values(Base.loaded_modules)])))
    write_terminal(terminal)
    status===:failed ? 1 : 0
end
exit(main())
