# Transport/ownership checks use explicitly injected core-only worker scripts.
# They do not substitute for actual replay parity or measured Qt interaction.
using Test,Hammerhead,TOML
include("worker_client.jl")
using .ReplayWorkerClient: start_replay,poll!,active,outcome,acknowledge_progress!,shutdown!
include("experiment_fixture.jl")

function injected_worker(path,body)
    protocol=repr(joinpath(@__DIR__,"worker_protocol.jl"))
    write(path,"include($protocol)\nusing .WorkerProtocol\nr=WorkerProtocol.request_data(WorkerProtocol.read_control(only(ARGS)))\nd=dirname(only(ARGS))\n"*
        "deadline=time()+30\nwhile !isfile(joinpath(d,\"enrolled.toml\"))\n time()<deadline || error(\"enrollment timed out\")\n sleep(.01)\nend\n"*body)
    path
end
function collect_transport(job;timeout=60.)
    deadline=time()+timeout
    while active(job) && time()<deadline
        poll!(job);sleep(.005)
    end
    active(job) && shutdown!(job;timeout=0.)
    active(job) && error("worker cleanup unverified; stop all further launches")
    outcome(job)
end

function prove_owner_loss(record_file,worker_script,root)
    announcement=joinpath(root,"owner-started.toml")
    command=`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $(joinpath(@__DIR__,"worker_owner_loss_child.jl")) $record_file $worker_script $root $announcement`
    command=Cmd(command;windows_hide=true)
    process=nothing;worker_handle=C_NULL;proof=Dict{String,Any}("scope"=>"injected enrolled worker wait; not mid-PIV interruption")
    open(joinpath(root,"owner_stdout.log"),"w") do out
        open(joinpath(root,"owner_stderr.log"),"w") do err
            try
                process=run(pipeline(command;stdout=out,stderr=err);wait=false)
                timedwait(()->isfile(announcement)||process_exited(process),90.;pollint=.02)==:ok || error("test owner startup timed out")
                isfile(announcement) || error("test owner exited before enrollment announcement")
                info=TOML.parsefile(announcement)
                info["owner_pid"]==Int(getpid(process)) || error("owner announcement PID mismatch")
                proof["owner_pid"]=info["owner_pid"];proof["worker_pid"]=info["worker_pid"];proof["job_id"]=info["job_id"]
                ready_file=joinpath(info["directory"],"injected_wait_ready.toml")
                timedwait(()->isfile(ready_file)||process_exited(process),60.;pollint=.02)==:ok || error("injected waiting boundary was not reached")
                isfile(ready_file) || error("test owner exited before worker wait boundary")
                ready=TOML.parsefile(ready_file)
                ready["job_id"]==info["job_id"] && ready["worker_pid"]===info["worker_pid"] || error("crossed worker wait identity")
                proof["injected_wait_acknowledged"]=true
                # Bind to the exact live process object before terminating the
                # test owner; no later PID lookup or process-name kill is used.
                worker_handle=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x00100000,0,UInt32(info["worker_pid"]))
                worker_handle!=C_NULL || error("cannot open owned worker synchronization handle")
                ccall((:WaitForSingleObject,"kernel32"),UInt32,(Ptr{Cvoid},UInt32),worker_handle,0)==0x00000102 || error("enrolled worker was not alive before owner loss")
                proof["worker_alive_before_owner_loss"]=true
                kill(process,Base.SIGKILL);wait(process)
                proof["owner_exit_code"]=Int(process.exitcode)
                signaled=ccall((:WaitForSingleObject,"kernel32"),UInt32,(Ptr{Cvoid},UInt32),worker_handle,15000)==0
                proof["worker_exit_verified"]=signaled
                signaled || error("worker survived owner loss; refuse subsequent launches")
                proof
            finally
                if process!==nothing && !process_exited(process)
                    kill(process,Base.SIGKILL);wait(process)
                end
                if worker_handle!=C_NULL
                    exited=ccall((:WaitForSingleObject,"kernel32"),UInt32,(Ptr{Cvoid},UInt32),worker_handle,15000)==0
                    proof["cleanup_worker_exit_verified"]=exited
                    ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),worker_handle)
                    worker_handle=C_NULL
                    exited || error("owned worker exit remains unverified; stop subsequent launches")
                end
                open(io->TOML.print(io,proof;sorted=true),joinpath(root,"owner_loss_proof.toml"),"w")
            end
        end
    end
    proof
end

function worker_client_checks(directory)
    fixture=experiment_fixture(joinpath(directory,"inputs"))
    artifacts=joinpath(directory,"jobs")
    if !(Sys.iswindows() && Sys.WORD_SIZE==64)
        @testset "Unsupported native ownership refuses before spawning" begin
            @test_throws ArgumentError start_replay(fixture.record;output=fixture.output,artifact_root=artifacts)
            @test !ispath(artifacts) && !ispath(fixture.output)
        end
        return
    end
    @testset "Unrepresentable deadlines refuse before spawning" begin
        protected_before=read(fixture.path)
        invalid_root=joinpath(directory,"invalid-deadlines")
        for key in (:ack_timeout,:shutdown_timeout),value in (big"1e1000",big"1e-1000")
            kwargs=Dict(key=>value)
            @test_throws ArgumentError start_replay(fixture.record;output=fixture.output,artifact_root=invalid_root,kwargs...)
            @test !ispath(invalid_root) && !ispath(fixture.output)
            @test read(fixture.path)==protected_before
        end
    end
    stalled=injected_worker(joinpath(directory,"stalled.jl"),
        "WorkerProtocol.write_control(joinpath(d,\"injected_wait_ready.toml\"),Dict(\"job_id\"=>r[\"job_id\"],\"worker_pid\"=>Int(getpid())))\nwhile true; sleep(.1); end\n")
    # Run first. Any unverified descendant stops the file before further work.
    lossdir=joinpath(directory,"owner-loss");mkdir(lossdir)
    proof=prove_owner_loss(fixture.path,stalled,lossdir)
    @testset "Abrupt owner death terminates the enrolled worker" begin
        @test proof["worker_alive_before_owner_loss"]===true
        @test proof["injected_wait_acknowledged"]===true
        @test proof["worker_exit_verified"]===true && proof["cleanup_worker_exit_verified"]===true
        @test proof["owner_exit_code"]!=0
    end
    @testset "Failed startup reaps an unassigned child before returning" begin
        startup_root=joinpath(directory,"startup-failure");mkdir(startup_root)
        command=`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $(joinpath(@__DIR__,"worker_startup_failure_child.jl")) $(fixture.path) $stalled $startup_root`
        command=Cmd(command;windows_hide=true)
        process=nothing
        open(joinpath(startup_root,"stdout.log"),"w") do out
            open(joinpath(startup_root,"stderr.log"),"w") do err
                try
                    process=run(pipeline(command;stdout=out,stderr=err);wait=false)
                    timedwait(()->process_exited(process),90.;pollint=.02)==:ok || error("startup-failure helper timed out; no further launches allowed")
                    wait(process)
                    @test process.exitcode==0
                    process.exitcode==0 || error("startup cleanup test failed; stop subsequent launches")
                finally
                    if process!==nothing && !process_exited(process)
                        kill(process,Base.SIGKILL);wait(process)
                    end
                end
            end
        end
        proof=TOML.parsefile(joinpath(startup_root,"startup_cleanup_proof.toml"))
        @test proof["cleanup_verified"]===true && proof["process_exit_verified"]===true
        @test !get(proof,"forced_test_cleanup",false)
    end
    @testset "Nonblocking bounded progress requires an exact explicit acknowledgement" begin
        script=injected_worker(joinpath(directory,"progress_wait.jl"),
            "WorkerProtocol.write_control(joinpath(d,\"progress.toml\"),Dict(\"version\"=>1,\"job_id\"=>r[\"job_id\"],\"sequence\"=>1,\"written\"=>1,\"total\"=>r[\"total\"]))\nwhile true; sleep(.1); end\n")
        job=start_replay(fixture.record;output=fixture.output,artifact_root=artifacts,worker_script=script)
        try
            event=Ref{Any}(nothing)
            @test timedwait(()->begin event[]=poll!(job);event[]!==nothing end,60.;pollint=.01)==:ok
            @test event[].kind===:progress && event[].written==1
            @test !isfile(joinpath(job.directory,"ack-1.toml"))
            @test all(_->poll!(job)===nothing,1:100)
            @test active(job) && outcome(job)===nothing
            stale=merge(event[],(job_id="stale-request",))
            @test_throws ArgumentError acknowledge_progress!(job,stale)
            @test !isfile(joinpath(job.directory,"ack-1.toml"))
            acknowledge_progress!(job,event[])
            @test isfile(joinpath(job.directory,"ack-1.toml"))
            @test_throws ArgumentError acknowledge_progress!(job,event[])
        finally
            result=shutdown!(job;timeout=0.)
            @test result.status===:timeout && !result.cleanup_confirmed && !active(job)
        end
    end
    @testset "Child exit and crossed metadata cannot fabricate completion" begin
        for mode in ("exit-zero","crossed-progress","oversized-progress","malformed-terminal")
            body=mode=="exit-zero" ? "exit(0)\n" : mode=="crossed-progress" ?
                "WorkerProtocol.write_control(joinpath(d,\"progress.toml\"),Dict(\"version\"=>1,\"job_id\"=>\"stale-request\",\"sequence\"=>1,\"written\"=>1,\"total\"=>r[\"total\"]))\nwhile true;sleep(.1);end\n" :
                mode=="oversized-progress" ? "write(joinpath(d,\"progress.toml\"),zeros(UInt8,WorkerProtocol.MAX_CONTROL_BYTES+1))\nwhile true;sleep(.1);end\n" :
                "using Hammerhead\nHammerhead.jldopen(joinpath(d,\"terminal.jld2\"),\"w\") do f; f[\"terminal\"]=Dict(\"version\"=>1); end\nexit(0)\n"
            script=injected_worker(joinpath(directory,"$mode.jl"),body)
            output=joinpath(directory,"$mode-output.jld2");write(output,"existing bytes")
            job=start_replay(fixture.record;output,artifact_root=artifacts,worker_script=script)
            result=try collect_transport(job) finally
                active(job) && shutdown!(job;timeout=0.)
            end
            @test result.status===(mode=="exit-zero" ? :crashed : :protocol_failed)
            @test result.run===nothing && !result.cleanup_confirmed && !active(job)
            @test read(output,String)=="existing bytes"
            if mode=="exit-zero"
                @test result.exit_code===0
            end
        end
    end
    @testset "Pending progress still permits authoritative child failure delivery" begin
        output=joinpath(directory,"ack-expired.jld2");history=joinpath(directory,"ack-expired-history.jld2")
        job=start_replay(fixture.record;output,run_record=history,artifact_root=artifacts,ack_timeout=1.)
        event=Ref{Any}(nothing)
        result=try
            @test timedwait(()->begin
                update=poll!(job)
                update!==nothing && update.kind===:progress && (event[]=update)
                event[]!==nothing
            end,90.;pollint=.01)==:ok
            @test event[].written==1 && !isfile(joinpath(job.directory,"ack-1.toml"))
            collect_transport(job;timeout=60.) # Deliberately never acknowledges.
        finally
            active(job) && shutdown!(job;timeout=0.)
        end
        @test result.status===:failed && result.exit_code===1 && result.cleanup_confirmed
        @test result.written==1 && result.run isa ExperimentRun && result.run.status===:failed
        @test occursin("acknowledgement deadline expired",result.message)
        @test length(load_results(output))==1 && !active(job)
    end
end

# Evidence survives Julia exit. No repository source is changed by fixtures.
mkpath(joinpath(@__DIR__,"artifacts"))
const worker_test_root=mktempdir(joinpath(@__DIR__,"artifacts");prefix="worker-client-checks-",cleanup=false)
println("Worker client evidence: ",worker_test_root);flush(stdout)
worker_client_checks(worker_test_root)
