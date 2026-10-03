# Actual core PIV through the subprocess boundary; no Qt/GLFW imports.
using Test,Hammerhead
include("worker_client.jl")
using .ReplayWorkerClient: start_replay,poll!,active,outcome,acknowledge_progress!,request_cancel!,shutdown!
include("experiment_fixture.jl")
Sys.iswindows() && Sys.WORD_SIZE==64 || error("saved-planar worker capability gate failed: validated Windows Job Object ownership is unavailable on this host")

function settle_worker(job;on_progress=event->nothing,timeout=180.)
    deadline=time()+timeout
    events=Any[]
    try
        while active(job)
            time()<deadline || error("test worker did not stop before deadline")
            event=poll!(job)
            if event!==nothing
                push!(events,event)
                if event.kind===:progress
                    abort=on_progress(event)
                    acknowledge_progress!(job,event;abort=abort isa Exception ? abort : nothing)
                end
            end
            sleep(.005)
        end
    catch
        # Even a broken assertion/fixture may not abandon its owned replay.
        try
            shutdown!(job;timeout=10.)
        catch cleanup_error
            @error "Test worker cleanup failed; further worker launches must stop" exception=cleanup_error
        end
        rethrow()
    end
    # Reaping must expose validated terminal truth, not an assumed completion.
    result=outcome(job)
    result===nothing && error("reaped worker has no outcome")
    result,events
end

@testset "Subprocess saved-planar replay preserves scientific fields and snapshot" begin
    mktempdir() do directory
        fixture=experiment_fixture(joinpath(directory,"inputs"))
        direct_path=joinpath(directory,"direct.jld2")
        direct=replay_experiment(fixture.record;output=direct_path)
        request=deepcopy(fixture.record)
        job=start_replay(request;output=fixture.output,run_record=fixture.history,
            artifact_root=joinpath(directory,"jobs"))
        # The caller's mutable object changes after submission. The child must
        # consume its detached saved request rather than this later mutation.
        request.recipe.mask[20,20]=!request.recipe.mask[20,20]
        request.pairs[1]=reverse(request.pairs[1])
        result,events=settle_worker(job)
        @test result.status===:completed && result.run isa ExperimentRun
        @test result.exit_code===0 && result.cleanup_confirmed
        @test result.written==result.total==3 && result.run.status===:completed
        @test result.run.recipe_id==fixture.record.recipe.recipe_id && result.run.input_id==fixture.record.input_id
        @test [e.written for e in events if e.kind===:progress]==[1,2,3]
        @test all(e->e.job_id==result.job_id,events)
        native=load_results(fixture.output);reference=load_results(direct_path)
        @test length(native)==length(reference)==direct.completed_pairs
        for (actual,expected) in zip(native,reference),field in fieldnames(typeof(expected))
            @test isequal(getfield(actual,field),getfield(expected,field))
        end
        @test last(load_experiment(fixture.history).runs).run_id==result.run.run_id
        @test !active(job)
        @test shutdown!(job;timeout=5.).job_id==result.job_id
    end
end

@testset "Subprocess cancellation retains ordinary failed-prefix semantics" begin
    mktempdir() do directory
        fixture=experiment_fixture(joinpath(directory,"inputs"))
        for boundary in (0,1,3)
            output=joinpath(directory,"cancel-$boundary.jld2")
            history=joinpath(directory,"cancel-$boundary-record.jld2")
            if boundary==0
                write(output,"previous destination bytes")
            end
            job=start_replay(fixture.record;output,run_record=history,
                artifact_root=joinpath(directory,"jobs"),initial_cancel=boundary==0)
            boundary==0 && request_cancel!(job)
            result,events=settle_worker(job;on_progress=e->begin
                e.written==boundary && request_cancel!(job)
                nothing
            end)
            @test result.status===(boundary==3 ? :completed : :cancelled)
            @test result.exit_code===0 && result.cleanup_confirmed
            @test result.written==boundary
            if boundary==0
                @test read(output,String)=="previous destination bytes"
                @test result.run===nothing
            else
                @test result.run isa ExperimentRun && result.run.status===(boundary==3 ? :completed : :failed)
                @test result.run.completed_pairs==boundary
                @test length(load_results(output))==boundary
                @test last(load_experiment(history).runs).run_id==result.run.run_id
            end
            @test !active(job)
        end
    end
end

@testset "Worker callback and history failures do not invent completion" begin
    mktempdir() do directory
        fixture=experiment_fixture(joinpath(directory,"inputs"))
        output=joinpath(directory,"original-failure.jld2")
        history=joinpath(directory,"original-failure-record.jld2")
        fault=ErrorException("distinct original progress failure")
        job=start_replay(fixture.record;output,run_record=history,artifact_root=joinpath(directory,"jobs"))
        result,_=settle_worker(job;on_progress=e->e.written==1 ? fault : nothing)
        @test result.status===:failed
        @test occursin("distinct original progress failure",result.message)
        @test result.run isa ExperimentRun && result.run.status===:failed && result.run.completed_pairs==1
        @test length(load_results(output))==1
        @test result.exit_code===1 && result.cleanup_confirmed

        output=joinpath(directory,"final-callback-failure.jld2")
        history=joinpath(directory,"final-callback-failure-record.jld2")
        job=start_replay(fixture.record;output,run_record=history,artifact_root=joinpath(directory,"jobs"))
        result,_=settle_worker(job;on_progress=e->e.written==e.total ? fault : nothing)
        @test result.status===:failed && result.written==result.total==3
        @test result.run isa ExperimentRun && result.run.status===:failed
        @test occursin("distinct original progress failure",result.message)
        @test length(load_results(output))==3
        @test result.exit_code===1 && result.cleanup_confirmed

        bad_history=joinpath(directory,"history-is-directory");mkdir(bad_history)
        output=joinpath(directory,"history-failure-output.jld2")
        job=start_replay(fixture.record;output,run_record=bad_history,artifact_root=joinpath(directory,"jobs"))
        result,_=settle_worker(job)
        @test result.status===:failed && result.written==result.total==3
        @test result.run===nothing # Output bytes/total writes cannot fabricate a run.
        @test length(load_results(output))==3
        @test isdir(bad_history) && isempty(readdir(bad_history))
        @test result.exit_code===1 && result.cleanup_confirmed

        output=joinpath(directory,"failure-plus-history-output.jld2")
        job=start_replay(fixture.record;output,run_record=bad_history,artifact_root=joinpath(directory,"jobs"))
        result,_=settle_worker(job;on_progress=e->e.written==1 ? fault : nothing)
        @test result.status===:failed && occursin("distinct original progress failure",result.message)
        @test result.run===nothing
        @test length(load_results(output))==1
        @test result.exit_code===1 && result.cleanup_confirmed
    end
end
