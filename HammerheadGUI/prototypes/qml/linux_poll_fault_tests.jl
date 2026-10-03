using Test,Hammerhead
include("worker_client.jl")
using .ReplayWorkerClient
include("experiment_fixture.jl")
const C=ReplayWorkerClient
const L=C.LinuxOwnership
L.subreaper!()
mkpath(joinpath(@__DIR__,"artifacts"))
const ROOT=mktempdir(joinpath(@__DIR__,"artifacts");prefix="linux-poll-fault-",cleanup=false)
println("LINUX_POLL_FAULT_ARTIFACTS=",ROOT)
const fixture=experiment_fixture(joinpath(ROOT,"inputs"))
@testset "Guardian OS status schema bounds" begin
    req=Dict("job_id"=>"test-job","owner_pid"=>Int(getpid()))
    good=Dict("version"=>1,"job_id"=>req["job_id"],"owner_pid"=>req["owner_pid"],
        "guardian_pid"=>123,"worker_pid"=>124,"request_sha256"=>"test-hash",
        "root_reaped"=>true,"group_empty"=>true,"children_echild"=>true,
        "worker_exit_code"=>255,"worker_term_signal"=>127,"cleanup_confirmed"=>true,
        "stopped_by_owner"=>true,"residual_group_killed"=>false)
    @test L.proof_data(good,req,123,124,"test-hash")===good
    for (key,value) in (("worker_exit_code",-1),("worker_exit_code",256),
                        ("worker_term_signal",-1),("worker_term_signal",128),
                        ("worker_term_signal",typemax(Int)),("worker_exit_code",true))
        bad=Dict{String,Any}(good);bad[key]=value
        @test_throws ErrorException L.proof_data(bad,req,123,124,"test-hash")
    end
end
@testset "Malformed enrollment is owner-visible and requests stop" begin
    script=joinpath(ROOT,"waiting.jl");write(script,"sleep(120.)\n")
    output=joinpath(ROOT,"preserved.jld2");write(output,"preserved output")
    job=start_replay(fixture.record;output,artifact_root=joinpath(ROOT,"jobs"),worker_script=script)
    ready_path=joinpath(job.directory,"guardian_ready.toml")
    try
        # No owner poll, so root remains blocked before any user fixture code.
        timedwait(()->isfile(ready_path),30.;pollint=.01)==:ok || error("guardian readiness timeout")
        good=C.WorkerProtocol.read_control(ready_path)
        bad=copy(good);bad["guardian_pid"]+=1
        C.WorkerProtocol.write_control(ready_path,bad)
        @test poll!(job)===nothing
        @test active(job) && outcome(job)===nothing
        @test job.fault===:protocol_failed
        reason=ownership_error(job)
        @test reason isa String && occursin("guardian enrollment",reason)
        @test isfile(joinpath(job.directory,"terminate.toml"))
        @test !isfile(joinpath(job.directory,"enrolled.toml"))
        @test read(output,String)=="preserved output"
        # Fixture-only restore permits independent cleanup acceptance. Production
        # never rewrites corruption to manufacture a successful scientific run.
        C.WorkerProtocol.write_control(ready_path,good)
        timedwait(()->begin poll!(job);!active(job) end,15.;pollint=.01)==:ok || error("fault cleanup timeout")
        @test outcome(job).status===:protocol_failed && !outcome(job).cleanup_confirmed
        @test occursin(reason,outcome(job).message)
        @test outcome(job).run===nothing
        @test process_exited(job.process) && job.lease.proof["children_echild"]
        @test ownership_error(job)===nothing
        @test L.no_children()
    finally
        active(job) && shutdown!(job;timeout=0.)
    end
end
