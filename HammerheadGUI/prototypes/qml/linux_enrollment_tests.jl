using Test,Hammerhead
include("worker_client.jl")
using .ReplayWorkerClient
include("experiment_fixture.jl")
const C=ReplayWorkerClient
mkpath(joinpath(@__DIR__,"artifacts"))
const ROOT=mktempdir(joinpath(@__DIR__,"artifacts");prefix="linux-enrollment-",cleanup=false)
println("LINUX_ENROLLMENT_ARTIFACTS=",ROOT)
fixture=experiment_fixture(joinpath(ROOT,"inputs"))
function service_until(job,predicate;timeout=30.)
    timedwait(()->begin poll!(job);predicate() end,timeout;pollint=.01)==:ok || error("bounded client service timeout")
end
@testset "Transferred enrollment and forced shutdown proof" begin
    script=joinpath(ROOT,"waiting_worker.jl")
    write(script,"r=WorkerProtocol.request_data(WorkerProtocol.read_control(only(ARGS)))\nd=dirname(only(ARGS))\nwhile !isfile(joinpath(d,\"enrolled.toml\")); sleep(.01);end\nWorkerProtocol.write_control(joinpath(d,\"fixture_ready.toml\"),Dict(\"pid\"=>Int(getpid()),\"job_id\"=>r[\"job_id\"]))\nsleep(90.)\n")
    output=joinpath(ROOT,"protected-output.jld2");write(output,"preserved")
    job=start_replay(fixture.record;output,artifact_root=joinpath(ROOT,"jobs"),worker_script=script)
    @test pid(job)==0 # No guardian PID is presented as the scientific worker.
    try
        service_until(job,()->isfile(joinpath(job.directory,"fixture_ready.toml")))
        @test pid(job)>0 && pid(job)!=job.lease.guardian_pid
        @test job.lease.worker_fd>=0 && job.lease.guardian_fd>=0
        @test C.LinuxOwnership.descriptor_pid(job.lease.worker_fd)==pid(job)
        @test C.LinuxOwnership.descriptor_pid(job.lease.guardian_fd)==job.lease.guardian_pid
        result=shutdown!(job;timeout=0.)
        @test result.status===:timeout && !result.cleanup_confirmed
        @test job.lease.proof["worker_term_signal"]==9
        @test result.exit_code==128+job.lease.proof["worker_term_signal"]
        @test !active(job)
        @test job.lease.proof["children_echild"] && job.lease.proof["group_empty"]
        @test job.lease.proof["root_reaped"]
        @test read(output,String)=="preserved"
    finally
        active(job) && shutdown!(job;timeout=0.)
    end
end
@testset "Bootstrap deadline before enrollment retains queued reaped identity" begin
    script=joinpath(ROOT,"early_failure.jl");write(script,"error(\"intentional early enrollment fixture failure\")\n")
    job=start_replay(fixture.record;output=joinpath(ROOT,"not-written.jld2"),artifact_root=joinpath(ROOT,"jobs"),worker_script=script)
    # Deliberately leave both self-opened descriptors queued past child exit/reap.
    sleep(12.)
    try
        service_until(job,()->!active(job))
        result=outcome(job)
        @test result.status===:crashed && !result.cleanup_confirmed
        @test result.run===nothing && result.written==0
        @test result.exit_code==1
        @test job.lease.proof["children_echild"] && job.lease.proof["group_empty"]
        @test !isfile(joinpath(ROOT,"not-written.jld2"))
    finally
        active(job) && shutdown!(job;timeout=0.)
    end
end
