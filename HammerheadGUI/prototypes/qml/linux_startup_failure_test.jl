using Test,Hammerhead
include("worker_client.jl")
include("experiment_fixture.jl")
const C=ReplayWorkerClient
C.LinuxOwnership.subreaper!() # Independent fixture audit, including failure-only orphan adoption.
const seen=Ref{Any}(nothing)
# More-specific method only in this isolated owner process. The original captures
# the guardian Process; then a parent startup fault happens before first poll.
@eval C function _assign!(lease::_LinuxLease,process::Base.Process)
    invoke(_assign!,Tuple{_LinuxLease,Any},lease,process)
    Main.seen[]=(lease=lease,process=process)
    throw(ErrorException("original injected post-spawn startup failure"))
end
mkpath(joinpath(@__DIR__,"artifacts"))
const ROOT=mktempdir(joinpath(@__DIR__,"artifacts");prefix="linux-startup-failure-",cleanup=false)
println("LINUX_STARTUP_FAILURE_ARTIFACTS=",ROOT)
fixture=experiment_fixture(joinpath(ROOT,"inputs"))
write(fixture.output,"original destination")
@testset "Parent startup fault preserves guardian to prove subtree cleanup" begin
    primary=try
        C.start_replay(fixture.record;output=fixture.output,artifact_root=joinpath(ROOT,"jobs"))
        nothing
    catch error
        error
    end
    @test primary isa ErrorException
    @test sprint(showerror,primary)=="original injected post-spawn startup failure"
    @test seen[]!==nothing
    lease=seen[].lease
    @test process_exited(seen[].process)
    @test seen[].process.exitcode==0 # Guardian survived, stopped root, proved/reaped, then exited normally.
    @test lease.proof!==nothing && lease.proof["root_reaped"] && lease.proof["children_echild"] && lease.proof["group_empty"]
    @test lease.proof["stopped_by_owner"]
    @test !isfile(joinpath(lease.directory,"enrolled.toml"))
    @test !isfile(joinpath(lease.directory,"terminal.jld2"))
    @test read(fixture.output,String)=="original destination"
    @test !C.startup_cleanup_pending() && C._owned_job[]===nothing
    @test lease.guardian_fd==-1 && lease.worker_fd==-1
    @test timedwait(C.LinuxOwnership.no_children,5.;pollint=.01)==:ok
end
