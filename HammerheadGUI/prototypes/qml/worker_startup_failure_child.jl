# Method injection exists only in this isolated test process.
using Test,Hammerhead,TOML
include("worker_client.jl")
using .ReplayWorkerClient
const captured_process=Ref{Union{Nothing,Base.Process}}(nothing)
const captured_pid=Ref(0)
function ReplayWorkerClient._assign!(lease::ReplayWorkerClient._WindowsLease,process::Base.Process)
    captured_process[]=process
    captured_pid[]=Int(getpid(process))
    throw(ErrorException("injected enrollment assignment failure"))
end
record_file,worker_script,root=ARGS
output=joinpath(root,"prior-native.jld2");write(output,"untouched prior destination")
proof=Dict{String,Any}("scope"=>"assignment failure before worker enrollment/processing")
try
    failure=try
        start_replay(load_experiment(record_file);output,worker_script,artifact_root=root)
        nothing
    catch exception
        exception
    end
    @test failure isa ErrorException && occursin("injected enrollment assignment failure",sprint(showerror,failure))
    @test captured_process[]!==nothing
    process=captured_process[]
    proof["worker_pid"]=captured_pid[]
    confirmed=timedwait(()->begin
        poll_startup_cleanup!()
        process_exited(process) && !startup_cleanup_pending()
    end,15.;pollint=.02)==:ok
    @test confirmed
    confirmed || error("startup cleanup remains unverified; stop later launches")
    wait(process)
    proof["exit_code"]=Int(process.exitcode);proof["cleanup_verified"]=true
    @test !startup_cleanup_pending()
    @test read(output,String)=="untouched prior destination"
    @test !any(directory->isfile(joinpath(directory,"enrolled.toml")),filter(isdir,readdir(root;join=true)))
    @test !any(directory->isfile(joinpath(directory,"incomplete_owner.toml")),filter(isdir,readdir(root;join=true)))
finally
    process=captured_process[]
    if process!==nothing && !process_exited(process)
        kill(process,Base.SIGKILL);wait(process)
        proof["forced_test_cleanup"]=true
    end
    process===nothing || (proof["process_exit_verified"]=process_exited(process))
    open(io->TOML.print(io,proof;sorted=true),joinpath(root,"startup_cleanup_proof.toml"),"w")
end
