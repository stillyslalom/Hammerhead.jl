using Test,TOML
include("worker_protocol.jl")
const WP=WorkerProtocol

@testset "Replay protocol rejects stale identity and malformed scalar counts" begin
    id="4f57b25c-3240-4faf-9e20-d3b9a47cbe99"
    stale="3d1000e4-60c6-4643-bc73-725a1f866e90"
    request=Dict{String,Any}("version"=>1,"job_id"=>id,"owner_pid"=>Int(getpid()),
        "record_file"=>"captured.jld2","record_sha256"=>repeat("a",64),"recipe_id"=>repeat("b",64),
        "input_id"=>repeat("c",64),"total"=>3,"output"=>"result.jld2","run_record"=>"",
        "allow_environment_change"=>false,"ack_timeout"=>30.,"project"=>"Project.toml",
        "threads"=>1,"fftw_threads"=>1,"protected_record_paths"=>String[],
        "worker_sources"=>Dict("replay_worker.jl"=>repeat("d",64)))
    @test WP.request_data(request)===request
    for (key,value) in (("version",true),("version",2),("job_id","invalid"),("total",true),
                        ("total",3.),("total",0),("owner_pid",-1),("threads",false),
                        ("ack_timeout",Inf),("ack_timeout",0.),("ack_timeout",30),
                        ("allow_environment_change",0),("record_sha256","bad"),
                        ("worker_sources",Dict("worker"=>"bad")))
        candidate=deepcopy(request);candidate[key]=value
        @test_throws ArgumentError WP.request_data(candidate)
    end
    candidate=deepcopy(request);candidate["unexpected"]=1
    @test_throws ArgumentError WP.request_data(candidate)

    progress=Dict{String,Any}("version"=>1,"job_id"=>id,"sequence"=>1,"written"=>1,"total"=>3)
    @test WP.progress_data(progress,id,3)==(kind=:progress,job_id=id,sequence=1,written=1,total=3)
    for (key,value) in (("job_id",stale),("version",true),("total",3.),("sequence",true),
                        ("written",2),("sequence",0),("sequence",4))
        candidate=deepcopy(progress);candidate[key]=value
        @test_throws ArgumentError WP.progress_data(candidate,id,3)
    end
    ack=WP.acknowledgement(id,1)
    @test WP.acknowledgement_data(ack,id,1)===ack
    @test_throws ArgumentError WP.acknowledgement_data(ack,stale,1)
    @test_throws ArgumentError WP.acknowledgement_data(ack,id,2)
    candidate=deepcopy(ack);candidate["sequence"]=true
    @test_throws ArgumentError WP.acknowledgement_data(candidate,id,1)
    candidate=deepcopy(ack);candidate["message"]="a continue cannot carry an abort"
    @test_throws ArgumentError WP.acknowledgement_data(candidate,id,1)
    fault=ErrorException("original abort")
    abort=WP.acknowledgement(id,1;abort=fault)
    @test WP.acknowledgement_data(abort,id,1)["action"]=="abort"
    @test abort["error_type"]=="ErrorException" && occursin("original abort",abort["message"])
    cancellation=Dict{String,Any}("version"=>1,"job_id"=>id,"cancel"=>true)
    @test WP.cancellation_data(cancellation,id)
    @test_throws ArgumentError WP.cancellation_data(cancellation,stale)
    cancellation["cancel"]=1
    @test_throws ArgumentError WP.cancellation_data(cancellation,id)
end

@testset "Terminal envelope cannot replace completion truth with exit or counts" begin
    id="4f57b25c-3240-4faf-9e20-d3b9a47cbe99"
    # Run contents are validated against the saved record by the client, not
    # reconstructed by this envelope validator. This tests transport only.
    terminal=Dict{String,Any}("version"=>1,"job_id"=>id,"status"=>"completed","written"=>3,
        "total"=>3,"run"=>Dict{String,Any}(),"error_type"=>"","message"=>"","backtrace"=>"",
        "history_error"=>"","core_cleanup_returned"=>true,"worker_pid"=>Int(getpid()),
        "loaded_modules"=>["Hammerhead","TOML","SHA"])
    @test WP.terminal_data(terminal,id,3)===terminal
    for (key,value) in (("written",2),("written",3.),("total",true),("run",nothing),
                        ("core_cleanup_returned",false),("worker_pid",true),
                        ("error_type","ErrorException"),("loaded_modules",["Hammerhead","QML"]),
                        ("loaded_modules",["HammerheadGUI"]),("loaded_modules",["GLMakie"]),
                        ("loaded_modules",["QMLMakie"]),("status","crashed"))
        candidate=deepcopy(terminal);candidate[key]=value
        @test_throws ArgumentError WP.terminal_data(candidate,id,3)
    end
    @test_throws ArgumentError WP.terminal_data(terminal,"stale",3)
    failed=deepcopy(terminal);merge!(failed,Dict("status"=>"failed","run"=>nothing,
        "error_type"=>"ErrorException","message"=>"history failed"))
    @test WP.terminal_data(failed,id,3)===failed # All writes still need not complete a run.
    cancelled=deepcopy(failed);merge!(cancelled,Dict("status"=>"cancelled","written"=>1))
    @test WP.terminal_data(cancelled,id,3)===cancelled
    cancelled["written"]=3
    @test_throws ArgumentError WP.terminal_data(cancelled,id,3)
end

@testset "Bounded controls preserve destinations on refusal" begin
    mktempdir() do directory
        path=joinpath(directory,"control.toml")
        original=Dict("message"=>"captured control","value"=>7)
        WP.write_control(path,original)
        @test WP.read_control(path)==original
        before=read(path)
        @test_throws ArgumentError WP.write_control(path,Dict("message"=>repeat("x",WP.MAX_CONTROL_BYTES)))
        @test read(path)==before
        @test !any(endswith(".partial"),readdir(directory))
        replacement=Dict("message"=>"next control","value"=>8)
        WP.write_control(path,replacement)
        @test WP.read_control(path)==replacement
        oversized=joinpath(directory,"oversized.toml")
        open(oversized,"w") do io
            write(io,zeros(UInt8,WP.MAX_CONTROL_BYTES+1))
        end
        @test_throws ArgumentError WP.read_control(oversized)
        @test_throws ArgumentError WP.read_control(joinpath(directory,"missing"))
        unicode_abort=WP.acknowledgement("4f57b25c-3240-4faf-9e20-d3b9a47cbe99",1;
            abort=ErrorException(repeat("\U0001f642",2 * WP.MAX_ERROR_CHARS)))
        WP.write_control(path,unicode_abort)
        @test filesize(path)<=WP.MAX_CONTROL_BYTES
        @test WP.acknowledgement_data(WP.read_control(path),unicode_abort["job_id"],1)["action"]=="abort"
        @test occursin("worker protocol text truncated",unicode_abort["message"])
        @test isvalid(unicode_abort["message"])
        symlink_path=joinpath(directory,"linked.toml")
        linked=try
            symlink(path,symlink_path);true
        catch
            false
        end
        if linked
            @test_throws ArgumentError WP.read_control(symlink_path)
        end
    end
end
