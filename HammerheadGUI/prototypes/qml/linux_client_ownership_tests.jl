using Test,Hammerhead,UUIDs
include("worker_client.jl")
using .ReplayWorkerClient
include("experiment_fixture.jl")
include("linux_test_socket.jl")
const C=ReplayWorkerClient
const L=C.LinuxOwnership
const T=C.LinuxFDTransport
L.subreaper!() # Fixture rescue proof only; this is NOT the production owner's guarantee.
const mode=only(ARGS)
mkpath(joinpath(@__DIR__,"artifacts"))
const ROOT=mktempdir(joinpath(@__DIR__,"artifacts");prefix="linux-$mode-",cleanup=false)
println("LINUX_CLIENT_OWNERSHIP_ARTIFACTS=",ROOT)
const fixture=experiment_fixture(joinpath(ROOT,"inputs"))
const name="hammerhead-fixture-"*string(uuid4())
const listener=mode=="escaped" ? LinuxTestSocket.listen(name) : -1
const script=joinpath(ROOT,"fixture_worker.jl")
const leaf_script=joinpath(@__DIR__,"linux_escaped_leaf.jl")
body="r=WorkerProtocol.request_data(WorkerProtocol.read_control(only(ARGS)))\nd=dirname(only(ARGS))\nwhile !isfile(joinpath(d,\"enrolled.toml\"));sleep(.01);end\n"
if mode=="escaped"
    body*="run(`\$(Base.julia_cmd()) --startup-file=no --threads=1 $(repr(leaf_script)) $(repr(name)) \$(r[\"job_id\"])`;wait=false)\n"
end
body*="WorkerProtocol.write_control(joinpath(d,\"fixture_ready.toml\"),Dict(\"pid\"=>Int(getpid())))\nsleep(120.)\n"
write(script,body)
const output=joinpath(ROOT,"preserved.jld2");write(output,"previous bytes")
const job=start_replay(fixture.record;output,artifact_root=joinpath(ROOT,"jobs"),worker_script=script)
leaf_fd=-1;peer_fd=-1
try
    timedwait(()->begin poll!(job);isfile(joinpath(job.directory,"fixture_ready.toml")) end,30.;pollint=.01)==:ok || error("fixture enrollment timeout")
    @testset "Linux $mode ownership truth" begin
        @test job.worker_pid>0 && job.lease.worker_fd>=0 && job.lease.guardian_fd>=0
        if mode=="escaped"
            peer=Ref{Any}(nothing)
            timedwait(()->begin peer[]=LinuxTestSocket.accept(listener);peer[]!==nothing end,15.;pollint=.01)==:ok || error("leaf socket timeout")
            peer_fd=peer[]
            packet=Ref{Any}(nothing)
            timedwait(()->begin packet[]=T.receive_fd(peer_fd);packet[]!==nothing end,10.;pollint=.01)==:ok || error("leaf pidfd timeout")
            @assert packet[].kind===:fd
            leaf_fd=packet[].fd
            @test packet[].data["job_id"]==job.job_id && packet[].data["role"]=="escaped_fixture"
            @test L.descriptor_pid(leaf_fd)==packet[].data["pid"]
            @test !L.exited(leaf_fd)
            @test_throws ErrorException shutdown!(job;timeout=0.)
            refusal=C.WorkerProtocol.read_control(joinpath(job.directory,"cleanup_unconfirmed.toml"))
            @test refusal["group_empty"] && !refusal["children_echild"] && !refusal["cleanup_confirmed"]
            @test active(job) && outcome(job)===nothing
            @test !L.exited(leaf_fd) && !process_exited(job.process)
            @test_throws ArgumentError start_replay(fixture.record;output=joinpath(ROOT,"forbidden.jld2"))
            @test read(output,String)=="previous bytes"
            # Only this test knows the escaped child's independently transferred
            # reference. Production may not invent it or certify group ESRCH alone.
            @test L.signal_fd(leaf_fd,9).rc==0
            timedwait(()->begin poll!(job);!active(job) end,10.;pollint=.01)==:ok || error("post-rescue guardian proof timeout")
            @test outcome(job).status===:timeout && !outcome(job).cleanup_confirmed
            @test job.lease.proof["children_echild"] && job.lease.proof["root_reaped"]
            @test process_exited(job.process)
        elseif mode=="guardian_loss"
            # Inject protocol-shaped progress into this non-scientific fixture.
            # Ownership loss must suppress both an unacknowledged event and a
            # newer staged event, even if their scalar envelopes are valid.
            progress_path=joinpath(job.directory,"progress.toml")
            C.WorkerProtocol.write_control(progress_path,Dict("version"=>1,"job_id"=>job.job_id,
                "sequence"=>1,"written"=>1,"total"=>job.request["total"]))
            pending_event=poll!(job)
            @test pending_event.kind===:progress && pending_event.sequence==1
            @test L.signal_fd(job.lease.guardian_fd,9).rc==0
            timedwait(()->L.exited(job.lease.guardian_fd),5.;pollint=.01)==:ok || error("guardian death timeout")
            wait(job.process) # Reap before checking the client, matching the reviewed race.
            # A structurally valid-looking file cannot override the guardian's actual
            # abnormal OS termination or restore its lost descendant inspection.
            forged=Dict("version"=>1,"job_id"=>job.job_id,"owner_pid"=>Int(getpid()),"guardian_pid"=>job.lease.guardian_pid,
                "worker_pid"=>job.worker_pid,"request_sha256"=>job.lease.request_sha,"root_reaped"=>true,"group_empty"=>true,
                "children_echild"=>true,"worker_exit_code"=>0,"worker_term_signal"=>0,"cleanup_confirmed"=>true,
                "stopped_by_owner"=>false,"residual_group_killed"=>false)
            C.WorkerProtocol.write_control(joinpath(job.directory,"guardian_proof.toml"),forged)
            C.WorkerProtocol.write_control(progress_path,Dict("version"=>1,"job_id"=>job.job_id,
                "sequence"=>2,"written"=>2,"total"=>job.request["total"]))
            @test poll!(job)===nothing
            @test_throws ArgumentError acknowledge_progress!(job,pending_event)
            @test !isfile(joinpath(job.directory,"ack-1.toml"))
            # Exercise new delivery independently of the pending-event guard.
            # This is fixture state, never a way for application code to bypass ACK.
            job.pending=nothing
            @test poll!(job)===nothing && job.last_sequence==1
            @test timedwait(()->L.exited(job.lease.worker_fd),5.;pollint=.01)==:ok
            @test active(job) && outcome(job)===nothing
            @test job.lease.failed && occursin("without descendant cleanup proof",job.lease.failure_reason)
            @test ownership_error(job)==job.lease.failure_reason
            @test_throws ArgumentError start_replay(fixture.record;output=joinpath(ROOT,"forbidden.jld2"))
            rc=L.signal_fd(job.lease.worker_fd,9;group=true)
            @test rc.rc==0 || rc.errno==3
            timedwait(()->L.exited(job.lease.worker_fd),5.;pollint=.01)==:ok || error("fixture root rescue timeout")
            @test timedwait(L.no_children,5.;pollint=.01)==:ok
            poll!(job)
            @test active(job) && outcome(job)===nothing # Exact rescue cannot restore lost proof.
            @test isfile(joinpath(job.directory,"guardian_proof.toml")) && job.lease.proof===nothing
            @test read(output,String)=="previous bytes"
        else
            error("unknown fixture mode")
        end
    end
finally
    leaf_fd>=0 && L.signal_fd(leaf_fd,9)
    job.lease.worker_fd>=0 && L.signal_fd(job.lease.worker_fd,9;group=true)
    if !process_exited(job.process)
        C._terminate!(job.lease)
        timedwait(()->process_exited(job.process),15.;pollint=.01)==:ok || error("guardian remains alive; cleanup unverified")
    end
    wait(job.process)
    timedwait(L.no_children,5.;pollint=.01)==:ok || error("known fixture descendants unreaped; stop launches")
    for fd in (leaf_fd,peer_fd,listener);L.close_fd(fd);end
end
println("LINUX_CLIENT_OWNERSHIP_PASSED mode=",mode," os_fixture_children_reaped=true")
