using Test,Hammerhead,UUIDs
include("worker_client.jl")
include("experiment_fixture.jl")
include("linux_test_socket.jl")
const C=ReplayWorkerClient
const L=C.LinuxOwnership
const T=C.LinuxFDTransport
L.subreaper!() # Outer fixture only, to inspect actual orphan guardian exit.
mkpath(joinpath(@__DIR__,"artifacts"))
const ROOT=mktempdir(joinpath(@__DIR__,"artifacts");prefix="linux-owner-loss-",cleanup=false)
println("LINUX_OWNER_LOSS_ARTIFACTS=",ROOT)
const fixture=experiment_fixture(joinpath(ROOT,"inputs"))
const record_path=joinpath(ROOT,"record.jld2")
save_experiment(record_path,fixture.record)
const output=joinpath(ROOT,"preserved.jld2");write(output,"previous output")
const name="hammerhead-owner-loss-"*string(uuid4())
const listener=LinuxTestSocket.listen(name)
const leaf_name="hammerhead-owner-leaf-"*string(uuid4())
const leaf_listener=LinuxTestSocket.listen(leaf_name)
const script=joinpath(ROOT,"nonyielding_worker.jl")
const leaf_script=joinpath(@__DIR__,"linux_worker_owner_loss_leaf.jl")
write(script,"r=WorkerProtocol.request_data(WorkerProtocol.read_control(only(ARGS)))\nd=dirname(only(ARGS))\nleafready=joinpath(d,\"leaf_ready.toml\")\nleaf=run(`\$(Base.julia_cmd()) --startup-file=no --threads=1 $(repr(leaf_script)) $(repr(leaf_name)) \$(r[\"job_id\"]) \$leafready`;wait=false)\ntimedwait(()->isfile(leafready),15.;pollint=.01)==:ok || error(\"leaf readiness timeout\")\nWorkerProtocol.write_control(joinpath(d,\"fixture_ready.toml\"),Dict(\"pid\"=>Int(getpid()),\"leaf_pid\"=>Int(getpid(leaf))))\nx=UInt(1)\nwhile true; global x=x*UInt(1664525)+UInt(1013904223); end\n")
const helper=joinpath(@__DIR__,"linux_worker_owner_loss_child.jl")
const log=open(joinpath(ROOT,"owner.log"),"w")
const process=run(pipeline(`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(dirname(Base.active_project())) $helper $name $record_path $ROOT $script`;stdout=log,stderr=log);wait=false)
const owner_pid=Int(getpid(process))
handles=Dict{String,Int}();packets=Dict{String,Any}();peer=-1;leaf_peer=-1;verified=false
try
    timedwait(()->begin p=LinuxTestSocket.accept(listener);p===nothing || (global peer=p);peer>=0 || process_exited(process) end,
        30.;pollint=.01)==:ok && peer>=0 || error("owner identity handshake failed; no descendant cleanup claim")
    deadline=time()+40.
    while length(handles)<3 && time()<deadline
        packet=T.receive_fd(peer)
        if packet!==nothing
            packet.kind===:fd || error("owner identity handshake ended before all references")
            role=packet.data["role"]
            role in ("owner","guardian","worker") && !haskey(handles,role) || begin
                L.close_fd(packet.fd);error("crossed/repeated fixture descriptor")
            end
            handles[role]=packet.fd;packets[role]=packet.data
        end
        sleep(.01)
    end
    length(handles)==3 || error("fixture identities incomplete; stop all subsequent launches")
    timedwait(()->begin p=LinuxTestSocket.accept(leaf_listener);p===nothing || (global leaf_peer=p);leaf_peer>=0 end,
        15.;pollint=.01)==:ok || error("leaf identity connection timeout")
    leaf_packet=Ref{Any}(nothing)
    timedwait(()->begin leaf_packet[]=T.receive_fd(leaf_peer);leaf_packet[]!==nothing end,
        10.;pollint=.01)==:ok || error("leaf identity transfer timeout")
    leaf_packet[].kind===:fd || error("leaf identity transfer ended")
    handles["leaf"]=leaf_packet[].fd;packets["leaf"]=leaf_packet[].data
    @testset "Actual owner loss with socket guardian and nonyielding subtree" begin
        @test packets["owner"]["pid"]==owner_pid && L.descriptor_pid(handles["owner"])==owner_pid
        @test all(role->L.descriptor_pid(handles[role])==packets[role]["pid"],("guardian","worker","leaf"))
        @test packets["leaf"]["role"]=="leaf" && packets["leaf"]["job_id"]==packets["worker"]["job_id"]
        @test packets["worker"]["job_id"]==packets["guardian"]["job_id"]
        @test packets["worker"]["directory"]==packets["guardian"]["directory"]
        leaf_alive_before=!L.exited(handles["leaf"])
        @test leaf_alive_before && all(fd->!L.exited(fd),values(handles))
        directory=packets["worker"]["directory"]
        ready=C.WorkerProtocol.read_control(joinpath(directory,"fixture_ready.toml"))
        @test ready["leaf_pid"]==packets["leaf"]["pid"] && ready["pid"]==packets["worker"]["pid"]
        @test L.signal_fd(handles["owner"],9).rc==0
        wait(process)
        @test process.termsignal==9 && process_exited(process)
        @test timedwait(()->L.exited(handles["worker"]) && L.exited(handles["guardian"]) && L.exited(handles["leaf"]),15.;pollint=.01)==:ok
        proof=C.WorkerProtocol.read_control(joinpath(directory,"guardian_proof.toml"))
        request=C.WorkerProtocol.request_data(C.WorkerProtocol.read_control(joinpath(directory,"request.toml")))
        @test L.proof_data(proof,request,packets["guardian"]["pid"],packets["worker"]["pid"],packets["worker"]["request_sha256"])===proof
        @test proof["stopped_by_owner"] && proof["worker_term_signal"]==9
        @test proof["root_reaped"] && proof["group_empty"] && proof["children_echild"]
        @test L.group_empty(handles["worker"])
        # The owner was libuv-reaped first. Guardian is now our adopted child;
        # inspect its actual status rather than accepting pidfd HUP as proof.
        status=Ref{Cint}(0)
        guardian_reaped=timedwait(()->begin
            p=ccall(:waitpid,Cint,(Cint,Ptr{Cint},Cint),-1,status,1)
            p==packets["guardian"]["pid"]
        end,5.;pollint=.01)==:ok
        @test guardian_reaped && status[]==0
        @test L.no_children()
        @test read(output,String)=="previous output"
        @test !isfile(joinpath(directory,"terminal.jld2"))
        global verified=guardian_reaped && status[]==0 && L.no_children()
        C.WorkerProtocol.write_control(joinpath(ROOT,"owner_loss_proof.toml"),
            Dict("version"=>1,"owner_pid"=>owner_pid,"worker_pid"=>packets["worker"]["pid"],
                "guardian_pid"=>packets["guardian"]["pid"],"job_id"=>request["job_id"],
                "leaf_pid"=>packets["leaf"]["pid"],"leaf_alive_before_owner_loss"=>leaf_alive_before,
                "leaf_exited_after_owner_loss"=>L.exited(handles["leaf"]),
                "owner_sigkill"=>9,"guardian_exit_status"=>Int(status[]),"fixture_children_echild"=>verified,
                "workload"=>"injected nonyielding CPU loop with child; not a PIV timing benchmark"))
    end
finally
    if !verified
        haskey(handles,"leaf") && L.signal_fd(handles["leaf"],9)
        haskey(handles,"worker") && L.signal_fd(handles["worker"],9;group=true)
        if haskey(handles,"owner")
            L.signal_fd(handles["owner"],9)
        elseif !process_exited(process)
            kill(process,Base.SIGKILL) # Exact fresh helper Process, not a PID lookup.
        end
        timedwait(()->process_exited(process),10.;pollint=.01)==:ok || error("fixture owner exit unverified; stop launches")
        wait(process)
        if haskey(handles,"guardian")
            timedwait(()->L.exited(handles["guardian"]),15.;pollint=.01)==:ok || error("healthy guardian cleanup unverified; stop launches")
        else
            error("guardian identity was not transferred; descendant cleanup unverified, stop launches")
        end
        timedwait(L.no_children,5.;pollint=.01)==:ok || error("fixture descendants remain; stop launches")
    end
    for fd in values(handles);L.close_fd(fd);end
    L.close_fd(peer);L.close_fd(leaf_peer);L.close_fd(listener);L.close_fd(leaf_listener);close(log)
end
println("LINUX_OWNER_LOSS_PASSED actual_socket_EOF=true nonyielding=true fixture_children_reaped=true")
