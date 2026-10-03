# Core-free guardian. It alone owns/reaps the scientific child and adopted descendants.
include("worker_protocol.jl")
using .WorkerProtocol
include("linux_ownership.jl")
using .LinuxOwnership
include("linux_fd_transport.jl")
length(ARGS)==3 || error("guardian expects captured request, worker script and thread pools")
const request_file=abspath(ARGS[1])
const request=WorkerProtocol.request_data(WorkerProtocol.read_control(request_file))
const directory=dirname(request_file)
const source_hash=WorkerProtocol.digest(request_file)
const owner_lost=Ref(false)
function stop_requested()
    # No owner data is expected on this socket; recvmsg is nonblocking and EOF is
    # tied to the exclusive owner endpoint, independently of numerical scheduling.
    if !owner_lost[]
        packet=LinuxFDTransport.receive_fd(0)
        if packet!==nothing && packet.kind!==:eof
            LinuxOwnership.close_fd(packet.fd)
            error("unexpected owner socket data")
        end
        packet!==nothing && (owner_lost[]=true)
    end
    owner_lost[] && return true
    path=joinpath(directory,"terminate.toml")
    isfile(path) || return false
    packet=WorkerProtocol.read_control(path)
    WorkerProtocol.check_keys(packet,("version","job_id","terminate"))
    packet["version"]===1 && packet["job_id"]==request["job_id"] && packet["terminate"]===true || error("invalid owner termination")
    true
end
function guardian_main()
    Sys.islinux() && Sys.WORD_SIZE==64 || error("Linux guardian requires validated x86_64 syscall ABI")
    LinuxOwnership.subreaper!()
    self_fd=LinuxOwnership.open_pidfd(getpid())
    try LinuxFDTransport.send_fd(0,self_fd,LinuxOwnership.descriptor_packet(request,"guardian",getpid(),source_hash))
    finally LinuxOwnership.close_fd(self_fd) end
    pair=LinuxFDTransport.pair()
    worker_socket=Base.fdio(pair[2],true)
    bootstrap=joinpath(@__DIR__,"linux_worker_bootstrap.jl")
    cmd=`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(request["project"])) --threads=$(ARGS[3]) $bootstrap $(ARGS[2]) $request_file`
    worker=run(pipeline(cmd;stdin=worker_socket);wait=false)
    close(worker_socket)
    worker_pid=Int(getpid(worker))
    fd=-1;armed=false;stopped=false;residual=false
    try
        startup=time()+30.
        packet=nothing
        while packet===nothing
            packet=LinuxFDTransport.receive_fd(pair[1])
            packet!==nothing && break
            # A queued parent startup failure must not destroy the guardian or
            # discard the worker's self-opened reference. Bootstrap cannot enter
            # any script before enrollment, so this bounded receive remains safe.
            stop_requested() # Observe EOF/validate stop envelope; apply after acquiring fd.
            if process_exited(worker) || time()>startup
                error("worker group enrollment failed before processing")
            end
            sleep(.01)
        end
        packet.kind===:fd || error("worker failed before transferring identity")
        fd=packet.fd
        LinuxOwnership.descriptor(packet.data,request,"worker",worker_pid,source_hash)
        support=LinuxOwnership.signal_fd(fd,0;group=true)
        support.rc==0 || support.errno==3 || error("Linux identity-bound group signaling unavailable errno=$(support.errno); kernel6.9+ required")
        armed=true
        LinuxFDTransport.send_fd(0,fd,packet.data)
        ready=Dict("version"=>1,"job_id"=>request["job_id"],"owner_pid"=>request["owner_pid"],"guardian_pid"=>Int(getpid()),
            "worker_pid"=>worker_pid,"request_sha256"=>source_hash,"group_signal_supported"=>true)
        WorkerProtocol.write_control(joinpath(directory,"guardian_ready.toml"),ready)
        while !process_exited(worker)
            if stop_requested() && !stopped
                result=LinuxOwnership.signal_fd(fd,9;group=true)
                result.rc==0 || result.errno==3 || error("bound termination failed errno=$(result.errno)")
                stopped=true
            end
            sleep(.01)
        end
        wait(worker) # Registered libuv child is reaped before raw adopted-child waits.
        if !LinuxOwnership.group_empty(fd)
            result=LinuxOwnership.signal_fd(fd,9;group=true)
            result.rc==0 || result.errno==3 || error("residual group termination failed")
            residual=true
        end
        announced=false
        while true
            children_empty=LinuxOwnership.no_children()
            group_empty=LinuxOwnership.group_empty(fd)
            children_empty && group_empty && break
            if !announced
                WorkerProtocol.write_control(joinpath(directory,"cleanup_unconfirmed.toml"),Dict("version"=>1,"job_id"=>request["job_id"],
                    "guardian_pid"=>Int(getpid()),"worker_pid"=>worker_pid,"group_empty"=>group_empty,
                    "children_echild"=>children_empty,"cleanup_confirmed"=>false))
                announced=true
            end
            # A live escaped child cannot be silently certified or abandoned. The owner
            # retains busy/incomplete state; no numeric PID/tree kill is substituted.
            sleep(.02)
        end
        proof=Dict("version"=>1,"job_id"=>request["job_id"],"owner_pid"=>request["owner_pid"],"guardian_pid"=>Int(getpid()),
            "worker_pid"=>worker_pid,"request_sha256"=>source_hash,"root_reaped"=>true,"group_empty"=>true,
            "children_echild"=>true,"worker_exit_code"=>Int(worker.exitcode),"worker_term_signal"=>Int(worker.termsignal),"cleanup_confirmed"=>true,
            "stopped_by_owner"=>stopped,"residual_group_killed"=>residual)
        LinuxOwnership.proof_data(proof,request,Int(getpid()),worker_pid,source_hash)
        WorkerProtocol.write_control(joinpath(directory,"guardian_proof.toml"),proof)
        0
    finally
        if !process_exited(worker)
            if fd>=0
                LinuxOwnership.signal_fd(fd,9;group=armed)
            else
                kill(worker,Base.SIGKILL) # Exact fresh Base.Process only, before user script enrollment.
            end
            timedwait(()->process_exited(worker),10.;pollint=.02)==:ok || error("guardian startup cleanup unconfirmed")
        end
        wait(worker)
        LinuxOwnership.close_fd(fd)
        LinuxOwnership.close_fd(pair[1])
    end
end
exit(guardian_main())
