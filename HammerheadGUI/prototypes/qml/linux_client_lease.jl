struct _LinuxChannel
    in::IOStream
    out::IOStream
end
mutable struct _LinuxLease <: _OwnershipLease
    pipe::_LinuxChannel
    guardian::Any
    guardian_pid::Int
    guardian_fd::Int
    worker_fd::Int
    worker_pid::Int
    directory::String
    request::Dict{String,Any}
    request_sha::String
    proof::Any
    failed::Bool
    failure_reason::String
    startup_deadline::Float64
    stop_published::Bool
end
function _linux_lease()
    fds=LinuxFDTransport.pair()
    pipe=_LinuxChannel(Base.fdio(fds[1],true),Base.fdio(fds[2],true))
    _LinuxLease(pipe,nothing,0,-1,-1,0,"",Dict{String,Any}(),"",nothing,false,"",Inf,false)
end
function _assign!(lease::_LinuxLease,process)
    lease.guardian=process;lease.guardian_pid=Int(getpid(process))
    lease.startup_deadline=time()+30.
end
function _worker_sources(script,lease)
    # Capture every included ownership helper on both hosts: their definitions
    # participate in the loaded client even when host-specific syscalls are unused.
    sources=[script,joinpath(@__DIR__,"worker_client.jl"),joinpath(@__DIR__,"worker_protocol.jl"),
        joinpath(@__DIR__,"replay_worker.jl"),joinpath(@__DIR__,"linux_client_lease.jl"),
        joinpath(@__DIR__,"linux_ownership.jl"),joinpath(@__DIR__,"linux_fd_transport.jl")]
    if lease isa _LinuxLease
        append!(sources,[joinpath(@__DIR__,"linux_guardian.jl"),joinpath(@__DIR__,"linux_worker_bootstrap.jl")])
    end
    unique(sources)
end
_scientific_exit_code(lease::_WindowsLease,process)=Int(process.exitcode)
_scientific_exit_code(lease::_LinuxLease,process)=lease.proof["worker_term_signal"]==0 ?
    lease.proof["worker_exit_code"] : 128+lease.proof["worker_term_signal"]
function _linux_ready!(lease)
    _linux_messages!(lease)
    lease.worker_pid>0 && lease.guardian_fd>=0 && lease.worker_fd>=0 || return false
    path=joinpath(lease.directory,"guardian_ready.toml")
    isfile(path) || return false
    ready=LinuxOwnership.ready_data(WorkerProtocol.read_control(path),lease.request,lease.guardian_pid,lease.request_sha)
    ready["worker_pid"]===lease.worker_pid || error("guardian ready disagrees with transferred worker identity")
    true
end
function _linux_messages!(lease)
    isopen(lease.pipe.in) || return
    for _ in 1:2
        packet=LinuxFDTransport.receive_fd(Base.fd(lease.pipe.in))
        packet===nothing && return
        packet.kind===:eof && return
        fd=packet.fd;adopted=false
        try
            role=packet.data["role"]
            actual_pid=LinuxOwnership.descriptor_pid(fd)
            announced=packet.data["pid"]
            announced isa Int && announced>0 && (actual_pid==announced || actual_pid == -1) || error("crossed pidfd kernel identity")
            if role=="guardian"
                lease.guardian_fd<0 || error("duplicate guardian pidfd")
                LinuxOwnership.descriptor(packet.data,lease.request,role,lease.guardian_pid,lease.request_sha)
                lease.guardian_fd=fd;adopted=true
            elseif role=="worker"
                lease.worker_fd<0 || error("duplicate worker pidfd")
                LinuxOwnership.descriptor(packet.data,lease.request,role,announced,lease.request_sha)
                lease.worker_pid=announced;lease.worker_fd=fd;adopted=true
            else
                error("unknown pidfd role")
            end
        finally
            adopted || LinuxOwnership.close_fd(fd)
        end
    end
end
function _lease_count(lease::_LinuxLease)
    if lease.guardian!==nothing && process_exited(lease.guardian) &&
            (lease.guardian.exitcode!=0 || lease.guardian.termsignal!=0)
        lease.failed=true
        isempty(lease.failure_reason) && (lease.failure_reason="guardian exited abnormally without descendant cleanup proof; refuse further launches")
        return 1
    end
    lease.proof!==nothing && return 0
    lease.failed && return 1
    isempty(lease.directory) && return lease.guardian===nothing ? 0 : 1
    path=joinpath(lease.directory,"guardian_proof.toml")
    if isfile(path)
        try
            _linux_ready!(lease) || error("cleanup proof without enrollment")
            lease.proof=LinuxOwnership.proof_data(WorkerProtocol.read_control(path),lease.request,
                lease.guardian_pid,lease.worker_pid,lease.request_sha)
            return 0
        catch error
            lease.failed=true
            isempty(lease.failure_reason) && (lease.failure_reason=sprint(showerror,error))
        end
    elseif lease.guardian!==nothing && process_exited(lease.guardian)
        lease.failed=true
        isempty(lease.failure_reason) && (lease.failure_reason="guardian exited without descendant cleanup proof; refuse further launches")
    end
    1
end
function _close_lease!(lease::_LinuxLease)
    lease.guardian===nothing || lease.proof!==nothing || error("Linux ownership cleanup remains unconfirmed")
    isopen(lease.pipe.in) && close(lease.pipe.in)
    isopen(lease.pipe.out) && close(lease.pipe.out)
    LinuxOwnership.close_fd(lease.guardian_fd);lease.guardian_fd=-1
    LinuxOwnership.close_fd(lease.worker_fd);lease.worker_fd=-1
end
function _terminate!(lease::_LinuxLease)
    isempty(lease.directory) && return
    if !lease.stop_published
        WorkerProtocol.write_control(joinpath(lease.directory,"terminate.toml"),Dict("version"=>1,"job_id"=>lease.request["job_id"],"terminate"=>true))
        lease.stop_published=true
    end
    if lease.worker_fd>=0
        rc=LinuxOwnership.signal_fd(lease.worker_fd,9;group=true)
        rc.rc==0 || rc.errno==3 || error("bound Linux group termination failed errno=$(rc.errno)")
    end
    # EOF is independent of the scientific worker's scheduling; guardian remains
    # alive to reap and prove no adopted descendants, including escaped groups.
    # Before enrollment this socket may still contain the only transferred kernel
    # identities. Keep it open while requesting stop, then close only after both
    # references were safely received. A healthy guardian must survive cleanup.
    lease.worker_fd>=0 && lease.guardian_fd>=0 && isopen(lease.pipe.in) && close(lease.pipe.in)
end
function _linux_startup_cleanup!(lease,process;timeout)
    lease.guardian===nothing && (lease.guardian=process)
    deadline=time()+timeout
    while time()<deadline
        try
            _linux_ready!(lease)
            _terminate!(lease)
            if process_exited(process) && _lease_count(lease)==0
                wait(process)
                return true
            end
        catch error
            lease.failed=true;lease.failure_reason=sprint(showerror,error)
            return false
        end
        sleep(.02)
    end
    false
end
function _linux_enroll!(job)
    lease=job.lease
    job.worker_pid>0 && return
    if !_linux_ready!(lease)
        if time()>lease.startup_deadline || process_exited(job.process)
            _fault!(job,:crashed,"Linux guardian enrollment failed or timed out")
        end
        return
    end
    job.worker_pid=lease.worker_pid
    # Root remains blocked until this owner acquires a kernel reference. Never
    # recover a group by numeric PGID after a process has been reaped.
    if !isfile(joinpath(job.directory,"guardian_proof.toml"))
        support=LinuxOwnership.signal_fd(lease.worker_fd,0;group=true)
        support.rc==0 || support.errno==3 || error("Linux group support/identity unavailable")
        support.errno==3 && return # Bound worker exited before enrollment; await guardian proof.
        owner=Dict("version"=>1,"job_id"=>job.job_id,"owner_pid"=>Int(getpid()),"worker_pid"=>job.worker_pid,
            "ownership"=>"linux_pidfd_subreaper_guardian","directory"=>job.directory,"request_sha256"=>lease.request_sha)
        WorkerProtocol.write_control(joinpath(job.directory,"owner.toml"),owner)
        WorkerProtocol.write_control(joinpath(job.directory,"enrolled.toml"),owner)
    end
end
