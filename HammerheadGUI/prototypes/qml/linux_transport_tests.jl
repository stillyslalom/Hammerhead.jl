using Test,TOML
include("worker_protocol.jl")
include("linux_ownership.jl")
include("linux_fd_transport.jl")
const L=LinuxOwnership
const T=LinuxFDTransport
const SCRIPT=abspath(@__FILE__)
fdcount()=length(readdir("/proc/self/fd"))
function raw_send(socket,fds,payload)
    len=16+4length(fds);space=(len+7)&~7
    control=zeros(UInt8,space)
    GC.@preserve control payload begin
        unsafe_store!(Ptr{Csize_t}(pointer(control)),len)
        unsafe_store!(Ptr{Cint}(pointer(control)+8),1)
        unsafe_store!(Ptr{Cint}(pointer(control)+12),1)
        for (i,fd) in enumerate(fds)
            unsafe_store!(Ptr{Cint}(pointer(control)+16+4(i-1)),fd)
        end
        iov=Ref(T.IOVec(pointer(payload),length(payload)))
        GC.@preserve iov begin
            msg=Ref(T.MsgHdr(C_NULL,0,Base.unsafe_convert(Ptr{T.IOVec},iov),1,pointer(control),length(control),0))
            rc=ccall(:sendmsg,Clong,(Cint,Ptr{T.MsgHdr},Cint),socket,msg,0x4000|0x40)
            rc==length(payload) || error("raw send errno=$(Libc.errno())")
        end
    end
end
function await_packet(fd)
    packet=Ref{Any}(nothing)
    timedwait(()->begin packet[]=T.receive_fd(fd);packet[]!==nothing end,10.;pollint=.01)==:ok || error("packet timeout")
    packet[]
end
function descendant_mode()
    fd=L.open_pidfd(getpid())
    try T.send_fd(0,fd,Dict("role"=>"leaf","pid"=>Int(getpid()))) finally L.close_fd(fd) end
    write(joinpath(ARGS[2],"leaf-ready"),"ready")
    sleep(90.)
end
function leader_mode()
    @assert ccall(:setpgid,Cint,(Cint,Cint),0,0)==0
    fd=L.open_pidfd(getpid())
    try T.send_fd(0,fd,Dict("role"=>"leader","pid"=>Int(getpid()))) finally L.close_fd(fd) end
    socket=Base.fdio(0,false)
    run(pipeline(`$(Base.julia_cmd()) --startup-file=no --threads=1 $SCRIPT leaf $(ARGS[2])`,stdin=socket);wait=false)
    sleep(90.)
end
if !isempty(ARGS)
    ARGS[1]=="leader" ? leader_mode() : descendant_mode()
else
    L.subreaper!()
    @testset "SCM_RIGHTS bounded packets and descriptor cleanup" begin
        pair=T.pair();self=L.open_pidfd(getpid())
        try
            T.send_fd(pair[1],self,Dict("role"=>"self","pid"=>Int(getpid())))
            packet=await_packet(pair[2])
            @test packet.kind===:fd
            @test packet.data["pid"]==Int(getpid())
            @test L.descriptor_pid(packet.fd)==Int(getpid())
            @test L.signal_fd(packet.fd,0).rc==0
            L.close_fd(packet.fd)
            baseline=fdcount()
            for (name,rights,payload) in (("zero",[self],UInt8[]),("multiple",[self,self],Vector{UInt8}(codeunits("x=1\n"))),
                ("malformed",[self],Vector{UInt8}(codeunits("[bad"))),
                ("truncated",[self],fill(UInt8('x'),T.MAX_PACKET+100)),
                ("control_truncated",fill(self,80),Vector{UInt8}(codeunits("x=1\n"))))
                @testset "$name" begin
                    raw_send(pair[1],rights,payload)
                    @test_throws Exception T.receive_fd(pair[2])
                    @test fdcount()==baseline
                end
            end
            @test T.receive_fd(pair[2])===nothing
        finally
            L.close_fd(self);L.close_fd(pair[1]);L.close_fd(pair[2])
        end
    end
    @testset "Delayed descriptor receipt after sender exit and libuv reap" begin
        pair=T.pair();io=Base.fdio(pair[2],true)
        mkpath(joinpath(@__DIR__,"artifacts"))
        directory=mktempdir(joinpath(@__DIR__,"artifacts");prefix="transport-fixture-",cleanup=false)
        proc=run(pipeline(`$(Base.julia_cmd()) --startup-file=no --threads=1 $SCRIPT leader $directory`,stdin=io);wait=false)
        close(io)
        leaderfd=-1;leaffd=-1
        try
            timedwait(()->isfile(joinpath(directory,"leaf-ready")),15.;pollint=.01)==:ok || error("leaf enrollment timeout")
            kill(proc,Base.SIGKILL)
            wait(proc)
            @test process_exited(proc)
            # Neither receiver has opened a numeric PID. Both references have been
            # queued by the still-live senders themselves and survive sender reap.
            first=await_packet(pair[1]);second=await_packet(pair[1])
            @test first.data["role"]=="leader" && second.data["role"]=="leaf"
            leaderfd=first.fd;leaffd=second.fd
            @test L.exited(leaderfd) && !L.exited(leaffd)
            @test L.descriptor_pid(leaderfd)==-1
            @test L.signal_fd(leaderfd,0;group=true).rc==0
            @test L.signal_fd(leaderfd,9;group=true).rc==0
            @test timedwait(()->L.exited(leaffd),5.;pollint=.01)==:ok
            @test timedwait(L.no_children,5.;pollint=.01)==:ok
            @test L.group_empty(leaderfd)
        finally
            leaderfd>=0 && L.signal_fd(leaderfd,9;group=true)
            leaffd>=0 && L.signal_fd(leaffd,9)
            if !process_exited(proc)
                kill(proc,Base.SIGKILL)
                timedwait(()->process_exited(proc),5.;pollint=.01)==:ok || error("exact sender cleanup unconfirmed")
            end
            wait(proc)
            timedwait(L.no_children,5.;pollint=.01)==:ok || error("descendant cleanup unconfirmed; stop launches")
            for fd in (leaderfd,leaffd,pair[1]);L.close_fd(fd);end
        end
    end
end
