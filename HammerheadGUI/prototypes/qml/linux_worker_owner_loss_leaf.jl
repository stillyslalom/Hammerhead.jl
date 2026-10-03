include("worker_protocol.jl")
include("linux_ownership.jl")
include("linux_fd_transport.jl")
include("linux_test_socket.jl")
length(ARGS)==3 || error("leaf fixture expects socket, job id and ready path")
const fd=LinuxOwnership.open_pidfd(getpid())
const peer=LinuxTestSocket.connect(ARGS[1])
try
    LinuxFDTransport.send_fd(peer,fd,Dict("role"=>"leaf","pid"=>Int(getpid()),"job_id"=>ARGS[2]))
    WorkerProtocol.write_control(ARGS[3],Dict("pid"=>Int(getpid()),"job_id"=>ARGS[2]))
    sleep(120.)
finally
    LinuxOwnership.close_fd(fd);LinuxOwnership.close_fd(peer)
end
