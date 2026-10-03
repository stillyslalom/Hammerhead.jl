include("worker_protocol.jl")
include("linux_ownership.jl")
include("linux_fd_transport.jl")
include("linux_test_socket.jl")
ccall(:setsid,Cint,())==Int(getpid()) || error("fixture cannot escape group")
const peer=LinuxTestSocket.connect(ARGS[1])
const self_fd=LinuxOwnership.open_pidfd(getpid())
try
    LinuxFDTransport.send_fd(peer,self_fd,Dict("role"=>"escaped_fixture","pid"=>Int(getpid()),"job_id"=>ARGS[2]))
finally
    LinuxOwnership.close_fd(self_fd);LinuxOwnership.close_fd(peer)
end
sleep(120.)
