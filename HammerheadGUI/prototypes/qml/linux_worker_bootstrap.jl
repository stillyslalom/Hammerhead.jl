# Self-open while alive; SCM_RIGHTS preserves kernel identity even if this process
# exits and libuv reaps it before either receiver services its socket.
ccall(:setpgid,Cint,(Cint,Cint),0,0)==0 || error("worker cannot create private group")
include("worker_protocol.jl")
using .WorkerProtocol
include("linux_ownership.jl")
include("linux_fd_transport.jl")
length(ARGS)==2 || error("worker bootstrap requires script and request")
const script=ARGS[1]
const bootstrap_request_file=abspath(ARGS[2])
const bootstrap_request=WorkerProtocol.request_data(WorkerProtocol.read_control(bootstrap_request_file))
const self_fd=LinuxOwnership.open_pidfd(getpid())
try
    LinuxFDTransport.send_fd(0,self_fd,LinuxOwnership.descriptor_packet(bootstrap_request,"worker",getpid(),WorkerProtocol.digest(bootstrap_request_file)))
finally
    LinuxOwnership.close_fd(self_fd)
end
const bootstrap_enrollment=joinpath(dirname(bootstrap_request_file),"enrolled.toml")
const bootstrap_deadline=time()+10.
while !isfile(bootstrap_enrollment)
    time()<bootstrap_deadline || error("bootstrap owner enrollment deadline expired before worker script/core imports")
    sleep(.01)
end
empty!(ARGS);push!(ARGS,bootstrap_request_file)
include(script)
