using Hammerhead
include("worker_client.jl")
include("linux_test_socket.jl")
const C=ReplayWorkerClient
const L=C.LinuxOwnership
const T=C.LinuxFDTransport
length(ARGS)==4 || error("owner fixture expects socket, record, root and worker script")
const peer=LinuxTestSocket.connect(ARGS[1])
const owner_fd=L.open_pidfd(getpid())
try
    T.send_fd(peer,owner_fd,Dict("role"=>"owner","pid"=>Int(getpid())))
finally
    L.close_fd(owner_fd)
end
const record=load_experiment(ARGS[2])
const job=C.start_replay(record;output=joinpath(ARGS[3],"preserved.jld2"),
    artifact_root=joinpath(ARGS[3],"jobs"),worker_script=ARGS[4])
try
    timedwait(()->begin C.poll!(job);isfile(joinpath(job.directory,"fixture_ready.toml")) end,
        30.;pollint=.01)==:ok || error("owner fixture worker enrollment timeout")
    for (role,fd,pid) in (("guardian",job.lease.guardian_fd,job.lease.guardian_pid),
                         ("worker",job.lease.worker_fd,C.pid(job)))
        T.send_fd(peer,fd,Dict("role"=>role,"pid"=>pid,"job_id"=>job.job_id,
            "directory"=>job.directory,"request_sha256"=>job.lease.request_sha,
            "owner_pid"=>Int(getpid())))
    end
    sleep(120.) # Outer fixture SIGKILL; no finally runs in that path.
finally
    C.active(job) && C.shutdown!(job;timeout=0.)
    L.close_fd(peer)
end
