# Explicit injected work for owner-loop concurrency evidence. This waiting
# interval is never described as PIV processing or a responsiveness benchmark.
module WorkerBarrierBootstrap
include("worker_protocol.jl")
using .WorkerProtocol

function await_release(args)
    length(args)==1 || error("worker barrier expects one captured request")
    request_file=abspath(only(args));directory=dirname(request_file)
    request=WorkerProtocol.request_data(WorkerProtocol.read_control(request_file))
    deadline=time()+30.
    enrollment=joinpath(directory,"enrolled.toml")
    while !isfile(enrollment)
        time()<deadline || error("injected worker enrollment deadline expired")
        sleep(.01)
    end
    owner=WorkerProtocol.read_control(enrollment)
    owner["job_id"]==request["job_id"] && owner["owner_pid"]===request["owner_pid"] &&
        owner["worker_pid"]===Int(getpid()) && owner["ownership"]==
            (Sys.islinux() ? "linux_pidfd_subreaper_guardian" : "windows_kill_on_close_job") &&
        owner["request_sha256"]==WorkerProtocol.digest(request_file) || error("injected worker enrollment identity disagrees")
    for (path,hash) in request["worker_sources"]
        WorkerProtocol.digest(path)==hash || error("worker source changed before injected barrier")
    end
    WorkerProtocol.write_control(joinpath(directory,"injected_barrier_ready.toml"),Dict(
        "job_id"=>request["job_id"],"worker_pid"=>Int(getpid()),"kind"=>"injected_worker_work"))
    release=joinpath(directory,"injected_barrier_release.toml")
    deadline=time()+request["ack_timeout"]
    while !isfile(release)
        time()<deadline || error("injected worker release deadline expired")
        sleep(.01)
    end
    released=WorkerProtocol.read_control(release)
    WorkerProtocol.check_keys(released,("job_id",))
    released["job_id"]==request["job_id"] || error("crossed injected worker release")
    nothing
end
end

WorkerBarrierBootstrap.await_release(ARGS)
include("replay_worker.jl")
