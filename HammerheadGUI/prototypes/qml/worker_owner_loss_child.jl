# Test owner: abrupt termination must close its non-inheritable native Job handle.
# No Qt/GLFW module is loaded and no process other than this owner's child is managed.
using Hammerhead
include("worker_client.jl")
using .ReplayWorkerClient: start_replay
record_file,worker_script,artifact_root,announcement=ARGS
job=start_replay(load_experiment(record_file);output=joinpath(artifact_root,"unused-output.jld2"),
    artifact_root,worker_script,ack_timeout=60.)
ReplayWorkerClient.WorkerProtocol.write_control(announcement,Dict(
    "owner_pid"=>Int(getpid()),"worker_pid"=>job.worker_pid,"job_id"=>job.job_id,
    "directory"=>job.directory,"ownership"=>"windows_kill_on_close_job"))
while true
    sleep(.1)
end
