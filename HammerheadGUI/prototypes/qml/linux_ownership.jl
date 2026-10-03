module LinuxOwnership
using ..WorkerProtocol
const GROUP = UInt32(4)
const ECHILD = 10
function open_pidfd(pid)
    fd=ccall(:pidfd_open,Cint,(Cint,Cuint),pid,0)
    fd>=0 || error("pidfd_open failed errno=$(Libc.errno())")
    Int(fd)
end
function signal_fd(fd,signal;group=false)
    rc=ccall(:pidfd_send_signal,Cint,(Cint,Cint,Ptr{Cvoid},Cuint),fd,signal,C_NULL,group ? GROUP : UInt32(0))
    (;rc=Int(rc),errno=rc==0 ? 0 : Int(Libc.errno()))
end
close_fd(fd)=fd<0 ? nothing : ccall(:close,Cint,(Cint,),fd)
struct PollFD
    fd::Cint
    events::Int16
    revents::Int16
end
function exited(fd)
    p=Ref(PollFD(fd,1,0))
    rc=ccall(:poll,Cint,(Ptr{PollFD},Culong,Cint),p,1,0)
    rc>=0 || error("pidfd poll failed errno=$(Libc.errno())")
    rc==1 && (p[].revents & 1)!=0
end
function group_empty(fd)
    rc=signal_fd(fd,0;group=true)
    rc.rc==0 && return false
    rc.errno==3 || error("bound group inspection failed errno=$(rc.errno)")
    true
end
function subreaper!()
    ccall(:prctl,Cint,(Cint,Culong,Culong,Culong,Culong),36,1,0,0,0)==0 || error("CHILD_SUBREAPER unavailable")
end
function descriptor(data,request,role,pid,request_sha)
    WorkerProtocol.check_keys(data,("version","job_id","owner_pid","role","pid","request_sha256"))
    data["version"]===1 && data["job_id"]==request["job_id"] && data["owner_pid"]===request["owner_pid"] &&
        data["role"]==role && data["pid"]===pid && data["request_sha256"]==request_sha || error("crossed pidfd descriptor")
    data
end
function descriptor_packet(request,role,pid,request_sha)
    Dict("version"=>1,"job_id"=>request["job_id"],"owner_pid"=>request["owner_pid"],"role"=>role,"pid"=>Int(pid),"request_sha256"=>request_sha)
end
function descriptor_pid(fd)
    text=read("/proc/self/fdinfo/$fd",String)
    matchpid=match(r"(?m)^Pid:\s*(-?\d+)$",text)
    matchpid===nothing && error("received descriptor is not a pidfd")
    parse(Int,matchpid.captures[1])
end
function no_children()
    st=Ref{Cint}(0)
    while true
        rc=ccall(:waitpid,Cint,(Cint,Ptr{Cint},Cint),-1,st,1)
        rc>0 && continue
        rc==0 && return false
        Libc.errno()==ECHILD || error("subreaper wait failed errno=$(Libc.errno())")
        return true
    end
end
function ready_data(data,request,guardian_pid,request_sha)
    WorkerProtocol.check_keys(data,("version","job_id","owner_pid","guardian_pid","worker_pid","request_sha256","group_signal_supported"))
    data["version"]===1 && data["job_id"]==request["job_id"] && data["owner_pid"]===request["owner_pid"] &&
        data["guardian_pid"]===guardian_pid && data["worker_pid"] isa Int && data["worker_pid"]>0 &&
        data["request_sha256"]==request_sha && data["group_signal_supported"]===true || error("crossed/malformed guardian enrollment")
    data
end
function proof_data(data,request,guardian_pid,worker_pid,request_sha)
    WorkerProtocol.check_keys(data,("version","job_id","owner_pid","guardian_pid","worker_pid","request_sha256",
        "root_reaped","group_empty","children_echild","worker_exit_code","worker_term_signal","cleanup_confirmed","stopped_by_owner","residual_group_killed"))
    data["version"]===1 && data["job_id"]==request["job_id"] && data["owner_pid"]===request["owner_pid"] &&
        data["guardian_pid"]===guardian_pid && data["worker_pid"]===worker_pid && data["request_sha256"]==request_sha || error("crossed guardian cleanup identity")
    all(k->data[k]===true,("root_reaped","group_empty","children_echild","cleanup_confirmed")) || error("guardian cleanup not confirmed")
    data["worker_exit_code"] isa Int && 0<=data["worker_exit_code"]<=255 &&
        data["worker_term_signal"] isa Int && 0<=data["worker_term_signal"]<=127 &&
        data["stopped_by_owner"] isa Bool && data["residual_group_killed"] isa Bool || error("malformed guardian cleanup fields")
    data
end
end
