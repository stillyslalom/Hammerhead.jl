# Finite owner-loop evidence. This file is included only by --worker-smoke.
# Injected Makie events exercise callbacks; they are not native desktop input.
const worker_command=Observable("")
const worker_ui_acknowledged=Ref("")
const worker_phase=Ref(0)
const worker_metrics=Dict{String,Any}("barrier_kind"=>"injected_worker_work",
    "owner_only_render_thread"=>true)
const worker_owner_thread=Threads.threadid()
const worker_previous_tick=Ref(0.0)
const worker_active_max_gap=Ref(0.0)
const worker_after_ack_max_gap=Ref(0.0)
const worker_first_ack_time=Ref(0.0)
const worker_started_time=Ref(0.0)
const worker_finished_time=Ref(0.0)
const worker_barrier_job=Ref{Any}(nothing)
const worker_prior_display=Ref("")
function worker_stage(name;kwargs...)
    isdefined(Main,:shell_journal) && LifecycleEvidence.stage!(shell_journal,name;kwargs...)
    nothing
end
function worker_ui_ack(command)
    text=String(command) # Qt wrapper never survives this callback
    text==worker_command[] || error("stale worker UI acknowledgement")
    worker_ui_acknowledged[]=text
    nothing
end
function worker_active_capture(success)
    Bool(success) || error("active worker Qt control capture failed")
    worker_assert_barrier("control_frame_captured")
    nothing
end
function worker_barrier_active()
    job=worker_barrier_job[]
    job!==nothing && Prototype.ReplayWorkerClient.active(job) &&
        !isfile(joinpath(job.directory,"injected_barrier_release.toml"))
end
function worker_assert_barrier(action)
    worker_barrier_active() || error("$action did not occur while injected worker barrier was active")
    worker_metrics[action*"_while_barrier_active"]=true
    worker_stage("worker_"*action;job_id=Prototype.ReplayWorkerClient.job_id(worker_barrier_job[]),
        worker_pid=Prototype.ReplayWorkerClient.pid(worker_barrier_job[]),barrier_active=true)
end
function worker_issue(command)
    worker_ui_acknowledged[]=""
    worker_command[]=command
end
function worker_pointer_events()
    worker_assert_barrier("render")
    fig,ax=viewports.active.payload
    # Scene coordinates are converted to event pixels using the actual axis.
    function position(x,y)
        rect=ax.scene.viewport[]; lim=ax.finallimits[]
        fx=(x-lim.origin[1])/lim.widths[1];fy=(y-lim.origin[2])/lim.widths[2]
        (Float64(rect.origin[1]+fx*rect.widths[1]),
            Float64(rect.origin[2]+(ax.yreversed[] ? 1-fy : fy)*rect.widths[2]))
    end
    result=HammerheadGUI.current_result(state.explorer)
    ev=events(fig);ev.entered_window[]=true
    state.explorer.selection[]=nothing
    ev.mouseposition[]=position(first(result.x),first(result.y))
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
    state.explorer.selection[]===nothing && error("injected pick was not serviced")
    worker_assert_barrier("pick")
    original=ax.finallimits[]
    ev.mouseposition[]=position(original.origin[1]+original.widths[1]/2,original.origin[2]+original.widths[2]/2)
    ev.scroll[]=(0.,1.)
    ax.finallimits[].widths[1]<original.widths[1] || error("injected zoom was not serviced")
    worker_assert_barrier("zoom")
    zoomed=ax.finallimits[];px,py=ev.mouseposition[]
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.right,Mouse.press)
    ev.mouseposition[]=(px+10,py+8);ev.mouseposition[]=(px+30,py+24)
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.right,Mouse.release)
    ax.finallimits[].origin!=zoomed.origin || error("injected pan was not serviced")
    worker_assert_barrier("pan")
end
function worker_progress(written,total)
    worker_stage("worker_native_write";job_id=worker_metrics["worker_request_id"],written,total)
    nothing # service_saved_replay! acknowledges after this returns
end
function worker_note_ack()
    if worker_first_ack_time[]==0 && state.experiment.controller.progress[][1]>=1 &&
            state.experiment.pending_progress===nothing && worker_started_time[]>0
        worker_first_ack_time[]=time() # immediately after owner service has written the ACK
    end
end
function worker_tick()
    Threads.threadid()==worker_owner_thread || error("worker evidence moved off render owner thread")
    now=time()
    if worker_started_time[]>0 && worker_finished_time[]==0
        worker_previous_tick[]>0 && (worker_active_max_gap[]=max(worker_active_max_gap[],now-worker_previous_tick[]))
        worker_first_ack_time[]>0 && worker_previous_tick[]>=worker_first_ack_time[] &&
            (worker_after_ack_max_gap[]=max(worker_after_ack_max_gap[],now-worker_previous_tick[]))
    end
    worker_previous_tick[]=now
    phase=worker_phase[];ec=state.experiment.controller
    if phase==0
        @assert open_saved_now(fixture.path)
        @assert Prototype.configure_saved_experiment(state,joinpath(dirname(fixture.output),"prior-vectors.jld2"),joinpath(dirname(fixture.history),"prior-history.jld2"),false)
        @assert Prototype.run_saved_experiment(state)
        worker_phase[]=-1
    elseif phase==-1 && !ec.running[]
        ec.state[]===:completed || error("prior fixture replay failed: $(ec.status[])")
        @assert inspect_saved_now() state.experiment.error[]
        result=HammerheadGUI.current_result(state.explorer)
        Prototype.pick(state,first(result.x),first(result.y))
        worker_prior_display[]=state.displayed[]
        worker_metrics["prior_displayed_run_id"]=ec.last_run[].run_id
        @assert Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        @assert Prototype.run_saved_experiment(state;progress=worker_progress,
            start_options=(worker_script=joinpath(@__DIR__,"worker_lifecycle_barrier.jl"),ack_timeout=120.0))
        job=state.experiment.job;worker_barrier_job[]=job
        worker_metrics["worker_request_id"]=Prototype.ReplayWorkerClient.job_id(job)
        worker_metrics["worker_pid"]=Prototype.ReplayWorkerClient.pid(job)
        worker_metrics["fixture_record_path"]=fixture.path
        worker_metrics["recipe_id"]=fixture.record.recipe.recipe_id
        worker_metrics["input_id"]=fixture.record.input_id
        worker_metrics["image_size"]=[64,64]
        worker_metrics["input_pairs"]=length(fixture.record.pairs)
        worker_metrics["processing_backend"]=String(fixture.record.recipe.backend)
        worker_metrics["processing_threaded"]=fixture.record.recipe.threaded
        worker_metrics["pass_window_sizes"]=[collect(p.window_size) for p in fixture.record.recipe.passes]
        worker_metrics["fixture_scope"]="64x64 shifted PNGs; ROI rows5:60 columns7:62; static mask; highpass and intensity cap; .02mm/pixel and .001s interval"
        worker_started_time[]=time();worker_previous_tick[]=worker_started_time[];worker_phase[]=1
    elseif phase==1
        job=worker_barrier_job[];ready=joinpath(job.directory,"injected_barrier_ready.toml")
        isfile(ready) || return
        data=Prototype.ReplayWorkerClient.WorkerProtocol.read_control(ready)
        Prototype.ReplayWorkerClient.WorkerProtocol.check_keys(data,("job_id","worker_pid","kind"))
        data["job_id"]==worker_metrics["worker_request_id"] && data["worker_pid"]==worker_metrics["worker_pid"] &&
            data["kind"]=="injected_worker_work" || error("crossed injected barrier")
        worker_issue("control");worker_phase[]=2
    elseif phase==2 && worker_ui_acknowledged[]=="control"
        state.displayed[]==worker_prior_display[] || error("active replay replaced the prior displayed identity")
        state.explorer.selection[]!==nothing || error("active replay discarded prior selection")
        worker_metrics["prior_display_preserved_while_barrier_active"]=true
        worker_assert_barrier("control_ack")
        OwnedGLFW.pump!(glfw_owner)===:rendered || error("active worker scientific render failed")
        worker_pointer_events()
        worker_issue("native_close");worker_phase[]=3
    elseif phase==3 && worker_ui_acknowledged[]=="native_close" && viewports.active===nothing
        worker_assert_barrier("close");worker_issue("reopen_1");worker_phase[]=4
    elseif phase==4 && worker_ui_acknowledged[]=="reopen_1" && viewports.active!==nothing
        worker_assert_barrier("reopen");worker_issue("close_2");worker_phase[]=5
    elseif phase==5 && worker_ui_acknowledged[]=="close_2" && viewports.active===nothing
        worker_issue("reopen_2");worker_phase[]=6
    elseif phase==6 && worker_ui_acknowledged[]=="reopen_2" && viewports.active!==nothing
        worker_issue("close_3");worker_phase[]=7
    elseif phase==7 && worker_ui_acknowledged[]=="close_3" && viewports.active===nothing
        worker_issue("reopen_3");worker_phase[]=8
    elseif phase==8 && worker_ui_acknowledged[]=="reopen_3" && viewports.active!==nothing
        worker_assert_barrier("reopen")
        job=worker_barrier_job[]
        Prototype.ReplayWorkerClient.WorkerProtocol.write_control(joinpath(job.directory,"injected_barrier_release.toml"),
            Dict("job_id"=>worker_metrics["worker_request_id"]))
        worker_stage("worker_barrier_released";job_id=worker_metrics["worker_request_id"])
        worker_phase[]=9
    elseif phase==9 && !ec.running[]
        outcome=state.experiment.outcome
        outcome===nothing && error("worker lost its joined outcome")
        outcome.status===:completed || error("worker did not complete: $(outcome.message)")
        worker_finished_time[]=time()
        merge!(worker_metrics,Dict("worker_exit_code"=>outcome.exit_code,
            "worker_terminal_received"=>true,"worker_terminal_status"=>String(outcome.status),
            "worker_closed"=>!Prototype.ReplayWorkerClient.active(worker_barrier_job[]),
            "whole_active_max_pump_gap_seconds"=>worker_active_max_gap[],
            "whole_active_interval_seconds"=>worker_finished_time[]-worker_started_time[],
            "whole_active_timing_scope"=>"owner servicing after start_replay returns to joined terminal; excludes prior fixture replay and owner request capture/spawn",
            "after_first_write_ack_max_pump_gap_seconds"=>worker_after_ack_max_gap[],
            "after_first_write_ack_interval_seconds"=>worker_finished_time[]-worker_first_ack_time[],
            "timing_scope"=>"owner service cycles; first native-write acknowledgment to joined terminal excludes imports and first pair but may include later JIT and I/O"))
        @assert inspect_saved_now() state.experiment.error[]
        ec.last_run[].run_id!=worker_metrics["prior_displayed_run_id"] || error("new replay reused prior displayed run identity")
        worker_metrics["completed_displayed_run_id"]=ec.last_run[].run_id
        @assert navigate_frame_now(3)
        old=state.explorer
        @assert !open_saved_now(joinpath(fixture.output*repeat("-long-path",12),"missing-record.jld2"))
        @assert state.explorer===old
        original_error=state.experiment.error[]
        @assert !isempty(original_error) && startswith(replace(state.experiment.text[],'\n'=>""),"Error: "*first(original_error,24))
        saved_page=state.experiment.page[]
        Prototype.experiment_page(state,-10000)
        text=String[]
        while true
            page=state.experiment.page[]
            push!(text,state.experiment.text[])
            Prototype.experiment_page(state,1)
            state.experiment.page[]==page && break
        end
        Prototype.experiment_page(state,saved_page-state.experiment.page[])
        @assert !isempty(original_error) && occursin(replace(original_error,'\n'=>""),replace(join(text,"\n"),'\n'=>""))
        worker_metrics["full_error_page_contains_original_error"]=true
        retained_display_seen[]=true;completed_replay_seen[]=true
        r=HammerheadGUI.current_result(old);Prototype.pick(state,r.x[1],r.y[1])
        worker_issue("capture_small");worker_phase[]=10
    elseif phase==10 && "small" in sidebar_probe_captures
        worker_issue("capture_large");worker_phase[]=11
    elseif phase==11 && "large" in sidebar_probe_captures
        open(io->TOML.print(io,worker_metrics;sorted=true),joinpath(@__DIR__,"artifacts","worker_report.toml"),"w")
        worker_stage("worker_joined";job_id=worker_metrics["worker_request_id"],worker_pid=worker_metrics["worker_pid"],exit_code=worker_metrics["worker_exit_code"])
        worker_issue("finish");worker_phase[]=12
    end
    nothing
end
