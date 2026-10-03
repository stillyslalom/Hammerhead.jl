# Separate bounded worker lane; existing native and software gates are unchanged.
include("lifecycle_runner.jl")

function validate_worker_evidence(directory)
    errors=String[]
    try
        report=TOML.parsefile(joinpath(directory,"worker_report.toml"))
        shell=TOML.parsefile(joinpath(directory,"shell_report.toml"))
        stages=[TOML.parsefile(path) for path in sort(readdir(joinpath(directory,"stages");join=true))]
        require(condition,message)=condition || push!(errors,message)
        flag(key)=get(report,key,nothing)===true
        require(get(report,"barrier_kind",nothing)=="injected_worker_work","active barrier must be explicitly injected")
        for key in ("control_ack_while_barrier_active","render_while_barrier_active",
                    "pick_while_barrier_active","pan_while_barrier_active","zoom_while_barrier_active",
                    "close_while_barrier_active","reopen_while_barrier_active",
                    "worker_terminal_received","worker_closed","owner_only_render_thread",
                    "control_frame_captured_while_barrier_active","full_error_page_contains_original_error")
            require(flag(key),"missing worker acknowledgement: $key")
        end
        require(get(report,"worker_request_id",nothing) isa String && !isempty(report["worker_request_id"]),"missing captured request identity")
        require(get(report,"worker_pid",nothing) isa Int && report["worker_pid"]>0,"missing worker PID")
        require(get(report,"worker_exit_code",nothing)===0,"worker did not exit cleanly")
        require(get(report,"worker_terminal_status",nothing)=="completed","worker terminal status is not completed")
        # Exit zero alone never substitutes for a validated terminal envelope.
        for name in ("sidebar_probe_small","sidebar_probe_large")
            probes=filter(s->get(s,"stage",nothing)==name,stages)
            require(length(probes)==1,"missing/duplicate $name")
            length(probes)==1 || continue
            probe=only(probes)
            for key in ("bottom_reachable","persistent_controls_visible","inner_bottom_reachable")
                require(get(probe,key,nothing)===true,"$name failed $key")
            end
            for prefix in ("outer","inner")
                values=[get(probe,"$(prefix)_$suffix",nothing) for suffix in ("content_height","viewport_height","content_y","max_y")]
                require(all(v->v isa Real && !(v isa Bool) && isfinite(v) && v>=0,values),"$name invalid $prefix scroll measurements")
                if all(v->v isa Real && !(v isa Bool) && isfinite(v) && v>=0,values)
                    content,viewport,position,maximum=values
                    require(abs(maximum-max(0,content-viewport))<=2,"$name inconsistent $prefix extent")
                    require(abs(position-maximum)<=2,"$name did not reach $prefix bottom")
                end
            end
        end
        require(get(shell,"plot_mode",nothing)=="glfw" && get(shell,"visible_desktop",nothing)===false,"worker lane must use hidden owned GLFW")
        require(get(shell,"scientific_capture_succeeded",nothing)===true,"scientific capture failed")
        before=shell["glfw_before_disposal"];after=shell["glfw_after_disposal"]
        require(after["visible"]===false && after["screen_active"]===false &&
            before["background_rendering"]===false && after["background_rendering"]===false,
            "owned scientific screen remains live or used background rendering")
        require(after["generations"] isa Int && after["generations"]>=2 &&
            after["releases"]===after["generations"] && after["frames"] isa Int && after["frames"]>0 &&
            after["current_screens"]===after["baseline_screens"] && shell["figures_still_reachable"]===0,
            "viewport lifetime counts are incomplete")
        for filename in ("framebuffer.png","scientific.png","worker_sidebar-active.png",
                         "worker_sidebar-small.png","worker_sidebar-large.png")
            require(nonempty_png(joinpath(directory,filename)),"missing/invalid $filename")
            require(get(get(report,"captures_sha256",Dict()),filename,nothing)==LifecycleEvidence.digest(joinpath(directory,filename)),
                "capture digest differs: $filename")
        end
        require(LifecycleEvidence.digest(joinpath(directory,"scientific.png"))==shell["scientific_sha256"],"scientific capture digest differs")
        disposed=filter(s->get(s,"stage",nothing)=="shell_subscriptions_disposed",stages)
        require(length(disposed)==1 && only(disposed)["remaining"]===0 && only(disposed)["replay_running"]===false,
            "shell callbacks or replay remain live")
        require(count(s->get(s,"stage",nothing)=="child_complete",stages)==1,"missing terminal child stage")
    catch exception
        push!(errors,sprint(showerror,exception))
    end
    errors
end

function worker_lifecycle_main(args=ARGS)
    option(name,default)=begin
        matches=filter(a->startswith(a,"--$name="),args)
        length(matches)<=1 || error("duplicate option $name")
        isempty(matches) ? default : split(only(matches),'=';limit=2)[2]
    end
    timeout=parse(Float64,option("timeout","360"))
    isfinite(timeout) && timeout>0 || error("timeout must be finite and positive")
    artifacts=joinpath(@__DIR__,"artifacts");mkpath(artifacts)
    root=mktempdir(artifacts;prefix="worker-lifecycle-",cleanup=false)
    println("Evidence directory: ",root);flush(stdout)
    directory=joinpath(root,"worker-active-glfw")
    command=`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $(joinpath(@__DIR__,"worker_lifecycle_child.jl")) --evidence=$directory`
    # Select software controls at process creation, before Qt's runtime loads.
    result=run_lifecycle_child(command,directory;timeout,expected=Dict("scenario"=>"shell","cycles"=>1,"plot"=>"glfw"))
    # Generic shell coverage is checked by the existing owner; worker assertions
    # additionally require actual acknowledgements and independent terminal truth.
    errors=validate_worker_evidence(directory)
    result["worker_evidence_errors"]=errors
    result["passed"]=result["passed"] && isempty(errors)
    # The Qt child's process handle is not the replay grandchild's handle.
    # Only a validated clean child report acknowledges worker reaping. An
    # outer timeout/error never implies that the descendant was disposed.
    result["descendant_cleanup_verified"]=result["passed"]
    if !result["descendant_cleanup_verified"]
        LifecycleEvidence.write_fresh(joinpath(root,"incomplete_owner.toml"),Dict(
            "descendant_cleanup_verified"=>false,"subsequent_launches_allowed"=>false,
            "reason"=>"Clean terminal worker lifetime was not verified; no further children may be launched"))
    end
    LifecycleEvidence.write_fresh(joinpath(root,"summary.toml"),Dict("case"=>result,"all_passed"=>result["passed"],
        "descendant_cleanup_verified"=>result["descendant_cleanup_verified"],
        "subsequent_launches_allowed"=>result["descendant_cleanup_verified"],
        "scope"=>"hidden Qt software controls and owned GLFW; injected-barrier concurrency separate from actual-work observations"))
    println("worker-active-glfw: passed=",result["passed"]," exit=",result["exit_code"]," timeout=",result["timed_out"])
    result["passed"] || exit(1)
    root
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    worker_lifecycle_main()
end
