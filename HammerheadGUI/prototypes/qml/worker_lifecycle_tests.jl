using Test
include("worker_lifecycle_runner.jl")

@testset "Worker lifecycle evidence requires acknowledgements and terminal truth" begin
    mktempdir() do directory
        journal=LifecycleEvidence.Journal(joinpath(directory,"stages"))
        for name in ("sidebar_probe_small","sidebar_probe_large")
            LifecycleEvidence.stage!(journal,name;bottom_reachable=true,persistent_controls_visible=true,
                inner_bottom_reachable=true,outer_content_height=900.,outer_viewport_height=600.,
                outer_content_y=300.,outer_max_y=300.,inner_content_height=500.,inner_viewport_height=200.,
                inner_content_y=300.,inner_max_y=300.)
        end
        LifecycleEvidence.stage!(journal,"shell_subscriptions_disposed";remaining=0,replay_running=false)
        LifecycleEvidence.stage!(journal,"child_complete")
        # Signature-only fixture tests the evidence validator, not PNG decoding
        # or successful rendering. Real child captures remain visually inspected.
        png=[UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a];zeros(UInt8,1024)]
        write(joinpath(directory,"scientific.png"),png);write(joinpath(directory,"framebuffer.png"),png)
        for filename in ("worker_sidebar-active.png","worker_sidebar-small.png","worker_sidebar-large.png")
            write(joinpath(directory,filename),png)
        end
        after=Dict("visible"=>false,"screen_active"=>false,"background_rendering"=>false,
            "generations"=>4,"releases"=>4,"frames"=>10,"current_screens"=>0,"baseline_screens"=>0)
        shell=Dict("plot_mode"=>"glfw","visible_desktop"=>false,"scientific_capture_succeeded"=>true,
            "glfw_before_disposal"=>Dict("background_rendering"=>false),"glfw_after_disposal"=>after,
            "figures_still_reachable"=>0,"scientific_sha256"=>LifecycleEvidence.digest(joinpath(directory,"scientific.png")))
        report=Dict{String,Any}("barrier_kind"=>"injected_worker_work","worker_request_id"=>"captured-request",
            "worker_pid"=>1234,"worker_exit_code"=>0,"worker_terminal_status"=>"completed")
        acknowledgements=("control_ack_while_barrier_active","render_while_barrier_active",
            "pick_while_barrier_active","pan_while_barrier_active","zoom_while_barrier_active",
            "close_while_barrier_active","reopen_while_barrier_active","worker_terminal_received",
            "worker_closed","owner_only_render_thread","control_frame_captured_while_barrier_active",
            "full_error_page_contains_original_error")
        for key in acknowledgements
            report[key]=true
        end
        report["captures_sha256"]=Dict(filename=>LifecycleEvidence.digest(joinpath(directory,filename))
            for filename in ("framebuffer.png","scientific.png","worker_sidebar-active.png","worker_sidebar-small.png","worker_sidebar-large.png"))
        publish(data)=open(io->TOML.print(io,data;sorted=true),joinpath(directory,"worker_report.toml"),"w")
        open(io->TOML.print(io,shell;sorted=true),joinpath(directory,"shell_report.toml"),"w")
        publish(report)
        corrupted=deepcopy(report);corrupted["captures_sha256"]["worker_sidebar-active.png"]=repeat("0",64)
        publish(corrupted)
        @test !isempty(validate_worker_evidence(directory))
        publish(report)
        @test isempty(validate_worker_evidence(directory))
        for key in acknowledgements
            candidate=deepcopy(report);candidate[key]=false;publish(candidate)
            @test !isempty(validate_worker_evidence(directory))
        end
        for (key,value) in (("worker_terminal_status","failed"),("worker_exit_code",7),("worker_pid",true),
                            ("barrier_kind","actual_piv"),("worker_request_id",""))
            candidate=deepcopy(report);candidate[key]=value;publish(candidate)
            @test !isempty(validate_worker_evidence(directory))
        end
        publish(report)
        first_probe=first(sort(readdir(journal.directory;join=true)))
        original=TOML.parsefile(first_probe)
        for (key,value) in (("bottom_reachable",false),("persistent_controls_visible",false),
                            ("inner_bottom_reachable",false),("outer_content_y",0.),
                            ("inner_max_y",200.),("inner_content_height",Inf))
            candidate=deepcopy(original);candidate[key]=value
            open(io->TOML.print(io,candidate;sorted=true),first_probe,"w")
            @test !isempty(validate_worker_evidence(directory))
        end
        open(io->TOML.print(io,original;sorted=true),first_probe,"w")
        @test isempty(validate_worker_evidence(directory))
    end
end
