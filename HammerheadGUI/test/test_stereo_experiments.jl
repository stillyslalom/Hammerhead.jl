using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie
const SGEC=HammerheadGUI.Controllers

function gui_stereo_experiment_fixture(directory;rich=true)
    mkpath(directory)
    sizes=((48,52),(52,56))
    cams=(PinholeCamera([100. 0 15. 0;0 100. 0 0;0 0 1. 100.]),
          PinholeCamera([100. 0 -15. 0;0 100. 0 0;0 0 1. 100.]))
    grid=DewarpGrid(x=1.:48.,y=1.:48.)
    dw=map((camera,dims)->ImageDewarper(camera,grid,dims),cams,sizes)
    paths=[joinpath(directory,"camera-$role-frame-$i.png") for role in 1:2,i in 1:4]
    for role in 1:2
        image=Hammerhead.Gray.([mod(37i+13j+7i*j,251)/250 for i in 1:sizes[role][1],j in 1:sizes[role][2]])
        for i in 1:4
            Hammerhead.FileIO.save(paths[role,i],circshift(image,(isodd(i) ? 0 : 1,isodd(i) ? 0 : 2)))
        end
    end
    passes=[PIVParameters(window_size=16,overlap=8,max_iterations=1,uncertainty=true,
        uod_enable=false,validation=(),replace_outliers=false)]
    mask=falses(48,48);mask[:,1:6].=true
    pipelines=([PreprocessStep(:subtract_background;background=fill(.01f0,sizes[1]...)),PreprocessStep(:highpass_filter;sigma=1.)],
               [PreprocessStep(:subtract_background;background=fill(.02f0,sizes[2]...))])
    recipe=StereoPIVRecipe(passes,dw...;threaded=false,image_type=rich ? Float32 : Float64,
        roi=rich ? ROI(5:44,5:44) : nothing,mask=rich ? mask : nothing,
        preprocessing=rich ? pipelines : (PreprocessStep[],PreprocessStep[]),
        scale=PhysicalScale(1.,.125,"mm","s"),world_unit="mm",coordinate_frame="fitted-sheet",
        calibration_note="supplied fit; no accuracy verification")
    pairs=([FramePair(paths[1,i],paths[1,i+1],.125) for i in (1,3)],
           [FramePair(paths[2,i],paths[2,i+1],.125) for i in (1,3)])
    epoch=big(2)^75
    stamps=Rational{BigInt}[epoch//1,(8epoch+1)//8,(8epoch+4)//8,(8epoch+5)//8]
    # Declared dt and timestamps use compatible seconds; large exact origin
    # exercises opaque clock metadata without unsafe Float64 subtraction.
    record=StereoExperimentRecord(pairs...,recipe;timestamps=(stamps,copy(stamps)),
        time_units=("s","s"),clock_ids=("shared-camera-clock","shared-camera-clock"),
        source_ids=("camera1","camera2"),frame_ids=(["A1","B1","A2","B2"],["A1","B1","A2","B2"]))
    (;record,paths,dw,pairs)
end
stereo_gui_button(fig,label)=only(filter(b->b isa Button && startswith(b.label[],label),fig.content))
stereo_gui_controls(fig)=only(filter(b->b isa Menu && ("Files",:files) in b.options[],fig.content))
stereo_gui_body(fig)=only(filter(b->b isa Label && b.fontsize[]==13 && !b.tellheight[],fig.content))
function stereo_gui_choose!(menu,value)
    menu.i_selected[]=findfirst(o->last(o)==value,menu.options[])
end
function stereo_gui_click!(fig,point)
    event=events(fig)
    event.mouseposition[]=Tuple(Float64.(point))
    event.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    event.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end
function stereo_gui_pages(fig)
    pager=only(filter(b->b isa Label && startswith(b.text[],"page "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    previous=stereo_gui_button(fig,"previous");next=stereo_gui_button(fig,"next")
    for _ in 1:count;previous.clicks[]+=1;end
    text=String[]
    for _ in 1:count
        push!(text,stereo_gui_body(fig).text[]);next.clicks[]+=1
    end
    replace(join(text,"\n"),"\n"=>"")
end
function stereo_gui_within(box,width,height)
    box.origin[1]>=-1 && box.origin[2]>=-1 &&
        box.origin[1]+box.widths[1]<=width+1 && box.origin[2]+box.widths[2]<=height+1
end
@noinline function stereo_gui_released(ec)
    explorer=experiment_results(ec;inspect_companions=true)
    ref=WeakRef(current_result(explorer).u)
    set_frame!(explorer,2)
    explorer,ref
end

@testset "Saved stereo snapshot and complete independent controller" begin
    mktempdir() do directory
        f=gui_stereo_experiment_fixture(directory)
        controller=StereoExperimentController(f.record)
        @test controller.record[]!==f.record && controller.record[].recipe._data==f.record.recipe._data
        @test controller.record[].recipe._data["processing"]["image_type"]=="Float32"
        summary=experiment_summary(controller)
        for label in ("processing.passes[1].window_size","preprocessing[1][1].operation","preprocessing[2][1].options.background",
                      "mask:","scale.dt","camera timing[1].source_id","reference_map_sha256")
            @test occursin(label,summary)
        end
        @test occursin("SHA-256",summary) && !occursin("Float32[0.02",summary)
        stored=joinpath(directory,"saved-stereo.jld2")
        save_experiment_record!(controller,stored)
        @test controller.run_record_path[]==stored
        reopened=StereoExperimentController(stored)
        @test reopened.record[].recipe._data==f.record.recipe._data
        before=reopened.record[];history=reopened.run_record_path[]
        wrong=joinpath(directory,"ordinary-results.jld2");save_results(wrong,PIVResult[])
        @test_throws ArgumentError open_experiment!(reopened,wrong)
        @test reopened.record[]===before && reopened.run_record_path[]==history
        @test_throws ArgumentError select_experiment_run!(reopened,"missing")
        @test reopened.selected_run_id[]===nothing
        batch=StereoBatchRunner(files1=collect(f.paths[1,:]),files2=collect(f.paths[2,:]),
            dewarpers=f.dw,window_schedule=[24,16],dt=.01,length_unit="mm",time_unit="s")
        snapshot=experiment_record(batch)
        @test snapshot.scaling_mode===:pair_list && length(snapshot.pairs)==2
        config=Hammerhead._stereo_recipe_config(snapshot.recipe._data)
        @test length(config.processing.passes)==2 && config.scale==build_scale(batch)
        @test config.processing.image_type===Float64 && config.processing.backend===:cpu
        @test config.processing.threaded==(Threads.nthreads()>1)
        set_effort!(batch,:low)
        preset=Hammerhead._stereo_recipe_config(experiment_record(batch).recipe._data).processing.passes
        @test Hammerhead._experiment_pass_data.(preset)==Hammerhead._experiment_pass_data.(Hammerhead.effort_schedule(:low;image_size=(48,48)))
        batch.running[]=true
        @test_throws ArgumentError experiment_record(batch)
        batch.running[]=false;batch.files1[]=[zeros(48,52),zeros(48,52)]
        @test_throws ArgumentError save_batch_experiment(stored,batch)
        @test load_stereo_experiment(stored).recipe._data==f.record.recipe._data
        edited=deepcopy(f.record);edited.recipe._data["grid"]["z"]=1.
        @test_throws ArgumentError open_experiment!(reopened,edited)
        @test reopened.record[]===before
    end
end

@testset "Saved stereo compact sections and actual hidden mouse routing" begin
    mktempdir() do directory
        f=gui_stereo_experiment_fixture(joinpath(directory,"long_recording_path_with_complete_saved_stereo_camera_settings"))
        ec=StereoExperimentController(f.record)
        ec.output_path[]=joinpath(dirname(f.paths[1]),"long_native_output_name_with_recipe_and_run_identity.jld2")
        ec.run_record_path[]=joinpath(dirname(f.paths[1]),"long_saved_stereo_history_name.jld2")
        ec.status[]="failure: "*repeat("complete-readable-cause-0123456789abcdef;",6)
        fig=stereo_experiment_workflow(ec;report_path_picker=()->"")
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        sections=stereo_gui_controls(fig)
        toggle=filter(b->b isa Toggle,fig.content)
        @test length(toggle)==5
        for dims in ((900,600),(1100,800))
            resize!(fig.scene,dims...)
            for section in (:files,:replay,:reports)
                stereo_gui_choose!(sections,section);colorbuffer(screen)
                @test size(colorbuffer(screen))==reverse(dims)
                for block in fig.content
                    block.blockscene.visible[] || continue
                    @test stereo_gui_within(block.layoutobservables.computedbbox[],dims...)
                end
                if haskey(ENV,"HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT")
                    path=ENV["HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT"]*"-$(dims[1])-$(section).png"
                    Hammerhead.FileIO.save(path,copy(colorbuffer(screen)))
                end
            end
        end
        stereo_gui_choose!(sections,:replay);colorbuffer(screen)
        record_toggle=toggle[2]
        box=record_toggle.layoutobservables.computedbbox[]
        point=box.origin.+box.widths./2
        stereo_gui_click!(fig,point)
        @test ec.record_diagnostics[]
        stereo_gui_choose!(sections,:files);colorbuffer(screen)
        @test record_toggle.layoutobservables.computedbbox[].origin[1]<-5000
        stereo_gui_click!(fig,point)
        @test ec.record_diagnostics[] # parked hidden region cannot change choice
        @test occursin(recipe_identity(ec.record[].recipe),stereo_gui_pages(fig))
        cancel=stereo_gui_button(fig,"cancel after acquisition")
        ec._cancel_token[]=Ref(false);ec.running[]=true
        box=cancel.layoutobservables.computedbbox[]
        stereo_gui_click!(fig,box.origin.+box.widths./2)
        @test ec._cancel_token[][] && ec.state[]===:cancel_requested
        SGEC._experiment_replay_cleanup!(ec)
        GLMakie.destroy!(screen);GLMakie.Makie.current_figure!(nothing)
        batch_figure=stereo_batch_runner(;size=(960,600))
        batch_screen=GLMakie.Screen(batch_figure.scene;visible=false,start_renderloop=false)
        colorbuffer(batch_screen)
        launch=stereo_gui_button(batch_figure,"saved stereo workflow")
        @test stereo_gui_within(launch.layoutobservables.computedbbox[],960,600)
        @test launch.fontsize[]==12
        if haskey(ENV,"HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT")
            Hammerhead.FileIO.save(ENV["HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT"]*"-960-batch-launch.png",copy(colorbuffer(batch_screen)))
        end
        GLMakie.destroy!(batch_screen);GLMakie.Makie.current_figure!(nothing)
    end
end

if get(ENV,"HAMMERHEAD_STEREO_GUI_STATE_ONLY","false")!="true"
@testset "Strict saved stereo replay, cancellation, verified history and reports" begin
    mktempdir() do directory
        f=gui_stereo_experiment_fixture(directory)
        output=joinpath(directory,"completed-stereo.jld2");history=joinpath(directory,"run-history.jld2")
        ec=StereoExperimentController(f.record;output_path=output,run_record_path=history,
            record_diagnostics=true,record_pair_timing=true)
        reentered=Ref(false)
        observer=on(ec.running) do busy
            busy || return
            ec.output_path[]=joinpath(directory,"next-choice.jld2")
            ec.record_diagnostics[]=false;ec.record_pair_timing[]=false
            @test ec.active_request[].output==output && ec.active_request[].record_diagnostics && ec.active_request[].record_pair_timing
            @test_throws ArgumentError open_experiment!(ec,f.record)
            start!(ec;async=false);reentered[]=true
        end
        start!(ec;async=false)
        off(observer)
        @test reentered[] && ec.state[]===:completed && !ec.running[]
        @test ec.error[]===nothing && ec.progress[]==(2,2) && ec.last_run[].output==output
        @test ec.active_request[]===nothing
        @test !isfile(joinpath(directory,"next-choice.jld2"))
        first_run=ec.last_run[];first_id=first_run.run_id
        @test ec.selected_run_id[]==first_id
        index=ResultFile(output)
        @test load_stereo_execution_diagnostics(index,1)!==nothing && load_stereo_pair_timing(index,1)!==nothing
        ex=experiment_results(ec;inspect_companions=true,verify_inputs=true)
        raw=index[1]
        @test isequal(current_result(ex).u,physical(raw).u)
        @test occursin("raw reconstructed and camera measurement binding verified",companion_summary(ex))
        @test current_result(ex).cam1.scale===nothing
        released,ref=stereo_gui_released(ec);GC.gc(true);GC.gc(true)
        @test ref.value===nothing && released.frame[]==2
        @test length(released.derived_cache)<=1
        report=experiment_quality_report(ec;include_execution_diagnostics=true)
        data=quality_report_data(report)
        @test data["quality_report_format_version"]==3 && data["provenance"]["run_id"]==first_id
        @test_throws ArgumentError experiment_quality_report(ec;include_measurement_history=true)
        for protected in (output,history,f.paths[1])
            bytes=read(protected)
            @test_throws ArgumentError save_experiment_quality_report(protected,ec;include_execution_diagnostics=true)
            @test read(protected)==bytes
        end
        # Cancellation is a failed v1 run, and selection retains the older success.
        ec.output_path[]=joinpath(directory,"cancelled-prefix.jld2")
        cancel_fig=stereo_experiment_workflow(ec;size=(900,600))
        cancel_screen=GLMakie.Screen(cancel_fig.scene;visible=false,start_renderloop=false)
        colorbuffer(cancel_screen)
        cancel_button=stereo_gui_button(cancel_fig,"cancel after acquisition")
        start!(ec;async=false,progress=(i,n)->begin
            @test ec.running[] && ec.progress[]==(i,n)
            box=cancel_button.layoutobservables.computedbbox[]
            stereo_gui_click!(cancel_fig,box.origin.+box.widths./2)
            @test ec.state[]===:cancel_requested && ec.running[]
        end)
        @test ec.state[]===:cancelled && ec.progress[]==(1,2) && !ec.running[]
        @test ec.last_run[].status===:failed && ec.last_run[].completed_pairs==1
        @test ec.selected_run_id[]==first_id && length(ec.record[].runs)==2
        @test length(ResultFile(ec.last_run[].output))==1
        if haskey(ENV,"HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT")
            Hammerhead.FileIO.save(ENV["HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT"]*"-900-cancelled.png",copy(colorbuffer(cancel_screen)))
        end
        GLMakie.destroy!(cancel_screen);GLMakie.Makie.current_figure!(nothing)
        failed_id=ec.last_run[].run_id
        select_experiment_run!(ec,failed_id)
        @test_throws ArgumentError experiment_results(ec)
        @test_throws ArgumentError experiment_quality_report(ec)
        select_experiment_run!(ec,first_id)
        @test current_result(experiment_results(ec)) isa StereoPIVResult
        ec.output_path[]=joinpath(directory,"final-boundary.jld2")
        start!(ec;async=false,progress=(i,n)->i==n && cancel!(ec))
        @test ec.state[]===:completed && ec.last_run[].completed_pairs==2
        @test ec.selected_run_id[]==first_id
        # Startup observer failures cannot strand request/task/busy state.
        broken=StereoExperimentController(f.record;output_path=joinpath(directory,"never-started.jld2"))
        err=ErrorException("startup-observer-marker")
        subscription=on(_->throw(err),broken.running)
        start!(broken;async=false)
        @test broken.error[]===err && broken.state[]===:failed && !broken.running[]
        @test broken._task[]===nothing && broken._cancel_token[]===nothing && !isfile(broken.output_path[])
        @test broken.active_request[]===nothing
        off(subscription)
        # Terminal observer failures cannot replace the original processing error.
        failed=StereoExperimentController(f.record;output_path=joinpath(directory,"observer-failure.jld2"),
            run_record_path=joinpath(directory,"observer-failure-history.jld2"))
        primary=ErrorException("stereo-progress-original-marker")
        terminal=on(failed.state) do state
            state===:failed && error("terminal-observer-secondary-marker")
        end
        closing=on(failed.running) do busy
            busy || error("cleanup-observer-secondary-marker")
        end
        start!(failed;async=false,progress=(i,n)->throw(primary))
        @test failed.error[]===primary && failed.state[]===:failed && !failed.running[]
        @test failed.active_request[]===nothing && failed._cancel_token[]===nothing && failed._task[]===nothing
        @test failed.last_run[].status===:failed && failed.last_run[].completed_pairs==1
        @test occursin("stereo-progress-original-marker",failed.last_run[].error)
        @test length(ResultFile(failed.last_run[].output))==1
        off(terminal);off(closing)
        # Async cancellation before scheduling execution leaves artifacts unchanged.
        staged=StereoExperimentController(f.record;output_path=joinpath(directory,"pre-cancel.jld2"),run_record_path=joinpath(directory,"pre-cancel-history.jld2"))
        start!(staged);task=staged._task[];cancel!(staged);wait(task)
        @test staged.state[]===:cancelled && !staged.running[]
        @test !isfile(staged.output_path[]) && !isfile(staged.run_record_path[])

        destination=Ref(joinpath(directory,"selected-run-quality.toml"));calls=Ref(0);toggles=Ref{Any}(nothing)
        picker=()->begin
            calls[]+=1
            # The dialog changes next choices; request identity/options are frozen.
            select_experiment_run!(ec,failed_id)
            toggles[][4].active[]=false;toggles[][5].active[]=false
            destination[]
        end
        fig=stereo_experiment_workflow(ec;report_path_picker=picker)
        toggles[]=filter(b->b isa Toggle,fig.content)
        toggles[][4].active[]=true;toggles[][5].active[]=true
        save_button=stereo_gui_button(fig,"save quality report")
        save_button.clicks[]+=1
        @test calls[]==1 && ec.status[]=="quality report saved"
        @test quality_report_data(load_quality_report(destination[]))["provenance"]["run_id"]==first_id
        @test quality_report_data(load_quality_report(destination[]))["quality_report_format_version"]==3
        @test occursin("Reported run: "*first_id,stereo_gui_pages(fig))
        destination[]=output;bytes=read(output)
        save_button.clicks[]+=1
        @test occursin("failed",ec.status[]) && read(output)==bytes
        @test occursin("Reported run: "*first_id,stereo_gui_pages(fig))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        controls=stereo_gui_controls(fig)
        for dims in ((900,600),(1100,800))
            resize!(fig.scene,dims...);stereo_gui_choose!(controls,:reports);colorbuffer(screen)
            for _ in 1:20;stereo_gui_button(fig,"previous").clicks[]+=1;end
            colorbuffer(screen)
            @test stereo_gui_within(stereo_gui_body(fig).layoutobservables.computedbbox[],dims...)
            if haskey(ENV,"HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT")
                Hammerhead.FileIO.save(ENV["HAMMERHEAD_STEREO_WORKFLOW_SCREENSHOT"]*"-$(dims[1])-retained-report.png",copy(colorbuffer(screen)))
            end
        end
        GLMakie.destroy!(screen);GLMakie.Makie.current_figure!(nothing)
        # Existing displayed output remains independent when current inputs change.
        select_experiment_run!(ec,first_id)
        bytes=read(f.paths[1]);write(f.paths[1],reverse(bytes))
        @test_throws ArgumentError experiment_results(ec;verify_inputs=true)
        @test current_result(ex) isa StereoPIVResult
        write(f.paths[1],bytes)
    end
end
end
