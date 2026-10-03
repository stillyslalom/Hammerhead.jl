using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie
const EGEC=HammerheadGUI.Controllers

function gui_ensemble_fixture(directory;backend=:cpu,image_type=Float32)
    mkpath(directory)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    paths=[joinpath(directory,"frame-$i.png") for i in 1:4]
    for (i,path) in enumerate(paths)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,isodd(i) ? (0,0) : (1,2))))
    end
    mask=falses(48,48);mask[:,1:6].=true
    passes=[PIVParameters(window_size=24,overlap=12,max_iterations=7,convergence_tol=Inf,
        uod_enable=false,validation=(),replace_outliers=false),
        PIVParameters(window_size=16,overlap=8,max_iterations=3,convergence_tol=.123,
        uncertainty=true,uod_enable=false,validation=(),replace_outliers=false)]
    steps=[PreprocessStep(:subtract_background;background=fill(.01f0,48,48)),PreprocessStep(:highpass_filter;sigma=1.)]
    recipe=EnsemblePIVRecipe(passes;preprocessing=steps,mask,
        scale=PhysicalScale(.02,.001,"mm","s"),backend,image_type,threaded=false)
    record=EnsembleExperimentRecord([(paths[1],paths[2]),(paths[3],paths[4])],recipe)
    (;record,paths,image)
end
ensemble_gui_button(fig,label)=only(filter(b->b isa Button && startswith(b.label[],label),fig.content))
ensemble_gui_controls(fig)=only(filter(b->b isa Menu && ("Files",:files) in b.options[],fig.content))
ensemble_gui_body(fig)=only(filter(b->b isa Label && b.fontsize[]==13 && !b.tellheight[],fig.content))
function ensemble_gui_choose!(menu,value)
    menu.i_selected[]=findfirst(o->last(o)==value,menu.options[])
end
function ensemble_gui_click!(fig,point)
    event=events(fig);event.mouseposition[]=Tuple(Float64.(point))
    event.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    event.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end
ensemble_gui_within(box,width,height)=box.origin[1]>=-1 && box.origin[2]>=-1 &&
    box.origin[1]+box.widths[1]<=width+1 && box.origin[2]+box.widths[2]<=height+1
function ensemble_gui_pages(fig)
    pager=only(filter(b->b isa Label && startswith(b.text[],"page "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    previous=ensemble_gui_button(fig,"previous");next=ensemble_gui_button(fig,"next")
    for _ in 1:count;previous.clicks[]+=1;end
    text=String[]
    for _ in 1:count
        push!(text,ensemble_gui_body(fig).text[]);next.clicks[]+=1
    end
    replace(join(text,"\n"),"\n"=>"")
end
function ensemble_gui_capture(screen,suffix)
    haskey(ENV,"HAMMERHEAD_ENSEMBLE_WORKFLOW_SCREENSHOT") || return
    Hammerhead.FileIO.save(ENV["HAMMERHEAD_ENSEMBLE_WORKFLOW_SCREENSHOT"]*suffix*".png",copy(colorbuffer(screen)))
end

@testset "Saved ensemble exact snapshot and imported settings" begin
    mktempdir() do directory
        f=gui_ensemble_fixture(directory)
        ec=EnsembleExperimentController(f.record)
        @test ec.record[]!==f.record && recipe_identity(ec.record[].recipe)==recipe_identity(f.record.recipe)
        @test ec.record[].recipe.passes[1].max_iterations==7 && ec.record[].recipe.passes[1].convergence_tol==Inf
        text=experiment_summary(ec)
        for label in ("passes[1].window_size","passes[1].max_iterations","passes[1].convergence_tol","preprocessing[1].options.background","scale.dt","input pair 2","ignored")
            @test occursin(label,text)
        end
        @test occursin("SHA-256",text) && !occursin("Float32[0.01",text)
        saved=joinpath(directory,"recipe.jld2");save_experiment_record!(ec,saved)
        reopened=EnsembleExperimentController(saved)
        @test Hammerhead._ensemble_recipe_data(reopened.record[].recipe)==Hammerhead._ensemble_recipe_data(f.record.recipe)
        prior=reopened.record[];wrong=joinpath(directory,"ordinary.jld2");save_results(wrong,PIVResult[])
        @test_throws ArgumentError open_experiment!(reopened,wrong)
        @test reopened.record[]===prior && reopened.run_record_path[]==saved
        @test_throws ArgumentError select_experiment_run!(reopened,"unknown")
        modified=deepcopy(f.record);modified.recipe.mask[1]=!modified.recipe.mask[1]
        @test_throws ArgumentError open_experiment!(reopened,modified)
        @test reopened.record[]===prior
        bc=BatchRunner(files=f.paths,window_schedule=[24,16],mask=falses(48,48),
            pixel_size=.02,dt=.001,length_unit="mm",time_unit="s")
        pp=PreprocessPreview(f.paths[1];enabled=[:highpass_filter])
        EGEC.set_background!(pp,f.paths[1:2];method=:min);enable_step!(pp,:subtract_background)
        set_preprocess!(bc,pp)
        snapshot=ensemble_experiment_record(bc;backend=:ka,image_type=Float32,threaded=false)
        @test snapshot.recipe.backend===:ka && snapshot.recipe.image_type===Float32
        @test snapshot.recipe.preprocessing[1].options["background"]!==pp.background[]
        @test snapshot.recipe.scale==build_scale(bc)
        @test Hammerhead._experiment_pass_data.(snapshot.recipe.passes)==Hammerhead._experiment_pass_data.(EGEC.build_parameters(bc))
        set_effort!(bc,:high)
        @test Hammerhead._experiment_pass_data.(ensemble_experiment_record(bc).recipe.passes)==
            Hammerhead._experiment_pass_data.(Hammerhead.effort_schedule(:high;ensemble=true,image_size=(48,48)))
        bc.roi[]=ROI(1:40,1:40);@test_throws ArgumentError ensemble_experiment_record(bc);bc.roi[]=nothing
        bc.running[]=true;@test_throws ArgumentError ensemble_experiment_record(bc);bc.running[]=false
        bc.preprocess[]=identity;@test_throws ArgumentError ensemble_experiment_record(bc)
        set_preprocess!(bc,pp);bc.preprocess_snapshot[].steps[1].options["background"][1]=.9
        @test_throws ArgumentError ensemble_experiment_record(bc)
        @test_throws ArgumentError ensemble_experiment_record(BatchRunner(files=[f.image,f.image]))
        @test_throws ArgumentError ensemble_experiment_record(BatchRunner())
        bytes=read(saved);@test_throws ArgumentError save_ensemble_batch_experiment(saved,bc);@test read(saved)==bytes
    end
end

@testset "Saved ensemble compact sections, actual mouse, discoverable launch" begin
    mktempdir() do directory
        f=gui_ensemble_fixture(joinpath(directory,"long_recording_directory_with_full_saved_ensemble_processing_settings"))
        ec=EnsembleExperimentController(f.record;output_path=joinpath(directory,"long_complete_pooled_output.jld2"),run_record_path=joinpath(directory,"long_complete_run_history.jld2"))
        ec.status[]="failure: "*repeat("complete-long-cause-0123456789abcdef;",6)
        fig=ensemble_experiment_workflow(ec;report_path_picker=()->"")
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        sections=ensemble_gui_controls(fig)
        for dims in ((900,600),(1100,800))
            resize!(fig.scene,dims...)
            for section in (:files,:replay,:reports)
                ensemble_gui_choose!(sections,section);colorbuffer(screen)
                @test size(colorbuffer(screen))==reverse(dims)
                for block in fig.content
                    block.blockscene.visible[] || continue
                    ensemble_gui_within(block.layoutobservables.computedbbox[],dims...) ||
                        @warn "ensemble block outside allocation" dims section block=typeof(block) box=block.layoutobservables.computedbbox[]
                    @test ensemble_gui_within(block.layoutobservables.computedbbox[],dims...)
                end
                ensemble_gui_capture(screen,"-$(dims[1])-$(section)")
            end
        end
        ensemble_gui_choose!(sections,:replay);colorbuffer(screen)
        toggles=filter(b->b isa Toggle,fig.content);@test length(toggles)==4
        box=toggles[2].layoutobservables.computedbbox[];point=box.origin.+box.widths./2
        ensemble_gui_click!(fig,point);@test ec.record_diagnostics[]
        ensemble_gui_choose!(sections,:files);colorbuffer(screen)
        ensemble_gui_click!(fig,point);@test ec.record_diagnostics[]
        @test toggles[2].layoutobservables.computedbbox[].origin[1]<-5000
        @test occursin(recipe_identity(ec.record[].recipe),ensemble_gui_pages(fig))
        ec._cancel_token[]=Ref(false);ec.running[]=true
        cancel=ensemble_gui_button(fig,"cancel between contributions");box=cancel.layoutobservables.computedbbox[]
        ensemble_gui_click!(fig,box.origin.+box.widths./2)
        @test ec._cancel_token[][] && ec.state[]===:cancel_requested && ec.running[]
        EGEC._experiment_replay_cleanup!(ec)
        GLMakie.destroy!(screen);GLMakie.Makie.current_figure!(nothing)
        launched=Ref{Any}(nothing)
        batch=BatchRunner(files=f.paths,window_schedule=[16])
        batchfig=batch_runner(batch;ensemble_workflow_launcher=figure->(launched[]=figure))
        batchscreen=GLMakie.Screen(batchfig.scene;visible=false,start_renderloop=false);colorbuffer(batchscreen)
        button=ensemble_gui_button(batchfig,"saved ensemble")
        box=button.layoutobservables.computedbbox[];@test ensemble_gui_within(box,960,720)
        ensemble_gui_click!(batchfig,box.origin.+box.widths./2)
        @test launched[] isa Figure
        @test any(b->b isa Button && startswith(b.label[],"snapshot ensemble batch"),launched[].content)
        ensemble_gui_capture(batchscreen,"-960-batch-launch")
        GLMakie.destroy!(batchscreen);GLMakie.Makie.current_figure!(nothing)
    end
end

if get(ENV,"HAMMERHEAD_ENSEMBLE_GUI_STATE_ONLY","false")!="true"
@testset "Strict ensemble replay, captured requests, cancellation and associated reports" begin
    mktempdir() do directory
        f=gui_ensemble_fixture(directory)
        output=joinpath(directory,"completed.jld2");history=joinpath(directory,"history.jld2")
        ec=EnsembleExperimentController(f.record;output_path=output,run_record_path=history,record_diagnostics=true)
        observation=on(ec.running) do busy
            busy || return
            @test ec.active_request[].output==output && ec.active_request[].record_diagnostics
            ec.output_path[]=joinpath(directory,"next.jld2");ec.record_diagnostics[]=false
            @test_throws ArgumentError open_experiment!(ec,f.record)
            start!(ec;async=false)
        end
        start!(ec;async=false);off(observation)
        @test ec.state[]===:completed && !ec.running[] && ec.error[]===nothing
        @test ec.progress[]==(4,4) && ec.pool_progress[]==(1,1) && ec.active_request[]===nothing
        @test ec.last_run[].output==output && !isfile(joinpath(directory,"next.jld2"))
        first=ec.last_run[];first_id=first.run_id
        ex=experiment_results(ec;verify_inputs=true)
        index=ResultFile(output);raw=index[1]
        @test nframes(ex)==1 && isequal(current_result(ex).u,physical(raw).u)
        @test occursin("ensemble",lowercase(companion_summary(ex)))
        direct=run_piv_ensemble([(f.paths[1],f.paths[2]),(f.paths[3],f.paths[4])],f.record.recipe.passes;
            preprocess=Hammerhead._experiment_preprocess(f.record.recipe,nothing),mask=f.record.recipe.mask,
            image_type=Float32,threaded=false,scale=f.record.recipe.scale,progress=false)
        @test isequal(raw.u,direct.u) && isequal(raw.v,direct.v)
        report=experiment_quality_report(ec)
        @test quality_report_data(report)["quality_report_format_version"]==5
        @test quality_report_data(report)["provenance"]["run_id"]==first_id
        for path in (output,history,f.paths[1])
            bytes=read(path);@test_throws ArgumentError save_experiment_quality_report(path,ec);@test read(path)==bytes
        end
        ec.output_path[]=joinpath(directory,"cancelled.jld2")
        cancel_fig=ensemble_experiment_workflow(ec;size=(900,600))
        cancel_screen=GLMakie.Screen(cancel_fig.scene;visible=false,start_renderloop=false);colorbuffer(cancel_screen)
        cancel_button=ensemble_gui_button(cancel_fig,"cancel between contributions")
        start!(ec;async=false,progress=event->begin
            @test ec.progress[]==(event.completed_contributions,event.total_contributions) && ec.running[]
            box=cancel_button.layoutobservables.computedbbox[];ensemble_gui_click!(cancel_fig,box.origin.+box.widths./2)
        end)
        @test ec.state[]===:cancelled && ec.last_run[].status===:cancelled && ec.progress[]==(1,4)
        @test !isfile(ec.last_run[].output) && ec.pool_progress[]==(0,0) && !ec.running[]
        @test ec.selected_run_id[]==first_id && length(ec.record[].runs)==2
        ensemble_gui_capture(cancel_screen,"-900-cancelled")
        GLMakie.destroy!(cancel_screen);GLMakie.Makie.current_figure!(nothing)
        cancelled_id=ec.last_run[].run_id
        select_experiment_run!(ec,cancelled_id);@test_throws ArgumentError experiment_results(ec)
        @test_throws ArgumentError experiment_quality_report(ec);select_experiment_run!(ec,first_id)
        ec.output_path[]=joinpath(directory,"last-contribution-cancel.jld2")
        start!(ec;async=false,progress=event->event.completed_contributions==event.total_contributions && cancel!(ec))
        @test ec.state[]===:cancelled && ec.progress[]==(4,4) && ec.pool_progress[]==(0,0)
        @test !isfile(ec.last_run[].output)
        staged=EnsembleExperimentController(f.record;output_path=joinpath(directory,"pre-cancel.jld2"),run_record_path=joinpath(directory,"pre-cancel-history.jld2"))
        start!(staged);task=staged._task[];cancel!(staged);wait(task)
        @test staged.last_run[].status===:cancelled && staged.progress[]==(0,4) && !isfile(staged.output_path[])
        @test length(load_ensemble_experiment(staged.run_record_path[]).runs)==1 && staged.active_request[]===nothing
        broken=EnsembleExperimentController(f.record;output_path=joinpath(directory,"never.jld2"))
        primary=ErrorException("startup-original-marker");subscription=on(_->throw(primary),broken.running)
        start!(broken;async=false);off(subscription)
        @test broken.error[]===primary && !broken.running[] && broken.active_request[]===nothing
        @test broken._task[]===nothing && broken._cancel_token[]===nothing && !isfile(broken.output_path[])
        failed=EnsembleExperimentController(f.record;output_path=joinpath(directory,"failed.jld2"),run_record_path=joinpath(directory,"failed-history.jld2"))
        original=ErrorException("progress-original-marker")
        terminal=on(s->s===:failed && error("terminal-secondary-marker"),failed.state)
        cleanup=on(b->b || error("cleanup-secondary-marker"),failed.running)
        start!(failed;async=false,progress=event->throw(original));off(terminal);off(cleanup)
        @test failed.error[]===original && failed.last_run[].status===:failed && !failed.running[]
        @test failed.last_run[].completed_contributions==1 && !isfile(failed.output_path[])
        # A successful pool survives separate history publication failure.
        partial=EnsembleExperimentController(f.record;output_path=joinpath(directory,"history-failed-pool.jld2"),run_record_path=joinpath(directory,"history-becomes-directory"))
        start!(partial;async=false,progress=event->event.completed_contributions==event.total_contributions && mkdir(partial.run_record_path[]))
        @test partial.state[]===:history_save_failed && partial.last_run[].status===:completed
        @test partial.error[] isa EnsembleRunRecordError && partial.pool_progress[]==(1,1) && !partial.running[]
        @test current_result(experiment_results(partial)) isa PIVResult
        @test quality_report_data(experiment_quality_report(partial))["provenance"]["run_id"]==partial.last_run[].run_id
        refreshed=EnsembleExperimentController(f.record)
        token=Ref(false);refreshed.running[]=true;refreshed._cancel_token[]=token
        refresh_error=ErrorException("post-return-history-refresh-marker")
        EGEC._replay_gui_ensemble!(refreshed,deepcopy(f.record),joinpath(directory,"refresh-pool.jld2"),joinpath(directory,"refresh-history.jld2"),false,false,token,nothing;
            history_loader=path->throw(refresh_error))
        @test refreshed.state[]===:history_refresh_failed && refreshed.error[]===refresh_error && !refreshed.running[]
        @test refreshed.last_run[].status===:completed && refreshed.pool_progress[]==(1,1)
        @test last(refreshed.record[].runs).run_id==refreshed.last_run[].run_id
        # Dialog mutation affects next choices, not the captured report/identity.
        destination=Ref(joinpath(directory,"selected-quality.toml"));toggles=Ref{Any}(nothing);calls=Ref(0)
        picker=()->begin
            calls[]+=1;select_experiment_run!(ec,cancelled_id);toggles[][3].active[]=false;toggles[][4].active[]=false
            destination[]
        end
        report_fig=ensemble_experiment_workflow(ec;report_path_picker=picker)
        toggles[]=filter(b->b isa Toggle,report_fig.content);toggles[][3].active[]=true
        button=ensemble_gui_button(report_fig,"save quality report");button.clicks[]+=1
        @test calls[]==1 && ec.status[]=="quality report saved"
        saved_report=quality_report_data(load_quality_report(destination[]))
        @test saved_report["provenance"]["run_id"]==first_id
        @test saved_report["provenance"]["input_bytes_checked"]===true
        @test haskey(saved_report,"ensemble_execution_diagnostics")
        @test occursin("Reported run: "*first_id,ensemble_gui_pages(report_fig))
        destination[]=output;bytes=read(output);button.clicks[]+=1
        @test occursin("failed",ec.status[]) && read(output)==bytes
        @test occursin("Reported run: "*first_id,ensemble_gui_pages(report_fig))
        screen=GLMakie.Screen(report_fig.scene;visible=false,start_renderloop=false)
        for dims in ((900,600),(1100,800))
            resize!(report_fig.scene,dims...);ensemble_gui_choose!(ensemble_gui_controls(report_fig),:reports)
            ensemble_gui_pages(report_fig)
            for _ in 1:20;ensemble_gui_button(report_fig,"previous").clicks[]+=1;end
            colorbuffer(screen);@test ensemble_gui_within(ensemble_gui_body(report_fig).layoutobservables.computedbbox[],dims...)
            ensemble_gui_capture(screen,"-$(dims[1])-retained-report")
        end
        GLMakie.destroy!(screen);GLMakie.Makie.current_figure!(nothing)
    end
end
end
