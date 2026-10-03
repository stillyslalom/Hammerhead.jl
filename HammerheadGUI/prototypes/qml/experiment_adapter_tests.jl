using Test,Hammerhead,HammerheadGUI,Observables,GLMakie
include("adapter.jl")
include("viewport.jl")
include("viewport_lifecycle.jl")
include("experiment_fixture.jl")

function wait_saved(state)
    deadline=time()+120
    while Prototype.busy(state) && time()<deadline
        Prototype.service_saved_replay!(state)
        sleep(.005)
    end
    @test !Prototype.busy(state)
end
function dispose_view(registry)
    lease=registry.active
    figure=lease.payload[1]
    ViewportLifecycle.request_release!(registry)
    for screen in copy(figure.scene.current_screens)
        GLMakie.destroy!(screen)
    end
    ViewportLifecycle.acknowledge_release!(registry,lease.generation)
    GLMakie.Makie.current_figure()===figure && GLMakie.Makie.current_figure!(nothing)
    ViewportLifecycle.detach_viewport!(registry,lease.generation)
    lease
end
function reopen_saved_views(state)
    registry=ViewportLifecycle.Registry()
    references=WeakRef[]
    leases=Any[]
    for _ in 1:2
        lease=ViewportLifecycle.create_viewport!(registry,()->begin
            fig,ax,refresh,subscriptions,detach=viewport(state;managed=true)
            ((fig,ax),refresh,subscriptions,detach)
        end)
        fig,ax=lease.payload
        push!(references,WeakRef(fig))
        @test size(colorbuffer(fig;px_per_unit=1,visible=false))==(650,900)
        @test ax.xlabel[]=="x (mm)" && ax.ylabel[]=="y (mm)"
        @test ax.yreversed[]
        push!(leases,dispose_view(registry))
        @test state.refresh()===nothing
    end
    leases,references
end

@testset "Qt saved-experiment adapter" begin
    mktempdir() do dir
        fixture=experiment_fixture(dir)
        state=Prototype.State()
        ec=state.experiment.controller
        @test Prototype.open_saved_experiment(state,fixture.path)
        @test recipe_identity(ec.record[].recipe)==recipe_identity(fixture.record.recipe)
        @test ec.record[].recipe.mask==fixture.record.recipe.mask
        @test ec.record[].recipe.roi==fixture.record.recipe.roi
        @test [s.operation for s in ec.record[].recipe.preprocessing]==[:highpass_filter,:intensity_cap]
        @test Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        # No unsupported callback is silently omitted from a subprocess request.
        ec.custom_preprocess[]=identity
        @test !Prototype.run_saved_experiment(state)
        @test occursin("custom preprocessing",state.experiment.error[])
        @test !ec.running[] && state.experiment.job===nothing
        ec.custom_preprocess[]=nothing
        # Startup observers cannot strand busy state or spawn processing after
        # their original failure. The request already exists when notified.
        startup_error=ErrorException("startup observer failure")
        startup=on(ec.running) do running
            if running
                @test state.experiment.request.record.recipe.recipe_id==fixture.record.recipe.recipe_id
                @test state.experiment.request.output==fixture.output
                throw(startup_error)
            end
        end
        @test !Prototype.run_saved_experiment(state)
        @test ec.error[]===startup_error && !ec.running[]
        @test state.experiment.job===nothing && state.experiment.request===nothing
        off(startup)
        @test Prototype.run_saved_experiment(state;progress=(i,n)->i==1 && Prototype.cancel_saved_experiment(state))
        @test !Prototype.configure_saved_experiment(state,"next-output","",true)
        @test !Prototype.run_batch(state)
        wait_saved(state)
        @test ec.state[]===:cancelled && ec.progress[]==(1,3)
        @test ec.last_run[].status===:failed && length(load_results(fixture.output))==1
        before=state.explorer
        @test !Prototype.inspect_saved_experiment(state)
        @test state.explorer===before && state.dataset[]===:demo
        @test Prototype.run_saved_experiment(state;progress=(i,n)->i==n && Prototype.cancel_saved_experiment(state))
        wait_saved(state)
        @test ec.state[]===:completed && ec.progress[]==(3,3)
        @test Prototype.inspect_saved_experiment(state)
        @test state.dataset[]===:experiment
        @test state.explorer.results isa HammerheadGUI.Controllers._LazyDisplayResults
        @test nframes(state.explorer)==3 && state.explorer.results.index==1
        state.drawing=true
        Prototype.pick(state,first(current_result(state.explorer).x),first(current_result(state.explorer).y);drawing=true)
        @test !state.drawing && state.explorer.selection[]===CartesianIndex(1,1)
        @test occursin(ec.last_run[].run_id,state.displayed[])
        raw=load_results(fixture.output;lazy=true)[1]
        displayed=current_result(state.explorer)
        @test displayed.x≈raw.x*.02 && displayed.y≈raw.y*.02
        @test isequal(displayed.u,raw.u*20) && isequal(displayed.v,raw.v*20)
        @test physical(displayed)===displayed
        @test Prototype.navigate(state,3) && state.frame[]==3
        displayed=current_result(state.explorer)
        Prototype.pick(state,displayed.x[1],displayed.y[1])
        @test selection_point(displayed,state.explorer.selection[])==(displayed.x[1],displayed.y[1])
        mask_count=length(state.mask.polygons[])
        Prototype.pick(state,20.,20.;drawing=true)
        Prototype.close_mask(state)
        @test length(state.mask.polygons[])==mask_count
        old=(state.explorer,state.explorer.selection[],state.frame[],state.displayed[])
        @test !Prototype.open_saved_experiment(state,joinpath(dir,"missing.jld2"))
        @test (state.explorer,state.explorer.selection[],state.frame[],state.displayed[])==old
        previous_record=ec.record[]
        @test occursin("window_size",state.experiment.recipe_text)
        @test occursin("highpass_filter",state.experiment.recipe_text)
        Prototype.experiment_page(state,10000)
        @test !isempty(state.experiment.text[]) && occursin("Page",state.experiment.pages[])
        Prototype.experiment_section(state,true)
        @test occursin(ec.last_run[].run_id,state.experiment.history_text)

        # Changed output cannot silently replace the retained bound display.
        open(fixture.output,"a") do io
            write(io,UInt8[1,2,3])
        end
        @test !Prototype.inspect_saved_experiment(state)
        @test (state.explorer,state.explorer.selection[],state.frame[],state.displayed[])==old
        # Create an intact latest run for viewport ownership/units assertions.
        @test Prototype.run_saved_experiment(state)
        wait_saved(state)
        @test Prototype.inspect_saved_experiment(state)
        leases,references=reopen_saved_views(state)
        GC.gc(true);GC.gc(true)
        @test all(ref->ref.value===nothing,references)
        @test all(lease->lease.payload===nothing && isempty(lease.subscriptions),leases)

        # Metadata can inspect a script reference; shell never executes it.
        script=joinpath(dir,"unsafe.jl");write(script,"error(\"must never execute\")\n")
        referenced=ExperimentRecord([(fixture.record.input_files[p[1]]["path"],fixture.record.input_files[p[2]]["path"]) for p in fixture.record.pairs],
            PIVRecipe(fixture.record.recipe.passes;external_preprocess=ScriptReference(script;entrypoint="unsafe(image)")))
        saved=save_experiment(joinpath(dir,"script-record.jld2"),referenced)
        @test Prototype.open_saved_experiment(state,saved)
        @test !Prototype.run_saved_experiment(state)
        @test occursin("does not load or execute",state.experiment.error[])
        @test state.dataset[]===:experiment # labelled old display remains
        @test occursin(previous_record.recipe.recipe_id,state.displayed[])

        # View closure does not own/cancel replay; shutdown does, then waits.
        @test Prototype.open_saved_experiment(state,fixture.path)
        @test Prototype.configure_saved_experiment(state,joinpath(dir,"shutdown.jld2"),joinpath(dir,"shutdown-record.jld2"),false)
        @test Prototype.run_saved_experiment(state)
        Prototype.request_shutdown(state)
        @test_throws ArgumentError Prototype.dispose_state(state)
        @test !Prototype.run_saved_experiment(state)
        wait_saved(state)
        @test ec.state[]===:cancelled && ec.progress[]==(0,3)
        Prototype.dispose_state(state)
        @test isempty(state.subscriptions) && isempty(state.experiment.subscriptions)
        @test state.explorer===nothing
    end
end

@testset "Owner observer abort and deferred shutdown retain joined prefix" begin
    mktempdir() do dir
        fixture=experiment_fixture(dir);state=Prototype.State();ec=state.experiment.controller
        @test Prototype.open_saved_experiment(state,fixture.path)
        @test Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        original=ErrorException("owner progress observer failed")
        @test Prototype.run_saved_experiment(state;progress=(i,n)->throw(original))
        wait_saved(state)
        @test ec.error[]===original && ec.state[]===:failed && ec.progress[]==(1,3)
        @test state.experiment.outcome.cleanup_confirmed && state.experiment.outcome.exit_code==1
        @test length(load_results(fixture.output))==1 && ec.last_run[].status===:failed
        @test Prototype.run_saved_experiment(state;progress=(i,n)->:defer)
        deadline=time()+120
        while state.experiment.pending_progress===nothing && ec.running[] && time()<deadline
            Prototype.service_saved_replay!(state);sleep(.005)
        end
        @test state.experiment.pending_progress!==nothing && ec.running[]
        Prototype.request_shutdown(state)
        @test state.experiment.pending_progress===nothing
        wait_saved(state)
        @test ec.state[]===:cancelled && ec.progress[]==(1,3)
        @test state.experiment.outcome.cleanup_confirmed && state.experiment.job===nothing
        Prototype.dispose_state(state)
    end
end
