using Test,Hammerhead,HammerheadGUI,Observables,GLMakie
include("adapter.jl")
include("viewport.jl")
include("experiment_fixture.jl")
@testset "Saved inspection render transaction" begin
    mktempdir() do dir
        fixture=experiment_fixture(dir)
        state=Prototype.State()
        @test Prototype.open_saved_experiment(state,fixture.path)
        @test Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        @test Prototype.run_saved_experiment(state)
        deadline=time()+120
        while Prototype.busy(state) && time()<deadline
            Prototype.service_saved_replay!(state)
            sleep(.005)
        end
        @test !Prototype.busy(state)
        @test Prototype.inspect_saved_experiment(state)
        @test Prototype.navigate(state,3)
        fig,ax,refresh,subscriptions,detach=viewport(state;managed=true)
        old=current_result(state.explorer)
        before=copy(colorbuffer(fig;px_per_unit=1,visible=false))
        identity=state.displayed[]
        fault=ErrorException("injected after plot publication")
        state.refresh=()->begin
            refresh()
            current_result(state.explorer)===old || throw(fault)
        end
        @test !Prototype.inspect_saved_experiment(state)
        @test current_result(state.explorer)===old && state.frame[]==3
        @test state.displayed[]==identity && state.render_available[]
        @test colorbuffer(fig;px_per_unit=1,visible=false)==before
        @test occursin("injected",state.experiment.error[])

        state.refresh=()->begin
            refresh()
            throw(fault)
        end
        @test !Prototype.inspect_saved_experiment(state)
        @test !state.render_available[] && occursin("Plot unavailable",state.displayed[])
        @test ax.title[]=="Plot unavailable after rendering failure"
        @test current_result(state.explorer)===old
        @test colorbuffer(fig;px_per_unit=1,visible=false)!=before
        state.refresh=refresh
        @test Prototype.inspect_saved_experiment(state)
        @test state.render_available[] && state.frame[]==1
        @test state.displayed[]==identity
        # A failed native open must restore the saved-run plot as well.
        native=joinpath(dir,"native.jld2")
        save_results(native,[Prototype.dense_result(8)])
        restored=copy(colorbuffer(fig;px_per_unit=1,visible=false))
        saved_result=current_result(state.explorer)
        state.refresh=()->begin
            refresh()
            state.dataset[]===:native && throw(fault)
        end
        @test !Prototype.open_results(state,native)
        @test state.dataset[]===:experiment && state.displayed[]==identity
        @test colorbuffer(fig;px_per_unit=1,visible=false)==restored
        state.refresh=()->begin
            refresh()
            state.explorer.frame[]==2 && throw(fault)
        end
        @test !Prototype.navigate(state,2)
        @test state.frame[]==1 && state.explorer.frame[]==1
        @test isequal(current_result(state.explorer).u,saved_result.u)
        @test colorbuffer(fig;px_per_unit=1,visible=false)==restored
        foreach(off,subscriptions);detach()
        state.refresh=()->nothing
        for screen in copy(fig.scene.current_screens)
            GLMakie.destroy!(screen)
        end
        GLMakie.Makie.current_figure()===fig && GLMakie.Makie.current_figure!(nothing)
        Prototype.dispose_state(state)
    end
end

@testset "Physical viewport singleton geometry" begin
    state=Prototype.State()
    raw=PIVResult([4.],[7.],fill(1.,1,1),fill(0.,1,1),ones(1,1),ones(1,1),
        fill(NaN,1,1),fill(NaN,1,1),falses(1,1),falses(1,1),PIVParameters(window_size=16,overlap=(8,8)))
    state.explorer=ResultExplorer(with_scale(raw,PhysicalScale(pixel_size=1e-6,dt=.001,length_unit="m",time_unit="s")))
    state.dataset[]=:experiment
    fig,ax,refresh,subscriptions,detach=viewport(state;managed=true)
    displayed=current_result(state.explorer)
    @test size(colorbuffer(fig;px_per_unit=1,visible=false))==(650,900)
    @test ax.xlabel[]=="x (m)" && ax.ylabel[]=="y (m)"
    @test 0<arrow_spacing(displayed.x,displayed.y)<1e-6
    @test .65arrow_spacing(displayed.x,displayed.y)<minimum(ax.finallimits[].widths)
    @test arrow_spacing([1e-6],[2e-6,3e-6])≈1e-6
    @test arrow_spacing([2e-6,3e-6],[1e-6])≈1e-6
    Prototype.pick(state,only(displayed.x),only(displayed.y))
    @test selection_point(displayed,state.explorer.selection[])==(only(displayed.x),only(displayed.y))
    if haskey(ENV,"HAMMERHEAD_QT_TINY_SCREENSHOT")
        GLMakie.save(joinpath(dirname(ENV["HAMMERHEAD_QT_TINY_SCREENSHOT"]),"singleton_units.png"),fig;visible=false,start_renderloop=false)
    end
    foreach(off,subscriptions);detach()
    for screen in copy(fig.scene.current_screens)
        GLMakie.destroy!(screen)
    end
    GLMakie.Makie.current_figure()===fig && GLMakie.Makie.current_figure!(nothing)
    Prototype.dispose_state(state)
end

@testset "Physical viewport tiny-unit geometry" begin
    state=Prototype.State()
    r=with_scale(Prototype.dense_result(8),PhysicalScale(pixel_size=1e-6,dt=.001,length_unit="m",time_unit="s"))
    state.explorer=ResultExplorer(r)
    state.dataset[]=:experiment
    fig,ax,refresh,subscriptions,detach=viewport(state;managed=true)
    @test size(colorbuffer(fig;px_per_unit=1,visible=false))==(650,900)
    displayed=current_result(state.explorer)
    @test ax.xlabel[]=="x (m)" && ax.ylabel[]=="y (m)"
    @test occursin("m/s",ax.title[])
    @test ax.finallimits[].widths[1]<2(maximum(displayed.x)-minimum(displayed.x))
    @test ax.finallimits[].widths[2]<2(maximum(displayed.y)-minimum(displayed.y))
    Prototype.pick(state,displayed.x[2],displayed.y[3])
    @test selection_point(displayed,state.explorer.selection[])==(displayed.x[2],displayed.y[3])
    if haskey(ENV,"HAMMERHEAD_QT_TINY_SCREENSHOT")
        GLMakie.save(ENV["HAMMERHEAD_QT_TINY_SCREENSHOT"],fig;visible=false,start_renderloop=false)
    end
    foreach(off,subscriptions);detach()
    for screen in copy(fig.scene.current_screens)
        GLMakie.destroy!(screen)
    end
    GLMakie.Makie.current_figure()===fig && GLMakie.Makie.current_figure!(nothing)
    Prototype.dispose_state(state)
end
