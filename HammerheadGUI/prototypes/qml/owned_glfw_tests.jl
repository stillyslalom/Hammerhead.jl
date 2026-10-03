using Test, GLMakie, Observables, Hammerhead, HammerheadGUI
include("adapter.jl")
include("viewport.jl")
include("viewport_lifecycle.jl")
include("owned_glfw.jl")
include("experiment_fixture.jl")
include("shell_cleanup.jl")

@testset "Cleanup preserves primary failure and separately reports disposal failure" begin
    disposed=Ref(false)
    @test ShellCleanup.with_cleanup(()->42,()->disposed[]=true)==42
    @test disposed[]
    primary=ErrorException("original render failure"); cleanup=ErrorException("secondary disposal failure")
    reported=Any[]
    caught=try
        ShellCleanup.with_cleanup(()->throw(primary),()->throw(cleanup);report=e->push!(reported,e))
    catch e
        e
    end
    @test caught===primary
    @test only(reported).ex===cleanup
    @test_throws ErrorException ShellCleanup.with_cleanup(()->42,()->throw(cleanup))
    @test_throws InterruptException ShellCleanup.with_cleanup(()->throw(InterruptException()),()->disposed[]=true)
end

function open_owned(registry, owner, state)
    lease=ViewportLifecycle.create_viewport!(registry,()->begin
        fig,ax,refresh,subscriptions,detach=viewport(state;managed=true)
        (fig,ax),refresh,subscriptions,detach
    end)
    try
        OwnedGLFW.open!(owner,lease.payload[1])
    catch
        ViewportLifecycle.request_release!(registry)
        ViewportLifecycle.acknowledge_release!(registry,lease.generation)
        ViewportLifecycle.detach_viewport!(registry,lease.generation)
        rethrow()
    end
    lease
end
function close_owned(registry, owner)
    lease=registry.active
    lease===nothing && return
    fig=lease.payload[1]
    ViewportLifecycle.request_release!(registry)
    OwnedGLFW.release!(owner)
    ViewportLifecycle.acknowledge_release!(registry,lease.generation)
    GLMakie.Makie.current_figure()===fig && GLMakie.Makie.current_figure!(nothing)
    ViewportLifecycle.detach_viewport!(registry,lease.generation)
end
function pixel_position(ax,x,y)
    rect=ax.scene.viewport[]; limits=ax.finallimits[]
    fx=(x-limits.origin[1])/limits.widths[1]
    fy=(y-limits.origin[2])/limits.widths[2]
    (Float64(rect.origin[1]+fx*rect.widths[1]),
     Float64(rect.origin[2]+(ax.yreversed[] ? 1-fy : fy)*rect.widths[2]))
end
function synthetic_click(fig,ax,x,y)
    ev=events(fig)
    ev.entered_window[]=true
    ev.mouseposition[]=pixel_position(ax,x,y)
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end

function independent_glfw_case()
    # An existing unrelated screen must keep its scene, rendering and callbacks.
    unrelated=Figure(size=(240,180)); Axis(unrelated[1,1]); lines!(0:3,[0,1,0,2])
    sentinel=GLMakie.Screen(;visible=false,start_renderloop=false,px_per_unit=1)
    GLMakie.display_scene!(sentinel,unrelated.scene)
    sentinel_before=copy(colorbuffer(sentinel))
    registry=ViewportLifecycle.Registry(); state=Prototype.State()
    owner=OwnedGLFW.WindowOwner()
    references=WeakRef[]
    try
        lease=open_owned(registry,owner,state)
        fig,ax=lease.payload
        push!(references,WeakRef(fig))
        @test owner.screen !== sentinel
        @test sentinel.scene===unrelated.scene && only(unrelated.scene.current_screens)===sentinel
        @test !owner.visible && !GLMakie.renderloop_running(owner.screen)
        @test OwnedGLFW.pump!(owner)===:rendered
        before=OwnedGLFW.capture(owner)
        @test size(before)==(650,900)
        @test ax.yreversed[] && length(current_result(state.explorer).u)==16384
        @test_throws ErrorException OwnedGLFW.open!(owner,fig)
        @test colorbuffer(sentinel)==sentinel_before

        ev=events(fig); ev.entered_window[]=true
        ev.mouseposition[]=pixel_position(ax,48,48)
        original=ax.finallimits[]
        ev.scroll[]=(0.,1.)
        @test ax.finallimits[].widths[1]<original.widths[1]
        @test ax.finallimits[].widths[2]<original.widths[2]
        zoomed=ax.finallimits[]
        ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.right,Mouse.press)
        px,py=ev.mouseposition[]
        ev.mouseposition[]=(px+10,py+8)
        ev.mouseposition[]=(px+30,py+24)
        ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.right,Mouse.release)
        @test ax.finallimits[].origin != zoomed.origin
        @test ax.finallimits[].widths≈zoomed.widths
        @test ax.yreversed[]
        @test OwnedGLFW.capture(owner)!=before

        synthetic_click(fig,ax,32,32)
        @test state.explorer.selection[] !== nothing
        @test occursin("u",state.selection[])
        limits!(ax,0,96,0,96)
        state.drawing=true
        for point in ((10,10),(30,10),(30,30))
            synthetic_click(fig,ax,point...)
        end
        Prototype.close_mask(state)
        @test any(state.batch.mask[])
        @test length(state.mask.polygons[])==1
        @test OwnedGLFW.pump!(owner)===:rendered
        @test !GLMakie.renderloop_running(owner.screen)

        # Exercise the close flag and recreate once, with new scene ownership.
        GLMakie.GLFW.SetWindowShouldClose(owner.screen.glscreen,true)
        @test OwnedGLFW.pump!(owner)===:closed
        old_figure=objectid(fig)
        close_owned(registry,owner)
        @test isempty(fig.scene.current_screens) && isempty(lease.subscriptions)
        @test owner.releases==1 && length(GLMakie.ALL_SCREENS)==owner.baseline_screens
        lease=open_owned(registry,owner,state)
        @test objectid(lease.payload[1]) != old_figure
        push!(references,WeakRef(lease.payload[1]))
        @test OwnedGLFW.pump!(owner)===:rendered
        close_owned(registry,owner)
        # The caller checks lifetime after this exercise's local stack has gone.
        fig=nothing;ax=nothing;ev=nothing;lease=nothing
        @test colorbuffer(sentinel)==sentinel_before
        @test owner.generations==owner.releases==2
        @test owner.frames>=3 && owner.max_gap_seconds>=0
        previous=owner.owner_thread;owner.owner_thread=0
        @test_throws ErrorException OwnedGLFW.pump!(owner)
        @test_throws ErrorException OwnedGLFW.capture(owner)
        @test_throws ErrorException OwnedGLFW.release!(owner)
        owner.owner_thread=previous
    finally
        close_owned(registry,owner)
        Prototype.dispose_state(state)
        GLMakie.destroy!(sentinel)
        GLMakie.Makie.current_figure!(nothing)
    end
    references
end
@testset "Independent hidden GLFW ownership and Makie event routing" begin
    references=independent_glfw_case()
    @test timedwait(()->begin GC.gc();all(ref->ref.value===nothing,references) end,5)==:ok
end

@testset "Saved replay outlives visualization and preserves physical selection" begin
    mktempdir() do directory
        fixture=experiment_fixture(directory)
        state=Prototype.State();registry=ViewportLifecycle.Registry();owner=OwnedGLFW.WindowOwner()
        try
            open_owned(registry,owner,state)
            @test Prototype.open_saved_experiment(state,fixture.path)
            @test Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
            @test Prototype.run_saved_experiment(state;progress=(i,n)->begin
                if i==1
                    close_owned(registry,owner)
                    Prototype.cancel_saved_experiment(state)
                end
            end)
            deadline=time()+90
            while Prototype.busy(state) && time()<deadline
                OwnedGLFW.pump!(owner)
                sleep(.005)
            end
            @test !Prototype.busy(state)
            @test registry.active===nothing && owner.screen===nothing
            @test state.experiment.controller.progress[]==(1,3)
            @test state.experiment.controller.last_run[].status===:failed
            lease=open_owned(registry,owner,state)
            @test Prototype.run_saved_experiment(state)
            deadline=time()+90
            while Prototype.busy(state) && time()<deadline
                OwnedGLFW.pump!(owner)
                sleep(.005)
            end
            @test !Prototype.busy(state)
            @test state.experiment.controller.progress[]==(3,3)
            @test Prototype.inspect_saved_experiment(state)
            @test Prototype.navigate(state,3)
            fig,ax=lease.payload
            @test ax.xlabel[]=="x (mm)" && ax.ylabel[]=="y (mm)"
            @test occursin("mm/s",ax.title[])
            result=current_result(state.explorer)
            synthetic_click(fig,ax,result.x[2],result.y[3])
            @test selection_point(result,state.explorer.selection[])==(result.x[2],result.y[3])
            image=OwnedGLFW.capture(owner);identity=state.displayed[]
            @test !Prototype.open_saved_experiment(state,joinpath(directory,"absent.jld2"))
            @test state.displayed[]==identity
            @test OwnedGLFW.capture(owner)==image
            @test !GLMakie.renderloop_running(owner.screen)
        finally
            Prototype.request_shutdown(state)
            deadline=time()+90
            while Prototype.busy(state) && time()<deadline
                sleep(.005)
            end
            close_owned(registry,owner)
            Prototype.dispose_state(state)
        end
        @test isempty(state.subscriptions) && isempty(state.experiment.subscriptions)
        @test owner.generations==owner.releases && length(GLMakie.ALL_SCREENS)==owner.baseline_screens
    end
end

@testset "Automatic limits contain normalized glyphs without resetting manual view" begin
    scale=PhysicalScale(pixel_size=1e-6,dt=.001,length_unit="m",time_unit="s")
    for (xs,ys,scaling) in (([4.,20.,36.],[7.,39.,71.],nothing),
                            ([4.],[7.,39.],scale),([4.,20.],[7.],scale),([4.],[7.],scale))
        dimensions=(length(ys),length(xs))
        raw=PIVResult(xs,ys,fill(4.,dimensions),fill(-3.,dimensions),ones(dimensions),ones(dimensions),
            fill(NaN,dimensions),fill(NaN,dimensions),falses(dimensions),falses(dimensions),PIVParameters(window_size=16,overlap=(8,8)))
        state=Prototype.State();registry=ViewportLifecycle.Registry();owner=OwnedGLFW.WindowOwner()
        state.explorer=ResultExplorer(scaling===nothing ? raw : with_scale(raw,scaling))
        state.dataset[]=scaling===nothing ? :native : :experiment
        try
            lease=open_owned(registry,owner,state)
            _,ax=lease.payload
            @test OwnedGLFW.pump!(owner)===:rendered
            @test size(OwnedGLFW.capture(owner))==(650,900)
            points=ax.scene.plots[2][1][]
            low,high=extrema(ax.finallimits[])
            @test !isempty(points)
            @test all(p->low[1]<p[1]<high[1] && low[2]<p[2]<high[2],points)
            displayed=current_result(state.explorer)
            @test any(p->p[1]>maximum(displayed.x),points)
            @test !ax.scene.plots[1].visible[]
            limits!(ax,low[1],(low[1]+high[1])/2,low[2],(low[2]+high[2])/2)
            manual=ax.finallimits[]
            state.refresh()
            @test ax.finallimits[]==manual
        finally
            close_owned(registry,owner)
            Prototype.dispose_state(state)
        end
    end
end
