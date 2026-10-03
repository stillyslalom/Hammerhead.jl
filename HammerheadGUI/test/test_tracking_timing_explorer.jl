using Test, HammerheadGUI
using HammerheadGUI.Hammerhead, HammerheadGUI.GLMakie
using HammerheadGUI.Controllers: field_label, selection_point, current_field_values

function timed_explorer_fixture(;scale=nothing,unit="s",epoch=big(10)^40,clock="camera-clock-"*repeat("opaque",10))
    source=FrameSource(6,i->zeros(2,2);timestamps=epoch.+[0,1,2,4,6,8],time_unit=unit,clock_id=clock,source_id="camera")
    refs=[FrameRef(source,i) for i in (1,3,6)]
    snapshot=Hammerhead._tracking_preflight(refs,:source,nothing,nothing,scale,nothing)
    trajectories=[Trajectory(1,[10.0,40.0],[20.0,55.0],[1,3]),
        Trajectory(2,[25.0],[35.0],[2]),
        Trajectory(1,[60.0,NaN,75.0],[20.0,30.0,45.0],[1,2,3]),
        Trajectory(1,Float64[],Float64[],Int[])]
    Hammerhead._tracking_bind(TrackingResult(trajectories,3,PTVParameters(),scale),snapshot.data)
end

@noinline function timed_explorer_release_fixture()
    raw=timed_explorer_fixture(scale=PhysicalScale(pixel_size=.25,dt=300,length_unit="mm",time_unit="s"))
    refs=(WeakRef(raw.result),WeakRef(raw.result.trajectories[1].x),WeakRef(raw.timing._data))
    ResultExplorer(raw),refs
end

@testset "Dedicated actual-time trajectory explorer" begin
    C=HammerheadGUI.Controllers
    timed=timed_explorer_fixture()
    ex=ResultExplorer(timed)
    @test current_result(ex)===timed && nframes(ex)==1
    @test ex.results isa C._TimedDisplayResult
    @test eltype(ex.results)==TimedTrackingResult
    @test available_fields(current_result(ex))==[:speed]
    @test field_values(current_result(ex),:speed)[1]≈hypot(30,35)/8
    @test all(isnan,field_values(current_result(ex),:speed)[2:4])
    @test occursin("observation-mean secant speed (px/s)",field_label(current_result(ex),:speed))
    @test_throws ArgumentError field_values(timed,:u)
    @test_throws ArgumentError set_tool!(ex,:profile)
    @test_throws ArgumentError set_companion_inspection!(ex)
    @test !ex.companion_enabled[]
    select_nearest!(ex,40,55)
    @test ex.selection[]==1 && selection_point(current_result(ex),1)==(10,20)
    selection=describe_selection(ex)
    @test occursin("whole track",selection) && occursin("selected frames 1–3",selection)
    @test occursin("gaps in selected frames: 1",selection)
    @test occursin(string(big(10)^40),selection) && occursin("elapsed: 8 s",selection)
    @test occursin("not instantaneous",selection)
    ex.selection[]=2
    @test occursin("unavailable (singleton)",describe_selection(ex))
    ex.selection[]=3
    @test occursin("nonfinite position",describe_selection(ex))
    @test selection_point(timed,3)==(60,20)
    ex.selection[]=4
    @test occursin("actual time: unavailable",describe_selection(ex))
    @test !occursin("nothing",describe_selection(ex))
    @test selection_point(timed,4)===nothing
    ex.selection[]=1
    prior=current_result(ex)
    @test_throws BoundsError (ex.frame[]=2)
    @test ex.frame[]==1 && ex.selection[]==1 && current_result(ex)===prior
    @test set_frame!(ex,99)==1
    @test_throws ArgumentError push_result!(ex,timed.result)
    @test_throws ArgumentError push_result!(ex,timed)
    @test_throws ArgumentError ResultExplorer([timed])
    @test_throws ArgumentError ResultExplorer([timed,timed.result])

    unknown=ResultExplorer(timed_explorer_fixture(unit=nothing))
    @test occursin("unknown sample-time unit",field_label(current_result(unknown),:speed))
    unknown.selection[]=1
    @test occursin("Sample-time unit unknown",describe_selection(unknown))
    assumed=ResultExplorer(timed_explorer_fixture(unit=nothing,scale=PhysicalScale(pixel_size=.25,dt=300,length_unit="mm",time_unit="s")))
    @test current_field_values(assumed)[1]≈current_field_values(ex)[1]/4
    assumed.selection[]=1
    @test occursin("Time unit assumed from scale",describe_selection(assumed))
    @test current_result(assumed).result.scale.dt==300
    @test physical(current_result(assumed))===current_result(assumed)
    @test_throws ArgumentError with_scale(current_result(assumed),PhysicalScale(pixel_size=2,length_unit="mm",time_unit="s"))

    mktempdir() do dir
        path=joinpath(dir,"timed.jld2")
        save_timed_tracking(path,timed)
        bytes=read(path)
        loaded=ResultExplorer(path;format=:timed_tracking)
        @test loaded.path==path && current_result(loaded) isa TimedTrackingResult
        @test current_field_values(loaded)[1]≈current_field_values(ex)[1]
        @test_throws ArgumentError ResultExplorer(path)
        @test_throws ArgumentError ResultExplorer(path;lazy=true,format=:timed_tracking)
        @test_throws ArgumentError ResultExplorer(path;format=:guess)
        @test_throws ArgumentError ResultFile(path)
        @test_throws ArgumentError save_results(path,loaded.results)
        @test_throws ArgumentError save_timed_tracking(path,current_result(loaded))
        @test read(path)==bytes
        copy_path=joinpath(dir,"copy.jld2")
        save_timed_tracking(copy_path,current_result(loaded))
        @test tracking_speed_summary(load_timed_tracking(copy_path)).speeds[1]≈current_field_values(ex)[1]
        export_table(joinpath(dir,"timed.csv"),current_result(loaded))
        @test occursin("sample_time_numerator",read(joinpath(dir,"timed.csv"),String))
        # Preserve native eager/lazy eltypes and save dispatch.
        native=ResultExplorer(timed.result)
        native_path=joinpath(dir,"native.jld2")
        @test eltype(native.results)==C.AnyResult
        save_results(native_path,native.results)
        lazy=ResultExplorer(native_path;lazy=true)
        @test eltype(lazy.results)==C.AnyResult
        save_results(joinpath(dir,"native-copy.jld2"),lazy.results)
        @test load_results(joinpath(dir,"native-copy.jld2"))[1] isa TrackingResult
        @test_throws ArgumentError ResultExplorer(native_path;format=:timed_tracking)
        # Physical edits invalidate timing binding; navigation preserves state.
        loaded.selection[]=1
        current_result(loaded).result.trajectories[1].x[1]+=1
        @test_throws ArgumentError set_frame!(loaded,1)
        @test loaded.frame[]==1 && loaded.selection[]==1
        @test occursin("binding",loaded.status[])
        @test_throws ArgumentError current_field_values(loaded)
        @test_throws ArgumentError selection_point(loaded.results.result,1)
        @test read(path)==bytes
    end

    retained,refs=timed_explorer_release_fixture()
    GC.gc(true)
    @test all(ref->ref.value===nothing,refs)
    @test current_result(retained) isa TimedTrackingResult

    view_ex=ResultExplorer(timed_explorer_fixture())
    view_ex.selection[]=1
    fig=result_explorer(view_ex) # default size, huge exact timestamps/clock token
    img=copy(colorbuffer(fig;px_per_unit=1))
    @test size(img)==(700,1000)
    axis=only(block for block in fig.content if block isa Axis)
    @test axis.scene.viewport[].widths[1]>300 && axis.scene.viewport[].widths[2]>300
    @test occursin("3 / 4 speeds unavailable (gray)",axis.title[])
    @test axis.yreversed[]
    @test any(block->block isa Label && block.text[]=="bundle",fig.content)
    next=only(block for block in fig.content if block isa Button && block.label[]=="next")
    next.clicks[]+=1
    @test colorbuffer(fig;px_per_unit=1)!=img
    @test any(block->block isa Label && block.text[]=="2 / 4",fig.content) ||
        any(block->block isa Label && startswith(block.text[],"2 /"),fig.content)
    if haskey(ENV,"HAMMERHEAD_TIMED_EXPLORER_SCREENSHOT")
        GLMakie.save(ENV["HAMMERHEAD_TIMED_EXPLORER_SCREENSHOT"],fig)
    end
    view_ex.selection[]=2
    next.clicks[]+=1
    @test !isempty(colorbuffer(fig;px_per_unit=1))
    @test any(block->block isa Label && occursin("singleton",block.text[]),fig.content)
    if haskey(ENV,"HAMMERHEAD_TIMED_EXPLORER_SCREENSHOT")
        GLMakie.save(replace(ENV["HAMMERHEAD_TIMED_EXPLORER_SCREENSHOT"],".png"=>"-singleton.png"),fig)
    end
    # Exact digits survive wrapping/paging, with no truncated identifiers.
    token=repeat("1234567890",20)
    @test replace(join(HammerheadGUI._timed_selection_pages(token)),"\n"=>"")==token
end
