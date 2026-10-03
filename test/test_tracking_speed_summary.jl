using Test, Hammerhead

function speed_summary_fixture(trajectories,times;scale=nothing,unit=nothing)
    snapshot=Hammerhead._tracking_preflight([zeros(2,2) for _ in times],times,unit,nothing,scale,nothing)
    Hammerhead._tracking_bind(TrackingResult(trajectories,length(times),PTVParameters(),scale),snapshot.data)
end

@testset "Actual-time scalar speed summary" begin
    track=Trajectory(1,[0.0,1.0,9.0],zeros(3),[1,2,3])
    timed=speed_summary_fixture([track],[0,1,3];unit="s")
    before=tracking_timing_data(timed)
    summary=tracking_speed_summary(timed)
    @test summary.speeds≈[8/3]
    @test summary.available==[true] && summary.reasons==[:none]
    @test summary.tracks[1]==(observations=3,first_frame=1,last_frame=3,gaps=0,first_time="0",last_time="3",elapsed="3")
    @test summary.mean_convention=="observation_mean_secant_magnitude"
    @test summary.velocity_convention=="endpoint_one_sided_interior_outer_secant"
    @test summary.time_unit=="s" && summary.time_unit_provenance=="provided_sample_unit"
    @test summary.length_unit=="px" && summary.clock_id===nothing
    @test before==tracking_timing_data(timed)
    summary.speeds[1]=-99
    empty!(summary.tracks)
    @test tracking_speed_summary(timed).speeds≈[8/3]
    @test timed.result.trajectories[1].x==[0,1,9]

    scale=PhysicalScale(pixel_size=2,dt=100,length_unit="mm",time_unit="s")
    scaled=with_scale(timed,scale)
    display=physical(scaled)
    @test tracking_speed_summary(scaled).speeds≈[16/3]
    @test tracking_speed_summary(display).speeds≈tracking_speed_summary(scaled).speeds
    @test physical(display)===display
    @test tracking_speed_summary(display).length_unit=="mm"
    @test display.result.scale.dt==100
    unknown=speed_summary_fixture([deepcopy(track)],[0,1,3])
    @test tracking_speed_summary(unknown).time_unit===nothing
    assumed=with_scale(unknown,scale)
    @test tracking_speed_summary(assumed).time_unit_provenance=="legacy_scale_same_unit"
    @test_throws ArgumentError with_scale(timed,PhysicalScale(time_unit="ms"))

    epoch=big(10)^100
    gapped=Trajectory(1,[0.0,9.0],[0.0,0.0],[1,3])
    gap=speed_summary_fixture([gapped],epoch.+[0,1,3];unit="ns")
    gs=tracking_speed_summary(gap)
    @test gs.speeds==[3.0] && gs.tracks[1].gaps==1
    @test gs.tracks[1].first_time==string(epoch) && gs.tracks[1].elapsed=="3"
    rational=speed_summary_fixture([deepcopy(track)],[0//1,1//2,3//2])
    @test tracking_speed_summary(rational).tracks[1].last_time=="3/2"

    bad=Trajectory(1,[0.0,NaN,9.0],zeros(3),[1,2,3])
    single=Trajectory(2,[5.0],[1.0],[2])
    emptytrack=Trajectory(1,Float64[],Float64[],Int[])
    mixed=speed_summary_fixture([deepcopy(track),bad,single,emptytrack],[0,1,3])
    ms=tracking_speed_summary(mixed)
    @test ms.available==[true,false,false,false]
    @test ms.reasons==[:none,:nonfinite_position,:singleton,:empty]
    @test all(isnan,ms.speeds[2:4])
    @test ms.tracks[4].first_time===nothing
    @test isempty(tracking_speed_summary(speed_summary_fixture(Trajectory{Float64}[],[0,1])).speeds)

    huge=Trajectory(1,[0.0,floatmax(Float64)/4,floatmax(Float64)/2],zeros(3),[1,2,3])
    @test tracking_speed_summary(speed_summary_fixture([huge],[0,1,2])).speeds==[floatmax(Float64)/4]
    overflow=Trajectory(1,[-floatmax(Float64),floatmax(Float64)],zeros(2),[1,2])
    @test tracking_speed_summary(speed_summary_fixture([overflow],[0,1])).reasons==[:arithmetic_range]
    normoverflow=Trajectory(1,[0.0,floatmax(Float64)],[0.0,floatmax(Float64)],[1,2])
    @test tracking_speed_summary(speed_summary_fixture([normoverflow],[0,1])).reasons==[:nonfinite_magnitude]
    tiny=nextfloat(0.0)
    underflow=Trajectory(1,[0.0,tiny,0,0,0,0,0],zeros(7),collect(1:7))
    @test tracking_speed_summary(speed_summary_fixture([underflow],collect(0:6)./2)).reasons==[:arithmetic_range]
    zero=Trajectory(1,zeros(7),zeros(7),collect(1:7))
    @test tracking_speed_summary(speed_summary_fixture([zero],collect(0:6)./2)).speeds==[0.0]

    # Integrity failure propagates rather than masquerading as an unavailable speed.
    display.result.trajectories[1].x[2]+=1
    @test_throws ArgumentError tracking_speed_summary(display)
    timed.timing._data["clock_id"]="edited"
    @test_throws ArgumentError tracking_speed_summary(timed)
end
