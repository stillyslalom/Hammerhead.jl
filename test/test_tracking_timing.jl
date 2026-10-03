using Test, Hammerhead, JLD2

function timing_tracking_frames(times; velocity=2.0, image_type=Float64)
    T=image_type
    images = [zeros(T,96,96) for _ in times]
    for (k,time) in enumerate(times), (x,y) in ((20.0,20.0),(30.0,45.0),(50.0,65.0),(65.0,30.0))
        Hammerhead.SyntheticData.generate_gaussian_particle!(images[k],(x+velocity*Float64(time),y),3.0,1.0)
    end
    images
end
function timing_tracking_wrap(trajectory,times;scale=nothing,unit=nothing)
    n=length(times)
    snapshot=Hammerhead._tracking_preflight([zeros(2,2) for _ in 1:n],times,unit,nothing,scale,nothing)
    Hammerhead._tracking_bind(TrackingResult([trajectory],n,PTVParameters(),scale),snapshot.data)
end
function timing_tracking_rows(path)
    chars=collect(read(path,String)); records=Vector{String}[]; record=String[]
    field=IOBuffer(); quoted=false; i=1
    while i<=length(chars)
        c=chars[i]
        if c=='"'
            if quoted && i<length(chars) && chars[i+1]=='"'
                print(field,'"');i+=1
            else
                quoted=!quoted
            end
        elseif !quoted && c==','
            push!(record,String(take!(field)))
        elseif !quoted && c=='\n'
            push!(record,String(take!(field)));push!(records,record);record=String[]
        elseif !quoted && c=='\r'
        else
            print(field,c)
        end
        i+=1
    end
    @test !quoted && isempty(record) && position(field)==0
    header=first(records)
    @test all(r->length(r)==length(header),records[2:end])
    [Dict(zip(header,row)) for row in records[2:end]]
end

@testset "Actual-time metadata and linking" begin
    times=[0,1,4,5]
    frames=timing_tracking_frames(times)
    p=PTVParameters(search_radius=0.7,uod_enable=false)
    pred=(x=[1.0,96.0],y=[1.0,96.0],u=fill(2.0,2,2),v=zeros(2,2))
    timed=track_particles(frames,p;sample_times=times,time_unit="s",predictor=pred,progress=false,min_track_length=4)
    @test timed isa TimedTrackingResult{Float64}
    @test length(timed.result.trajectories)==4
    @test all(t->t.frames==[1,2,3,4],timed.result.trajectories)
    @test all(id->all(isapprox.(trajectory_velocities(timed,id)[1],2.0;atol=1e-6)),1:4)
    legacy=track_particles(frames,p;predictor=pred,progress=false,min_track_length=4)
    @test legacy isa TrackingResult
    @test isempty(legacy.trajectories)
    uniform=timing_tracking_frames(0:3)
    old=track_particles(uniform,p;predictor=pred,progress=false,min_track_length=4)
    new=track_particles(uniform,p;predictor=pred,progress=false,min_track_length=4,sample_times=collect(0:3))
    @test [(t.x,t.y,t.frames) for t in old.trajectories]==[(t.x,t.y,t.frames) for t in new.result.trajectories]
    for variant in (times.*1000,BigInt(typemax(Int64)).+times)
        other=track_particles(frames,p;predictor=pred,progress=false,min_track_length=4,sample_times=variant)
        @test [(t.x,t.y,t.frames) for t in other.result.trajectories]==[(t.x,t.y,t.frames) for t in timed.result.trajectories]
    end
    @test tracking_timing_data(timed)["reference_interval"]["numerator"]=="1"
    copied=tracking_timing_data(timed)
    copied["sample_times"][1]["numerator"]="999"
    @test tracking_timing_data(timed)["sample_times"][1]["numerator"]=="0"
    @test occursin("TimedTrackingResult",sprint(show,timed))

    loads=Int[]
    epoch=BigInt(typemax(Int64))+10
    stamps=epoch.+[0,1,4,5,8,9]
    labels=["frame-$i" for i in 1:6]
    ids=["id-$i" for i in 1:6]
    source=FrameSource(6,i->(push!(loads,i);zeros(32,32));timestamps=stamps,labels,
        source_id="opaque",frame_ids=ids,time_unit="ns",clock_id="camera")
    refs=[FrameRef(source,i) for i in (1,3,6)]
    captured=track_particles(refs;predictor=nothing,progress=(k,n)->begin
        stamps[6]=epoch+1000;labels[6]="edited";source.frame_ids[6]="edited"
        refs[end]=FrameRef(source,2)
    end,sample_times=:source,min_track_length=2)
    data=tracking_timing_data(captured)
    @test loads==[1,3,6]
    @test [f["frame_index"] for f in data["frames"]]==[1,3,6]
    @test data["frames"][3]["label"]=="frame-6"
    @test data["frames"][3]["frame_id"]=="id-6"
    @test data["sample_times"][3]["numerator"]==string(epoch+9)
    @test data["time_unit"]=="ns" && data["clock_id"]=="camera"
    @test data["frames"][1]["source_identity"]=="opaque_provided"
    @test captured.result.n_frames==3
    @test_throws ArgumentError track_particles(refs;sample_times=:source,time_unit="s",progress=false)
    @test_throws ArgumentError track_particles(refs;sample_times=:source,clock_id="other",progress=false)
    @test_throws ArgumentError track_particles(frames;sample_times=:source,progress=false)
    @test_throws ArgumentError track_particles(frames;time_unit="s",progress=false)
    @test_throws ArgumentError track_particles(frames;sample_times=:other,progress=false)

    for invalid in (Any[0,1,true,4],[0,1,NaN,4],[0,1,Inf,4],[0,1,1,4],
                    [0,1,0,4],Any[0,1,nothing,4],BigFloat[0,1,2,3],[0,1],
                    [0//1,1//big(10)^1000,1//1,2//1])
        empty!(loads)
        s=FrameSource(4,i->(push!(loads,i);frames[i]))
        @test_throws ArgumentError track_particles([FrameRef(s,i) for i in 1:4];sample_times=invalid,progress=false)
        @test isempty(loads)
    end
    missing_source=FrameSource(4,i->(push!(loads,i);frames[i]);timestamps=Any[0,1,nothing,4])
    @test_throws ArgumentError track_particles([FrameRef(missing_source,i) for i in 1:4];sample_times=:source,progress=false)
    @test isempty(loads)
    blank=PIVResult([1.0,96.0],[1.0,96.0],fill(2.0,2,2),zeros(2,2),ones(2,2),ones(2,2),
        zeros(2,2),zeros(2,2),falses(2,2),falses(2,2),PIVParameters())
    @test_throws ArgumentError track_particles(frames;sample_times=times,predictor=with_scale(blank,PhysicalScale()),progress=false)
    @test_throws ArgumentError track_particles(frames;sample_times=times,time_unit="s",scale=PhysicalScale(time_unit="ms"),progress=false)
    @test_throws ArgumentError track_particles(frames;sample_times=times,scale=PhysicalScale(time_unit=""),progress=false)
    s1=FrameSource(2,i->zeros(32,32);timestamps=[0,1],time_unit="s",clock_id="camera")
    s2=FrameSource(2,i->zeros(32,32);timestamps=[2,3],time_unit="s",clock_id="camera")
    combined=[FrameRef(s1,1),FrameRef(s1,2),FrameRef(s2,1),FrameRef(s2,2)]
    joined=track_particles(combined;sample_times=:source,predictor=nothing,progress=false)
    @test tracking_timing_data(joined)["source_scope"]=="mixed_sources"
    unknown_source=FrameSource(2,i->zeros(32,32);timestamps=[2,3])
    mixed_unknown=[FrameRef(s1,1),FrameRef(s1,2),FrameRef(unknown_source,1),FrameRef(unknown_source,2)]
    @test_throws ArgumentError track_particles(mixed_unknown;sample_times=:source,predictor=nothing,progress=false)
    explicitly_labelled=track_particles(mixed_unknown;sample_times=:source,time_unit="s",clock_id="camera",predictor=nothing,progress=false)
    @test tracking_timing_data(explicitly_labelled)["frames"][3]["clock_id"]===nothing
    source_data=tracking_timing_data(joined)
    for mutate in (d->d["clock_id"]="other",d->d["time_unit"]="ms",
                   d->d["frames"][2]["clock_id"]="other",d->d["frames"][2]["time_unit"]="ms",
                   d->d["clock_id"]=nothing,d->d["source_scope"]="provided_vector")
        bad=deepcopy(source_data);mutate(bad)
        packet=TrackingTiming(bad,Hammerhead._history_digest(bad))
        @test_throws ArgumentError tracking_timing_data(packet)
    end
    # Explicit vectors define a separate provided coordinate. The original
    # acquisition unit/clock remain unchanged rather than being overwritten.
    separate=track_particles(combined;sample_times=[0,10,20,30],time_unit="ms",clock_id="provided",predictor=nothing,progress=false)
    separate_data=tracking_timing_data(separate)
    @test separate_data["time_unit"]=="ms" && separate_data["clock_id"]=="provided"
    @test separate_data["frames"][1]["time_unit"]=="s" && separate_data["frames"][1]["clock_id"]=="camera"
end

function timing_tracking_dense(times; drop=Dict{Int,Vector{Int}}(), fresh=false,image_type=Float64)
    centers=[(x,y) for x in 18.0:6:54 for y in 18.0:6:54]
    images=[zeros(image_type,96,96) for _ in times]
    for (k,time) in enumerate(times), (id,(x,y)) in enumerate(centers)
        id in get(drop,k,Int[]) && continue
        Hammerhead.SyntheticData.generate_gaussian_particle!(images[k],(x+Float64(time),y),3.0,1.0)
    end
    if fresh
        for k in 2:length(times)
            Hammerhead.SyntheticData.generate_gaussian_particle!(images[k],(70.0+Float64(times[k]),70.0),3.0,1.0)
        end
    end
    images,length(centers)
end

@testset "Actual-time gaps, UOD and prior fields" begin
    times=[0,1,4,5,8]
    pred=(x=[1.0,96.0],y=[1.0,96.0],u=ones(2,2),v=zeros(2,2))
    p=PTVParameters(search_radius=0.6,uod_enable=true)
    # A spatial minority has a longer observation span at transition 2. UOD
    # must compare equal px/reference-interval velocities, not differing slopes
    # divided by selected-frame counts. The initial field also predicts a fresh
    # one-point head through the first missing observation.
    mixed,particle_count=timing_tracking_dense(times;drop=Dict(2=>[8,9,10]))
    result=track_particles(mixed,p;sample_times=times,predictor=pred,max_gap=1,min_track_length=4,progress=false)
    @test length(result.result.trajectories)==particle_count
    @test count(t->t.frames==[1,3,4,5],result.result.trajectories)==3
    @test all(id->all(isapprox.(trajectory_velocities(result,id)[1],1;atol=1e-6)),1:particle_count)
    # Now established two-point heads must bridge a missed frame using elapsed
    # time/last-link duration. A second empty frame exceeds max_gap and ends them.
    gap,n=timing_tracking_dense(times;drop=Dict(3=>collect(1:particle_count)))
    bridged=track_particles(gap,p;sample_times=times,predictor=pred,max_gap=1,min_track_length=4,progress=false)
    @test length(bridged.result.trajectories)==n
    @test all(t->t.frames==[1,2,4,5],bridged.result.trajectories)
    ended=track_particles(gap,p;sample_times=times,predictor=pred,max_gap=0,min_track_length=4,progress=false)
    @test isempty(ended.result.trajectories)
    # This particle first appears at input 2. The previous accepted grid field
    # must expand its 1px reference displacement to3px, then contract the next
    # prior matches back to1px rather than reusing the previous3px displacement.
    fresh,n=timing_tracking_dense(times;fresh=true)
    newcomers=track_particles(fresh,p;sample_times=times,predictor=pred,min_track_length=4,progress=false)
    @test length(newcomers.result.trajectories)==n+1
    @test count(t->t.frames==[2,3,4,5],newcomers.result.trajectories)==1
    for variant in (times.*1000,BigInt(typemax(Int64)).+times)
        other=track_particles(mixed,p;sample_times=variant,predictor=pred,max_gap=1,min_track_length=4,progress=false)
        @test [(t.x,t.y,t.frames) for t in other.result.trajectories]==[(t.x,t.y,t.frames) for t in result.result.trajectories]
    end
    passes=PIVParameters(window_size=(32,32),overlap=(16,16),uod_enable=false)
    initial=track_particles(fresh,p;sample_times=times,predictor=:piv,piv_passes=passes,min_track_length=4,progress=false)
    @test length(initial.result.trajectories)==n+1
    @test count(t->t.frames==[2,3,4,5],initial.result.trajectories)==1
    frames32,_=timing_tracking_dense(times;fresh=true,image_type=Float32)
    single=track_particles(frames32,p;sample_times=times,time_unit="s",predictor=pred,min_track_length=4,progress=false,
        scale=PhysicalScale(pixel_size=0.3,dt=123,length_unit="mm",time_unit="s"))
    @test single isa TimedTrackingResult{Float32}
    @test length(single.result.trajectories)==n+1
    converted=physical(single)
    for id in eachindex(single.result.trajectories)
        @test trajectory_velocities(single,id)[1] ≈ trajectory_velocities(converted,id)[1] atol=5e-6
    end
end

@testset "Actual-time secants, conversion and integrity" begin
    t=Trajectory{Float64}(1,[0.0,1.0,16.0],[0.0,2.0,32.0],[1,2,3])
    timed=timing_tracking_wrap(t,[0,1,4];unit="s")
    u,v=trajectory_velocities(timed,1)
    @test u==[1,4,5] && v==[2,8,10]
    # Interior outer secant =4; derivative of x=t² at the middle time is2.
    @test u[2]!=2
    @test eltype(u)==Float64
    @test_throws BoundsError trajectory_velocities(timed,2)
    single=timing_tracking_wrap(Trajectory{Float64}(2,[1.0],[2.0],[2]),[0,1,4])
    @test_throws ArgumentError trajectory_velocities(single,1)
    gapped=timing_tracking_wrap(Trajectory{Float64}(1,[0.0,8.0,10.0],[0.0,4.0,5.0],[1,3,4]),[0,1,4,5])
    @test trajectory_velocities(gapped,1)==([2,2,2],[1,1,1])
    for dt in (0.1,99.0)
        scaled=with_scale(timed,PhysicalScale(pixel_size=0.25,dt=dt,length_unit="mm",time_unit="s"))
        @test trajectory_velocities(scaled,1)==(u.*0.25,v.*0.25)
        converted=physical(scaled)
        @test converted isa TimedTrackingResult
        @test trajectory_velocities(converted,1)==trajectory_velocities(scaled,1)
        @test converted.result.scale.dt==dt
        @test converted.result.scale.pixel_size==1
        @test physical(converted)===converted
        @test tracking_timing_data(converted)["position_basis"]=="scaled_length"
        @test_throws ArgumentError with_scale(converted,nothing)
        @test_throws ArgumentError with_scale(converted,PhysicalScale(pixel_size=2,time_unit="s"))
    end
    unknown=timing_tracking_wrap(t,[0,1,4];scale=PhysicalScale(time_unit="ms"))
    d=tracking_timing_data(unknown)
    @test d["time_unit"]===nothing && d["effective_time_unit"]=="ms"
    @test d["effective_time_unit_provenance"]=="legacy_scale_same_unit"
    @test_throws ArgumentError with_scale(timed,PhysicalScale(time_unit="ms"))
    epoch=BigInt(typemax(Int64))+1
    half=timing_tracking_wrap(Trajectory(1,[0.0,1.0,2.0],[0.0,0.0,0.0]),[epoch,epoch+1//2,epoch+1])
    @test trajectory_velocities(half,1)[1]==[2,2,2]
    huge=big(10)^1000
    slow=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,1.0]),[0,huge])
    @test_throws ArgumentError trajectory_velocities(slow,1)
    @test_throws ArgumentError Hammerhead._tracking_product(Float32,1.0,1//big(10)^100)
    @test_throws ArgumentError Hammerhead._tracking_product(Float64,1.0,1//big(10)^1000)
    @test_throws ArgumentError Hammerhead._tracking_predict(Float64,floatmax(Float64),floatmax(Float64),2//1)
    finite_positions=timing_tracking_wrap(Trajectory(1,[-floatmax(Float64),floatmax(Float64)],[0.0,0.0]),[0,big(10)^310])
    @test isfinite(trajectory_velocities(finite_positions,1)[1][1])
    tiny_positions=timing_tracking_wrap(Trajectory{Float32}(1,Float32[1,2],Float32[1,2],[1,2]),[0,1])
    @test_throws ArgumentError physical(with_scale(tiny_positions,PhysicalScale(pixel_size=1e-50)))
    @test_throws ArgumentError physical(with_scale(tiny_positions,PhysicalScale(pixel_size=1e50)))
    forged=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,0.0]),[0,1])
    forged.result.trajectories[1].x[2]=9
    for operation in (tracking_timing_data,physical,r->with_scale(r,PhysicalScale()),r->trajectory_velocities(r,1))
        @test_throws ArgumentError operation(forged)
    end
    packet=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,0.0]),[0,1])
    packet.timing._data["clock_id"]="changed"
    @test_throws ArgumentError tracking_timing_data(packet)
end

@testset "Actual-time artifact and table" begin
    timed=timing_tracking_wrap(Trajectory{Float64}(1,[0.0,8.0,10.0],[0.0,4.0,5.0],[1,3,4]),[0,1,4,5];unit="s")
    mktempdir() do dir
        file=joinpath(dir,"timed.jld2")
        csv=joinpath(dir,"timed.csv")
        @test save_timed_tracking(file,timed)==file
        loaded=load_timed_tracking(file)
        @test trajectory_velocities(loaded,1)==trajectory_velocities(timed,1)
        @test loaded.result.trajectories[1].frames==[1,3,4]
        @test_throws ArgumentError load_results(file)
        @test_throws ArgumentError ResultFile(file)
        @test_throws ArgumentError save_timed_tracking(file,loaded)
        @test_throws ArgumentError export_table(file,loaded)
        destination=joinpath(dir,"preserved.jld2")
        write(destination,"existing")
        @test_throws ArgumentError save_results(destination,loaded)
        @test read(destination,String)=="existing"
        @test_throws ArgumentError save_timed_tracking(dir,timed)
        # Source-path and hard-link aliases are guarded even after an artifact
        # round trip. Use the committed fixture path without loading unrelated
        # frames; preflight only snapshots path provenance for this binding test.
        sourcepath=joinpath(pkgdir(Hammerhead),"test","reference_images","A","A001_1.tif")
        snapshot=Hammerhead._tracking_preflight([sourcepath,sourcepath],[0,1],nothing,nothing,nothing,nothing)
        source_bound=Hammerhead._tracking_bind(TrackingResult([Trajectory(1,[0.0,1.0],[0.0,0.0])],2,PTVParameters()),snapshot.data)
        @test_throws ArgumentError save_timed_tracking(sourcepath,source_bound)
        @test_throws ArgumentError export_table(sourcepath,source_bound)
        sourceartifact=joinpath(dir,"source.jld2")
        save_timed_tracking(sourceartifact,source_bound)
        @test_throws ArgumentError save_timed_tracking(sourcepath,load_timed_tracking(sourceartifact))
        alias=joinpath(dir,"alias.jld2")
        hardlink(file,alias)
        @test_throws ArgumentError save_timed_tracking(alias,loaded)
        @test_throws ArgumentError export_table(alias,loaded)
        @test_throws ArgumentError save_results(destination,[loaded])
        @test read(destination,String)=="existing"
        too_slow=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,1.0]),[0,big(10)^1000])
        write(csv,"preserved")
        @test_throws ArgumentError export_table(csv,too_slow)
        @test read(csv,String)=="preserved"
        # Finite zero secants permit an unrepresentable elapsed-time coordinate:
        # only its optional decimal column is empty, exact metadata is retained.
        zero=timing_tracking_wrap(Trajectory(1,[1.0,1.0],[2.0,2.0]),[0,big(10)^1000])
        export_table(csv,zero)
        exact_rows=timing_tracking_rows(csv)
        @test exact_rows[2]["elapsed_time"]==""
        @test exact_rows[2]["elapsed_time_numerator"]==string(big(10)^1000)
        rawpath=joinpath(dir,"legacy.jld2")
        save_results(rawpath,timed.result)
        @test_throws ArgumentError load_timed_tracking(rawpath)
        @test only(load_results(rawpath)) isa TrackingResult
        @test export_table(csv,loaded)==csv
        rows=timing_tracking_rows(csv)
        @test length(rows)==3
        @test all(r->r["schema_version"]=="hammerhead-tracking-time-table-1",rows)
        @test [r["frame_index"] for r in rows]==["1","3","4"]
        @test [r["elapsed_time"] for r in rows]==["0.0","4.0","5.0"]
        @test [r["gap_before"] for r in rows]==["0","1","0"]
        @test [r["u"] for r in rows]==fill("2.0",3)
        @test [r["velocity_start_frame"] for r in rows]==["1","1","3"]
        @test [r["velocity_end_frame"] for r in rows]==["3","4","4"]
        @test all(r->r["time_provenance"]=="actual_sample_times" && r["velocity_unit"]=="px/s",rows)
        @test [r["sample_time_numerator"] for r in rows]==["0","4","5"]
        scaled=with_scale(loaded,PhysicalScale(pixel_size=0.25,dt=99,length_unit="mm",time_unit="s"))
        export_table(csv,scaled)
        expected=read(csv,String)
        export_table(csv,physical(scaled))
        @test read(csv,String)==expected
        anonymous=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,0.0]),[0,1])
        export_table(csv,anonymous)
        @test all(r->r["time_unit"]==r["velocity_unit"]=="",timing_tracking_rows(csv))
        original=tracking_timing_data(loaded)
        for mutate in (d->d["tracking_timing_format_version"]=true,
                       d->d["reference_interval"]["numerator"]="2",
                       d->d["sample_times"][2]["numerator"]="0",
                       d->d["effective_time_unit"]="ms",
                       d->d["position_basis"]="scaled_length",
                       d->d["result_sha256"]=repeat("0",64))
            bad=deepcopy(original);mutate(bad)
            invalid=joinpath(dir,"malformed.jld2")
            jldopen(invalid,"w") do f
                f["timed_tracking_format_version"]=1
                f["tracking_result"]=loaded.result
                f["tracking_timing"]=bad
                f["tracking_timing_sha256"]=Hammerhead._history_digest(bad)
            end
            @test_throws ArgumentError load_timed_tracking(invalid)
        end
        for version in (true,1.0,2)
            badfile=joinpath(dir,"version.jld2")
            jldopen(badfile,"w") do f;f["timed_tracking_format_version"]=version;end
            @test_throws ArgumentError load_timed_tracking(badfile)
        end
        broken=timing_tracking_wrap(Trajectory(1,[0.0,1.0],[0.0,0.0]),[0,1])
        broken.result.trajectories[1].x[1]=7
        write(csv,"preserved")
        @test_throws ArgumentError export_table(csv,broken)
        @test read(csv,String)=="preserved"
        @test_throws ArgumentError save_timed_tracking(destination,broken)
        @test read(destination,String)=="existing"
    end
end
