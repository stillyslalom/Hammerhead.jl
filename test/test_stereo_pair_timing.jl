using Hammerhead, Test, Random, JLD2

stiming_exact(v)=v===nothing ? nothing : parse(BigInt,v["numerator"])//parse(BigInt,v["denominator"])
function stiming_fixture(::Type{T}=Float64;grid=DewarpGrid(x=1.:48.,y=1.:48.)) where T
    cams=(PinholeCamera([100.0 0 15.0 0;0 100.0 0 0;0 0 1.0 100.0]),
          PinholeCamera([100.0 0 -15.0 0;0 100.0 0 0;0 0 1.0 100.0]))
    dw=map(c->ImageDewarper(c,grid,(48,48)),cams)
    rng=MersenneTwister(1295);a=rand(rng,T,48,48);b=rand(rng,T,48,48)
    (;frames=(a,circshift(a,(1,2)),b,circshift(b,(-1,1))),dw,grid)
end
stiming_params()=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,validation=(),replace_outliers=false)
function stiming_source(f,role,timestamps;kwargs...)
    loads=Int[]
    source=FrameSource(length(timestamps),i->(push!(loads,i);f.frames[2role-1+iseven(i)]);
        timestamps,labels=["camera$(role)-$i" for i in eachindex(timestamps)],source_id="camera$role",kwargs...)
    source,loads
end
function stiming_pairs(source)
    [(FrameRef(source,i),FrameRef(source,i+1)) for i in 1:2:length(source)-1]
end
function stiming_equal(a,b)
    all(k->isequal(getfield(a,k),getfield(b,k)),(:x,:y,:z,:u,:v,:w,:uncertainty_u,:uncertainty_v,:uncertainty_w,:mask,:outliers,:scale)) &&
        all(c->all(k->isequal(getfield(getfield(a,c),k),getfield(getfield(b,c),k)),
            (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,:mask,:outliers)),(:cam1,:cam2))
end
function stiming_no_payloads(x)
    x isa AbstractDict && return all(stiming_no_payloads,values(x))
    x isa AbstractArray && return ndims(x)==1 && all(stiming_no_payloads,x)
    x===nothing || x isa Union{Number,AbstractString}
end
function stiming_preflight(p1,p2;scale=nothing,atol=0.,rtol=sqrt(eps(Float64)),sync_atol=0.,sync_rtol=0.,missing_timestamps=:allow)
    a=[(x[1],x[2],y[1],y[2]) for (x,y) in zip(p1,p2)]
    Hammerhead._stereo_timing_preflight(a,p1,p2,scale,atol,rtol,sync_atol,sync_rtol,missing_timestamps)
end

@testset "Stereo timing exact preflight and two camera references" begin
    f=stiming_fixture();epoch=typemax(Int64)-20
    s1,l1=stiming_source(f,1,[epoch,epoch+3];time_unit="ns",clock_id="clock",frame_ids=["A","B"])
    s2,l2=stiming_source(f,2,[epoch+1,epoch+4];time_unit="ns",clock_id="clock")
    p1,p2=stiming_pairs(s1),stiming_pairs(s2)
    d=only(stiming_preflight(p1,p2;sync_atol=1.))
    @test isempty(l1) && isempty(l2)
    @test stiming_exact(d["cameras"][1]["sample_time"])==big(epoch)+3//2
    @test stiming_exact(d["cameras"][2]["sample_time"])==big(epoch)+5//2
    @test d["reconstructed_time_reference"]==d["cameras"][1]["sample_time"]
    @test d["reconstructed_time_reference_convention"]=="camera1_provided_timestamp_midpoint_not_common_exposure_time"
    @test stiming_exact.(d["synchronization"]["exposure_offsets_camera2_minus_camera1"])==[1,1]
    @test d["synchronization"]["clock_unit_status"]=="provided_labels_consistent"
    @test d["cameras"][1]["frames"][1]["source_identity"]=="opaque_provided"
    @test d["effective_delay"]===nothing && d["effective_delay_provenance"]=="unavailable"
    @test_throws ArgumentError stiming_preflight(p1,p2)
    @test_throws ArgumentError stiming_preflight(p1,p2;sync_atol=-1)
    @test_throws ArgumentError stiming_preflight(p1,p2;sync_rtol=Inf)
    for bad in (Any[0,true],Any[0,NaN],Any[0,Inf],[1,0],[0,0],BigFloat[0,1])
        source,_=stiming_source(f,1,bad)
        @test_throws ArgumentError stiming_preflight(stiming_pairs(source),stiming_pairs(source))
    end
    # Native integer subtraction would overflow: encode/promote before subtracting.
    wide,_=stiming_source(f,1,[typemin(Int64),typemax(Int64)])
    w=only(stiming_preflight(stiming_pairs(wide),stiming_pairs(wide)))
    @test stiming_exact(w["cameras"][1]["observed_delay"])==big(typemax(Int64))-big(typemin(Int64))
    @test stiming_exact(w["reconstructed_time_reference"])==-1//2
    floating,_=stiming_source(f,1,[.1,.4])
    fp=image_pairs(floating)
    @test only(stiming_preflight(fp,fp))["cameras"][1]["timestamp_status"]=="complete"
    @test_throws ArgumentError stiming_preflight(fp,fp;rtol=0.)
    conflicting=[FramePair(p1[1][1],p1[1][2],2.)]
    @test_throws ArgumentError stiming_preflight(conflicting,conflicting)
    missing_source,_=stiming_source(f,1,Any[missing,2])
    partial=only(stiming_preflight(stiming_pairs(missing_source),stiming_pairs(missing_source)))
    @test partial["cameras"][1]["timestamp_status"]=="partial"
    @test partial["reconstructed_time_reference"]===nothing
    @test partial["synchronization"]["comparison_status"]=="available_numeric_comparisons_only"
    @test_throws ArgumentError stiming_preflight(stiming_pairs(missing_source),stiming_pairs(missing_source);missing_timestamps=:error)
    for metadata in ((;time_unit="s",clock_id="clock"),(;time_unit="ns",clock_id="other"))
        other,_=stiming_source(f,2,[epoch,epoch+3];metadata...)
        @test_throws ArgumentError stiming_preflight(p1,stiming_pairs(other))
    end
    @test_throws ArgumentError stiming_preflight(p1,p2;sync_atol=1.,scale=PhysicalScale(time_unit="s"))
end

@testset "Stereo timing numerical parity and legacy scale provenance" begin
    for T in (Float32,Float64),backend in (:cpu,:ka)
        f=stiming_fixture(T;grid=DewarpGrid(x=1.:.5:24.5,y=48.:-1.:1.));dw1,dw2=f.dw
        s1,_=stiming_source(f,1,[0,2]);s2,_=stiming_source(f,2,[0,2])
        p1,p2=image_pairs(s1),image_pairs(s2)
        scale=PhysicalScale(dt=7.,length_unit="mm",time_unit="ns")
        options=(;backend,threaded=false,progress=false,image_type=T,roi=ROI(5:44,7:42),scale)
        plain=run_piv_stereo_sequence(p1,p2,dw1,dw2,stiming_params();options...)
        packets=StereoPairTiming[]
        measured=run_piv_stereo_sequence(p1,p2,dw1,dw2,stiming_params();options...,on_pair_timing=(i,d)->push!(packets,d))
        @test stiming_equal(only(plain),only(measured))
        packet=only(packets);data=pair_timing_data(packet;result=only(measured))
        @test stiming_no_payloads(data) && isimmutable(packet)
        @test data["effective_delay_provenance"]=="pair_dt_scale_override" && stiming_exact(data["effective_delay"])==2
        @test data["scale_applied"]===true && data["geometry"]["y"]["step"]==-1.
        @test data["verification"]["inspection_state"]=="supplied_measurement_fields_verified"
        @test data["verification"]["calibration"]===data["verification"]["source_inputs"]===false
        @test pair_timing_data(packet)["verification"]["inspection_state"]=="captured_measurement_fields"
        @test_throws ArgumentError pair_timing_data(packet;result=physical(only(measured)))
        acq=[(p1[1][1],p1[1][2],p2[1][1],p2[1][2])]
        tuple_plain=run_piv_stereo_sequence(acq,dw1,dw2,stiming_params();options...)
        tuple_measured=run_piv_stereo_sequence(acq,dw1,dw2,stiming_params();options...,on_pair_timing=(i,d)->push!(packets,d))
        @test stiming_equal(only(tuple_plain),only(tuple_measured))
        tuple_data=pair_timing_data(last(packets))
        @test stiming_exact(tuple_data["effective_delay"])==7 && tuple_data["effective_delay_provenance"]=="supplied_scale_dt"
        @test tuple_data["effective_agrees_with_camera1_observed"]===false
    end
    f=stiming_fixture();dw1,dw2=f.dw
    s,_=stiming_source(f,1,[0,2]);p=image_pairs(s);packet=Ref{Any}(nothing)
    run_piv_stereo_sequence(p,p,dw1,dw2,stiming_params();threaded=false,progress=false,on_pair_timing=(i,d)->(packet[]=d))
    data=pair_timing_data(packet[])
    @test data["effective_delay_provenance"]=="declared_pair_dt_metadata_only" && data["scale_applied"]===false
    @test data["geometry"]["camera_basis"]=="dewarped_pixels_x_columns_y_rows"
    copied=pair_timing_data(packet[]);copied["cameras"][1]["frames"][1]["label"]="edited"
    @test pair_timing_data(packet[])["cameras"][1]["frames"][1]["label"]!="edited"
end

function stiming_rewrite(path,change;rehash=true)
    jldopen(path,"r+") do file
        key="stereo_pair_timing/000001";entry=file[key];change(entry)
        rehash && (entry["timing_sha256"]=Hammerhead._experiment_digest(entry["timing"]))
        delete!(file,key);file[key]=entry
    end
end
@testset "Stereo timing persistence, schema and raw linkage" begin
    f=stiming_fixture();dw1,dw2=f.dw;p=stiming_params()
    s1,_=stiming_source(f,1,[0,2,4,7]);s2,_=stiming_source(f,2,[0,2,4,7]);p1,p2=image_pairs(s1),image_pairs(s2)
    mktempdir() do dir
        path=joinpath(dir,"shared.jld2");events=Tuple{Symbol,Int}[]
        results=run_piv_stereo_sequence(p1,p2,dw1,dw2,p;threaded=false,output=path,record_pair_timing=true,record_diagnostics=true,
            on_diagnostics=(i,d)->push!(events,(:diagnostics,i)),on_pair_timing=(i,d)->push!(events,(:timing,i)),
            on_result=(i,r)->push!(events,(:result,i)),progress=(i,n)->push!(events,(:progress,i)))
        @test events==[(k,i) for i in 1:2 for k in (:diagnostics,:timing,:result,:progress)]
        index=ResultFile(path)
        @test jldopen(f->f["stereo_pair_timing_format_version"],path,"r")===1
        @test load_pair_timing(index,1)===nothing
        @test load_stereo_execution_diagnostics(index,2;verify_result=true) isa StereoPIVExecutionDiagnostics
        @test load_stereo_pair_timing(index,2) isa StereoPairTiming
        @test pair_timing_data(load_stereo_pair_timing(index,2))["verification"]["measurement_field_binding_checked"]===false
        @test pair_timing_data(load_stereo_pair_timing(path,2;verify_result=true))["verification"]["measurement_field_binding_checked"]===true
        @test pair_timing_data(load_stereo_pair_timing(path,2))["input_sequence_index"]==2
        @test jldopen(f->f["sources/000002"],path,"r")==["camera1-3","camera1-4","camera2-3","camera2-4"]
        @test_throws BoundsError load_stereo_pair_timing(index,3)
        bare=joinpath(dir,"bare.jld2");save_results(bare,results)
        @test load_stereo_pair_timing(bare)===nothing
        paths=[joinpath(dir,"pair$i.jld2") for i in 1:2]
        run_piv_stereo_sequence(p1,p2,dw1,dw2,p;threaded=false,progress=false,record_pair_timing=true,output=(i,a)->paths[i])
        @test pair_timing_data(load_stereo_pair_timing(paths[2];verify_result=true))["input_sequence_index"]==2
        packet=load_stereo_pair_timing(path)
        raw=index[1];raw.cam2.u[1]+=1
        @test_throws ArgumentError pair_timing_data(packet;result=raw)
        saved=read(path)
        for change in (e->(e["result_key"]="results/000002"),
                       e->(e["timing"]["stereo_pair_timing_format_version"]=2),
                       e->(e["timing"]["cameras"][1]["camera_role"]=2),
                       e->(e["timing"]["scale_applied"]=0),
                       e->(e["timing"]["effective_agrees_with_camera1_observed"]=1),
                       e->(e["timing"]["reconstructed_time_reference"]=e["timing"]["cameras"][2]["declared_delay"]),
                       e->begin
                           e["timing"]["cameras"][1]["frames"][1]["clock_id"]="clock"
                           e["timing"]["cameras"][2]["frames"][1]["clock_id"]="other"
                       end,
                       e->(e["timing"]["cameras"][1]["frames"]=nothing),
                       e->(e["timing"]["cameras"][1]["frames"]=[]))
            bad=joinpath(dir,"malformed.jld2");write(bad,saved);stiming_rewrite(bad,change)
            @test_throws ArgumentError load_stereo_pair_timing(bad;verify_result=true)
        end
        shape=joinpath(dir,"shape.jld2");write(shape,saved)
        stiming_rewrite(shape,e->(e["timing"]["geometry"]["measurement_shape"]=[1,1]))
        @test load_stereo_pair_timing(shape) isa StereoPairTiming
        @test_throws ArgumentError load_stereo_pair_timing(shape;verify_result=true)
        snapshot=first(stiming_preflight(p1,p2;scale=PhysicalScale(dt=7.)))
        applied=Hammerhead._stereo_timing_bind(snapshot,with_scale(results[1],PhysicalScale(dt=2.)),f.grid)
        inconsistent=deepcopy(applied._data);inconsistent["scale"]["dt"]=3.
        forged=StereoPairTiming(inconsistent,Hammerhead._experiment_digest(inconsistent),:metadata_only)
        @test_throws ArgumentError pair_timing_data(forged)
        bad=joinpath(dir,"tamper.jld2");write(bad,saved)
        stiming_rewrite(bad,e->(e["timing"]["cameras"][1]["frames"][1]["label"]="changed");rehash=false)
        @test_throws ArgumentError load_stereo_pair_timing(bad)
        for marker in (true,2)
            jldopen(bad,"w") do file
                file["format_version"]=1;file["results/000001"]=results[1];file["stereo_pair_timing_format_version"]=marker
            end
            @test_throws ArgumentError load_stereo_pair_timing(bad)
        end
    end
end

struct StimingCountingSource <: AbstractFrameSource
    timestamps::Vector{Float32}
    reads::Vector{Int}
end
Base.length(s::StimingCountingSource)=length(s.timestamps)
Hammerhead.frame_timestamp(s::StimingCountingSource,i)=(s.reads[i]+=1;s.timestamps[i])
Hammerhead.frame_source_label(::StimingCountingSource,i)="counted-$i"
@testset "Stereo timestamp getter snapshot and normalized tolerance" begin
    source=StimingCountingSource(Float32[0,1,0,1],zeros(Int,4))
    p1=[(FrameRef(source,1),FrameRef(source,2))];p2=[(FrameRef(source,3),FrameRef(source,4))]
    data=only(stiming_preflight(p1,p2;sync_atol=Float32(.01),sync_rtol=Float32(.02)))
    @test source.reads==ones(Int,4)
    @test data["synchronization_policy"]["atol"]===Float64(Float32(.01))
    @test stiming_exact(data["synchronization"]["delay_scaled_bound"])==Rational{BigInt}(Float64(Float32(.01))+Float64(Float32(.02)))
    @test all(f->f["timestamp"]["numeric_type"]=="Float32",Iterators.flatten(c["frames"] for c in data["cameras"]))
    for role1 in (Float32[0,.3],Float32[0,1]), delta in (Float32(.01),Float32(.03))
        a=FrameSource(2,identity;timestamps=role1)
        b=FrameSource(2,identity;timestamps=role1.+delta)
        p1,p2=stiming_pairs(a),stiming_pairs(b)
        # All-Float64 tolerances preserve the native floating gate boundary.
        options=(;sync_atol=Float64(delta),sync_rtol=.001)
        legacy=try Hammerhead._check_stereo_pair_times(p1,p2;options...);true catch e;e isa ArgumentError || rethrow();false end
        captured=try stiming_preflight(p1,p2;options...);true catch e;e isa ArgumentError || rethrow();false end
        @test captured===legacy
    end
    for (ta,tb) in (([0,2],[0.,1.]),([0.,1.],[0,2]),(Any[big(0),2.],[0.,2.]),
                    (Any[0//big(1),2.],[0,2]))
        a=FrameSource(2,identity;timestamps=ta);b=FrameSource(2,identity;timestamps=tb)
        d=only(stiming_preflight(stiming_pairs(a),stiming_pairs(b);sync_atol=1.))
        @test stiming_exact(d["synchronization"]["delay_scaled_bound"])==1
        # Declared floating and exact intervals must not accidentally promote to
        # unsupported BigFloat arithmetic inside synchronization checks.
        pa=[FramePair(FrameRef(a,1),FrameRef(a,2),Float64(ta[2]-ta[1]))]
        pb=[FramePair(FrameRef(b,1),FrameRef(b,2),Rational{BigInt}(tb[2]-tb[1]))]
        @test only(stiming_preflight(pa,pb;sync_atol=1.))["synchronization"]["comparison_status"]=="all_numeric_comparisons_checked"
    end
    huge=big(10)^400
    for rtol in (0.,1e-300)
        a=FrameSource(2,identity;timestamps=[big(0),huge])
        b=FrameSource(2,identity;timestamps=[0.,1.])
        pa=[FramePair(FrameRef(a,1),FrameRef(a,2),huge)]
        pb=[FramePair(FrameRef(b,1),FrameRef(b,2),1.)]
        # The complete acquisition refuses actual delay/skew disagreement. Test
        # the bound itself separately so rejection cannot conceal type/range bugs.
        @test_throws ArgumentError stiming_preflight(pa,pb;sync_atol=1.,sync_rtol=rtol)
        t=Hammerhead._timing_tolerance(0.,rtol)
        @test Hammerhead._stereo_timing_sync_bound(Any[huge//big(1),1.],t)==Rational{BigInt}(rtol)*huge
        d=only(stiming_preflight(stiming_pairs(a),[(zeros(1,1),zeros(1,1))];sync_rtol=rtol))
        @test stiming_exact(d["synchronization"]["delay_scaled_bound"])==Rational{BigInt}(rtol)*huge
    end
end

@testset "Stereo timing callbacks and loader cleanup" begin
    f=stiming_fixture();dw1,dw2=f.dw;p=stiming_params()
    mktempdir() do dir
        path=joinpath(dir,"callbacks.jld2");held=Ref{Any}(nothing)
        write(path,"preserved")
        @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;threaded=false,progress=false,
            on_pair_timing=(i,d)->(held[]=d),on_result=(i,r)->(r.v[1]+=1),output=(i,a)->path)
        @test read(path,String)=="preserved"
        # The progress mutation is detected after publication; its entry survives.
        @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;threaded=false,output=path,record_pair_timing=true,
            on_pair_timing=(i,d)->(held[]=d),progress=(i,n)->(held[]._data["input_sequence_index"]=99))
        @test length(ResultFile(path))==1
        @test load_stereo_pair_timing(path;verify_result=true) isa StereoPairTiming
        for callback in (:diagnostics,:timing)
            started=Channel{Nothing}(1);release=Channel{Nothing}(1);failed=Channel{Nothing}(1)
            joined=Threads.Atomic{Bool}(false)
            source=FrameSource(8,i->begin
                if i==5
                    put!(started,nothing);take!(release);joined[]=true;error("background loader failed")
                end
                f.frames[mod1(i,4)]
            end)
            acquisitions=[ntuple(k->FrameRef(source,4*(i-1)+k),4) for i in 1:2]
            primary=ErrorException("$callback consumer failed")
            consumer=(i,d)->begin
                timedwait(()->isready(started),30.)==:ok || error("prefetch did not start")
                take!(started);put!(failed,nothing);throw(primary)
            end
            driver=Threads.@spawn try
                run_piv_stereo_sequence(acquisitions,dw1,dw2,p;threaded=false,progress=false,collect_results=false,
                    on_diagnostics=callback===:diagnostics ? consumer : nothing,
                    on_pair_timing=callback===:timing ? consumer : (i,d)->nothing)
                nothing
            catch e
                e
            end
            try
                @test timedwait(()->isready(failed) || istaskdone(driver),30.)==:ok
                isready(failed) && (@test timedwait(()->istaskdone(driver),.05)==:timed_out)
            finally
                put!(release,nothing)
            end
            @test fetch(driver)===primary && joined[]
        end
        completed=Ref(0);loads=Int[]
        source=FrameSource(8,i->(push!(loads,i);f.frames[mod1(i,4)]))
        acq=[ntuple(k->FrameRef(source,4*(i-1)+k),4) for i in 1:2]
        @test run_piv_stereo_sequence(acq,dw1,dw2,p;threaded=false,progress=false,collect_results=false,
            record_pair_timing=true,output=path,on_pair_timing=(i,d)->(completed[]+=1),cancel=()->completed[]>0)===nothing
        @test completed[]==1 && sort(loads)==collect(1:8)
        @test length(ResultFile(path))==1 && load_stereo_pair_timing(path;verify_result=true) isa StereoPairTiming
    end
end

@testset "Stereo timing frozen selections and pre-open rejection" begin
    f=stiming_fixture();dw1,dw2=f.dw;p=stiming_params()
    mktempdir() do dir
        path=joinpath(dir,"protected.jld2");write(path,"preserved")
        s1,l1=stiming_source(f,1,Any[0,1,2,NaN]);s2,l2=stiming_source(f,2,[0,1,2,3])
        @test_throws ArgumentError run_piv_stereo_sequence(stiming_pairs(s1),stiming_pairs(s2),dw1,dw2,p;record_pair_timing=true,output=path,progress=false)
        @test isempty(l1) && isempty(l2) && read(path,String)=="preserved"
        @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;record_pair_timing=true,progress=false)
        for kw in ((;timing_pairs2=[]),(;scale_pairs=[]),(;_timing_snapshots=[]))
            @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;record_pair_timing=true,output=path,progress=false,kw...)
        end
        timestamps=collect(0:5);s1,l1=stiming_source(f,1,timestamps;frame_ids=string.(1:6))
        s2,l2=stiming_source(f,2,collect(0:5));pairs1=[collect(x) for x in stiming_pairs(s1)];pairs2=stiming_pairs(s2)
        packets=StereoPairTiming[];seen=Any[]
        run_piv_stereo_sequence(pairs1,pairs2,dw1,dw2,p;threaded=false,progress=false,
            on_pair_timing=(i,d)->begin
                push!(packets,d)
                if i==1
                    timestamps[5]=999;s1.labels[5]="changed";s1.frame_ids[5]="changed"
                    pairs1[3][1]=FrameRef(s1,1)
                end
            end,output=(i,a)->(push!(seen,a);joinpath(dir,"frozen$i.jld2")),record_pair_timing=true)
        lastdata=pair_timing_data(last(packets))
        @test lastdata["cameras"][1]["frames"][1]["frame_index"]==5
        @test stiming_exact(lastdata["cameras"][1]["frames"][1]["timestamp"])==4
        @test lastdata["cameras"][1]["frames"][1]["label"]=="camera1-5"
        @test lastdata["cameras"][1]["frames"][1]["frame_id"]=="5"
        @test last(seen)[1].index==5 && count(==(5),l1)==1
        held=Ref{Any}(nothing);raw=Ref{Any}(nothing);diag=Ref{Any}(nothing)
        for mutate in (:packet,:result,:diagnostics)
            write(path,"preserved")
            @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;threaded=false,progress=false,
                on_pair_timing=(i,d)->(held[]=d),on_diagnostics=(i,d)->(diag[]=d),on_result=(i,r)->(raw[]=r),
                output=(i,a)->begin
                    if mutate===:packet
                        held[]._data["input_sequence_index"]=99
                    elseif mutate===:result
                        raw[].w[1]+=1
                    else
                        # Diagnostics are immutable, but their associated captured
                        # camera fields can still be modified by a consumer.
                        raw[].cam1.v[1]+=1
                    end
                    path
                end)
            @test read(path,String)=="preserved"
        end
        @test_throws ArgumentError run_piv_stereo_sequence([f.frames],dw1,dw2,p;threaded=false,progress=false,
            on_pair_timing=(i,d)->(d._data["scale_applied"]=true),output=(i,a)->path)
        @test read(path,String)=="preserved"
        # Selected path aliases are rejected without loading their invalid bytes.
        inputs=fill((path,path,path,path),1)
        @test_throws ArgumentError run_piv_stereo_sequence(inputs,dw1,dw2,p;record_pair_timing=true,output=path,progress=false)
        @test read(path,String)=="preserved"
        @test_throws ArgumentError run_piv_stereo_ensemble([(f.frames[1],f.frames[2])],[(f.frames[3],f.frames[4])],dw1,dw2,p;record_pair_timing=true)
        @test_throws ArgumentError run_piv_stereo(f.frames...,dw1,dw2,p;on_pair_timing=x->nothing)
    end
end

Base.@noinline function stiming_release_probe()
    f=stiming_fixture();dw1,dw2=f.dw;refs=WeakRef[];packets=WeakRef[]
    result=run_piv_stereo_sequence(fill(f.frames,3),dw1,dw2,stiming_params();threaded=false,progress=false,collect_results=false,
        on_result=(i,r)->append!(refs,WeakRef.((r,r.u,r.cam1.u,r.cam2.u))),on_pair_timing=(i,p)->push!(packets,WeakRef(p._data)))
    (;result,refs,packets)
end
@testset "Stereo timing current-payload lifetime" begin
    probe=stiming_release_probe();GC.gc();GC.gc()
    @test probe.result===nothing
    @test all(x->x.value===nothing,probe.refs)
    @test all(x->x.value===nothing,probe.packets)
end
