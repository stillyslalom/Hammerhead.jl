using Hammerhead, Test, Random, JLD2

function stereo_execution_fixture(::Type{T}=Float64;grid=DewarpGrid(x=1.:48.,y=1.:48.)) where T
    cameras=(PinholeCamera([100.0 0 15.0 0;0 100.0 0 0;0 0 1.0 100.0]),
             PinholeCamera([100.0 0 -15.0 0;0 100.0 0 0;0 0 1.0 100.0]))
    dewarpers=map(c->ImageDewarper(c,grid,(48,48)),cameras)
    rng=MersenneTwister(915)
    A1=rand(rng,T,48,48);A2=rand(rng,T,48,48)
    (frames=(A1,circshift(A1,(1,2)),A2,circshift(A2,(-1,1))),dewarpers=dewarpers,grid=grid)
end
stereo_execution_params(;kwargs...)=PIVParameters(;window_size=16,overlap=8,padding=true,
    uod_enable=false,validation=(),replace_outliers=false,kwargs...)
function stereo_execution_equal(a,b)
    fields=(:x,:y,:z,:u,:v,:w,:uncertainty_u,:uncertainty_v,:uncertainty_w,:mask,:outliers,:scale)
    all(k->isequal(getfield(a,k),getfield(b,k)),fields) &&
        all(k->isequal(getfield(a.cam1,k),getfield(b.cam1,k)) && isequal(getfield(a.cam2,k),getfield(b.cam2,k)),
            (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,:mask,:outliers,:correlation_planes))
end
function stereo_execution_rebuild(result;kwargs...)
    StereoPIVResult((get(kwargs,k,getfield(result,k)) for k in fieldnames(typeof(result)))...)
end
function stereo_execution_unsafe_digest(result)
    data=Dict{String,Any}(String(k)=>getfield(result,k) for k in
        (:x,:y,:z,:u,:v,:w,:uncertainty_u,:uncertainty_v,:uncertainty_w,:mask,:outliers))
    data["scale"]=Hammerhead._timing_scale(result.scale)
    data["cam1_measurement_sha256"]=Hammerhead._history_result_digest(result.cam1)
    data["cam2_measurement_sha256"]=Hammerhead._history_result_digest(result.cam2)
    Hammerhead._history_digest(data)
end
function stereo_execution_rewrite(path;mutate=identity,result=nothing,rehash=true)
    jldopen(path,"r+") do file
        key=Hammerhead._stereo_execution_key(Hammerhead.result_key(1))
        entry=file[key];mutate(entry)
        if result!==nothing
            delete!(file,Hammerhead.result_key(1));file[Hammerhead.result_key(1)]=result
            entry["diagnostics"]["measurement_sha256"]=stereo_execution_unsafe_digest(result)
        end
        rehash && (entry["diagnostics_sha256"]=Hammerhead._history_digest(entry["diagnostics"]))
        delete!(file,key);file[key]=entry
    end
end

@testset "Stereo camera diagnostics preserve numerical results" begin
    for T in (Float32,Float64),backend in (:cpu,:ka)
        fixture=stereo_execution_fixture(T);frames=fixture.frames;dw1,dw2=fixture.dewarpers
        schedule=multipass_parameters([24,16];uod_enable=false,validation=(),replace_outliers=false,
            final=(;max_iterations=4,convergence_tol=1e6,uncertainty=true))
        plain=run_piv_stereo(frames...,dw1,dw2,schedule;backend,threaded=false)
        packet=Ref{Any}(nothing);calls=Ref(0)
        measured=run_piv_stereo(frames...,dw1,dw2,schedule;backend,threaded=false,
            on_diagnostics=d->(packet[]=d;calls[]+=1))
        @test calls[]==1 && stereo_execution_equal(plain,measured)
        d=packet[];data=execution_diagnostics_data(d)
        @test isimmutable(d) && isimmutable(d.cam1) && isimmutable(d.geometry)
        @test length(unique([d.execution_id,d.cam1.execution_id,d.cam2.execution_id]))==3
        @test d.pair_index===d.cam1.pair_index===d.cam2.pair_index===nothing
        @test d.cam1.backend==d.cam2.backend==backend && d.cam1.image_type==d.cam2.image_type==string(T)
        @test length(d.cam1.passes)==length(d.cam2.passes)==2
        @test last(d.cam1.passes).executed_iterations==last(d.cam2.passes).executed_iterations==2
        @test last(d.cam1.passes).checks==last(d.cam2.passes).checks==1
        @test all(c->c["parent_execution_id"]==d.execution_id,data["cameras"])
        @test getindex.(data["cameras"],"camera_role")==[1,2]
        @test all(p->p.residual.unit=="px",(d.cam1.passes...,d.cam2.passes...))
        @test data["verification"]["inspection_state"]=="captured_measurement_fields"
        @test data["verification"]["measurement_field_binding_checked"]===true
        @test data["verification"]["calibration"]===data["verification"]["source_inputs"]===false
        @test d.geometry.measurement_shape==size(measured.u)
        @test occursin("dewarped pixels",sprint(show,MIME"text/plain"(),d))
        data["cameras"][1]["diagnostics"]["passes"][1]["executed_iterations"]=99
        data["geometry"]["x"]["step"]=99.
        @test d.cam1.passes[1].executed_iterations==1 && d.geometry.x.step==1.
    end
    fixture=stereo_execution_fixture();dw1,dw2=fixture.dewarpers;frames=fixture.frames
    d=Ref{Any}(nothing)
    result=run_piv_stereo(frames...,dw1,dw2,stereo_execution_params();threaded=false,on_diagnostics=x->(d[]=x))
    @test only(d[].cam1.passes).stop_reason===:single_sweep
    @test only(d[].cam1.passes).residual.mean_magnitude!=only(d[].cam2.passes).residual.mean_magnitude
    @test_throws ErrorException run_piv_stereo(frames...,dw1,dw2,stereo_execution_params();on_diagnostics=x->error("consumer"))
    flat=zeros(48,48)
    run_piv_stereo(flat,flat,flat,flat,dw1,dw2,stereo_execution_params(max_iterations=4,convergence_tol=1.);
        threaded=false,on_diagnostics=x->(d[]=x))
    @test all(c->only(c.passes).last_check.included_count==0 && only(c.passes).residual.mean_magnitude===nothing,(d[].cam1,d[].cam2))
    @test all(c->only(c.passes).stop_reason===:tolerance_condition_met,(d[].cam1,d[].cam2))
    effort_plain=run_piv_stereo(frames...,dw1,dw2;effort=:low,threaded=false)
    effort_capture=run_piv_stereo(frames...,dw1,dw2;effort=:low,threaded=false,on_diagnostics=x->(d[]=x))
    @test stereo_execution_equal(effort_plain,effort_capture)
end

@testset "Stereo diagnostics signed world grid, ROI and scale" begin
    grid=DewarpGrid(x=1.:0.5:24.5,y=48.:-1.:1.,z=0.)
    for T in (Float32,Float64)
        fixture=stereo_execution_fixture(T;grid);dw1,dw2=fixture.dewarpers
        scale=PhysicalScale(dt=.001,length_unit="mm",time_unit="s")
        packet=Ref{Any}(nothing)
        options=(;threaded=false,roi=ROI(5:44,7:42),scale)
        plain=run_piv_stereo(fixture.frames...,dw1,dw2,stereo_execution_params();options...)
        result=run_piv_stereo(fixture.frames...,dw1,dw2,stereo_execution_params();options...,on_diagnostics=d->(packet[]=d))
        d=packet[]
        @test stereo_execution_equal(plain,result)
        @test d.geometry.x.step==.5 && d.geometry.y.step==-1.
        @test d.cam1.processing_size==d.cam2.processing_size==(40,36)
        @test all(>(0),diff(result.cam1.x)) && all(<(0),diff(result.y))
        @test result.scale===scale && result.cam1.scale===result.cam2.scale===nothing
        @test only(d.cam1.passes).residual.unit=="px"
        @test_throws ArgumentError Hammerhead._stereo_execution_check_result(d,physical(result))
    end
end

@testset "Stereo sequence callbacks, persistence and bounded lifetime" begin
    fixture=stereo_execution_fixture();frames=fixture.frames;dw1,dw2=fixture.dewarpers;p=stereo_execution_params()
    mktempdir() do dir
        path=joinpath(dir,"shared.jld2");events=Tuple{Symbol,Int}[];packets=StereoPIVExecutionDiagnostics[];refs=WeakRef[]
        acquisitions=fill(frames,3)
        @test run_piv_stereo_sequence(acquisitions,dw1,dw2,p;output=path,record_diagnostics=true,
            collect_results=false,threaded=false,progress=(i,n)->push!(events,(:progress,i)),
            on_diagnostics=(i,d)->(push!(events,(:diagnostics,i));push!(packets,d)),
            on_result=(i,r)->(push!(events,(:result,i));append!(refs,WeakRef.((r.u,r.cam1.u,r.cam2.u)))))===nothing
        @test events==[(kind,i) for i in 1:3 for kind in (:diagnostics,:result,:progress)]
        GC.gc();GC.gc()
        @test all(w->w.value===nothing,refs) # Retaining scalar packets does not retain results.
        index=ResultFile(path)
        @test length(index)==3 && length(load_results(path))==3
        for i in 1:3
            meta=load_stereo_execution_diagnostics(index,i)
            verified=load_stereo_execution_diagnostics(index,i;verify_result=true)
            @test meta.pair_index==meta.cam1.pair_index==meta.cam2.pair_index==i
            @test execution_diagnostics_data(meta)["verification"]["inspection_state"]=="metadata_only"
            @test execution_diagnostics_data(meta)["verification"]["measurement_field_binding_checked"]===false
            @test execution_diagnostics_data(verified)["verification"]["inspection_state"]=="loaded_measurement_fields_verified"
            @test verified.measurement_sha256==packets[i].measurement_sha256
            @test load_execution_diagnostics(index,i)===nothing
        end
        for effort in (nothing,:low)
            output=(i,acq)->joinpath(dir,"pair-$(effort)-$i.jld2")
            first_pairs=fill(frames[1:2],2);second_pairs=fill(frames[3:4],2)
            arguments=effort===nothing ? (p,) : ()
            keywords=effort===nothing ? (;) : (;effort)
            run_piv_stereo_sequence(first_pairs,second_pairs,dw1,dw2,arguments...;keywords...,output,
                record_diagnostics=true,collect_results=false,progress=false,threaded=false)
            @test load_stereo_execution_diagnostics(output(2,nothing);verify_result=true).pair_index==2
        end
        direct=run_piv_stereo(frames...,dw1,dw2,p;threaded=false)
        plain=joinpath(dir,"bare.jld2");save_results(plain,direct)
        @test load_stereo_execution_diagnostics(plain)===nothing
        preloaded=Ref(false)
        @test_throws ArgumentError run_piv_stereo_sequence(acquisitions,dw1,dw2,p;record_diagnostics=true,
            preprocess=x->(preloaded[]=true;x))
        @test !preloaded[]
        private_sentinel=joinpath(dir,"private-preserved.jld2");write(private_sentinel,"preserved")
        path_calls=Ref(0)
        for output in (private_sentinel,(i,a)->(path_calls[]+=1;private_sentinel))
            @test_throws ArgumentError run_piv_stereo_sequence(acquisitions,dw1,dw2,p;output,
                record_diagnostics=true,_camera_diagnostics=(Ref{Any}(nothing),Ref{Any}(nothing)),
                preprocess=x->(preloaded[]=true;x),progress=false)
            @test filesize(private_sentinel)==9 && read(private_sentinel,String)=="preserved"
        end
        @test !preloaded[] && path_calls[]==0
        # Callback-only capture must check mutations before opening function output.
        protected=joinpath(dir,"preserved.jld2");write(protected,"preserved")
        current=Ref{Any}(nothing)
        @test_throws ArgumentError run_piv_stereo_sequence([frames],dw1,dw2,p;
            threaded=false,progress=false,collect_results=false,on_diagnostics=(i,d)->nothing,
            on_result=(i,r)->(current[]=r),output=(i,a)->(current[].cam2.u[1]+=1;protected))
        @test filesize(protected)==9 && read(protected,String)=="preserved"
        progressed=joinpath(dir,"progressed.jld2")
        @test_throws ArgumentError run_piv_stereo_sequence([frames],dw1,dw2,p;threaded=false,
            record_diagnostics=true,output=(i,a)->progressed,on_result=(i,r)->(current[]=r),
            progress=(i,n)->(current[].cam1.u[1]=99.))
        @test load_stereo_execution_diagnostics(progressed;verify_result=true) isa StereoPIVExecutionDiagnostics
        @test_throws ArgumentError run_piv_stereo_sequence([frames],dw1,dw2,p;
            threaded=false,progress=false,on_diagnostics=(i,d)->nothing,
            on_result=(i,r)->(r.w[1]=1.),output=(i,a)->protected)
        @test filesize(protected)==9 && read(protected,String)=="preserved"
        @test_throws ErrorException run_piv_stereo_sequence([frames],dw1,dw2,p;
            threaded=false,progress=false,on_diagnostics=(i,d)->error("consumer"),output=(i,a)->protected)
        @test filesize(protected)==9 && read(protected,String)=="preserved"
        calls=Ref(0);cancelled=joinpath(dir,"cancelled.jld2")
        @test run_piv_stereo_sequence(acquisitions,dw1,dw2,p;threaded=false,progress=false,
            record_diagnostics=true,output=cancelled,collect_results=false,
            on_diagnostics=(i,d)->(calls[]+=1),cancel=()->calls[]>=1)===nothing
        @test calls[]==1 && length(ResultFile(cancelled))==1
        @test load_stereo_execution_diagnostics(cancelled;verify_result=true).pair_index==1
        @test_throws ArgumentError run_piv_stereo_ensemble([frames[1:2]],[frames[3:4]],dw1,dw2,p;on_diagnostics=identity)

        # A failed diagnostics consumer must join an outstanding loader, and
        # its original exception takes precedence over the loader's failure.
        started=Channel{Nothing}(1);release=Channel{Nothing}(1);failing=Channel{Nothing}(1)
        finished=Threads.Atomic{Bool}(false)
        source=FrameSource(8,i->begin
            if i==5
                put!(started,nothing);take!(release);finished[]=true
                error("background stereo load failed")
            end
            frames[mod1(i,4)]
        end)
        selected=[ntuple(k->FrameRef(source,4*(i-1)+k),4) for i in 1:2]
        primary=ErrorException("stereo diagnostics consumer failed")
        driver=Threads.@spawn try
            run_piv_stereo_sequence(selected,dw1,dw2,p;threaded=false,progress=false,
                collect_results=false,on_diagnostics=(i,d)->begin
                    timedwait(()->isready(started),30.)==:ok || error("stereo prefetch gate was not reached")
                    take!(started);put!(failing,nothing);throw(primary)
                end)
            nothing
        catch err
            err
        end
        try
            @test timedwait(()->isready(failing) || istaskdone(driver),30.)==:ok
            if isready(failing)
                take!(failing)
                @test timedwait(()->istaskdone(driver),.1)==:timed_out
            end
        finally
            put!(release,nothing)
        end
        @test fetch(driver)===primary && finished[]
    end
end

@testset "Stereo diagnostic schema and independent binding checks" begin
    fixture=stereo_execution_fixture();frames=fixture.frames;dw1,dw2=fixture.dewarpers;p=stereo_execution_params()
    mktempdir() do dir
        original=joinpath(dir,"original.jld2")
        run_piv_stereo_sequence([frames],dw1,dw2,p;threaded=false,progress=false,record_diagnostics=true,output=original)
        packet=load_stereo_execution_diagnostics(original;verify_result=true)
        for (i,mutate) in enumerate((e->e["result_key"]="results/999999",
            e->e["diagnostics"]["stereo_diagnostics_format_version"]=true,
            e->e["diagnostics"]["cameras"][1]["camera_role"]=true,
            e->e["diagnostics"]["cameras"][1]["parent_execution_id"]="wrong",
            e->e["diagnostics"]["cameras"][2]["diagnostics"]["execution_id"]=e["diagnostics"]["cameras"][1]["diagnostics"]["execution_id"],
            e->e["diagnostics"]["cameras"][1]["diagnostics"]["processing_size"]=[40,48],
            e->e["diagnostics"]["cameras"][1]["diagnostics"]["passes"][1]["requested_tolerance"]=1.,
            e->e["diagnostics"]["geometry"]["x"]["step"]=0.,
            e->e["diagnostics"]["geometry"]["measurement_shape"]=[1,1],
            e->e["diagnostics"]["cameras"][1]["diagnostics"]["passes"][1]["executed_iterations"]=0))
            path=joinpath(dir,"malformed-$i.jld2");cp(original,path)
            stereo_execution_rewrite(path;mutate)
            @test_throws ArgumentError load_stereo_execution_diagnostics(path)
        end
        mismatch=joinpath(dir,"checksum.jld2");cp(original,mismatch)
        stereo_execution_rewrite(mismatch;mutate=e->e["diagnostics"]["geometry"]["z"]=1.,rehash=false)
        @test_throws ArgumentError load_stereo_execution_diagnostics(mismatch)
        result=only(load_results(original))
        for (i,changed) in enumerate((stereo_execution_rebuild(result;u=ones(1,1)),
            stereo_execution_rebuild(result;x=result.x .+ 1),
            stereo_execution_rebuild(result;cam1=result.cam2,cam2=result.cam1)))
            path=joinpath(dir,"edited-$i.jld2");cp(original,path)
            stereo_execution_rewrite(path;result=changed)
            @test load_stereo_execution_diagnostics(path) isa StereoPIVExecutionDiagnostics
            @test_throws ArgumentError load_stereo_execution_diagnostics(path;verify_result=true)
        end
        wrong_camera=PIVResult((k===:mask ? falses(1,1) : getfield(result.cam1,k)
            for k in fieldnames(typeof(result.cam1)))...)
        malformed_camera=joinpath(dir,"camera-dimensions.jld2");cp(original,malformed_camera)
        stereo_execution_rewrite(malformed_camera;result=stereo_execution_rebuild(result;cam1=wrong_camera),
            mutate=e->e["diagnostics"]["cameras"][1]["measurement_sha256"]=Hammerhead._history_result_digest(wrong_camera))
        @test load_stereo_execution_diagnostics(malformed_camera) isa StereoPIVExecutionDiagnostics
        @test_throws ArgumentError load_stereo_execution_diagnostics(malformed_camera;verify_result=true)
        edited=deepcopy(result);edited.cam1.u[1]+=1
        @test_throws ArgumentError Hammerhead._stereo_execution_check_result(packet,edited)
        @test_throws BoundsError load_stereo_execution_diagnostics(original,2)
        version=joinpath(dir,"version.jld2");cp(original,version)
        jldopen(version,"r+") do file
            delete!(file,"stereo_execution_diagnostics_format_version");file["stereo_execution_diagnostics_format_version"]=2
        end
        @test_throws ArgumentError load_stereo_execution_diagnostics(version)
    end
end
