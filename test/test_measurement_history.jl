using Test, Hammerhead, Random, JLD2

struct HistoryCustomValidator <: PIVValidator
    calls::Base.RefValue{Int}
end
function Hammerhead.apply_validator!(r::PIVResult,v::HistoryCustomValidator)
    v.calls[]+=1
    r.u[1]=7
    r.outliers[1]=true
    r
end

function history_fixture(p; shape=(3,3),u=zeros(shape),v=zeros(shape),ratio=fill(2.,shape),mask=falses(shape))
    PIVResult(collect(1.:shape[2]),collect(1.:shape[1]),copy(u),copy(v),copy(ratio),zeros(shape),
        fill(NaN,shape),fill(NaN,shape),falses(shape),copy(mask),p)
end
function history_observe_fixture(r;alternatives=nothing,force=false)
    h=Hammerhead._HistoryObservation(eltype(r.u),size(r.u),r.parameters)
    Hammerhead._history_begin!(h,r.u,r.v,r.u,r.v,1)
    Hammerhead.validate_and_replace!(r,r.parameters,force;alternatives,history=h)
    h,Hammerhead._history_finish(h,r,:cpu,(64,64),1)
end
function history_same_result(a,b)
    all(k->isequal(getfield(a,k),getfield(b,k)),(:x,:y,:u,:v,:peak_ratio,:correlation_moment,
        :uncertainty_u,:uncertainty_v,:outliers,:mask,:correlation_planes,:scale))
end

@testset "Final-sweep planar measurement history" begin
    rng=MersenneTwister(2821)
    A=rand(rng,48,48); B=circshift(A,(1,2))
    base=(;window_size=16,overlap=8,padding=true,uod_enable=false,validation=(),replace_outliers=false)
    @testset "Unchanged CPU/KA numerics, final pass/sweep, coordinates" begin
        for T in (Float32,Float64),backend in (:cpu,:ka),iterations in (1,2,4)
            a,b=T.(A),T.(B)
            p=PIVParameters(;base...,max_iterations=iterations,convergence_tol=1e6,uncertainty=true,n_peaks=3)
            plain=run_piv(a,b,p;backend,threaded=true)
            captured=Ref{Any}(); d=Ref{Any}()
            observed=run_piv(a,b,p;backend,threaded=true,on_measurement_history=h->(captured[]=h),on_diagnostics=x->(d[]=x))
            @test history_same_result(plain,observed)
            h=captured[]; data=measurement_history_data(h)
            @test data["pass_index"]==1 && data["sweep_index"]==(iterations==4 ? 2 : iterations)
            @test data["image_type"]==string(T) && data["backend"]==String(backend)
            @test data["x"]==observed.x && data["y"]==observed.y && data["mask"]==observed.mask
            @test data["final_outliers"]==observed.outliers
            @test data["history_id"]!=d[].execution_id
            @test data["measurement_sha256"]==Hammerhead._history_result_digest(observed)
            @test data["association"]===nothing && data["pair_index"]===nothing
            @test eltype(data["primary_u"])==T && size(data["primary_u"])==size(observed.u)
            @test occursin("Applicability, accuracy and coverage are not established",sprint(show,MIME"text/plain"(),h))
            data["primary_u"][1]=999
            @test measurement_history_data(h)["primary_u"][1]!=999
        end
        mask=falses(64,64); mask[15:33,19:35].=true
        image=rand(rng,64,64); roi=ROI(9:56,13:60); scale=PhysicalScale(pixel_size=.2,dt=.05)
        captured=Ref{Any}()
        passes=[PIVParameters(;base...,window_size=24,overlap=12),PIVParameters(;base...,max_iterations=2)]
        r=run_piv(image,image,passes;roi,mask,scale,on_measurement_history=h->(captured[]=h))
        data=measurement_history_data(captured[])
        @test data["pass_index"]==2 && data["sweep_index"]==2 && data["processing_size"]==[48,48]
        @test data["x"]==r.x && data["y"]==r.y && minimum(data["x"])>13 && minimum(data["y"])>9
        @test data["measurement_sha256"]==Hammerhead._history_result_digest(r)
        @test all(isnan,data["primary_residual_u"][data["mask"]])
        @test all(==(4),data["final_origin"][data["mask"]])
        @test all(==(0),data["first_rejection_stage"][data["mask"]])
        @test all(==(4),data["uncertainty_u_status"][data["mask"]])
        @test Hammerhead._history_digest(data)==Hammerhead._experiment_digest(data) # spans canonical hash buffer boundaries
    end

    @testset "Actual rejection, alternative and median events" begin
        p=PIVParameters(;base...,n_peaks=3,uncertainty=true,validation=(VelocityMagnitudeValidator(0,1),))
        u=zeros(3,3); u[2,2]=20
        r=history_fixture(p;u)
        r.uncertainty_u .= .2; r.uncertainty_v .= .3
        altu=fill(NaN,3,3,2); altv=copy(altu)
        altu[2,2,1]=3; altv[2,2,1]=0; altu[2,2,2]=0; altv[2,2,2]=0
        _,h=history_observe_fixture(r;alternatives=(altu,altv))
        data=measurement_history_data(h)
        @test data["first_rejection_stage"][2,2]==3
        @test data["primary_u"][2,2]==20 && data["primary_residual_u"][2,2]==20
        @test data["accepted_peak_rank"][2,2]==3 && data["final_origin"][2,2]==2
        @test data["pre_substitution_outliers"][2,2] && !r.outliers[2,2] && r.u[2,2]==0
        @test !any(data["fill_attempted"]) && r.peak_ratio[2,2]==2 # remains primary metric
        @test data["uncertainty_u_status"][2,2]==1 && r.uncertainty_u[2,2]==.2 # not reestimated for rank 3
        p=PIVParameters(;base...,validation=(VelocityMagnitudeValidator(0,1),PeakRatioValidator(3)))
        _,h=history_observe_fixture(history_fixture(p;u))
        data=measurement_history_data(h)
        @test data["first_rejection_stage"][2,2]==3
        @test data["first_rejection_stage"][1,1]==4 # first cause only, not both criteria
        p=PIVParameters(;base...,replace_outliers=true,uncertainty=true)
        ratio=fill(2.,3,3); ratio[2,2]=.5
        r=history_fixture(p;ratio)
        r.uncertainty_u .= .2
        _,h=history_observe_fixture(r)
        data=measurement_history_data(h)
        @test data["fill_attempted"][2,2] && data["fill_assigned"][2,2] && r.u[2,2]==data["primary_u"][2,2]==0
        @test data["final_origin"][2,2]==3 && data["final_outliers"][2,2]
        @test data["uncertainty_u_status"][2,2]==1 && r.uncertainty_u[2,2]==.2 # not propagated for a fill
        p=PIVParameters(;base...,replace_outliers=true)
        _,h=history_observe_fixture(history_fixture(p;shape=(1,2),u=reshape([NaN,0.],1,2)))
        data=measurement_history_data(h)
        @test data["fill_attempted"][1] && !data["fill_assigned"][1] && data["final_origin"][1]==0
        @test data["first_rejection_stage"][1]==1
        mask=falses(3,3); mask[2,2]=true
        masked=history_fixture(p;mask,u=fill(NaN,3,3),ratio=fill(NaN,3,3))
        _,h=history_observe_fixture(masked)
        @test !measurement_history_data(h)["fill_attempted"][2,2]
        # The existing median branch can assign nonfinite donors; record its
        # execution even when no finite replacement results.
        obs=Hammerhead._HistoryObservation(Float64,(3,3),p)
        invalid=falses(3,3); invalid[2,2]=true
        ru=fill(NaN,3,3); rv=copy(ru)
        Hammerhead.replace_vectors!(ru,rv,invalid;history=obs,history_mask=falses(3,3))
        @test obs.fill_assigned[2,2] && isnan(ru[2,2])
        calls=Ref(0); custom=HistoryCustomValidator(calls)
        p=PIVParameters(;base...,validation=(custom,))
        captured=Ref{Any}()
        plain=run_piv(A,B,p;threaded=false); @test calls[]==1
        calls[]=0
        observed=run_piv(A,B,p;threaded=false,on_measurement_history=h->(captured[]=h))
        @test calls[]==1 && history_same_result(plain,observed)
        data=measurement_history_data(captured[])
        @test data["custom_validators_present"] && data["final_origin"][1]==5
        @test last(data["rejection_stages"])["kind"]=="custom"
        # Iteration fills are internal when unreplaced primary values are
        # restored at still-flagged nodes after the final sweep.
        p=PIVParameters(;base...,max_iterations=2,validation=(PeakRatioValidator(1e9),))
        run_piv(A,B,p;on_measurement_history=h->(captured[]=h))
        data=measurement_history_data(captured[])
        @test all(data["primary_restored"]) && all(data["fill_attempted"]) && !any(data["fill_assigned"])
        @test all(==(1),data["final_origin"])
        # Hand-driven final-sweep restoration also covers successful internal
        # assignment followed by restoration, not an inferred final fill.
        p=PIVParameters(;base...)
        ratio=fill(2.,3,3); ratio[2,2]=.5
        r=history_fixture(p;ratio,u)
        obs=Hammerhead._HistoryObservation(Float64,(3,3),p)
        Hammerhead._history_begin!(obs,r.u,r.v,r.u,r.v,2)
        Hammerhead.validate_and_replace!(r,p,true;history=obs)
        r.u[r.outliers]=obs.primary_u[r.outliers]; r.v[r.outliers]=obs.primary_v[r.outliers]
        obs.primary_restored .= r.outliers
        data=measurement_history_data(Hammerhead._history_finish(obs,r,:cpu,(64,64),1))
        @test data["fill_assigned"][2,2] && data["primary_restored"][2,2] && data["final_origin"][2,2]==1
        @test r.u[2,2]==20
        values=reshape([0.,-1.,NaN,Inf],2,2)
        @test Hammerhead._history_uq_status(values,falses(2,2),true)==UInt8[1 3;2 3]
        @test all(==(0),Hammerhead._history_uq_status(values,falses(2,2),false))
        @test occursin("alternatives_not_reestimated",data["uncertainty_basis"])
    end

    mktempdir() do dir
        p=PIVParameters(;base...); pairs=[(A,B),(A,B),(A,B)]
        @testset "Streaming lifetime, callback ordering and mutation guards" begin
            path=joinpath(dir,"stream.jld2"); events=Tuple[]; weak=WeakRef[]
            result=run_piv_sequence(pairs,p;output=path,record_measurement_history=true,collect_results=false,progress=(i,n)->push!(events,(:progress,i)),
                on_measurement_history=(i,h)->begin
                    push!(events,(:history,i)); push!(weak,WeakRef(h._data["primary_u"]))
                end,on_result=(i,r)->push!(events,(:result,i)))
            @test result===nothing && events==[(:history,1),(:result,1),(:progress,1),(:history,2),(:result,2),(:progress,2),(:history,3),(:result,3),(:progress,3)]
            GC.gc(true); @test all(w->w.value===nothing,weak)
            @test length(ResultFile(path))==3 && load_measurement_history(path,3;verify_result=true)._data["pair_index"]==3
            @test_throws ArgumentError run_piv_sequence([("missing","missing")],p;record_measurement_history=true,progress=false)
            @test_throws ErrorException run_piv(A,B,p;on_measurement_history=h->error("callback"))
            @test_throws ArgumentError run_piv(A,B,p;on_measurement_history=h->(h._data["primary_u"][1]=999))
            for mode in (:single,:pair),mutation in (:result,:history)
                target=joinpath(dir,"$(mode)-$(mutation).jld2")
                mode==:pair && write(target,"unchanged")
                output=mode==:single ? target : (i,pair)->target
                @test_throws ArgumentError run_piv_sequence(pairs,p;output,record_measurement_history=true,collect_results=false,progress=false,
                    on_measurement_history=(i,h)->(mutation==:history && (h._data["primary_u"][1]=123)),
                    on_result=(i,r)->(mutation==:result && (r.u[1]=123)))
                @test mode==:single ? isempty(ResultFile(target)) : read(target,String)=="unchanged"
            end
            failure=joinpath(dir,"prefix.jld2")
            @test_throws ErrorException run_piv_sequence(pairs,p;output=failure,record_measurement_history=true,progress=false,on_measurement_history=(i,h)->(i==2 && error("pair2")))
            @test length(ResultFile(failure))==1 && load_measurement_history(failure)._data["pair_index"]==1
            prefetched=fill(false,4)
            source=FrameSource(4,i->begin
                i>=3 && sleep(.05)
                prefetched[i]=true
                A
            end)
            lazy_pairs=[(FrameRef(source,1),FrameRef(source,2)),(FrameRef(source,3),FrameRef(source,4))]
            @test_throws ErrorException run_piv_sequence(lazy_pairs,p;progress=false,collect_results=false,on_measurement_history=(i,h)->error("delivery"))
            @test all(prefetched) # pending loader joined on history callback failure
            run_piv_sequence(pairs,p;output=(i,pair)->joinpath(dir,"pair-$i.jld2"),record_measurement_history=true,progress=false,collect_results=false)
            @test load_measurement_history(joinpath(dir,"pair-3.jld2"))._data["pair_index"]==3
            run_piv_sequence(pairs;effort=:low,output=path,record_measurement_history=true,progress=false,collect_results=false)
            @test load_measurement_history(path,2;verify_result=true)._data["pair_index"]==2
            for key in (:x,:uncertainty_u,:mask,:outliers)
                target=joinpath(dir,"binding-$(key).jld2")
                @test_throws ArgumentError run_piv_sequence(pairs,p;output=target,record_measurement_history=true,collect_results=false,progress=false,
                    on_result=(i,r)->begin
                        array=getfield(r,key)
                        array[1]=eltype(array)===Bool ? !array[1] : 123
                    end)
                @test isempty(ResultFile(target))
            end
            mixed=joinpath(dir,"both.jld2")
            combined=Symbol[]
            run_piv_sequence(pairs[1:1],p;output=mixed,record_measurement_history=true,record_diagnostics=true,progress=false,
                on_diagnostics=(i,d)->push!(combined,:diagnostics),on_measurement_history=(i,h)->push!(combined,:history),on_result=(i,r)->push!(combined,:result))
            @test combined==[:diagnostics,:history,:result]
            @test load_measurement_history(mixed;verify_result=true) isa PIVMeasurementHistory && load_execution_diagnostics(mixed) isa PIVExecutionDiagnostics
        end

        @testset "Native metadata-only loading, schema, exact binding" begin
            captured=Ref{Any}(); r=run_piv(A,B,p;on_measurement_history=h->(captured[]=h))
            data=measurement_history_data(captured[]); path=joinpath(dir,"schema.jld2")
            function write_history(change=identity;version=1,key="results/000007",declared=key,payload=r,digest=true)
                candidate=deepcopy(data); change(candidate)
                jldopen(path,"w") do file
                    file["format_version"]=1; file[key]=payload
                    file["measurement_history_format_version"]=version
                    file[Hammerhead._history_key(key)]=Dict("result_key"=>declared,"history"=>candidate,
                        "history_sha256"=>digest ? Hammerhead._history_digest(candidate) : "0"^64)
                end
            end
            write_history(;payload="not a result")
            @test measurement_history_data(load_measurement_history(path))==data
            @test_throws ArgumentError ResultFile(path)[1]
            @test_throws ArgumentError load_measurement_history(path;verify_result=true)
            write_history(); @test measurement_history_data(load_measurement_history(path;verify_result=true))==data
            for change in (d->d["history_format_version"]=true,d->d["pair_index"]=true,
                           d->d["sweep_index"]=0,d->d["n_peaks"]=false,d->d["final_origin"][1]=3,
                           d->d["accepted_peak_rank"][1]=1,d->d["first_rejection_stage"][1]=999,
                           d->d["fill_assigned"][1]=true,d->d["uncertainty_u_status"][1]=4,
                           d->d["mask"][1]=true,d->d["primary_u"]=zeros(Float32,2,2),
                           d->d["rejection_stages"][1]["name"]="incorrect",
                           d->begin
                               d["n_peaks"]=3; d["accepted_peak_rank"][1]=2
                               d["first_rejection_stage"][1]=2; d["pre_substitution_outliers"][1]=true
                               d["final_origin"][1]=2; d["fill_attempted"][1]=true
                           end,
                           d->d["processing_size"]=[typemax(Int),typemax(Int)])
                write_history(change); @test_throws ArgumentError load_measurement_history(path)
            end
            write_history(;version=2); @test_throws ArgumentError load_measurement_history(path)
            @test ResultFile(path)[1] isa PIVResult
            write_history(;declared="results/000001"); @test_throws ArgumentError load_measurement_history(path)
            write_history(;digest=false); @test_throws ArgumentError load_measurement_history(path)
            changed=deepcopy(r); changed.u[1]+=1
            write_history(;payload=changed)
            @test load_measurement_history(path) isa PIVMeasurementHistory
            @test_throws ArgumentError load_measurement_history(path;verify_result=true)
            save_results(path,r)
            @test load_measurement_history(path)===nothing
            jldopen(path,"w") do f
                f["format_version"]=1; f["results/000001"]=r
                f["measurement_history/000001"]=Dict()
            end
            @test_throws ArgumentError load_measurement_history(path)
            stale=ResultFile(path); save_results(path,r)
            @test_throws ArgumentError load_measurement_history(stale,1)
            @test_throws BoundsError load_measurement_history(path,2)
        end

        @testset "Unsupported workflow guards and replay association" begin
            target=joinpath(dir,"protected.jld2"); write(target,"unchanged")
            @test_throws ArgumentError run_ptv_sequence([("missing","missing")];output=target,record_measurement_history=true,progress=false)
            @test read(target,String)=="unchanged"
            @test_throws ArgumentError run_piv_ensemble([("missing","missing")];effort=:low,on_measurement_history=identity)
            @test_throws MethodError run_piv_ensemble([("missing","missing")],p;on_measurement_history=identity)
            @test_throws ArgumentError Hammerhead._reject_execution_diagnostics((on_measurement_history=nothing,),"stereo")
            fixtures=joinpath(pkgdir(Hammerhead),"test","reference_images","A")
            files=sort(filter(f->endswith(lowercase(f),".tif"),readdir(fixtures;join=true)))
            recipe=PIVRecipe(p;roi=ROI(1:48,1:48),image_type=Float32,threaded=false)
            record=ExperimentRecord([(files[1],files[2])],recipe)
            path=joinpath(dir,"replay.jld2"); delivered=Ref{Any}()
            run=replay_experiment(record;output=path,record_measurement_history=true,on_measurement_history=(i,h)->(delivered[]=h))
            data=measurement_history_data(load_measurement_history(path;verify_result=true))
            @test data["association"]==Dict("recipe_id"=>recipe_identity(recipe),"input_id"=>record.input_id)
            @test data==measurement_history_data(delivered[])
            @test run.recipe_id==recipe_identity(recipe) && run.output_sha256==Hammerhead._experiment_file_digest(path)
            bad=joinpath(dir,"failed-replay.jld2")
            @test_throws ArgumentError replay_experiment(record;output=bad,record_measurement_history=true,on_measurement_history=(i,h)->(h._data["primary_u"][1]=123))
            @test isempty(ResultFile(bad))
        end
    end
end
