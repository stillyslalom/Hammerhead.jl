using Test, Hammerhead, Random, JLD2

struct EnsembleCaptureValidator <: PIVValidator
    result::Base.RefValue{Any}
end
Hammerhead.apply_validator!(r::PIVResult,v::EnsembleCaptureValidator)=(v.result[]=r;r)
ensemble_diag_params(;kwargs...)=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,validation=(),replace_outliers=false;kwargs...)
function ensemble_diag_same(a,b)
    all(k->isequal(getfield(a,k),getfield(b,k)),(:x,:y,:u,:v,:peak_ratio,:correlation_moment,
        :uncertainty_u,:uncertainty_v,:mask,:outliers,:correlation_planes,:scale))
end
function ensemble_diag_observe(pairs,passes;kwargs...)
    d=Ref{Any}();r=run_piv_ensemble(pairs,passes;threaded=false,progress=false,on_diagnostics=x->(d[]=x),kwargs...)
    r,d[]
end
function ensemble_diag_no_arrays(x)
    x isa Union{PIVResult,PIVParameters} && return false
    x isa AbstractDict && return all(ensemble_diag_no_arrays,values(x))
    x isa AbstractArray && return ndims(x)==1 && all(ensemble_diag_no_arrays,x)
    x isa Union{NamedTuple,Tuple} && return all(ensemble_diag_no_arrays,x)
    x isa Union{EnsemblePassDiagnostics,EnsemblePIVExecutionDiagnostics} && return all(k->ensemble_diag_no_arrays(getfield(x,k)),fieldnames(typeof(x)))
    x===nothing || x isa Union{Number,AbstractString,Symbol}
end
function ensemble_diag_rewrite(path,mutate;rehash=true)
    key="ensemble_execution_diagnostics/000001"
    jldopen(path,"r+") do f
        entry=f[key];mutate(entry)
        rehash && (entry["diagnostics_sha256"]=Hammerhead._experiment_digest(entry["diagnostics"]))
        delete!(f,key);f[key]=entry
    end
end
function ensemble_diag_replace(path,key,value)
    jldopen(path,"r+") do f
        haskey(f,key) && delete!(f,key);f[key]=value
    end
end

@testset "Ensemble pooled execution diagnostics" begin
    rng=MersenneTwister(4717);A=rand(rng,48,48);B=circshift(A,(1,2))
    @testset "Numerical parity and ignored iteration settings" begin
        for T in (Float32,Float64),backend in (:cpu,:ka)
            pairs=[(T.(A),T.(B)),(T.(A),T.(B))]
            passes=[ensemble_diag_params(max_iterations=7,convergence_tol=Inf,uncertainty=true),
                    ensemble_diag_params(max_iterations=4,convergence_tol=1e9,uncertainty=true)]
            scale=PhysicalScale(pixel_size=.01,dt=2.)
            plain=run_piv_ensemble(pairs,passes;backend,threaded=false,progress=false,scale)
            result,d=ensemble_diag_observe(pairs,passes;backend,scale)
            @test ensemble_diag_same(plain,result)
            @test d isa EnsemblePIVExecutionDiagnostics && d.pair_count==2
            @test all(p->p.executed_pooling_sweeps==1 && p.tolerance_checks==0 && p.stop_reason===:pooled_once_iteration_settings_ignored,d.passes)
            @test [p.requested_iterations for p in d.passes]==[7,4]
            @test first(d.passes).requested_tolerance==Inf
            @test !first(d.passes).predictor_present && last(d.passes).predictor_present
            @test first(d.passes).uncertainty.reason=="intermediate_pass" && last(d.passes).uncertainty.evaluated
            @test all(p->p.residual.unit=="px" && p.uncertainty.unit=="px",d.passes)
            @test last(d.passes).residual.mean_magnitude<first(d.passes).residual.mean_magnitude
            @test execution_diagnostics_data(d;result)["verification"]["inspection_state"]=="supplied_measurement_fields_verified"
            @test_throws ArgumentError execution_diagnostics_data(d;result=physical(result))
            @test ensemble_diag_no_arrays(d) && ensemble_diag_no_arrays(execution_diagnostics_data(d))
            data=execution_diagnostics_data(d);data["passes"][1]["contributions"]["nonfinite_planes"]=999
            @test execution_diagnostics_data(d)["passes"][1]["contributions"]["nonfinite_planes"]==0
            @test occursin("ignored",sprint(show,MIME"text/plain"(),d))
        end
        result,d=ensemble_diag_observe([(A,B)],ensemble_diag_params();preprocess=img->Float32.(img),image_type=Float64)
        @test d.requested_image_type=="Float64" && only(d.passes).image_type=="Float32" && result isa PIVResult{Float32}
        d=Ref{Any}();result=run_piv_ensemble([(A,B)];effort=:high,window_size=16,progress=false,threaded=false,on_diagnostics=x->(d[]=x))
        @test length(d[].passes)>1 && all(p->p.executed_pooling_sweeps==1,d[].passes)
    end

    @testset "Contribution/source-support/UQ populations" begin
        for T in (Float32,Float64),backend in (:cpu,:ka)
            a,b=T.(A),T.(B);Z=fill(T(.1),48,48)
            p=ensemble_diag_params(uncertainty=true)
            _,single=ensemble_diag_observe([(Z,Z),(a,b)],p;backend)
            s=only(single.passes)
            @test s.contributions.finite_zero_planes==s.contributor_nodes.eligible_count
            @test s.contributions.source_gated_window_pairs==0
            @test s.uncertainty.admitted_window_pair_updates==2s.contributor_nodes.eligible_count
            @test s.contributor_nodes.some_finite_nonzero_count==s.contributor_nodes.eligible_count
            r,multi=ensemble_diag_observe([(Z,Z),(a,b)],[p,p];backend)
            s=last(multi.passes)
            @test s.source_support.evaluated_pairs==2
            @test s.contributions.source_gated_window_pairs>=s.contributor_nodes.eligible_count
            @test s.contributions.accumulated_window_pairs+s.contributions.source_gated_window_pairs==2s.contributor_nodes.eligible_count
            @test s.uncertainty.admitted_window_pair_updates==s.contributions.accumulated_window_pairs
            empty,d=ensemble_diag_observe([(Z,Z),(Z,Z)],[p,p];backend)
            @test all(empty.outliers) && all(isnan,empty.uncertainty_u)
            @test !last(d.passes).predictor_present && last(d.passes).source_support.no_predictor_pairs==2
            @test last(d.passes).contributions.finite_zero_planes==2last(d.passes).contributor_nodes.eligible_count
            @test last(d.passes).residual.finite_count==0 && last(d.passes).residual.mean_magnitude===nothing
            masked,d=ensemble_diag_observe([(a,b)],p;backend,mask=trues(48,48))
            s=only(d.passes)
            @test s.contributor_nodes.eligible_count==0 && s.contributions.masked_window_pairs==length(masked.mask)
            @test s.contributor_nodes.minimum_finite_nonzero_contributions===nothing && s.uncertainty.reason=="no_eligible_windows"
            bad=copy(a);bad[20,20]=T(NaN)
            _,d=ensemble_diag_observe([(a,b),(bad,b)],[p,p];backend)
            @test last(d.passes).source_support.disabled_nonfinite_source_pairs==1
            @test last(d.passes).contributions.nonfinite_planes>0
            tiny,td=ensemble_diag_observe([(T(1e-6).*a,T(1e-6).*b)],p;backend)
            @test only(td.passes).contributions.finite_nonflat_planes>0 && any(isfinite,tiny.u)
        end
        # Masked source texture does not create an informative original stencil.
        mask=falses(48,48);mask[1:8,:].=true
        _,d=ensemble_diag_observe([(A,B)],ensemble_diag_params();mask)
        @test only(d.passes).contributions.masked_window_pairs>0
        _,d=ensemble_diag_observe([(A,B)],PIVParameters(window_size=16,search_area_size=24,overlap=8,uncertainty=true,uod_enable=false);backend=:cpu)
        @test only(d.passes).grid.search_area_size==(24,24)
        # Actual component populations are separate; zero is a finite estimate.
        summary=Hammerhead._ensemble_component_summary([0. -1.;NaN 2.],falses(2,2))
        @test summary==(finite_nonnegative_count=2,finite_negative_count=1,nonfinite_count=1)
        for (plane,category) in ((zeros(2,2),1),(ones(2,2),2),([0. 1.;0. 2.],3),([0. NaN;0. 1.],4))
            @test Hammerhead._ensemble_plane_category(plane)==category
        end
        # Direct analysis proves residual capture precedes large-predictor rounding.
        p=ensemble_diag_params();grid=Hammerhead.pass_grid(Float64,(16,16),p,nothing,.5)
        o=Hammerhead._ensemble_observation(grid,Float64,1)
        plane=[exp(-((i-9.3)^2+(j-9.25)^2)/5) for i in 1:32,j in 1:32]
        engine=only(Hammerhead.piv_correlation_engines(Hammerhead._resolve_backend(:cpu),nothing,p,Float64,1))
        u=fill(1e16,1,1);v=fill(1e16,1,1);ratio=zeros(1,1);moment=zeros(1,1);uu=fill(NaN,1,1);uv=copy(uu)
        Hammerhead.ensemble_analyze!([plane],engine,u,v,ratio,moment,uu,uv,nothing,nothing,nothing,grid.jobs,p,nothing;diagnostics=o)
        @test isfinite(o.residual_u[1]) && o.residual_u[1]!=u[1]-1e16
        # More than one KA tile: stale padded slices cannot contribute counts.
        a=rand(MersenneTwister(718),Float32,200,200)
        _,d=ensemble_diag_observe([(a,circshift(a,(1,1)))],p;backend=:ka,image_type=Float32)
        @test only(d.passes).contributions.accumulated_window_pairs==only(d.passes).contributor_nodes.eligible_count>Hammerhead._KA_BATCH
    end

    @testset "Schedule/selection snapshots precede preprocessor callbacks" begin
        schedule=[ensemble_diag_params(),ensemble_diag_params()]
        mutation=Ref(false)
        pre=image->begin
            if !mutation[]
                mutation[]=true;empty!(schedule)
            end
            image
        end
        expected=run_piv_ensemble([(A,B)],[ensemble_diag_params(),ensemble_diag_params()];progress=false,threaded=false)
        result,d=ensemble_diag_observe([(A,B)],schedule;preprocess=pre)
        @test isempty(schedule) && length(d.passes)==2 && ensemble_diag_same(result,expected)
        selection=[(A,B)];mask=falses(48,48)
        pre=image->begin
            selection[1]=(zeros(48,48),zeros(48,48));mask[:].=true
            image
        end
        expected=run_piv_ensemble([(A,B)];effort=:low,window_size=16,progress=false,threaded=false)
        d=Ref{Any}()
        result=run_piv_ensemble(selection;effort=:low,window_size=16,mask,preprocess=pre,progress=false,threaded=false,on_diagnostics=x->(d[]=x))
        @test all(mask) && all(iszero,selection[1][1]) && ensemble_diag_same(result,expected)
        @test first(d[].passes).contributor_nodes.masked_count==0
    end

    @testset "Preflight, callback checks and native companions" begin
        mktempdir() do dir
            path=joinpath(dir,"result.jld2");calls=Ref(0)
            source=FrameSource(2,i->(calls[]+=1;isodd(i) ? A : B))
            pairs=[(FrameRef(source,1),FrameRef(source,2))]
            for opts in ((;record_diagnostics=true),(;record_diagnostics=1,output=path),(;on_diagnostics=1),
                         (;output=(_,_) -> path),(;backend=:cuda,on_diagnostics=x->nothing),
                         (;backend=:amdgpu,on_diagnostics=x->nothing),(;image_type=Float16,on_diagnostics=x->nothing))
                @test_throws ArgumentError run_piv_ensemble(pairs;effort=:low,progress=false,opts...)
                @test calls[]==0 && !ispath(path)
            end
            write(path,"sentinel")
            @test_throws ErrorException run_piv_ensemble([(A,B)],ensemble_diag_params();progress=false,output=path,on_diagnostics=d->error("stop"))
            @test read(path,String)=="sentinel"
            held=Ref{Any}();p=ensemble_diag_params(validation=(EnsembleCaptureValidator(held),))
            @test_throws ArgumentError run_piv_ensemble([(A,B)],p;progress=false,output=path,on_diagnostics=d->(held[].u[1]+=1))
            @test read(path,String)=="sentinel"
            raw,d=ensemble_diag_observe([(A,B),(A,B)],ensemble_diag_params();output=path,record_diagnostics=true)
            @test ensemble_diag_same(only(load_results(path)),raw)
            @test load_execution_diagnostics(path)===nothing
            loaded=load_ensemble_execution_diagnostics(path)
            @test loaded.execution_id==d.execution_id && execution_diagnostics_data(loaded)["verification"]["inspection_state"]=="metadata_only"
            @test execution_diagnostics_data(load_ensemble_execution_diagnostics(path;verify_result=true))["verification"]["measurement_field_binding_checked"]
            @test execution_diagnostics_data(loaded;result=raw)["verification"]["inspection_state"]=="supplied_measurement_fields_verified"
            @test execution_diagnostics_data(loaded)["verification"]["inspection_state"]=="metadata_only"
            saved=read(path);bad=joinpath(dir,"bad.jld2")
            for change in (e->e["result_key"]="results/000002",e->e["diagnostics"]["pair_count"]=true,
                           e->e["diagnostics"]["passes"][1]["executed_pooling_sweeps"]=2,
                           e->e["diagnostics"]["passes"][1]["source_support"]["evaluated_pairs"]=2,
                           e->e["diagnostics"]["passes"][1]["contributor_nodes"]["maximum_finite_nonzero_contributions"]=0,
                           e->e["diagnostics"]["passes"][1]["uncertainty"]["evaluated"]=0,
                           e->e["diagnostics"]["passes"][1]["residual"]["masked_count"]=typemax(Int),
                           e->e["diagnostics"]["passes"][1]["grid"]["x"]["count"]=1)
                write(bad,saved);ensemble_diag_rewrite(bad,change)
                @test_throws ArgumentError load_ensemble_execution_diagnostics(bad)
            end
            write(bad,saved);ensemble_diag_rewrite(bad,e->e["diagnostics"]["passes"][1]["requested_iterations"]=99;rehash=false)
            @test_throws ArgumentError load_ensemble_execution_diagnostics(bad)
            write(bad,saved);changed=deepcopy(raw);changed.u[1]+=1
            ensemble_diag_replace(bad,Hammerhead.result_key(1),changed)
            @test load_ensemble_execution_diagnostics(bad)!==nothing
            @test_throws ArgumentError load_ensemble_execution_diagnostics(bad;verify_result=true)
            write(bad,saved);changed=deepcopy(raw);changed.x[1]+=1
            ensemble_diag_replace(bad,Hammerhead.result_key(1),changed)
            ensemble_diag_rewrite(bad,e->e["diagnostics"]["measurement_sha256"]=Hammerhead._history_result_digest(changed))
            @test load_ensemble_execution_diagnostics(bad)!==nothing
            @test_throws ArgumentError load_ensemble_execution_diagnostics(bad;verify_result=true)
            for marker in (true,0,2)
                write(bad,saved);ensemble_diag_replace(bad,"ensemble_execution_diagnostics_format_version",marker)
                @test_throws ArgumentError load_ensemble_execution_diagnostics(bad)
            end
            write(bad,saved)
            jldopen(f->delete!(f,"ensemble_execution_diagnostics_format_version"),bad,"r+")
            @test_throws ArgumentError load_ensemble_execution_diagnostics(bad)
            bare=joinpath(dir,"bare.jld2");save_results(bare,raw)
            @test load_ensemble_execution_diagnostics(bare)===nothing
            ensemble_diag_observe([(A,B)],ensemble_diag_params();output=bare)
            @test load_ensemble_execution_diagnostics(bare)===nothing
            # Native file-path alias refusal occurs before trying to decode a sentinel.
            input=joinpath(dir,"input.tif");write(input,"must not decode or truncate")
            @test_throws ArgumentError run_piv_ensemble([(input,input)];effort=:low,progress=false,output=input,record_diagnostics=true)
            @test read(input,String)=="must not decode or truncate"
            if Sys.iswindows()
                @test_throws ArgumentError run_piv_ensemble([(input,input)];effort=:low,progress=false,output=uppercase(input),record_diagnostics=true)
            end
        end
    end

    @testset "Packets do not retain frames or per-pair traces" begin
        refs=WeakRef[]
        source=FrameSource(8,i->begin
            image=rand(MersenneTwister(isodd(i) ? 93 : 94),32,32);push!(refs,WeakRef(image));image
        end)
        pairs=[(FrameRef(source,i),FrameRef(source,i+1)) for i in 1:2:7]
        r,d=ensemble_diag_observe(pairs,ensemble_diag_params())
        GC.gc(true);GC.gc(true)
        @test all(ref->ref.value===nothing,refs[1:end-2])
        @test ensemble_diag_no_arrays(d)
        _,small=ensemble_diag_observe([(A,B)],ensemble_diag_params())
        _,large=ensemble_diag_observe(fill((A,B),20),ensemble_diag_params())
        @test Base.summarysize(large)<=Base.summarysize(small)+256
    end
end
