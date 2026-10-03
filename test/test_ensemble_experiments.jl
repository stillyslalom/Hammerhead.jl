using Test, Hammerhead, Random, JLD2
using FileIO: save
using ImageCore: Gray, N0f8

function een_fixture(dir)
    a=Gray{N0f8}.(rand(MersenneTwister(12091),48,48));b=circshift(a,(1,2))
    paths=[joinpath(dir,"frame$i.png") for i in 1:4]
    for (path,image) in zip(paths,(a,b,a,b));save(path,image);end
    (;paths,pairs=[(paths[1],paths[2]),(paths[3],paths[4])])
end
een_passes()=multipass_parameters([24,16];padding=true,uod_enable=false,validation=(),replace_outliers=false,
    max_iterations=3,convergence_tol=Inf)
een_same(a,b)=all(k->isequal(getfield(a,k),getfield(b,k)),
    (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,:mask,:outliers,:scale))
function een_run(run;kwargs...)
    data=Dict{Symbol,Any}(k=>getfield(run,k) for k in fieldnames(EnsembleExperimentRun));merge!(data,Dict(kwargs))
    EnsembleExperimentRun((data[k] for k in fieldnames(EnsembleExperimentRun))...)
end
function een_rewrite(path,key,value)
    jldopen(path,"r+") do f
        haskey(f,key) && delete!(f,key);f[key]=value
    end
end
een_rehash(run,path)=een_run(run;output_sha256=Hammerhead._experiment_file_digest(path))
function een_record_payload(record)
    data=deepcopy(Hammerhead._ensemble_record_core(record));data["runs"]=Any[];data["record_sha256"]=record._sha256;data
end
function een_write_record(path,data;version=1)
    jldopen(path,"w") do f
        f["ensemble_experiment_format_version"]=version;f["ensemble_experiment"]=data
    end
end
function een_record_environment(record,environment)
    provisional=EnsembleExperimentRecord(record.recipe,record.input_files,record.pairs,record.input_id,
        environment,record.runs,record.record_paths,"")
    EnsembleExperimentRecord(provisional.recipe,provisional.input_files,provisional.pairs,provisional.input_id,
        environment,provisional.runs,provisional.record_paths,Hammerhead._experiment_digest(Hammerhead._ensemble_record_core(provisional)))
end

@testset "Ensemble recipe snapshots and dedicated schema" begin
    mktempdir() do dir
        fixture=een_fixture(dir);passes=een_passes();mask=falses(48,48);bg=zeros(48,48)
        steps=[PreprocessStep(:subtract_background;background=bg)]
        recipe=EnsemblePIVRecipe(passes;mask,preprocessing=steps,scale=PhysicalScale(.2,.5,"mm","s"))
        id=recipe_identity(recipe);mask[1]=true;bg[1]=1.;passes[1]=PIVParameters()
        @test recipe_identity(recipe)==id
        @test !recipe.mask[1] && recipe.preprocessing[1].options["background"][1]==0
        @test recipe.passes[1].max_iterations==3 && isinf(recipe.passes[1].convergence_tol)
        record=EnsembleExperimentRecord(fixture.pairs,recipe)
        path=joinpath(dir,"recipe.jld2");save_experiment(path,record)
        loaded=load_ensemble_experiment(path)
        @test loaded.recipe.recipe_id==id && loaded.input_id==record.input_id && loaded._sha256==record._sha256
        @test loaded.record_paths==[realpath(path)] && isempty(loaded.runs)
        @test_throws ArgumentError load_experiment(path)
        @test_throws ArgumentError load_results(path)
        copies=[joinpath(dir,"copy$i.png") for i in 1:4]
        for (a,b) in zip(fixture.paths,copies);Base.cp(a,b);end
        equal_record=EnsembleExperimentRecord([(copies[1],copies[2]),(copies[3],copies[4])],recipe)
        @test equal_record.input_id==record.input_id && equal_record._sha256!=record._sha256
        @test EnsembleExperimentRecord(reverse(fixture.pairs),recipe).input_id==record.input_id # equal-byte pairs; order still bound in full snapshot
        @test EnsembleExperimentRecord([(fixture.paths[2],fixture.paths[1])],recipe).input_id!=
            EnsembleExperimentRecord([first(fixture.pairs)],recipe).input_id
        for kwargs in ((;roi=(1:48,1:48)),(;external_preprocess="script.jl"),(;backend=:cuda),
                       (;uncertainty_backend=:cpu),(;image_type=Float16))
            @test_throws ArgumentError EnsemblePIVRecipe(een_passes();kwargs...)
        end
        @test_throws ArgumentError EnsembleExperimentRecord([[fixture.paths[1],fixture.paths[2]]],recipe)
        @test_throws ArgumentError EnsembleExperimentRecord(fixture.pairs,EnsemblePIVRecipe(een_passes();mask=falses(4,4)))
        payload=een_record_payload(record)
        for (label,mutate) in (("policy",d->(d["recipe"]["threaded"]=1)),
                              ("pass",d->(d["recipe"]["passes"][1]["max_iterations"]=true)),
                              ("pair",d->(d["pairs"][1][1]=0)),
                              ("hash",d->(d["record_sha256"]=repeat("a",64))))
            bad=deepcopy(payload);mutate(bad);dest=joinpath(dir,"bad-$label.jld2");een_write_record(dest,bad)
            @test_throws ArgumentError load_ensemble_experiment(dest)
        end
        een_write_record(joinpath(dir,"bad-version.jld2"),payload;version=2)
        @test_throws ArgumentError load_ensemble_experiment(joinpath(dir,"bad-version.jld2"))
        bytes=read(fixture.paths[1]);@test_throws ArgumentError save_experiment(fixture.paths[1],record)
        @test read(fixture.paths[1])==bytes
        changed=deepcopy(recipe);changed.mask[2]=true
        @test_throws ArgumentError recipe_identity(changed)
    end
end

@testset "Saved ensemble exact direct parity and contribution progress" begin
    mktempdir() do dir
        fixture=een_fixture(dir)
        for backend in (:cpu,:ka), T in (Float32,Float64)
            mask=falses(48,48);mask[1:12,1:12].=true
            recipe=EnsemblePIVRecipe(een_passes();backend,image_type=T,mask,scale=PhysicalScale(.2,.5,"mm","s"),
                preprocessing=[PreprocessStep(:subtract_background;background=zeros(48,48))])
            record=EnsembleExperimentRecord(fixture.pairs,recipe)
            direct=run_piv_ensemble(fixture.pairs,recipe.passes;backend,image_type=T,mask,
                scale=recipe.scale,preprocess=Hammerhead._experiment_preprocess(recipe,nothing),threaded=false,progress=false)
            events=Any[];packet=Ref{Any}(nothing);output=joinpath(dir,"$backend-$T.jld2");history=joinpath(dir,"$backend-$T-record.jld2")
            run=replay_experiment(record;output,run_record=history,record_diagnostics=true,on_diagnostics=d->(packet[]=d),progress=e->push!(events,e))
            @test run.status===:completed && run.completed_pools==run.published_results==1
            @test run.input_pairs==2 && run.scheduled_passes==2 && run.total_contributions==run.completed_contributions==4
            @test length(events)==4 && [e.completed_contributions for e in events]==1:4
            @test [(e.pass_index,e.pair_index) for e in events]==[(1,1),(1,2),(2,1),(2,2)]
            @test all(e->e.completed_pools==e.published_results==0 && e.total_contributions==4,events)
            raw=ResultFile(output)[1];@test een_same(direct,raw)
            checked=verify_ensemble_experiment_run(record,run;verify_results=true,verify_inputs=true)
            @test checked["measurement_fields_checked"] && checked["run_association_checked"] && checked["current_input_bytes_checked"]
            @test packet[].pair_count==2 && all(d->d.executed_pooling_sweeps==1 && d.requested_iterations==3 && isinf(d.requested_tolerance),packet[].passes)
            @test length(load_ensemble_experiment(history).runs)==1
            @test load_ensemble_experiment(history).runs[1].output_sha256==run.output_sha256
            @test_throws ArgumentError save_experiment(output,record;runs=[run])
        end
    end
end

@testset "Ensemble cancellation, failures, snapshots and publication boundaries" begin
    mktempdir() do dir
        fixture=een_fixture(dir);record=EnsembleExperimentRecord(fixture.pairs,EnsemblePIVRecipe(een_passes()))
        output=joinpath(dir,"result.jld2");history=joinpath(dir,"history.jld2");sentinel=UInt8[4,2,9];write(output,sentinel)
        for boundary in (0,1,4)
            token=Ref(boundary==0);events=Any[]
            run=replay_experiment(record;output,run_record=history,cancel_requested=()->token[],
                progress=e->begin push!(events,e);e.completed_contributions==boundary && (token[]=true);end)
            @test run.status===:cancelled && run.completed_contributions==boundary
            @test run.completed_pools==run.published_results==0 && run.output_sha256===run.measurement_sha256===nothing
            @test length(events)==boundary && read(output)==sentinel
            @test !verify_ensemble_experiment_run(record,run)["run_association_checked"]
            @test_throws ArgumentError verify_ensemble_experiment_run(record,run;verify_results=true)
            @test load_ensemble_experiment(history).runs[end].status===:cancelled
        end
        @test_throws ArgumentError replay_experiment(record;output,run_record=history,cancel_requested=()->1)
        @test read(output)==sentinel && load_ensemble_experiment(history).runs[end].status===:failed
        original=ErrorException("joined contribution observer failed")
        caught=try replay_experiment(record;output,run_record=history,progress=e->throw(original));nothing catch e;e end
        @test caught===original && read(output)==sentinel
        failed=load_ensemble_experiment(history).runs[end]
        @test failed.status===:failed && failed.completed_contributions==1 && failed.published_results==0
        # Internal hooks are validated before either explicit/effort first load.
        loads=Ref(0);source=FrameSource(2,i->begin loads[]+=1;zeros(48,48);end)
        pairs=[(FrameRef(source,1),FrameRef(source,2))]
        @test_throws ArgumentError run_piv_ensemble(pairs,first(een_passes());_on_contribution=1,progress=false)
        @test_throws ArgumentError run_piv_ensemble(pairs;effort=:low,_cancel_requested=1,progress=false)
        @test loads[]==0
        # Callers can edit their containers; captured settings/selection stay fixed.
        mutable_record=deepcopy(record);events=Any[]
        run=replay_experiment(mutable_record;output,progress=e->begin
            push!(events,e)
            if e.completed_contributions==1
                mutable_record.pairs[2][1]=2;mutable_record.recipe.passes[2]=PIVParameters(window_size=(24,24),overlap=(12,12))
                fixture.pairs[2]=(fixture.paths[2],fixture.paths[1])
            end
        end)
        @test run.status===:completed && length(events)==4
        @test verify_ensemble_experiment_run(record,run;verify_results=true)["measurement_fields_checked"]
        # A late cancellation predicate is a mutation boundary even when false.
        old=read(output);input_bytes=read(fixture.paths[1]);changed=Ref(false)
        callback=()->begin
            if !changed[] && any(endswith(".partial"),readdir(dir))
                open(fixture.paths[1],"a") do io;write(io,UInt8(0));end
                changed[]=true
            end
            false
        end
        @test_throws ArgumentError replay_experiment(record;output,cancel_requested=callback)
        @test changed[] && read(output)==old
        write(fixture.paths[1],input_bytes)
        @test !any(endswith(".partial"),readdir(dir))
        staged_mutation=Ref(false)
        mutate_staging=()->begin
            partials=filter(endswith(".partial"),readdir(dir;join=true))
            if !staged_mutation[] && !isempty(partials)
                open(first(partials),"a") do io;write(io,UInt8(0));end
                staged_mutation[]=true
            end
            false
        end
        @test_throws ArgumentError replay_experiment(record;output,cancel_requested=mutate_staging)
        @test staged_mutation[] && read(output)==old
        @test !any(endswith(".partial"),readdir(dir))
        # History failures carry completed/cancelled truth, never a failed pool.
        bad_history=joinpath(dir,"missing","history.jld2")
        caught=try replay_experiment(record;output,run_record=bad_history);nothing catch e;e end
        @test caught isa EnsembleRunRecordError && caught.run.status===:completed && caught.run.published_results==1
        @test occursin("pooled output completed",sprint(showerror,caught))
        @test verify_ensemble_experiment_run(record,caught.run;verify_results=true)["measurement_fields_checked"]
        caught=try replay_experiment(record;output,run_record=bad_history,cancel_requested=()->true);nothing catch e;e end
        @test caught isa EnsembleRunRecordError && caught.run.status===:cancelled && caught.run.published_results==0
        caught=try replay_experiment(record;output,run_record=bad_history,progress=e->throw(original));nothing catch e;e end
        @test caught===original
        save_experiment(history,record);record_bytes=read(history)
        @test_throws ArgumentError replay_experiment(record;output=history)
        @test read(history)==record_bytes
        @test_throws ArgumentError replay_experiment(record;output,run_record=output)
        @test_throws ArgumentError replay_experiment(record;output=fixture.paths[1])
    end
end

@testset "Ensemble association and raw corruption are independently checked" begin
    mktempdir() do dir
        fixture=een_fixture(dir);record=EnsembleExperimentRecord(fixture.pairs,EnsemblePIVRecipe(een_passes()))
        output=joinpath(dir,"pooled.jld2");run=replay_experiment(record;output,record_diagnostics=false)
        bytes=read(output)
        environment=deepcopy(run.environment);environment["julia_version"]="1.10.99"
        @test_throws ArgumentError verify_ensemble_experiment_run(record,een_run(run;environment))
        @test verify_ensemble_experiment_run(record,run)["output_integrity_checked"]
        previous=read(output);old_environment=deepcopy(record.creation_environment);old_environment["julia_version"]="0.0.0"
        other=een_record_environment(record,old_environment)
        @test_throws ArgumentError replay_experiment(other;output)
        @test read(output)==previous
        override_path=joinpath(dir,"explicit-environment-override.jld2")
        overridden=replay_experiment(other;output=override_path,allow_environment_change=true)
        @test overridden.environment["julia_version"]==string(VERSION)
        @test verify_ensemble_experiment_run(other,overridden;verify_results=true)["measurement_fields_checked"]
        for (label,mutate,raw_only) in (
                ("sources",p->een_rewrite(p,"ensemble_sources",reverse(String[record.input_files[i]["path"] for pair in record.pairs for i in pair])),false),
                ("raw",p->begin raw=ResultFile(p)[1];raw.u[1]+=1;een_rewrite(p,"results/000001",raw);end,true),
                ("parameters",p->begin raw=ResultFile(p)[1];new=PIVResult(raw.x,raw.y,raw.u,raw.v,raw.peak_ratio,raw.correlation_moment,raw.uncertainty_u,raw.uncertainty_v,
                    raw.outliers,raw.mask,PIVParameters(window_size=(24,24),overlap=(12,12)),raw.correlation_planes,raw.scale);een_rewrite(p,"results/000001",new);end,true),
                ("extra",p->een_rewrite(p,"results/000002",ResultFile(p)[1]),false),
                ("crossed",p->een_rewrite(p,"pair_timing_format_version",1),false),
                ("association",p->begin d=jldopen(f->f["ensemble_experiment_run"],p,"r");d["association"]["record_diagnostics"]=0;
                    d["association_sha256"]=Hammerhead._experiment_digest(d["association"]);een_rewrite(p,"ensemble_experiment_run",d);end,false))
            @testset "$label" begin
                write(output,bytes);mutate(output);forged=een_rehash(run,output)
                raw_only && @test verify_ensemble_experiment_run(record,forged)["output_integrity_checked"]
                @test_throws ArgumentError verify_ensemble_experiment_run(record,forged;verify_results=true)
            end
        end
        write(output,bytes);relocated=joinpath(dir,"relocated.jld2");Base.cp(output,relocated)
        @test verify_ensemble_experiment_run(record,run;output=relocated,verify_results=true)["measurement_fields_checked"]
        recording=replay_experiment(record;output,record_diagnostics=true)
        een_rewrite(output,"ensemble_execution_diagnostics_format_version",2)
        @test_throws ArgumentError verify_ensemble_experiment_run(record,een_rehash(recording,output))
        input=read(fixture.paths[1]);open(fixture.paths[1],"a") do io;write(io,UInt8(0));end
        @test_throws ArgumentError verify_ensemble_experiment_run(record,run;output=relocated,verify_inputs=true)
        @test verify_ensemble_experiment_run(record,run;output=relocated)["output_integrity_checked"]
        write(fixture.paths[1],input)
    end
end

@testset "Ensemble raw geometry, retained planes and detached lifetime" begin
    mktempdir() do dir
        fixture=een_fixture(dir)
        p=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,validation=(),
            replace_outliers=false,uncertainty=true,keep_correlation_planes=true)
        recipe=EnsemblePIVRecipe(p;threaded=true)
        record=EnsembleExperimentRecord(fixture.pairs,recipe);output=joinpath(dir,"planes.jld2")
        packet_refs=WeakRef[]
        run=replay_experiment(record;output,record_diagnostics=true,on_diagnostics=d->push!(packet_refs,WeakRef(d)))
        @test verify_ensemble_experiment_run(record,run;verify_results=true)["measurement_fields_checked"]
        raw=ResultFile(output)[1]
        serial=run_piv_ensemble(fixture.pairs,p;threaded=false,progress=false)
        @test een_same(raw,serial)
        @test raw.correlation_planes!==nothing && any(p->p!==nothing,raw.correlation_planes)
        original=read(output)
        shorter=PIVResult(raw.x[1:end-1],raw.y,raw.u,raw.v,raw.peak_ratio,raw.correlation_moment,
            raw.uncertainty_u,raw.uncertainty_v,raw.outliers,raw.mask,raw.parameters,raw.correlation_planes,raw.scale)
        een_rewrite(output,"results/000001",shorter)
        @test_throws ArgumentError verify_ensemble_experiment_run(record,een_rehash(run,output);verify_results=true)
        write(output,original)
        without_planes=PIVResult(raw.x,raw.y,raw.u,raw.v,raw.peak_ratio,raw.correlation_moment,
            raw.uncertainty_u,raw.uncertainty_v,raw.outliers,raw.mask,raw.parameters,nothing,raw.scale)
        een_rewrite(output,"results/000001",without_planes)
        @test_throws ArgumentError verify_ensemble_experiment_run(record,een_rehash(run,output);verify_results=true)
        GC.gc(true)
        @test all(r->r.value===nothing,packet_refs)
        @test all(k->!(getfield(run,k) isa AbstractArray || getfield(run,k) isa PIVResult),fieldnames(EnsembleExperimentRun))
    end
end
