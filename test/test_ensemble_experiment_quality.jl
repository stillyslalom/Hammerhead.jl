using Test, Hammerhead, Random, JLD2, TOML, UUIDs
using FileIO: save
using ImageCore: Gray, N0f8

# A detached schema fixture checks serialization/validation, not execution
# provenance. Replay and association integration fixtures follow below.
function eeq_schema_fixture(dir)
    a=rand(MersenneTwister(6017),32,32)
    p=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,
        validation=(),replace_outliers=false)
    r=run_piv(a,circshift(a,(1,2)),p;threaded=false)
    path=joinpath(dir,"schema-source.jld2");save_results(path,[r])
    data=quality_report_data(quality_report(ResultFile(path)))
    data["quality_report_format_version"]=5
    data["entry_kinds"]=Dict{String,Any}("planar"=>1,"stereo"=>0,"ptv"=>0,"tracking"=>0)
    data["unavailable"]=Hammerhead._quality_unavailable(4;include_history=false)
    merge!(data["provenance"],Dict{String,Any}(
        "association"=>"recorded_ensemble_output_verified","workflow"=>"planar_ensemble",
        "verification_time"=>"report_generation","recipe_id"=>repeat("a",64),
        "input_id"=>repeat("b",64),"run_id"=>string(uuid4()),"run_environment_id"=>repeat("c",64),
        "input_pairs"=>2,"scheduled_passes"=>2,"total_contributions"=>4,"completed_contributions"=>4,
        "completed_pools"=>1,"published_results"=>1,"record_diagnostics"=>false,
        "input_bytes_checked"=>false,"raw_measurement_fields_checked"=>true,
        "recipe_grid_mask_scale_checked"=>true,"requested_companions_checked"=>true))
    data,path
end

function eeq_fixture(dir)
    rng=MersenneTwister(7719)
    paths=String[]
    for (i,a) in enumerate((rand(rng,32,32),rand(rng,32,32)))
        for (j,img) in enumerate((a,circshift(a,(1,2))))
            path=joinpath(dir,"pair$i-$j.png")
            save(path,Gray{N0f8}.(img));push!(paths,path)
        end
    end
    p=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,
        validation=(),replace_outliers=false,uncertainty=true)
    mask=falses(32,32);mask[1:8,1:8].=true
    recipe=EnsemblePIVRecipe([p,p];mask,threaded=false,
        preprocessing=[PreprocessStep(:invert_image)],
        scale=PhysicalScale(pixel_size=.02,dt=.01,length_unit="mm",time_unit="s"))
    record=EnsembleExperimentRecord([(paths[1],paths[2]),(paths[3],paths[4])],recipe)
    (;record,paths)
end

eeq_run(run;kwargs...)=EnsembleExperimentRun((get(kwargs,k,deepcopy(getfield(run,k))) for k in fieldnames(EnsembleExperimentRun))...)
function eeq_replace(path,key,value)
    jldopen(path,"r+") do f
        haskey(f,key) && delete!(f,key)
        f[key]=value
    end
end

@testset "Verified saved ensemble quality reports" begin
    mktempdir() do dir
        fixture=eeq_fixture(dir);record=fixture.record
        record_path=joinpath(dir,"ensemble-record.jld2");save_experiment(record_path,record)
        for recorded in (false,true)
            @testset "diagnostics policy $recorded" begin
                output=joinpath(dir,"output-$recorded.jld2")
                run=replay_experiment(record;output,run_record=record_path,record_diagnostics=recorded)
                saved=load_ensemble_experiment(record_path)
                for pooled in (false,true)
                    report=quality_report(saved,run;include_ensemble_execution_diagnostics=pooled,verify_inputs=true)
                    d=quality_report_data(report);p=d["provenance"]
                    @test d["quality_report_format_version"]===5
                    @test p["association"]=="recorded_ensemble_output_verified"
                    @test p["recipe_id"]==recipe_identity(record.recipe) && p["input_id"]==record.input_id
                    @test p["run_id"]==run.run_id && p["source_sha256"]==run.output_sha256
                    @test p["input_pairs"]==p["scheduled_passes"]==2
                    @test p["total_contributions"]==p["completed_contributions"]==4
                    @test p["completed_pools"]==p["published_results"]==p["source_index_entries"]==1
                    @test p["record_diagnostics"]===recorded && p["input_bytes_checked"]===true
                    @test !haskey(p,"completed_pairs")
                    @test d["groups"]==quality_report_data(quality_report(ResultFile(output)))["groups"]
                    @test haskey(d,"ensemble_execution_diagnostics")===pooled
                    if pooled
                        section=d["ensemble_execution_diagnostics"]
                        @test section==quality_report_data(quality_report(ResultFile(output);include_ensemble_execution_diagnostics=true))["ensemble_execution_diagnostics"]
                        @test section["classification"]["recorded_ensemble_entries"]==Int(recorded)
                        @test section["classification"]["entries_without_execution_metadata"]==Int(!recorded)
                    end
                    @test all(path->path in d["protected_locators"],[fixture.paths;record_path;output])
                    target=joinpath(dir,"report-$recorded-$pooled.toml")
                    save_quality_report(target,report)
                    @test quality_report_data(load_quality_report(target))==d
                    for protected in [fixture.paths;record_path;output]
                        bytes=read(protected)
                        @test_throws ArgumentError save_quality_report(protected,report)
                        @test read(protected)==bytes
                    end
                end
                @test_throws ArgumentError quality_report(saved,run;include_measurement_history=true)
                @test_throws ArgumentError quality_report(saved,run;include_execution_diagnostics=true)
                for status in (:failed,:cancelled)
                    stopped=eeq_run(run;status,completed_pools=0,published_results=0,
                        output_sha256=nothing,measurement_sha256=nothing,error="stopped")
                    @test_throws ArgumentError quality_report(saved,stopped)
                end
                for altered in (eeq_run(run;run_id=string(uuid4())),eeq_run(run;output_sha256=nothing),
                        eeq_run(run;input_pairs=3),eeq_run(run;completed_contributions=3))
                    @test_throws ArgumentError quality_report(saved,altered)
                end
                moved=joinpath(dir,"moved-$recorded.jld2");Base.cp(output,moved)
                foreign=Sys.iswindows() ? "/historical/ensemble.jld2" : "Z:\\historical\\ensemble.jld2"
                relocated=eeq_run(run;output=foreign)
                d=quality_report_data(quality_report(saved,relocated;output=moved))
                @test d["provenance"]["source_path"]==moved
                @test !(foreign in d["protected_locators"])
                @test !(abspath(foreign) in d["protected_locators"])
                @test moved in d["protected_locators"] && output in d["protected_locators"]
                @test_throws ArgumentError quality_report(saved,relocated)
                old=read(fixture.paths[1]);write(fixture.paths[1],"changed input")
                @test quality_report_data(quality_report(saved,run))["provenance"]["input_bytes_checked"]===false
                @test_throws ArgumentError quality_report(saved,run;verify_inputs=true)
                write(fixture.paths[1],old)

                # Reseal the outer file hash: raw association still must catch
                # changed arrays, final settings, crossed packets and sources.
                original=read(output);raw=ResultFile(output)[1]
                changed=deepcopy(raw)
                finite_index=findfirst(isfinite,changed.u)
                @test finite_index!==nothing
                changed.u[finite_index]+=17
                eeq_replace(output,Hammerhead.result_key(1),changed)
                resealed=eeq_run(run;output_sha256=Hammerhead._experiment_file_digest(output))
                @test_throws ArgumentError quality_report(saved,resealed;include_ensemble_execution_diagnostics=false)
                write(output,original)
                settings=(;(k=>getfield(raw.parameters,k) for k in fieldnames(PIVParameters))...)
                parameters=PIVParameters(;merge(settings,(max_iterations=raw.parameters.max_iterations+1,))...)
                altered_parameters=PIVResult((k===:parameters ? parameters : getfield(raw,k) for k in fieldnames(typeof(raw)))...)
                eeq_replace(output,Hammerhead.result_key(1),altered_parameters)
                @test_throws ArgumentError quality_report(saved,eeq_run(run;output_sha256=Hammerhead._experiment_file_digest(output)))
                write(output,original)
                association=jldopen(f->deepcopy(f["ensemble_experiment_run"]),output,"r")
                association["association"]["input_pairs"]+=1
                association["association_sha256"]=Hammerhead._experiment_digest(association["association"])
                eeq_replace(output,"ensemble_experiment_run",association)
                @test_throws ArgumentError quality_report(saved,eeq_run(run;output_sha256=Hammerhead._experiment_file_digest(output)))
                write(output,original)
                for (key,value) in (("ensemble_sources",[fixture.paths[2],fixture.paths[1],fixture.paths[3],fixture.paths[4]]),
                        ("measurement_history_format_version",1),("ensemble_experiment_run_format_version",2))
                    eeq_replace(output,key,value)
                    resealed=eeq_run(run;output_sha256=Hammerhead._experiment_file_digest(output))
                    @test_throws ArgumentError quality_report(saved,resealed;include_ensemble_execution_diagnostics=false)
                    write(output,original)
                end
                eeq_replace(output,"results/000002",raw)
                @test_throws ArgumentError quality_report(saved,eeq_run(run;output_sha256=Hammerhead._experiment_file_digest(output)))
                write(output,original)
                altered=deepcopy(saved);altered.recipe.mask[1,1]=!altered.recipe.mask[1,1]
                @test_throws ArgumentError quality_report(altered,run)
                record=saved # retain prior history when appending the next policy run
            end
        end
    end
end

@testset "Associated ensemble report schema" begin
    mktempdir() do dir
        data,source=eeq_schema_fixture(dir)
        report=Hammerhead._quality_wrap(data)
        raw=ResultFile(source)[1]
        stereo=StereoPIVResult(raw.x,raw.y,0.,raw.u,raw.v,raw.u,raw.uncertainty_u,
            raw.uncertainty_v,raw.uncertainty_u,raw.outliers,raw.mask,raw,raw,raw.parameters)
        crossed=deepcopy(data)
        crossed["groups"]=quality_report_data(quality_report([stereo]))["groups"]
        @test_throws ArgumentError Hammerhead._quality_wrap(crossed)
        output=joinpath(dir,"report.toml");save_quality_report(output,report)
        @test quality_report_data(load_quality_report(output))==data
        text=sprint(show,MIME"text/plain"(),report)
        @test occursin("verified recorded ensemble output",text)
        @test occursin("2 ordered input pairs",text)
        @test occursin("4 / 4 processed pair contributions",text)
        @test occursin("1 published pooled result",text)
        @test !haskey(data["provenance"],"completed_pairs")
        for (section,key,value) in (("entry_kinds","planar",true),("entry_kinds","planar",1.0),
                ("entry_kinds","stereo",false),("provenance","input_pairs",true),
                ("provenance","completed_pools",2),("provenance","published_results",0),
                ("provenance","completed_contributions",3),("provenance","record_diagnostics",0),
                ("provenance","raw_measurement_fields_checked",1))
            altered=deepcopy(data);altered[section][key]=value
            malformed=joinpath(dir,"malformed.toml")
            open(io->TOML.print(io,altered),malformed,"w")
            @test_throws ArgumentError load_quality_report(malformed)
        end
        overflow=deepcopy(data);overflow["provenance"]["input_pairs"]=typemax(Int)
        @test_throws ArgumentError Hammerhead._quality_wrap(overflow)
        for value in (0,0.0)
            altered=deepcopy(data);altered["unavailable"]["accuracy"]["available"]=value
            malformed=joinpath(dir,"unavailable-type.toml")
            open(io->TOML.print(io,altered),malformed,"w")
            @test_throws ArgumentError load_quality_report(malformed)
        end
        changed=deepcopy(data);changed["provenance"]["completed_pairs"]=2
        @test_throws ArgumentError Hammerhead._quality_wrap(changed)
        bytes=read(source)
        @test_throws ArgumentError save_quality_report(source,report)
        @test read(source)==bytes
        rm(source)
        @test quality_report_data(load_quality_report(output))==data # past verification only
    end
end
