using Test, Hammerhead, Random, JLD2, UUIDs
using FileIO: save
using ImageCore: Gray, N0f8

function srq_fixture(dir)
    rng = MersenneTwister(8042)
    cameras = (PinholeCamera([100. 0 15. 0; 0 100. 0 0; 0 0 1. 100.]),
               PinholeCamera([100. 0 -15. 0; 0 100. 0 0; 0 0 1. 100.]))
    grid = DewarpGrid(x=1.:32., y=32.:-1.:1.)
    dewarpers = map(c -> ImageDewarper(c, grid, (36,36)), cameras)
    files = [joinpath(dir,"camera$camera-frame$frame.png") for camera in 1:2, frame in 1:4]
    for camera in 1:2
        a = Gray{N0f8}.(rand(rng,36,36))
        for (frame,img) in enumerate((a,circshift(a,(1,1)),a,circshift(a,(1,1))))
            save(files[camera,frame],img)
        end
    end
    pairs = [(files[1,i],files[1,i+1],files[2,i],files[2,i+1]) for i in (1,3)]
    parameters = PIVParameters(window_size=16,overlap=8,padding=true,
        uod_enable=false,validation=(),replace_outliers=false)
    recipe = StereoPIVRecipe(parameters,dewarpers...;threaded=false,
        scale=PhysicalScale(pixel_size=.02,dt=.01,length_unit="mm",time_unit="s"))
    record = StereoExperimentRecord(pairs,recipe;
        timestamps=([0,1,2,3],[0,1,2,3]),time_units=("s","s"))
    (;record,files)
end

function srq_run(run; output=run.output, hash=run.output_sha256,
        status=run.status, count=run.completed_pairs, error=run.error, id=run.run_id)
    ExperimentRun(id,run.recipe_id,run.input_id,run.started_at,run.finished_at,
        status,count,output,hash,deepcopy(run.environment),error)
end

function srq_replace(path,key,value)
    jldopen(path,"r+") do file
        haskey(file,key) && delete!(file,key)
        file[key]=value
    end
end

@testset "Verified stereo run quality reports" begin
    mktempdir() do dir
        f=srq_fixture(dir)
        record_path=joinpath(dir,"experiment.jld2")
        output=joinpath(dir,"vectors.jld2")
        save_experiment(record_path,f.record)
        run=replay_experiment(f.record;output,run_record=record_path,
            record_diagnostics=true,record_pair_timing=true)
        record=load_stereo_experiment(record_path)
        @test length(record.runs)==1

        @testset "Existing report schemas, separate camera observations and identity" begin
            report=quality_report(record,run)
            data=quality_report_data(report)
            @test data["quality_report_format_version"]===1
            @test data["groups"]==quality_report_data(quality_report(ResultFile(output)))["groups"]
            @test Set(keys(data["groups"]))==Set(["stereo"])
            provenance=data["provenance"]
            @test provenance["association"]=="recorded_output_verified"
            @test provenance["run_id"]==run.run_id
            @test provenance["recipe_id"]==recipe_identity(record.recipe)
            @test provenance["input_id"]==record.input_id
            @test provenance["completed_pairs"]==provenance["source_index_entries"]==2
            @test provenance["source_sha256"]==run.output_sha256
            @test !haskey(data,"execution_diagnostics")
            @test data["unavailable"]["accuracy"]["available"]===false
            @test data["unavailable"]["uncertainty_coverage"]["available"]===false
            @test all(path -> path in data["protected_locators"], [record_path,output,vec(f.files)...])
            data["groups"]["stereo"]["counts"]["entries"]=12345
            @test quality_report_data(report)["groups"]["stereo"]["counts"]["entries"]==2

            execution=quality_report(record,run;include_execution_diagnostics=true,verify_inputs=true)
            edata=quality_report_data(execution)
            @test edata["quality_report_format_version"]===3
            @test !haskey(edata,"measurement_history")
            @test edata["groups"]==quality_report_data(report)["groups"]
            @test edata["execution_diagnostics"]==quality_report_data(
                quality_report(ResultFile(output);include_execution_diagnostics=true))["execution_diagnostics"]
            for role in ("cam1","cam2")
                group=edata["execution_diagnostics"]["groups"][role]
                @test group["counts"]["recorded_entries"]==2
                @test group["counts"]["missing_entries"]==0
                @test group["binding"]=="raw_measurement_fields_checked_at_report_generation"
            end
            @test_throws ArgumentError quality_report(record,run;include_measurement_history=true)
            @test_throws ArgumentError quality_report(record,run;
                include_measurement_history=true,include_execution_diagnostics=true)
            report_path=joinpath(dir,"quality.toml")
            save_quality_report(report_path,execution)
            @test quality_report_data(load_quality_report(report_path))==edata
            for path in (record_path,output,f.files[1,1])
                before=read(path)
                @test_throws ArgumentError save_quality_report(path,execution)
                @test read(path)==before
            end
            # A saved report is a detached historical snapshot, not a live verifier.
            original=read(f.files[1,1])
            write(f.files[1,1],"changed input bytes")
            @test quality_report_data(quality_report(record,run))["groups"]==edata["groups"]
            @test_throws ArgumentError quality_report(record,run;verify_inputs=true)
            @test quality_report_data(load_quality_report(report_path))==edata
            write(f.files[1,1],original)
        end

        @testset "Relocated output and failed or inconsistent associations" begin
            relocated=joinpath(dir,"relocated.jld2")
            cp(output,relocated)
            foreign=Sys.iswindows() ? "/historical/stereo-results.jld2" : "Z:\\historical\\stereo-results.jld2"
            moved=srq_run(run;output=foreign)
            data=quality_report_data(quality_report(record,moved;output=relocated))
            @test data["provenance"]["source_path"]==relocated
            @test data["provenance"]["run_id"]==run.run_id
            @test !(foreign in data["protected_locators"])
            @test !(abspath(foreign) in data["protected_locators"])
            @test relocated in data["protected_locators"] && output in data["protected_locators"]
            @test moved.output==foreign
            @test_throws ArgumentError quality_report(record,moved)
            @test_throws ArgumentError quality_report(record,srq_run(run;status=:failed,count=1,error="cancelled"))
            @test_throws ArgumentError quality_report(record,srq_run(run;id=string(uuid4())))
            @test_throws ArgumentError quality_report(record,srq_run(run;hash=nothing))
            edited=deepcopy(record)
            edited.recipe._data["calibration_provenance"]["note"]="changed after capture"
            @test_throws ArgumentError quality_report(edited,run)
        end

        @testset "Resealed output still requires raw fields and ordered sources" begin
            bytes=read(output)
            original=ResultFile(output)[1]
            changed=deepcopy(original)
            changed.u[1,1]=321.0
            srq_replace(output,Hammerhead.result_key(1),changed)
            resealed=srq_run(run;hash=Hammerhead._experiment_file_digest(output))
            @test_throws ArgumentError quality_report(record,resealed)
            @test_throws ArgumentError quality_report(record,resealed;include_execution_diagnostics=true)
            write(output,bytes)
            srq_replace(output,Hammerhead.source_key(1),reverse(String[record.input_files[j]["path"] for j in record.pairs[1]]))
            @test_throws ArgumentError quality_report(record,srq_run(run;hash=Hammerhead._experiment_file_digest(output)))
            write(output,bytes)
            for key in ("ensemble_execution_diagnostics_format_version","ensemble_execution_diagnostics")
                srq_replace(output,key,1)
                @test_throws ArgumentError quality_report(record,srq_run(run;hash=Hammerhead._experiment_file_digest(output)))
                write(output,bytes)
            end
            open(io -> write(io,UInt8[1,2,3]),output,"a")
            @test_throws ArgumentError quality_report(record,run)
            write(output,bytes)
        end

        @testset "Unrecorded execution is a coverage gap" begin
            plain=joinpath(dir,"plain.jld2")
            plain_run=replay_experiment(record;output=plain)
            data=quality_report_data(quality_report(record,plain_run;include_execution_diagnostics=true))
            for role in ("cam1","cam2")
                counts=data["execution_diagnostics"]["groups"][role]["counts"]
                @test counts["eligible_entries"]==2
                @test counts["recorded_entries"]==0
                @test counts["missing_entries"]==2
                @test counts["executed_sweeps"]==0
            end
        end
    end
end

@testset "Execution reports refuse distinct ensemble metadata" begin
    mktempdir() do dir
        for key in ("ensemble_execution_diagnostics_format_version","ensemble_execution_diagnostics")
            path=joinpath(dir,key*".jld2")
            save_results(path,PIVResult[])
            srq_replace(path,key,1)
            @test isempty(quality_report_data(quality_report(ResultFile(path)))["groups"])
            @test_throws ArgumentError quality_report(ResultFile(path);include_execution_diagnostics=true)
        end
    end
end
