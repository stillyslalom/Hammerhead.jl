using Test
using Hammerhead
using JLD2
using Random
using FileIO: save
using ImageCore: Gray, N0f8

function experiment_test_files(dir)
    rng=MersenneTwister(829)
    a=Gray{N0f8}.(rand(rng,48,48))
    b=circshift(a,(1,2))
    paths=[joinpath(dir,"frame-$i.png") for i in 1:4]
    for (path,image) in zip(paths,(a,b,a,b))
        save(path,image)
    end
    [(paths[1],paths[2]),(paths[3],paths[4])], paths
end

function experiment_same_result(a,b)
    all(k -> isequal(getfield(a,k),getfield(b,k)),
        (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,
         :outliers,:mask,:correlation_planes)) &&
    Hammerhead._experiment_pass_data(a.parameters)==Hammerhead._experiment_pass_data(b.parameters) &&
    (a.scale===nothing ? b.scale===nothing : b.scale!==nothing &&
       all(k -> getfield(a.scale,k)==getfield(b.scale,k),fieldnames(PhysicalScale)))
end

@testset "Versioned planar experiments" begin
    @testset "preprocessing validation and snapshots" begin
        for args in (
            ()->PreprocessStep(:unknown), ()->PreprocessStep(:invert_image; wrong=true),
            ()->PreprocessStep(:highpass_filter; sigma=Inf),
            ()->PreprocessStep(:intensity_cap; n_sigma=0),
            ()->PreprocessStep(:clahe; tiles=(0,8)), ()->PreprocessStep(:clahe; nbins=1),
            ()->PreprocessStep(:clahe; clip_limit=0.5),
            ()->PreprocessStep(:percentile_stretch; low=99,high=2),
            ()->PreprocessStep(:local_variance_normalize; epsilon=0),
            ()->PreprocessStep(:subtract_background),
            ()->PreprocessStep(:subtract_background; background=fill(NaN,2,2)),
        )
            @test_throws ArgumentError args()
        end
        bg=zeros(Float32,2,2)
        step=PreprocessStep(:subtract_background; background=bg)
        bg[1]=1
        @test step.options["background"][1]==0
        @test PreprocessStep(:clahe).options==Dict("tiles"=>[8,8],"nbins"=>256,"clip_limit"=>2.0)
        params=PIVParameters(window_size=16,overlap=8)
        for kwargs in ((; backend=:cuda),(; image_type=Float16),(; mask=(i,a,b)->nothing),
                       (; mask_threshold=0),(; uncertainty_backend=:unknown))
            @test_throws ArgumentError PIVRecipe(params; kwargs...)
        end
        @test_throws ArgumentError PIVRecipe(PIVParameters[])
        @test_throws ArgumentError PIVRecipe(PIVParameters(uod_threshold=Inf))
        @test PIVRecipe(PIVParameters(validation=(:velocity_magnitude=>(max=Inf,),))).passes[1].validation[1].max==Inf
    end

    mktempdir() do dir
        pairs,paths=experiment_test_files(dir)
        mask=falses(48,48); mask[:,1:10].=true
        steps=[PreprocessStep(:subtract_background;background=fill(0.01f0,48,48)),
               PreprocessStep(:intensity_cap;n_sigma=2),PreprocessStep(:highpass_filter;sigma=3),
               PreprocessStep(:clahe;tiles=(3,4),clip_limit=2,nbins=64),
               PreprocessStep(:percentile_stretch;low=2,high=98),
               PreprocessStep(:invert_image),PreprocessStep(:local_variance_normalize;sigma=2,epsilon=0.02)]
        passes=multipass_parameters([32,16]; padding=true,
            validation=(:peak_ratio=>1.2,:velocity_magnitude=>(min=0.0,max=20.0),
                        :uod=>(threshold=3.0,neighborhood_size=1,epsilon=0.2),:correlation_moment=>10.0),
            final=(keep_correlation_planes=true,))
        scale=PhysicalScale(pixel_size=0.02,dt=0.001,length_unit="mm",time_unit="s")
        recipe=PIVRecipe(passes;preprocessing=steps,mask,roi=ROI(5:44,5:44),scale,
            image_type=Float32,predictor_smoothing=false,mask_threshold=0.25)
        id=recipe_identity(recipe)
        @test length(id)==64
        mask[1,1]=false; steps[1].options["background"][1,1]=1
        @test recipe.mask[1,1] && recipe.preprocessing[1].options["background"][1,1]==0.01f0
        @test recipe_identity(recipe)==id
        record=ExperimentRecord(pairs,recipe)
        path=joinpath(dir,"experiment.jld2")
        output=joinpath(dir,"results.jld2")

        @testset "primitive round trip and deterministic replay" begin
            @test save_experiment(path,record)==path
            reopened=load_experiment(path)
            @test recipe_identity(reopened.recipe)==id && reopened.input_id==record.input_id
            @test reopened.pairs==[[1,2],[3,4]]
            @test length(reopened.input_files)==4 && isempty(reopened.runs)
            @test reopened.recipe.image_type===Float32 && reopened.recipe.roi.rows==5:44
            @test reopened.recipe.mask==recipe.mask && reopened.recipe.scale.dt==0.001
            @test reopened.creation_environment==record.creation_environment
            payload=JLD2.load(path,"experiment")
            @test payload["recipe"] isa Dict && payload["recipe"]["passes"] isa Vector
            @test all(p -> Set(keys(p))==Set(String.(fieldnames(PIVParameters))),payload["recipe"]["passes"])
            @test !any(v -> v isa Function,values(payload))
            @test payload["recipe"]["preprocessing"][1]["options"]["background"] isa Matrix{Float32}
            function manual(image)
                image=subtract_background(image,recipe.preprocessing[1].options["background"])
                image=intensity_cap(image;n_sigma=2)
                image=highpass_filter(image;sigma=3)
                image=clahe(image;tiles=(3,4),clip_limit=2,nbins=64)
                image=percentile_stretch(image;low=2,high=98)
                image=invert_image(image)
                local_variance_normalize(image;sigma=2,epsilon=0.02)
            end
            expected=run_piv_sequence(pairs,recipe.passes;preprocess=manual,
                progress=false,image_type=Float32,mask=recipe.mask,roi=recipe.roi,scale,
                threaded=false,predictor_smoothing=false,mask_threshold=0.25)
            run=replay_experiment(reopened;output,run_record=path)
            @test run.status===:completed && run.completed_pairs==2 && run.error===nothing
            @test run.recipe_id==id && run.input_id==record.input_id
            @test run.started_at<=run.finished_at
            @test run.output_sha256==Hammerhead._experiment_file_digest(output)
            @test run.environment==reopened.creation_environment
            actual=load_results(output)
            @test length(actual)==2 && all(experiment_same_result.(actual,expected))
            @test actual[1] isa PIVResult{Float32} && any(actual[1].mask)
            saved_run=only(load_experiment(path).runs)
            @test saved_run.run_id==run.run_id && saved_run.completed_pairs==2
            @test saved_run.environment==run.environment
            another=joinpath(dir,"replay.jld2")
            rerun=replay_experiment(load_experiment(path);output=another,run_record=path)
            @test rerun.run_id!=run.run_id && rerun.recipe_id==run.recipe_id
            @test length(load_experiment(path).runs)==2
            @test all(experiment_same_result.(load_results(another),expected))
        end

        @testset "location-independent content and setting identities" begin
            moved=joinpath(dir,"moved"); mkpath(moved)
            relocated=[joinpath(moved,basename(p)) for p in paths]
            for (from,to) in zip(paths,relocated); cp(from,to); end
            second=ExperimentRecord([(relocated[1],relocated[2]),(relocated[3],relocated[4])],recipe)
            @test second.input_id==record.input_id
            @test recipe_identity(second.recipe)==id
            reused=ExperimentRecord([pairs[1],pairs[1]],recipe)
            @test reused.input_id==record.input_id && length(reused.input_files)==2
            reordered=ExperimentRecord(reverse(pairs),recipe)
            # Equal contents in this fixture mean reversed identical pairs are
            # scientifically identical; swapping frame A/B changes identity.
            swapped=ExperimentRecord([(p[2],p[1]) for p in pairs],recipe)
            @test swapped.input_id!=record.input_id
            @test reordered.input_id==record.input_id
            changed=PIVRecipe(passes;preprocessing=recipe.preprocessing,mask=recipe.mask,
                roi=recipe.roi,scale,image_type=Float32,predictor_smoothing=true,mask_threshold=0.25)
            @test recipe_identity(changed)!=id
            @test recipe_identity(PIVRecipe(PIVParameters(window_size=16,overlap=8))) !=
                  recipe_identity(PIVRecipe(PIVParameters(window_size=16,overlap=4)))
            @test Hammerhead._experiment_digest(reshape(Float32[1,2,3,4],2,2)) !=
                  Hammerhead._experiment_digest(reshape(Float32[1,2,3,4],1,4))
            @test Hammerhead._experiment_digest(Float32[1,2])!=Hammerhead._experiment_digest(Float64[1,2])
            @test Hammerhead._experiment_digest(Dict("a"=>1,"b"=>2))==
                  Hammerhead._experiment_digest(Dict("b"=>2,"a"=>1))
        end

        @testset "mixed background precision and unbounded validators" begin
            parameters=PIVParameters(window_size=16,overlap=8,
                validation=(:velocity_magnitude=>(min=0.0,max=Inf),))
            background=fill(0.01,48,48)
            mixed=PIVRecipe(parameters;image_type=Float32,
                preprocessing=[PreprocessStep(:subtract_background;background)])
            mixed_record=ExperimentRecord(pairs,mixed)
            mixed_path=joinpath(dir,"mixed.jld2")
            save_experiment(mixed_path,mixed_record)
            reopened=load_experiment(mixed_path)
            @test reopened.recipe.preprocessing[1].options["background"] isa Matrix{Float64}
            @test reopened.recipe.passes[1].validation[1].max==Inf
            @test recipe_identity(reopened.recipe)==recipe_identity(mixed)
            mixed_output=joinpath(dir,"mixed-results.jld2")
            @test replay_experiment(reopened;output=mixed_output).status===:completed
            expected=run_piv_sequence(pairs,parameters;image_type=Float32,threaded=false,
                progress=false,preprocess=image->subtract_background(image,background))
            @test all(experiment_same_result.(load_results(mixed_output),expected))
            @test load_results(mixed_output)[1] isa PIVResult{Float32}
        end

        @testset "KA backend replay" begin
            parameters=PIVParameters(window_size=16,overlap=8,uod_enable=false)
            ka_recipe=PIVRecipe(parameters;backend=:ka,image_type=Float32)
            ka_record=ExperimentRecord([pairs[1]],ka_recipe)
            ka_path=joinpath(dir,"ka-experiment.jld2")
            save_experiment(ka_path,ka_record)
            reopened=load_experiment(ka_path)
            @test reopened.recipe.backend===:ka
            ka_output=joinpath(dir,"ka-results.jld2")
            @test replay_experiment(reopened;output=ka_output).status===:completed
            expected=run_piv_sequence([pairs[1]],parameters;backend=:ka,
                image_type=Float32,threaded=false,progress=false)
            @test experiment_same_result(only(load_results(ka_output)),only(expected))
        end

        @testset "refusal protects files and malformed records" begin
            sentinel=joinpath(dir,"sentinel.jld2"); write(sentinel,"leave unchanged")
            original_bytes=read(paths[1])
            for destination in (paths[1],path)
                before=read(destination)
                @test_throws ArgumentError replay_experiment(record;output=destination)
                @test read(destination)==before
            end
            @test_throws ArgumentError save_experiment(paths[1],record)
            @test read(paths[1])==original_bytes
            image_alias=joinpath(dir,"image-alias.png")
            Base.hardlink(paths[1],image_alias)
            @test_throws ArgumentError replay_experiment(record;output=image_alias)
            @test_throws ArgumentError save_experiment(image_alias,record)
            @test_throws ArgumentError replay_experiment(record;output=sentinel,run_record=image_alias)
            @test read(paths[1])==original_bytes
            record_alias=joinpath(dir,"record-alias.jld2")
            Base.hardlink(path,record_alias)
            before_record=read(path)
            @test_throws ArgumentError replay_experiment(record;output=record_alias)
            @test read(path)==before_record
            other_record=joinpath(dir,"other-record.jld2")
            save_experiment(other_record,deepcopy(record))
            @test_throws ArgumentError replay_experiment(record;output=other_record)
            @test_throws ArgumentError replay_experiment(record;output=sentinel,run_record=paths[2])
            @test_throws ArgumentError replay_experiment(record;output=sentinel,run_record=sentinel)
            @test_throws ArgumentError replay_experiment(record;output=sentinel,custom_preprocess=identity)
            @test read(sentinel,String)=="leave unchanged"

            broken=deepcopy(record)
            broken.recipe.mask[1,1]=!broken.recipe.mask[1,1]
            @test_throws ArgumentError replay_experiment(broken;output=sentinel)
            @test_throws ArgumentError save_experiment(sentinel,broken)
            broken=deepcopy(record); broken.pairs[1][1]=99
            @test_throws ArgumentError replay_experiment(broken;output=sentinel)
            @test read(sentinel,String)=="leave unchanged"

            saved=read(paths[1]); open(io -> write(io,UInt8(0)),paths[1],"a")
            @test_throws ArgumentError replay_experiment(record;output=sentinel)
            @test read(sentinel,String)=="leave unchanged"
            write(paths[1],saved)

            malformed=joinpath(dir,"malformed.jld2")
            payload=JLD2.load(path,"experiment")
            mutations=[
                d->delete!(d,"pairs"),
                d->delete!(d["recipe"]["passes"][1],"window_size"),
                d->(d["recipe"]["backend"]="cuda"),
                d->(d["recipe"]["uncertainty_backend"]="unknown"),
                d->(d["recipe"]["image_type"]="Float16"),
                d->(d["recipe"]["threaded"]=1),
                d->(d["recipe"]["preprocessing"][1]["operation"]="eval"),
                d->delete!(d["recipe"]["preprocessing"][2]["options"],"n_sigma"),
                d->(d["recipe_id"]=repeat("0",64)),
                d->(d["input_id"]=repeat("0",64)),
                d->(d["pairs"]=[[0,1]]),
                d->(d["creation_environment"]["julia_threads"]=0),
                d->(d["runs"][1]["completed_pairs"]=0),
            ]
            for mutate in mutations
                changed=deepcopy(payload); mutate(changed)
                jldopen(malformed,"w") do f
                    f["experiment_format_version"]=1; f["experiment"]=changed
                end
                @test_throws ArgumentError load_experiment(malformed)
                @test read(sentinel,String)=="leave unchanged"
            end
            for version in (2,999,"1",true)
                jldopen(malformed,"w") do f
                    f["experiment_format_version"]=version; f["experiment"]=payload
                end
                @test_throws ArgumentError load_experiment(malformed)
            end
            badmask=PIVRecipe(PIVParameters(window_size=16,overlap=8);mask=falses(40,40))
            @test_throws ArgumentError ExperimentRecord(pairs,badmask)
            badroi=PIVRecipe(PIVParameters(window_size=16,overlap=8);roi=ROI(1:60,1:30))
            @test_throws ArgumentError ExperimentRecord(pairs,badroi)
            bigpass=PIVRecipe(PIVParameters(window_size=64,overlap=32))
            @test_throws ArgumentError ExperimentRecord(pairs,bigpass)
        end

        @testset "explicit script references and failed-run metadata" begin
            script=joinpath(dir,"custom.jl")
            write(script,"error(\"this script must never be executed automatically\")\n")
            reference=ScriptReference(script;entrypoint="custom_preprocess(image)")
            simple=PIVRecipe(PIVParameters(window_size=16,overlap=8);external_preprocess=reference)
            scripted=ExperimentRecord(pairs,simple)
            scripted_path=joinpath(dir,"scripted.jld2"); save_experiment(scripted_path,scripted)
            @test_throws ArgumentError replay_experiment(scripted;output=output)
            before=read(script)
            @test_throws ArgumentError replay_experiment(scripted;output=script,custom_preprocess=identity)
            @test_throws ArgumentError save_experiment(script,scripted)
            @test read(script)==before
            script_alias=joinpath(dir,"script-alias.jl")
            Base.hardlink(script,script_alias)
            @test_throws ArgumentError replay_experiment(scripted;output=script_alias,custom_preprocess=identity)
            @test_throws ArgumentError save_experiment(script_alias,scripted)
            @test read(script)==before
            copied_script=joinpath(dir,"copied-script.jl"); cp(script,copied_script)
            relocated=PIVRecipe(simple.passes;external_preprocess=ScriptReference(copied_script;entrypoint=reference.entrypoint))
            @test recipe_identity(relocated)==recipe_identity(simple)
            ok=replay_experiment(load_experiment(scripted_path);output,custom_preprocess=identity)
            @test ok.status===:completed
            write(script,"changed\n")
            before_output=read(output)
            @test_throws ArgumentError replay_experiment(scripted;output,custom_preprocess=identity)
            @test read(output)==before_output
            write(script,before)
            failed_output=joinpath(dir,"failed.jld2")
            calls=Ref(0); primary=ErrorException("manual preprocessor failed")
            failing=image->begin
                calls[]+=1
                calls[]>=3 && throw(primary)
                image
            end
            caught=try
                replay_experiment(scripted;output=failed_output,custom_preprocess=failing,run_record=scripted_path)
                nothing
            catch err
                err
            end
            @test caught===primary
            failed=only(load_experiment(scripted_path).runs)
            @test failed.status===:failed && failed.completed_pairs==1
            @test occursin("manual preprocessor failed",failed.error)
            @test length(load_results(failed_output))==1
            @test failed.output_sha256==Hammerhead._experiment_file_digest(failed_output)
            @test_throws ArgumentError replay_experiment(scripted;output=failed_output,
                custom_preprocess=image->Float32.(image))
            @test_throws ArgumentError replay_experiment(scripted;output=failed_output,
                custom_preprocess=image->image[1:32,1:32])
        end

        @testset "creation versus actual execution environment" begin
            different=deepcopy(record)
            different.creation_environment["julia_version"]="0.0.0"
            sentinel=joinpath(dir,"environment-sentinel.jld2"); write(sentinel,"preserve")
            @test_throws ArgumentError replay_experiment(different;output=sentinel)
            @test read(sentinel,String)=="preserve"
            run=replay_experiment(different;output=sentinel,allow_environment_change=true)
            @test run.environment["julia_version"]==string(VERSION)
            @test different.creation_environment["julia_version"]=="0.0.0"
            @test run.status===:completed
        end
    end
end
