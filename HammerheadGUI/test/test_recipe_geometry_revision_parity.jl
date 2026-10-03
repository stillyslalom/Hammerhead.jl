using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

function geometry_parity_files(directory)
    base=[mod(37r+19c+div(r*c,3),251)/250 for r in 1:56,c in 1:72]
    files=[joinpath(directory,"geometry frame $i.png") for i in 1:2]
    for (i,path) in enumerate(files)
        save(path,Gray{N0f8}.(circshift(base,(i-1,2(i-1)))))
    end
    files
end

function geometry_parity_recipe(T,backend)
    background_type=T===Float32 ? Float64 : Float32
    background=background_type.([.03+mod(r+2c,9)/1000 for r in 1:56,c in 1:72])
    mask=falses(56,72);mask[:,1:23].=true;mask[54:56,69:72].=true
    # Enlarged search is a CPU capability; the portable KA engine requires
    # search_area_size == window_size. Exercise both admitted configurations.
    coarse_search=backend===:ka ? 16 : 20
    final_search=backend===:ka ? 8 : 12
    passes=[PIVParameters(window_size=16,search_area_size=coarse_search,overlap=8,
                padding=true,uod_enable=false,n_peaks=2,replace_outliers=false),
            PIVParameters(window_size=8,search_area_size=final_search,overlap=4,
                padding=true,uod_enable=false,uncertainty=true,
                validation=(:peak_ratio=>1.0,),replace_outliers=false)]
    PIVRecipe(passes;image_type=T,backend,threaded=false,predictor_smoothing=false,
        uncertainty_backend=:cpu,mask_threshold=.5,mask,roi=ROI(7:50,11:66),
        scale=PhysicalScale(.012345678901234567,.004567890123456789," mm ","s"),
        preprocessing=[PreprocessStep(:subtract_background;background),
            PreprocessStep(:highpass_filter;sigma=1.3)])
end

# Independent numerical oracle: standalone core operations on each complete
# decoded image, then the public PIV driver. No saved preprocessing closure or
# production revision/preview helper is used to construct expected arrays.
function geometry_parity_condition(image,recipe)
    first_step,second_step=recipe.preprocessing
    highpass_filter(subtract_background(image,first_step.options["background"]);
        sigma=second_step.options["sigma"])
end
function geometry_parity_direct(a,b,recipe;roi=recipe.roi,mask=recipe.mask,scale=recipe.scale)
    run_piv(a,b,recipe.passes;backend=recipe.backend,
        uncertainty_backend=recipe.uncertainty_backend,threaded=recipe.threaded,
        predictor_smoothing=recipe.predictor_smoothing,mask_threshold=recipe.mask_threshold,
        roi,mask,scale)
end
function geometry_parity_measurements(a,b)
    all(key->isequal(getfield(a,key),getfield(b,key)),
        (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,
         :outliers,:mask,:correlation_planes))
end
function geometry_parity_scale(a,b)
    a===nothing || b===nothing ? a===b :
        all(k->isequal(getfield(a,k),getfield(b,k)),fieldnames(PhysicalScale))
end

@testset "Geometry revisions preserve exact imported and disabled metadata" begin
    H=HammerheadGUI.Hammerhead
    mktempdir() do directory
        files=geometry_parity_files(directory)
        for T in (Float32,Float64)
            source=ExperimentRecord([(files[2],files[1]),(files[1],files[2])],geometry_parity_recipe(T,:cpu))
            old_output=joinpath(directory,"old-output-$T.jld2");write(old_output,"prior output fixture")
            push!(source.runs,ExperimentRun("b58f888f-9e9b-4e4d-ad85-5a8c4d5c4085",source.recipe.recipe_id,
                source.input_id,1.,2.,:failed,0,old_output,nothing,deepcopy(source.creation_environment),"prior failure"))
            original_data=H._experiment_recipe_data(source.recipe)
            rc=RecipeRevisionController(source)
            @test recipe_identity(revision_recipe(rc))==recipe_identity(source.recipe)
            @test rc.roi_draft[].enabled && rc.scale_draft[].enabled
            @test rc.scale_draft[].values[:length_unit]==" mm "
            @test parse(Float64,rc.scale_draft[].values[:pixel_size])===source.recipe.scale.pixel_size
            @test parse(Float64,rc.scale_draft[].values[:dt])===source.recipe.scale.dt
            set_revision_roi!(rc,Dict(:row_first=>"9",:row_last=>"48",:col_first=>"13",:col_last=>"64"))
            set_revision_scale!(rc,Dict(:pixel_size=>"0.025",:dt=>"0.01",:length_unit=>"mm",:time_unit=>"s"))
            candidate=revision_recipe(rc)
            @test candidate.roi.rows==9:48 && candidate.roi.cols==13:64
            @test geometry_parity_scale(candidate.scale,PhysicalScale(.025,.01,"mm","s"))
            data=H._experiment_recipe_data(candidate)
            @test all(key->isequal(data[key],original_data[key]),setdiff(collect(keys(data)),["roi","scale"]))
            @test size(candidate.mask)==(56,72) && candidate.mask==source.recipe.mask
            @test size(candidate.preprocessing[1].options["background"])==(56,72)
            @test eltype(candidate.preprocessing[1].options["background"])!==T
            @test H._experiment_recipe_data(source.recipe)==original_data
            @test recipe_identity(candidate)!=recipe_identity(source.recipe)
            fresh=revision_record(rc)
            @test fresh.input_id==source.input_id && fresh.pairs==source.pairs
            @test isempty(fresh.runs) && isempty(fresh.record_paths)
            @test length(rc.original.runs)==1
            set_revision_roi!(rc,Dict(:row_first=>"invalid retained bound");enabled=false)
            set_revision_scale!(rc,Dict(:dt=>"invalid retained delay");enabled=false)
            disabled=revision_recipe(rc)
            @test disabled.roi===nothing && disabled.scale===nothing
            @test rc.roi_draft[].values[:row_first]=="invalid retained bound"
            @test rc.scale_draft[].values[:dt]=="invalid retained delay"
            disabled_path=joinpath(directory,"disabled-$T.jld2")
            save_recipe_revision!(rc,disabled_path;async=false)
            @test rc.state[]===:completed
            persisted=load_experiment(disabled_path)
            @test persisted.recipe.roi===nothing && persisted.recipe.scale===nothing
            @test isempty(persisted.runs) && persisted.input_id==source.input_id
            set_revision_roi!(rc,Dict{Symbol,String}();enabled=true)
            @test_throws ArgumentError revision_recipe(rc)
        end
        absent=ExperimentRecord([(files[1],files[2])],PIVRecipe(PIVParameters(window_size=8,overlap=4)))
        rc=RecipeRevisionController(absent)
        @test !rc.roi_draft[].enabled && !rc.scale_draft[].enabled
        @test all(isempty,values(rc.scale_draft[].values))
        @test revision_recipe(rc).roi===nothing && revision_recipe(rc).scale===nothing
    end
end

@testset "Revised replay remains pixel native with original ROI coordinates" begin
    H=HammerheadGUI.Hammerhead
    mktempdir() do directory
        files=geometry_parity_files(directory)
        for T in (Float32,Float64),backend in (:cpu,:ka)
            source=ExperimentRecord([(files[2],files[1]),(files[1],files[2])],geometry_parity_recipe(T,backend))
            source_data=H._experiment_recipe_data(source.recipe)
            source_path=joinpath(directory,"source-$T-$backend.jld2");save_experiment(source_path,source)
            rc=RecipeRevisionController(source_path)
            set_revision_roi!(rc,Dict(:row_first=>"9",:row_last=>"48",:col_first=>"13",:col_last=>"64"))
            set_revision_scale!(rc,Dict(:pixel_size=>"0.025",:dt=>"0.01",:length_unit=>"mm",:time_unit=>"s"))
            set_revision_pass!(rc,2,Dict(:min_peak_ratio=>"0.75"))
            set_revision_preprocess!(rc,2,Dict(:sigma=>"1.6"))
            revised_path=joinpath(directory,"revised-$T-$backend.jld2")
            source_bytes=read(source_path)
            save_recipe_revision!(rc,revised_path;async=false)
            @test rc.state[]===:completed && !rc.running[]
            revised=load_experiment(revised_path)
            @test revised.input_id==source.input_id && revised.pairs==source.pairs
            @test revised.input_files==source.input_files && isempty(revised.runs)
            @test revised.recipe.recipe_id!=source.recipe.recipe_id
            @test revised.creation_environment["julia_version"]==string(VERSION)
            @test revised.recipe.passes[2].min_peak_ratio==.75
            @test revised.recipe.passes[2].validation==source.recipe.passes[2].validation
            @test revised.recipe.preprocessing[2].options["sigma"]==1.6
            @test H._experiment_recipe_data(source.recipe)==source_data
            @test read(source_path)==source_bytes
            recipe=revised.recipe;roi=recipe.roi
            output=joinpath(directory,"replayed-$T-$backend.jld2")
            run=replay_experiment(revised;output)
            @test run.status===:completed && run.completed_pairs==2
            raw_index=load_results(output;lazy=true)
            for (i,pair) in enumerate(source.pairs)
                raw_a,raw_b=(load_image(T,source.input_files[k]["path"]) for k in pair)
                a,b=geometry_parity_condition(raw_a,recipe),geometry_parity_condition(raw_b,recipe)
                direct=geometry_parity_direct(a,b,recipe)
                raw=raw_index[i]
                @test raw isa PIVResult{T} && geometry_parity_measurements(raw,direct)
                @test geometry_parity_scale(raw.scale,recipe.scale)
                @test raw.mask==direct.mask && any(raw.mask) && any(.!raw.mask)
                @test any(isfinite,raw.u[.!raw.mask])
                unscaled=geometry_parity_direct(a,b,recipe;scale=nothing)
                @test unscaled.scale===nothing && geometry_parity_measurements(raw,unscaled)
                local_result=geometry_parity_direct(a[roi.rows,roi.cols],b[roi.rows,roi.cols],recipe;
                    roi=nothing,mask=recipe.mask[roi.rows,roi.cols],scale=nothing)
                @test raw.x==local_result.x .+ T(first(roi.cols)-1)
                @test raw.y==local_result.y .+ T(first(roi.rows)-1)
                @test isequal(raw.u,local_result.u) && isequal(raw.v,local_result.v)
                @test a[roi.rows,roi.cols]!=highpass_filter(
                    subtract_background(raw_a[roi.rows,roi.cols],recipe.preprocessing[1].options["background"][roi.rows,roi.cols]);sigma=1.6)
                expected=physical(raw)
                @test expected.x==raw.x .* T(.025) && expected.y==raw.y .* T(.025)
                @test isequal(expected.u,raw.u .* T(2.5)) && isequal(expected.v,raw.v .* T(2.5))
                explorer=ResultExplorer(raw_index)
                set_frame!(explorer,i)
                displayed=current_result(explorer)
                @test geometry_parity_measurements(displayed,expected)
                @test displayed.scale.pixel_size==1 && displayed.scale.dt==1
                @test displayed.scale.length_unit=="mm" && displayed.scale.time_unit=="s"
                @test physical(displayed)===displayed
                @test geometry_parity_measurements(raw_index[i],raw)
            end
        end
    end
end

@testset "Geometry capture, offline validation and strict failure retention" begin
    mktempdir() do directory
        files=geometry_parity_files(directory)
        source=ExperimentRecord([(files[2],files[1])],geometry_parity_recipe(Float32,:cpu))
        rc=RecipeRevisionController(source)
        apply_recipe_revision!(rc;async=false)
        previous=rc.candidate[]
        for options in (Dict(:row_first=>"0"),Dict(:row_last=>"6"),Dict(:row_last=>"57"),
                        Dict(:col_last=>"73"),Dict(:row_first=>"true"),Dict(:col_first=>"10.5"),
                        Dict(:row_first=>"7",:row_last=>"10"))
            bad=RecipeRevisionController(source)
            set_revision_roi!(bad,options)
            @test_throws ArgumentError revision_recipe(bad)
            destination=joinpath(directory,"bad-bounds.jld2")
            save_recipe_revision!(bad,destination;async=false)
            @test bad.state[]===:failed && !ispath(destination)
        end
        for field in (:pixel_size,:dt),text in ("0","-1","NaN","Inf","1e999","1e-999","bad text")
            bad=RecipeRevisionController(source)
            set_revision_scale!(bad,Dict(field=>text))
            @test_throws ArgumentError revision_recipe(bad)
        end
        # Backgrounds are never resized to an edited crop; incompatible full-frame
        # replacements fail during recipe composition, not as silent interpolation.
        set_revision_background!(rc,1,zeros(Float64,44,56))
        apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:failed && rc.candidate[]===previous
        set_revision_background!(rc,1,source.recipe.preprocessing[1].options["background"])
        set_revision_roi!(rc,Dict(:row_first=>"9",:row_last=>"48",:col_first=>"13",:col_last=>"64"))
        set_revision_scale!(rc,Dict(:pixel_size=>"0.025",:dt=>"0.01"))
        captured=revision_recipe(rc)
        pc=RecipeImagePreviewController()
        observer=on(pc.running) do active
            active || return
            set_revision_roi!(rc,Dict(:row_first=>"7",:row_last=>"50",:col_first=>"11",:col_last=>"66"))
            set_revision_scale!(rc,Dict(:pixel_size=>"0.05",:dt=>"0.02"))
        end
        preview_recipe_images!(pc,rc;async=false)
        off(observer)
        bundle=pc.bundle[]
        @test pc.state[]===:completed && bundle!==nothing
        @test bundle.recipe_id==captured.recipe_id && bundle.roi.rows==9:48 && bundle.roi.cols==13:64
        @test bundle.recipe_id!=revision_recipe(rc).recipe_id
        @test size(bundle.raw_a)==(56,72) && size(bundle.processed_a)==(56,72)
        @test bundle.mask==source.recipe.mask
        @test bundle.processed_a==geometry_parity_condition(bundle.raw_a,captured)
        @test bundle.raw_a==load_image(Float32,source.input_files[source.pairs[1][1]]["path"])
        set_revision_roi!(rc,Dict(:row_last=>"57"))
        preview_recipe_images!(pc,rc;async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        set_revision_roi!(rc,Dict(:row_last=>"50"))
        # Metadata can be composed offline, but neither pixel preview nor a new
        # saved record may replace prior data using changed/missing acquisition.
        bytes=read(files[1]);rm(files[1])
        apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:completed
        preview_recipe_images!(pc,rc;async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        destination=joinpath(directory,"missing-input.jld2")
        save_recipe_revision!(rc,destination;async=false)
        @test rc.state[]===:failed && !ispath(destination)
        write(files[1],bytes)
        changed=copy(bytes);changed[end]⊻=0x01;write(files[1],changed)
        preview_recipe_images!(pc,rc;async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        write(files[1],bytes)
    end
end

@testset "ROI revision validates later acquisition dimensions" begin
    mktempdir() do directory
        files=geometry_parity_files(directory)
        smaller=[joinpath(directory,"smaller-$i.png") for i in 1:2]
        for (i,path) in enumerate(smaller)
            save(path,Gray{N0f8}.([mod(13r+29c+i,251)/250 for r in 1:40,c in 1:50]))
        end
        source=ExperimentRecord([(files[1],files[2]),(smaller[1],smaller[2])],
            PIVRecipe(PIVParameters(window_size=8,search_area_size=12,overlap=4);roi=ROI(5:32,7:38)))
        rc=RecipeRevisionController(source)
        set_revision_roi!(rc,Dict(:row_last=>"48"))
        # This fits the first 56-row pair but not the second 40-row pair.
        @test_throws ArgumentError revision_recipe(rc)
        destination=joinpath(directory,"invalid-later-pair.jld2")
        save_recipe_revision!(rc,destination;async=false)
        @test rc.state[]===:failed && !ispath(destination)
    end
end
