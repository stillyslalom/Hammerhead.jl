using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

function geometry_revision_files(dir;dimensions=((48,64),(48,64),(48,64)))
    files=[joinpath(dir,"geometry frame $i.png") for i in eachindex(dimensions)]
    for (i,path) in enumerate(files)
        nr,nc=dimensions[i]
        save(path,Gray{N0f8}.([mod(31r+7c+i,251)/250 for r in 1:nr,c in 1:nc]))
    end
    files
end

@testset "Lossless imported ROI and isotropic scale drafts" begin
    @test [f.key for f in revision_roi_fields()]==[:row_first,:row_last,:col_first,:col_last]
    @test [f.key for f in revision_scale_fields()]==[:pixel_size,:dt,:length_unit,:time_unit]
    fields=revision_roi_fields();pop!(fields);@test length(revision_roi_fields())==4
    fields=revision_scale_fields();pop!(fields);@test length(revision_scale_fields())==4
    mktempdir() do dir
        files=geometry_revision_files(dir)
        background=fill(.01,48,64);mask=falses(48,64);mask[14:19,22:27].=true
        scale=PhysicalScale(nextfloat(.2),prevfloat(.7)," µm "," s ")
        passes=[PIVParameters(window_size=16,search_area_size=24,overlap=8,
            validation=(:peak_ratio=>1.1,),max_iterations=3,convergence_tol=.025),
            PIVParameters(window_size=8,overlap=4,uod_enable=false,replace_outliers=false)]
        recipe=PIVRecipe(passes;preprocessing=[PreprocessStep(:subtract_background;background),
            PreprocessStep(:intensity_cap;n_sigma=2.2)],roi=ROI(5:44,7:58),scale,mask,
            image_type=Float32,threaded=false,predictor_smoothing=false,mask_threshold=.7,uncertainty_backend=:cpu)
        source=ExperimentRecord([(files[2],files[1]),(files[1],files[1]),(files[2],files[3])],recipe)
        source_path=joinpath(dir,"geometry source.jld2");save_experiment(source_path,source)
        source_bytes=read(source_path);original=Hammerhead._experiment_recipe_data(source.recipe)
        history=joinpath(dir,"external history.jld2");write(history,"known history")
        output=joinpath(dir,"old output.jld2");write(output,"known output")
        rc=RecipeRevisionController(source_path;protected_paths=[history,output])
        @test rc.roi_draft[].enabled && rc.scale_draft[].enabled
        @test rc.roi_draft[].values==Dict(:row_first=>"5",:row_last=>"44",:col_first=>"7",:col_last=>"58")
        @test parse(Float64,rc.scale_draft[].values[:pixel_size])===scale.pixel_size
        @test parse(Float64,rc.scale_draft[].values[:dt])===scale.dt
        @test rc.scale_draft[].values[:length_unit]==" µm " && rc.scale_draft[].values[:time_unit]==" s "
        @test isempty(revision_diff(rc)) && revision_recipe(rc).recipe_id==recipe.recipe_id
        apply_recipe_revision!(rc;async=false);@test !rc.dirty[]
        set_revision_roi!(rc,Dict());set_revision_scale!(rc,copy(rc.scale_draft[].values));@test !rc.dirty[]
        @test_throws ArgumentError set_revision_roi!(rc,Dict(:row_first=>3))
        @test_throws ArgumentError set_revision_scale!(rc,Dict(:dt=>.5))
        @test_throws ArgumentError set_revision_roi!(rc,Dict(:unknown=>"3"))
        @test_throws ArgumentError set_revision_scale!(rc,Dict(:window_size=>"8"))
        set_revision_roi!(rc,Dict(:row_first=>"8",:row_last=>"39",:col_first=>"9",:col_last=>"48"))
        set_revision_scale!(rc,Dict(:pixel_size=>"0.125",:dt=>"0.25",:length_unit=>"mm",:time_unit=>"s"))
        set_revision_pass!(rc,2,Dict(:max_iterations=>"2"))
        set_revision_preprocess!(rc,2,Dict(:n_sigma=>"2.7"))
        candidate=revision_recipe(rc)
        @test candidate.roi.rows==8:39 && candidate.roi.cols==9:48
        @test candidate.scale.pixel_size==.125 && candidate.scale.dt==.25
        @test candidate.scale.length_unit=="mm" && candidate.scale.time_unit=="s"
        @test candidate.passes[2].max_iterations==2 && candidate.preprocessing[2].options["n_sigma"]==2.7
        @test Hammerhead._experiment_pass_data(candidate.passes[1])==Hammerhead._experiment_pass_data(recipe.passes[1])
        @test candidate.preprocessing[1].options["background"] isa Matrix{Float64}
        @test candidate.preprocessing[1].options["background"]==background
        @test size(candidate.mask)==(48,64) && candidate.mask==mask
        for key in (:backend,:image_type,:threaded,:predictor_smoothing,:mask_threshold,:uncertainty_backend)
            @test getfield(candidate,key)==getfield(recipe,key)
        end
        apply_recipe_revision!(rc;async=false);valid=rc.candidate[]
        @test !rc.dirty[] && valid.recipe_id==candidate.recipe_id
        for bounds in (Dict(:row_first=>"9",:row_last=>"8"),
                       Dict(:row_first=>"0",:row_last=>"39"),
                       Dict(:row_first=>"8",:row_last=>"49"),
                       Dict(:row_first=>"8",:row_last=>"23"),
                       Dict(:row_first=>"unsubmitted invalid",:row_last=>"39"))
            set_revision_roi!(rc,bounds)
            apply_recipe_revision!(rc;async=false)
            @test rc.state[]===:failed && rc.candidate[]===valid
        end
        set_revision_roi!(rc,Dict(:row_first=>"invalid row",:row_last=>"invalid end");enabled=false)
        @test revision_recipe(rc).roi===nothing
        @test rc.roi_draft[].values[:row_first]=="invalid row"
        set_revision_roi!(rc,Dict();enabled=true)
        @test_throws ArgumentError revision_recipe(rc)
        set_revision_roi!(rc,Dict(:row_first=>"8",:row_last=>"39"))
        for bad in ("0","-1","Inf","NaN","1e-999","invalid delay")
            set_revision_scale!(rc,Dict(:dt=>bad))
            apply_recipe_revision!(rc;async=false)
            @test rc.state[]===:failed && rc.candidate[]===valid
        end
        set_revision_scale!(rc,Dict(:pixel_size=>"invalid size",:dt=>"invalid delay");enabled=false)
        @test revision_recipe(rc).scale===nothing
        @test rc.scale_draft[].values[:dt]=="invalid delay"
        set_revision_scale!(rc,Dict();enabled=true)
        @test_throws ArgumentError revision_recipe(rc)
        set_revision_scale!(rc,Dict(:pixel_size=>"1.0",:dt=>"1.0",:length_unit=>"",:time_unit=>""))
        identity=revision_recipe(rc)
        @test identity.scale!==nothing && identity.scale.pixel_size==identity.scale.dt==1
        @test isempty(identity.scale.length_unit) && isempty(identity.scale.time_unit)
        set_revision_scale!(rc,Dict();enabled=false)
        @test revision_recipe(rc).recipe_id!=identity.recipe_id
        set_revision_scale!(rc,Dict(:pixel_size=>"0.125",:dt=>"0.25",:length_unit=>"mm",:time_unit=>"s");enabled=true)
        fresh=revision_record(rc)
        @test fresh.input_id==source.input_id && fresh.pairs==source.pairs
        @test isempty(fresh.runs) && isempty(fresh.record_paths)
        @test fresh.creation_environment["julia_version"]==string(VERSION)
        destination=joinpath(dir,"geometry revision.jld2")
        save_recipe_revision!(rc,destination;async=false)
        @test rc.state[]===:completed && rc.saved_path[]==realpath(destination)
        saved=rc.saved_record[];saved_bytes=read(destination)
        @test load_experiment(destination).recipe.recipe_id==revision_recipe(rc).recipe_id
        @test read(source_path)==source_bytes && Hammerhead._experiment_recipe_data(source.recipe)==original
        for path in (source_path,files[1],history,output,destination)
            bytes=read(path);save_recipe_revision!(rc,path;async=false)
            @test rc.state[]===:failed && rc.saved_record[]===saved && read(path)==bytes
        end
        @test read(destination)==saved_bytes
        rm(files[3]);apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:completed
        save_recipe_revision!(rc,joinpath(dir,"offline geometry.jld2");async=false)
        @test rc.state[]===:failed && !isfile(joinpath(dir,"offline geometry.jld2"))
    end
end

@testset "Absent geometry and every-frame/pass validation" begin
    mktempdir() do dir
        files=geometry_revision_files(dir;dimensions=((48,64),(48,64),(32,40)))
        recipe=PIVRecipe([PIVParameters(window_size=8,overlap=4),PIVParameters(window_size=16,overlap=8)])
        source=ExperimentRecord([(files[1],files[2]),(files[3],files[3])],recipe)
        rc=RecipeRevisionController(source)
        @test !rc.roi_draft[].enabled && !rc.scale_draft[].enabled
        @test rc.roi_draft[].values[:row_last]=="48" && rc.roi_draft[].values[:col_last]=="64"
        @test all(isempty,values(rc.scale_draft[].values))
        @test revision_recipe(rc).roi===nothing && revision_recipe(rc).scale===nothing
        set_revision_roi!(rc,Dict();enabled=true)
        @test_throws ArgumentError revision_recipe(rc) # exceeds later original frame
        set_revision_roi!(rc,Dict(:row_last=>"32",:col_last=>"40"))
        @test revision_recipe(rc).roi.rows==1:32
        set_revision_roi!(rc,Dict(:row_last=>"12"))
        @test_throws ArgumentError revision_recipe(rc) # later 16px pass, not merely first 8px pass
        set_revision_roi!(rc,Dict();enabled=false)
        @test Hammerhead._experiment_pass_data.(revision_recipe(rc).passes)==Hammerhead._experiment_pass_data.(recipe.passes)
        set_revision_scale!(rc,Dict();enabled=true)
        @test_throws ArgumentError revision_recipe(rc) # absent scale never fabricates factors
        set_revision_scale!(rc,Dict(:pixel_size=>"2",:dt=>"3"))
        @test revision_recipe(rc).scale.pixel_size==2 && revision_recipe(rc).scale.dt==3
    end
end

@testset "Captured geometry, preview identity and notification cleanup" begin
    mktempdir() do dir
        files=geometry_revision_files(dir)
        recipe=PIVRecipe(PIVParameters(window_size=8,overlap=4);image_type=Float32,
            roi=ROI(5:44,7:58),scale=PhysicalScale(.2,.4,"mm","s"),
            preprocessing=[PreprocessStep(:invert_image)],mask=falses(48,64))
        record=ExperimentRecord([(files[2],files[1]),(files[1],files[3])],recipe)
        rc=RecipeRevisionController(record);pc=RecipeImagePreviewController()
        set_revision_roi!(rc,Dict(:row_first=>"8",:row_last=>"39",:col_first=>"9",:col_last=>"48"))
        captured=revision_recipe(rc)
        observer=on(rc.running) do busy
            busy || return
            @test_throws ArgumentError set_revision_roi!(rc,Dict(:row_first=>"10"))
            @test_throws ArgumentError set_revision_scale!(rc,Dict(:dt=>"0.9"))
            row=deepcopy(rc.roi_draft[]);row.values[:row_first]="10";rc.roi_draft.val=row
            row=deepcopy(rc.scale_draft[]);row.values[:dt]="0.9";rc.scale_draft.val=row
        end
        destination=joinpath(dir,"captured geometry.jld2")
        save_recipe_revision!(rc,()->destination;async=false);off(observer)
        @test rc.state[]===:completed && rc.saved_record[].recipe.recipe_id==captured.recipe_id
        @test rc.saved_record[].recipe.roi.rows==8:39 && rc.saved_record[].recipe.scale.dt==.4
        @test rc.dirty[] && revision_recipe(rc).roi.rows==10:39 && revision_recipe(rc).scale.dt==.9
        image_recipe=revision_recipe(rc)
        observer=on(pc.running) do busy
            busy || return
            set_revision_roi!(rc,Dict(:row_first=>"11"))
            set_revision_scale!(rc,Dict(:pixel_size=>"0.3"))
        end
        preview_recipe_images!(pc,rc;pair_index=1,async=false);off(observer)
        @test pc.state[]===:completed && pc.bundle[].recipe_id==image_recipe.recipe_id
        @test pc.bundle[].roi.rows==10:39 && size(pc.bundle[].processed_a)==(48,64)
        @test pc.bundle[].raw_a==load_image(Float32,files[2])
        @test pc.bundle[].processed_a==invert_image(load_image(Float32,files[2]))
        @test revision_recipe(rc).recipe_id!=pc.bundle[].recipe_id
        prior=pc.bundle[]
        set_revision_roi!(rc,Dict(:row_first=>"invalid latest ROI"))
        preview_recipe_images!(pc,rc;async=false)
        @test pc.state[]===:failed && pc.bundle[]===prior
        failing=on(rc.running) do busy
            busy && error("geometry startup observer")
        end
        apply_recipe_revision!(rc;async=false);off(failing)
        @test rc.state[]===:failed && !rc.running[] && rc.task[]===nothing
        @test rc.saved_record[].recipe.recipe_id==captured.recipe_id
    end
end
