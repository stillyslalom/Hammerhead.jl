using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

# Return no strong reference to the old bundle, request, controller or local
# image variables. Collection happens after this helper's stack has returned.
Base.@noinline function preprocessing_superseded_bundle_fixture(record)
    revision=RecipeRevisionController(record);preview=RecipeImagePreviewController()
    preview_recipe_images!(preview,revision;pair_index=1,async=false)
    preview.state[]===:completed || error("initial lifetime fixture preview failed")
    old=preview.bundle[]
    refs=[WeakRef(old.raw_a),WeakRef(old.raw_b),WeakRef(old.processed_a),WeakRef(old.processed_b),WeakRef(old.mask)]
    preview_recipe_images!(preview,revision;pair_index=2,async=false)
    preview.state[]===:completed || error("replacement lifetime fixture preview failed")
    preview,refs
end

@testset "Ordered preprocessing recipe revisions" begin
    C=HammerheadGUI.Controllers
    mktempdir() do dir
        files=[joinpath(dir,"acquisition $i.png") for i in 1:3]
        for (i,path) in enumerate(files)
            save(path,Gray{N0f8}.([mod(19r+7c+i,251)/250 for r in 1:32,c in 1:40]))
        end
        background=fill(.01f0,32,40);mask=falses(32,40);mask[9,10]=true
        steps=[PreprocessStep(:subtract_background;background),PreprocessStep(:intensity_cap;n_sigma=2.3),
            PreprocessStep(:highpass_filter;sigma=1.2),PreprocessStep(:clahe;tiles=(3,4),nbins=48,clip_limit=2.5),
            PreprocessStep(:percentile_stretch;low=2,high=98),PreprocessStep(:invert_image),
            PreprocessStep(:local_variance_normalize;sigma=1.4,epsilon=.012),PreprocessStep(:invert_image)]
        recipe=PIVRecipe(PIVParameters(window_size=8,overlap=4,validation=(:peak_ratio=>1.2,));
            preprocessing=steps,roi=ROI(4:29,5:36),mask,
            scale=PhysicalScale(pixel_size=.3,dt=.4,length_unit="mm",time_unit="s"),
            image_type=Float32,threaded=false,predictor_smoothing=false,mask_threshold=.7,uncertainty_backend=:cpu)
        source=ExperimentRecord([(files[2],files[1]),(files[1],files[1]),(files[3],files[2])],recipe)
        source_path=joinpath(dir,"original.jld2");save_experiment(source_path,source)
        original_data=Hammerhead._experiment_recipe_data(source.recipe)
        rc=RecipeRevisionController(source_path)
        @test rc.preprocessing_selected[]==1
        @test length(rc.preprocessing_drafts[])==8
        @test rc.preprocessing_drafts[][1].background isa Matrix{Float32}
        @test reinterpret(UInt8,vec(rc.preprocessing_drafts[][1].background))==reinterpret(UInt8,vec(background))
        @test isempty(revision_diff(rc))
        @test Set(f.key for f in preprocessing_fields(:clahe))==Set([:tiles,:nbins,:clip_limit])
        @test Set(f.key for f in preprocessing_fields(:local_variance_normalize))==Set([:sigma,:epsilon])
        @test isempty(preprocessing_fields(:subtract_background))
        @test isempty(preprocessing_fields(:invert_image))
        @test_throws ArgumentError preprocessing_fields(:custom)
        fields=preprocessing_fields(:clahe);pop!(fields);@test length(preprocessing_fields(:clahe))==3
        apply_recipe_revision!(rc;async=false);@test !rc.dirty[]
        set_revision_preprocess!(rc,4,copy(rc.preprocessing_drafts[][4].options));@test !rc.dirty[]
        move_revision_preprocess!(rc,4,4);@test !rc.dirty[]
        @test_throws ArgumentError set_revision_preprocess!(rc,4,Dict(:nbins=>48))
        @test_throws ArgumentError set_revision_preprocess!(rc,4,Dict(:unknown=>"1"))
        @test_throws ArgumentError set_revision_preprocess!(rc,true,Dict())
        set_revision_preprocess!(rc,4,Dict(:tiles=>"(2, 5)",:nbins=>"64",:clip_limit=>"3.0"))
        set_revision_preprocess!(rc,7,Dict(:epsilon=>"0.02"))
        candidate=revision_recipe(rc)
        @test candidate.preprocessing[4].options["tiles"]==[2,5]
        @test candidate.preprocessing[4].options["nbins"]==64
        @test candidate.preprocessing[7].options["epsilon"]==.02
        for field in fieldnames(PIVRecipe)
            field in (:preprocessing,:recipe_id) && continue
            @test Hammerhead._experiment_recipe_data(candidate)[String(field)]==original_data[String(field)]
        end
        valid=rc.candidate[]
        set_revision_preprocess!(rc,4,Dict(:nbins=>"invalid hidden option"));move_revision_preprocess!(rc,4,8)
        @test rc.preprocessing_drafts[][8].options[:nbins]=="invalid hidden option"
        insert_revision_preprocess!(rc,9;source=8)
        @test rc.preprocessing_drafts[][9].options[:nbins]=="invalid hidden option"
        @test rc.preprocessing_selected[]==9
        apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:failed && rc.candidate[]===valid
        picked=Ref(false);save_recipe_revision!(rc,()->(picked[]=true;joinpath(dir,"bad.jld2"));async=false)
        @test !picked[] && !isfile(joinpath(dir,"bad.jld2"))
        delete_revision_preprocess!(rc,9);delete_revision_preprocess!(rc,8)
        while !isempty(rc.preprocessing_drafts[]);delete_revision_preprocess!(rc,1);end
        @test rc.preprocessing_selected[]==0 && isempty(revision_recipe(rc).preprocessing)
        insert_revision_preprocess!(rc,1,:subtract_background)
        @test_throws ArgumentError revision_recipe(rc)
        set_revision_background!(rc,1,background)
        @test rc.preprocessing_drafts[][1].background!==background
        apply_recipe_revision!(rc;async=false)
        old_precision_id=rc.candidate[].recipe_id
        set_revision_background!(rc,1,Float64.(background))
        @test rc.dirty[] && rc.preprocessing_drafts[][1].background isa Matrix{Float64}
        @test revision_recipe(rc).recipe_id!=old_precision_id
        apply_recipe_revision!(rc;async=false)
        set_revision_background!(rc,1,Float64.(background));@test !rc.dirty[]
        set_revision_background!(rc,1,background)
        changed_during_capture=on(rc.running) do busy
            busy || return
            rows=deepcopy(rc.preprocessing_drafts[])
            rows[1]=merge(rows[1],(background=Float64.(background),))
            rc.preprocessing_drafts.val=rows
        end
        apply_recipe_revision!(rc;async=false);off(changed_during_capture)
        @test rc.candidate[].preprocessing[1].options["background"] isa Matrix{Float32}
        @test rc.dirty[]
        set_revision_background!(rc,1,background)
        insert_revision_preprocess!(rc,2;source=1)
        @test rc.preprocessing_drafts[][1].background!==rc.preprocessing_drafts[][2].background
        @test rc.preprocessing_drafts[][2].background==background
        @test_throws ArgumentError set_revision_background!(rc,1,fill(NaN,32,40))
        delete_revision_preprocess!(rc,2)
        bgfile=joinpath(dir,"background replacement.png");save(bgfile,Gray{N0f8}.(fill(.2,32,40)))
        replaced=on(rc.running) do busy
            busy || return
            rows=deepcopy(rc.preprocessing_drafts[])
            rows[1]=merge(rows[1],(background=Float64.(rows[1].background),))
            rc.preprocessing_drafts.val=rows
        end
        load_revision_background!(rc,1,bgfile;async=false);off(replaced)
        @test rc.state[]===:failed && rc.preprocessing_drafts[][1].background isa Matrix{Float64}
        set_revision_background!(rc,1,background)
        previous=deepcopy(rc.preprocessing_drafts[])
        load_revision_background!(rc,1,()->nothing;async=false)
        @test rc.state[]===:cancelled && rc.preprocessing_drafts[]==previous
        seen=Ref(false)
        listener=on(rc.running) do busy
            busy || return
            seen[]=true
            @test_throws ArgumentError move_revision_preprocess!(rc,1,1)
            rc.preprocessing_selected.val=0
        end
        load_revision_background!(rc,1,()->bgfile;async=false);off(listener)
        @test seen[] && rc.state[]===:completed && rc.preprocessing_selected[]==1
        @test rc.preprocessing_drafts[][1].background isa Matrix{Float32}
        # File replacement protects the normalized absolute spelling; it need
        # not match realpath's expanded Windows short-name spelling.
        @test Hammerhead._artifact_local_path(bgfile) in rc.protected_paths[]
        # Public path-list mutation cannot erase lifetime consumed-source guards.
        erased=on(rc.preprocessing_drafts) do _
            rc.protected_paths[]=String[]
        end
        load_revision_background!(rc,1,bgfile;async=false);off(erased)
        @test isempty(rc.protected_paths[])
        bytes=read(bgfile);save_recipe_revision!(rc,bgfile;async=false)
        @test rc.state[]===:failed && read(bgfile)==bytes
        fresh=revision_record(rc)
        @test fresh.input_id==source.input_id && fresh.pairs==source.pairs
        @test isempty(fresh.runs) && isempty(fresh.record_paths)
        @test Hammerhead._experiment_recipe_data(source.recipe)==original_data
        queued=RecipeRevisionController(source)
        apply_recipe_revision!(queued)
        @test queued.running[] && queued.candidate[]===nothing
        wait(queued.task[]);@test queued.state[]===:completed
        # Metadata preview works offline; pixel work and fresh save refuse absent bytes.
        rm(files[3]);apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:completed
        save_recipe_revision!(rc,joinpath(dir,"offline.jld2");async=false)
        @test rc.state[]===:failed && !isfile(joinpath(dir,"offline.jld2"))
    end
end

@testset "Explicit image preview identity and failure retention" begin
    mktempdir() do dir
        files=[joinpath(dir,"image $i.png") for i in 1:3]
        for (i,path) in enumerate(files)
            save(path,Gray{N0f8}.([mod(23r+11c+i,251)/250 for r in 1:24,c in 1:32]))
        end
        mask=falses(24,32);mask[1,1]=true
        recipe=PIVRecipe(PIVParameters(window_size=8,overlap=4);image_type=Float32,
            preprocessing=[PreprocessStep(:invert_image)],roi=ROI(3:22,4:29),mask)
        record=ExperimentRecord([(files[2],files[1]),(files[1],files[3])],recipe)
        retained,released=preprocessing_superseded_bundle_fixture(record)
        GC.gc(true)
        @test all(ref->ref.value===nothing,released)
        @test retained.bundle[].pair_index==2 && size(retained.bundle[].processed_a)==(24,32)
        @test retained.task[]===nothing && !retained.running[]
        rc=RecipeRevisionController(record);pc=RecipeImagePreviewController()
        listener=on(pc.running) do busy
            busy || return
            insert_revision_preprocess!(rc,2,:invert_image)
        end
        preview_recipe_images!(pc,rc;pair_index=1,async=false);off(listener)
        @test pc.state[]===:completed && !pc.running[]
        bundle=pc.bundle[]
        @test bundle.recipe_id==recipe.recipe_id && bundle.input_id==record.input_id && bundle.pair_index==1
        @test bundle.input_paths==[f["path"] for f in record.input_files[record.pairs[1]]]
        @test bundle.raw_a isa Matrix{Float32} && bundle.processed_a isa Matrix{Float32}
        @test bundle.processed_a==invert_image(bundle.raw_a)
        @test size(bundle.processed_a)==(24,32) && bundle.roi==recipe.roi
        @test bundle.mask==mask && bundle.mask!==mask
        @test bundle.raw_a!==bundle.processed_a && bundle.raw_b!==bundle.processed_b
        @test revision_recipe(rc).recipe_id!=bundle.recipe_id
        failing=on(pc.bundle) do _
            error("image publication observer")
        end
        preview_recipe_images!(pc,rc;pair_index=1,async=false);off(failing)
        @test pc.state[]===:failed && !pc.running[] && pc.bundle[]!==bundle
        @test pc.bundle[].recipe_id==revision_recipe(rc).recipe_id
        bundle=pc.bundle[]
        @test_throws ArgumentError preview_recipe_images!(pc,rc;pair_index=true)
        @test_throws ArgumentError preview_recipe_images!(pc,rc;pair_index=0)
        set_revision_pass!(rc,1,Dict(:n_peaks=>"not a number"))
        preview_recipe_images!(pc,rc;async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        set_revision_pass!(rc,1,Dict(:n_peaks=>"1"))
        write(files[3],"changed source")
        preview_recipe_images!(pc,rc;pair_index=2,async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        script=joinpath(dir,"must not execute.jl");write(script,"error(\"executed script\")")
        scripted=PIVRecipe(recipe.passes;image_type=Float32,external_preprocess=ScriptReference(script;entrypoint="condition"))
        scriptrecord=ExperimentRecord([(files[2],files[1])],scripted)
        scriptrevision=RecipeRevisionController(scriptrecord)
        apply_recipe_revision!(scriptrevision;async=false);@test scriptrevision.state[]===:completed
        preview_recipe_images!(pc,scriptrevision;async=false)
        @test pc.state[]===:failed && pc.bundle[]===bundle
        @test occursin("never executes",sprint(showerror,pc.error[]))
        throwing=on(pc.running) do busy
            busy && error("preview startup observer")
        end
        preview_recipe_images!(pc,scriptrevision;async=false);off(throwing)
        @test !pc.running[] && pc.task[]===nothing && pc.state[]===:failed
    end
end
