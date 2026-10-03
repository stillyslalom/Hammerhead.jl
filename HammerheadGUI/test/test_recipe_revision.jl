using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8

@testset "Lossless saved planar pass revisions" begin
    C=HammerheadGUI.Controllers
    fields=revision_fields()
    @test Set(f.key for f in fields)==setdiff(Set(fieldnames(PIVParameters)),Set([:validation]))
    @test length(fields)==18
    pop!(fields)
    @test length(revision_fields())==18
    mktempdir() do dir
        files=[joinpath(dir,"frame $i.png") for i in 1:3]
        for (i,path) in enumerate(files)
            save(path,Gray{N0f8}.([mod(37r+13c+i,251)/250 for r in 1:48,c in 1:48]))
        end
        passes=[PIVParameters(window_size=(16,20),search_area_size=(20,24),overlap=(8,9),
            correlation_method=:phase,padding=false,apodization=:gauss,subpixel_method=:gauss9,
            n_peaks=2,peak_finder=:regionalmax,uncertainty=true,uod_enable=true,uod_threshold=3.5,
            uod_neighborhood=2,min_peak_ratio=1.1,validation=(:peak_ratio=>1.2,:velocity_magnitude=>(min=0.,max=Inf)),
            replace_outliers=false,max_iterations=3,convergence_tol=.125,keep_correlation_planes=true),
            PIVParameters(window_size=8,overlap=4,uod_enable=false,validation=(:correlation_moment=>2.0,))]
        background=fill(.02f0,48,48);mask=falses(48,48);mask[9,10]=true
        recipe=PIVRecipe(passes;preprocessing=[PreprocessStep(:subtract_background;background),
            PreprocessStep(:clahe;tiles=(3,4),nbins=64,clip_limit=2)],mask,roi=ROI(5:44,3:42),
            scale=PhysicalScale(pixel_size=.2,dt=.5,length_unit="mm",time_unit="s"),
            backend=:cpu,image_type=Float32,threaded=false,predictor_smoothing=false,
            mask_threshold=.75,uncertainty_backend=:cpu)
        source=ExperimentRecord([(files[2],files[1]),(files[1],files[1]),(files[2],files[3])],recipe)
        source.creation_environment["julia_version"]="older test environment"
        old_output=joinpath(dir,"old-output.jld2");write(old_output,"old output")
        push!(source.runs,ExperimentRun("aeff9c3e-5f18-4917-ae07-9a198078eecb",recipe.recipe_id,
            source.input_id,1.,2.,:failed,0,old_output,nothing,deepcopy(source.creation_environment),"fixture failure"))
        source_path=joinpath(dir,"source.jld2");save_experiment(source_path,source)
        history=joinpath(dir,"external history.jld2");write(history,"external history")
        report=joinpath(dir,"report.toml");write(report,"report bytes")
        source_data=Hammerhead._experiment_recipe_data(source.recipe)
        rc=RecipeRevisionController(source_path;protected_paths=[history,report])
        @test rc.original!==source
        @test rc.original.recipe.mask!==source.recipe.mask
        @test rc.original.recipe.preprocessing[1].options["background"]!==background
        @test recipe_identity(revision_recipe(rc))==recipe_identity(source.recipe)
        @test isempty(revision_diff(rc))
        @test_throws ArgumentError set_revision_pass!(rc,1,Dict(:validation=>"()"))
        @test_throws ArgumentError set_revision_pass!(rc,true,Dict(:n_peaks=>"3"))
        @test_throws ArgumentError set_revision_pass!(rc,1,Dict(:n_peaks=>3))
        set_revision_pass!(rc,1,Dict(:max_iterations=>"4",:window_size=>"(16,20)",:convergence_tol=>"0.01"))
        candidate=revision_recipe(rc)
        for key in fieldnames(PIVRecipe)
            key in (:passes,:recipe_id) && continue
            # Exact primitive comparison includes payload precision/order/scale/ROI.
            @test Hammerhead._experiment_recipe_data(candidate)[String(key)]==source_data[String(key)]
        end
        @test candidate.passes[1].validation==source.recipe.passes[1].validation
        @test candidate.passes[1].max_iterations==4
        @test candidate.passes[1].convergence_tol==.01
        apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:completed && !rc.dirty[]
        set_revision_pass!(rc,1,copy(rc.drafts[][1]))
        @test !rc.dirty[]
        move_revision_pass!(rc,1,1);@test !rc.dirty[]
        @test !isempty(rc.diff[]) && occursin("max_iterations",rc.preview_text[])
        valid=rc.candidate[]
        set_revision_pass!(rc,2,Dict(:window_size=>"invalid invisible row"))
        move_revision_pass!(rc,2,1)
        @test rc.drafts[][1][:window_size]=="invalid invisible row"
        @test rc.templates[][1].validation==source.recipe.passes[2].validation
        apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:failed && rc.error[] isa ArgumentError && rc.candidate[]===valid
        picked=Ref(false)
        save_recipe_revision!(rc,()->(picked[]=true;joinpath(dir,"invalid.jld2"));async=false)
        @test !picked[] && !isfile(joinpath(dir,"invalid.jld2"))
        set_revision_pass!(rc,1,Dict(:window_size=>"8,8"))
        rc.selected[]=2;insert_revision_pass!(rc,3)
        @test rc.selected[]==3 && rc.drafts[][3]==rc.drafts[][2]
        @test rc.templates[][3].validation==rc.templates[][2].validation
        move_revision_pass!(rc,3,1);@test rc.selected[]==1
        delete_revision_pass!(rc,1);@test length(rc.drafts[])==2
        move_revision_pass!(rc,2,1)
        # Every source scientific option is still detached and unchanged.
        @test Hammerhead._experiment_recipe_data(source.recipe)==source_data
        new=revision_record(rc)
        @test new.input_id==source.input_id && new.pairs==source.pairs
        @test [f["path"] for f in new.input_files]==[f["path"] for f in source.input_files]
        @test isempty(new.runs) && isempty(new.record_paths)
        @test new.creation_environment["julia_version"]==string(VERSION)
        @test new.creation_environment!=source.creation_environment
        ka=ExperimentRecord([(files[1],files[2])],PIVRecipe(PIVParameters(window_size=8,overlap=4);
            backend=:ka,image_type=Float32))
        @test revision_recipe(RecipeRevisionController(ka)).backend===:ka
        destination=joinpath(dir,"revision.jld2")
        save_recipe_revision!(rc,destination;async=false)
        @test rc.state[]===:completed && rc.saved_path[]==realpath(destination)
        saved=rc.saved_record[];bytes=read(destination);preview=rc.candidate[]
        @test isempty(saved.runs) && saved.input_id==source.input_id
        @test load_experiment(destination).recipe.recipe_id==saved.recipe.recipe_id
        save_recipe_revision!(rc,()->nothing;async=false)
        @test rc.state[]===:cancelled && rc.saved_record[]===saved && rc.candidate[]===preview
        @test read(destination)==bytes
        save_recipe_revision!(rc,destination;async=false)
        @test rc.state[]===:failed && read(destination)==bytes && rc.saved_record[]===saved
        for protected in (source_path,files[1],old_output,history,report)
            preserved=read(protected)
            save_recipe_revision!(rc,protected;async=false)
            @test rc.state[]===:failed && rc.error[] isa ArgumentError
            @test read(protected)==preserved && rc.saved_record[]===saved
        end
        if Sys.iswindows()
            original_bytes=read(source_path)
            for protected in (uppercase(source_path),source_path*".",source_path*" ")
                save_recipe_revision!(rc,protected;async=false)
                @test rc.state[]===:failed && read(source_path)==original_bytes
            end
        end
        record_alias=joinpath(dir,"source-hardlink.jld2")
        Base.hardlink(source_path,record_alias)
        alias_bytes=read(source_path)
        save_recipe_revision!(rc,record_alias;async=false)
        @test rc.state[]===:failed && read(source_path)==alias_bytes && read(record_alias)==alias_bytes
        for (key,text) in ((:padding,"1"),(:n_peaks,"1.5"),(:window_size,"16"),
                           (:convergence_tol,"Inf"),(:uod_threshold,"NaN"),(:correlation_method,"unknown"),
                           (:search_area_size,"200,200"))
            old=rc.drafts[][1][key];set_revision_pass!(rc,1,Dict(key=>text))
            @test_throws ArgumentError revision_recipe(rc)
            set_revision_pass!(rc,1,Dict(key=>old))
        end
        # Requests are detached before running listeners and before path picker.
        captured=revision_recipe(rc).passes[1].max_iterations
        protected_bytes=read(report)
        observer=C.on(rc.running) do running
            running || return
            @test_throws ArgumentError apply_recipe_revision!(rc;async=false)
            rc.drafts[][1][:max_iterations]="99"
            empty!(rc.protected_paths[])
        end
        captured_path=joinpath(dir,"captured.jld2")
        save_recipe_revision!(rc,()->begin
            rc.drafts[][1][:max_iterations]="100"
            captured_path
        end;async=false)
        @test rc.saved_record[].recipe.passes[1].max_iterations==captured && rc.dirty[]
        @test rc.drafts[][1][:max_iterations]=="100" && read(report)==protected_bytes
        C.off(observer)
        # Async action returns from native callback before picker/hash/I/O work.
        set_revision_pass!(rc,1,Dict(:max_iterations=>"5"))
        async_chosen=Ref(false);async_path=joinpath(dir,"queued.jld2")
        save_recipe_revision!(rc,()->(async_chosen[]=true;async_path))
        @test rc.running[] && !async_chosen[] && !isfile(async_path)
        queued=rc.task[]
        @test queued isa Task
        # Direct next-choice mutations after scheduling cannot alter captured request.
        rc.drafts[][1][:max_iterations]="6"
        wait(queued)
        @test async_chosen[] && rc.saved_record[].recipe.passes[1].max_iterations==5
        @test !rc.running[] && rc.task[]===nothing && rc.dirty[]
        # Relative direct destination and extra protections bind to request cwd.
        other_dir=mkpath(joinpath(dir,"other cwd"))
        cd(dir) do
            rc.protected_paths[]=[basename(report)]
            listener=C.on(rc.running) do running
                running && cd(other_dir)
            end
            try
                save_recipe_revision!(rc,"cwd-bound.jld2";async=false)
                @test isfile(joinpath(dir,"cwd-bound.jld2")) && !isfile(joinpath(other_dir,"cwd-bound.jld2"))
                cd(dir)
                save_recipe_revision!(rc,()->report;async=false)
                @test rc.state[]===:failed && read(report)==protected_bytes
            finally
                C.off(listener);cd(dir)
            end
        end
        # Original protected paths are captured even if an observer clears next paths.
        rc.protected_paths[]=[report]
        observer=C.on(rc.running) do running
            running && empty!(rc.protected_paths[])
        end
        save_recipe_revision!(rc,()->report;async=false)
        @test rc.state[]===:failed && read(report)==protected_bytes
        C.off(observer)
        # Startup observers cannot strand a busy controller or replace the error.
        original_error=ErrorException("startup observer")
        observer=C.on(rc.running) do running
            running && throw(original_error)
        end
        apply_recipe_revision!(rc;async=false)
        @test !rc.running[] && rc.task[]===nothing && rc.error[]===original_error
        C.off(observer)
        # Offline inspection/preview works, saving requires unchanged original bytes.
        rm(files[3]);apply_recipe_revision!(rc;async=false)
        @test rc.state[]===:completed
        missing_path=joinpath(dir,"offline-save.jld2")
        save_recipe_revision!(rc,missing_path;async=false)
        @test rc.state[]===:failed && !isfile(missing_path)
        @test rc.saved_path[]==realpath(joinpath(dir,"cwd-bound.jld2"))
        # Mutation identity is checked before constructing a candidate.
        rc.original.recipe.mask[1,1]=!rc.original.recipe.mask[1,1]
        @test_throws ArgumentError revision_recipe(rc)
        @test source.recipe.mask==recipe.mask
    end
end

@testset "Revision script provenance and failed save notifications" begin
    C=HammerheadGUI.Controllers
    mktempdir() do dir
        a=joinpath(dir,"a.png");b=joinpath(dir,"b.png")
        save(a,Gray{N0f8}.(fill(.2,32,32)));save(b,Gray{N0f8}.(fill(.3,32,32)))
        script=joinpath(dir,"referenced script.jl")
        write(script,"error(\"this script must never run\")")
        source=ExperimentRecord([(a,b)],PIVRecipe(PIVParameters(window_size=8,overlap=4);
            external_preprocess=ScriptReference(script;entrypoint="preprocess(image)")))
        rc=RecipeRevisionController(source)
        set_revision_pass!(rc,1,Dict(:max_iterations=>"2"))
        @test revision_recipe(rc).external_preprocess==source.recipe.external_preprocess
        @test_throws ArgumentError delete_revision_pass!(rc,1)
        save_recipe_revision!(rc,script;async=false)
        @test rc.state[]===:failed && read(script,String)=="error(\"this script must never run\")"
        # Publication happened even if notification fails; identity must be retained.
        notification=ErrorException("saved notification")
        observer=C.on(rc.saved_record) do record
            record===nothing || throw(notification)
        end
        path=joinpath(dir,"saved.jld2");save_recipe_revision!(rc,path;async=false)
        @test isfile(path) && rc.saved_path[]==realpath(path) && rc.saved_record[]!==nothing
        @test rc.error[]===notification && !rc.running[]
        C.off(observer)
        write(script,"changed script bytes")
        @test revision_recipe(rc).external_preprocess.sha256==source.recipe.external_preprocess.sha256
        failed=joinpath(dir,"changed-script.jld2");save_recipe_revision!(rc,failed;async=false)
        @test rc.state[]===:failed && !isfile(failed)
    end
end
