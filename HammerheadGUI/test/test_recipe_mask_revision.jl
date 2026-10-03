using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

function mask_revision_inputs(dir;dimensions=((24,32),(24,32),(24,32)))
    files=[joinpath(dir,"mask frame $i.png") for i in eachindex(dimensions)]
    for (i,path) in enumerate(files)
        nr,nc=dimensions[i]
        save(path,Gray{N0f8}.([mod(17r+31c+7i,251)/250 for r in 1:nr,c in 1:nc]))
    end
    files
end
function mask_revision_recipe(;mask=nothing,external_preprocess=nothing)
    PIVRecipe(PIVParameters(window_size=8,overlap=4,uod_enable=false);
        preprocessing=[PreprocessStep(:invert_image)],mask,roi=ROI(5:20,7:26),
        scale=PhysicalScale(.025,.01," mm ","s"),image_type=Float32,
        threaded=false,predictor_smoothing=false,mask_threshold=.7,external_preprocess)
end
function mask_revision_release(record)
    rc=RecipeRevisionController(record);mc=RecipeMaskReferenceController()
    load_recipe_mask_reference!(mc,rc;async=false)
    old=mc.bundle[]
    refs=[WeakRef(old.raw_image),WeakRef(old.editor.image),WeakRef(old.editor.raster[]),
          WeakRef(old.mask_draft.raster),WeakRef(old.editor)]
    load_recipe_mask_reference!(mc,rc;pair_index=2,frame=:b,async=false)
    mc,refs
end

@testset "Lossless optional static-mask revision" begin
    mktempdir() do dir
        files=mask_revision_inputs(dir);bits=falses(24,32)
        bits[1,32]=true;bits[8:12,13:17].=true
        recipe=mask_revision_recipe(;mask=bits)
        record=ExperimentRecord([(files[2],files[1]),(files[1],files[1]),(files[2],files[3])],recipe)
        path=joinpath(dir,"source.jld2");save_experiment(path,record);sourcebytes=read(path)
        rc=RecipeRevisionController(path)
        @test rc.mask_draft[].enabled && rc.mask_draft[].raster==bits
        @test rc.mask_draft[].raster!==record.recipe.mask
        @test revision_recipe(rc).recipe_id==recipe.recipe_id
        apply_recipe_revision!(rc;async=false)
        set_revision_mask!(rc);@test !rc.dirty[]
        supplied=copy(bits);set_revision_mask!(rc;raster=supplied)
        supplied[2,3]=true;@test !rc.mask_draft[].raster[2,3]
        set_revision_mask!(rc;enabled=false)
        @test rc.mask_draft[].raster==bits && revision_recipe(rc).mask===nothing
        set_revision_mask!(rc;enabled=true)
        @test revision_recipe(rc).recipe_id==recipe.recipe_id
        set_revision_mask!(rc;raster=falses(24,32),enabled=true)
        empty_id=revision_recipe(rc).recipe_id
        set_revision_mask!(rc;enabled=false)
        @test revision_recipe(rc).recipe_id!=empty_id
        @test all(!,rc.mask_draft[].raster)
        reset_revision_mask!(rc);@test revision_recipe(rc).recipe_id==recipe.recipe_id
        @test_throws ArgumentError set_revision_mask!(rc;raster=zeros(24,32))
        set_revision_mask!(rc;raster=falses(16,20),enabled=false)
        @test revision_recipe(rc).mask===nothing
        set_revision_mask!(rc;enabled=true);@test_throws ArgumentError revision_recipe(rc)
        set_revision_mask!(rc;raster=nothing,enabled=true);@test_throws ArgumentError revision_recipe(rc)
        reset_revision_mask!(rc)
        set_revision_pass!(rc,1,Dict(:max_iterations=>"2"))
        set_revision_preprocess!(rc,1,Dict())
        set_revision_roi!(rc,Dict(:row_first=>"6"))
        set_revision_scale!(rc,Dict(:dt=>"0.02"))
        changed=copy(bits);changed[2,3]=true;set_revision_mask!(rc;raster=changed)
        candidate=revision_recipe(rc)
        @test candidate.mask==changed && candidate.mask_threshold==.7
        @test candidate.roi.rows==6:20 && candidate.scale.dt==.02
        @test candidate.passes[1].max_iterations==2 && candidate.preprocessing[1].operation===:invert_image
        dest=joinpath(dir,"revision.jld2");save_recipe_revision!(rc,dest;async=false)
        saved=load_experiment(dest)
        @test rc.state[]===:completed && saved.recipe.mask==changed
        @test saved.input_id==record.input_id && isempty(saved.runs)
        @test saved.pairs==record.pairs && read(path)==sourcebytes
        @test record.recipe.mask==bits
        nil=RecipeRevisionController(ExperimentRecord([(files[1],files[2])],mask_revision_recipe()))
        @test !nil.mask_draft[].enabled && nil.mask_draft[].raster===nothing
        set_revision_mask!(nil;raster=falses(24,32),enabled=true)
        @test revision_recipe(nil).mask!==nothing
        reset_revision_mask!(nil);@test revision_recipe(nil).mask===nothing
    end
end

@testset "Seed raster survives ordered drawing and morphology" begin
    bits=falses(24,32);bits[3:8,5:12].=true;bits[20,30]=true
    baseline=copy(bits);me=MaskEditor(zeros(24,32);raster=bits)
    bits[1,1]=true;@test polygon_mask(me)==baseline
    returned=polygon_mask(me);returned[2,2]=true;@test !polygon_mask(me)[2,2]
    @test_throws ArgumentError MaskEditor(zeros(24,32);raster=falses(20,20))
    @test_throws ArgumentError MaskEditor(zeros(24,32);raster=zeros(24,32))
    add=[(14.,10.),(22.,10.),(22.,18.),(14.,18.)]
    hole=[(6.,4.),(10.,4.),(10.,7.),(6.,7.)]
    me=MaskEditor(zeros(24,32);raster=baseline,polygons=[add,hole],holes=[false,true])
    expected=(baseline .| polygon_mask((24,32),add)) .& .!polygon_mask((24,32),hole)
    @test polygon_mask(me)==expected && !polygon_mask(me)[5,7]
    me.selected[]=2;delete_selected!(me)
    @test polygon_mask(me)==(baseline .| polygon_mask((24,32),add))
    me.selected[]=1;delete_selected!(me);@test polygon_mask(me)==baseline
    grow_mask!(me,1);@test polygon_mask(me)==grow_mask(baseline,1)
    enlarged=polygon_mask(me);shrink_mask!(me,1)
    @test polygon_mask(me)==shrink_mask(enlarged,1)
    clear_polygons!(me);@test me.raster[]===nothing && all(!,polygon_mask(me))
end

@testset "Captured imports and lifetime mask-source protection" begin
    mktempdir() do dir
        files=mask_revision_inputs(dir)
        record=ExperimentRecord([(files[1],files[2])],mask_revision_recipe())
        rc=RecipeRevisionController(record)
        path=joinpath(dir,"replacement mask.png")
        save(path,Gray{N0f8}.([c>16 ? .8 : .2 for r in 1:24,c in 1:32]))
        originalbytes=read(path)
        attempts=Ref(0)
        listener=on(rc.running) do busy
            busy || return
            @test_throws ArgumentError set_revision_mask!(rc;raster=falses(24,32),enabled=true)
        end
        erase=on(rc.mask_draft) do _
            empty!(rc.protected_paths[])
        end
        load_revision_mask!(rc,()->(attempts[]+=1;path);threshold=.5,invert=true,async=false)
        off(listener);off(erase)
        @test attempts[]==1 && rc.state[]===:completed && !rc.running[]
        @test rc.mask_draft[].enabled && rc.mask_draft[].raster==load_mask(path;threshold=.5,invert=true)
        @test isempty(rc.protected_paths[])
        save_recipe_revision!(rc,path;async=false)
        @test rc.state[]===:failed && read(path)==originalbytes
        @test occursin("protected",sprint(showerror,rc.error[]))
        prior=deepcopy(rc.mask_draft[])
        load_revision_mask!(rc,()->nothing;async=false)
        @test rc.state[]===:cancelled && rc.mask_draft[]==prior
        @test_throws ArgumentError load_revision_mask!(rc,path;threshold=NaN)
        @test_throws ArgumentError load_revision_mask!(rc,path;threshold=Inf)
        wrong=joinpath(dir,"cropped.png");save(wrong,Gray{N0f8}.(zeros(16,20)))
        load_revision_mask!(rc,wrong;async=false)
        @test rc.state[]===:failed && rc.mask_draft[]==prior
        load_revision_mask!(rc,()->begin
            rc.mask_draft[]=(enabled=false,raster=copy(prior.raster));path
        end;async=false)
        @test rc.state[]===:failed && !rc.mask_draft[].enabled
        @test occursin("changed",sprint(showerror,rc.error[]))
        mixed=ExperimentRecord([(files[1],files[2]),(files[3],files[3])],mask_revision_recipe())
        mixed.input_files[end]["image_size"]=[20,28]
        # Rebuild a valid identity for this metadata-only dimension fixture.
        mixed=ExperimentRecord(mixed.recipe,mixed.input_files,mixed.pairs,
            Hammerhead._experiment_digest(Hammerhead._experiment_input_data(mixed.input_files,mixed.pairs)),
            mixed.creation_environment,mixed.runs,mixed.record_paths)
        multi=RecipeRevisionController(mixed)
        load_revision_mask!(multi,path;async=false)
        @test multi.state[]===:failed && multi.mask_draft[].raster===nothing
        startup=ErrorException("mask loading startup")
        listener=on(rc.running) do busy;busy && throw(startup);end
        load_revision_mask!(rc,path;async=false);off(listener)
        @test rc.error[]===startup && !rc.running[] && rc.task[]===nothing
        # Queued import does not call a native picker in the invoking callback.
        queued=RecipeRevisionController(record);picked=Ref(false)
        load_revision_mask!(queued,()->(picked[]=true;path))
        @test queued.running[] && !picked[]
        wait(queued.task[])
        @test picked[] && queued.state[]===:completed && !queued.running[]
    end
end

@testset "Verified raw references and captured mask applications" begin
    mktempdir() do dir
        files=mask_revision_inputs(dir);bits=falses(24,32);bits[8:12,13:17].=true
        recipe=mask_revision_recipe(;mask=bits)
        record=ExperimentRecord([(files[2],files[1]),(files[1],files[3])],recipe)
        rc=RecipeRevisionController(record);mc=RecipeMaskReferenceController()
        observer=on(mc.running) do busy
            busy || return
            set_revision_roi!(rc,Dict(:row_first=>"6"))
            set_revision_mask!(rc;enabled=false)
        end
        load_recipe_mask_reference!(mc,rc;pair_index=2,frame=:b,async=false);off(observer)
        bundle=mc.bundle[]
        @test mc.state[]===:completed && !mc.running[] && mc.task[]===nothing
        @test bundle.input_id==record.input_id && bundle.recipe_id==recipe.recipe_id
        @test bundle.pair_index==2 && bundle.frame===:b && bundle.input_descriptor==record.input_files[record.pairs[2][2]]
        @test bundle.raw_image==load_image(Float32,files[3]) && bundle.raw_image isa Matrix{Float32}
        @test bundle.raw_image!=invert_image(bundle.raw_image)
        @test size(bundle.raw_image)==(24,32) && bundle.roi.rows==5:20
        @test bundle.editor.raster[]==bits && bundle.mask_draft.enabled
        @test_throws ArgumentError apply_revision_mask!(rc,mc;async=false)
        load_recipe_mask_reference!(mc,rc;async=false)
        @test !mc.bundle[].mask_draft.enabled && mc.bundle[].editor.raster[]==bits
        @test revision_recipe(rc).mask===nothing
        add_vertex!(mc.bundle[].editor,2,2)
        @test_throws ArgumentError apply_revision_mask!(rc,mc;async=false)
        undo_vertex!(mc.bundle[].editor)
        set_revision_pass!(rc,1,Dict(:max_iterations=>"2"))
        clear_polygons!(mc.bundle[].editor)
        apply_revision_mask!(rc,mc;async=false)
        @test rc.state[]===:completed && rc.mask_draft[].enabled && all(!,rc.mask_draft[].raster)
        @test rc.candidate[].passes[1].max_iterations==2 && rc.candidate[].mask_threshold==.7
        @test mc.bundle[].roi.rows==6:20 && mc.bundle[].recipe_id!=rc.candidate[].recipe_id
        for xy in ((2,2),(5,2),(5,5),(2,5));add_vertex!(mc.bundle[].editor,xy...);end
        close_active!(mc.bundle[].editor)
        apply_revision_mask!(rc,mc;async=false)
        @test rc.mask_draft[].raster[3,3] && !rc.running[] && !mc.running[]
        # A same-size, same-mask reference must not apply to another experiment.
        different=RecipeRevisionController(ExperimentRecord([(files[3],files[3])],recipe))
        set_revision_mask!(different;raster=copy(rc.mask_draft[].raster),enabled=true)
        @test_throws ArgumentError apply_revision_mask!(different,mc;async=false)
        changed_recipe=mask_revision_recipe(;mask=rc.mask_draft[].raster)
        other=RecipeRevisionController(ExperimentRecord([(files[2],files[1]),(files[1],files[3])],changed_recipe))
        @test_throws ArgumentError apply_revision_mask!(other,mc;async=false)
        before=mc.bundle[]
        write(files[3],"changed reference bytes")
        load_recipe_mask_reference!(mc,rc;pair_index=2,frame=:b,async=false)
        @test mc.state[]===:failed && mc.bundle[]===before
        @test_throws ArgumentError load_recipe_mask_reference!(mc,rc;frame=:c)
        @test_throws ArgumentError load_recipe_mask_reference!(mc,rc;pair_index=true)
        # Apply is a detached edit; saving is where ALL original bytes are checked.
        apply_revision_mask!(rc,mc;async=false);@test rc.state[]===:completed
        output=joinpath(dir,"changed-input-save.jld2");save_recipe_revision!(rc,output;async=false)
        @test rc.state[]===:failed && !isfile(output)
        script=joinpath(dir,"never execute.jl");write(script,"error(\"must not execute\")")
        scripted=ExperimentRecord([(files[2],files[1])],mask_revision_recipe(;
            external_preprocess=ScriptReference(script;entrypoint="condition")))
        src=RecipeRevisionController(scripted)
        load_recipe_mask_reference!(mc,src;async=false)
        @test mc.state[]===:completed && mc.bundle[].raw_image==load_image(Float32,files[2])
        @test mc.bundle[].editor.raster[]===nothing && revision_recipe(src).mask===nothing
        error=ErrorException("raw-reference publication")
        listener=on(mc.bundle) do _;throw(error);end
        load_recipe_mask_reference!(mc,src;frame=:b,async=false);off(listener)
        @test mc.error[]===error && mc.state[]===:failed && !mc.running[]
        @test mc.bundle[].frame===:b && mc.bundle[].raw_image==load_image(Float32,files[1])
        error=ErrorException("mask apply startup")
        listener=on(rc.running) do busy;busy && throw(error);end
        # Reload the correct unchanged first frame into this revision first.
        load_recipe_mask_reference!(mc,rc;async=false)
        apply_revision_mask!(rc,mc;async=false);off(listener)
        @test rc.error[]===error && !rc.running[] && !mc.running[] && rc.task[]===nothing && mc.task[]===nothing
    end
end

@testset "Superseded mask reference release" begin
    mktempdir() do dir
        files=mask_revision_inputs(dir);bits=falses(24,32);bits[3:7,5:9].=true
        record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],mask_revision_recipe(;mask=bits))
        retained,refs=mask_revision_release(record)
        GC.gc(true);GC.gc(true)
        @test all(ref->ref.value===nothing,refs)
        @test retained.bundle[].pair_index==2 && retained.bundle[].frame===:b
        @test retained.task[]===nothing && !retained.running[]
    end
end
