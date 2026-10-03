# Explicit, framework-free image conditioning. No PIV execution/result cache.

"""
    RecipeImagePreviewController()

Hold only the latest successful selected-pair raw/processed image bundle and its
captured recipe/input/pair identities. Draft edits and failures before publication
do not relabel or replace that bundle. A throwing publication observer can leave
the newly published bundle visible; notification failures do not roll it back.
`bundle` arrays are detached, publicly mutable
snapshots, not an ongoing proof of file or display integrity.
"""
struct RecipeImagePreviewController
    bundle::Observable{Union{Nothing,NamedTuple}}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    error::Observable{Any}
    task::Base.RefValue{Union{Nothing,Task}}
end
RecipeImagePreviewController()=RecipeImagePreviewController(
    Observable{Union{Nothing,NamedTuple}}(nothing),Observable(false),Observable(:ready),
    Observable("No image preview. Choose an original pair and preview explicitly."),
    Observable{Any}(nothing),Ref{Union{Nothing,Task}}(nothing))

function _recipe_image_failure!(pc,err)
    _revision_safe_set!(pc.error,err);_revision_safe_set!(pc.state,:failed)
    _revision_safe_set!(pc.status,"Image preview failed: "*sprint(showerror,err)*
        ". Any displayed image bundle keeps its captured identity; notification errors do not roll back published values.")
end
function _recipe_image_run!(pc,request,pair_index)
    try
        recipe=_revision_recipe(request)
        recipe.external_preprocess===nothing || throw(ArgumentError(
            "Referenced external preprocessing scripts are inspectable, but numerical image preview never executes them."))
        source=request.original;pair=source.pairs[pair_index]
        files=deepcopy(source.input_files[pair])
        raw_a=Hammerhead._experiment_load_image(files[1],recipe.image_type)
        raw_b=Hammerhead._experiment_load_image(files[2],recipe.image_type)
        preprocess=Hammerhead._experiment_preprocess(recipe,nothing)
        processed_a=preprocess(copy(raw_a));processed_b=preprocess(copy(raw_b))
        # Processing can take time: bind both images to unchanged acquisition bytes
        # at completion without loading a second pair or retaining all acquisitions.
        for f in files
            isfile(f["path"]) && filesize(f["path"])==f["size_bytes"] &&
                Hammerhead._experiment_file_digest(f["path"])==f["sha256"] ||
                throw(ArgumentError("selected experiment input changed during image preview: $(f["path"])"))
        end
        bundle=(raw_a=raw_a,raw_b=raw_b,processed_a=processed_a,processed_b=processed_b,
            roi=recipe.roi,mask=recipe.mask===nothing ? nothing : copy(recipe.mask),
            recipe_id=recipe.recipe_id,source_recipe_id=source.recipe.recipe_id,input_id=source.input_id,
            pair_index=pair_index,input_paths=String[f["path"] for f in files],
            input_descriptors=files,image_type=recipe.image_type)
        pc.bundle[]=bundle
        pc.state[]=:completed;pc.status[]="Selected pair conditioned in saved precision on full frames; ROI/mask are overlays. No PIV or accuracy assessment."
    catch err
        _recipe_image_failure!(pc,err)
    finally
        pc.task[]=nothing;_revision_safe_set!(pc.running,false)
    end
    nothing
end

"""
    preview_recipe_images!(preview, revision; pair_index=1, async=true)

Capture the complete revision drafts, source and ordered pair index before any
observer notification. Validate this captured recipe and verify the selected
original input bytes using the exact core loader, then run the replay built-ins
in `recipe.image_type` on full images **before** ROI cropping. Any referenced
script refuses numerical preview. Metadata-only `apply_recipe_revision!` remains
separate and works without acquisition files.

The detached bundle contains `raw_a`, `raw_b`, `processed_a`, `processed_b`,
full-image-coordinate `roi`/`mask`, `recipe_id`, `source_recipe_id`, `input_id`,
`pair_index`, `input_paths`, `input_descriptors`, and `image_type`. Only one
current bundle is retained; input-file integrity is checked again after processing.
No vector, uncertainty, stationarity or accuracy claim follows from these images.
Queued work yields outside native callbacks but can pause rendering. No cancellation.
"""
function preview_recipe_images!(pc::RecipeImagePreviewController,rc::RecipeRevisionController;
                               pair_index=1,async::Bool=true)
    pc.running[] && throw(ArgumentError("image preview is busy; wait for it to finish"))
    request=_revision_capture(rc)
    pair_index isa Integer && !(pair_index isa Bool) && 1<=pair_index<=length(request.original.pairs) ||
        throw(ArgumentError("pair index is outside the original ordered acquisition pairs"))
    index=Int(pair_index)
    try
        pc.running[]=true;pc.error[]=nothing;pc.state[]=:busy
        pc.status[]="Verifying and conditioning the captured pair; cancellation is unavailable."
    catch err
        _recipe_image_failure!(pc,err);_revision_safe_set!(pc.running,false);return pc
    end
    if async
        pc.task[]=errormonitor(@async begin yield();_recipe_image_run!(pc,request,index);end)
    else
        _recipe_image_run!(pc,request,index)
    end
    pc
end
