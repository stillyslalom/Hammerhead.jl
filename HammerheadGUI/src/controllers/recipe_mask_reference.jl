# One explicitly loaded raw full-frame reference and a detached editing session.

"""
    RecipeMaskReferenceController()

Hold the latest successful raw-reference/editor bundle and its captured recipe,
input, ordered pair and frame identities. Loading verifies original input bytes;
it never conditions pixels, runs PIV or executes referenced scripts. Draft edits
and prepublication failures retain the previous bundle without relabeling it.
A throwing publication observer can leave newly published values visible.
Arrays are detached mutable snapshots, not an ongoing display/file proof.
"""
struct RecipeMaskReferenceController
    bundle::Observable{Union{Nothing,NamedTuple}}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    error::Observable{Any}
    task::Base.RefValue{Union{Nothing,Task}}
    baseline::Base.RefValue{Union{Nothing,NamedTuple}}
end
RecipeMaskReferenceController()=RecipeMaskReferenceController(
    Observable{Union{Nothing,NamedTuple}}(nothing),Observable(false),Observable(:ready),
    Observable("Choose an original pair/frame and load a raw reference explicitly."),
    Observable{Any}(nothing),Ref{Union{Nothing,Task}}(nothing),Ref{Union{Nothing,NamedTuple}}(nothing))

function _recipe_mask_failure!(mc,err)
    _revision_safe_set!(mc.error,err);_revision_safe_set!(mc.state,:failed)
    _revision_safe_set!(mc.status,"Mask reference/action failed: "*sprint(showerror,err)*
        ". Displayed reference keeps its captured identity; notification errors do not roll back published values.")
end
function _recipe_mask_reference_run!(mc,request,index,frame)
    try
        recipe=_revision_recipe(request)
        source=request.original;descriptor=deepcopy(source.input_files[source.pairs[index][frame===:a ? 1 : 2]])
        raw=Hammerhead._experiment_load_image(descriptor,recipe.image_type)
        editor=MaskEditor(raw;raster=request.mask.raster)
        # Include the display-copy work in the verified loading boundary.
        isfile(descriptor["path"]) && filesize(descriptor["path"])==descriptor["size_bytes"] &&
            Hammerhead._experiment_file_digest(descriptor["path"])==descriptor["sha256"] ||
            throw(ArgumentError("original input changed during raw reference loading"))
        bundle=(editor=editor,raw_image=raw,roi=recipe.roi,mask_draft=deepcopy(request.mask),
            recipe_id=recipe.recipe_id,source_recipe_id=source.recipe.recipe_id,input_id=source.input_id,
            pair_index=index,frame=frame,input_descriptor=descriptor)
        mc.baseline[]=deepcopy(request.mask)
        mc.bundle[]=bundle
        mc.state[]=:completed;mc.status[]="Verified raw full-frame reference loaded; no preprocessing or scripts executed."
    catch err
        _recipe_mask_failure!(mc,err)
    finally
        mc.task[]=nothing;_revision_safe_set!(mc.running,false)
    end
    nothing
end

"""
    load_recipe_mask_reference!(reference, revision; pair_index=1,
                                frame=:a, async=true)

Capture all revision drafts and one original ordered pair/frame before observers.
Validate the complete recipe metadata, then load that single raw full image in
saved precision with original file-byte/dimension checks before and after loading.
No preprocessing or scripts are executed, even when recorded in the recipe.

The bundle contains `editor`, `raw_image`, `roi`, `mask_draft`, `recipe_id`,
`source_recipe_id`, `input_id`, `pair_index`, `frame` and `input_descriptor`.
Its editor copies the retained full-image raster even when the draft is disabled;
loading an absent mask leaves it absent until explicit Apply. The captured ROI is
an overlay, never a crop. Only the current bundle is retained. Queued work may
pause rendering; it is not cancellable. Later mask application does not reverify
the reference file; saving the new record verifies all original input bytes.
"""
function load_recipe_mask_reference!(mc::RecipeMaskReferenceController,rc::RecipeRevisionController;
                                     pair_index=1,frame::Symbol=:a,async::Bool=true)
    mc.running[] && throw(ArgumentError("mask reference is busy; wait for it to finish"))
    request=_revision_capture(rc)
    pair_index isa Integer && !(pair_index isa Bool) && 1<=pair_index<=length(request.original.pairs) ||
        throw(ArgumentError("pair index is outside the original ordered pairs"))
    frame in (:a,:b) || throw(ArgumentError("frame must be :a or :b"))
    index=Int(pair_index)
    try
        mc.running[]=true;mc.error[]=nothing;mc.state[]=:busy
        mc.status[]="Verifying captured raw reference; cancellation is unavailable."
    catch err
        _recipe_mask_failure!(mc,err);_revision_safe_set!(mc.running,false);return mc
    end
    if async
        mc.task[]=errormonitor(@async begin yield();_recipe_mask_reference_run!(mc,request,index,frame);end)
    else
        _recipe_mask_reference_run!(mc,request,index,frame)
    end
    mc
end

function _revision_mask_apply_run!(rc,mc,request,bundle,baseline,draft)
    try
        updated=merge(request,(mask=draft,))
        recipe=_revision_recipe(updated);difference=recipe_diff(request.original.recipe,recipe)
        mc.bundle[]===bundle && _revision_mask_equal(mc.baseline[],baseline) ||
            throw(ArgumentError("mask reference session changed before application"))
        _revision_mask_equal(rc.mask_draft[],baseline) ||
            throw(ArgumentError("mask draft changed since this reference was seeded; reload the reference"))
        # Publish the baseline coherently with the new bits before listeners.
        mc.baseline[]=deepcopy(draft)
        _revision_publish_mask!(rc,draft)
        _revision_publish_preview!(rc,recipe,difference,updated)
        rc.state[]=:completed;rc.status[]="Edited full-image raster applied; no record saved or PIV run."
        mc.state[]=:completed;mc.status[]="Raster applied. Reference image/ROI keep their captured loading identity."
    catch err
        _revision_failure!(rc,err);_recipe_mask_failure!(mc,err)
    finally
        rc.task[]=nothing;mc.task[]=nothing
        _revision_safe_set!(rc.running,false);_revision_safe_set!(mc.running,false)
    end
    nothing
end

"""
    apply_revision_mask!(revision, reference; enabled=true, async=true)

Capture the editor's combined raster and complete current revision before any
observer. Refuse unfinished drawing and a mask whose enabled flag or bits differ
from the session baseline; replacing then restoring identical content is allowed.
Non-mask draft edits are composed normally. Validate every original full-frame
size and pass/ROI before publishing. By default this explicitly enables the edited
raster, including all-false; `enabled=false` retains edited bits as disabled.
Successful application advances the session baseline for subsequent edits.

Loading identity and ROI remain those of the reference snapshot, not an assertion
that the displayed image matches later draft settings. Apply reads no pixels or
scripts and does not reverify files; save verifies original inputs. It does not
infer polygons or alter `mask_threshold`. Clear-all in the editor applies an empty
raster; `reset_revision_mask!` separately restores the imported optional mask.
Queued validation may pause rendering; there is no processing cancellation.
"""
function apply_revision_mask!(rc::RecipeRevisionController,mc::RecipeMaskReferenceController;
                              enabled::Bool=true,async::Bool=true)
    _revision_idle(rc)
    mc.running[] && throw(ArgumentError("mask reference is busy; wait for it to finish"))
    bundle=mc.bundle[]
    bundle===nothing && throw(ArgumentError("load a raw reference before applying edits"))
    isempty(bundle.editor.active[]) || throw(ArgumentError("finish or cancel the active polygon before applying"))
    baseline=deepcopy(mc.baseline[])
    baseline===nothing && throw(ArgumentError("mask reference has no editing baseline"))
    _revision_mask_equal(rc.mask_draft[],baseline) ||
        throw(ArgumentError("mask draft changed since this reference was seeded; reload the reference"))
    request=_revision_capture(rc)
    request.original.input_id==bundle.input_id && request.original.recipe.recipe_id==bundle.source_recipe_id ||
        throw(ArgumentError("mask reference belongs to another original experiment"))
    draft=(enabled=enabled,raster=polygon_mask(bundle.editor))
    try
        rc.running.val=true;mc.running.val=true
        notify(rc.running);notify(mc.running)
        rc.error[]=nothing;mc.error[]=nothing;rc.state[]=:busy;mc.state[]=:busy
        rc.status[]="Validating captured mask edit; cancellation is unavailable."
    catch err
        _revision_failure!(rc,err);_recipe_mask_failure!(mc,err)
        _revision_safe_set!(rc.running,false);_revision_safe_set!(mc.running,false);return rc
    end
    if async
        task=errormonitor(@async begin yield();_revision_mask_apply_run!(rc,mc,request,bundle,baseline,draft);end)
        rc.task[]=task;mc.task[]=task
    else
        _revision_mask_apply_run!(rc,mc,request,bundle,baseline,draft)
    end
    rc
end
