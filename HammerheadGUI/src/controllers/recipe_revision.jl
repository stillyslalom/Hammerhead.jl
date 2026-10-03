# Lossless pass editing; no native widgets, pixel/result cache or script execution.

"""
    revision_fields()

Return a detached ordered editor schema (`key`, `label`, `kind`, `group`) for
the eighteen non-validation fields of `PIVParameters`. Ordered validation
tuples remain unchanged. Text values are parsed without evaluating Julia code.
"""
revision_fields() = [
    (key=:window_size,label="Window size",kind=:tuple,group=:geometry),
    (key=:search_area_size,label="Search area size",kind=:tuple,group=:geometry),
    (key=:overlap,label="Overlap",kind=:tuple,group=:geometry),
    (key=:correlation_method,label="Correlation method",kind=:symbol,group=:correlation),
    (key=:padding,label="Padding",kind=:bool,group=:correlation),
    (key=:apodization,label="Apodization",kind=:symbol,group=:correlation),
    (key=:subpixel_method,label="Subpixel method",kind=:symbol,group=:correlation),
    (key=:n_peaks,label="Number of peaks",kind=:int,group=:correlation),
    (key=:peak_finder,label="Peak finder",kind=:symbol,group=:correlation),
    (key=:uncertainty,label="Uncertainty",kind=:bool,group=:validation),
    (key=:uod_enable,label="UOD enabled",kind=:bool,group=:validation),
    (key=:uod_threshold,label="UOD threshold",kind=:real,group=:validation),
    (key=:uod_neighborhood,label="UOD neighborhood",kind=:int,group=:validation),
    (key=:min_peak_ratio,label="Minimum peak ratio",kind=:real,group=:validation),
    (key=:replace_outliers,label="Replace outliers",kind=:bool,group=:validation),
    (key=:max_iterations,label="Maximum iterations",kind=:int,group=:iteration),
    (key=:convergence_tol,label="Convergence tolerance",kind=:real,group=:iteration),
    (key=:keep_correlation_planes,label="Keep correlation planes",kind=:bool,group=:iteration)]

_revision_text(value::Tuple) = join(value,", ")
_revision_text(value) = string(value)
_revision_draft(pass) = Dict(f.key=>_revision_text(getfield(pass,f.key)) for f in revision_fields())

"""
    RecipeRevisionController(record_or_path; protected_paths=[])

Keep a detached complete planar experiment and ordered raw pass text drafts.
All non-pass recipe fields and each pass's ordered validation tuple are retained.
`candidate`, `diff` and `preview_text` describe the last successful preview;
`dirty` marks subsequent edits, including invalid text. `saved_record` and
`saved_path` retain the last successfully written revision independently of drafts.

Extra known output/history/report destinations must be supplied in
`protected_paths`: arbitrary external history paths cannot be recovered from a
record. Original record/input/script/run output paths are protected automatically.
All revision destinations successfully saved by this controller are also protected
for its lifetime; subsequent saves require another destination.
Construction validates metadata and hashes embedded recipe arrays; launch outside
native callbacks. Preview/save asynchronous scheduling can still pause rendering;
processing cancellation is unavailable. No parent-recipe lineage is persisted.
"""
struct RecipeRevisionController
    original::ExperimentRecord
    drafts::Observable{Vector{Dict{Symbol,String}}}
    templates::Observable{Vector{PIVParameters}}
    selected::Observable{Int}
    candidate::Observable{Union{Nothing,PIVRecipe}}
    diff::Observable{Union{Nothing,RecipeDiff}}
    preview_text::Observable{String}
    dirty::Observable{Bool}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    error::Observable{Any}
    saved_record::Observable{Union{Nothing,ExperimentRecord}}
    saved_path::Observable{String}
    protected_paths::Observable{Vector{String}}
    task::Base.RefValue{Union{Nothing,Task}}
    saved_protected::Base.RefValue{Vector{String}}
end
function RecipeRevisionController(value;protected_paths=String[])
    record=value isa AbstractString ? load_experiment(value) : value
    record isa ExperimentRecord || throw(ArgumentError("pass revision needs a planar ExperimentRecord or path"))
    original=deepcopy(record);_validate_gui_experiment(original)
    all(p->p isa AbstractString,protected_paths) || throw(ArgumentError("protected_paths must contain paths"))
    paths=String[Hammerhead._artifact_local_path(p) for p in protected_paths]
    RecipeRevisionController(original,Observable(_revision_draft.(original.recipe.passes)),
        Observable(deepcopy(original.recipe.passes)),Observable(1),
        Observable{Union{Nothing,PIVRecipe}}(nothing),Observable{Union{Nothing,RecipeDiff}}(nothing),
        Observable("No preview yet."),Observable(true),Observable(false),Observable(:ready),
        Observable("Edit passes, preview, then save a distinct experiment record."),Observable{Any}(nothing),
        Observable{Union{Nothing,ExperimentRecord}}(nothing),Observable(""),Observable(paths),
        Ref{Union{Nothing,Task}}(nothing),Ref(String[]))
end
_revision_idle(rc)=rc.running[] && throw(ArgumentError("recipe revision is busy; wait for it to finish"))
function _revision_index(rc,index;insertion=false)
    index isa Integer && !(index isa Bool) && 1<=index<=length(rc.drafts[])+(insertion ? 1 : 0) ||
        throw(ArgumentError("pass index is outside the draft sequence"))
    Int(index)
end
function _revision_edit!(rc,drafts,templates,selected)
    # Coherent state before listeners; draft/template changes never parse text.
    rc.drafts.val=drafts;rc.templates.val=templates;rc.selected.val=selected;rc.dirty.val=true
    for obs in (rc.drafts,rc.templates,rc.selected,rc.dirty);notify(obs);end
    rc
end

"""
    set_revision_pass!(controller, index, draft)

Merge raw string fields into one pass. Invalid text is retained for correction;
unknown fields and validation-tuple edits are refused. Saving parses every row,
including rows not currently visible, rather than using a prior valid preview.
"""
function set_revision_pass!(rc::RecipeRevisionController,index,draft::AbstractDict)
    _revision_idle(rc);i=_revision_index(rc,index)
    keys_allowed=Set(f.key for f in revision_fields())
    all(k->k isa Symbol && k in keys_allowed,keys(draft)) || throw(ArgumentError("unknown or read-only pass field"))
    all(v->v isa AbstractString,values(draft)) || throw(ArgumentError("pass drafts must contain text"))
    rows=deepcopy(rc.drafts[]);merge!(rows[i],Dict(k=>String(v) for (k,v) in draft))
    rows==rc.drafts[] && return rc
    _revision_edit!(rc,rows,copy(rc.templates[]),rc.selected[])
end

"""
    insert_revision_pass!(controller, index; source=controller.selected[])

Insert a copy of a complete raw pass draft and its unchanged validation tuple,
and select the new row. Invalid raw text remains invalid after copying.
"""
function insert_revision_pass!(rc::RecipeRevisionController,index;source=rc.selected[])
    _revision_idle(rc);i=_revision_index(rc,index;insertion=true);s=_revision_index(rc,source)
    rows=deepcopy(rc.drafts[]);templates=deepcopy(rc.templates[])
    insert!(rows,i,deepcopy(rows[s]));insert!(templates,i,deepcopy(templates[s]))
    _revision_edit!(rc,rows,templates,i)
end

"""
    move_revision_pass!(controller, from, to)

Move a complete raw draft and validation template together, keeping selection
attached to the same row. Pass order is scientific recipe content.
"""
function move_revision_pass!(rc::RecipeRevisionController,from,to)
    _revision_idle(rc);a=_revision_index(rc,from);b=_revision_index(rc,to)
    a==b && return rc
    rows=deepcopy(rc.drafts[]);templates=copy(rc.templates[])
    insert!(rows,b,popat!(rows,a));insert!(templates,b,popat!(templates,a))
    s=rc.selected[];s=s==a ? b : a<s<=b ? s-1 : b<=s<a ? s+1 : s
    _revision_edit!(rc,rows,templates,s)
end

"""
    delete_revision_pass!(controller, index)

Delete one ordered draft/template pair. At least one pass must remain.
"""
function delete_revision_pass!(rc::RecipeRevisionController,index)
    _revision_idle(rc);i=_revision_index(rc,index)
    length(rc.drafts[])>1 || throw(ArgumentError("a recipe needs at least one pass"))
    rows=deepcopy(rc.drafts[]);templates=copy(rc.templates[]);deleteat!(rows,i);deleteat!(templates,i)
    s=rc.selected[];s=s>i ? s-1 : min(s,length(rows))
    _revision_edit!(rc,rows,templates,s)
end

function _revision_parse(text,kind,key)
    value=strip(text)
    try
        if kind===:tuple
            startswith(value,"(") && endswith(value,")") && (value=strip(value[2:end-1]))
            parts=split(value,',');length(parts)==2 || throw(ArgumentError("needs two integers"))
            return (parse(Int,strip(parts[1])),parse(Int,strip(parts[2])))
        elseif kind===:bool
            value in ("true","false") || throw(ArgumentError("needs true or false"))
            return value=="true"
        elseif kind===:symbol
            startswith(value,":") && (value=value[2:end]);return Symbol(value)
        elseif kind===:int
            return parse(Int,value)
        else
            x=parse(Float64,value);isfinite(x) || throw(ArgumentError("needs a finite number"));return x
        end
    catch err
        throw(ArgumentError("Invalid $key text $(repr(text)): $(sprint(showerror,err))"))
    end
end
function _revision_capture(rc)
    protected=String[Hammerhead._artifact_local_path(p) for p in rc.protected_paths[]]
    append!(protected,rc.saved_protected[])
    saved=rc.saved_record[]
    if saved!==nothing
        append!(protected,saved.record_paths);append!(protected,[run.output for run in saved.runs])
    end
    (original=deepcopy(rc.original),drafts=deepcopy(rc.drafts[]),templates=deepcopy(rc.templates[]),
     protected=protected)
end
function _revision_recipe(request)
    original=request.original;_validate_gui_experiment(original)
    length(request.drafts)==length(request.templates)>0 || throw(ArgumentError("draft/template sequence changed"))
    fields=revision_fields();allowed=Set(f.key for f in fields)
    passes=PIVParameters[]
    for (row,template) in zip(request.drafts,request.templates)
        Set(keys(row))==allowed || throw(ArgumentError("draft fields do not match the complete editor schema"))
        options=Dict(k=>getfield(template,k) for k in fieldnames(PIVParameters))
        for f in fields;options[f.key]=_revision_parse(row[f.key],f.kind,f.key);end
        push!(passes,PIVParameters(;options...))
    end
    source=original.recipe
    options=Dict(k=>getfield(source,k) for k in fieldnames(PIVRecipe) if k ∉ (:passes,:recipe_id))
    recipe=PIVRecipe(passes;options...)
    # Metadata-only geometry validation; do not claim a fresh record/environment.
    scratch=ExperimentRecord(recipe,deepcopy(original.input_files),deepcopy(original.pairs),
        original.input_id,deepcopy(original.creation_environment),ExperimentRun[],String[])
    Hammerhead._experiment_preflight(scratch)
    recipe
end

"""
    revision_recipe(controller) -> PIVRecipe

Synchronously validate all raw drafts and original metadata, returning a detached
candidate preserving every non-pass field. Hashes embedded arrays; does not read
input pixels or evaluate referenced scripts. Invalid text never uses a prior preview.
Floating point text is parsed as Float64, matching the stored parameter precision.
"""
revision_recipe(rc::RecipeRevisionController)=_revision_recipe(_revision_capture(rc))

"""
    revision_diff(controller) -> RecipeDiff

Synchronously compare the complete original recipe with freshly parsed drafts.
Reports describe changed settings, not accuracy or numerical result differences.
"""
function revision_diff(rc::RecipeRevisionController)
    request=_revision_capture(rc);recipe_diff(request.original.recipe,_revision_recipe(request))
end
function _revision_record(request,recipe)
    source=request.original
    # Verify bytes against the ORIGINAL descriptors before constructing new ones.
    Hammerhead._experiment_preflight(source;verify_files=true)
    pairs=[(source.input_files[p[1]]["path"],source.input_files[p[2]]["path"]) for p in source.pairs]
    record=ExperimentRecord(pairs,recipe)
    record.input_id==source.input_id || throw(ArgumentError("original inputs changed; revision input identity differs"))
    isempty(record.runs) && isempty(record.record_paths) || error("fresh revision inherited history")
    record
end

"""
    revision_record(controller) -> ExperimentRecord

Synchronously build a fresh record using the public constructor and the exact
original ordered pairs. Verify original input/script bytes, preserve `input_id`,
and capture the current creation environment. Runs and record paths start empty.
This hashes/decodes image files and can be expensive; never call from native UI
callbacks. Referenced scripts are verified as content but never evaluated.
"""
function revision_record(rc::RecipeRevisionController)
    request=_revision_capture(rc);_revision_record(request,_revision_recipe(request))
end

function _revision_guard(path,request)
    isempty(strip(path)) && throw(ArgumentError("choose a new revision destination"))
    destination=Hammerhead._artifact_local_path(path);source=request.original
    paths=String[request.protected...;source.record_paths...;[f["path"] for f in source.input_files]...;
        [run.output for run in source.runs]...]
    source.recipe.external_preprocess===nothing || push!(paths,source.recipe.external_preprocess.path)
    for p in paths
        isempty(p) && continue
        # Foreign historical locators are provenance, not current-host files.
        Hammerhead._artifact_absolute_locator(p) && !Hammerhead._artifact_local_locator(p) && continue
        Hammerhead._experiment_alias(destination,p) && throw(ArgumentError("revision destination aliases a protected original/input/output/history path: $p"))
    end
    destination
end
function _revision_safe_set!(obs,value)
    try obs[]=value catch;end
end
function _revision_failure!(rc,err;saved=false)
    _revision_safe_set!(rc.error,err);_revision_safe_set!(rc.state,:failed)
    _revision_safe_set!(rc.status,(saved ? "Revision saved, but notification failed: " : "Revision failed: ")*
        sprint(showerror,err)*". Displayed preview and saved identity remain explicit; notification errors do not roll back published values.")
end
function _revision_publish_preview!(rc,recipe,difference,request)
    rc.candidate.val=recipe;rc.diff.val=difference;rc.preview_text.val=sprint(show,MIME"text/plain"(),difference)
    rc.dirty.val=rc.drafts[]!=request.drafts || rc.templates[]!=request.templates
    for obs in (rc.candidate,rc.diff,rc.preview_text,rc.dirty);notify(obs);end
end
function _revision_run!(rc,request,destination)
    saved=false
    try
        recipe=_revision_recipe(request);difference=recipe_diff(request.original.recipe,recipe)
        if destination!==nothing
            chosen=destination isa Function ? destination() : destination
            if chosen===nothing || (chosen isa AbstractString && isempty(chosen))
                rc.state[]=:cancelled;rc.status[]="Save choice cancelled; prior saved revision and preview retained."
                return nothing
            end
            chosen isa AbstractString || throw(ArgumentError("save picker must return a path or nothing"))
            path=_revision_guard(String(chosen),request)
            record=_revision_record(request,recipe)
            # Recheck immediately before writer opens destination.
            _revision_guard(path,request);save_experiment(path,record);saved=true
            rc.saved_record.val=record;rc.saved_path.val=realpath(path)
            push!(rc.saved_protected[],rc.saved_path[])
            notify(rc.saved_record);notify(rc.saved_path)
        end
        _revision_publish_preview!(rc,recipe,difference,request)
        rc.state[]=:completed;rc.status[]=saved ? "Revision saved as a fresh record; original history unchanged." : "Preview updated; no inputs processed or record saved."
    catch err
        _revision_failure!(rc,err;saved)
    finally
        rc.task[]=nothing;_revision_safe_set!(rc.running,false)
    end
    nothing
end
function _revision_start!(rc,destination;async)
    _revision_idle(rc)
    # No hashing, pixel decoding or picker callbacks before detaching this request.
    request=_revision_capture(rc)
    try
        rc.running[]=true;rc.error[]=nothing;rc.state[]=:busy
        rc.status[]="Validating captured pass drafts; cancellation is unavailable."
    catch err
        _revision_failure!(rc,err);rc.task[]=nothing;_revision_safe_set!(rc.running,false);return rc
    end
    if async
        rc.task[]=errormonitor(@async begin yield();_revision_run!(rc,request,destination);end)
    else
        _revision_run!(rc,request,destination)
    end
    rc
end

"""
    apply_recipe_revision!(controller; async=true)

Capture all draft rows, validation templates and original metadata before busy
notifications, then validate and publish a preview. Failure retains prior valid
preview and exposes `error`/`:failed`. Asynchronous work yields before validation
but can pause the event loop during embedded-array hashing. No cancellation.
"""
apply_recipe_revision!(rc::RecipeRevisionController;async::Bool=true)=_revision_start!(rc,nothing;async)

"""
    save_recipe_revision!(controller, path_or_picker; async=true)

Capture complete drafts/source/known protected paths before notifications or
picker invocation. Validate all rows, invoke a zero-argument picker if supplied,
verify original inputs, construct a fresh record, and save to a distinct guarded
destination. A picker returning `nothing` or empty text cancels only the save
choice. Failure/cancellation retain the previous saved identity. Writes use core
save semantics, without atomic publication, resume or persisted lineage claims.
Asynchronous work runs outside native callbacks but CPU/I/O may pause rendering.
"""
function save_recipe_revision!(rc::RecipeRevisionController,path::Union{AbstractString,Function};async::Bool=true)
    destination=path isa AbstractString ? (isempty(strip(path)) ? String(path) : Hammerhead._artifact_local_path(path)) : path
    _revision_start!(rc,destination;async)
end
