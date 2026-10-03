# Lossless recipe editing; no native widgets, pixel/result cache or script execution.

"""
    preprocessing_fields(operation)

Return a detached ordered text-option schema (`key`, `label`, `kind`) for one
of the seven core built-ins. Embedded backgrounds use explicit array/file actions,
not text. Operation order and repetitions are scientific recipe content.
"""
function preprocessing_fields(operation::Symbol)
    definitions = operation===:subtract_background ? [] :
        operation===:intensity_cap ? [(:n_sigma,"Sigma limit",:real)] :
        operation===:highpass_filter ? [(:sigma,"Filter sigma",:real)] :
        operation===:clahe ? [(:tiles,"Tiles",:tuple),(:clip_limit,"Clip limit",:real),(:nbins,"Histogram bins",:int)] :
        operation===:percentile_stretch ? [(:low,"Lower percentile",:real),(:high,"Upper percentile",:real)] :
        operation===:invert_image ? [] :
        operation===:local_variance_normalize ? [(:sigma,"Filter sigma",:real),(:epsilon,"Normalization epsilon",:real)] :
        throw(ArgumentError("unsupported preprocessing operation $operation"))
    [(key=k,label=l,kind=t) for (k,l,t) in definitions]
end
function _revision_preprocess_draft(step::PreprocessStep)
    options=Dict{Symbol,String}(f.key=>_revision_text(step.options[String(f.key)]) for f in preprocessing_fields(step.operation))
    # CLAHE's canonical tiles are a vector, unlike pass tuple fields.
    haskey(options,:tiles) && (options[:tiles]=join(step.options["tiles"],", "))
    (operation=step.operation,options=options,
     background=step.operation===:subtract_background ? copy(step.options["background"]) : nothing)
end
function _revision_preprocess_steps(rows)
    steps=PreprocessStep[]
    for row in rows
        keys(row)==(:operation,:options,:background) || throw(ArgumentError("invalid preprocessing draft fields"))
        fields=preprocessing_fields(row.operation)
        Set(keys(row.options))==Set(f.key for f in fields) || throw(ArgumentError("preprocessing option schema changed"))
        options=Dict{Symbol,Any}(f.key=>_revision_parse(row.options[f.key],f.kind,f.key) for f in fields)
        if row.operation===:subtract_background
            row.background===nothing && throw(ArgumentError("subtract_background needs an explicitly supplied background"))
            options[:background]=row.background
        else
            row.background===nothing || throw(ArgumentError("only subtract_background accepts an embedded background"))
        end
        push!(steps,PreprocessStep(row.operation;options...))
    end
    steps
end
_revision_background_equal(a,b)=typeof(a)===typeof(b) &&
    (a===nothing || (size(a)==size(b) && isequal(a,b)))
function _revision_preprocessing_equal(a,b)
    length(a)==length(b) && all(x.operation===y.operation && x.options==y.options &&
        _revision_background_equal(x.background,y.background) for (x,y) in zip(a,b))
end

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
    revision_roi_fields()

Return a detached ordered raw-text schema (`key`, `label`, `kind`) for inclusive
original-frame ROI bounds. Disabled ROI drafts compose `nothing`, meaning the
full image, while retaining their text for later re-enabling.
"""
revision_roi_fields() = [(key=:row_first,label="First row",kind=:int),
    (key=:row_last,label="Last row",kind=:int),
    (key=:col_first,label="First column",kind=:int),
    (key=:col_last,label="Last column",kind=:int)]

"""
    revision_scale_fields()

Return a detached ordered raw-text schema for isotropic `PhysicalScale`:
`pixel_size`, `dt`, `length_unit`, `time_unit`. Enabled numeric values must be
finite positive Float64 values. Unit strings are preserved verbatim and are
caller assertions, not verified calibration. No line endpoints are invented.
"""
revision_scale_fields() = [(key=:pixel_size,label="Length per pixel",kind=:real),
    (key=:dt,label="Pair delay",kind=:real),
    (key=:length_unit,label="Length unit",kind=:string),
    (key=:time_unit,label="Time unit",kind=:string)]

function _revision_roi_draft(roi,dims)
    rows,cols=roi===nothing ? (1:dims[1],1:dims[2]) : (roi.rows,roi.cols)
    (enabled=roi!==nothing,values=Dict(:row_first=>string(first(rows)),:row_last=>string(last(rows)),
        :col_first=>string(first(cols)),:col_last=>string(last(cols))))
end
function _revision_scale_draft(scale)
    values=scale===nothing ? Dict(f.key=>"" for f in revision_scale_fields()) :
        Dict(:pixel_size=>repr(scale.pixel_size),:dt=>repr(scale.dt),
            :length_unit=>scale.length_unit,:time_unit=>scale.time_unit)
    (enabled=scale!==nothing,values=values)
end

function _revision_geometry_values(draft,fields)
    keys(draft)==(:enabled,:values) && draft.enabled isa Bool && draft.values isa AbstractDict ||
        throw(ArgumentError("invalid recipe geometry draft schema"))
    Set(keys(draft.values))==Set(f.key for f in fields) && all(v->v isa AbstractString,values(draft.values)) ||
        throw(ArgumentError("recipe geometry drafts need the complete raw-text schema"))
    draft.values
end
function _revision_roi_value(draft)
    values=_revision_geometry_values(draft,revision_roi_fields())
    draft.enabled || return nothing
    parsed=Dict(f.key=>_revision_parse(values[f.key],f.kind,f.key) for f in revision_roi_fields())
    ROI(parsed[:row_first]:parsed[:row_last],parsed[:col_first]:parsed[:col_last])
end
function _revision_scale_value(draft)
    values=_revision_geometry_values(draft,revision_scale_fields())
    draft.enabled || return nothing
    PhysicalScale(pixel_size=_revision_parse(values[:pixel_size],:real,:pixel_size),
        dt=_revision_parse(values[:dt],:real,:dt),
        length_unit=values[:length_unit],time_unit=values[:time_unit])
end

"""
    RecipeRevisionController(record_or_path; protected_paths=[])

Keep a detached complete planar experiment, ordered raw pass text drafts and
ordered built-in preprocessing and ROI/scale drafts. All other recipe fields and each pass's
ordered validation tuple are retained. Untouched embedded backgrounds retain
their exact type/content; repetitions and an empty preprocessing chain are valid.
`candidate`, `diff` and `preview_text` describe the last successful preview;
`dirty` marks subsequent edits, including invalid text. `saved_record` and
`saved_path` retain the last successfully written revision independently of drafts.

Extra known output/history/report destinations must be supplied in
`protected_paths`: arbitrary external history paths cannot be recovered from a
record. Original record/input/script/run output paths are protected automatically.
All revision destinations successfully saved by this controller are also protected
for its lifetime; subsequent saves require another destination. Consumed background
file paths are also lifetime-protected, independently of the public path list.
Construction validates metadata and hashes embedded recipe arrays; launch outside
native callbacks. Preview/save asynchronous scheduling can still pause rendering;
processing cancellation is unavailable. No parent-recipe lineage is persisted.
"""
struct RecipeRevisionController
    original::ExperimentRecord
    drafts::Observable{Vector{Dict{Symbol,String}}}
    templates::Observable{Vector{PIVParameters}}
    selected::Observable{Int}
    preprocessing_drafts::Observable{Vector{NamedTuple}}
    preprocessing_selected::Observable{Int}
    roi_draft::Observable{NamedTuple}
    scale_draft::Observable{NamedTuple}
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
        Observable{Vector{NamedTuple}}(NamedTuple[_revision_preprocess_draft(s) for s in original.recipe.preprocessing]),
        Observable(isempty(original.recipe.preprocessing) ? 0 : 1),
        Observable{NamedTuple}(_revision_roi_draft(original.recipe.roi,original.input_files[1]["image_size"])),
        Observable{NamedTuple}(_revision_scale_draft(original.recipe.scale)),
        Observable{Union{Nothing,PIVRecipe}}(nothing),Observable{Union{Nothing,RecipeDiff}}(nothing),
        Observable("No preview yet."),Observable(true),Observable(false),Observable(:ready),
        Observable("Edit recipe settings, preview, then save a distinct experiment record."),Observable{Any}(nothing),
        Observable{Union{Nothing,ExperimentRecord}}(nothing),Observable(""),Observable(paths),
        Ref{Union{Nothing,Task}}(nothing),Ref(String[]))
end
_revision_idle(rc)=rc.running[] && throw(ArgumentError("recipe revision is busy; wait for it to finish"))
function _revision_geometry_edit!(rc,obs,fields,options,enabled)
    _revision_idle(rc)
    all(k->k isa Symbol && k in Set(f.key for f in fields),keys(options)) ||
        throw(ArgumentError("unknown recipe geometry field"))
    all(v->v isa AbstractString,values(options)) || throw(ArgumentError("recipe geometry drafts must contain text"))
    row=deepcopy(obs[]);merge!(row.values,Dict(k=>String(v) for (k,v) in options))
    updated=(enabled=enabled,values=row.values)
    updated==obs[] && return rc
    obs.val=updated;rc.dirty.val=true
    notify(obs);notify(rc.dirty)
    rc
end

"""
    set_revision_roi!(controller, values; enabled=controller.roi_draft[].enabled)

Merge raw ROI bounds without parsing or changing a previous valid candidate.
Disable explicitly with `enabled=false` to compose `nothing`; raw disabled text,
including invalid values, is retained. Re-enabling parses it on preview/save.
Metadata validation checks every input and pass; no schedule is regenerated.
"""
set_revision_roi!(rc::RecipeRevisionController,values::AbstractDict;enabled::Bool=rc.roi_draft[].enabled)=
    _revision_geometry_edit!(rc,rc.roi_draft,revision_roi_fields(),values,enabled)

"""
    set_revision_scale!(controller, values; enabled=controller.scale_draft[].enabled)

Merge raw scale values/labels. Disabled scale composes `nothing`, distinct from
an explicit identity scale. Disabled text is retained; enabled preview/save parses
finite positive Float64 factors. Labels are not trimmed or converted. This changes
saved metadata only; neither input pixels nor vector arrays are converted here.
"""
set_revision_scale!(rc::RecipeRevisionController,values::AbstractDict;enabled::Bool=rc.scale_draft[].enabled)=
    _revision_geometry_edit!(rc,rc.scale_draft,revision_scale_fields(),values,enabled)
function _revision_index(rc,index;insertion=false)
    index isa Integer && !(index isa Bool) && 1<=index<=length(rc.drafts[])+(insertion ? 1 : 0) ||
        throw(ArgumentError("pass index is outside the draft sequence"))
    Int(index)
end
function _revision_preprocess_index(rc,index;insertion=false)
    index isa Integer && !(index isa Bool) && 1<=index<=length(rc.preprocessing_drafts[])+(insertion ? 1 : 0) ||
        throw(ArgumentError("preprocessing index is outside the draft sequence"))
    Int(index)
end
function _revision_preprocess_edit!(rc,rows,selected)
    rc.preprocessing_drafts.val=rows;rc.preprocessing_selected.val=selected;rc.dirty.val=true
    for obs in (rc.preprocessing_drafts,rc.preprocessing_selected,rc.dirty);notify(obs);end
    rc
end

"""
    set_revision_preprocess!(controller, index, options)

Merge raw option strings into an ordered step. Invalid text remains visible and
is parsed afresh for preview/save; a previous valid candidate is never substituted.
Unknown options and textual background replacement are refused.
"""
function set_revision_preprocess!(rc::RecipeRevisionController,index,options::AbstractDict)
    _revision_idle(rc);i=_revision_preprocess_index(rc,index);rows=deepcopy(rc.preprocessing_drafts[])
    allowed=Set(f.key for f in preprocessing_fields(rows[i].operation))
    all(k->k isa Symbol && k in allowed,keys(options)) || throw(ArgumentError("unknown preprocessing option"))
    all(v->v isa AbstractString,values(options)) || throw(ArgumentError("preprocessing option drafts must contain text"))
    merge!(rows[i].options,Dict(k=>String(v) for (k,v) in options))
    _revision_preprocessing_equal(rows,rc.preprocessing_drafts[]) && return rc
    _revision_preprocess_edit!(rc,rows,rc.preprocessing_selected[])
end

"""
    insert_revision_preprocess!(controller, index, operation; background=nothing)
    insert_revision_preprocess!(controller, index; source=controller.preprocessing_selected[])

Insert a built-in with explicit core defaults, or duplicate a complete raw draft
(including invalid text and exact background precision). Repetitions are allowed.
A new subtraction may have no background yet; preview/save then refuses it.
"""
function insert_revision_preprocess!(rc::RecipeRevisionController,index,operation::Symbol;background=nothing)
    _revision_idle(rc);i=_revision_preprocess_index(rc,index;insertion=true)
    preprocessing_fields(operation)
    row=operation===:subtract_background && background===nothing ?
        (operation=operation,options=Dict{Symbol,String}(),background=nothing) :
        _revision_preprocess_draft(operation===:subtract_background ? PreprocessStep(operation;background) : PreprocessStep(operation))
    operation===:subtract_background || background===nothing || throw(ArgumentError("only subtraction accepts a background"))
    rows=deepcopy(rc.preprocessing_drafts[]);insert!(rows,i,row)
    _revision_preprocess_edit!(rc,rows,i)
end
function insert_revision_preprocess!(rc::RecipeRevisionController,index;source=rc.preprocessing_selected[])
    _revision_idle(rc);i=_revision_preprocess_index(rc,index;insertion=true);s=_revision_preprocess_index(rc,source)
    rows=deepcopy(rc.preprocessing_drafts[]);insert!(rows,i,deepcopy(rows[s]))
    _revision_preprocess_edit!(rc,rows,i)
end

"""
    move_revision_preprocess!(controller, from, to)

Move a whole raw step/background, keeping selection attached to its step.
"""
function move_revision_preprocess!(rc::RecipeRevisionController,from,to)
    _revision_idle(rc);a=_revision_preprocess_index(rc,from);b=_revision_preprocess_index(rc,to)
    a==b && return rc
    rows=deepcopy(rc.preprocessing_drafts[]);insert!(rows,b,popat!(rows,a))
    s=rc.preprocessing_selected[];s=s==a ? b : a<s<=b ? s-1 : b<=s<a ? s+1 : s
    _revision_preprocess_edit!(rc,rows,s)
end

"""
    delete_revision_preprocess!(controller, index)

Delete a complete step. An empty chain is valid and selects index zero.
"""
function delete_revision_preprocess!(rc::RecipeRevisionController,index)
    _revision_idle(rc);i=_revision_preprocess_index(rc,index)
    rows=deepcopy(rc.preprocessing_drafts[]);deleteat!(rows,i)
    s=rc.preprocessing_selected[];s=s>i ? s-1 : min(s,length(rows))
    _revision_preprocess_edit!(rc,rows,s)
end

"""
    set_revision_background!(controller, index, background)

Explicitly replace a subtraction background with a detached finite real matrix,
using core `PreprocessStep` conversion semantics. Untouched imported arrays retain
their exact precision/content. Geometry is validated when composing the recipe.
"""
function set_revision_background!(rc::RecipeRevisionController,index,background::AbstractMatrix{<:Real})
    _revision_idle(rc);i=_revision_preprocess_index(rc,index)
    rows=deepcopy(rc.preprocessing_drafts[])
    rows[i].operation===:subtract_background || throw(ArgumentError("select a subtraction step to replace its background"))
    bg=PreprocessStep(:subtract_background;background).options["background"]
    rows[i]=merge(rows[i],(background=bg,))
    _revision_preprocessing_equal(rows,rc.preprocessing_drafts[]) && return rc
    _revision_preprocess_edit!(rc,rows,rc.preprocessing_selected[])
end

function _revision_background_run!(rc,request,index,path_or_picker)
    try
        chosen=path_or_picker isa Function ? path_or_picker() : path_or_picker
        if chosen===nothing || (chosen isa AbstractString && isempty(chosen))
            rc.state[]=:cancelled;rc.status[]="Background choice cancelled; draft and displayed identities retained.";return
        end
        chosen isa AbstractString || throw(ArgumentError("background picker must return a path or nothing"))
        path=Hammerhead._artifact_local_path(chosen)
        digest=Hammerhead._experiment_file_digest(path)
        image=load_image(request.original.recipe.image_type,path)
        Hammerhead._experiment_file_digest(path)==digest || throw(ArgumentError("background file changed during loading"))
        bg=PreprocessStep(:subtract_background;background=image).options["background"]
        all(f->collect(size(bg))==f["image_size"],request.original.input_files) ||
            throw(ArgumentError("background must match every original full-image size"))
        _revision_preprocessing_equal(rc.preprocessing_drafts[],request.preprocessing) || throw(ArgumentError("preprocessing drafts changed while loading background"))
        rows=deepcopy(request.preprocessing);rows[index]=merge(rows[index],(background=bg,))
        # Protection is session provenance; the saved recipe embeds the snapshot,
        # not a continuing dependency on this source file.
        rc.protected_paths.val=unique(String[rc.protected_paths[];path])
        path in rc.saved_protected[] || push!(rc.saved_protected[],path)
        _revision_preprocess_edit!(rc,rows,index);notify(rc.protected_paths)
        rc.state[]=:completed;rc.status[]="Background snapshot loaded in recipe precision; preview is pending."
    catch err
        _revision_failure!(rc,err)
    finally
        rc.task[]=nothing;_revision_safe_set!(rc.running,false)
    end
end

"""
    load_revision_background!(controller, index, path_or_picker; async=true)

Capture the target step/drafts before notifications or a zero-argument picker.
Decode an explicitly chosen file in recipe precision, verify bytes across decoding,
and protect the consumed local path against later revision saves. No background is
estimated from the pair. Cancellation retains drafts; queued work can pause rendering.
"""
function load_revision_background!(rc::RecipeRevisionController,index,path_or_picker::Union{AbstractString,Function};async::Bool=true)
    _revision_idle(rc);i=_revision_preprocess_index(rc,index);request=_revision_capture(rc)
    request.preprocessing[i].operation===:subtract_background || throw(ArgumentError("select a subtraction step"))
    destination=path_or_picker isa AbstractString ? (isempty(strip(path_or_picker)) ? String(path_or_picker) : Hammerhead._artifact_local_path(path_or_picker)) : path_or_picker
    try
        rc.running[]=true;rc.error[]=nothing;rc.state[]=:busy;rc.status[]="Loading captured background; cancellation is unavailable."
    catch err
        _revision_failure!(rc,err);_revision_safe_set!(rc.running,false);return rc
    end
    if async
        rc.task[]=errormonitor(@async begin yield();_revision_background_run!(rc,request,i,destination);end)
    else
        _revision_background_run!(rc,request,i,destination)
    end
    rc
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
     preprocessing=deepcopy(rc.preprocessing_drafts[]),
     roi=deepcopy(rc.roi_draft[]),scale=deepcopy(rc.scale_draft[]),
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
    options[:preprocessing]=_revision_preprocess_steps(request.preprocessing)
    options[:roi]=_revision_roi_value(request.roi)
    options[:scale]=_revision_scale_value(request.scale)
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
candidate preserving every field outside pass/preprocessing/ROI/scale edits. Hashes embedded arrays; does not read
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
    rc.dirty.val=rc.drafts[]!=request.drafts || rc.templates[]!=request.templates ||
        !_revision_preprocessing_equal(rc.preprocessing_drafts[],request.preprocessing) ||
        rc.roi_draft[]!=request.roi || rc.scale_draft[]!=request.scale
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
        rc.status[]="Validating captured recipe drafts; cancellation is unavailable."
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
