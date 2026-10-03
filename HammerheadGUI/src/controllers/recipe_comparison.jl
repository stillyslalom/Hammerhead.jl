# Framework-free selected-pair reruns; no result/image cache or recipe projection.

"""
    RecipeComparisonController(before=nothing, after=nothing;
        pair_indices=(1,1), basis=:pixels, allow_environment_change=false,
        protected_paths=[])

Compare two complete saved planar recipes on explicitly selected ordered pairs.
Records are detached copies; no form reconstructs their settings. Observable
`before`, `after`, `pair_indices`, `basis` and `allow_environment_change` describe
the next attempt. `report` is the last completed or loaded detached core report,
whose own provenance remains visible after an unsuccessful attempt. `running`,
`state`, `status` and `error` expose execution state. Custom preprocessing is
refused by the core comparison contract. No experiment run history is changed.

[`compare!`](@ref) captures its complete request before notifications/tasks.
Asynchronous scheduling does not guarantee responsive CPU preflight/computation;
live progress and cancellation are unavailable. Only two recipe snapshots and
scalar report metadata remain after completion, with no image/result arrays.
Caller edits to nested record arrays are checked by the core identity contract.
"""
struct RecipeComparisonController
    before::Observable{Union{Nothing,ExperimentRecord}}
    after::Observable{Union{Nothing,ExperimentRecord}}
    pair_indices::Observable{Tuple{Int,Int}}
    basis::Observable{Symbol}
    allow_environment_change::Observable{Bool}
    protected_paths::Observable{Vector{String}}
    report::Observable{Union{Nothing,RecipePairComparison}}
    report_origin::Observable{String}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    error::Observable{Any}
    task::Base.RefValue{Union{Nothing,Task}}
    report_protected::Base.RefValue{Tuple{Vararg{String}}}
end

function _comparison_record(value)
    value===nothing && return nothing
    record=value isa AbstractString ? load_experiment(value) : value
    record isa ExperimentRecord || throw(ArgumentError("comparison needs an ExperimentRecord or saved experiment path"))
    snapshot=deepcopy(record)
    _validate_gui_experiment(snapshot)
    snapshot
end
function _comparison_indices(indices)
    (indices isa Tuple || indices isa AbstractVector) && length(indices)==2 &&
        all(i->i isa Integer && !(i isa Bool) && i>0,indices) ||
        throw(ArgumentError("choose two positive integer pair indices"))
    (Int(indices[1]),Int(indices[2]))
end
function _comparison_basis(basis)
    basis in (:pixels,:physical) || throw(ArgumentError("comparison basis must be :pixels or :physical"))
    basis
end
function RecipeComparisonController(before=nothing,after=nothing;
                                    pair_indices=(1,1),basis::Symbol=:pixels,
                                    allow_environment_change::Bool=false,protected_paths=String[])
    a,b=_comparison_record(before),_comparison_record(after)
    indices=_comparison_indices(pair_indices)
    _comparison_basis(basis)
    all(p->p isa AbstractString,protected_paths) || throw(ArgumentError("protected_paths must contain paths"))
    RecipeComparisonController(Observable{Union{Nothing,ExperimentRecord}}(a),
        Observable{Union{Nothing,ExperimentRecord}}(b),Observable(indices),Observable(basis),
        Observable(allow_environment_change),Observable(abspath.(String[protected_paths...])),
        Observable{Union{Nothing,RecipePairComparison}}(nothing),Observable(""),Observable(false),
        Observable(a===nothing || b===nothing ? :empty : :ready),Observable("Choose both records and pair indices."),
        Observable{Any}(nothing),Ref{Union{Nothing,Task}}(nothing),Ref{Tuple{Vararg{String}}}(()))
end
_comparison_idle(cc)=cc.running[] && throw(ArgumentError("pair comparison is busy; wait for it to finish"))

"""
    open_comparison_record!(controller, side, record_or_path)

Replace `side=:before` or `:after` only after complete metadata validation, without
executing scripts or opening images. Reset that side's pair index to one. Failed
loading preserves both records, choices and previous report. Successful changes
leave the previous report explicitly labeled with its own historical provenance.
Busy controllers refuse changes.
"""
function open_comparison_record!(cc::RecipeComparisonController,side::Symbol,value)
    _comparison_idle(cc)
    side in (:before,:after) || throw(ArgumentError("comparison side must be :before or :after"))
    record=_comparison_record(value)
    record===nothing && throw(ArgumentError("choose a saved experiment"))
    getproperty(cc,side)[]=record
    old=cc.pair_indices[]
    cc.pair_indices[]=side===:before ? (1,old[2]) : (old[1],1)
    cc.state[]=cc.before[]===nothing || cc.after[]===nothing ? :empty : :ready
    cc.status[]="$(String(side)) record loaded; previous report describes its own recorded request."
    cc
end

"""
    set_comparison_pairs!(controller, before_index, after_index)

Choose two explicit one-based pair indices, checking each against its loaded
record. Pair content/order is verified by the core when comparing; equal list
positions or filenames alone never establish identical input.
"""
function set_comparison_pairs!(cc::RecipeComparisonController,before_index,after_index)
    _comparison_idle(cc)
    indices=_comparison_indices((before_index,after_index))
    for (record,index) in zip((cc.before[],cc.after[]),indices)
        record!==nothing && index<=length(record.pairs) || throw(ArgumentError("selected pair is outside a loaded record"))
    end
    cc.pair_indices[]=indices
    cc
end

"""
    compare!(controller; async=true)

Rerun both selected built-in recipes through [`compare_recipe_pair`](@ref),
checking ordered pair bytes/dimensions, exact recipe/input identities, units and
environment. Capture detached records, indices, basis, environment override and
protected paths before notifying observers or scheduling a task. Success replaces
the metadata-only report; failure keeps the previous report and exposes the
exception in `error` with `state=:failed`. No output is saved automatically.
There is no progress/cancellation guarantee; `async=false` executes synchronously.
Invalid record metadata or request syntax throws during preflight, preserving
state/report before execution begins; processing failures use `error`/`:failed`.
"""
function compare!(cc::RecipeComparisonController;async::Bool=true)
    _comparison_idle(cc)
    before,after=_comparison_record(cc.before[]),_comparison_record(cc.after[])
    before!==nothing && after!==nothing || throw(ArgumentError("choose both comparison records first"))
    indices=_comparison_indices(cc.pair_indices[])
    basis=_comparison_basis(cc.basis[])
    allow=cc.allow_environment_change[]
    protected=Tuple(abspath.(copy(cc.protected_paths[])))
    cc.running[]=true
    cc.error[]=nothing
    cc.state[]=:busy
    cc.status[]="Comparing one ordered pair; live progress and cancellation are unavailable."
    run=()->_run_recipe_comparison!(cc,before,after,indices,basis,allow,protected)
    if async
        cc.task[]=errormonitor(@async run())
    else
        run()
    end
    cc
end
function _run_recipe_comparison!(cc,before,after,indices,basis,allow,protected)
    try
        report=compare_recipe_pair(before,after;pair_indices=indices,basis,allow_environment_change=allow)
        cc.report_protected[]=protected
        cc.report_origin[]="Completed comparison; recorded request below is independent of current choices."
        cc.report[]=report
        cc.state[]=:completed
        cc.status[]="Comparison complete; differences describe sensitivity, not accuracy."
    catch err
        cc.error[]=err
        cc.state[]=:failed
        cc.status[]="Comparison failed: $(_errmsg(err)). Previous report, if present, remains unchanged."
    finally
        cc.task[]=nothing
        cc.running[]=false
    end
    nothing
end

"""
    open_comparison_report!(controller, path)

Load a validated past TOML comparison for read-only inspection without opening
images, rerunning recipes or replacing current record choices. Loading does not
reverify original files. Failure preserves the previous report and origin label.
"""
function open_comparison_report!(cc::RecipeComparisonController,path::AbstractString)
    _comparison_idle(cc)
    report=load_pair_comparison(path)
    cc.report_protected[]=()
    cc.report_origin[]="Loaded past report: $(abspath(path)). Original inputs have not been reverified."
    cc.report[]=report
    cc.status[]="Past report loaded for read-only inspection; no computation performed."
    cc
end

"""
    save_comparison_report!(controller, path) -> path

Save the last completed/loaded validated report through [`save_pair_comparison`](@ref),
without rerunning current choices. Protect both records' persisted dependencies
and captured/current extra GUI destinations. Refuse busy/missing/mutated reports
and known aliases before opening output. Writes are not atomic publication.
"""
function save_comparison_report!(cc::RecipeComparisonController,path::AbstractString)
    _comparison_idle(cc)
    report=cc.report[]
    report===nothing && throw(ArgumentError("no completed or loaded comparison report"))
    protected=[cc.report_protected[]...;cc.protected_paths[]...]
    for record in (cc.before[],cc.after[])
        record===nothing && continue
        append!(protected,[f["path"] for f in record.input_files])
        append!(protected,record.record_paths)
        append!(protected,[run.output for run in record.runs])
        record.recipe.external_preprocess===nothing || push!(protected,record.recipe.external_preprocess.path)
    end
    saved=save_pair_comparison(path,report;protected_paths=protected)
    cc.status[]="Saved last report; its recorded provenance is unchanged."
    saved
end

function _comparison_setting_text(data)
    kind=data["kind"]
    kind in ("missing","nothing") && return kind
    value=data["value"]
    kind=="array_summary" && return "array $(Tuple(value["size"])) $(value["element_type"]); SHA-256 $(value["sha256"])"
    kind=="tuple" && return "("*join(_comparison_setting_text.(value),", ")*")"
    kind=="mapping" && return "("*join([k*"="*_comparison_setting_text(value[k]) for k in sort!(collect(keys(value)))],", ")*")"
    kind=="string" && return repr(value)
    kind=="symbol" && return ":"*value
    string(value)
end
function _comparison_tree(io,value;indent="")
    if value isa AbstractDict
        for key in sort!(collect(keys(value)))
            label=replace(key,'_'=>' ')
            nested=value[key] isa AbstractDict
            println(io,indent,label,nested ? ":" : ": $(value[key])")
            nested && _comparison_tree(io,value[key];indent=indent*"  ")
        end
    else
        println(io,indent,value)
    end
end

"""
    comparison_summary(controller; section=:summary) -> String

Readable complete text for paged inspection. `:request` shows current record/pair
choices; `:summary`, `:settings`, `:populations` and `:provenance` describe only the
last report, never relabel it as current after edits/failure. Settings preserve
typed before/after values and compact array digests. Populations include exact
common nodes, separate native grids, mask/flag selection and unavailable metrics.
Differences are after minus before and do not establish accuracy or UQ coverage.
"""
function comparison_summary(cc::RecipeComparisonController;section::Symbol=:summary)
    section in (:request,:summary,:settings,:populations,:provenance) || throw(ArgumentError("unknown comparison section"))
    if section===:request
        io=IOBuffer()
        println(io,"Current request (not the last report): $(cc.basis[]); environment override $(cc.allow_environment_change[])")
        for (side,record,index) in zip(("Before","After"),(cc.before[],cc.after[]),cc.pair_indices[])
            println(io,"\n$side pair $index")
            if record===nothing
                println(io,"No record selected.")
            else
                println(io,"Recipe: $(record.recipe.recipe_id)\nInput collection: $(record.input_id)")
                if 1<=index<=length(record.pairs)
                    for file in record.input_files[record.pairs[index]]
                        println(io,"  $(file["path"])\n  SHA-256: $(file["sha256"]); size $(file["size_bytes"]) bytes; dimensions $(Tuple(file["image_size"]))")
                    end
                else
                    println(io,"Pair index is outside this record.")
                end
            end
        end
        return String(take!(io))
    end
    cc.report[]===nothing && return "No completed or loaded comparison report."
    data=pair_comparison_data(cc.report[])
    io=IOBuffer();println(io,cc.report_origin[])
    for side in ("before","after")
        selected=data["provenance"][side]
        println(io,"$(uppercasefirst(side)) report: pair $(selected["pair_index"])\n  Recipe ID: $(selected["recipe_id"])\n  Input ID: $(selected["record_input_id"])")
    end
    println(io,"Report basis: $(data["basis"]["mode"]), $(data["basis"]["unit"]); after minus before.\n")
    if section===:summary
        show(io,MIME"text/plain"(),cc.report[])
    elseif section===:settings
        println(io,"$(length(data["settings_changes"])) recorded settings changes")
        for change in data["settings_changes"]
            println(io,"\n$(change["path"])\n  Before: $(_comparison_setting_text(change["before"]))\n  After: $(_comparison_setting_text(change["after"]))")
        end
    elseif section===:populations
        println(io,"Native-grid populations (different grids remain separate):")
        _comparison_tree(io,data["native"])
        println(io,"\nExact common-grid populations and numerical differences:")
        _comparison_tree(io,data["common"])
        println(io,"\nUnavailable claims:")
        _comparison_tree(io,data["unavailable"])
    else
        println(io,"Recorded provenance; loading this report does not reverify inputs:")
        _comparison_tree(io,data["provenance"])
        println(io,"\nActual comparison environment:")
        _comparison_tree(io,data["actual_environment"])
        _comparison_tree(io,data["generator"])
    end
    String(take!(io))
end
