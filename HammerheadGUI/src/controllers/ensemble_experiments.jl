# The saved pool is a distinct execution/association, never a planar sequence.

"""
    ensemble_experiment_record(batch::BatchRunner; backend=:cpu,
        image_type=Float64, threaded=Threads.nthreads()>1) -> EnsembleExperimentRecord

Snapshot idle image-file pairs, exact effective ensemble passes, static mask,
built-in preprocessing and scale. Presets resolve with `ensemble=true` from the
first full image, matching the pool driver. Custom passes preserve their requested
iteration/tolerance settings even though pooling ignores them. CPU/KA and
Float32/64 are explicit snapshot choices. ROI, scripts/custom callbacks, arrays
and loaders refuse rather than silently dropping settings. Imported recipes use
`EnsembleExperimentController` intact, without projection through this form.
"""
function ensemble_experiment_record(batch::BatchRunner;backend::Symbol=:cpu,
        image_type::DataType=Float64,threaded::Bool=Threads.nthreads()>1)
    batch.running[] && throw(ArgumentError("wait for the batch to finish before snapshotting"))
    pairs=deepcopy(frame_pairs(batch))
    isempty(pairs) && throw(ArgumentError("add image-file pairs first"))
    all(p->all(f->f isa AbstractString,p),pairs) ||
        throw(ArgumentError("saved ensembles require image-file pairs, not arrays or loaders"))
    batch.roi[]===nothing || throw(ArgumentError("saved ensemble pooling requires full images; clear ROI before snapshotting"))
    effort=batch.effort[]
    custom=effort===:custom ? deepcopy(build_parameters(batch)) : nothing
    mask=deepcopy(batch.mask[]);scale=build_scale(batch)
    callback=batch.preprocess[];metadata=batch.preprocess_snapshot[]
    steps=if callback===nothing
        PreprocessStep[]
    else
        metadata!==nothing && metadata.callback===callback ||
            throw(ArgumentError("saved ensemble pooling supports built-in preprocessing only; custom callbacks/scripts are unsupported"))
        recipe_identity(PIVRecipe(PIVParameters();preprocessing=metadata.steps))==metadata.identity ||
            throw(ArgumentError("built-in preprocessing metadata changed; attach the preview again"))
        deepcopy(metadata.steps)
    end
    passes=custom===nothing ? Hammerhead.effort_schedule(effort;ensemble=true,image_size=size(load_image(first(pairs)[1]))) : custom
    recipe=EnsemblePIVRecipe(passes;preprocessing=steps,mask,scale,backend,image_type,threaded)
    EnsembleExperimentRecord(pairs,recipe)
end

"""
    save_ensemble_batch_experiment(path, batch::BatchRunner; kwargs...)

Capture and save the supported full-image ensemble form. Snapshot and core
protected-locator validation happen before destination replacement.
"""
function save_ensemble_batch_experiment(path::AbstractString,batch::BatchRunner;kwargs...)
    record=ensemble_experiment_record(batch;kwargs...)
    save_experiment(path,record)
    record
end

"""
    EnsembleExperimentController(record=nothing; output_path="",
        run_record_path="", record_diagnostics=false)
    EnsembleExperimentController(path; run_record_path=path, kwargs...)

Framework-free saved planar-ensemble lane. Preserve the complete detached recipe
and metadata-only run history. `progress` counts joined input contributions over
all passes; `location` is `(pass,pair)`, and `pool_progress` is
`(completed_pools,published_results)`. Reaching the contribution budget is not pool
publication. `selected_run_id` is separate from the latest attempt and active
request. Cancellation is cooperative before loading, between contributions and
before publication, and stays busy through cleanup/history publication. No resume,
stationarity, independent-sample, convergence or accuracy claim is made.
`async=true` schedules a task; preflight/current contribution/I/O can pause GUI
rendering. Saved preprocessing never loads scripts.
`state=:history_save_failed` or `:history_refresh_failed` retains a known
completed/cancelled `last_run`: history failure does not undo pool publication.
"""
struct EnsembleExperimentController
    record::Observable{Union{Nothing,EnsembleExperimentRecord}}
    output_path::Observable{String}
    run_record_path::Observable{String}
    allow_environment_change::Observable{Bool}
    record_diagnostics::Observable{Bool}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    last_run::Observable{Union{Nothing,EnsembleExperimentRun}}
    selected_run_id::Observable{Union{Nothing,String}}
    error::Observable{Any}
    progress::Observable{Tuple{Int,Int}}
    location::Observable{Tuple{Int,Int}}
    pool_progress::Observable{Tuple{Int,Int}}
    active_request::Observable{Union{Nothing,NamedTuple}}
    _cancel_token::Base.RefValue{Union{Nothing,Base.RefValue{Bool}}}
    _task::Base.RefValue{Union{Nothing,Task}}
end
function _validate_gui_ensemble(record)
    Hammerhead._ensemble_record_preflight(record)
    foreach(r->Hammerhead._ensemble_run_validate(Hammerhead._ensemble_run_data(r),record),record.runs)
    record
end
_ensemble_budget(record)=record===nothing ? 0 : Base.Checked.checked_mul(length(record.pairs),length(record.recipe.passes))
function EnsembleExperimentController(record::Union{Nothing,EnsembleExperimentRecord}=nothing;
        output_path::AbstractString="",run_record_path::AbstractString="",record_diagnostics::Bool=false)
    snapshot=deepcopy(record)
    snapshot===nothing || _validate_gui_ensemble(snapshot)
    latest=snapshot===nothing || isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    EnsembleExperimentController(Observable{Union{Nothing,EnsembleExperimentRecord}}(snapshot),
        Observable(String(output_path)),Observable(String(run_record_path)),Observable(false),Observable(record_diagnostics),
        Observable(false),Observable(snapshot===nothing ? :empty : :ready),Observable(""),
        Observable{Union{Nothing,EnsembleExperimentRun}}(latest),Observable{Union{Nothing,String}}(latest===nothing ? nothing : latest.run_id),
        Observable{Any}(nothing),Observable((0,_ensemble_budget(snapshot))),Observable((0,0)),Observable((0,0)),
        Observable{Union{Nothing,NamedTuple}}(nothing),Ref{Union{Nothing,Base.RefValue{Bool}}}(nothing),Ref{Union{Nothing,Task}}(nothing))
end
EnsembleExperimentController(path::AbstractString;run_record_path::AbstractString=path,kwargs...)=
    EnsembleExperimentController(load_ensemble_experiment(path);run_record_path,kwargs...)

"""
    open_experiment!(controller::EnsembleExperimentController, path_or_record)

Validate before publishing an intact detached recipe. Invalid files/settings
preserve the previous record, paths and selection. Inspection loads no input
pixels or scripts. Replay options reset; prior independently displayed/reported
run identities remain historical identities.
"""
function open_experiment!(ec::EnsembleExperimentController,record::EnsembleExperimentRecord)
    _experiment_idle(ec)
    snapshot=_validate_gui_ensemble(deepcopy(record))
    latest=isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    ec.output_path.val="";ec.run_record_path.val="";ec.allow_environment_change.val=false;ec.record_diagnostics.val=false
    ec.last_run.val=latest;ec.selected_run_id.val=latest===nothing ? nothing : latest.run_id
    ec.progress.val=(0,_ensemble_budget(snapshot));ec.location.val=(0,0);ec.pool_progress.val=(0,0)
    ec.error.val=nothing;ec.state.val=:ready;ec.status.val="ensemble recipe loaded; complete settings are read-only";ec.record.val=snapshot
    for field in (:record,:output_path,:run_record_path,:allow_environment_change,:record_diagnostics,
            :last_run,:selected_run_id,:progress,:location,:pool_progress,:error,:state,:status)
        notify(getfield(ec,field))
    end
    ec
end
function open_experiment!(ec::EnsembleExperimentController,path::AbstractString)
    _experiment_idle(ec)
    record=load_ensemble_experiment(path)
    open_experiment!(ec,record)
    ec.run_record_path[]=String(path)
    ec
end

"""
    save_experiment_record!(controller::EnsembleExperimentController, path)

Save intact settings/history with core alias protection. Update the history
destination only after the save succeeds.
"""
function save_experiment_record!(ec::EnsembleExperimentController,path::AbstractString)
    _experiment_idle(ec)
    ec.record[]===nothing && throw(ArgumentError("open or snapshot an ensemble experiment first"))
    save_experiment(path,ec.record[])
    ec.run_record_path[]=String(path);notify(ec.record);ec.status[]="ensemble experiment saved"
    ec
end
function _ensemble_selected_request(ec,id=ec.selected_run_id[])
    record=deepcopy(ec.record[])
    record===nothing && throw(ArgumentError("open or snapshot an ensemble experiment first"))
    _validate_gui_ensemble(record)
    matches=filter(r->r.run_id==id,record.runs)
    length(matches)==1 || throw(ArgumentError("select a unique recorded ensemble run first"))
    record,only(matches)
end
"""
    select_experiment_run!(controller::EnsembleExperimentController, run_id)

Select historical metadata without loading output. Failed/cancelled runs remain
inspectable but cannot be opened/reported as completed pools. Busy/invalid
selection leaves the previous selection intact.
"""
function select_experiment_run!(ec::EnsembleExperimentController,id::AbstractString)
    _experiment_idle(ec);_ensemble_selected_request(ec,String(id));ec.selected_run_id[]=String(id);ec
end

"""
    start!(controller::EnsembleExperimentController; async=true, progress=nothing)

Capture complete settings, paths, override, recording choice and optional
`progress(event)` observer before notification/task scheduling. Joined contribution
events are not result writes. Cancellation at the final contribution can still
prevent the one pool's publication. The core cancellation run is retained after
cleanup; failures preserve original exceptions and reopen only newly recorded
matching history. Startup/terminal observer failures cannot strand busy state.
Historical selection is preserved. No checkpoint/resume behavior is provided.
"""
function start!(ec::EnsembleExperimentController;async::Bool=true,progress::Union{Nothing,Function}=nothing)
    ec.running[] && return ec
    record=deepcopy(ec.record[]);output=ec.output_path[];history=ec.run_record_path[]
    allow=ec.allow_environment_change[];diagnostics=ec.record_diagnostics[]
    token=Ref(false);ec._cancel_token[]=token
    ec.active_request.val=(recipe_id=record===nothing ? nothing : record.recipe.recipe_id,
        input_id=record===nothing ? nothing : record.input_id,output=output,run_record=history,
        allow_environment_change=allow,record_diagnostics=diagnostics,
        input_pairs=record===nothing ? 0 : length(record.pairs),scheduled_passes=record===nothing ? 0 : length(record.recipe.passes))
    try
        ec.running[]=true;notify(ec.active_request)
        ec.error[]=nothing;ec.last_run[]=nothing;ec.progress[]=(0,_ensemble_budget(record))
        ec.location[]=(0,0);ec.pool_progress[]=(0,0)
        ec.state[]=token[] ? :cancel_requested : :busy
        ec.status[]="pooling exact ensemble recipe; contributions are not persisted results"
        work=()->_replay_gui_ensemble!(ec,record,output,history,allow,diagnostics,token,progress)
        if async
            task=Task(work);ec._task[]=task;errormonitor(task);schedule(task)
        else
            work()
        end
    catch err
        _experiment_notify_safely!(ec.error,err);_experiment_notify_safely!(ec.state,:failed)
        _experiment_notify_safely!(ec.status,"failed to start: $(sprint(showerror,err))")
        _experiment_replay_cleanup!(ec)
    end
    ec
end
function _experiment_replay_cleanup!(ec::EnsembleExperimentController)
    ec._cancel_token[]=nothing;ec._task[]=nothing
    _experiment_notify_safely!(ec.active_request,nothing);_experiment_notify_safely!(ec.running,false)
end
"""
    cancel!(controller::EnsembleExperimentController)

Request cancellation before the next input load/after joined contribution or
before pool publication. Remain busy through cleanup/history save. Finishing
all input contributions does not make cancellation too late; publication is a
separate boundary. Idle requests are no-ops.
"""
function cancel!(ec::EnsembleExperimentController)
    token=ec._cancel_token[]
    ec.running[] && token!==nothing || return ec
    token[]=true
    ec.state[]=:cancel_requested;ec.status[]="cancel requested; waiting for contribution and cleanup before pool publication"
    ec
end
function _ensemble_history_ids(path)
    isempty(path) && return Set{String}()
    try
        Set(r.run_id for r in load_ensemble_experiment(path).runs)
    catch
        Set{String}()
    end
end
function _retain_ensemble_run(record,run)
    snapshot=deepcopy(record);run=deepcopy(run)
    index=findfirst(r->r.run_id==run.run_id,snapshot.runs)
    index===nothing ? push!(snapshot.runs,run) : (snapshot.runs[index]=run)
    _validate_gui_ensemble(snapshot)
    snapshot
end
function _publish_gui_ensemble_outcome!(ec,record,run;issue=nothing,issue_state=:history_refresh_failed)
    # Install the entire outcome before notifying observers of any part of it.
    ec.record.val=record;ec.last_run.val=run
    ec.selected_run_id[]===nothing && (ec.selected_run_id.val=run.run_id)
    ec.progress.val=(run.completed_contributions,run.total_contributions)
    ec.pool_progress.val=(run.completed_pools,run.published_results)
    ec.error.val=issue;ec.state.val=issue===nothing ? run.status : issue_state
    suffix=issue===nothing ? "" : "; run history $(issue_state===:history_save_failed ? "save" : "refresh") failed: $(sprint(showerror,issue))"
    ec.status.val="$(run.status): $(run.completed_contributions)/$(run.total_contributions) contributions; $(run.completed_pools) completed pool, $(run.published_results) persisted result"*suffix
    for observable in (ec.record,ec.last_run,ec.selected_run_id,ec.progress,ec.pool_progress,ec.error,ec.state,ec.status)
        _experiment_notify_safely!(observable,observable[])
    end
    ec
end
function _replay_gui_ensemble!(ec,record,output,history,allow,diagnostics,token,observer;
        history_loader::Function=load_ensemble_experiment)
    completed=Ref(0);total=_ensemble_budget(record);prior_ids=_ensemble_history_ids(history)
    outcome=nothing
    try
        record===nothing && throw(ArgumentError("open or snapshot an ensemble experiment first"))
        isempty(strip(output)) && throw(ArgumentError("choose ensemble result output first"))
        delivery=event->begin
            completed[]=event.completed_contributions
            ec.location[]=(event.pass_index,event.pair_index)
            ec.progress[]=(event.completed_contributions,event.total_contributions)
            ec.status[]="$(event.completed_contributions) / $(event.total_contributions) contributions joined; no pool published"*(token[] ? "; cancel requested" : "")
            observer===nothing || observer(event)
            yield()
            nothing
        end
        run=replay_experiment(record;output,run_record=isempty(history) ? nothing : history,
            allow_environment_change=allow,record_diagnostics=diagnostics,progress=delivery,cancel_requested=()->token[])
        outcome=deepcopy(run)
        record=_retain_ensemble_run(record,outcome)
        if !isempty(history)
            updated=history_loader(history)
            updated._sha256==record._sha256 && !isempty(updated.runs) &&
                Hammerhead._ensemble_run_data(last(updated.runs))==Hammerhead._ensemble_run_data(outcome) ||
                throw(ArgumentError("ensemble run history changed before reopening"))
            record=updated
        end
        _publish_gui_ensemble_outcome!(ec,record,outcome)
    catch err
        if err isa Hammerhead.EnsembleRunRecordError
            outcome=deepcopy(err.run);record=_retain_ensemble_run(record,outcome)
            _publish_gui_ensemble_outcome!(ec,record,outcome;issue=err,issue_state=:history_save_failed)
            return ec
        elseif outcome!==nothing
            record=_retain_ensemble_run(record,outcome)
            _publish_gui_ensemble_outcome!(ec,record,outcome;issue=err)
            return ec
        end
        record===nothing || _experiment_notify_safely!(ec.record,record)
        if record!==nothing && !isempty(history) && isfile(history)
            try
                updated=load_ensemble_experiment(history)
                latest=isempty(updated.runs) ? nothing : last(updated.runs)
                if latest!==nothing && !(latest.run_id in prior_ids) && updated.input_id==record.input_id &&
                        recipe_identity(updated.recipe)==recipe_identity(record.recipe)
                    _experiment_notify_safely!(ec.record,updated);_experiment_notify_safely!(ec.last_run,latest)
                    _experiment_notify_safely!(ec.pool_progress,(latest.completed_pools,latest.published_results))
                    ec.selected_run_id[]===nothing && _experiment_notify_safely!(ec.selected_run_id,latest.run_id)
                end
            catch
                # Preserve the original computation/callback/publication error.
            end
        end
        _experiment_notify_safely!(ec.progress,(completed[],total));_experiment_notify_safely!(ec.error,err)
        _experiment_notify_safely!(ec.state,:failed);_experiment_notify_safely!(ec.status,"failed: $(sprint(showerror,err))")
    finally
        _experiment_replay_cleanup!(ec)
    end
    ec
end

"""
    experiment_results(controller::EnsembleExperimentController; run_id=nothing,
        verify_inputs=false, inspect_companions=true) -> ResultExplorer

Verify the selected completed pool's native/content association before opening
one lazy physical display. Optional input-byte checks are explicit; recorded
ensemble inspection is enabled by default when present. Check output identity
again after opening. Verification is at opening, not scientific validity or
ongoing immutability; concurrent writers are unsupported. Caller windows keep
their own captured displayed run IDs after workflow selection changes.
"""
function experiment_results(ec::EnsembleExperimentController;run_id=nothing,
        verify_inputs::Bool=false,inspect_companions::Bool=true)
    _experiment_idle(ec)
    record,run=_ensemble_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    run.status===:completed || throw(ArgumentError("selected ensemble run is not completed"))
    verify_ensemble_experiment_run(record,run;verify_results=true,verify_inputs)
    explorer=ResultExplorer(run.output;lazy=true)
    inspect_companions && run.record_diagnostics && set_companion_inspection!(explorer,true)
    Hammerhead._experiment_file_digest(run.output)==run.output_sha256 || throw(ArgumentError("ensemble output changed while opening"))
    explorer
end

"""
    experiment_quality_report(controller::EnsembleExperimentController; run_id=nothing,
        include_ensemble_execution_diagnostics=true, verify_inputs=false)

Generate a verified associated ensemble report (format 5) for the selected
completed run. Disabling pooled observations still preserves explicit ensemble
association; it does not coerce a pool into the planar sequence schemas.
Verification is at generation, not stationarity, accuracy or UQ applicability.
"""
function experiment_quality_report(ec::EnsembleExperimentController;run_id=nothing,
        include_ensemble_execution_diagnostics::Bool=true,verify_inputs::Bool=false)
    _experiment_idle(ec)
    record,run=_ensemble_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    quality_report(record,run;include_ensemble_execution_diagnostics,verify_inputs)
end
"""
    save_experiment_quality_report(path, controller::EnsembleExperimentController; kwargs...)

Generate/save the selected run's associated TOML report. Protect the captured
workflow destinations in addition to core input/history/result locators. Options
match `experiment_quality_report`; validation precedes output opening, but
filesystem failures may leave partial output and publication is not atomic.
"""
function save_experiment_quality_report(path::AbstractString,ec::EnsembleExperimentController;run_id=nothing,
        include_ensemble_execution_diagnostics::Bool=true,verify_inputs::Bool=false)
    _experiment_idle(ec)
    record,run=_ensemble_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    protected=filter(!isempty,[ec.output_path[],ec.run_record_path[]])
    report=quality_report(record,run;include_ensemble_execution_diagnostics,verify_inputs)
    save_quality_report(path,report;protected_paths=protected)
    report
end

"""
    experiment_summary(controller::EnsembleExperimentController) -> String

Readable complete effective recipe, source descriptors and ordered pool inputs.
Masks/backgrounds use shape/type/content digests instead of dumping their values.
Ignored requested iterations/tolerance remain visible. This validates the saved
snapshot, not stationarity or measurement accuracy; no pixels/scripts are loaded.
"""
function experiment_summary(ec::EnsembleExperimentController)
    record=ec.record[]
    record===nothing && return "Open a saved ensemble, or snapshot file pairs from the planar batch form."
    _validate_gui_ensemble(record)
    lines=["recipe: $(recipe_identity(record.recipe))","input: $(record.input_id)",
        "$(length(record.pairs)) ordered input pairs; $(length(record.recipe.passes)) pooled passes; one output pool",
        "One pooled sweep per pass; requested iterations/tolerances are ignored, not convergence checks."]
    function visit(value,path)
        if value isa AbstractDict
            foreach(k->visit(value[k],isempty(path) ? String(k) : "$path.$k"),sort!(collect(keys(value));by=String))
        elseif value isa AbstractVector && !isempty(value) && any(v->v isa Union{AbstractDict,AbstractVector},value)
            foreach(i->visit(value[i],"$path[$i]"),eachindex(value))
        else
            push!(lines,"$path: "*_stereo_summary_value(value))
        end
    end
    visit(Hammerhead._ensemble_recipe_data(record.recipe),"")
    for (i,metadata) in enumerate(record.input_files)
        visit(metadata,"input_files[$i]")
    end
    for (i,pair) in enumerate(record.pairs)
        push!(lines,"input pair $i: "*join([record.input_files[j]["path"] for j in pair]," | "))
    end
    push!(lines,"Contribution counts do not establish stationarity, effective independent sample size, accuracy or uncertainty coverage.")
    join(lines,"\n")
end
"""
    experiment_run_history(controller::EnsembleExperimentController) -> String

Historical run identities/status, joined contributions, completed/persisted pools,
destinations and original errors. Counts are not per-pair result counts.
"""
function experiment_run_history(ec::EnsembleExperimentController)
    record=ec.record[]
    record===nothing || isempty(record.runs) ? "No recorded ensemble runs." :
        join(["$(r.run_id): $(r.status); $(r.completed_contributions)/$(r.total_contributions) joined contributions; $(r.completed_pools) completed pool, $(r.published_results) persisted result\n"*
            "  recipe=$(r.recipe_id)\n  input=$(r.input_id)\n  output=$(r.output)"*
            (r.error===nothing ? "" : "\n  $(r.error)") for r in record.runs],"\n")
end
