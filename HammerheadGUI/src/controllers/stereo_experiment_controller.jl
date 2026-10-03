# Saved fitted-stereo recipes have their own lane; never project them onto a form.

"""
    experiment_record(batch::StereoBatchRunner; timestamps=(nothing,nothing),
        time_units=(nothing,nothing), clock_ids=(nothing,nothing), source_ids=(nothing,nothing),
        frame_ids=(nothing,nothing), world_unit=nothing, coordinate_frame=nothing,
        calibration_note=nothing, self_calibration_report=nothing) -> StereoExperimentRecord

Snapshot an idle file-based stereo form with its fitted dewarpers, exact effective
passes and scale. Capture CPU/Float64 and the current thread default used by the
form. Extra keywords supply explicit timing/calibration provenance, not fitting
or settings overrides. Reopened recipes may contain richer preprocessing, ROI
and mask settings; these are preserved by `StereoExperimentController`, never
reconstructed through this narrower form. Arrays/loaders and edited maps refuse.
"""
function experiment_record(batch::StereoBatchRunner; timestamps=(nothing,nothing),
    time_units=(nothing,nothing),clock_ids=(nothing,nothing),source_ids=(nothing,nothing),
    frame_ids=(nothing,nothing),world_unit=nothing,coordinate_frame=nothing,
    calibration_note=nothing,self_calibration_report=nothing)
    batch.running[] && throw(ArgumentError("wait for the stereo batch to finish before snapshotting"))
    pairs=deepcopy(stereo_pairs(batch))
    dw=deepcopy(batch.dewarpers[])
    dw===nothing && throw(ArgumentError("build fitted stereo dewarpers first"))
    !isempty(pairs[1]) && length(pairs[1])==length(pairs[2]) || throw(ArgumentError("add equal nonempty camera pair lists"))
    all(p->all(f->f isa AbstractString,p),Iterators.flatten(pairs)) ||
        throw(ArgumentError("saved stereo experiments require image-file paths, not arrays or loaders"))
    passes=batch.effort[]===:custom ? build_parameters(batch) :
        Hammerhead.effort_schedule(batch.effort[];image_size=size(dw[1].grid))
    scale=build_scale(batch)
    recipe=StereoPIVRecipe(passes,dw...;scale,backend=:cpu,image_type=Float64,
        threaded=Threads.nthreads()>1,world_unit,coordinate_frame,calibration_note,self_calibration_report)
    StereoExperimentRecord(pairs...,recipe;timestamps=deepcopy(timestamps),time_units,clock_ids,source_ids,frame_ids=deepcopy(frame_ids))
end

"""
    save_batch_experiment(path, batch::StereoBatchRunner; kwargs...) -> StereoExperimentRecord

Capture and validate the supported stereo form before saving its intact record.
Unsupported settings or input/record/result aliases refuse before writing.
"""
function save_batch_experiment(path::AbstractString,batch::StereoBatchRunner;kwargs...)
    record=experiment_record(batch;kwargs...)
    save_experiment(path,record)
    record
end

"""
    StereoExperimentController(record=nothing; output_path="", run_record_path="",
        record_diagnostics=false, record_pair_timing=false)
    StereoExperimentController(path; run_record_path=path, kwargs...)

Framework-free saved fitted-stereo workflow. Keep the complete detached core
record, explicit destinations/environment override and companion choices.
`last_run` describes the latest attempt; `selected_run_id` chooses historical
inspection/reporting independently. No scripts, fitting, checkpoint/resume or
planar recipe coercion occurs. `progress` counts persisted acquisitions.
Cooperative replay can pause rendering during preflight/current-pair computation
and I/O; `async=true` is task scheduling, not a responsiveness guarantee.
"""
struct StereoExperimentController
    record::Observable{Union{Nothing,StereoExperimentRecord}}
    output_path::Observable{String}
    run_record_path::Observable{String}
    allow_environment_change::Observable{Bool}
    record_diagnostics::Observable{Bool}
    record_pair_timing::Observable{Bool}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    last_run::Observable{Union{Nothing,ExperimentRun}}
    selected_run_id::Observable{Union{Nothing,String}}
    error::Observable{Any}
    progress::Observable{Tuple{Int,Int}}
    active_request::Observable{Union{Nothing,NamedTuple}}
    _cancel_token::Base.RefValue{Union{Nothing,Base.RefValue{Bool}}}
    _task::Base.RefValue{Union{Nothing,Task}}
end
function _validate_gui_stereo_experiment(record)
    Hammerhead._stereo_record_preflight(record)
    foreach(r->Hammerhead._stereo_run_validate(Hammerhead._experiment_run_data(r),record),record.runs)
    record
end
function StereoExperimentController(record::Union{Nothing,StereoExperimentRecord}=nothing;
    output_path::AbstractString="",run_record_path::AbstractString="",record_diagnostics::Bool=false,record_pair_timing::Bool=false)
    snapshot=deepcopy(record)
    snapshot===nothing || _validate_gui_stereo_experiment(snapshot)
    latest=snapshot===nothing || isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    StereoExperimentController(Observable{Union{Nothing,StereoExperimentRecord}}(snapshot),
        Observable(String(output_path)),Observable(String(run_record_path)),Observable(false),
        Observable(record_diagnostics),Observable(record_pair_timing),Observable(false),
        Observable(snapshot===nothing ? :empty : :ready),Observable(""),
        Observable{Union{Nothing,ExperimentRun}}(latest),Observable{Union{Nothing,String}}(latest===nothing ? nothing : latest.run_id),
        Observable{Any}(nothing),Observable((0,snapshot===nothing ? 0 : length(snapshot.pairs))),
        Observable{Union{Nothing,NamedTuple}}(nothing),
        Ref{Union{Nothing,Base.RefValue{Bool}}}(nothing),Ref{Union{Nothing,Task}}(nothing))
end
StereoExperimentController(path::AbstractString;run_record_path::AbstractString=path,kwargs...)=
    StereoExperimentController(load_stereo_experiment(path);run_record_path,kwargs...)

"""
    open_experiment!(controller::StereoExperimentController, path_or_record)

Validate the candidate before publishing a detached intact stereo record. Failed
loads preserve the previous record, destinations and selected run. No input image
or script is executed just to inspect saved settings. Replay options reset.
"""
function open_experiment!(ec::StereoExperimentController,record::StereoExperimentRecord)
    _experiment_idle(ec)
    snapshot=_validate_gui_stereo_experiment(deepcopy(record))
    latest=isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    # Publish a coherent bundle before observers can read another field.
    ec.output_path.val="";ec.run_record_path.val="";ec.allow_environment_change.val=false
    ec.record_diagnostics.val=false;ec.record_pair_timing.val=false
    ec.last_run.val=latest;ec.selected_run_id.val=latest===nothing ? nothing : latest.run_id
    ec.progress.val=(0,length(snapshot.pairs));ec.error.val=nothing;ec.state.val=:ready
    ec.status.val="stereo recipe loaded; fitted settings are read-only";ec.record.val=snapshot
    for field in (:record,:output_path,:run_record_path,:allow_environment_change,:record_diagnostics,
                  :record_pair_timing,:last_run,:selected_run_id,:progress,:error,:state,:status)
        notify(getfield(ec,field))
    end
    ec
end
function open_experiment!(ec::StereoExperimentController,path::AbstractString)
    _experiment_idle(ec)
    record=load_stereo_experiment(path)
    open_experiment!(ec,record)
    ec.run_record_path[]=String(path)
    ec
end

"""
    save_experiment_record!(controller::StereoExperimentController, path)

Save complete fitted settings/history, preserving core protected locators. The
saved destination becomes the run-history destination only after successful save.
"""
function save_experiment_record!(ec::StereoExperimentController,path::AbstractString)
    _experiment_idle(ec)
    ec.record[]===nothing && throw(ArgumentError("open or snapshot a stereo experiment first"))
    save_experiment(path,ec.record[])
    ec.run_record_path[]=String(path)
    notify(ec.record)
    ec.status[]="stereo experiment saved"
    ec
end

"""
    select_experiment_run!(controller::StereoExperimentController, run_id)

Select a recorded run by ID without loading results. A failed run remains
inspectable as history but cannot be opened as completed output or reported.
Invalid selection/busy actions preserve the previous selection.
"""
function select_experiment_run!(ec::StereoExperimentController,id::AbstractString)
    _experiment_idle(ec)
    _stereo_selected_request(ec,String(id))
    ec.selected_run_id[]=String(id)
    ec
end
function _stereo_selected_request(ec,id=ec.selected_run_id[])
    record=deepcopy(ec.record[])
    record===nothing && throw(ArgumentError("open or snapshot a stereo experiment first"))
    _validate_gui_stereo_experiment(record)
    matches=filter(r->r.run_id==id,record.runs)
    length(matches)==1 || throw(ArgumentError("select a unique recorded stereo run first"))
    record,only(matches)
end

"""
    start!(controller::StereoExperimentController; async=true, progress=nothing)

Capture the complete record, destinations, override and companion flags before
the first notification. Progress runs after native acquisition persistence.
Cancellation waits for a written boundary and cleanup; final-write cancellation
completes normally. Earlier cancellation leaves a failed core run/native prefix,
not a checkpoint. Before task execution, cancellation leaves files untouched.
Original failures and observer failures release busy state; terminal notification
errors never replace the recorded outcome. Historical selection is preserved.
"""
function start!(ec::StereoExperimentController;async::Bool=true,progress::Union{Nothing,Function}=nothing)
    ec.running[] && return ec
    record=deepcopy(ec.record[]);output=ec.output_path[];history=ec.run_record_path[]
    allow=ec.allow_environment_change[];diagnostics=ec.record_diagnostics[];timing=ec.record_pair_timing[]
    token=Ref(false);ec._cancel_token[]=token
    ec.active_request.val=(recipe_id=record===nothing ? nothing : record.recipe.recipe_id,
        input_id=record===nothing ? nothing : record.input_id,output=output,run_record=history,
        allow_environment_change=allow,record_diagnostics=diagnostics,record_pair_timing=timing)
    try
        ec.running[]=true
        notify(ec.active_request)
        ec.error[]=nothing;ec.last_run[]=nothing
        ec.progress[]=(0,record===nothing ? 0 : length(record.pairs))
        ec.state[]=token[] ? :cancel_requested : :busy
        ec.status[]="replaying fitted stereo recipe; current acquisition may pause rendering"
        work=()->_replay_gui_stereo!(ec,record,output,history,allow,diagnostics,timing,token,progress)
        if async
            task=Task(work);ec._task[]=task;errormonitor(task);schedule(task)
        else
            work()
        end
    catch err
        _experiment_notify_safely!(ec.error,err)
        _experiment_notify_safely!(ec.state,:failed)
        _experiment_notify_safely!(ec.status,"failed to start: $(_errmsg(err))")
        _experiment_replay_cleanup!(ec)
    end
    ec
end
function _experiment_replay_cleanup!(ec::StereoExperimentController)
    ec._cancel_token[]=nothing
    ec._task[]=nothing
    _experiment_notify_safely!(ec.active_request,nothing)
    _experiment_notify_safely!(ec.running,false)
end
"""
    cancel!(controller::StereoExperimentController)

Request cancellation at the next persisted acquisition boundary. Remain busy
through loading/output/history cleanup. Idle requests are no-ops. Final-boundary
cancellation is completion; partial native output is not resumable.
"""
function cancel!(ec::StereoExperimentController)
    token=ec._cancel_token[]
    ec.running[] && token!==nothing || return ec
    token[]=true
    ec.state[]=:cancel_requested
    ec.status[]="cancellation requested; waiting for acquisition write and cleanup"
    ec
end
function _replay_gui_stereo!(ec,record,output,history,allow,diagnostics,timing,token,observer)
    total=record===nothing ? 0 : length(record.pairs);written=Ref(0)
    try
        record===nothing && throw(ArgumentError("open or snapshot a stereo experiment first"))
        isempty(strip(output)) && throw(ArgumentError("choose stereo result output first"))
        token[] && throw(_ExperimentCancelled(0,total))
        delivery=(i,n)->begin
            written[]=i;ec.progress[]=(i,n)
            ec.status[]="$i / $n acquisitions written"*(token[] ? "; cancellation requested" : "")
            observer===nothing || observer(i,n)
            yield()
            token[] && i<n && throw(_ExperimentCancelled(i,n))
            nothing
        end
        run=replay_experiment(record;output,run_record=isempty(history) ? nothing : history,
            allow_environment_change=allow,record_diagnostics=diagnostics,record_pair_timing=timing,progress=delivery)
        if isempty(history)
            push!(record.runs,run)
        else
            record=load_stereo_experiment(history)
            !isempty(record.runs) && last(record.runs).run_id==run.run_id || throw(ArgumentError("stereo run history changed before reopening"))
        end
        _experiment_notify_safely!(ec.record,record)
        _experiment_notify_safely!(ec.last_run,run)
        ec.selected_run_id[]===nothing && _experiment_notify_safely!(ec.selected_run_id,run.run_id)
        _experiment_notify_safely!(ec.progress,(run.completed_pairs,total))
        _experiment_notify_safely!(ec.state,:completed)
        _experiment_notify_safely!(ec.status,"completed: $(run.completed_pairs) written acquisitions")
    catch err
        record===nothing || _experiment_notify_safely!(ec.record,record)
        if record!==nothing && !isempty(history) && isfile(history)
            try
                updated=load_stereo_experiment(history)
                if updated.input_id==record.input_id && recipe_identity(updated.recipe)==recipe_identity(record.recipe) && length(updated.runs)>length(record.runs)
                    _experiment_notify_safely!(ec.record,updated)
                    _experiment_notify_safely!(ec.last_run,last(updated.runs))
                    ec.selected_run_id[]===nothing && _experiment_notify_safely!(ec.selected_run_id,last(updated.runs).run_id)
                end
            catch
                # Preserve the original replay exception if history publication/reopen failed.
            end
        end
        _experiment_notify_safely!(ec.progress,(written[],total))
        _experiment_notify_safely!(ec.error,err)
        _experiment_notify_safely!(ec.state,err isa _ExperimentCancelled ? :cancelled : :failed)
        _experiment_notify_safely!(ec.status,err isa _ExperimentCancelled ? "stereo replay cancelled after $(written[]) / $total written acquisitions" : "failed: $(_errmsg(err))")
    finally
        _experiment_replay_cleanup!(ec)
    end
    ec
end

"""
    experiment_results(controller::StereoExperimentController; run_id=nothing,
        verify_inputs=false, inspect_companions=false) -> ResultExplorer

Verify the selected completed run's native association/content and raw fields
before publishing a lazy explorer. Retain one physical display frame. Optional
current input-byte checks and companion mode are explicit. Verification occurs
at opening; concurrent writers are unsupported. A separately opened explorer
keeps its own captured run identity even if the workflow selection later changes.
"""
function experiment_results(ec::StereoExperimentController;run_id=nothing,verify_inputs::Bool=false,inspect_companions::Bool=false)
    _experiment_idle(ec)
    record,run=_stereo_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    run.status===:completed || throw(ArgumentError("selected stereo run is not completed"))
    verify_stereo_experiment_run(record,run;verify_results=true,verify_inputs)
    explorer=ResultExplorer(run.output;lazy=true)
    inspect_companions && set_companion_inspection!(explorer,true)
    Hammerhead._experiment_file_digest(run.output)==run.output_sha256 || throw(ArgumentError("stereo output changed while opening"))
    explorer
end

"""
    experiment_quality_report(controller::StereoExperimentController; run_id=nothing,
        include_measurement_history=false, include_execution_diagnostics=false, verify_inputs=false)

Generate an associated core report for the selected completed run. Default
format 1 counts current stored fields; execution inclusion selects format 3.
Stereo per-node measurement history is unsupported and explicitly refused.
Verification describes generation time, not accuracy or continuing validity.
"""
function experiment_quality_report(ec::StereoExperimentController;run_id=nothing,
    include_measurement_history::Bool=false,include_execution_diagnostics::Bool=false,verify_inputs::Bool=false)
    _experiment_idle(ec)
    record,run=_stereo_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    quality_report(record,run;include_measurement_history,include_execution_diagnostics,verify_inputs)
end

"""
    save_experiment_quality_report(path, controller::StereoExperimentController; kwargs...)

Save the selected run's validated associated TOML report, protecting captured
workflow destinations in addition to core input/record/run-output locators.
Capture selection/options before scanning. Failure leaves the destination intact.
"""
function save_experiment_quality_report(path::AbstractString,ec::StereoExperimentController;run_id=nothing,
    include_measurement_history::Bool=false,include_execution_diagnostics::Bool=false,verify_inputs::Bool=false)
    _experiment_idle(ec)
    record,run=_stereo_selected_request(ec,run_id===nothing ? ec.selected_run_id[] : run_id)
    protected=filter(!isempty,[ec.output_path[],ec.run_record_path[]])
    report=quality_report(record,run;include_measurement_history,include_execution_diagnostics,verify_inputs)
    save_quality_report(path,report;protected_paths=protected)
    report
end

function _stereo_summary_value(value)
    value isa AbstractVector && length(value)<=8 && all(v->v isa Union{Number,AbstractString,Bool},value) && return repr(value)
    value isa AbstractArray && return "$(size(value)) $(eltype(value)); SHA-256 $(Hammerhead._history_digest(Dict{String,Any}("value"=>value)))"
    value isa AbstractDict && return "{"*join(["$k="*_stereo_summary_value(v) for (k,v) in sort!(collect(value);by=x->String(first(x)))],", ")*"}"
    repr(value)
end
"""
    experiment_summary(controller::StereoExperimentController) -> String

Readable complete fitted-stereo settings/provenance. Every primitive recipe key
is retained; arrays use dimensions/type/content digest instead of dumping or
copying masks, backgrounds or camera matrices. These summaries are not calibration
accuracy claims. Exact values remain in the intact saved record.
"""
function experiment_summary(ec::StereoExperimentController)
    record=ec.record[]
    record===nothing && return "Open a stereo experiment, or snapshot a fitted stereo batch."
    _validate_gui_stereo_experiment(record)
    lines=["recipe: $(recipe_identity(record.recipe))","input: $(record.input_id)",
        "$(length(record.pairs)) ordered acquisitions; scaling mode=$(record.scaling_mode)"]
    function visit(value,path)
        if value isa AbstractDict
            foreach(k->visit(value[k],isempty(path) ? String(k) : "$path.$k"),sort!(collect(keys(value));by=String))
        elseif value isa AbstractVector && !isempty(value) && any(v->v isa Union{AbstractDict,AbstractVector},value)
            foreach(i->visit(value[i],"$path[$i]"),eachindex(value))
        else
            push!(lines,"$path: "*_stereo_summary_value(value))
        end
    end
    visit(record.recipe._data,"")
    visit(record.timing_metadata,"camera timing")
    for (i,pair) in enumerate(record.pairs)
        push!(lines,"acquisition $i: "*join([record.input_files[j]["path"] for j in pair]," | "))
    end
    push!(lines,"calibration provenance is supplied/unknown, not verified accuracy; replay never refits cameras")
    join(lines,"\n")
end
"""
    experiment_run_history(controller::StereoExperimentController) -> String

Recorded run IDs, status, persisted acquisition counts, output and original
failure summary. Selection is independent of the latest attempt.
"""
function experiment_run_history(ec::StereoExperimentController)
    record=ec.record[]
    record===nothing || isempty(record.runs) ? "No recorded stereo runs." :
        join(["$(r.run_id): $(r.status), $(r.completed_pairs)/$(length(record.pairs)) acquisitions, $(r.output)"*
            (r.error===nothing ? "" : "\n  $(r.error)") for r in record.runs],"\n")
end
