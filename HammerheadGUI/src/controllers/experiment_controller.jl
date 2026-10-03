# Dedicated lossless recipe lane. No Makie or filesystem dialogs here.

"""
    experiment_record(batch::BatchRunner; script_reference=nothing) -> ExperimentRecord

Snapshot a file-based planar batch into the core experiment format. Resolve
effort presets into their exact effective pass schedule using full-image/ROI
dimensions; reject pair sizes that would require different schedules. Preserve
the form's mask, ROI, scale and built-in preprocessing snapshot. Current batch
execution uses CPU, Float64, and the core's current thread default.

A bare or directly assigned preprocessing callback requires an explicit
`ScriptReference`; its implementation is never serialized. In-memory images,
running batches, and an unrelated script reference are rejected. Reopened core
recipes use [`ExperimentController`](@ref), not this narrow form projection.
"""
function experiment_record(bc::BatchRunner; script_reference::Union{Nothing,ScriptReference}=nothing)
    bc.running[] && throw(ArgumentError("wait for the batch to finish before saving its settings"))
    pairs = deepcopy(frame_pairs(bc))
    isempty(pairs) && throw(ArgumentError("add image-file pairs first"))
    all(p -> all(f -> f isa AbstractString, p), pairs) ||
        throw(ArgumentError("experiment records require image-file paths; in-memory frames are unsupported"))
    roi, mask, scale = deepcopy(bc.roi[]), deepcopy(bc.mask[]), build_scale(bc)
    effort = bc.effort[]
    passes = if effort === :custom
        build_parameters(bc)
    else
        schedules = [Hammerhead.effort_schedule(effort;
            image_size=roi === nothing ? size(load_image(p[1])) : (length(roi.rows),length(roi.cols))) for p in pairs]
        first_id = recipe_identity(PIVRecipe(first(schedules)))
        all(s -> recipe_identity(PIVRecipe(s))==first_id,schedules) ||
            throw(ArgumentError("pair sizes produce different effort schedules; use an explicit custom schedule"))
        first(schedules)
    end
    callback, metadata = bc.preprocess[], bc.preprocess_snapshot[]
    steps = if callback === nothing
        script_reference === nothing || throw(ArgumentError("a script reference requires a custom preprocessing callback"))
        PreprocessStep[]
    elseif metadata !== nothing && metadata.callback === callback
        script_reference === nothing || throw(ArgumentError("built-in preprocessing does not need a script reference"))
        recipe_identity(PIVRecipe(PIVParameters();preprocessing=metadata.steps))==metadata.identity ||
            throw(ArgumentError("built-in preprocessing metadata changed; attach the preview again"))
        deepcopy(metadata.steps)
    else
        script_reference === nothing && throw(ArgumentError("custom preprocessing requires an explicit ScriptReference to save; functions are not serialized"))
        PreprocessStep[]
    end
    recipe = PIVRecipe(passes; preprocessing=steps,external_preprocess=script_reference,
        roi,mask,scale,image_type=Float64,backend=:cpu,threaded=Threads.nthreads()>1)
    ExperimentRecord(pairs,recipe)
end

"""
    save_batch_experiment(path, batch::BatchRunner; script_reference=nothing) -> ExperimentRecord

Build and save an exact supported snapshot of the current batch settings.
Preflight rejects unsupported callbacks/inputs before replacing the destination.
"""
function save_batch_experiment(path::AbstractString,bc::BatchRunner; kwargs...)
    record = experiment_record(bc; kwargs...)
    save_experiment(path,record)
    record
end

"""
    ExperimentController(record_or_path; output_path="", run_record_path="")
    ExperimentController()

Keep a deep-copied complete core recipe apart from the batch form. Observable
state includes `record`, `running`, `progress` (`written,total`), `state`
(`:empty`, `:ready`, `:busy`, `:cancel_requested`, `:cancelled`, `:completed`,
`:failed`), `status`, `last_run`, and `error`. `output_path`
selects native results; optional `run_record_path` saves success/failure history.
`allow_environment_change` is an explicit opt-in, initially false.
`custom_preprocess` is caller-supplied for a recorded script and never loaded
from it. No recipe fields are reconstructed from GUI widgets.

Use [`open_experiment!`](@ref), [`save_experiment_record!`](@ref), [`start!`](@ref)
[`cancel!`](@ref) and [`experiment_results`](@ref). Replay starts from the first
pair, with progress after native pair writes and cancellation at those boundaries.
A GUI cancelled attempt is a failed v1 core run with a readable cancellation
reason when run history is persisted; without a history destination its failed
run metadata is unavailable. This is not checkpoint/resume support. Metadata-only
histories are kept; completed result browsing is lazy. Cooperative scheduling
yields between pairs, but preflight/current computation/I/O can pause rendering.
"""
struct ExperimentController
    record::Observable{Union{Nothing,ExperimentRecord}}
    output_path::Observable{String}
    run_record_path::Observable{String}
    allow_environment_change::Observable{Bool}
    custom_preprocess::Observable{Union{Nothing,Function}}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    last_run::Observable{Union{Nothing,ExperimentRun}}
    error::Observable{Any}
    progress::Observable{Tuple{Int,Int}}
    _cancel_token::Base.RefValue{Union{Nothing,Base.RefValue{Bool}}}
    _task::Base.RefValue{Union{Nothing,Task}}
end

function _validate_gui_experiment(record::ExperimentRecord)
    # Reuse the versioned core contract for direct objects as well as file loads,
    # without requiring the original input files to be present just to inspect.
    Hammerhead._experiment_preflight(record)
    Hammerhead._experiment_validate_environment(record.creation_environment)
    for run in record.runs
        Hammerhead._experiment_run(Hammerhead._experiment_run_data(run),record)
    end
    record
end

function ExperimentController(record::Union{Nothing,ExperimentRecord}=nothing;
                              output_path::AbstractString="",run_record_path::AbstractString="")
    snapshot = deepcopy(record)
    snapshot === nothing || _validate_gui_experiment(snapshot)
    latest = snapshot===nothing || isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    ExperimentController(Observable{Union{Nothing,ExperimentRecord}}(snapshot),
        Observable(String(output_path)),Observable(String(run_record_path)),Observable(false),
        Observable{Union{Nothing,Function}}(nothing),Observable(false),
        Observable(snapshot===nothing ? :empty : :ready),Observable(""),
        Observable{Union{Nothing,ExperimentRun}}(latest),Observable{Any}(nothing),
        Observable((0,snapshot===nothing ? 0 : length(snapshot.pairs))),
        Ref{Union{Nothing,Base.RefValue{Bool}}}(nothing),Ref{Union{Nothing,Task}}(nothing))
end
ExperimentController(path::AbstractString;run_record_path::AbstractString=path,kwargs...) =
    ExperimentController(load_experiment(path);run_record_path,kwargs...)

_experiment_idle(ec) = ec.running[] && throw(ArgumentError("experiment replay is busy; wait for it to finish"))

"""
    open_experiment!(controller, path_or_record)

Replace the complete recipe snapshot after successful loading/validation.
Loading failures preserve the previous record. No saved scripts are executed.
"""
function open_experiment!(ec::ExperimentController,record::ExperimentRecord)
    _experiment_idle(ec)
    snapshot = deepcopy(record)
    _validate_gui_experiment(snapshot)
    ec.output_path[] = ""
    ec.run_record_path[] = ""
    ec.custom_preprocess[] = nothing
    ec.allow_environment_change[] = false
    ec.last_run[] = isempty(snapshot.runs) ? nothing : last(snapshot.runs)
    ec.progress[] = (0,length(snapshot.pairs))
    ec.error[] = nothing
    ec.state[] = :ready
    ec.status[] = "recipe loaded; settings are read-only"
    ec.record[] = snapshot
    ec
end
function open_experiment!(ec::ExperimentController,path::AbstractString)
    _experiment_idle(ec)
    record = load_experiment(path)
    open_experiment!(ec,record)
    ec.run_record_path[] = String(path)
    ec
end

"""
    save_experiment_record!(controller, path)

Save the intact recipe/history, keeping the saved location as the run-record
destination. Inputs/scripts/results are protected by the core preflight.
"""
function save_experiment_record!(ec::ExperimentController,path::AbstractString)
    _experiment_idle(ec)
    ec.record[] === nothing && throw(ArgumentError("open or snapshot an experiment first"))
    save_experiment(path,ec.record[])
    notify(ec.record)
    ec.run_record_path[] = String(path)
    ec.status[] = "experiment saved"
    ec
end

"""
    start!(controller::ExperimentController; async=true, progress=nothing)

Replay the captured complete recipe to the selected result file. Capture all
execution state before notifying observers or scheduling work. Errors are
exposed in `error`/`status` with `state=:failed`; completed output uses
`state=:completed`. Core preflight preserves existing files on rejection.
With a run-record destination, reread appended history after execution, even
on processing failure. The original replay exception remains the reported
error if rereading history also fails. No arbitrary recipe overrides occur.
Optional `progress(written,total)` is captured with the request and runs after
the writer progress notification. Its exceptions remain failures. The GUI yields
between pair writes, then checks an independent per-run cancellation token;
cancellation after the final write resolves as completion. Cancellation before
scheduled execution starts leaves files/history unchanged. Once replay starts,
there is no in-pass/preflight interruption: a request waits for the first/next
pair write, prefetched loading cleanup, closure, hashing and history handling.
`running` stays true until all handling finishes. No worker-thread Observable
mutation or event-loop responsiveness is promised by `async=true`.
Startup/progress notification exceptions become workflow failures. Terminal
notifications assign final state while suppressing observer exceptions, preserving
the completed outcome or original replay error and reliably releasing busy state.
"""
function start!(ec::ExperimentController; async::Bool=true,progress::Union{Nothing,Function}=nothing)
    ec.running[] && return ec
    record = deepcopy(ec.record[])
    output, history = ec.output_path[],ec.run_record_path[]
    allow, custom = ec.allow_environment_change[],ec.custom_preprocess[]
    token=Ref(false)
    ec._cancel_token[]=token
    try
        # This is the first notification: observers cannot reenter idle actions.
        ec.running[] = true
        ec.error[] = nothing
        ec.last_run[] = nothing
        ec.progress[] = (0,record===nothing ? 0 : length(record.pairs))
        ec.state[] = token[] ? :cancel_requested : :busy
        ec.status[] = token[] ? "cancellation requested; replay has not started" :
            "replaying; validation/current-pair work may pause rendering"
        run = () -> _replay_gui_experiment!(ec,record,output,history,allow,custom,token,progress)
        if async
            task=Task(run)
            ec._task[]=task
            errormonitor(task)
            schedule(task)
        else
            run()
        end
    catch err
        # No replay was scheduled if startup failed. Terminal observers may
        # also throw; their exceptions cannot strand busy state or replace err.
        _experiment_notify_safely!(ec.last_run,nothing)
        _experiment_notify_safely!(ec.progress,(0,record===nothing ? 0 : length(record.pairs)))
        _experiment_notify_safely!(ec.error,err)
        _experiment_notify_safely!(ec.state,:failed)
        _experiment_notify_safely!(ec.status,"failed to start: $(_errmsg(err))")
        _experiment_replay_cleanup!(ec)
    end
    ec
end

function _experiment_notify_safely!(observable,value)
    try
        observable[]=value # assignment occurs before listeners are notified
    catch
        # A terminal notification must not replace a replay/startup exception.
    end
    nothing
end
function _experiment_replay_cleanup!(ec)
    ec._cancel_token[]=nothing
    ec._task[]=nothing
    _experiment_notify_safely!(ec.running,false)
end

struct _ExperimentCancelled <: Exception
    written::Int
    total::Int
end
Base.showerror(io::IO,err::_ExperimentCancelled)=
    print(io,"Experiment replay cancelled after ",err.written," of ",err.total," written pairs")

"""
    cancel!(controller::ExperimentController)

Request cancellation after the current pair is written. Idle requests are no-ops.
The captured per-run token cannot be cleared by editing display Observables.
The GUI remains busy until pending loading/output/history cleanup finishes.
Cancellation after the final write means completion; earlier cancellation may
leave a native written prefix and a failed v1 core run record, not a resumable
checkpoint. Before scheduled execution starts no output/run record is written.
"""
function cancel!(ec::ExperimentController)
    token=ec._cancel_token[]
    ec.running[] && token!==nothing || return ec
    token[]=true
    ec.state[]=:cancel_requested
    ec.status[]="cancellation requested; waiting for a written-pair boundary and cleanup"
    ec
end

function _replay_gui_experiment!(ec,record,output,history,allow,custom,token,observer)
    total=record===nothing ? 0 : length(record.pairs)
    written=Ref(0)
    try
        record === nothing && throw(ArgumentError("open or snapshot an experiment first"))
        isempty(strip(output)) && throw(ArgumentError("choose a result output file first"))
        token[] && throw(_ExperimentCancelled(0,total))
        delivery=(i,n)->begin
            written[]=i # independent of mutable progress display state
            ec.progress[]=(i,n)
            ec.status[]="$i / $n pairs written"*(token[] ? "; cancellation requested" : "")
            observer===nothing || observer(i,n)
            yield()
            token[] && i<n && throw(_ExperimentCancelled(i,n))
            nothing
        end
        run = replay_experiment(record;output,run_record=isempty(history) ? nothing : history,
            allow_environment_change=allow,custom_preprocess=custom,progress=delivery)
        if isempty(history)
            push!(record.runs,run)
        else
            record=load_experiment(history)
            !isempty(record.runs) && last(record.runs).run_id==run.run_id ||
                throw(ArgumentError("saved run history changed before it could be reopened"))
        end
        _experiment_notify_safely!(ec.record,record)
        _experiment_notify_safely!(ec.last_run,run)
        _experiment_notify_safely!(ec.progress,(run.completed_pairs,total))
        _experiment_notify_safely!(ec.state,:completed)
        _experiment_notify_safely!(ec.status,"completed: $(run.completed_pairs) written pairs")
    catch err
        # Core has drained prefetch/closed output/recorded failure before this
        # catch. Keep busy/requested state until history recovery also finishes.
        record===nothing || _experiment_notify_safely!(ec.record,record)
        if record !== nothing && !isempty(history) && isfile(history)
            try
                updated = load_experiment(history)
                if updated.input_id==record.input_id && recipe_identity(updated.recipe)==recipe_identity(record.recipe) &&
                   length(updated.runs)>length(record.runs)
                    _experiment_notify_safely!(ec.record,updated)
                    _experiment_notify_safely!(ec.last_run,isempty(updated.runs) ? nothing : last(updated.runs))
                end
            catch
                # Preserve the original exception, including failed metadata saves.
            end
        end
        _experiment_notify_safely!(ec.progress,(written[],total))
        _experiment_notify_safely!(ec.error,err)
        _experiment_notify_safely!(ec.state,err isa _ExperimentCancelled ? :cancelled : :failed)
        _experiment_notify_safely!(ec.status,err isa _ExperimentCancelled ? sprint(showerror,err) : "failed: $(_errmsg(err))")
    finally
        _experiment_replay_cleanup!(ec)
    end
    ec
end

"""
    experiment_results(controller) -> ResultExplorer

Open the latest completed run lazily, retaining one display frame. Verify its
recorded output SHA-256 before browsing; refuse changed/unverifiable output,
busy replay, or a failed latest run. Concurrent result-file mutation is unsupported.
"""
function experiment_results(ec::ExperimentController)
    _experiment_idle(ec)
    run = ec.last_run[]
    run !== nothing && run.status===:completed || throw(ArgumentError("no completed experiment run to explore"))
    record=ec.record[]
    record!==nothing || throw(ArgumentError("no current experiment record"))
    _validate_gui_experiment(record)
    run.recipe_id==recipe_identity(record.recipe) && run.input_id==record.input_id ||
        throw(ArgumentError("completed run identities do not match the current experiment"))
    Hammerhead._experiment_run(Hammerhead._experiment_run_data(run),record)
    run.output_sha256!==nothing && isfile(run.output) ||
        throw(ArgumentError("completed output is missing or has no recorded content identity"))
    digest=open(io->bytes2hex(SHA.sha256(io)),run.output,"r")
    digest==run.output_sha256 || throw(ArgumentError("completed result output changed since the recorded run; choose an intact run output"))
    ResultExplorer(run.output;lazy=true)
end

"""
    experiment_summary(controller) -> String

Read-only description of every pass field and recipe option. Embedded arrays
are summarized by size/type; their exact values remain in the core record.
"""
function experiment_summary(ec::ExperimentController)
    record = ec.record[]
    record===nothing && return "Open an experiment, or snapshot the current batch."
    r=record.recipe
    lines=["recipe: $(recipe_identity(r))", "input: $(record.input_id)",
        "$(length(record.pairs)) pairs; backend=$(r.backend), precision=$(r.image_type), threaded=$(r.threaded)",
        "predictor_smoothing=$(r.predictor_smoothing), mask_threshold=$(r.mask_threshold), uncertainty_backend=$(r.uncertainty_backend)",
        "ROI: $(r.roi)","mask: $(r.mask===nothing ? "none" : "$(size(r.mask)), $(count(r.mask)) excluded pixels")",
        "scale: $(r.scale)","created: Julia $(record.creation_environment["julia_version"]), Hammerhead $(record.creation_environment["hammerhead_version"])"]
    for (i,pass) in enumerate(r.passes)
        push!(lines,"pass $i: "*join(["$k=$(repr(getfield(pass,k)))" for k in fieldnames(PIVParameters)],", "))
    end
    for (i,step) in enumerate(r.preprocessing)
        options=join(["$k="*(k=="background" ? "$(size(v)) $(eltype(v)) array" : repr(v)) for (k,v) in sort!(collect(step.options);by=first)],", ")
        push!(lines,"preprocess $i: $(step.operation) ($options)")
    end
    isempty(r.preprocessing) && push!(lines,"preprocessing: none")
    ref=r.external_preprocess
    push!(lines,ref===nothing ? "custom script: none" : "custom script: $(ref.path); $(ref.entrypoint); SHA-256 $(ref.sha256) (caller function required)")
    join(lines,"\n")
end

"""
    experiment_quality_report(controller; include_measurement_history=false) -> RunQualityReport

Summarize the latest completed run using the core report contract. Verify the
captured record/run identities and output content before scanning one result at
a time. The report distinguishes current flags and finite output from unavailable
measurement/replacement history; stored uncertainty availability is not accuracy.
Busy, missing, failed, or changed runs are refused. No input images or saved
scripts are executed to generate the report.
Opt into `include_measurement_history=true` for core format-v2 verified
final-sweep event counts; the default remains the stored-field v1 report.
"""
function experiment_quality_report(ec::ExperimentController;include_measurement_history::Bool=false)
    _experiment_idle(ec)
    record,run=deepcopy(ec.record[]),deepcopy(ec.last_run[])
    record!==nothing && run!==nothing ||
        throw(ArgumentError("no completed experiment run to report"))
    quality_report(record,run;include_measurement_history)
end

"""
    save_experiment_quality_report(path, controller; include_measurement_history=false) -> RunQualityReport

Generate and save the shared core TOML quality report. In addition to the core
report's protected sources, preserve the controller's selected result and run
record destinations. A rejected report or destination leaves existing files
unchanged. This scans completed output and does not modify the experiment.
"""
function save_experiment_quality_report(path::AbstractString,ec::ExperimentController;include_measurement_history::Bool=false)
    _experiment_idle(ec)
    protected=filter(!isempty,[ec.output_path[],ec.run_record_path[]])
    report=experiment_quality_report(ec;include_measurement_history)
    save_quality_report(path,report;protected_paths=protected)
    report
end

"""
    experiment_run_history(controller) -> String

Read-only run IDs, statuses, pair counts, output locations and failure summaries.
"""
function experiment_run_history(ec::ExperimentController)
    record=ec.record[]
    record===nothing || isempty(record.runs) ? "No recorded runs." :
        join(["$(run.run_id): $(run.status), $(run.completed_pairs)/$(length(record.pairs)) pairs, $(run.output)" *
            (run.error===nothing ? "" : "\n  $(run.error)") for run in record.runs],"\n")
end
