# Checkpoint execution lives in the framework-free controller layer.

"""
    CheckpointController(record::ExperimentRecord; checkpoint_dir="", output_dir="")
    CheckpointController(checkpoint::ExperimentCheckpoint)
    CheckpointController(path::AbstractString; output_dir=nothing)
    CheckpointController()

Keep a complete planar recipe and checkpoint apart from the batch form. The
`checkpoint`, `record`, directory selections, `progress` (`committed,total`),
`checkpoint_status`, `data_complete`, `running`, `state`, `status`, `error` and
`last_attempt` are Observables. State is initially `:empty` or `:recipe_ready`;
an opened store reflects its native attempt status. During execution it is
`:busy` or `:cancel_requested`. Native cancellation is distinct from failure.

`recover_interrupted` is an explicit stopped-writer assertion, initially false
and reset after opening/creating a store and after every start. There is no
environment override. Optional `resume_record` supplies equivalent relocated
input locators; the core checks exact recipe/ordered input identities and bytes.
No callbacks/scripts or recipe settings are reconstructed from widgets.

Use [`create_checkpoint!`](@ref), [`open_checkpoint!`](@ref), [`start!`](@ref),
[`cancel!`](@ref), [`refresh_checkpoint!`](@ref), [`checkpoint_explorer`](@ref)
and [`export_checkpoint_results!`](@ref). No result arrays are accumulated.
"""
struct CheckpointController
    checkpoint::Observable{Union{Nothing,ExperimentCheckpoint}}
    record::Observable{Union{Nothing,ExperimentRecord}}
    resume_record::Observable{Union{Nothing,ExperimentRecord}}
    checkpoint_dir::Observable{String}
    output_dir::Observable{String}
    export_path::Observable{String}
    recover_interrupted::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    checkpoint_status::Observable{Symbol}
    data_complete::Observable{Bool}
    running::Observable{Bool}
    state::Observable{Symbol}
    status::Observable{String}
    error::Observable{Any}
    last_attempt::Observable{Union{Nothing,CheckpointAttempt}}
    _cancel_token::Base.RefValue{Union{Nothing,Base.RefValue{Bool}}}
end

function _checkpoint_gui_record(record)
    snapshot=deepcopy(record)
    snapshot===nothing && return nothing
    _validate_gui_experiment(snapshot)
    snapshot
end

function CheckpointController(record::Union{Nothing,ExperimentRecord}=nothing;
                              checkpoint_dir::AbstractString="",output_dir::AbstractString="")
    snapshot=_checkpoint_gui_record(record)
    CheckpointController(Observable{Union{Nothing,ExperimentCheckpoint}}(nothing),
        Observable{Union{Nothing,ExperimentRecord}}(snapshot),
        Observable{Union{Nothing,ExperimentRecord}}(nothing),
        Observable(String(checkpoint_dir)),Observable(String(output_dir)),Observable(""),Observable(false),
        Observable((0,snapshot===nothing ? 0 : length(snapshot.pairs))),Observable(:ready),Observable(false),
        Observable(false),Observable(snapshot===nothing ? :empty : snapshot.recipe.external_preprocess===nothing ? :recipe_ready : :recipe_unsupported),
        Observable(snapshot!==nothing && snapshot.recipe.external_preprocess!==nothing ?
            "selected recipe references an external script; checkpoint creation is unsupported; open an existing store instead" : ""),
        Observable{Any}(nothing),Observable{Union{Nothing,CheckpointAttempt}}(nothing),
        Ref{Union{Nothing,Base.RefValue{Bool}}}(nothing))
end

function CheckpointController(cp::ExperimentCheckpoint)
    snapshot=deepcopy(cp)
    Hammerhead._checkpoint_record(snapshot.record)
    current,state=Hammerhead._checkpoint_fresh(snapshot)
    cc=CheckpointController()
    _adopt_checkpoint!(cc,current,state)
end
CheckpointController(path::AbstractString;output_dir=nothing)=
    CheckpointController(load_checkpoint(path;output_dir))

_checkpoint_idle(cc)=cc.running[] && throw(ArgumentError("checkpoint execution is busy; wait for the current pair to finish"))

function _checkpoint_gui_state!(cc,state,total)
    native=isempty(state.begins) ? :ready : haskey(state.endings,last(state.begins)["attempt_id"]) ?
        Symbol(state.endings[last(state.begins)["attempt_id"]]["status"]) : :unfinished
    complete=length(state.commits)==total
    cc.progress[]=(length(state.commits),total)
    cc.checkpoint_status[]=native
    cc.data_complete[]=complete
    cc.state[]=native
    cc.status[]="$(length(state.commits)) / $total committed; attempt: $native; data complete: $complete"
    cc
end

function _adopt_checkpoint!(cc,current,state)
    cc.checkpoint[]=current
    cc.record[]=deepcopy(current.record)
    cc.resume_record[]=nothing
    cc.checkpoint_dir[]=current.path
    cc.output_dir[]=current.output_dir
    cc.export_path[]=""
    cc.recover_interrupted[]=false
    cc.last_attempt[]=nothing
    cc.error[]=nothing
    _checkpoint_gui_state!(cc,state,length(current.record.pairs))
end

# The view may select a complete recipe for a new store. Validate before any
# observable replacement so refused records keep the previous store intact.
function _checkpoint_recipe!(cc,record::ExperimentRecord)
    _checkpoint_idle(cc)
    snapshot=_checkpoint_gui_record(record)
    cc.checkpoint[]=nothing
    cc.record[]=snapshot
    cc.resume_record[]=nothing
    cc.checkpoint_dir[]=""
    cc.output_dir[]=""
    cc.export_path[]=""
    cc.recover_interrupted[]=false
    cc.progress[]=(0,length(snapshot.pairs))
    cc.checkpoint_status[]=:ready
    cc.data_complete[]=false
    cc.last_attempt[]=nothing
    cc.error[]=nothing
    cc.state[]=snapshot.recipe.external_preprocess===nothing ? :recipe_ready : :recipe_unsupported
    cc.status[]=snapshot.recipe.external_preprocess===nothing ?
        "complete recipe selected; choose separate new/empty directories" :
        "selected recipe references an external script; checkpoint creation is unsupported; open an existing store instead"
    cc
end

"""
    create_checkpoint!(controller, record=controller.record[];
                       checkpoint_dir=controller.checkpoint_dir[], output_dir=controller.output_dir[])

Create a new store from a deep copy of the complete recipe and selected separate
metadata/result directories. Built-ins only; core input, environment and alias
guards apply. Failed creation preserves the previous controller snapshot and
never replaces existing store data. Directory selections are not recipe options.
"""
function create_checkpoint!(cc::CheckpointController,record=cc.record[];
                            checkpoint_dir::AbstractString=cc.checkpoint_dir[],output_dir::AbstractString=cc.output_dir[])
    _checkpoint_idle(cc)
    snapshot=_checkpoint_gui_record(record)
    snapshot===nothing && throw(ArgumentError("choose a complete experiment recipe first"))
    snapshot.recipe.external_preprocess===nothing ||
        throw(ArgumentError("checkpoints support built-in preprocessing only; script/callback state cannot be resumed"))
    isempty(strip(checkpoint_dir)) && throw(ArgumentError("choose a checkpoint metadata directory"))
    isempty(strip(output_dir)) && throw(ArgumentError("choose a separate per-pair output directory"))
    cp=create_checkpoint(String(checkpoint_dir),snapshot;output_dir=String(output_dir))
    current,state=Hammerhead._checkpoint_fresh(cp)
    _adopt_checkpoint!(cc,current,state)
end

"""
    open_checkpoint!(controller, path_or_checkpoint; output_dir=nothing)

Open and fully verify a store/prefix before replacing controller state. An
explicitly relocated result directory is supported. Original images need not
exist just to inspect. Recovery permission is reset; opening never resumes or
recovers a writer automatically. Loading failures preserve the previous state.
"""
function open_checkpoint!(cc::CheckpointController,path::AbstractString;output_dir=nothing)
    _checkpoint_idle(cc)
    cp=load_checkpoint(path;output_dir)
    current,state=Hammerhead._checkpoint_fresh(cp)
    _adopt_checkpoint!(cc,current,state)
end
function open_checkpoint!(cc::CheckpointController,cp::ExperimentCheckpoint;output_dir=nothing)
    _checkpoint_idle(cc)
    snapshot=deepcopy(cp)
    Hammerhead._checkpoint_record(snapshot.record)
    if output_dir!==nothing
        candidate=load_checkpoint(snapshot.path;output_dir)
        candidate.checkpoint_id==snapshot.checkpoint_id &&
            recipe_identity(candidate.record.recipe)==recipe_identity(snapshot.record.recipe) &&
            candidate.record.input_id==snapshot.record.input_id || throw(ArgumentError("checkpoint identities differ"))
        snapshot=candidate
    end
    current,state=Hammerhead._checkpoint_fresh(snapshot)
    _adopt_checkpoint!(cc,current,state)
end

"""
    refresh_checkpoint!(controller)

Verify and refresh native progress/status while idle, preserving the selected
equivalent input record and export destination. This reads all committed bytes;
it is an explicit refresh, not a per-pair status poll or live result reader.
"""
function refresh_checkpoint!(cc::CheckpointController)
    _checkpoint_idle(cc)
    cp=cc.checkpoint[]
    cp===nothing && throw(ArgumentError("open or create a checkpoint first"))
    current,state=Hammerhead._checkpoint_fresh(deepcopy(cp))
    cc.checkpoint[]=current
    _checkpoint_gui_state!(cc,state,length(current.record.pairs))
end

"""
    start!(controller::CheckpointController; async=true, recover_interrupted=nothing,
           record=nothing, progress=nothing)

Resume the captured checkpoint and exact recipe/input record from its committed
prefix. Capture all execution values before Observable notifications or yielding.
`record` may supply equivalent relocated input locators; arbitrary recipe
overrides are unsupported. Recovery uses the explicit argument or the current
unchecked-by-default assertion, then resets it for the next invocation.

Progress receives absolute `(committed,total)` counts after descriptor publication
and yields between commits. No payloads or full-store polls occur in this
callback. Cancellation acknowledges after the pair in flight; the final pair is
completion. Callback exceptions are failures. Failures retain their original
exception in `error`, plus the verified committed prefix where available.
Preflight/current-pair processing may pause rendering; there is no in-pass
progress, immediate interruption or worker-thread Observable mutation.
"""
function start!(cc::CheckpointController;async::Bool=true,recover_interrupted::Union{Nothing,Bool}=nothing,
                record::Union{Nothing,ExperimentRecord}=nothing,progress::Union{Nothing,Function}=nothing,_phase_hook=nothing)
    cc.running[] && return cc
    cp=deepcopy(cc.checkpoint[])
    candidate=record===nothing ? cc.resume_record[] : record
    snapshot=deepcopy(candidate===nothing ? (cp===nothing ? cc.record[] : cp.record) : candidate)
    recover=recover_interrupted===nothing ? cc.recover_interrupted[] : recover_interrupted
    token=Ref(false)
    cc._cancel_token[]=token
    cc.running[]=true
    cc.recover_interrupted[]=false
    cc.error[]=nothing
    cc.last_attempt[]=nothing
    cc.state[]=token[] ? :cancel_requested : :busy
    cc.status[]=token[] ? "cancellation requested; waiting for input/prefix verification" :
        "verifying inputs and committed prefix; cancellation is checked at pair boundaries"
    run=()->_run_checkpoint!(cc,cp,snapshot,recover,token,progress,_phase_hook)
    async ? errormonitor(@async run()) : run()
    cc
end

function _run_checkpoint!(cc,cp,record,recover,token,observer,hook)
    try
        cp===nothing && throw(ArgumentError("open or create a checkpoint first"))
        delivery=(done,total)->begin
            cc.progress[]=(done,total)
            cc.data_complete[]=done==total
            cc.status[]="$done / $total committed"*(token[] ? "; cancellation requested" : "")
            observer===nothing || observer(done,total)
            yield()
        end
        attempt=resume_checkpoint!(cp,record;recover_interrupted=recover,cancel=()->token[],progress=delivery,_phase_hook=hook)
        _checkpoint_execution_snapshot!(cc,cp,record)
        cc.last_attempt[]=attempt
        cc.progress[]=(attempt.committed,attempt.total_pairs)
        cc.data_complete[]=attempt.committed==attempt.total_pairs
        cc.checkpoint_status[]=attempt.status
        cc.state[]=attempt.status
        cc.status[]="$(attempt.committed) / $(attempt.total_pairs) committed; attempt: $(attempt.status); data complete: $(cc.data_complete[])"
    catch err
        cc.error[]=err
        if cp!==nothing
            try
                _checkpoint_execution_snapshot!(cc,cp,record)
                state=checkpoint_state(cp)
                cc.progress[]=(state.committed,state.total_pairs)
                cc.checkpoint_status[]=state.status
                cc.data_complete[]=state.data_complete
            catch
                # Do not replace the original error with a damaged-store scan.
            end
        end
        cc.state[]=:failed
        cc.status[]="failed: $(_errmsg(err)); committed progress: $(cc.progress[])"
    finally
        cc._cancel_token[]=nothing
        cc.recover_interrupted[]=false
        cc.running[]=false
    end
    cc
end

function _checkpoint_execution_snapshot!(cc,cp,record)
    cc.checkpoint[]=cp
    cc.record[]=deepcopy(cp.record)
    cc.resume_record[]=deepcopy(record)
    cc.checkpoint_dir[]=cp.path
    cc.output_dir[]=cp.output_dir
end

"""
    cancel!(controller::CheckpointController)

Request native cancellation at the next committed pair boundary. The independent
per-run token remains set even if display Observables change. An idle request is
a no-op. Cancellation after the final commit records completion, not failure.
"""
function cancel!(cc::CheckpointController)
    token=cc._cancel_token[]
    cc.running[] && token!==nothing || return cc
    token[]=true
    cc.state[]=:cancel_requested
    cc.status[]="cancellation requested; waiting for the current pair to commit"
    cc
end

"""
    checkpoint_explorer(controller) -> ResultExplorer

Verify an idle checkpoint and open its nonempty committed prefix lazily, keeping
one physical display frame. The index length is fixed; resume does not append to
an existing explorer. Open another explorer after explicit refresh to see later
commits. Read errors preserve the previous displayed frame.
"""
function checkpoint_explorer(cc::CheckpointController)
    _checkpoint_idle(cc)
    cp=cc.checkpoint[]
    cp===nothing && throw(ArgumentError("open or create a checkpoint first"))
    ResultExplorer(checkpoint_results(cp);path=cp.path)
end

"""
    export_checkpoint_results!(controller, path=controller.export_path[]) -> path

Stream complete data to a fresh ordinary native aggregate outside both owned
directories. Core alias, existing-destination and exclusive export-lock guards
apply. Failure preserves the checkpoint and previous controller selections.
A terminated export's remaining lock requires a different fresh destination.
"""
function export_checkpoint_results!(cc::CheckpointController,path::AbstractString=cc.export_path[])
    _checkpoint_idle(cc)
    cp=deepcopy(cc.checkpoint[])
    cp===nothing && throw(ArgumentError("open or create a checkpoint first"))
    isempty(strip(path)) && throw(ArgumentError("choose a fresh native aggregate destination"))
    output=save_checkpoint_results(String(path),cp)
    cc.export_path[]=String(path)
    cc.status[]="native aggregate saved: $output"
    output
end
