# Included inside Prototype: the shell owns this lane, not a viewport lease.
mutable struct ExperimentLane
    controller::ExperimentController
    error::Observable{String}
    identity::Observable{String}
    written::Observable{String}
    section::Observable{Symbol}
    page::Observable{Int}
    text::Observable{String}
    pages::Observable{String}
    subscriptions::Vector{Any}
    recipe_text::String
    history_text::String
    job::Any
    request::Any
    observer::Union{Nothing,Function}
    pending_progress::Any
    owner_error::Any
    outcome::Any
    cancel_requested::Bool
    owner_thread::Int
end

function wrapped_pages(text;columns=40,lines=8)
    rows=String[]
    for line in split(text,'\n')
        chars=collect(line)
        isempty(chars) && push!(rows,"")
        for i in 1:columns:length(chars)
            push!(rows,String(chars[i:min(i+columns-1,end)]))
        end
    end
    [join(rows[i:min(i+lines-1,end)],'\n') for i in 1:lines:length(rows)]
end
function refresh_experiment(lane;record_changed=false)
    ec=lane.controller
    record=ec.record[]
    if record_changed
        lane.recipe_text=experiment_summary(ec)
        lane.history_text=experiment_run_history(ec)
    end
    lane.identity[]=lane.request===nothing ? (record===nothing ? "No saved experiment open" :
        "Configured recipe: $(recipe_identity(record.recipe))\nInput: $(record.input_id)") :
        "Captured replay recipe: $(lane.request.record.recipe.recipe_id)\nInput: $(lane.request.record.input_id)\nOutput: $(lane.request.output)"
    lane.written[]="Written pairs: $(ec.progress[][1]) / $(ec.progress[][2])"
    details=lane.section[]===:history ? lane.history_text : lane.recipe_text
    full="State: $(ec.state[])\nStatus: $(ec.status[])\nResult output: $(ec.output_path[])\nRun record: $(ec.run_record_path[])\n"*
        "Replay starts at pair 1; cancel waits for writes/cleanup.\n\n"*details
    isempty(lane.error[]) || (full="Error: $(lane.error[])\n"*full)
    lane.request===nothing || (full="Captured output: $(lane.request.output)\nCaptured run record: $(lane.request.history)\nEnvironment override: $(lane.request.allow)\n"*full)
    chunks=wrapped_pages(full)
    lane.page[]=clamp(lane.page[],1,length(chunks))
    lane.text[]=chunks[lane.page[]]
    lane.pages[]="Page $(lane.page[]) / $(length(chunks))"
    nothing
end
function ExperimentLane()
    ec=ExperimentController()
    lane=ExperimentLane(ec,Observable(""),Observable(""),Observable(""),
        Observable(:recipe),Observable(1),Observable(""),Observable(""),Any[],"","",
        nothing,nothing,nothing,nothing,nothing,nothing,false,Threads.threadid())
    push!(lane.subscriptions,on(_->refresh_experiment(lane;record_changed=true),ec.record))
    for source in (ec.status,ec.state,ec.progress,ec.output_path,ec.run_record_path,lane.section,lane.error)
        push!(lane.subscriptions,on(_->refresh_experiment(lane),source))
    end
    refresh_experiment(lane;record_changed=true)
    lane
end
function experiment_action(state,action)
    lane=state.experiment
    try
        state.shutdown && throw(ArgumentError("shell shutdown requested; wait for cleanup"))
        lane.controller.running[] && throw(ArgumentError("saved replay is busy; wait for cleanup"))
        state.batch.running[] && throw(ArgumentError("demo is running; wait for it to finish"))
        action()
        lane.error[]=""
        refresh_experiment(lane)
        true
    catch err
        lane.error[]=message(err)
        false
    end
end
function open_saved_experiment(state,path)
    experiment_action(state,()->begin
        open_experiment!(state.experiment.controller,String(path))
        state.experiment.page[]=1
    end)
end
function configure_saved_experiment(state,output,history,allow)
    experiment_action(state,()->begin
        ec=state.experiment.controller
        ec.output_path[]=String(output)
        ec.run_record_path[]=String(history)
        ec.allow_environment_change[]=Bool(allow)
    end)
end
function run_saved_experiment(state;progress=nothing,start_options=NamedTuple())
    experiment_action(state,()->begin
        lane=state.experiment;ec=lane.controller
        record=deepcopy(ec.record[])
        record===nothing && throw(ArgumentError("open a saved experiment first"))
        record.recipe.external_preprocess===nothing ||
            throw(ArgumentError("referenced scripts can be inspected; this shell does not load or execute them"))
        ec.custom_preprocess[]===nothing || throw(ArgumentError("custom preprocessing callbacks are unsupported in the subprocess shell"))
        isempty(strip(ec.output_path[])) && throw(ArgumentError("choose a result output first"))
        # Capture before running/status observers or spawning the child. Only the
        # owner mutates controller observables; the child sees a native snapshot.
        options=deepcopy(start_options) # private finite-harness seam, captured like user choices
        lane.request=(record=record,output=String(ec.output_path[]),history=String(ec.run_record_path[]),allow=ec.allow_environment_change[],start_options=options)
        lane.observer=progress;lane.pending_progress=nothing;lane.owner_error=nothing
        lane.outcome=nothing;lane.cancel_requested=false
        ec.running.val=true;ec.state.val=:busy;ec.error.val=nothing;ec.last_run.val=nothing
        ec.progress.val=(0,length(record.pairs));ec.status.val="starting saved replay subprocess"
        try
            for source in (ec.running,ec.state,ec.error,ec.last_run,ec.progress,ec.status)
                notify(source)
            end
            request=lane.request
            lane.job=ReplayWorkerClient.start_replay(request.record;output=request.output,
                run_record=isempty(request.history) ? nothing : request.history,
                allow_environment_change=request.allow,initial_cancel=lane.cancel_requested,request.start_options...)
        catch error
            lane.owner_error=error;lane.observer=nothing
            _saved_notify!(ec.error,error);_saved_notify!(ec.state,:failed)
            pending=ReplayWorkerClient.startup_cleanup_pending()
            pending || (lane.request=nothing)
            _saved_notify!(ec.status,"failed to start saved subprocess: $(sprint(showerror,error))"*
                (pending ? "; still waiting for owned startup process cleanup" : ""))
            _saved_notify!(ec.running,pending)
            rethrow()
        end
    end)
end
_saved_notify!(source,value)=HammerheadGUI.Controllers._experiment_notify_safely!(source,value)
function cancel_saved_experiment(state)
    lane=state.experiment;ec=lane.controller
    ec.running[] || return nothing
    lane.cancel_requested=true
    lane.job===nothing || ReplayWorkerClient.request_cancel!(lane.job)
    _saved_notify!(ec.state,:cancel_requested)
    _saved_notify!(ec.status,"cancel requested; waiting for a written-pair boundary and subprocess cleanup")
    nothing
end
function acknowledge_saved_progress!(state;abort=nothing)
    lane=state.experiment;event=lane.pending_progress
    event===nothing && return false
    ReplayWorkerClient.outcome(lane.job)===nothing || return false
    ReplayWorkerClient.ownership_error(lane.job)===nothing || return false
    ReplayWorkerClient.acknowledge_progress!(lane.job,event;abort)
    lane.pending_progress=nothing
    true
end
function _adopt_saved_replay_outcome!(state,outcome)
    lane=state.experiment;ec=lane.controller
    record=deepcopy(lane.request.record)
    run=outcome.run
    if run!==nothing
        run=Hammerhead._experiment_run(Hammerhead._experiment_run_data(run),record)
        position=findfirst(r->r.run_id==run.run_id,record.runs)
        position===nothing ? push!(record.runs,run) : (record.runs[position]=run)
    end
    # Detached terminal metadata, never result arrays. A completed core run is
    # retained even when saving its requested history failed separately.
    ec.record.val=record;ec.last_run.val=run
    ec.progress.val=(outcome.written,outcome.total)
    history_error=outcome.history_error
    history_failed=history_error!==nothing && !isempty(string(history_error))
    ec.error.val=lane.owner_error===nothing ? (outcome.status===:completed ?
        (history_failed ? ErrorException("run history save failed: $history_error") : nothing) : ErrorException(outcome.message)) : lane.owner_error
    lane.error.val=ec.error[]===nothing ? "" : sprint(showerror,ec.error[])
    ec.state.val=outcome.status===:completed && history_failed ? :history_save_failed : outcome.status
    ec.status.val="$(outcome.status): $(outcome.written)/$(outcome.total) written pairs; subprocess exited"*
        (history_failed ? "; history save failed: $history_error" : "")
    outcome.cleanup_confirmed || (ec.status.val*= "; joined processing cleanup was not confirmed")
    lane.outcome=outcome
    for source in (ec.record,ec.last_run,ec.progress,ec.error,ec.state,ec.status,lane.error)
        _saved_notify!(source,source[])
    end
    nothing
end
function finish_saved_replay!(state,outcome)
    lane=state.experiment;ec=lane.controller
    try
        _adopt_saved_replay_outcome!(state,outcome)
    catch error
        lane.outcome=outcome
        original=lane.owner_error===nothing ? error : lane.owner_error
        _saved_notify!(ec.error,original);_saved_notify!(lane.error,sprint(showerror,original))
        _saved_notify!(ec.state,:owner_failed)
        _saved_notify!(ec.status,"subprocess ended ($(outcome.status)); terminal metadata could not be adopted: $(sprint(showerror,error))")
    finally
        # outcome() is available only after OS exit/reap, including fault paths.
        lane.job=nothing;lane.request=nothing;lane.observer=nothing;lane.pending_progress=nothing
        _saved_notify!(ec.running,false)
        try
            refresh_experiment(lane)
        catch error
            ec.error[]===nothing && _saved_notify!(ec.error,error)
            _saved_notify!(lane.error,sprint(showerror,ec.error[]))
            _saved_notify!(ec.status,"subprocess ended; recipe/history display refresh failed: $(sprint(showerror,error))")
        end
    end
    nothing
end
function service_saved_replay!(state)
    lane=state.experiment
    Threads.threadid()==lane.owner_thread || throw(ArgumentError("saved replay updates require the shell owner thread"))
    job=lane.job
    if job===nothing
        if lane.controller.running[] && lane.request!==nothing && ReplayWorkerClient.poll_startup_cleanup!()
            lane.request=nothing
            _saved_notify!(lane.controller.status,"saved replay could not start; owned startup process has exited: $(sprint(showerror,lane.owner_error))")
            _saved_notify!(lane.controller.running,false)
            refresh_experiment(lane)
        end
        return nothing
    end
    # One nonblocking protocol event per service pass. Deferred acknowledgments
    # allow owner-loop tests/actions without a spin inside a progress callback.
    event=ReplayWorkerClient.poll!(job)
    diagnostic=ReplayWorkerClient.ownership_error(job)
    if diagnostic!==nothing
        # This is a live ownership refusal, not a terminal outcome. Keep the
        # captured request/job and running guard until actual cleanup is proved.
        status="Owned worker cleanup remains unverified: $diagnostic"
        details=lane.owner_error===nothing ? diagnostic : sprint(showerror,lane.owner_error)*"\nOwnership: "*diagnostic
        if lane.controller.state[]!==:cleanup_unverified || lane.controller.status[]!=status || lane.error[]!=details
            original=lane.owner_error===nothing ? ErrorException(diagnostic) : lane.owner_error
            lane.controller.error[]===nothing && _saved_notify!(lane.controller.error,original)
            _saved_notify!(lane.error,details)
            _saved_notify!(lane.controller.state,:cleanup_unverified)
            _saved_notify!(lane.controller.status,status)
        end
        # Even a previously deferred event must not run its observer or release
        # an ACK after ownership proof is lost. Cancellation/cleanup polling
        # remain available; a diagnostic never invents a terminal outcome.
        return nothing
    end
    if state.shutdown && lane.pending_progress!==nothing
        ReplayWorkerClient.request_cancel!(job)
        acknowledge_saved_progress!(state;abort=lane.owner_error)
    end
    event===nothing || event.job_id==ReplayWorkerClient.job_id(job) ||
        throw(ArgumentError("stale saved replay event refused"))
    if event!==nothing && event.kind===:progress && lane.pending_progress!==nothing
        previous=lane.pending_progress
        event.job_id==previous.job_id && event.sequence==previous.sequence ||
            throw(ArgumentError("unexpected second progress event before acknowledgment"))
        event=nothing
    end
    if event!==nothing && event.kind===:progress
        lane.pending_progress=event
        try
            lane.controller.progress[]=(event.written,event.total)
            lane.controller.status[]="$(event.written)/$(event.total) pairs written"*(lane.cancel_requested ? "; cancellation requested; waiting for cleanup" : "")
            reply=lane.observer===nothing || state.shutdown ? nothing : lane.observer(event.written,event.total)
            reply===:defer || acknowledge_saved_progress!(state)
        catch error
            lane.owner_error=error
            _saved_notify!(lane.controller.state,:abort_requested)
            _saved_notify!(lane.error,sprint(showerror,error))
            acknowledge_saved_progress!(state;abort=CapturedException(error,catch_backtrace()))
        end
    end
    outcome=ReplayWorkerClient.outcome(job)
    outcome===nothing || outcome.job_id==ReplayWorkerClient.job_id(job) ||
        throw(ArgumentError("stale saved replay outcome refused"))
    outcome===nothing || finish_saved_replay!(state,outcome)
    event
end
function experiment_page(state,delta)
    state.experiment.page[]=max(1,state.experiment.page[]+Int(delta))
    refresh_experiment(state.experiment)
    nothing
end
function experiment_section(state,history)
    state.experiment.section[]=Bool(history) ? :history : :recipe
    state.experiment.page[]=1
    refresh_experiment(state.experiment)
    nothing
end
function inspect_saved_experiment(state)
    experiment_action(state,()->begin
        ec=state.experiment.controller
        candidate=experiment_results(ec)
        supported(current_result(candidate);allow_physical=true)
        display_transaction(state) do
            state.explorer=candidate
            state.dataset[]=:experiment
            run=ec.last_run[]
            state.displayed[]="Displayed completed run: $(run.run_id)\nRecipe: $(run.recipe_id)\nOutput: $(run.output)"
            state.frame[]=candidate.frame[]
            state.count[]=nframes(candidate)
            state.selection[]=describe_selection(candidate)
            state.render_available[]=true
            state.refresh()
        end
    end)
end

# One bounded old/new payload transaction for saved inspection, native open and
# navigation. A failed render must not leave vectors under an old run label.
function display_transaction(action,state)
    previous=state.explorer
    snapshot=(state.dataset[],state.displayed[],state.frame[],state.count[],state.selection[],
        state.render_available[],previous===nothing ? 0 : previous.frame[],
        previous===nothing ? nothing : previous.selection[])
    try
        action()
    catch
        state.explorer=previous
        dataset,displayed,frame,count,selection,available,model_frame,model_selection=snapshot
        state.dataset[]=dataset; state.displayed[]=displayed
        state.frame[]=frame; state.count[]=count; state.selection[]=selection
        state.render_available[]=available
        try
            if previous!==nothing
                previous.frame[]==model_frame || set_frame!(previous,model_frame)
                previous.selection[]=model_selection
            end
            state.refresh()
        catch
            state.render_available[]=false
            try state.invalidate() catch end
            state.displayed[]="Plot unavailable after rendering failure; open/inspect again to recover.\nRetained previous display: $displayed"
        end
        rethrow()
    end
end
busy(state)=state.batch.running[] || state.experiment.controller.running[]
function request_shutdown(state)
    state.shutdown=true
    cancel!(state.batch)
    cancel_saved_experiment(state)
    state.experiment.pending_progress===nothing || acknowledge_saved_progress!(state;abort=state.experiment.owner_error)
    nothing
end
function dispose_state(state)
    busy(state) && throw(ArgumentError("wait for replay/loading/history cleanup before disposing shell"))
    foreach(off,state.subscriptions)
    empty!(state.subscriptions)
    foreach(off,state.experiment.subscriptions)
    empty!(state.experiment.subscriptions)
    state.refresh=()->nothing
    state.invalidate=()->nothing
    state.explorer=nothing
    nothing
end
