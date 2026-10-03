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
        lane.identity[]=record===nothing ? "No saved experiment open" :
            "Active recipe: $(recipe_identity(record.recipe))\nInput: $(record.input_id)"
        lane.recipe_text=experiment_summary(ec)
        lane.history_text=experiment_run_history(ec)
    end
    lane.written[]="Written pairs: $(ec.progress[][1]) / $(ec.progress[][2])"
    details=lane.section[]===:history ? lane.history_text : lane.recipe_text
    full="State: $(ec.state[])\nStatus: $(ec.status[])\nResult output: $(ec.output_path[])\nRun record: $(ec.run_record_path[])\n"*
        "Replay starts at pair 1; cancel waits for writes/cleanup.\n\n"*details
    chunks=wrapped_pages(full)
    lane.page[]=clamp(lane.page[],1,length(chunks))
    lane.text[]=chunks[lane.page[]]
    lane.pages[]="Page $(lane.page[]) / $(length(chunks))"
    nothing
end
function ExperimentLane()
    ec=ExperimentController()
    lane=ExperimentLane(ec,Observable(""),Observable(""),Observable(""),
        Observable(:recipe),Observable(1),Observable(""),Observable(""),Any[],"","")
    push!(lane.subscriptions,on(_->refresh_experiment(lane;record_changed=true),ec.record))
    for source in (ec.status,ec.state,ec.progress,ec.output_path,ec.run_record_path,lane.section)
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
function run_saved_experiment(state;progress=nothing)
    experiment_action(state,()->begin
        ec=state.experiment.controller
        record=ec.record[]
        record===nothing && throw(ArgumentError("open a saved experiment first"))
        record.recipe.external_preprocess===nothing ||
            throw(ArgumentError("referenced scripts can be inspected; this shell does not load or execute them"))
        start!(ec;progress)
    end)
end
cancel_saved_experiment(state)=(cancel!(state.experiment.controller);nothing)
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
