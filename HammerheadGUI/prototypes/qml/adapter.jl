# Candidate shell adapter. Production controllers remain independent of QML.
module Prototype
using Hammerhead, HammerheadGUI, HammerheadGUI.Controllers, Observables
using Hammerhead.SyntheticData: generate_synthetic_piv_pair, linear_flow
include("worker_client.jl")
include("experiment_adapter.jl")

mutable struct State
    batch::BatchRunner
    explorer::Union{Nothing,ResultExplorer}
    mask::MaskEditor
    image::Matrix{Float64}
    schedule_error::Observable{String}
    open_error::Observable{String}
    status::Observable{String}
    frame::Observable{Int}
    count::Observable{Int}
    selection::Observable{String}
    refresh::Function
    ticks::Int
    visualizations::Int
    shutdown::Bool
    drawing::Bool
    experiment::ExperimentLane
    dataset::Observable{Symbol}
    displayed::Observable{String}
    subscriptions::Vector{Any}
    render_available::Observable{Bool}
    invalidate::Function
end

function dense_result(n = 128)
    xs = collect(range(4.0, 92.0; length = n))
    u = [3sin(y / 15) for y in xs, x in xs]
    v = [3cos(x / 15) for y in xs, x in xs]
    PIVResult(xs, copy(xs), u, v, ones(n, n), ones(n, n),
              fill(NaN, n, n), fill(NaN, n, n), falses(n, n), falses(n, n),
              PIVParameters(window_size = 16, overlap = (8, 8)))
end

function State()
    a, b, _, _ = generate_synthetic_piv_pair(linear_flow(2, 1, 0, 0, 0, 0, 0),
                                              (96, 96), 1.0; z_range = (-1., 1.))
    batch = BatchRunner(files = [a, b, a, b, a, b], window_schedule = [32])
    state = State(batch, ResultExplorer(dense_result()), MaskEditor(a), a,
                  Observable(""), Observable(""), Observable("Ready: dense demo (16,384 vectors)"),
                  Observable(1), Observable(1), Observable("No selection"),
                  () -> nothing, 0, 0, false, false,ExperimentLane(),
                  Observable(:demo),Observable("Displayed: synthetic demo"),Any[],Observable(true),()->nothing)
    push!(state.subscriptions,on(batch.status) do status
        state.status[] = status
    end)
    push!(state.subscriptions,on(batch.completed) do completed
        isempty(completed) && return
        state.dataset[]=:demo
        state.displayed[]="Displayed: synthetic demo (not a saved run)"
        if length(completed) == 1
            state.explorer = ResultExplorer(completed[1])
        else
            push_result!(state.explorer, completed[end])
        end
        state.count[] = nframes(state.explorer)
        navigate(state, nframes(state.explorer))
    end)
    state
end

message(err) = first(split(sprint(showerror, err), '\n'))
function supported(result;allow_physical=false)
    result isa PIVResult || throw(ArgumentError("this prototype viewport supports planar PIV entries only"))
    allow_physical || result.scale === nothing || throw(ArgumentError("this prototype viewport requires unscaled pixel results"))
    (isempty(result.x) || isempty(result.y)) && throw(ArgumentError("empty vector grid"))
    result
end
function schedule(state, text)
    try
        parsed = parse_schedule(String(text))
        all(n -> n >= 8 && n <= minimum(size(state.image)), parsed) ||
            throw(ArgumentError("windows must fit the 96 x 96 demo images (8 to 96)"))
        set_schedule!(state.batch, parsed)
        state.schedule_error[] = ""
        return true
    catch err
        state.schedule_error[] = message(err)
        return false
    end
end

function navigate(state, i)
    try
        display_transaction(state) do
            index = clamp(Int(i), 1, nframes(state.explorer))
            results = state.explorer.results
            if state.dataset[]!==:experiment
                source = results isa HammerheadGUI.Controllers._LazyDisplayResults ? results.source : results
                supported(source[index])
            end
            set_frame!(state.explorer, Int(i))
            state.frame[] = state.explorer.frame[]
            state.count[] = nframes(state.explorer)
            state.selection[] = describe_selection(state.explorer)
            state.render_available[]=true
            state.refresh()
        end
        state.open_error[] = ""
        return true
    catch err
        state.open_error[] = message(err)
        return false
    end
end

function open_results(state, path)
    try
        candidate = ResultExplorer(String(path); lazy = true)
        supported(current_result(candidate))
        display_transaction(state) do
            state.explorer = candidate
            state.dataset[]=:native
            state.displayed[]="Displayed completed native file: $(abspath(String(path))) (no saved-run association)"
            state.frame[]=1; state.count[]=nframes(candidate)
            state.selection[]=describe_selection(candidate)
            state.render_available[]=true
            state.refresh()
        end
        state.open_error[] = ""
        state.status[] = "Indexed $(nframes(candidate)) completed results; one display payload"
        return true
    catch err
        state.open_error[] = message(err)
        return false
    end
end

function pick(state, x, y; drawing = false)
    # Demo-only mask mode cannot swallow ordinary file/experiment inspection.
    state.dataset[]===:demo && state.explorer.path===nothing || (drawing=false;state.drawing=false)
    if drawing
        state.dataset[]===:demo && state.explorer.path === nothing || return nothing
        click!(state.mask, x, y)
        state.selection[] = status_text(state.mask)
    else
        select_nearest!(state.explorer, x, y)
        state.selection[] = describe_selection(state.explorer)
    end
    state.refresh()
    nothing
end
function close_mask(state)
    state.dataset[]===:demo || return nothing
    close_active!(state.mask)
    state.batch.mask[] = polygon_mask(state.mask)
    state.selection[] = status_text(state.mask)
    state.refresh()
    nothing
end
function run_batch(state)
    state.shutdown && return false
    state.experiment.controller.running[] && return false
    isempty(state.schedule_error[]) || return false
    start!(state.batch)
    state.batch.running[]
end
cancel_batch(state) = (cancel!(state.batch); nothing)
function wait_batch(state; timeout = 120.)
    deadline = time() + timeout
    while state.batch.running[]
        time() < deadline || error("batch timeout")
        sleep(0.005)
    end
end
end
