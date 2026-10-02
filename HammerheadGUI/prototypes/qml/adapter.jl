# Candidate shell adapter. Production controllers remain independent of QML.
module Prototype
using Hammerhead, HammerheadGUI, HammerheadGUI.Controllers, Observables
using Hammerhead.SyntheticData: generate_synthetic_piv_pair, linear_flow

mutable struct State
    batch::BatchRunner
    explorer::ResultExplorer
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
                  () -> nothing, 0, 0, false, false)
    on(batch.status) do status
        state.status[] = status
    end
    on(batch.completed) do completed
        isempty(completed) && return
        if length(completed) == 1
            state.explorer = ResultExplorer(completed[1])
        else
            push_result!(state.explorer, completed[end])
        end
        state.count[] = nframes(state.explorer)
        navigate(state, nframes(state.explorer))
    end
    state
end

message(err) = first(split(sprint(showerror, err), '\n'))
function supported(result)
    result isa PIVResult || throw(ArgumentError("this prototype viewport supports planar PIV entries only"))
    result.scale === nothing || throw(ArgumentError("this prototype viewport requires unscaled pixel results"))
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
        index = clamp(Int(i), 1, nframes(state.explorer))
        results = state.explorer.results
        # Reject unsupported entries before replacing the display cache/state.
        source = results isa HammerheadGUI.Controllers._LazyDisplayResults ? results.source : results
        supported(source[index])
        set_frame!(state.explorer, Int(i))
        state.frame[] = state.explorer.frame[]
        state.count[] = nframes(state.explorer)
        state.selection[] = describe_selection(state.explorer)
        state.refresh()
        state.open_error[] = ""
        return true
    catch err
        state.open_error[] = message(err)
        return false
    end
end

function open_results(state, path)
    previous = (state.explorer, state.frame[], state.count[], state.selection[], state.status[])
    try
        candidate = ResultExplorer(String(path); lazy = true)
        supported(current_result(candidate))
        state.explorer = candidate
        state.open_error[] = ""
        state.status[] = "Indexed $(nframes(candidate)) completed results; one display payload"
        navigate(state, 1) || error(state.open_error[])
    catch err
        state.explorer, frame, count, selection, status = previous
        state.frame[] = frame
        state.count[] = count
        state.selection[] = selection
        state.status[] = status
        state.open_error[] = message(err)
        return false
    end
end

function pick(state, x, y; drawing = false)
    if drawing
        state.explorer.path === nothing || return nothing # mask belongs to demo frames
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
    close_active!(state.mask)
    state.batch.mask[] = polygon_mask(state.mask)
    state.selection[] = status_text(state.mask)
    state.refresh()
    nothing
end
function run_batch(state)
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
