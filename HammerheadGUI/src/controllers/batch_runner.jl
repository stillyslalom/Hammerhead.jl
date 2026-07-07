# Batch-runner controller: PIVParameters form state plus sequence-execution
# state (file list, output path, progress, cancellation). Framework-free —
# the run itself goes through Hammerhead.run_piv_sequence with a progress
# callback, launched as a cooperative task so a GL render loop stays live.

"""
    BatchCancelled()

Thrown from the progress callback to abort a running batch; pairs finished
before the abort stay in the incremental output file.
"""
struct BatchCancelled <: Exception end

"""
    BatchRunner(; kwargs...)

Controller for the parameter form + batch runner. Holds the frame list and
pairing mode, a curated `PIVParameters` form (multi-pass window schedule +
the accuracy-relevant options), the output path and analysis mask, and the
run state — all as `Observables`.

# Keyword defaults
- `files = Any[]` (frame paths and/or in-memory matrices), `pair_mode = :paired`
- `window_schedule = [64, 32, 32]`, `overlap_fraction = 0.5`
- `correlation_method = :cross`, `padding = true`, `apodization = :gauss`
  (the accuracy configuration), `subpixel_method = :gauss3`,
  `uncertainty = false`
- `output_path = ""` (empty = keep results in memory only), `mask = nothing`

Drive it with [`add_files!`](@ref), [`set_schedule!`](@ref),
[`start!`](@ref) and [`cancel!`](@ref); watch `progress` (`(done, total)`),
`status`, `running`, and `results` (a `Vector{PIVResult}` after a completed
run, `nothing` before).
"""
struct BatchRunner
    files::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    output_path::Observable{String}
    mask::Observable{Union{Nothing,BitMatrix}}
    window_schedule::Observable{Vector{Int}}
    overlap_fraction::Observable{Float64}
    correlation_method::Observable{Symbol}
    padding::Observable{Bool}
    apodization::Observable{Symbol}
    subpixel_method::Observable{Symbol}
    uncertainty::Observable{Bool}
    running::Observable{Bool}
    cancel::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    results::Observable{Union{Nothing,Vector{PIVResult}}}
end

function BatchRunner(; files = Any[], pair_mode::Symbol = :paired,
                     output_path::AbstractString = "", mask = nothing,
                     window_schedule::AbstractVector{<:Integer} = [64, 32, 32],
                     overlap_fraction::Real = 0.5,
                     correlation_method::Symbol = :cross,
                     padding::Bool = true, apodization::Symbol = :gauss,
                     subpixel_method::Symbol = :gauss3,
                     uncertainty::Bool = false)
    return BatchRunner(Observable{Vector{Any}}(collect(Any, files)),
                       Observable(pair_mode), Observable(String(output_path)),
                       Observable{Union{Nothing,BitMatrix}}(mask === nothing ? nothing : BitMatrix(mask)),
                       Observable(collect(Int, window_schedule)),
                       Observable(Float64(overlap_fraction)),
                       Observable(correlation_method), Observable(padding),
                       Observable(apodization), Observable(subpixel_method),
                       Observable(uncertainty),
                       Observable(false), Observable(false),
                       Observable((0, 0)), Observable(""),
                       Observable{Union{Nothing,Vector{PIVResult}}}(nothing))
end

function Base.show(io::IO, bc::BatchRunner)
    print(io, "BatchRunner($(length(bc.files[])) frames, schedule ",
          bc.window_schedule[], bc.running[] ? ", running)" : ")")
end

"""
    add_files!(bc::BatchRunner, entries)

Append frames (file paths and/or matrices) to the frame list.
"""
function add_files!(bc::BatchRunner, entries)
    append!(bc.files[], entries)
    notify(bc.files)
    return bc
end

"""
    clear_files!(bc::BatchRunner)

Empty the frame list.
"""
clear_files!(bc::BatchRunner) = (empty!(bc.files[]); notify(bc.files); bc)

"""
    frame_pairs(bc::BatchRunner)

The correlation pairs of the current frame list and pairing mode
(`Hammerhead.image_pairs`; throws on an odd `:paired` frame count).
"""
frame_pairs(bc::BatchRunner) = image_pairs(bc.files[]; mode = bc.pair_mode[])

"""
    parse_schedule(str) -> Vector{Int}

Parse a window-schedule entry: positive integers separated by commas and/or
spaces, e.g. `"64, 32, 32"`. Throws `ArgumentError` on anything else.
"""
function parse_schedule(str::AbstractString)
    tokens = split(str, r"[,\s]+"; keepempty = false)
    isempty(tokens) && throw(ArgumentError("empty window schedule"))
    sizes = Int[]
    for t in tokens
        n = tryparse(Int, t)
        (n === nothing || n <= 0) &&
            throw(ArgumentError("window sizes must be positive integers, got \"$t\""))
        push!(sizes, n)
    end
    return sizes
end

"""
    set_schedule!(bc::BatchRunner, schedule)

Set the multi-pass window schedule from a vector of sizes or a string
(see [`parse_schedule`](@ref)).
"""
set_schedule!(bc::BatchRunner, s::AbstractString) = set_schedule!(bc, parse_schedule(s))
function set_schedule!(bc::BatchRunner, s::AbstractVector{<:Integer})
    isempty(s) && throw(ArgumentError("empty window schedule"))
    bc.window_schedule[] = collect(Int, s)
    return bc
end

"""
    build_parameters(bc::BatchRunner) -> Vector{PIVParameters}

The multi-pass schedule for the current form state
(`Hammerhead.multipass_parameters`; propagates `PIVParameters` validation
errors).
"""
build_parameters(bc::BatchRunner) =
    multipass_parameters(bc.window_schedule[];
                         overlap_fraction = bc.overlap_fraction[],
                         correlation_method = bc.correlation_method[],
                         padding = bc.padding[],
                         apodization = bc.apodization[],
                         subpixel_method = bc.subpixel_method[],
                         uncertainty = bc.uncertainty[])

"""
    validate(bc::BatchRunner) -> Union{Nothing,String}

`nothing` when the batch can start, otherwise a human-readable reason.
"""
function validate(bc::BatchRunner)
    isempty(bc.files[]) && return "add frames first"
    prs = try
        frame_pairs(bc)
    catch err
        return _errmsg(err)
    end
    isempty(prs) && return "no pairs to process"
    try
        build_parameters(bc)
    catch err
        return _errmsg(err)
    end
    return nothing
end

_errmsg(err) = first(split(sprint(showerror, err), '\n'))

"""
    start!(bc::BatchRunner; async = true)

Validate and start the batch. With `async = true` the run executes in a
cooperative task (`@async`) so a GLMakie render loop keeps updating —
`run_piv`'s internal thread spawns provide the yield points; observables
are still written from the primary thread. Progress lands in
`bc.progress`/`bc.status`, results in `bc.results`, and the output file
(when `output_path` is set) is written incrementally.
"""
function start!(bc::BatchRunner; async::Bool = true)
    bc.running[] && return bc
    msg = validate(bc)
    msg === nothing || (bc.status[] = msg; return bc)
    bc.cancel[] = false
    bc.running[] = true
    async ? errormonitor(@async _run!(bc)) : _run!(bc)
    return bc
end

"""
    cancel!(bc::BatchRunner)

Request cancellation of the running batch; it stops after the pair in
flight, keeping finished pairs in the incremental output.
"""
cancel!(bc::BatchRunner) = (bc.cancel[] = true; bc)

function _run!(bc::BatchRunner)
    try
        prs = frame_pairs(bc)
        params = build_parameters(bc)
        bc.progress[] = (0, length(prs))
        bc.status[] = "running…"
        callback = (i, n) -> begin
            bc.progress[] = (i, n)
            bc.cancel[] && throw(BatchCancelled())
        end
        output = isempty(bc.output_path[]) ? nothing : bc.output_path[]
        kwargs = bc.mask[] === nothing ? (;) : (; mask = bc.mask[])
        results = run_piv_sequence(prs, params; progress = callback, output, kwargs...)
        bc.results[] = results
        bc.status[] = "done: $(length(results)) pairs" *
                      (output === nothing ? "" : " → $(basename(output))")
    catch err
        if err isa BatchCancelled
            done, total = bc.progress[]
            bc.status[] = "cancelled after $done of $total pairs"
        else
            bc.status[] = "failed: $(_errmsg(err))"
        end
    finally
        bc.running[] = false
    end
    return bc
end
