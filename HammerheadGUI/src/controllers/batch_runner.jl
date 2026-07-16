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
- `effort = :custom` — `:custom` runs the manual window schedule below;
  `:low` / `:medium` / `:high` use [`run_piv_sequence`](@ref)'s effort presets
  and ignore the schedule/option widgets.
- `window_schedule = [64, 32, 32]`, `overlap_fraction = 0.5`
- `correlation_method = :cross`, `padding = true`, `apodization = :gauss`
  (the accuracy configuration), `subpixel_method = :gauss3`,
  `uncertainty = false`
- `pixel_size = 1.0`, `dt = 1.0`, `length_unit = "px"`, `time_unit = "frame"` —
  a [`PhysicalScale`](@ref) is attached to the outputs only when any of these
  differs from its default.
- `output_path = ""` (empty = keep results in memory only), `mask = nothing`

Drive it with [`add_files!`](@ref), [`set_schedule!`](@ref),
[`set_effort!`](@ref), [`start!`](@ref) and [`cancel!`](@ref); watch
`progress` (`(done, total)`), `status`, `running`, `results` (a
`Vector{PIVResult}` after a completed run, `nothing` before), and
`completed` (the finished pairs' results so far, appended and notified
live while the batch runs — feed them to a [`ResultExplorer`](@ref) via
[`push_result!`](@ref) to browse a batch in progress).
"""
struct BatchRunner
    files::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    output_path::Observable{String}
    mask::Observable{Union{Nothing,BitMatrix}}
    effort::Observable{Symbol}
    window_schedule::Observable{Vector{Int}}
    overlap_fraction::Observable{Float64}
    correlation_method::Observable{Symbol}
    padding::Observable{Bool}
    apodization::Observable{Symbol}
    subpixel_method::Observable{Symbol}
    uncertainty::Observable{Bool}
    pixel_size::Observable{Float64}
    dt::Observable{Float64}
    length_unit::Observable{String}
    time_unit::Observable{String}
    running::Observable{Bool}
    cancel::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    results::Observable{Union{Nothing,Vector{PIVResult}}}
    completed::Observable{Vector{PIVResult}}
end

function BatchRunner(; files = Any[], pair_mode::Symbol = :paired,
                     output_path::AbstractString = "", mask = nothing,
                     effort::Symbol = :custom,
                     window_schedule::AbstractVector{<:Integer} = [64, 32, 32],
                     overlap_fraction::Real = 0.5,
                     correlation_method::Symbol = :cross,
                     padding::Bool = true, apodization::Symbol = :gauss,
                     subpixel_method::Symbol = :gauss3,
                     uncertainty::Bool = false,
                     pixel_size::Real = 1.0, dt::Real = 1.0,
                     length_unit::AbstractString = "px",
                     time_unit::AbstractString = "frame")
    return BatchRunner(Observable{Vector{Any}}(collect(Any, files)),
                       Observable(pair_mode), Observable(String(output_path)),
                       Observable{Union{Nothing,BitMatrix}}(mask === nothing ? nothing : BitMatrix(mask)),
                       Observable(effort),
                       Observable(collect(Int, window_schedule)),
                       Observable(Float64(overlap_fraction)),
                       Observable(correlation_method), Observable(padding),
                       Observable(apodization), Observable(subpixel_method),
                       Observable(uncertainty),
                       Observable(Float64(pixel_size)), Observable(Float64(dt)),
                       Observable(String(length_unit)), Observable(String(time_unit)),
                       Observable(false), Observable(false),
                       Observable((0, 0)), Observable(""),
                       Observable{Union{Nothing,Vector{PIVResult}}}(nothing),
                       Observable(PIVResult[]))
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

const EFFORT_LEVELS = (:custom, :low, :medium, :high)

"""
    set_effort!(bc::BatchRunner, effort::Symbol)

Set the analysis effort. `:custom` uses the manual window schedule and
option widgets; `:low` / `:medium` / `:high` use [`run_piv_sequence`](@ref)'s
effort presets and ignore the manual schedule.
"""
function set_effort!(bc::BatchRunner, effort::Symbol)
    effort in EFFORT_LEVELS ||
        throw(ArgumentError("effort must be one of $(EFFORT_LEVELS), got :$effort"))
    bc.effort[] = effort
    return bc
end

# Parse a positive-number textbox entry (pixel size / dt).
function _parse_positive(str::AbstractString, what::AbstractString)
    v = tryparse(Float64, strip(str))
    (v === nothing || !(isfinite(v) && v > 0)) &&
        throw(ArgumentError("$what must be a positive number, got \"$str\""))
    return v
end

"""
    set_pixel_size!(bc::BatchRunner, value)
    set_dt!(bc::BatchRunner, value)

Set the physical `pixel_size` / frame interval `dt` from a positive number or
its string form (see [`set_scale!`](@ref)).
"""
set_pixel_size!(bc::BatchRunner, v::Real) =
    (v > 0 || throw(ArgumentError("pixel_size must be positive")); bc.pixel_size[] = Float64(v); bc)
set_pixel_size!(bc::BatchRunner, s::AbstractString) =
    (bc.pixel_size[] = _parse_positive(s, "pixel size"); bc)
set_dt!(bc::BatchRunner, v::Real) =
    (v > 0 || throw(ArgumentError("dt must be positive")); bc.dt[] = Float64(v); bc)
set_dt!(bc::BatchRunner, s::AbstractString) =
    (bc.dt[] = _parse_positive(s, "dt"); bc)

"""
    set_scale!(bc::BatchRunner; pixel_size, dt, length_unit, time_unit)

Set any of the physical-scale form fields at once; omitted fields are left
unchanged. Positive numbers are required for `pixel_size`/`dt`.
"""
function set_scale!(bc::BatchRunner; pixel_size = nothing, dt = nothing,
                    length_unit = nothing, time_unit = nothing)
    pixel_size === nothing || set_pixel_size!(bc, pixel_size)
    dt === nothing || set_dt!(bc, dt)
    length_unit === nothing || (bc.length_unit[] = String(length_unit))
    time_unit === nothing || (bc.time_unit[] = String(time_unit))
    return bc
end

"""
    build_scale(bc::BatchRunner) -> Union{Nothing,PhysicalScale}

The [`PhysicalScale`](@ref) attached to the batch outputs, or `nothing` when
every scale field is at its default (`pixel_size = dt = 1`, units `px`/`frame`)
so the results stay in pixel/frame units.
"""
function build_scale(bc::BatchRunner)
    (bc.pixel_size[] == 1.0 && bc.dt[] == 1.0 &&
     bc.length_unit[] == "px" && bc.time_unit[] == "frame") && return nothing
    return PhysicalScale(bc.pixel_size[], bc.dt[], bc.length_unit[], bc.time_unit[])
end

"""
    build_parameters(bc::BatchRunner) -> Vector{PIVParameters}

The multi-pass schedule for the current form state
(`Hammerhead.multipass_parameters`; propagates `PIVParameters` validation
errors). Only relevant when `effort == :custom`.
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

`nothing` when the batch can start, otherwise a human-readable reason. With a
non-`:custom` effort the manual schedule is not consulted.
"""
function validate(bc::BatchRunner)
    isempty(bc.files[]) && return "add frames first"
    prs = try
        frame_pairs(bc)
    catch err
        return _errmsg(err)
    end
    isempty(prs) && return "no pairs to process"
    if bc.effort[] === :custom
        try
            build_parameters(bc)
        catch err
            return _errmsg(err)
        end
    elseif !(bc.effort[] in EFFORT_LEVELS)
        return "unknown effort :$(bc.effort[])"
    end
    try
        build_scale(bc)
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
        bc.progress[] = (0, length(prs))
        bc.status[] = "running…"
        bc.completed[] = PIVResult[]   # fresh accumulator for this run
        callback = (i, n) -> begin
            bc.progress[] = (i, n)
            bc.cancel[] && throw(BatchCancelled())
        end
        # Live accumulator: run_piv_sequence stores results on this (serial)
        # task, so the push happens on the task driving the batch and open
        # views can follow along.
        on_result = (i, r) -> (push!(bc.completed[], r); notify(bc.completed))
        output = isempty(bc.output_path[]) ? nothing : bc.output_path[]
        scale = build_scale(bc)
        maskkw = bc.mask[] === nothing ? (;) : (; mask = bc.mask[])
        results = if bc.effort[] === :custom
            run_piv_sequence(prs, build_parameters(bc);
                             progress = callback, on_result, output, scale, maskkw...)
        else
            run_piv_sequence(prs; effort = bc.effort[],
                             progress = callback, on_result, output, scale, maskkw...)
        end
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
