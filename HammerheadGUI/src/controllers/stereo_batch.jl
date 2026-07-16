# Stereo-batch controller: two synchronized camera frame lists, a shared
# dewarper pair, and the parameter/scale form state driving
# Hammerhead.run_piv_stereo_sequence — including its native zero-argument
# `cancel` predicate (no exception; the completed prefix is returned and
# stays persisted). Framework-free.

"""
    StereoBatchRunner(; kwargs...)

Controller for the stereo batch runner. Holds the two synchronized camera
frame lists (`files1`/`files2`) and pairing mode, the [`ImageDewarper`](@ref)
pair (`dewarpers`, `nothing` until built or supplied — see
[`set_dewarpers!`](@ref) and [`build_dewarpers`](@ref)), a curated parameter
form (effort preset or manual window schedule), the stereo scale form
(`dt`/`time_unit`/`length_unit`; stereo results are already in world units,
so `pixel_size` is fixed at 1), the output path, and the run state — all as
`Observables`.

# Keyword defaults
- `files1 = Any[]`, `files2 = Any[]`, `pair_mode = :paired`
- `dewarpers = nothing` (a `(dw1, dw2)` tuple sharing one `DewarpGrid`)
- `effort = :custom`, `window_schedule = [64, 32, 32]`,
  `overlap_fraction = 0.5`, `uncertainty = false` (the manual schedule uses
  the accuracy configuration: padded, Gaussian apodization)
- `dt = 1.0`, `time_unit = "frame"`, `length_unit = "world units"` — a
  dt-only [`PhysicalScale`](@ref) is attached when any differs from default
- `output_path = ""` (empty = in memory only)

Drive it like a [`BatchRunner`](@ref): [`add_files!`](@ref) (with a
`camera` keyword), [`set_schedule!`](@ref), [`set_effort!`](@ref),
[`start!`](@ref), [`cancel!`](@ref) (the stereo driver's native
between-acquisition cancellation — the completed prefix is kept in
`results`); watch `progress`, `status`, `running`, `results`, and the live
`completed` accumulator.
"""
struct StereoBatchRunner
    files1::Observable{Vector{Any}}
    files2::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    dewarpers::Observable{Union{Nothing,Tuple{ImageDewarper,ImageDewarper}}}
    output_path::Observable{String}
    effort::Observable{Symbol}
    window_schedule::Observable{Vector{Int}}
    overlap_fraction::Observable{Float64}
    uncertainty::Observable{Bool}
    dt::Observable{Float64}
    length_unit::Observable{String}
    time_unit::Observable{String}
    running::Observable{Bool}
    cancel::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    results::Observable{Union{Nothing,Vector{StereoPIVResult}}}
    completed::Observable{Vector{StereoPIVResult}}
end

function StereoBatchRunner(; files1 = Any[], files2 = Any[],
                           pair_mode::Symbol = :paired,
                           dewarpers = nothing,
                           output_path::AbstractString = "",
                           effort::Symbol = :custom,
                           window_schedule::AbstractVector{<:Integer} = [64, 32, 32],
                           overlap_fraction::Real = 0.5,
                           uncertainty::Bool = false,
                           dt::Real = 1.0,
                           length_unit::AbstractString = "world units",
                           time_unit::AbstractString = "frame")
    sbc = StereoBatchRunner(Observable{Vector{Any}}(collect(Any, files1)),
                            Observable{Vector{Any}}(collect(Any, files2)),
                            Observable(pair_mode),
                            Observable{Union{Nothing,Tuple{ImageDewarper,ImageDewarper}}}(nothing),
                            Observable(String(output_path)),
                            Observable(effort),
                            Observable(collect(Int, window_schedule)),
                            Observable(Float64(overlap_fraction)),
                            Observable(uncertainty),
                            Observable(Float64(dt)),
                            Observable(String(length_unit)), Observable(String(time_unit)),
                            Observable(false), Observable(false),
                            Observable((0, 0)), Observable(""),
                            Observable{Union{Nothing,Vector{StereoPIVResult}}}(nothing),
                            Observable(StereoPIVResult[]))
    dewarpers === nothing || set_dewarpers!(sbc, dewarpers...)
    return sbc
end

function Base.show(io::IO, sbc::StereoBatchRunner)
    print(io, "StereoBatchRunner($(length(sbc.files1[]))+$(length(sbc.files2[])) frames, ",
          sbc.dewarpers[] === nothing ? "no dewarpers" : "dewarpers set",
          sbc.running[] ? ", running)" : ")")
end

"""
    add_files!(sbc::StereoBatchRunner, entries; camera::Integer = 1)

Append frames (file paths and/or matrices) to camera `camera`'s frame list.
"""
function add_files!(sbc::StereoBatchRunner, entries; camera::Integer = 1)
    camera in (1, 2) || throw(ArgumentError("camera must be 1 or 2, got $camera"))
    files = camera == 1 ? sbc.files1 : sbc.files2
    append!(files[], entries)
    notify(files)
    return sbc
end

"""
    clear_files!(sbc::StereoBatchRunner)

Empty both camera frame lists.
"""
function clear_files!(sbc::StereoBatchRunner)
    empty!(sbc.files1[]); notify(sbc.files1)
    empty!(sbc.files2[]); notify(sbc.files2)
    return sbc
end

"""
    set_dewarpers!(sbc::StereoBatchRunner, dw1, dw2)

Attach the two cameras' [`ImageDewarper`](@ref)s (they must share one
`DewarpGrid`) — built in the GUI ([`build_dewarpers`](@ref)) or constructed
at the REPL and passed in.
"""
function set_dewarpers!(sbc::StereoBatchRunner, dw1::ImageDewarper, dw2::ImageDewarper)
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid"))
    sbc.dewarpers[] = (dw1, dw2)
    return sbc
end

"""
    build_dewarpers(cr1::CalibrationReview, cr2::CalibrationReview;
                    z = 0.0, spacing = :auto, coverage = :intersection,
                    margin = 0.0) -> (dw1, dw2)

Build the stereo [`ImageDewarper`](@ref) pair from two fitted calibration
reviews: `Hammerhead.common_dewarp_grid` over both fitted cameras (each
review's plate image size), then one dewarper per camera on the shared grid.
Throws when either review has no fitted camera.
"""
function build_dewarpers(cr1::CalibrationReview, cr2::CalibrationReview;
                         z::Real = 0.0, spacing = :auto,
                         coverage::Symbol = :intersection, margin::Real = 0.0)
    cam1, cam2 = cr1.camera[], cr2.camera[]
    cam1 === nothing && throw(ArgumentError("camera 1 has no fitted calibration: $(cr1.fit_message[])"))
    cam2 === nothing && throw(ArgumentError("camera 2 has no fitted calibration: $(cr2.fit_message[])"))
    size1, size2 = size(cr1.images[1]), size(cr2.images[1])
    grid = common_dewarp_grid([cam1, cam2], [size1, size2], z;
                              spacing, coverage, margin)
    return ImageDewarper(cam1, grid, size1), ImageDewarper(cam2, grid, size2)
end

"""
    set_dewarpers!(sbc::StereoBatchRunner, cr1::CalibrationReview,
                   cr2::CalibrationReview; kwargs...)

Convenience composition of [`build_dewarpers`](@ref) and the direct
dewarper setter.
"""
set_dewarpers!(sbc::StereoBatchRunner, cr1::CalibrationReview,
               cr2::CalibrationReview; kwargs...) =
    set_dewarpers!(sbc, build_dewarpers(cr1, cr2; kwargs...)...)

"""
    stereo_pairs(sbc::StereoBatchRunner) -> (pairs1, pairs2)

The two cameras' synchronized correlation pair lists under the current
pairing mode (`Hammerhead.image_pairs` per camera).
"""
stereo_pairs(sbc::StereoBatchRunner) =
    (image_pairs(sbc.files1[]; mode = sbc.pair_mode[]),
     image_pairs(sbc.files2[]; mode = sbc.pair_mode[]))

"""
    set_schedule!(sbc::StereoBatchRunner, schedule)

Set the manual multi-pass window schedule (vector of sizes or a string, see
[`parse_schedule`](@ref)); used when `effort == :custom`.
"""
set_schedule!(sbc::StereoBatchRunner, s::AbstractString) =
    set_schedule!(sbc, parse_schedule(s))
function set_schedule!(sbc::StereoBatchRunner, s::AbstractVector{<:Integer})
    isempty(s) && throw(ArgumentError("empty window schedule"))
    sbc.window_schedule[] = collect(Int, s)
    return sbc
end

"""
    set_effort!(sbc::StereoBatchRunner, effort::Symbol)

Set the analysis effort (`:custom` manual schedule, or the
`:low`/`:medium`/`:high` presets of [`run_piv_stereo_sequence`](@ref)).
"""
function set_effort!(sbc::StereoBatchRunner, effort::Symbol)
    effort in EFFORT_LEVELS ||
        throw(ArgumentError("effort must be one of $(EFFORT_LEVELS), got :$effort"))
    sbc.effort[] = effort
    return sbc
end

"""
    set_dt!(sbc::StereoBatchRunner, value)

Set the frame interval from a positive number or its string form.
"""
set_dt!(sbc::StereoBatchRunner, v::Real) =
    (v > 0 || throw(ArgumentError("dt must be positive, got $v")); sbc.dt[] = Float64(v); sbc)
set_dt!(sbc::StereoBatchRunner, s::AbstractString) =
    (sbc.dt[] = _parse_positive(s, "dt"); sbc)

build_parameters(sbc::StereoBatchRunner) =
    multipass_parameters(sbc.window_schedule[];
                         overlap_fraction = sbc.overlap_fraction[],
                         uncertainty = sbc.uncertainty[])

"""
    build_scale(sbc::StereoBatchRunner) -> Union{Nothing,PhysicalScale}

The dt-only [`PhysicalScale`](@ref) attached to the stereo outputs
(`pixel_size = 1` — the arrays are already in world units), or `nothing`
when every scale field is at its default.
"""
function build_scale(sbc::StereoBatchRunner)
    (sbc.dt[] == 1.0 && sbc.time_unit[] == "frame" &&
     sbc.length_unit[] == "world units") && return nothing
    return PhysicalScale(1.0, sbc.dt[], sbc.length_unit[], sbc.time_unit[])
end

"""
    validate(sbc::StereoBatchRunner) -> Union{Nothing,String}

`nothing` when the stereo batch can start, otherwise a human-readable
reason.
"""
function validate(sbc::StereoBatchRunner)
    sbc.dewarpers[] === nothing && return "build or set the dewarpers first"
    (isempty(sbc.files1[]) || isempty(sbc.files2[])) && return "add frames for both cameras"
    prs1, prs2 = try
        stereo_pairs(sbc)
    catch err
        return _errmsg(err)
    end
    isempty(prs1) && return "no pairs to process"
    length(prs1) == length(prs2) ||
        return "camera frame lists give $(length(prs1)) vs $(length(prs2)) pairs"
    if sbc.effort[] === :custom
        try
            build_parameters(sbc)
        catch err
            return _errmsg(err)
        end
    end
    return nothing
end

"""
    start!(sbc::StereoBatchRunner; async = true)

Validate and start the stereo batch (same task semantics as the planar
[`start!`](@ref)). Progress lands in `progress`/`status`, finished
acquisitions stream into `completed`, results into `results`, and the
output file is written incrementally.
"""
function start!(sbc::StereoBatchRunner; async::Bool = true)
    sbc.running[] && return sbc
    msg = validate(sbc)
    msg === nothing || (sbc.status[] = msg; return sbc)
    sbc.cancel[] = false
    sbc.running[] = true
    async ? errormonitor(@async _run!(sbc)) : _run!(sbc)
    return sbc
end

"""
    cancel!(sbc::StereoBatchRunner)

Request cancellation: the stereo driver's native predicate stops the run
between acquisitions, and the completed prefix is returned in `results`
(and stays in the incremental output).
"""
cancel!(sbc::StereoBatchRunner) = (sbc.cancel[] = true; sbc)

function _run!(sbc::StereoBatchRunner)
    try
        prs1, prs2 = stereo_pairs(sbc)
        dw1, dw2 = sbc.dewarpers[]
        n = length(prs1)
        sbc.progress[] = (0, n)
        sbc.status[] = "running…"
        sbc.completed[] = StereoPIVResult[]   # fresh accumulator for this run
        on_result = (i, r) -> (push!(sbc.completed[], r); notify(sbc.completed))
        progress = (i, m) -> (sbc.progress[] = (i, m))
        cancel = () -> sbc.cancel[]           # the driver's native predicate
        output = isempty(sbc.output_path[]) ? nothing : sbc.output_path[]
        scale = build_scale(sbc)
        results = if sbc.effort[] === :custom
            run_piv_stereo_sequence(prs1, prs2, dw1, dw2, build_parameters(sbc);
                                    progress, on_result, cancel, output, scale)
        else
            run_piv_stereo_sequence(prs1, prs2, dw1, dw2; effort = sbc.effort[],
                                    progress, on_result, cancel, output, scale)
        end
        sbc.results[] = results
        sbc.status[] = if length(results) < n
            "cancelled after $(length(results)) of $n acquisitions"
        else
            "done: $n acquisitions" * (output === nothing ? "" : " → $(basename(output))")
        end
    catch err
        sbc.status[] = "failed: $(_errmsg(err))"
    finally
        sbc.running[] = false
    end
    return sbc
end
