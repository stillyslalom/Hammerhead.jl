# Result-explorer controller: all state and logic for browsing a loaded
# result sequence. Framework-free — views render from the Observables and
# mutate them through the API below, so everything here runs without a GL
# context.

# The explorer browses all four persisted result types. Gridded results
# (PIV/stereo) select a window by CartesianIndex; scattered results (PTV
# particles, tracking trajectories) select by linear index.
const GridResult = Union{PIVResult,StereoPIVResult}
const ScatteredResult = Union{PTVResult,TrackingResult}
const AnyResult = Union{GridResult,ScatteredResult}
const Selection = Union{Nothing,CartesianIndex{2},Int}

"""
    ResultExplorer(results; path = nothing)
    ResultExplorer(result)
    ResultExplorer(path::AbstractString)

Controller for the result explorer: a loaded sequence of `PIVResult`,
`StereoPIVResult`, `PTVResult`, or `TrackingResult` entries (mixed files are
allowed) plus the view state as `Observables` — `frame` (1-based index into
the sequence), `field` (the displayed scalar field, see
[`available_fields`](@ref)), `show_vectors` / `highlight_outliers` (overlay
toggles), `selection` (the inspected item: a `CartesianIndex` for a
gridded window, a linear `Int` index for a scattered particle/trajectory, or
`nothing`), and the colorbar-limit state `color_mode` / `color_min` /
`color_max` (see [`color_limits`](@ref) and [`set_color_limits!`](@ref);
manual overrides persist across frame/field switches until cleared).

Each entry is routed through [`physical`](@ref) at construction, so loaded
results carrying a [`PhysicalScale`](@ref) display in physical units with
labelled axes (`physical` is idempotent and identity-safe, so unscaled
results are untouched).

The string form loads the sequence with `Hammerhead.load_results`. Changing
frames resets `field` to the new result's default when the current field is
unavailable, and drops `selection` when it no longer refers to a valid item.
"""
struct ResultExplorer
    results::Vector{AnyResult}
    path::Union{Nothing,String}
    frame::Observable{Int}
    field::Observable{Symbol}
    show_vectors::Observable{Bool}
    highlight_outliers::Observable{Bool}
    selection::Observable{Selection}
    color_mode::Observable{Symbol}
    color_min::Observable{Union{Nothing,Float64}}
    color_max::Observable{Union{Nothing,Float64}}
end

function ResultExplorer(results::AbstractVector; path::Union{Nothing,AbstractString} = nothing)
    isempty(results) && throw(ArgumentError("no results to explore"))
    all(r -> r isa AnyResult, results) ||
        throw(ArgumentError("results must be PIVResult, StereoPIVResult, PTVResult, or TrackingResult entries"))
    conv = AnyResult[physical(r) for r in results]
    ex = ResultExplorer(conv,
                        path === nothing ? nothing : String(path),
                        Observable(1), Observable(first(available_fields(conv[1]))),
                        Observable(true), Observable(true),
                        Observable{Selection}(nothing),
                        Observable(:robust),
                        Observable{Union{Nothing,Float64}}(nothing),
                        Observable{Union{Nothing,Float64}}(nothing))
    on(ex.frame) do _
        r = current_result(ex)
        ex.field[] in available_fields(r) || (ex.field[] = first(available_fields(r)))
        ex.selection[] = _valid_selection(r, ex.selection[])
    end
    return ex
end

ResultExplorer(result::AnyResult; kwargs...) = ResultExplorer([result]; kwargs...)
ResultExplorer(path::AbstractString) = ResultExplorer(load_results(path); path)

function Base.show(io::IO, ex::ResultExplorer)
    print(io, "ResultExplorer($(length(ex.results)) frame",
          length(ex.results) == 1 ? "" : "s",
          ", frame $(ex.frame[]), field :$(ex.field[]))")
end

"""
    nframes(ex::ResultExplorer) -> Int

Number of results in the explored sequence.
"""
nframes(ex::ResultExplorer) = length(ex.results)

"""
    current_result(ex::ResultExplorer)

The result at the current frame (one of `PIVResult`, `StereoPIVResult`,
`PTVResult`, `TrackingResult`).
"""
current_result(ex::ResultExplorer) = ex.results[ex.frame[]]

"""
    set_frame!(ex::ResultExplorer, i::Integer)

Move to frame `i`, clamped to `1:nframes(ex)`.
"""
set_frame!(ex::ResultExplorer, i::Integer) = ex.frame[] = clamp(i, 1, nframes(ex))

# Whether a stored selection still refers to a valid item of the given result
# (a gridded window index for grids, a linear index for scattered results).
function _valid_selection(r, sel)
    sel === nothing && return nothing
    if r isa GridResult
        return (sel isa CartesianIndex{2} && checkbounds(Bool, r.u, sel)) ? sel : nothing
    elseif r isa PTVResult
        return (sel isa Int && 1 <= sel <= length(r.x)) ? sel : nothing
    else # TrackingResult
        return (sel isa Int && 1 <= sel <= length(r.trajectories)) ? sel : nothing
    end
end

"""
    available_fields(result) -> Vector{Symbol}

Scalar fields displayable for a result. Gridded results (`PIVResult` /
`StereoPIVResult`) offer `:magnitude` (in-plane, or 3-component for stereo,
displacement magnitude) plus the per-vector component/diagnostic fields, with
uncertainty fields included only when estimates are present. A `PTVResult`
offers `[:magnitude, :u, :v, :match_residual]`; a `TrackingResult` offers
`[:speed]` (per-trajectory mean speed).
"""
function available_fields(r::PIVResult)
    fields = [:magnitude, :u, :v, :peak_ratio, :correlation_moment]
    any(isfinite, r.uncertainty_u) && push!(fields, :uncertainty_u)
    any(isfinite, r.uncertainty_v) && push!(fields, :uncertainty_v)
    return fields
end

function available_fields(r::StereoPIVResult)
    fields = [:magnitude, :u, :v, :w]
    any(isfinite, r.uncertainty_u) &&
        append!(fields, (:uncertainty_u, :uncertainty_v, :uncertainty_w))
    return fields
end

available_fields(::PTVResult) = [:magnitude, :u, :v, :match_residual]
available_fields(::TrackingResult) = [:speed]

"""
    field_values(result, field::Symbol)

The scalar field to display. For gridded results this is a `Matrix`
(`:magnitude` is `hypot` of the displacement components, everything else the
result's own matrix). For a `PTVResult` it is a per-particle `Vector`
(`:magnitude`, `:u`, `:v`, `:match_residual`). For a `TrackingResult` it is a
per-trajectory `Vector` of mean speeds (`:speed`).
"""
function field_values(r::GridResult, field::Symbol)
    field === :magnitude &&
        return r isa StereoPIVResult ? hypot.(r.u, r.v, r.w) : hypot.(r.u, r.v)
    field in available_fields(r) ||
        throw(ArgumentError("field :$field is not available for this result"))
    return getproperty(r, field)
end

function field_values(r::PTVResult, field::Symbol)
    field === :magnitude && return hypot.(r.u, r.v)
    field in available_fields(r) ||
        throw(ArgumentError("field :$field is not available for this result"))
    return getproperty(r, field)
end

function field_values(r::TrackingResult, field::Symbol)
    field === :speed ||
        throw(ArgumentError("field :$field is not available for this result"))
    return [_mean_speed(t, r.scale) for t in r.trajectories]
end

const FIELD_NAMES = Dict(
    :magnitude => "|displacement|",
    :u => "u", :v => "v", :w => "w",
    :peak_ratio => "peak ratio",
    :correlation_moment => "correlation moment",
    :match_residual => "match residual",
    :speed => "speed",
    :uncertainty_u => "σu", :uncertainty_v => "σv", :uncertainty_w => "σw",
)

"""
    field_name(field::Symbol) -> String

Short display name of a scalar field (menu entries).
"""
field_name(field::Symbol) = FIELD_NAMES[field]

# Fallback units when no PhysicalScale is attached: pixels for planar / PTV /
# tracking, "world units" for stereo (world coordinates are already physical).
_fallback_unit(::StereoPIVResult) = "world units"
_fallback_unit(::AnyResult) = "px"

# Position (length) unit and displacement/velocity unit. After `physical`
# conversion a real attached scale leaves labelled unit strings; `nothing`
# means unscaled, so we use the type's fallback.
_length_unit(r::AnyResult) = r.scale === nothing ? _fallback_unit(r) : r.scale.length_unit
_field_unit(r::AnyResult) = r.scale === nothing ? _fallback_unit(r) :
    string(r.scale.length_unit, "/", r.scale.time_unit)

"""
    field_label(result, field::Symbol) -> String

Display name of a scalar field with the result's units appended (colorbar
label). Displacement/velocity fields and their uncertainties carry the
velocity unit (`length_unit/time_unit`, e.g. `mm/s`, or the `px`/`world units`
fallback when unscaled); a `PTVResult`'s `match_residual` carries the length
unit; dimensionless diagnostics carry no unit.
"""
function field_label(r::AnyResult, field::Symbol)
    field in (:peak_ratio, :correlation_moment) && return field_name(field)
    field === :match_residual && return string(field_name(field), " (", _length_unit(r), ")")
    return string(field_name(field), " (", _field_unit(r), ")")
end

"""
    set_field!(ex::ResultExplorer, field::Symbol)

Display `field` (must be in `available_fields(current_result(ex))`).
"""
function set_field!(ex::ResultExplorer, field::Symbol)
    field in available_fields(current_result(ex)) ||
        throw(ArgumentError("field :$field is not available for the current result"))
    ex.field[] = field
    return ex
end

# Item validity for the colorbar statistics: masked/outlier grid cells and
# flagged PTV particles are excluded from the robust range (they are exactly
# the values that stretch it); tracking speeds carry no flags.
_flagged(r::GridResult, i) = r.mask[i] || r.outliers[i]
_flagged(r::PTVResult, i) = r.outliers[i]
_flagged(::TrackingResult, i) = false

# Nearest-rank percentile band of (unsorted) values; mutates `vals` by sorting.
function _percentile_band(vals::Vector{Float64}, plo::Real, phi::Real)
    sort!(vals)
    n = length(vals)
    lo = vals[clamp(round(Int, plo * (n - 1)) + 1, 1, n)]
    hi = vals[clamp(round(Int, phi * (n - 1)) + 1, 1, n)]
    return (lo, hi)
end

"""
    color_limits(result, field::Symbol, mode::Symbol = :robust) -> (lo, hi)

Automatic colorbar limits for a displayed field. `:full` is the extrema of
all finite values. `:robust` (the default) is the 2–98% percentile band over
finite values at *valid* items (non-masked, non-outlier grid cells;
non-flagged PTV particles) so a few outliers cannot stretch the color range;
when no valid values exist it falls back to all finite values. A degenerate
range is padded by ±0.5.
"""
function color_limits(r::AnyResult, field::Symbol, mode::Symbol = :robust)
    mode in (:robust, :full) ||
        throw(ArgumentError("mode must be :robust or :full, got :$mode"))
    data = field_values(r, field)
    vals = Float64[]
    if mode === :robust
        for i in eachindex(data)
            (isfinite(data[i]) && !_flagged(r, i)) || continue
            push!(vals, Float64(data[i]))
        end
    end
    isempty(vals) && (vals = [Float64(v) for v in data if isfinite(v)])
    isempty(vals) && return (0.0, 1.0)
    lo, hi = mode === :robust ? _percentile_band(vals, 0.02, 0.98) : extrema(vals)
    lo == hi && ((lo, hi) = (lo - 0.5, hi + 0.5))
    return (lo, hi)
end

"""
    set_color_mode!(ex::ResultExplorer, mode::Symbol)

Set the automatic colorbar-limit mode: `:robust` (2–98% percentile band over
valid values, the default) or `:full` (extrema). See [`color_limits`](@ref).
"""
function set_color_mode!(ex::ResultExplorer, mode::Symbol)
    mode in (:robust, :full) ||
        throw(ArgumentError("mode must be :robust or :full, got :$mode"))
    ex.color_mode[] = mode
    return ex
end

# One manual colorbar bound: `nothing`, "", or "auto" clears the override;
# numbers and their string forms set it. Throws ArgumentError on junk.
_parse_limit(::Nothing) = nothing
_parse_limit(v::Real) =
    (isfinite(v) || throw(ArgumentError("color limit must be finite, got $v")); Float64(v))
function _parse_limit(s::AbstractString)
    t = strip(s)
    (isempty(t) || lowercase(t) == "auto") && return nothing
    v = tryparse(Float64, t)
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("color limit must be a number or \"auto\", got \"$s\""))
    return v
end

"""
    set_color_limits!(ex::ResultExplorer; min = missing, max = missing)

Manually override the colorbar limits, bound by bound. Each keyword accepts a
number (or its string form) to pin that bound, and `nothing` / `""` /
`"auto"` to clear it back to the automatic [`color_limits`](@ref); `missing`
leaves it unchanged. Manual bounds persist across frame and field switches
until cleared.
"""
function set_color_limits!(ex::ResultExplorer; min = missing, max = missing)
    min === missing || (ex.color_min[] = _parse_limit(min))
    max === missing || (ex.color_max[] = _parse_limit(max))
    return ex
end

"""
    current_color_limits(ex::ResultExplorer) -> (lo, hi)

The colorbar limits in effect for the current frame and field: the automatic
[`color_limits`](@ref) under `ex.color_mode`, with any manual
[`set_color_limits!`](@ref) overrides applied bound-wise (an inverted or
degenerate manual pair is padded to a valid range).
"""
function current_color_limits(ex::ResultExplorer)
    lo, hi = color_limits(current_result(ex), ex.field[], ex.color_mode[])
    ex.color_min[] === nothing || (lo = ex.color_min[])
    ex.color_max[] === nothing || (hi = ex.color_max[])
    if !(lo < hi)
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5
    end
    return (lo, hi)
end

"""
    select_nearest!(ex::ResultExplorer, x::Real, y::Real)

Select the item nearest to the data-space point `(x, y)` (e.g. a mouse
click) for inspection: the nearest grid node for a gridded result, the
nearest particle for a `PTVResult`, or the nearest trajectory (by vertex) for
a `TrackingResult`.
"""
function select_nearest!(ex::ResultExplorer, x::Real, y::Real)
    ex.selection[] = _nearest(current_result(ex), x, y)
    return ex
end

_nearest(r::GridResult, x, y) =
    CartesianIndex(argmin(i -> abs(r.y[i] - y), eachindex(r.y)),
                   argmin(j -> abs(r.x[j] - x), eachindex(r.x)))

function _nearest(r::PTVResult, x, y)
    isempty(r.x) && return nothing
    return argmin(k -> abs2(r.x[k] - x) + abs2(r.y[k] - y), eachindex(r.x))
end

function _nearest(r::TrackingResult, x, y)
    isempty(r.trajectories) && return nothing
    best, bestd = 1, Inf
    for (k, t) in pairs(r.trajectories), p in eachindex(t.x)
        d = abs2(t.x[p] - x) + abs2(t.y[p] - y)
        d < bestd && ((best, bestd) = (k, d))
    end
    return best
end

"""
    clear_selection!(ex::ResultExplorer)

Drop the current selection.
"""
clear_selection!(ex::ResultExplorer) = (ex.selection[] = nothing; ex)

# Data-space point (x, y) marking the current selection, or `nothing` when the
# selection is empty or stale — the view draws a marker there.
function selection_point(r, sel)
    sel = _valid_selection(r, sel)
    sel === nothing && return nothing
    if r isa GridResult
        return (r.x[sel[2]], r.y[sel[1]])
    elseif r isa PTVResult
        return (r.x[sel], r.y[sel])
    else # TrackingResult: mark the trajectory's first point
        t = r.trajectories[sel]
        return (t.x[1], t.y[1])
    end
end

_fmt(v::Real) = isfinite(v) ? @sprintf("%.4g", v) : "—"

_status(r::GridResult, idx) = r.mask[idx]     ? "masked (no measurement)" :
                              r.outliers[idx] ? "outlier (replaced)" : "valid"

"""
    describe_selection(ex::ResultExplorer) -> String

Multi-line summary of the selected item (empty string when nothing is
selected): position, displacement/velocity, diagnostics/uncertainty, and
validation status, all with units.
"""
function describe_selection(ex::ResultExplorer)
    r = current_result(ex)
    sel = _valid_selection(r, ex.selection[])
    sel === nothing && return ""
    return vector_summary(r, sel)
end

function vector_summary(r::PIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["window ($i, $j)",
             "x = $(_fmt(r.x[j])) $lu, y = $(_fmt(r.y[i])) $lu",
             "u = $(_fmt(r.u[idx])) $vu",
             "v = $(_fmt(r.v[idx])) $vu",
             "peak ratio = $(_fmt(r.peak_ratio[idx]))",
             "corr. moment = $(_fmt(r.correlation_moment[idx]))"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_v[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])) $vu, σv = $(_fmt(r.uncertainty_v[idx])) $vu")
    end
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

function vector_summary(r::StereoPIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["node ($i, $j)",
             "x = $(_fmt(r.x[j])), y = $(_fmt(r.y[i])), z = $(_fmt(r.z)) ($lu)",
             "u = $(_fmt(r.u[idx])) $vu",
             "v = $(_fmt(r.v[idx])) $vu",
             "w = $(_fmt(r.w[idx])) $vu"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_w[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])), σv = $(_fmt(r.uncertainty_v[idx])), σw = $(_fmt(r.uncertainty_w[idx])) ($vu)")
    end
    push!(lines, "cam peak ratios: $(_fmt(r.cam1.peak_ratio[idx])) / $(_fmt(r.cam2.peak_ratio[idx]))")
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

function vector_summary(r::PTVResult, k::Int)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["particle $k",
             "x = $(_fmt(r.x[k])) $lu, y = $(_fmt(r.y[k])) $lu",
             "u = $(_fmt(r.u[k])) $vu",
             "v = $(_fmt(r.v[k])) $vu",
             "match residual = $(_fmt(r.match_residual[k])) $lu",
             "status: $(r.outliers[k] ? "flagged (scattered UOD)" : "valid")"]
    return join(lines, "\n")
end

function vector_summary(r::TrackingResult, k::Int)
    t = r.trajectories[k]
    vu = _field_unit(r)
    n = length(t)
    spd = n >= 2 ? "$(_fmt(_mean_speed(t, r.scale))) $vu" : "—"
    lines = ["trajectory $k",
             "start frame $(t.start_frame)",
             "$n point" * (n == 1 ? "" : "s") * ", frames $(first(t.frames))–$(last(t.frames))",
             "gaps: $(trajectory_gap_count(t))",
             "mean speed = $spd"]
    return join(lines, "\n")
end

# Mean speed along a trajectory (physical when `scale` is attached). Zero for
# a single-point track (trajectory_velocities needs ≥ 2 points).
function _mean_speed(t, scale)
    length(t) < 2 && return 0.0
    u, v = trajectory_velocities(t, scale)
    s = 0.0
    for k in eachindex(u)
        s += hypot(u[k], v[k])
    end
    return s / length(u)
end

"""
    trajectory_gap_count(t) -> Int

Number of bridged frame gaps in a trajectory: steps where `t.frames`
increases by more than one (nonconsecutive frames linked across a gap).
"""
trajectory_gap_count(t) = count(>(1), diff(t.frames))

"""
    trajectory_points(t) -> (xs, ys)

Flat polyline coordinates for a trajectory with `NaN` breaks inserted at
frame gaps, so a single `lines!` call renders one polyline per continuous run
(the standard Makie NaN-separation approach).
"""
function trajectory_points(t)
    xs = Float64[]; ys = Float64[]
    for p in eachindex(t.x)
        if p > 1 && t.frames[p] - t.frames[p - 1] > 1
            push!(xs, NaN); push!(ys, NaN)
        end
        push!(xs, t.x[p]); push!(ys, t.y[p])
    end
    return (xs, ys)
end

"""
    vector_data(result) -> NamedTuple

Displayable vectors of the result as flat, NaN-free arrays
`(; x, y, u, v, outlier)` — the arrow-overlay input. Gridded results skip
masked and out-of-view nodes; a `PTVResult` returns one entry per matched
particle.
"""
function vector_data(r::GridResult)
    x = Float64[]; y = Float64[]; u = Float64[]; v = Float64[]; outlier = Bool[]
    for j in eachindex(r.x), i in eachindex(r.y)
        (isnan(r.u[i, j]) || isnan(r.v[i, j])) && continue
        push!(x, r.x[j]); push!(y, r.y[i])
        push!(u, r.u[i, j]); push!(v, r.v[i, j])
        push!(outlier, r.outliers[i, j])
    end
    return (; x, y, u, v, outlier)
end

function vector_data(r::PTVResult)
    keep = findall(k -> isfinite(r.u[k]) && isfinite(r.v[k]), eachindex(r.u))
    return (; x = Float64.(r.x[keep]), y = Float64.(r.y[keep]),
            u = Float64.(r.u[keep]), v = Float64.(r.v[keep]),
            outlier = collect(Bool, r.outliers[keep]))
end

"""
    auto_lengthscale(result, data = vector_data(result)) -> Float64

Arrow length scale that keeps the longest displayed vector at ~85% of the
vector spacing. Gridded results use the grid step; scattered (`PTVResult`)
data uses a robust `sqrt(area / n)` spacing proxy.
"""
function auto_lengthscale(r::GridResult, data = vector_data(r))
    dmax = 0.0
    for k in eachindex(data.u)
        dmax = max(dmax, hypot(data.u[k], data.v[k]))
    end
    spacing = min(minimum(abs.(diff(r.x)); init = Inf),
                  minimum(abs.(diff(r.y)); init = Inf))
    (isfinite(spacing) && dmax > 0) || return 1.0
    return 0.85 * spacing / dmax
end

function auto_lengthscale(r::PTVResult, data = vector_data(r))
    n = length(data.x)
    n == 0 && return 1.0
    dmax = 0.0
    for k in eachindex(data.u)
        dmax = max(dmax, hypot(data.u[k], data.v[k]))
    end
    dmax > 0 || return 1.0
    xspan = maximum(data.x) - minimum(data.x)
    yspan = maximum(data.y) - minimum(data.y)
    spacing = sqrt(max(xspan * yspan, 1.0) / n)
    return 0.85 * spacing / dmax
end
