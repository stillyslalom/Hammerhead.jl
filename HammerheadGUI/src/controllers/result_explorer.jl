# Result-explorer controller: all state and logic for browsing a loaded
# result sequence. Framework-free — views render from the Observables and
# mutate them through the API below, so everything here runs without a GL
# context.

const AnyResult = Union{PIVResult,StereoPIVResult}

"""
    ResultExplorer(results; path = nothing)
    ResultExplorer(result)
    ResultExplorer(path::AbstractString)

Controller for the result explorer: a loaded sequence of `PIVResult` /
`StereoPIVResult` entries (mixed files are allowed) plus the view state as
`Observables` — `frame` (1-based index into the sequence), `field` (the
displayed scalar field, see [`available_fields`](@ref)), `show_vectors` /
`highlight_outliers` (overlay toggles), and `selection` (the inspected
vector's `CartesianIndex`, or `nothing`).

The string form loads the sequence with `Hammerhead.load_results`. Changing
frames resets `field` to `:magnitude` when the new result lacks the current
field, and drops `selection` when it falls outside the new grid.
"""
struct ResultExplorer
    results::Vector{AnyResult}
    path::Union{Nothing,String}
    frame::Observable{Int}
    field::Observable{Symbol}
    show_vectors::Observable{Bool}
    highlight_outliers::Observable{Bool}
    selection::Observable{Union{Nothing,CartesianIndex{2}}}
end

function ResultExplorer(results::AbstractVector; path::Union{Nothing,AbstractString} = nothing)
    isempty(results) && throw(ArgumentError("no results to explore"))
    all(r -> r isa AnyResult, results) ||
        throw(ArgumentError("results must be PIVResult or StereoPIVResult entries"))
    ex = ResultExplorer(collect(AnyResult, results),
                        path === nothing ? nothing : String(path),
                        Observable(1), Observable(:magnitude),
                        Observable(true), Observable(true),
                        Observable{Union{Nothing,CartesianIndex{2}}}(nothing))
    on(ex.frame) do _
        r = current_result(ex)
        ex.field[] in available_fields(r) || (ex.field[] = :magnitude)
        sel = ex.selection[]
        sel === nothing || checkbounds(Bool, r.u, sel) || (ex.selection[] = nothing)
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
    current_result(ex::ResultExplorer) -> Union{PIVResult,StereoPIVResult}

The result at the current frame.
"""
current_result(ex::ResultExplorer) = ex.results[ex.frame[]]

"""
    set_frame!(ex::ResultExplorer, i::Integer)

Move to frame `i`, clamped to `1:nframes(ex)`.
"""
set_frame!(ex::ResultExplorer, i::Integer) = ex.frame[] = clamp(i, 1, nframes(ex))

"""
    available_fields(result) -> Vector{Symbol}

Scalar fields displayable for a `PIVResult` or `StereoPIVResult`:
`:magnitude` (in-plane, or 3-component for stereo, displacement magnitude)
and the per-vector component/diagnostic fields. Uncertainty fields are
included only when the result carries estimates (any finite value).
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

"""
    field_values(result, field::Symbol) -> Matrix

The scalar field to display: `:magnitude` is computed (`hypot` of the
displacement components), everything else is the result's own matrix.
"""
function field_values(r::AnyResult, field::Symbol)
    field === :magnitude &&
        return r isa StereoPIVResult ? hypot.(r.u, r.v, r.w) : hypot.(r.u, r.v)
    field in available_fields(r) ||
        throw(ArgumentError("field :$field is not available for this result"))
    return getproperty(r, field)
end

const FIELD_NAMES = Dict(
    :magnitude => "|displacement|",
    :u => "u", :v => "v", :w => "w",
    :peak_ratio => "peak ratio",
    :correlation_moment => "correlation moment",
    :uncertainty_u => "σu", :uncertainty_v => "σv", :uncertainty_w => "σw",
)

"""
    field_name(field::Symbol) -> String

Short display name of a scalar field (menu entries).
"""
field_name(field::Symbol) = FIELD_NAMES[field]

units(::PIVResult) = "px"
units(::StereoPIVResult) = "world units"

"""
    field_label(result, field::Symbol) -> String

Display name of a scalar field with the result's units appended
(colorbar label). Dimensionless diagnostics carry no unit.
"""
function field_label(r::AnyResult, field::Symbol)
    field in (:peak_ratio, :correlation_moment) && return field_name(field)
    return string(field_name(field), " (", units(r), ")")
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

"""
    select_nearest!(ex::ResultExplorer, x::Real, y::Real)

Select the vector whose grid node is nearest to the data-space point
`(x, y)` (e.g. a mouse click) for inspection.
"""
function select_nearest!(ex::ResultExplorer, x::Real, y::Real)
    r = current_result(ex)
    j = argmin(j -> abs(r.x[j] - x), eachindex(r.x))
    i = argmin(i -> abs(r.y[i] - y), eachindex(r.y))
    ex.selection[] = CartesianIndex(i, j)
    return ex
end

"""
    clear_selection!(ex::ResultExplorer)

Drop the vector selection.
"""
clear_selection!(ex::ResultExplorer) = (ex.selection[] = nothing; ex)

_fmt(v::Real) = isfinite(v) ? @sprintf("%.4g", v) : "—"

_status(r::AnyResult, idx) = r.mask[idx]     ? "masked (no measurement)" :
                             r.outliers[idx] ? "outlier (replaced)" : "valid"

"""
    describe_selection(ex::ResultExplorer) -> String

Multi-line summary of the selected vector (empty string when nothing is
selected): grid position, displacement components, diagnostics, uncertainty
(when finite), and validation status.
"""
function describe_selection(ex::ResultExplorer)
    idx = ex.selection[]
    idx === nothing && return ""
    return vector_summary(current_result(ex), idx)
end

function vector_summary(r::PIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    lines = ["window ($i, $j)",
             "x = $(_fmt(r.x[j])) px, y = $(_fmt(r.y[i])) px",
             "u = $(_fmt(r.u[idx])) px",
             "v = $(_fmt(r.v[idx])) px",
             "peak ratio = $(_fmt(r.peak_ratio[idx]))",
             "corr. moment = $(_fmt(r.correlation_moment[idx]))"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_v[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])) px, σv = $(_fmt(r.uncertainty_v[idx])) px")
    end
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

function vector_summary(r::StereoPIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    un = units(r)
    lines = ["node ($i, $j)",
             "x = $(_fmt(r.x[j])), y = $(_fmt(r.y[i])), z = $(_fmt(r.z)) ($un)",
             "u = $(_fmt(r.u[idx])) $un",
             "v = $(_fmt(r.v[idx])) $un",
             "w = $(_fmt(r.w[idx])) $un"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_w[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])), σv = $(_fmt(r.uncertainty_v[idx])), σw = $(_fmt(r.uncertainty_w[idx])) ($un)")
    end
    push!(lines, "cam peak ratios: $(_fmt(r.cam1.peak_ratio[idx])) / $(_fmt(r.cam2.peak_ratio[idx]))")
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

"""
    vector_data(result) -> NamedTuple

Displayable vectors of the result as flat, NaN-free arrays
`(; x, y, u, v, outlier)` — the arrow-overlay input (masked and
out-of-view nodes are skipped).
"""
function vector_data(r::AnyResult)
    x = Float64[]; y = Float64[]; u = Float64[]; v = Float64[]; outlier = Bool[]
    for j in eachindex(r.x), i in eachindex(r.y)
        (isnan(r.u[i, j]) || isnan(r.v[i, j])) && continue
        push!(x, r.x[j]); push!(y, r.y[i])
        push!(u, r.u[i, j]); push!(v, r.v[i, j])
        push!(outlier, r.outliers[i, j])
    end
    return (; x, y, u, v, outlier)
end

"""
    auto_lengthscale(result, data = vector_data(result)) -> Float64

Arrow length scale that keeps the longest displayed vector at ~85% of the
grid spacing.
"""
function auto_lengthscale(r::AnyResult, data = vector_data(r))
    dmax = 0.0
    for k in eachindex(data.u)
        dmax = max(dmax, hypot(data.u[k], data.v[k]))
    end
    spacing = min(minimum(abs.(diff(r.x)); init = Inf),
                  minimum(abs.(diff(r.y)); init = Inf))
    (isfinite(spacing) && dmax > 0) || return 1.0
    return 0.85 * spacing / dmax
end
