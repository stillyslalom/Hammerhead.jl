module HammerheadMakieExt

using Hammerhead
using Makie

# Resolve the arrow-length multiplier: a `Real` is used verbatim, `:auto`
# scales the 0.99-quantile of the selected vectors to one plotted grid cell.
function _resolve_lengthscale(lengthscale, u, v, valid, stride::Int, x, y)
    lengthscale isa Real && return Float64(lengthscale)
    lengthscale === :auto ||
        throw(ArgumentError("lengthscale must be :auto or a real number, got $lengthscale"))
    target = stride * min(Hammerhead.grid_axis_step(x), Hammerhead.grid_axis_step(y))
    ls = Hammerhead.arrow_lengthscale(u, v, valid, target)
    return ls === nothing ? 1.0 : ls
end

function Hammerhead.plot_vector_field!(ax::Makie.Axis,
                                       x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                       u::AbstractMatrix{<:Real}, v::AbstractMatrix{<:Real};
                                       stride::Int = 1, lengthscale = :auto, kwargs...)
    size(u) == size(v) == (length(y), length(x)) ||
        throw(ArgumentError("u and v must have dimensions (length(y), length(x)) = $((length(y), length(x))), got $(size(u)) and $(size(v))"))
    stride >= 1 || throw(ArgumentError("stride must be at least 1, got $stride"))
    ny, nx = length(y), length(x)
    # Directions are pre-scaled here (backend-independent) rather than via a
    # Makie length attribute.
    scale = _resolve_lengthscale(lengthscale, u, v, nothing, stride, x, y)
    points = Makie.Point2f[]
    directions = Makie.Vec2f[]
    keepidx = Int[]                              # column-major linear index into the full grid
    for j in 1:stride:nx, i in 1:stride:ny
        (isnan(u[i, j]) || isnan(v[i, j])) && continue
        push!(points, Makie.Point2f(x[j], y[i]))
        push!(directions, Makie.Vec2f(scale * u[i, j], scale * v[i, j]))
        push!(keepidx, (j - 1) * ny + i)
    end
    # Subsample any full-grid per-vector attribute vectors (color, etc.) to the
    # strided, NaN-filtered subset.
    kw = Dict{Symbol,Any}(kwargs)
    for (k, val) in kw
        val isa AbstractVector && length(val) == ny * nx && (kw[k] = val[keepidx])
    end
    isempty(points) && return ax
    Makie.arrows2d!(ax, points, directions; kw...)
    return ax
end

function Hammerhead.plot_vector_field!(ax::Makie.Axis, result::PIVResult;
                                       stride::Int = 1, lengthscale = :auto,
                                       show_replaced::Bool = true,
                                       replaced_color = :orangered,
                                       color = :black, kwargs...)
    stride >= 1 || throw(ArgumentError("stride must be at least 1, got $stride"))
    result = physical(result)   # plot in physical units when a scale is attached
    u, v = result.u, result.v
    # Valid vectors (not masked, not flagged) set the auto length scale.
    valid = .!(result.mask .| result.outliers)
    scale = _resolve_lengthscale(lengthscale, u, v, valid, stride, result.x, result.y)
    flagged = result.outliers .& .!result.mask
    if any(flagged)
        u2, v2 = copy(u), copy(v)
        if !show_replaced
            u2[flagged] .= convert(eltype(u2), NaN)
            v2[flagged] .= convert(eltype(v2), NaN)
        end
        colors = [flagged[i] ? replaced_color : color for i in eachindex(u)]
        return Hammerhead.plot_vector_field!(ax, result.x, result.y, u2, v2;
                                             stride, lengthscale = scale,
                                             color = colors, kwargs...)
    end
    return Hammerhead.plot_vector_field!(ax, result.x, result.y, u, v;
                                         stride, lengthscale = scale, color, kwargs...)
end

function Hammerhead.plot_vector_field(result::PIVResult;
                                      figure = (;), axis = (;), kwargs...)
    result = physical(result)
    xlabel, ylabel = Hammerhead.plot_axis_labels(result.scale)
    fig, ax = _vector_field_figure(; figure, axis, xlabel, ylabel)
    Hammerhead.plot_vector_field!(ax, result; kwargs...)
    return fig
end

# Scattered PTV vectors: arrows at each frame-A match position (not a grid).
function Hammerhead.plot_vector_field!(ax::Makie.Axis, result::PTVResult;
                                       highlight_outliers::Bool = true, kwargs...)
    result = physical(result)   # plot in physical units when a scale is attached
    points = [Makie.Point2f(result.x[i], result.y[i]) for i in eachindex(result.x)]
    directions = [Makie.Vec2f(result.u[i], result.v[i]) for i in eachindex(result.u)]
    if highlight_outliers && any(result.outliers)
        color = [o ? :red : :black for o in result.outliers]
        Makie.arrows2d!(ax, points, directions; color, kwargs...)
    else
        Makie.arrows2d!(ax, points, directions; kwargs...)
    end
    return ax
end

function Hammerhead.plot_vector_field(result::PTVResult;
                                      figure = (;), axis = (;), kwargs...)
    result = physical(result)
    xlabel, ylabel = Hammerhead.plot_axis_labels(result.scale)
    fig, ax = _vector_field_figure(; figure, axis, xlabel, ylabel)
    Hammerhead.plot_vector_field!(ax, result; kwargs...)
    return fig
end

function Hammerhead.plot_vector_field(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                      u::AbstractMatrix{<:Real}, v::AbstractMatrix{<:Real};
                                      figure = (;), axis = (;), kwargs...)
    fig, ax = _vector_field_figure(; figure, axis)
    Hammerhead.plot_vector_field!(ax, x, y, u, v; kwargs...)
    return fig
end

function _vector_field_figure(; figure, axis, xlabel = "x (px)", ylabel = "y (px)")
    fig = Makie.Figure(; figure...)
    # Image (row-down) coordinates: y axis reversed. `axis...` splats last, so
    # user-supplied labels still win.
    ax = Makie.Axis(fig[1, 1];
                    xlabel, ylabel,
                    yreversed = true, aspect = Makie.DataAspect(), axis...)
    return fig, ax
end

end # module HammerheadMakieExt
