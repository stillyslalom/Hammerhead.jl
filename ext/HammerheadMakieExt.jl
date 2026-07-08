module HammerheadMakieExt

using Hammerhead
using Makie

function Hammerhead.plot_vector_field!(ax::Makie.Axis,
                                       x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                       u::AbstractMatrix{<:Real}, v::AbstractMatrix{<:Real};
                                       kwargs...)
    size(u) == size(v) == (length(y), length(x)) ||
        throw(ArgumentError("u and v must have dimensions (length(y), length(x)) = $((length(y), length(x))), got $(size(u)) and $(size(v))"))
    points = vec([Makie.Point2f(x[j], y[i]) for i in eachindex(y), j in eachindex(x)])
    directions = vec([Makie.Vec2f(u[i, j], v[i, j]) for i in eachindex(y), j in eachindex(x)])
    # NaN vectors (e.g. masked interrogation windows) are skipped, along with
    # any matching per-vector attribute vectors (color, etc.).
    keep = map(d -> !(isnan(d[1]) || isnan(d[2])), directions)
    kw = Dict{Symbol,Any}(kwargs)
    if !all(keep)
        points = points[keep]
        directions = directions[keep]
        for (k, val) in kw
            val isa AbstractVector && length(val) == length(keep) && (kw[k] = val[keep])
        end
    end
    Makie.arrows2d!(ax, points, directions; kw...)
    return ax
end

function Hammerhead.plot_vector_field!(ax::Makie.Axis, result::PIVResult;
                                       highlight_outliers::Bool = true, kwargs...)
    if highlight_outliers && any(result.outliers)
        color = [o ? :red : :black for o in vec(result.outliers)]
        return Hammerhead.plot_vector_field!(ax, result.x, result.y, result.u, result.v;
                                             color, kwargs...)
    end
    return Hammerhead.plot_vector_field!(ax, result.x, result.y, result.u, result.v; kwargs...)
end

function Hammerhead.plot_vector_field(result::PIVResult;
                                      figure = (;), axis = (;), kwargs...)
    fig, ax = _vector_field_figure(; figure, axis)
    Hammerhead.plot_vector_field!(ax, result; kwargs...)
    return fig
end

# Scattered PTV vectors: arrows at each frame-A match position (not a grid).
function Hammerhead.plot_vector_field!(ax::Makie.Axis, result::PTVResult;
                                       highlight_outliers::Bool = true, kwargs...)
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
    fig, ax = _vector_field_figure(; figure, axis)
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

function _vector_field_figure(; figure, axis)
    fig = Makie.Figure(; figure...)
    # Image (row-down) coordinates: y axis reversed.
    ax = Makie.Axis(fig[1, 1];
                    xlabel = "x (px)", ylabel = "y (px)",
                    yreversed = true, aspect = Makie.DataAspect(), axis...)
    return fig, ax
end

end # module HammerheadMakieExt
