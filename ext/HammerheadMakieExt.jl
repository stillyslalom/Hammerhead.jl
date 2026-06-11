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
    Makie.arrows2d!(ax, points, directions; kwargs...)
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
