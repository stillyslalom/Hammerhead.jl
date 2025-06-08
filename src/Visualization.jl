module Visualization

using GLMakie

export plot_vector_field

"""
    plot_vector_field(x_coords, y_coords, u_vectors, v_vectors;
                      figure_kwargs=Dict(), arrow_kwargs=Dict(), axis_kwargs=Dict())

Plots a 2D vector field using GLMakie.

- `x_coords`, `y_coords`:
    - If 2D AbstractArrays: Assumed to be meshgrid-like matrices of point locations.
    - If 1D AbstractVectors: Assumed to be the unique x and y coordinates; they will be
      expanded into a 2D grid. `u_vectors` and `v_vectors` must match this grid shape.
- `u_vectors`, `v_vectors`: 2D AbstractArrays of U and V vector components.
- `figure_kwargs`: Dictionary of keyword arguments passed to `GLMakie.Figure()`.
- `arrow_kwargs`: Dictionary of keyword arguments passed to `GLMakie.arrows!()`.
- `axis_kwargs`: Dictionary of keyword arguments passed to `GLMakie.Axis()`.

Returns the `GLMakie.Figure` object. The plot is displayed automatically.
"""
function plot_vector_field(
    x_coords::AbstractArray{T},
    y_coords::AbstractArray{T},
    u_vectors::AbstractMatrix{S},
    v_vectors::AbstractMatrix{S};
    figure_kwargs::Dict = Dict(),
    arrow_kwargs::Dict = Dict(),
    axis_kwargs::Dict = Dict()
) where {T<:Real, S<:Real}

    local fig_x_coords, fig_y_coords

    if x_coords isa AbstractVector && y_coords isa AbstractVector
        # Convert 1D vectors to 2D grids for arrows!
        if size(u_vectors) != (length(y_coords), length(x_coords)) || size(v_vectors) != (length(y_coords), length(x_coords))
             error("For 1D x_coords and y_coords, u_vectors and v_vectors must have dimensions (length(y_coords), length(x_coords)).")
        end
        fig_x_coords = [x for _ in y_coords, x in x_coords]
        fig_y_coords = [y for y in y_coords, _ in x_coords]
    elseif x_coords isa AbstractMatrix && y_coords isa AbstractMatrix
        if size(x_coords) != size(u_vectors) || size(y_coords) != size(v_vectors) || size(x_coords) != size(y_coords)
            error("For 2D x_coords and y_coords, all input arrays (coords and vectors) must have the same dimensions.")
        end
        fig_x_coords = x_coords
        fig_y_coords = y_coords
    else
        error("x_coords and y_coords must be both AbstractVectors or both AbstractMatrices.")
    end

    # Default figure and arrow arguments
    default_fig_kwargs = Dict(:resolution => (800, 600))
    final_fig_kwargs = merge(default_fig_kwargs, figure_kwargs)

    default_axis_kwargs = Dict(:title => "Vector Field", :xlabel => "X", :ylabel => "Y", :aspect => DataAspect())
    final_axis_kwargs = merge(default_axis_kwargs, axis_kwargs)

    default_arrow_kwargs = Dict(:arrowsize => 10, :lengthscale => 0.1)
    final_arrow_kwargs = merge(default_arrow_kwargs, arrow_kwargs)

    fig = Figure(; final_fig_kwargs...)
    ax = Axis(fig[1,1]; final_axis_kwargs...)

    arrows!(ax, fig_x_coords, fig_y_coords, u_vectors, v_vectors; final_arrow_kwargs...)

    display(fig) # Show the plot
    return fig
end

end # module Visualization
