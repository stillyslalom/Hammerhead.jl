# Results step canvas: a scalar field with vectors and the selected node of a
# ResultExplorer. Same rule as the image canvas: plots are created once and
# only their inputs change. Frame, field and colour choices are made with the
# window's controls; clicking the canvas selects the nearest vector.

"""
    ResultsCanvas

Canvas for the Results step. Attach an explorer with `set_explorer!`.
"""
struct ResultsCanvas
    fig::Figure
    ax::Axis
    dirty::Base.RefValue{Bool}
    field::Any
    colorbar::Colorbar
    shafts::Any
    heads::Any
    selection::Any
    explorer::Base.RefValue{Union{Nothing,ResultExplorer}}
    listeners::Vector{Any}
    grid_size::Base.RefValue{Union{Nothing,Dims{2}}}
end

function results_canvas()
    fig = Figure(size = (900, 800), figure_padding = 6)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
              xlabel = "x (px)", ylabel = "y (px)")
    field = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = :viridis, colorrange = (0, 1),
                     nan_color = :transparent)
    cb = Colorbar(fig[1, 2], field; label = "")
    shafts = linesegments!(ax, [Point2f(NaN, NaN), Point2f(NaN, NaN)];
                           color = [:black, :black], linewidth = 1.4)
    heads = scatter!(ax, _NOPOINT; marker = :utriangle, rotation = [0.0f0], markersize = 8,
                     color = [RGBf(0, 0, 0)])
    sel = scatter!(ax, _NOPOINT; color = :transparent, strokecolor = :cyan,
                   strokewidth = 2.5, markersize = 16)
    translate!(field, 0, 0, -1)
    rc = ResultsCanvas(fig, ax, Ref(true), field, cb, shafts, heads, sel,
                       Ref{Union{Nothing,ResultExplorer}}(nothing), Any[],
                       Ref{Union{Nothing,Dims{2}}}(nothing))
    register_interaction!(ax, :select_vector) do event::MouseEvent, _
        event.type === MouseEventTypes.leftclick || return Consume(false)
        ex = rc.explorer[]
        ex === nothing && return Consume(false)
        select_nearest!(ex, event.data[1], event.data[2])
        return Consume(true)
    end
    return rc
end

"""
    set_explorer!(rc::ResultsCanvas, ex)

Show `ex` (a `ResultExplorer`, or `nothing`) on the canvas.
"""
function set_explorer!(rc::ResultsCanvas, ex::Union{Nothing,ResultExplorer})
    rc.explorer[] === ex && return rc
    foreach(off, rc.listeners)
    empty!(rc.listeners)
    rc.explorer[] = ex
    rc.grid_size[] = nothing
    if ex !== nothing
        for obs in (ex.frame, ex.field, ex.color_mode, ex.color_min, ex.color_max,
                    ex.show_vectors, ex.highlight_outliers)
            push!(rc.listeners, on(_ -> _draw_results!(rc), obs))
        end
        push!(rc.listeners, on(_ -> _draw_selection!(rc), ex.selection))
    end
    _draw_results!(rc)
    _draw_selection!(rc)
    return rc
end

function _draw_results!(rc::ResultsCanvas)
    ex = rc.explorer[]
    r = ex === nothing ? nothing : current_result(ex)
    if r isa Controllers.GridResult
        data = Float32.(current_field_values(ex))
        lo, hi = current_color_limits(ex)
        _update!(rc.field, collect(Float64, r.x), collect(Float64, r.y), permutedims(data);
                      colorrange = (lo, hi))
        rc.colorbar.label = field_label(r, ex.field[])
        unit = r.scale === nothing ? "px" : r.scale.length_unit
        rc.ax.xlabel = "x ($unit)"
        rc.ax.ylabel = "y ($unit)"
        _update_arrows!(rc.shafts, rc.heads, ex.show_vectors[] ? r : nothing;
                        valid_color = RGBf(0, 0, 0),
                        flagged_color = ex.highlight_outliers[] ? FLAGGED_COLOR : RGBf(0, 0, 0))
        if size(data) != rc.grid_size[]
            rc.grid_size[] = size(data)
            reset_limits!(rc.ax)
        end
    else
        _update!(rc.field, 1:2, 1:2, _EMPTY_IMAGE)
        rc.colorbar.label = ""
        _update_arrows!(rc.shafts, rc.heads, nothing)
    end
    rc.dirty[] = true
    return rc
end

function _draw_selection!(rc::ResultsCanvas)
    ex = rc.explorer[]
    p = ex === nothing ? nothing : selection_point(current_result(ex), ex.selection[])
    _update!(rc.selection, p === nothing ? _NOPOINT : [Point2f(p...)])
    rc.dirty[] = true
    return rc
end
