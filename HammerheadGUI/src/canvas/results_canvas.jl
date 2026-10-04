# Results step canvas: a scalar field with vectors (gridded results), particles
# colored by a field with their displacements (PTV), or trajectories colored
# by mean speed (tracking); the selected node, and the
# analysis tools of a ResultExplorer. Same rule as the image canvas: plots are
# created once and only their inputs change. Frame, field, color and tool
# choices are made with the window's controls; a click on the canvas goes to
# the explorer's tool (inspect: select the nearest vector; profile: line
# endpoints; circulation: contour vertices, right-click closes), Escape
# clears the tool's path.
#
# The profile panel is an Axis in a second layout row, created with the
# figure with its legend. Outside the profile tool the row collapses to zero
# height and the Axis's and legend's scenes are hidden (`visible = false`,
# which GLMakie skips when rendering), so showing or hiding the panel only
# changes layout and visibility, never the plots.

const TOOL_PATH_COLOR = RGBf(0.0, 0.95, 1.0)
const PROFILE_COLORS = (RGBf(0.27, 0.51, 0.71), RGBf(1.0, 0.55, 0.0), RGBf(0, 0, 0))  # u, v, |V|
const PROFILE_HEIGHT = 170

"""
    ResultsCanvas

Canvas for the Results step. Attach an explorer with `set_explorer!`.
"""
struct ResultsCanvas
    fig::Figure
    ax::Axis
    dirty::Base.RefValue{Bool}
    field::Any
    points::Any               # PTV particles, colored by the field
    tracks::Any               # trajectories, colored by mean speed
    colorbar::Colorbar
    shafts::Any
    heads::Any
    selection::Any
    tool_line::Any            # profile line / circulation contour
    tool_points::Any          # its vertices
    profile_box::GridLayout   # second layout row holding the profile Axis
    profile_ax::Axis
    profile_lines::Vector{Any}  # u, v, |V|
    profile_legend::Legend
    profile_shown::Base.RefValue{Bool}
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
    # scattered results color with the heatmap's colormap and range (the
    # heatmap then holds no data), so the colorbar serves every result type
    points = scatter!(ax, _NOPOINT; color = [0.0f0], colormap = :viridis, colorrange = (0, 1),
                      markersize = 8)
    tracks = lines!(ax, _NOPOINT; color = [0.0f0], colormap = :viridis, colorrange = (0, 1),
                    linewidth = 1.5)
    cb = Colorbar(fig[1, 2], field; label = "")
    shafts = linesegments!(ax, [Point2f(NaN, NaN), Point2f(NaN, NaN)];
                           color = [:black, :black], linewidth = 1.4)
    heads = scatter!(ax, _NOPOINT; marker = :utriangle, rotation = [0.0f0], markersize = 8,
                     color = [RGBf(0, 0, 0)])
    sel = scatter!(ax, _NOPOINT; color = :transparent, strokecolor = :cyan,
                   strokewidth = 2.5, markersize = 16)
    tool_line = lines!(ax, _NOPOINT; color = TOOL_PATH_COLOR, linewidth = 2.5)
    tool_points = scatter!(ax, _NOPOINT; color = TOOL_PATH_COLOR, markersize = 9,
                           strokecolor = :black, strokewidth = 1)
    translate!(field, 0, 0, -1)
    translate!(tool_line, 0, 0, 2)
    translate!(tool_points, 0, 0, 3)
    # `Outside` alignment keeps the profile Axis's tick labels inside its
    # own box, so the collapsed row takes no space from the field axis.
    box = GridLayout(fig[2, 1:2]; alignmode = Outside())
    pax = Axis(box[1, 1]; height = PROFILE_HEIGHT, xlabel = "distance along the line",
               ylabel = "")
    plines = Any[lines!(pax, [NaN, NaN], [NaN, NaN]; color = c, linewidth = 2,
                        label = l) for (c, l) in zip(PROFILE_COLORS, ("u", "v", "|V|"))]
    legend = Legend(box[1, 2], pax; framevisible = false, padding = (4, 4, 4, 4))
    rc = ResultsCanvas(fig, ax, Ref(true), field, points, tracks, cb, shafts, heads, sel,
                       tool_line, tool_points,
                       box, pax, plines, legend, Ref(true),
                       Ref{Union{Nothing,ResultExplorer}}(nothing), Any[],
                       Ref{Union{Nothing,Dims{2}}}(nothing))
    _show_profile!(rc, false)
    _register_gestures!(rc)
    return rc
end

# Left click → the explorer's tool; right click closes a circulation contour.
# Consumed only when an explorer uses it, so the Axis keeps its own clicks.
function _register_gestures!(rc::ResultsCanvas)
    register_interaction!(rc.ax, :results_gesture) do event::MouseEvent, _
        ex = rc.explorer[]
        (ex === nothing || _modifier_held(rc.fig)) && return Consume(false)
        t = event.type
        if t === MouseEventTypes.leftclick || t === MouseEventTypes.leftdoubleclick
            return Consume(_results_gesture(() -> (click!(ex, event.data[1], event.data[2]); true), rc))
        elseif t === MouseEventTypes.rightclick || t === MouseEventTypes.rightdoubleclick
            ex.tool[] === :circulation || return Consume(false)
            return Consume(_results_gesture(() -> (alt_click!(ex); true), rc))
        end
        return Consume(false)
    end
    on(events(rc.fig).keyboardbutton) do ev
        (ev.action === Keyboard.press && ev.key === Keyboard.escape) || return Consume(false)
        ex = rc.explorer[]
        ex === nothing && return Consume(false)
        return Consume(_results_gesture(() -> canvas_key!(ex, :escape), rc))
    end
    return rc
end

# A failing gesture reports in the explorer's status line instead of breaking
# Makie's event handling.
function _results_gesture(f, rc::ResultsCanvas)
    try
        return f()::Bool
    catch err
        ex = rc.explorer[]
        ex === nothing || (ex.status[] = "error: " * Controllers._errmsg(err))
        return true
    end
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
        for obs in (ex.tool, ex.tool_points, ex.profile_data, ex.circulation_result)
            push!(rc.listeners, on(_ -> _draw_tool!(rc), obs))
        end
    end
    _draw_results!(rc)
    _draw_selection!(rc)
    _draw_tool!(rc)
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
        stereo = r isa StereoPIVResult
        unit = r.scale !== nothing ? r.scale.length_unit : stereo ? "world units" : "px"
        rc.ax.xlabel = "x ($unit)"
        rc.ax.ylabel = "y ($unit)"
        # image results keep rows growing downward; stereo results are on world
        # axes (+Y up, as the dewarped images are displayed)
        flip = rc.ax.yreversed[] == stereo
        flip && (rc.ax.yreversed = !stereo)
        _update_arrows!(rc.shafts, rc.heads, ex.show_vectors[] ? r : nothing;
                        valid_color = RGBf(0, 0, 0), yreversed = !stereo,
                        flagged_color = ex.highlight_outliers[] ? FLAGGED_COLOR : RGBf(0, 0, 0))
        if size(data) != rc.grid_size[] || flip
            rc.grid_size[] = size(data)
            reset_limits!(rc.ax)
        end
        _update!(rc.points, _NOPOINT; color = [0.0f0])
        _update!(rc.tracks, _NOPOINT; color = [0.0f0])
    elseif r isa Union{PTVResult,TrackingResult}
        lo, hi = current_color_limits(ex)
        hi > lo || (hi = lo + 1)
        _update!(rc.field, 1:2, 1:2, _EMPTY_IMAGE; colorrange = (lo, hi))
        rc.colorbar.label = field_label(r, ex.field[])
        rc.ax.xlabel, rc.ax.ylabel = Hammerhead.plot_axis_labels(r.scale)
        rc.ax.yreversed[] || (rc.ax.yreversed = true)
        if r isa PTVResult
            vals = Float32.(current_field_values(ex))
            isempty(r.x) ? _update!(rc.points, _NOPOINT; color = [0.0f0]) :
                _update!(rc.points, Point2f.(r.x, r.y); color = vals, colorrange = (lo, hi))
            _update!(rc.tracks, _NOPOINT; color = [0.0f0])
            _update_arrows!(rc.shafts, rc.heads, ex.show_vectors[] ? r : nothing;
                            valid_color = RGBf(0, 0, 0),
                            flagged_color = ex.highlight_outliers[] ? FLAGGED_COLOR : RGBf(0, 0, 0))
        else
            speeds = current_field_values(ex)
            pts = Point2f[]; cols = Float32[]
            for (k, t) in pairs(r.trajectories)
                xs, ys = trajectory_points(t)
                length(xs) >= 2 || continue
                c = isfinite(speeds[k]) ? Float32(speeds[k]) : Float32(lo)
                append!(pts, Point2f.(xs, ys)); push!(pts, Point2f(NaN, NaN))
                append!(cols, fill(c, length(xs) + 1))
            end
            isempty(pts) ? _update!(rc.tracks, _NOPOINT; color = [0.0f0]) :
                _update!(rc.tracks, pts; color = cols, colorrange = (lo, hi))
            _update!(rc.points, _NOPOINT; color = [0.0f0])
            _update_arrows!(rc.shafts, rc.heads, nothing)
        end
        rc.grid_size[] === nothing || (rc.grid_size[] = nothing; reset_limits!(rc.ax))
    else
        _update!(rc.field, 1:2, 1:2, _EMPTY_IMAGE)
        _update!(rc.points, _NOPOINT; color = [0.0f0])
        _update!(rc.tracks, _NOPOINT; color = [0.0f0])
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

# The tool path (a closed circulation contour is drawn closed) and the
# profile panel.
function _draw_tool!(rc::ResultsCanvas)
    ex = rc.explorer[]
    pts = ex === nothing ? Point2f[] : [Point2f(p...) for p in ex.tool_points[]]
    closed = ex !== nothing && ex.tool[] === :circulation &&
             ex.circulation_result[] !== nothing && length(pts) >= 3
    path = closed ? push!(copy(pts), pts[1]) : pts
    _update!(rc.tool_line, length(path) >= 2 ? path : _NOPOINT)
    _update!(rc.tool_points, isempty(pts) ? _NOPOINT : pts)

    prof = ex === nothing ? nothing : profile_series(ex)
    if prof === nothing
        for l in rc.profile_lines
            _update!(l, [NaN, NaN], [NaN, NaN])
        end
    else
        for (l, y) in zip(rc.profile_lines, (prof.u, prof.v, prof.speed))
            _update!(l, prof.s, y)
        end
        rc.profile_ax.xlabel = prof.xlabel
        rc.profile_ax.ylabel = prof.ylabel
        _profile_limits!(rc.profile_ax, prof)
    end
    _show_profile!(rc, prof !== nothing)
    rc.dirty[] = true
    return rc
end

function _profile_limits!(pax::Axis, prof)
    s = filter(isfinite, prof.s)
    vals = filter(isfinite, vcat(prof.u, prof.v, prof.speed))
    (isempty(s) || isempty(vals)) && return
    s0, s1 = extrema(s)
    lo, hi = extrema(vals)
    s1 > s0 || (s1 = s0 + 1)
    pad = hi > lo ? 0.05 * (hi - lo) : max(abs(hi), 1.0) * 0.05
    xlims!(pax, s0, s1)
    ylims!(pax, lo - pad, hi + pad)
    return
end

function _set_visible!(scene::Scene, v::Bool)
    scene.visible[] == v || (scene.visible[] = v)
    foreach(c -> _set_visible!(c, v), scene.children)
    return
end

# Expand or collapse the profile row (layout and visibility only).
function _show_profile!(rc::ResultsCanvas, show::Bool)
    rc.profile_shown[] == show && return rc
    rc.profile_shown[] = show
    _set_visible!(rc.profile_ax.blockscene, show)
    _set_visible!(rc.profile_legend.blockscene, show)
    rc.profile_box.tellheight[] = show
    rowsize!(rc.fig.layout, 2, show ? Auto() : Fixed(0))
    rowgap!(rc.fig.layout, 1, Fixed(show ? 10 : 0))
    return rc
end
