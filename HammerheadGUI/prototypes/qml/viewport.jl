# Scientific viewport shared by native-bridge and static-image fallback probes.
using GLMakie, Hammerhead
using HammerheadGUI.Controllers

function viewport(state)
    fig = Figure(size = (900, 650))
    ax = Axis(fig[1, 1]; title = "Dense planar PIV: 16,384 vectors",
              xlabel = "image x (px)", ylabel = "image y (px)", yreversed = true,
              aspect = DataAspect())
    # 1024-square image workload, sampled from the same demo particle image.
    image = Float32[state.image[clamp(ceil(Int, y * 96 / 1024), 1, 96),
                               clamp(ceil(Int, x * 96 / 1024), 1, 96)]
                    for y in 1:1024, x in 1:1024]
    background = heatmap!(ax, range(0, 96; length = 1024), range(0, 96; length = 1024),
             image'; colormap = :grays)
    segments = Observable(Point2f[])
    selection = Observable(Point2f[])
    outline = Observable(Point2f[])
    last_geometry = Ref{Any}(nothing)
    linesegments!(ax, segments; color = :cyan, linewidth = 0.7)
    scatter!(ax, selection; color = :orange, markersize = 16, strokewidth = 2)
    lines!(ax, outline; color = :red, linewidth = 3)
    state.refresh = () -> begin
        r = current_result(state.explorer)
        r isa Hammerhead.PIVResult || error("viewport supports planar PIV entries only")
        background.visible[] = state.explorer.path === nothing
        data = vector_data(r)
        scale = length(r.x) > 32 ? 0.10f0 : 1.0f0
        points = Point2f[]
        for (x, y, u, v) in zip(data.x, data.y, data.u, data.v)
            a = Point2f(x, y)
            b = a + scale * Vec2f(u, v)
            push!(points, a, b)
            delta = b - a
            push!(points, b, b - 0.3f0 * delta + 0.15f0 * Vec2f(-delta[2], delta[1]))
            push!(points, b, b - 0.3f0 * delta - 0.15f0 * Vec2f(-delta[2], delta[1]))
        end
        segments[] = points
        point = selection_point(r, state.explorer.selection[])
        selection[] = point === nothing ? Point2f[] : [Point2f(point...)]
        poly = isempty(state.mask.polygons[]) ? state.mask.active[] : state.mask.polygons[][end]
        outline[] = isempty(poly) ? Point2f[] : Point2f[Point2f(p...) for p in [poly; [first(poly)]]]
        ax.title[] = "Planar PIV: $(length(data.x)) vectors | frame $(state.frame[]) / $(state.count[])"
        geometry = (extrema(r.x), extrema(r.y), state.explorer.path === nothing)
        if geometry != last_geometry[] && state.explorer.path === nothing
            limits!(ax, 0, 96, 0, 96)
        elseif geometry != last_geometry[]
            xmin, xmax = extrema(r.x); ymin, ymax = extrema(r.y)
            padx = max((xmax - xmin) * 0.03, 1.0)
            pady = max((ymax - ymin) * 0.03, 1.0)
            limits!(ax, xmin - padx, xmax + padx, ymin - pady, ymax + pady)
        end
        last_geometry[] = geometry
        ax.yreversed[] = true
        nothing
    end
    state.refresh()
    # The native bridge forwards input to the ordinary Makie axis interactions.
    on(events(fig).mousebutton) do event
        if event.action == Mouse.press && event.button == Mouse.left
            p = mouseposition(ax)
            Prototype.pick(state, p[1], p[2]; drawing = state.drawing)
        end
        Consume(false)
    end
    fig, ax
end
