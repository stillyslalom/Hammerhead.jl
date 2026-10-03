# Scientific viewport shared by native-bridge and static-image fallback probes.
using GLMakie, Hammerhead
using HammerheadGUI.Controllers

function axis_padding(values,other)
    span=maximum(values)-minimum(values)
    differences=filter(>(0),abs.(diff(values)))
    spacing=isempty(differences) ? 0. : minimum(differences)
    local_extent=max(maximum(abs,values),maximum(abs,other))
    fallback=local_extent>0 ? .05local_extent : eps(Float64)
    max(.03span,.35spacing,span>0 || spacing>0 ? 0. : fallback)
end

function arrow_spacing(xs,ys)
    spacings=filter(>(0),vcat(abs.(diff(xs)),abs.(diff(ys))))
    isempty(spacings) ? min(axis_padding(xs,ys),axis_padding(ys,xs)) : minimum(spacings)
end

function vector_glyph_padding(xs,ys)
    # The furthest normalized glyph vertex is its tip (.65 * spacing).
    # Arrowhead vertices have norm sqrt(.7^2+.15^2) of that displacement.
    # Reserve another .1 spacing for the stroked line, in coordinate units.
    max(axis_padding(xs,ys), .75arrow_spacing(xs,ys))
end

function viewport(state; managed = false)
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
    vector_color=Observable(RGBAf(0,1,1,1))
    linesegments!(ax, segments; color = vector_color, linewidth = 0.7)
    scatter!(ax, selection; color = :orange, markersize = 16, strokewidth = 2)
    lines!(ax, outline; color = :red, linewidth = 3)
    invalidate=()->begin
        segments[]=Point2f[];selection[]=Point2f[];outline[]=Point2f[]
        background.visible[]=false
        ax.title[]="Plot unavailable after rendering failure"
        nothing
    end
    state.invalidate=invalidate
    refresh = () -> begin
        state.render_available[] || return invalidate()
        vector_color[]=state.dataset[]===:demo ? RGBAf(0,1,1,1) : RGBAf(.04,.25,.35,1)
        r = current_result(state.explorer)
        r isa Hammerhead.PIVResult || error("viewport supports planar PIV entries only")
        background.visible[] = state.dataset[]===:demo && state.explorer.path === nothing
        ax.xlabel[],ax.ylabel[]=Hammerhead.plot_axis_labels(r.scale)
        data = vector_data(r)
        # Display lengths are normalized; dimensional velocities are not added
        # directly to dimensional positions, and physical values stay intact.
        spacing=arrow_spacing(r.x,r.y)
        magnitudes=[hypot(u,v) for (u,v) in zip(data.u,data.v) if isfinite(u) && isfinite(v)]
        largest=isempty(magnitudes) ? 0. : maximum(magnitudes)
        scale=largest>0 ? .65spacing/largest : 0.
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
        outline[] = state.dataset[]!==:demo || isempty(poly) ? Point2f[] : Point2f[Point2f(p...) for p in [poly; [first(poly)]]]
        ax.title[] = "Planar PIV: $(length(data.x)) vectors | frame $(state.frame[]) / $(state.count[])\nArrows normalized to grid spacing; $(field_label(r,:u)), $(field_label(r,:v))"
        geometry = (extrema(r.x), extrema(r.y), background.visible[])
        if geometry != last_geometry[] && background.visible[]
            limits!(ax, 0, 96, 0, 96)
        elseif geometry != last_geometry[]
            xmin, xmax = extrema(r.x); ymin, ymax = extrema(r.y)
            padx = vector_glyph_padding(r.x,r.y)
            pady = vector_glyph_padding(r.y,r.x)
            limits!(ax, xmin - padx, xmax + padx, ymin - pady, ymax + pady)
        end
        last_geometry[] = geometry
        ax.yreversed[] = true
        nothing
    end
    state.refresh = refresh
    refresh()
    # The native bridge forwards input to the ordinary Makie axis interactions.
    subscription = on(events(fig).mousebutton) do event
        if event.action == Mouse.press && event.button == Mouse.left
            p = mouseposition(ax)
            Prototype.pick(state, p[1], p[2]; drawing = state.drawing)
        end
        Consume(false)
    end
    if managed
        detach = () -> begin
            state.refresh === refresh && (state.refresh = () -> nothing)
            state.invalidate === invalidate && (state.invalidate = () -> nothing)
            nothing
        end
        return fig, ax, refresh, [subscription], detach
    end
    fig, ax
end
