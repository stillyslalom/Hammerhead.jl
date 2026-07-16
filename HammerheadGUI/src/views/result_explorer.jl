# Result-explorer view: the GLMakie shell around a Controllers.ResultExplorer.
# All state lives in the controller; this file only renders it and forwards
# widget/mouse input into the controller API.

"""
    result_explorer(source; size = (1000, 700)) -> Figure

Open the result explorer on `source`: a `PIVResult` / `StereoPIVResult`, a
vector of them, a results-file path (read with `Hammerhead.load_results`),
or a prebuilt [`ResultExplorer`](@ref) controller (pass the controller when
you want to drive the view programmatically — its observables stay live).

For a gridded (`PIVResult` / `StereoPIVResult`) result the view shows a
scalar field (menu: displacement magnitude, components, diagnostics,
uncertainty when present) as a heatmap in image orientation (y down) with
the vector field as arrows (outliers red). A `PTVResult` is drawn as a
colored particle scatter with optional displacement arrows, and a
`TrackingResult` as trajectory polylines colored by mean speed (breaks at
frame gaps). A frame slider scrubs a sequence, and a click-to-inspect panel
summarizes the selected item in physical units when a scale is attached.
"""
result_explorer(source; kwargs...) = result_explorer(ResultExplorer(source); kwargs...)

function result_explorer(ex::ResultExplorer; size = (1000, 700))
    fig = Figure(; size)
    result_explorer!(fig[1, 1], ex)
    return fig
end

"""
    result_explorer!(target, ex::ResultExplorer) -> GridLayout

Build the result-explorer view into `target` (a `GridPosition`, e.g.
`fig[1, 2]`), for embedding in a larger layout.
"""
function result_explorer!(target, ex::ResultExplorer)
    gl = GridLayout(target)
    n = nframes(ex)

    ax = Axis(gl[1, 1]; xlabel = "x", ylabel = "y",
              yreversed = true, aspect = DataAspect(),
              title = ex.path === nothing ? "" : basename(ex.path))

    # Standalone colorbar driven by observables so the heatmap can be
    # recreated per refresh (grid sizes may change across a mixed sequence).
    crange = Observable((0.0, 1.0))
    clabel = Observable("")
    Colorbar(gl[1, 2]; colormap = :viridis, limits = crange, label = clabel)

    controls = GridLayout(gl[1, 3]; tellheight = false, valign = :top)
    Label(controls[1, 1], "field"; halign = :left, font = :bold)
    menu = Menu(controls[2, 1]; options = [("|displacement|", :magnitude)])
    toggles = GridLayout(controls[3, 1]; halign = :left)
    vec_toggle = Toggle(toggles[1, 1]; active = ex.show_vectors[])
    Label(toggles[1, 2], "vectors"; halign = :left)
    out_toggle = Toggle(toggles[2, 1]; active = ex.highlight_outliers[])
    Label(toggles[2, 2], "flag outliers"; halign = :left)
    Label(controls[4, 1], "color range"; halign = :left, font = :bold)
    cgrid = GridLayout(controls[5, 1]; halign = :left)
    cmode_menu = Menu(cgrid[1, 1:2]; tellwidth = false,
                      options = [("robust (2–98%)", :robust), ("full range", :full)])
    cmin_box = Textbox(cgrid[2, 1]; placeholder = "min: auto", width = 100)
    cmax_box = Textbox(cgrid[2, 2]; placeholder = "max: auto", width = 100)
    Label(controls[6, 1], "click a vector to inspect"; halign = :left, font = :bold)
    info = Label(controls[7, 1], ""; halign = :left, justification = :left)
    colsize!(gl, 3, Fixed(230))

    Label(gl[2, 1:3][1, 1], "frame")
    slider = Slider(gl[2, 1:3][1, 2]; range = 1:max(n, 1), startvalue = ex.frame[])
    Label(gl[2, 1:3][1, 3], lift(i -> "$i / $n", ex.frame))

    # Widget -> controller (equality guards break the notification cycles;
    # Observables notify even when the value is unchanged).
    _sync_toggle!(vec_toggle, ex.show_vectors)
    _sync_toggle!(out_toggle, ex.highlight_outliers)
    on(slider.value) do i
        i == ex.frame[] || set_frame!(ex, i)
    end
    on(ex.frame) do i
        i == slider.value[] || set_close_to!(slider, i)
    end
    on(menu.selection) do f
        f === nothing || f == ex.field[] || set_field!(ex, f)
    end
    _sync_menu!(cmode_menu, ex.color_mode)
    # Manual colorbar bounds: one-way widget -> controller (the box's
    # placeholder documents the cleared state); junk entries are ignored.
    on(cmin_box.stored_string) do s
        s === nothing && return
        try
            set_color_limits!(ex; min = s)
        catch
        end
    end
    on(cmax_box.stored_string) do s
        s === nothing && return
        try
            set_color_limits!(ex; max = s)
        catch
        end
    end

    # Click to inspect (left press inside the axis).
    on(events(ax.scene).mousebutton) do mb
        if mb.button == Mouse.left && mb.action == Mouse.press &&
           GLMakie.Makie.is_mouseinside(ax.scene)
            pos = GLMakie.Makie.mouseposition(ax.scene)
            select_nearest!(ex, pos[1], pos[2])
        end
        return Consume(false)
    end

    # Selection marker and info panel.
    sel_points = Observable(Point2f[])
    sel_plot = scatter!(ax, sel_points; color = :transparent,
                        strokecolor = :cyan, strokewidth = 2.5, markersize = 16)
    translate!(sel_plot, 0, 0, 2)
    onany(ex.selection, ex.frame) do sel, _
        pt = selection_point(current_result(ex), sel)
        sel_points[] = pt === nothing ? Point2f[] : [Point2f(pt[1], pt[2])]
        info.text[] = describe_selection(ex)
    end

    function refresh_menu!()
        fields = available_fields(current_result(ex))
        opts = [(field_name(f), f) for f in fields]
        opts == menu.options[] || (menu.options[] = opts)
        i = something(findfirst(==(ex.field[]), fields), 1)
        i == menu.i_selected[] || (menu.i_selected[] = i)
    end
    on(_ -> refresh_menu!(), ex.frame)
    on(_ -> refresh_menu!(), ex.field) # sync the menu on programmatic set_field!

    # The plots are recreated per refresh rather than driven by per-argument
    # observables: grid sizes (and the plot type itself, across a mixed
    # sequence) can change between frames, and sequential x/y/data updates
    # would render transiently mismatched args.
    plots = Any[]
    has_drawn = Ref(false)

    # Quiver-style fast path: linesegments shafts + one rotated triangle
    # marker per arrow head. arrows2d is a poly/mesh recipe whose pixel-space
    # tip sizing recomputes per pan/zoom frame — interaction crawls with
    # thousands of arrows — while these two primitive plots are static in
    # data space and cheap to render.
    function _draw_arrows!(r)
        ex.show_vectors[] || return
        d = vector_data(r)
        isempty(d.x) && return
        ls = auto_lengthscale(r, d)
        n = length(d.x)
        segs = Vector{Point2f}(undef, 2n)
        tips = Vector{Point2f}(undef, n)
        rots = Vector{Float32}(undef, n)
        for k in 1:n
            tip = Point2f(d.x[k] + ls * d.u[k], d.y[k] + ls * d.v[k])
            segs[2k - 1] = Point2f(d.x[k], d.y[k])
            segs[2k] = tip
            tips[k] = tip
            # marker rotation is screen-space CCW and the axis is yreversed,
            # so (du, dv) points along (du, -dv) on screen; :utriangle points
            # up (screen +y), hence the -π/2 offset.
            rots[k] = Float32(atan(-d.v[k], d.u[k]) - π / 2)
        end
        if ex.highlight_outliers[] && any(d.outlier)
            cols = [o ? :red : :black for o in d.outlier]
            shafts = linesegments!(ax, segs; color = repeat(cols, inner = 2),
                                   linewidth = 1.5)
            heads = scatter!(ax, tips; marker = :utriangle, rotation = rots,
                             markersize = 9, color = cols)
        else
            shafts = linesegments!(ax, segs; color = :black, linewidth = 1.5)
            heads = scatter!(ax, tips; marker = :utriangle, rotation = rots,
                             markersize = 9, color = :black)
        end
        translate!(shafts, 0, 0, 1)
        translate!(heads, 0, 0, 1)
        push!(plots, shafts, heads)
        return
    end

    function _draw!(r::Union{PIVResult,StereoPIVResult})
        data = field_values(r, ex.field[])
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = field_label(r, ex.field[])
        h = heatmap!(ax, collect(r.x), collect(r.y), permutedims(data);
                     colormap = :viridis, colorrange = (lo, hi),
                     nan_color = :transparent)
        translate!(h, 0, 0, -1)
        push!(plots, h)
        _draw_arrows!(r)
        return
    end

    function _draw!(r::PTVResult)
        data = field_values(r, ex.field[])
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = field_label(r, ex.field[])
        if !isempty(r.x)
            sc = scatter!(ax, Point2f.(r.x, r.y); color = Float64.(data),
                          colormap = :viridis, colorrange = (lo, hi), markersize = 9)
            translate!(sc, 0, 0, -1)
            push!(plots, sc)
        end
        _draw_arrows!(r)
        return
    end

    function _draw!(r::TrackingResult)
        speeds = field_values(r, :speed)
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = field_label(r, :speed)
        for (k, t) in pairs(r.trajectories)
            xs, ys = trajectory_points(t)
            length(xs) >= 2 || continue
            col = isfinite(speeds[k]) ? speeds[k] : lo
            ln = lines!(ax, Point2f.(xs, ys); color = Float64(col),
                        colormap = :viridis, colorrange = (lo, hi))
            push!(plots, ln)
        end
        return
    end

    function refresh_plots!()
        r = current_result(ex)
        ax.xlabel[], ax.ylabel[] = Hammerhead.plot_axis_labels(r.scale)

        # capture/restore targetlimits (not limits!): the pre-reversal rect,
        # so the image-orientation yreversed flip survives the restore
        limits = ax.targetlimits[]
        for p in plots
            delete!(ax, p)
        end
        empty!(plots)
        _draw!(r)
        has_drawn[] && (ax.targetlimits[] = limits) # keep the user's zoom across refreshes
        has_drawn[] = true
        return
    end
    onany((args...) -> refresh_plots!(),
          ex.frame, ex.field, ex.show_vectors, ex.highlight_outliers,
          ex.color_mode, ex.color_min, ex.color_max)

    refresh_menu!()
    refresh_plots!()
    return gl
end
