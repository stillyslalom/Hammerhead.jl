# Result-explorer view: the GLMakie shell around a Controllers.ResultExplorer.
# All state lives in the controller; this file only renders it and forwards
# widget/mouse input into the controller API.

"""
    result_explorer(source; size = (1000, 700)) -> Figure

Open the result explorer on `source`: a `PIVResult` / `StereoPIVResult`, a
vector of them, a results-file path (read with `Hammerhead.load_results`),
or a prebuilt [`ResultExplorer`](@ref) controller (pass the controller when
you want to drive the view programmatically — its observables stay live).

The view shows a scalar field (menu: displacement magnitude, components,
diagnostics, uncertainty when present) as a heatmap in image orientation
(y down), the vector field as arrows (outliers red), a frame slider for
sequences, and a click-to-inspect panel for individual vectors.
"""
result_explorer(source; kwargs...) = result_explorer(ResultExplorer(source); kwargs...)

function result_explorer(ex::ResultExplorer; size = (1000, 700))
    fig = Figure(; size)
    n = nframes(ex)

    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y",
              yreversed = true, aspect = DataAspect(),
              title = ex.path === nothing ? "" : basename(ex.path))

    # Standalone colorbar driven by observables so the heatmap can be
    # recreated per refresh (grid sizes may change across a mixed sequence).
    crange = Observable((0.0, 1.0))
    clabel = Observable("")
    Colorbar(fig[1, 2]; colormap = :viridis, limits = crange, label = clabel)

    controls = GridLayout(fig[1, 3]; tellheight = false, valign = :top)
    Label(controls[1, 1], "field"; halign = :left, font = :bold)
    menu = Menu(controls[2, 1]; options = [("|displacement|", :magnitude)])
    toggles = GridLayout(controls[3, 1]; halign = :left)
    vec_toggle = Toggle(toggles[1, 1]; active = ex.show_vectors[])
    Label(toggles[1, 2], "vectors"; halign = :left)
    out_toggle = Toggle(toggles[2, 1]; active = ex.highlight_outliers[])
    Label(toggles[2, 2], "flag outliers"; halign = :left)
    Label(controls[4, 1], "click a vector to inspect"; halign = :left, font = :bold)
    info = Label(controls[5, 1], ""; halign = :left, justification = :left)
    colsize!(fig.layout, 3, Fixed(230))

    Label(fig[2, 1:3][1, 1], "frame")
    slider = Slider(fig[2, 1:3][1, 2]; range = 1:max(n, 1), startvalue = ex.frame[])
    Label(fig[2, 1:3][1, 3], lift(i -> "$i / $n", ex.frame))

    # Widget -> controller (equality guards break the notification cycles;
    # Observables notify even when the value is unchanged).
    on(a -> ex.show_vectors[] = a, vec_toggle.active)
    on(a -> ex.highlight_outliers[] = a, out_toggle.active)
    on(slider.value) do i
        i == ex.frame[] || set_frame!(ex, i)
    end
    on(ex.frame) do i
        i == slider.value[] || set_close_to!(slider, i)
    end
    on(menu.selection) do f
        f === nothing || f == ex.field[] || set_field!(ex, f)
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
        r = current_result(ex)
        sel_points[] = (sel === nothing || !checkbounds(Bool, r.u, sel)) ?
            Point2f[] : [Point2f(r.x[sel[2]], r.y[sel[1]])]
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

    # The heatmap and arrows are recreated per refresh rather than driven by
    # per-argument observables: grid sizes can change between frames, and
    # sequential x/y/data updates would render transiently mismatched args.
    heat = Ref{Any}(nothing)
    arrows = Ref{Any}(nothing)
    has_drawn = Ref(false)
    function refresh_plots!()
        r = current_result(ex)
        data = field_values(r, ex.field[])
        vals = [v for v in data if isfinite(v)]
        lo, hi = isempty(vals) ? (0.0, 1.0) : Float64.(extrema(vals))
        lo == hi && ((lo, hi) = (lo - 0.5, hi + 0.5))
        crange[] = (lo, hi)
        clabel[] = field_label(r, ex.field[])

        # capture/restore targetlimits (not limits!): the pre-reversal rect,
        # so the image-orientation yreversed flip survives the restore
        limits = ax.targetlimits[]
        for p in (heat, arrows)
            p[] === nothing || delete!(ax, p[])
            p[] = nothing
        end
        heat[] = heatmap!(ax, collect(r.x), collect(r.y), permutedims(data);
                          colormap = :viridis, colorrange = (lo, hi),
                          nan_color = :transparent)
        translate!(heat[], 0, 0, -1)
        if ex.show_vectors[]
            d = vector_data(r)
            if !isempty(d.x)
                colors = ex.highlight_outliers[] ?
                    [o ? :red : :black for o in d.outlier] : :black
                arrows[] = arrows2d!(ax, Point2f.(d.x, d.y), Vec2f.(d.u, d.v);
                                     color = colors,
                                     lengthscale = auto_lengthscale(r, d))
                translate!(arrows[], 0, 0, 1)
            end
        end
        has_drawn[] && (ax.targetlimits[] = limits) # keep the user's zoom across refreshes
        has_drawn[] = true
        return
    end
    onany((args...) -> refresh_plots!(),
          ex.frame, ex.field, ex.show_vectors, ex.highlight_outliers)

    refresh_menu!()
    refresh_plots!()
    return fig
end
