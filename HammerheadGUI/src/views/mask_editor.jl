# Mask-editor view: the GLMakie shell around a Controllers.MaskEditor.
# Mouse/keyboard input is forwarded into the controller's gesture API; all
# editing state lives in the controller.

"""
    mask_editor(source; size = (1000, 700)) -> Figure

Open the mask editor on `source`: an image matrix, an image-file path
(read with `Hammerhead.load_image`), or a prebuilt [`MaskEditor`](@ref)
controller — pass the controller when you want the drawn mask back
programmatically (`polygon_mask(editor)`), or to seed existing polygons.

Editing model: left-click adds vertices (on empty background it starts a
new polygon; inside an existing polygon it selects it), right-click closes
the active polygon, Backspace undoes the last vertex, Delete removes the
selected polygon. "Save mask…" writes the white-=-excluded grayscale image
`Hammerhead.load_mask` reads back.
"""
mask_editor(source; kwargs...) = mask_editor(MaskEditor(source); kwargs...)

function mask_editor(me::MaskEditor; size = (1000, 700))
    fig = Figure(; size)
    nr, nc = Base.size(me.image)

    ax = Axis(fig[1, 1]; xlabel = "x (px)", ylabel = "y (px)",
              yreversed = true, aspect = DataAspect(),
              title = "left click: add vertex / select · right click: close polygon")

    background = heatmap!(ax, 1:nc, 1:nr, permutedims(me.image);
                          colormap = :grays, nan_color = :black)
    translate!(background, 0, 0, -2)

    controls = GridLayout(fig[1, 2]; tellheight = false, valign = :top)
    status_obs = lift((args...) -> status_text(me),
                      me.polygons, me.active, me.hole_mode, me.selected)
    Label(controls[1, 1], status_obs; halign = :left, justification = :left,
          word_wrap = true, tellwidth = false)
    mask_toggle_grid = GridLayout(controls[2, 1]; halign = :left)
    mask_toggle = Toggle(mask_toggle_grid[1, 1]; active = me.show_mask[])
    Label(mask_toggle_grid[1, 2], "show mask"; halign = :left)
    close_btn = Button(controls[3, 1]; label = "close polygon", tellwidth = false)
    hole_btn = Button(controls[4, 1]; label = "draw hole", tellwidth = false)
    undo_btn = Button(controls[5, 1]; label = "undo vertex", tellwidth = false)
    grow_btn = Button(controls[6, 1]; label = "grow mask 1 px", tellwidth = false)
    shrink_btn = Button(controls[7, 1]; label = "shrink mask 1 px", tellwidth = false)
    delete_btn = Button(controls[8, 1]; label = "delete selected", tellwidth = false)
    clear_btn = Button(controls[9, 1]; label = "clear all", tellwidth = false)
    save_btn = Button(controls[10, 1]; label = "save mask…", tellwidth = false)
    colsize!(fig.layout, 2, Fixed(180))

    # Widget <-> controller.
    _sync_toggle!(mask_toggle, me.show_mask)
    on(_ -> close_active!(me), close_btn.clicks)
    on(_ -> begin_hole!(me), hole_btn.clicks)
    on(_ -> undo_vertex!(me), undo_btn.clicks)
    on(_ -> grow_mask!(me), grow_btn.clicks)
    on(_ -> shrink_mask!(me), shrink_btn.clicks)
    on(_ -> delete_selected!(me), delete_btn.clicks)
    on(_ -> clear_polygons!(me), clear_btn.clicks)
    on(save_btn.clicks) do _
        path = save_file(; filterlist = "png")
        isempty(path) || save_mask(me, path)
    end

    # Mouse and keyboard -> controller gestures.
    on(events(ax.scene).mousebutton) do mb
        if mb.action == Mouse.press && GLMakie.Makie.is_mouseinside(ax.scene)
            pos = GLMakie.Makie.mouseposition(ax.scene)
            if mb.button == Mouse.left
                click!(me, pos[1], pos[2])
            elseif mb.button == Mouse.right
                alt_click!(me)
            end
        end
        return Consume(false)
    end
    on(events(ax.scene).keyboardbutton) do kb
        if kb.action == Keyboard.press
            kb.key == Keyboard.backspace && undo_vertex!(me)
            kb.key == Keyboard.delete && delete_selected!(me)
        end
        return Consume(false)
    end

    # Active polygon: single-observable lifts, so updates stay consistent.
    active_points = lift(a -> [Point2f(v[1], v[2]) for v in a], me.active)
    active_lines = lines!(ax, active_points; color = :orange, linewidth = 2)
    active_verts = scatter!(ax, active_points; color = :orange, markersize = 8)
    translate!(active_lines, 0, 0, 3)
    translate!(active_verts, 0, 0, 3)

    # Committed polygons and the mask overlay are recreated per refresh
    # (their count varies; per-plot observables don't fit a changing set).
    poly_plots = Any[]
    overlay = Ref{Any}(nothing)
    function refresh_polygons!()
        for p in poly_plots
            delete!(ax, p)
        end
        empty!(poly_plots)
        overlay[] === nothing || delete!(ax, overlay[])
        overlay[] = nothing
        for (i, (p, hole)) in enumerate(zip(me.polygons[], me.holes[]))
            sel = me.selected[] == i
            plt = poly!(ax, [Point2f(v[1], v[2]) for v in p];
                        color = (hole ? :dodgerblue : :red, sel ? 0.45 : 0.2),
                        strokecolor = sel ? :yellow : hole ? :dodgerblue : :red,
                        strokewidth = sel ? 2.5 : 1.5)
            translate!(plt, 0, 0, 2)
            push!(poly_plots, plt)
        end
        if me.show_mask[] && (me.raster[] !== nothing || !isempty(me.polygons[]))
            m = polygon_mask(me)
            shade = [v ? 1.0f0 : NaN32 for v in permutedims(m)]
            overlay[] = heatmap!(ax, 1:nc, 1:nr, shade;
                                 colormap = [(:red, 0.35), (:red, 0.35)],
                                 nan_color = :transparent)
            translate!(overlay[], 0, 0, 1)
        end
        return
    end
    onany((args...) -> refresh_polygons!(),
          me.polygons, me.holes, me.raster, me.selected, me.show_mask)

    refresh_polygons!()
    return fig
end
