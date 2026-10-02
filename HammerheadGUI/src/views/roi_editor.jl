"""
    roi_editor(image_or_controller; batch = nothing, size = (900, 560)) -> Figure

Open a rectangular ROI editor. Click two opposite corners or edit the four
inclusive pixel bounds and press "set bounds". "Full image" resets the
selection. Pass `batch = bc` to apply the completed selection to a planar
[`BatchRunner`](@ref). Errors appear below the form.
"""
roi_editor(source; kwargs...) = roi_editor(ROIEditor(source); kwargs...)
function roi_editor(ed::ROIEditor; size = (900, 560), kwargs...)
    fig = Figure(; size)
    roi_editor!(fig[1, 1], ed; kwargs...)
    return fig
end

"""
    roi_editor!(position, editor::ROIEditor; batch = nothing)

Embed the ROI editor in a figure's `GridPosition`. Mouse gestures are
forwarded to the framework-free controller; the image stays in original
pixel coordinates (`x` = column, `y` = row).
"""
function roi_editor!(position, ed::ROIEditor; batch = nothing)
    layout = GridLayout(position)
    nr, nc = size(ed.image)
    ax = Axis(layout[1, 1]; yreversed = true, aspect = DataAspect(),
              xlabel = "column (px)", ylabel = "row (px)",
              title = "click opposite ROI corners")
    heatmap!(ax, 1:nc, 1:nr, permutedims(ed.image); colormap = :grays)

    outline = Observable(Point2f[])
    anchor = Observable(Point2f[])
    lines!(ax, outline; color = :cyan, linewidth = 2.5)
    scatter!(ax, anchor; color = :cyan, markersize = 12)
    form = GridLayout(layout[1, 2]; tellheight = false, valign = :top)
    Label(form[1, 1:2], "ROI: inclusive pixels"; halign = :left, font = :bold)
    summary = lift((args...) -> Controllers.roi_summary(ed), ed.roi, ed.anchor)
    Label(form[2, 1:2], summary; halign = :left, word_wrap = true, width = 220)
    boxes = Textbox[]
    for (i, name) in enumerate(("first row", "last row", "first column", "last column"))
        Label(form[i + 2, 1], name; halign = :left)
        push!(boxes, Textbox(form[i + 2, 2]; width = 90))
    end
    bounds_btn = Button(form[7, 1:2]; label = "set bounds", tellwidth = false)
    clear_btn = Button(form[8, 1:2]; label = "full image", tellwidth = false)
    apply_btn = batch === nothing ? nothing :
        Button(form[9, 1:2]; label = "apply to batch", tellwidth = false)
    status = Observable("")
    Label(form[10, 1:2], status; halign = :left, justification = :left,
          word_wrap = true, width = 220)
    colsize!(layout, 2, Fixed(240))

    on(ed.roi) do roi
        rows, cols = roi === nothing ? (1:nr, 1:nc) : (roi.rows, roi.cols)
        # The outline lies on pixel edges, so a one-pixel selection is visible.
        x1, x2 = first(cols) - 0.5, last(cols) + 0.5
        y1, y2 = first(rows) - 0.5, last(rows) + 0.5
        outline[] = Point2f[(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        values = (first(rows), last(rows), first(cols), last(cols))
        for (box, value) in zip(boxes, values)
            text = string(value)
            box.displayed_string[] == text || (box.displayed_string[] = text)
            box.stored_string[] == text || (box.stored_string[] = text)
        end
        status[] = ""
    end
    on(p -> anchor[] = p === nothing ? Point2f[] : [Point2f(p...)], ed.anchor)
    notify(ed.roi)
    notify(ed.anchor)

    on(events(ax.scene).mousebutton) do mb
        if mb.button == Mouse.left && mb.action == Mouse.press &&
           GLMakie.Makie.is_mouseinside(ax.scene)
            pos = GLMakie.Makie.mouseposition(ax.scene)
            Controllers.click!(ed, pos[1], pos[2])
        end
        return Consume(false)
    end
    on(bounds_btn.clicks) do _
        try
            # Read displayed text so the focused textbox need not be submitted.
            set_roi!(ed, (box.displayed_string[] for box in boxes)...)
        catch err
            status[] = Controllers._errmsg(err)
        end
    end
    on(_ -> clear_roi!(ed), clear_btn.clicks)
    apply_btn === nothing || on(apply_btn.clicks) do _
        try
            apply_roi!(batch, ed)
            status[] = "applied to batch"
        catch err
            status[] = Controllers._errmsg(err)
        end
    end
    return layout
end
