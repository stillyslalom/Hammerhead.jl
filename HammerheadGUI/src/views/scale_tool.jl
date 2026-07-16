# Scale-tool view: the GLMakie shell around a Controllers.ScaleTool. The
# image renders with the calibration line on top; clicks forward into the
# controller, and the form fields sync one-way into it.

"""
    scale_tool(source; batch = nothing, size = (900, 560)) -> Figure

Open the calibration-line scale tool on `source`: an image (matrix or path)
or a prebuilt [`ScaleTool`](@ref) controller. Click the two endpoints of a
feature of known physical size, enter the separation, units, and frame
interval, and the derived pixel size / scale updates live. Pass a
[`BatchRunner`](@ref) as `batch` to get an "apply to batch" button that
copies the scale into its form ([`apply_scale!`](@ref)).
"""
scale_tool(source; kwargs...) = scale_tool(ScaleTool(source); kwargs...)

function scale_tool(st::ScaleTool; batch = nothing, size = (900, 560))
    fig = Figure(; size)
    nr, nc = Base.size(st.image)
    ax = Axis(fig[1, 1]; yreversed = true, aspect = DataAspect(),
              title = "click two points of known separation")
    heatmap!(ax, 1:nc, 1:nr, permutedims(st.image); colormap = :grays)

    # Line + endpoint markers redrawn from the controller's points.
    pts_obs = Observable(Point2f[])
    lines!(ax, pts_obs; color = :cyan, linewidth = 2)
    scatter!(ax, pts_obs; color = :transparent, strokecolor = :cyan,
             strokewidth = 2.5, markersize = 14)
    on(p -> pts_obs[] = [Point2f(x, y) for (x, y) in p], st.points)
    notify(st.points)

    form = GridLayout(fig[1, 2]; tellheight = false, valign = :top)
    Label(form[1, 1:2], "calibration line"; halign = :left, font = :bold)
    summary_label = Label(form[2, 1:2], Controllers.scale_summary(st);
                          halign = :left, justification = :left,
                          word_wrap = true, width = 210)
    Label(form[3, 1], "separation"; halign = :left)
    sep_box = Textbox(form[3, 2]; placeholder = string(st.separation[]), width = 100)
    Label(form[4, 1], "length unit"; halign = :left)
    lu_box = Textbox(form[4, 2]; placeholder = st.length_unit[], width = 100)
    Label(form[5, 1], "dt"; halign = :left)
    dt_box = Textbox(form[5, 2]; placeholder = string(st.dt[]), width = 100)
    Label(form[6, 1], "time unit"; halign = :left)
    tu_box = Textbox(form[6, 2]; placeholder = st.time_unit[], width = 100)
    clear_btn = Button(form[7, 1:2]; label = "clear points", tellwidth = false)
    apply_btn = batch === nothing ? nothing :
        Button(form[8, 1:2]; label = "apply to batch", tellwidth = false)
    colsize!(fig.layout, 2, Fixed(230))

    # Clicks place endpoints (left press inside the axis).
    on(events(ax.scene).mousebutton) do mb
        if mb.button == Mouse.left && mb.action == Mouse.press &&
           GLMakie.Makie.is_mouseinside(ax.scene)
            pos = GLMakie.Makie.mouseposition(ax.scene)
            Controllers.click!(st, pos[1], pos[2])
        end
        return Consume(false)
    end

    # Form -> controller (one-way; placeholders show the current values).
    on(sep_box.stored_string) do s
        s === nothing && return
        try
            Controllers.set_separation!(st, s)
        catch
        end
    end
    on(dt_box.stored_string) do s
        s === nothing && return
        try
            Controllers.set_dt!(st, s)
        catch
        end
    end
    on(s -> s === nothing || (st.length_unit[] = s), lu_box.stored_string)
    on(s -> s === nothing || (st.time_unit[] = s), tu_box.stored_string)
    on(_ -> Controllers.clear_points!(st), clear_btn.clicks)
    apply_btn === nothing || on(apply_btn.clicks) do _
        try
            Controllers.apply_scale!(batch, st)
        catch
        end
    end

    onany((args...) -> summary_label.text[] = Controllers.scale_summary(st),
          st.points, st.separation, st.length_unit)
    return fig
end
