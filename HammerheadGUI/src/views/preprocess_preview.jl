# Preprocess-preview view: the GLMakie shell around a
# Controllers.PreprocessPreview. Side-by-side raw/processed images plus the
# step toggles/parameters; all pipeline state lives in the controller.

"""
    preprocess_preview(source; size = (1200, 640)) -> Figure

Open the preprocessing-pipeline preview on `source`: a representative image
(matrix or path) or a prebuilt [`PreprocessPreview`](@ref) controller. The
raw image renders on the left, the processed result on the right, and the
step list on the far right: a toggle per step, its parameters, and an "▲"
button that moves the step earlier in the pipeline (the summary line shows
the current order). "background…" picks frames to compute the
background-subtraction reference from. Hand the pipeline to a batch with
[`build_preprocess`](@ref) or [`set_preprocess!`](@ref).

When the controller carries a pair frame ([`set_pair!`](@ref)), clicking
the processed image places a single-window correlation probe: the window
outline is drawn at the click, and the panel below reports its
displacement and peak ratio live as steps are toggled and edited.
"""
preprocess_preview(source; kwargs...) =
    preprocess_preview(PreprocessPreview(source); kwargs...)

function preprocess_preview(pp::PreprocessPreview; size = (1200, 640))
    fig = Figure(; size)
    preprocess_preview!(fig[1, 1], pp)
    return fig
end

"""
    preprocess_preview!(target, pp::PreprocessPreview) -> GridLayout

Build the preprocess-preview view into `target` (a `GridPosition`), for
embedding in a larger layout — the embeddable form, like
[`result_explorer!`](@ref).
"""
function preprocess_preview!(target, pp::PreprocessPreview)
    gl = GridLayout(target)

    ax_raw = Axis(gl[1, 1]; title = "raw", yreversed = true, aspect = DataAspect())
    ax_proc = Axis(gl[1, 2]; title = "processed", yreversed = true, aspect = DataAspect())

    # Correlation-probe row under the images: window-size box + live numbers.
    probe_row = GridLayout(gl[2, 1:2]; halign = :left)
    Label(probe_row[1, 1], "probe"; halign = :left, font = :bold)
    probe_box = Textbox(probe_row[1, 2]; placeholder = "window: $(pp.probe_window[])",
                        width = 100)
    probe_label = Label(probe_row[1, 3], Controllers.probe_summary(pp);
                        halign = :left, justification = :left)

    # The steps column is content-sized (no colsize! override — Auto(false)
    # made it ignore content width, so the widgets overflowed onto the
    # processed image). Parameter textboxes get their own row under each
    # step, keeping the widest row at ~250 px.
    steps_col = GridLayout(gl[1, 3]; tellheight = false, valign = :top)
    Label(steps_col[1, 1:3], "pipeline"; halign = :left, font = :bold)
    order_label = Label(steps_col[2, 1:3], Controllers.pipeline_summary(pp);
                        halign = :left, justification = :left,
                        word_wrap = true, width = 240, tellwidth = false)

    # One widget row per catalogue step, its parameters on the row below
    # (rows keep catalogue order; the summary label above shows the live
    # pipeline order after "▲" reorders).
    toggles = Dict{Symbol,Any}()
    row = 3
    for (name, (label, defaults)) in Controllers.PREPROC_CATALOG
        tg = Toggle(steps_col[row, 1]; halign = :left,
                    active = Controllers._step(pp, name).enabled)
        toggles[name] = tg
        Label(steps_col[row, 2], label; halign = :left)
        up = Button(steps_col[row, 3]; label = "▲", halign = :right)
        on(_ -> Controllers.move_step!(pp, name, -1), up.clicks)
        row += 1
        isempty(defaults) && continue
        # span the full column width so two-parameter rows never widen the
        # label column (and never clip at the figure edge)
        pgrid = GridLayout(steps_col[row, 1:3]; halign = :left)
        for (k, (param, default)) in enumerate(defaults)
            box = Textbox(pgrid[1, k]; placeholder = "$param: $default",
                          width = 90)
            on(box.stored_string) do s
                s === nothing && return
                try
                    Controllers.set_step_param!(pp, name, param, s)
                catch
                end
            end
        end
        row += 1
    end
    bg_btn = Button(steps_col[row, 1:2]; label = "background…", tellwidth = false)
    bg_label = Label(steps_col[row, 3],
                     lift(b -> b === nothing ? "none" : "computed", pp.background);
                     halign = :right)
    rowgap!(steps_col, 6)   # 12 content rows must fit the default height

    # Widget -> controller (guards break notification cycles); step toggles
    # sync back from the controller on every steps notification.
    for (name, tg) in toggles
        on(tg.active) do a
            s = Controllers._step(pp, name)
            a == s.enabled && return
            try
                Controllers.enable_step!(pp, name, a)
            catch
                tg.active[] = s.enabled   # e.g. no background yet — revert
            end
        end
    end
    on(pp.steps) do steps
        for s in steps
            tg = toggles[s.name]
            tg.active[] == s.enabled || (tg.active[] = s.enabled)
        end
        order_label.text[] = Controllers.pipeline_summary(pp)
    end
    on(bg_btn.clicks) do _
        paths = pick_multi_file()
        isempty(paths) || Controllers.set_background!(pp, paths)
    end
    on(probe_box.stored_string) do s
        s === nothing && return
        try
            Controllers.set_probe_window!(pp, s)
        catch
        end
    end

    # Click on the processed image places the probe; the window outline and
    # the numbers panel follow the controller's probe_result.
    on(events(ax_proc.scene).mousebutton) do mb
        if mb.button == Mouse.left && mb.action == Mouse.press &&
           GLMakie.Makie.is_mouseinside(ax_proc.scene)
            pos = GLMakie.Makie.mouseposition(ax_proc.scene)
            Controllers.click!(pp, pos[1], pos[2])
        end
        return Consume(false)
    end
    probe_rect = Observable(Point2f[])
    rect_plot = lines!(ax_proc, probe_rect; color = :cyan, linewidth = 2)
    translate!(rect_plot, 0, 0, 1)
    function refresh_probe!()
        res = pp.probe_result[]
        if res === nothing
            probe_rect[] = Point2f[]
        else
            x0, y0, w = res.x0 - 0.5, res.y0 - 0.5, res.window
            probe_rect[] = [Point2f(x0, y0), Point2f(x0 + w, y0),
                            Point2f(x0 + w, y0 + w), Point2f(x0, y0 + w),
                            Point2f(x0, y0)]
        end
        probe_label.text[] = Controllers.probe_summary(pp)
        return
    end
    onany((args...) -> refresh_probe!(), pp.probe_result, pp.probe, pp.image2)
    refresh_probe!()   # a probe placed before the view opened still draws

    # Recreate the image plots per refresh (sizes can change with set_image!).
    raw_plot = Ref{Any}(nothing)
    proc_plot = Ref{Any}(nothing)
    function refresh_images!()
        raw_plot[] === nothing || delete!(ax_raw, raw_plot[])
        proc_plot[] === nothing || delete!(ax_proc, proc_plot[])
        img = pp.image[]
        out = pp.processed[]
        nr, nc = Base.size(img)
        raw_plot[] = heatmap!(ax_raw, 1:nc, 1:nr, permutedims(img);
                              colormap = :grays)
        nrp, ncp = Base.size(out)
        proc_plot[] = heatmap!(ax_proc, 1:ncp, 1:nrp, permutedims(out);
                               colormap = :grays)
        return
    end
    on(_ -> refresh_images!(), pp.processed)
    refresh_images!()
    return gl
end
