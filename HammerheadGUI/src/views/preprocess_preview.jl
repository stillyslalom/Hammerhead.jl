# Preprocess-preview view: the GLMakie shell around a
# Controllers.PreprocessPreview. Side-by-side raw/processed images plus the
# step toggles/parameters; all pipeline state lives in the controller.

"""
    preprocess_preview(source; size = (1100, 560)) -> Figure

Open the preprocessing-pipeline preview on `source`: a representative image
(matrix or path) or a prebuilt [`PreprocessPreview`](@ref) controller. The
raw image renders on the left, the processed result on the right, and the
step list on the far right: a toggle per step, its parameters, and an "▲"
button that moves the step earlier in the pipeline (the summary line shows
the current order). "background…" picks frames to compute the
background-subtraction reference from. Hand the pipeline to a batch with
[`build_preprocess`](@ref) or [`set_preprocess!`](@ref).
"""
preprocess_preview(source; kwargs...) =
    preprocess_preview(PreprocessPreview(source); kwargs...)

function preprocess_preview(pp::PreprocessPreview; size = (1100, 560))
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

    steps_col = GridLayout(gl[1, 3]; tellheight = false, valign = :top)
    Label(steps_col[1, 1:4], "pipeline"; halign = :left, font = :bold)
    order_label = Label(steps_col[2, 1:4], Controllers.pipeline_summary(pp);
                        halign = :left, justification = :left,
                        word_wrap = true, width = 240)

    # One fixed widget row per catalogue step (rows keep catalogue order; the
    # summary label above shows the live pipeline order after "▲" reorders).
    toggles = Dict{Symbol,Any}()
    row = 3
    for (name, (label, defaults)) in Controllers.PREPROC_CATALOG
        tg = Toggle(steps_col[row, 1]; halign = :right)
        toggles[name] = tg
        Label(steps_col[row, 2], label; halign = :left)
        up = Button(steps_col[row, 3]; label = "▲")
        on(_ -> Controllers.move_step!(pp, name, -1), up.clicks)
        col = 4
        for (param, default) in defaults
            box = Textbox(steps_col[row, col]; placeholder = "$param: $default",
                          width = 90)
            on(box.stored_string) do s
                s === nothing && return
                try
                    Controllers.set_step_param!(pp, name, param, s)
                catch
                end
            end
            col += 1
            col > 5 && (row += 1; col = 4)   # wrap two-param steps
        end
        row += 1
    end
    bg_btn = Button(steps_col[row, 1:2]; label = "background…", tellwidth = false)
    bg_label = Label(steps_col[row, 3:4],
                     lift(b -> b === nothing ? "none" : "computed", pp.background);
                     halign = :left)
    colsize!(gl, 3, Auto(false))

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
