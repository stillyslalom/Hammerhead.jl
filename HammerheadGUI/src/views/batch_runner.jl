# Batch-runner view: the GLMakie parameter form + run panel around a
# Controllers.BatchRunner. Widgets push into the controller with guarded
# two-way sync; the run itself is the controller's cooperative task.

"""
    batch_runner(bc = BatchRunner(); size = (900, 520)) -> Figure

Open the parameter form + batch runner. Pick frames ("add frames…") and the
pairing mode, edit the multi-pass window schedule ("64, 32, 32" style) and
the correlation/validation options, optionally choose an incremental JLD2
output file and an analysis mask image, then run — progress and status
update live, "cancel" stops after the pair in flight, and "explore results"
opens the finished batch in [`result_explorer`](@ref).

Pass a prebuilt [`BatchRunner`](@ref) to seed the form (e.g. with in-memory
frames) or to drive it programmatically.
"""
batch_runner(; kwargs...) = batch_runner(BatchRunner(); kwargs...)

function batch_runner(bc::BatchRunner; size = (900, 520))
    fig = Figure(; size)

    # -- frames column ------------------------------------------------------
    files_col = GridLayout(fig[1, 1]; tellheight = false, valign = :top)
    Label(files_col[1, 1], "frames"; halign = :left, font = :bold)
    files_info = lift(bc.files, bc.pair_mode) do files, _
        n = length(files)
        msg = validate(bc)
        pairs_txt = msg === nothing ? "$(length(frame_pairs(bc))) pairs" :
                    n == 0 ? "none" : msg
        return "$n frame" * (n == 1 ? "" : "s") * " · " * pairs_txt
    end
    Label(files_col[2, 1], files_info; halign = :left, justification = :left,
          word_wrap = true, width = 180)
    mode_menu = Menu(files_col[3, 1]; tellwidth = false,
                     options = [("paired: 1-2, 3-4, …", :paired),
                                ("chained: 1-2, 2-3, …", :chained)])
    add_btn = Button(files_col[4, 1]; label = "add frames…", tellwidth = false)
    clear_btn = Button(files_col[5, 1]; label = "clear frames", tellwidth = false)

    # -- parameters column --------------------------------------------------
    form = GridLayout(fig[1, 2]; tellheight = false, valign = :top)
    Label(form[1, 1:2], "parameters"; halign = :left, font = :bold)
    Label(form[2, 1], "windows"; halign = :left)
    schedule_box = Textbox(form[2, 2];
                           placeholder = join(bc.window_schedule[], ", "),
                           width = 110)
    Label(form[3, 1], "overlap"; halign = :left)
    overlap_slider = Slider(form[3, 2]; range = 0.0:0.05:0.75,
                            startvalue = bc.overlap_fraction[])
    Label(form[4, 1], "correlation"; halign = :left)
    corr_menu = Menu(form[4, 2]; tellwidth = false,
                     options = [("cross", :cross), ("phase", :phase)])
    Label(form[5, 1], "apodization"; halign = :left)
    apod_menu = Menu(form[5, 2]; options = [("none", :none), ("gauss", :gauss)])
    Label(form[6, 1], "subpixel"; halign = :left)
    subpx_menu = Menu(form[6, 2]; options = [("gauss3", :gauss3),
                                             ("gauss9", :gauss9),
                                             ("gauss2d", :gauss2d)])
    pad_toggle = Toggle(form[7, 1]; active = bc.padding[], halign = :right)
    Label(form[7, 2], "padding"; halign = :left)
    unc_toggle = Toggle(form[8, 1]; active = bc.uncertainty[], halign = :right)
    Label(form[8, 2], "uncertainty"; halign = :left)
    summary_obs = lift((args...) -> _form_summary(bc), bc.window_schedule,
                       bc.overlap_fraction, bc.correlation_method, bc.padding,
                       bc.apodization, bc.subpixel_method, bc.uncertainty)
    Label(form[9, 1:2], summary_obs; halign = :left, justification = :left,
          word_wrap = true, width = 230)

    # -- run column ----------------------------------------------------------
    run_col = GridLayout(fig[1, 3]; tellheight = false, valign = :top)
    Label(run_col[1, 1], "output"; halign = :left, font = :bold)
    output_obs = lift(p -> isempty(p) ? "in memory only" : basename(p), bc.output_path)
    Label(run_col[2, 1], output_obs; halign = :left, word_wrap = true, width = 160)
    output_btn = Button(run_col[3, 1]; label = "choose output…", tellwidth = false)
    mask_obs = lift(m -> m === nothing ? "mask: none" :
                         "mask: $(count(m)) px excluded", bc.mask)
    Label(run_col[4, 1], mask_obs; halign = :left, word_wrap = true, width = 160)
    mask_btn = Button(run_col[5, 1]; label = "load mask…", tellwidth = false)
    run_btn = Button(run_col[6, 1]; label = "run", tellwidth = false)
    cancel_btn = Button(run_col[7, 1]; label = "cancel", tellwidth = false)
    progress_obs = lift(bc.progress, bc.running) do (done, total), running
        total == 0 ? "" : "$done / $total pairs" * (running ? "…" : "")
    end
    Label(run_col[8, 1], progress_obs; halign = :left)
    Label(run_col[9, 1], bc.status; halign = :left, justification = :left,
          word_wrap = true, width = 160)
    explore_btn = Button(run_col[10, 1]; label = "explore results", tellwidth = false)

    colsize!(fig.layout, 1, Fixed(190))
    colsize!(fig.layout, 3, Fixed(170))
    colgap!(fig.layout, 30)

    # -- widget <-> controller wiring (guards break notification cycles) ----
    _sync_menu!(mode_menu, bc.pair_mode)
    _sync_menu!(corr_menu, bc.correlation_method)
    _sync_menu!(apod_menu, bc.apodization)
    _sync_menu!(subpx_menu, bc.subpixel_method)
    _sync_toggle!(pad_toggle, bc.padding)
    _sync_toggle!(unc_toggle, bc.uncertainty)
    on(overlap_slider.value) do v
        v == bc.overlap_fraction[] || (bc.overlap_fraction[] = v)
    end
    on(bc.overlap_fraction) do v
        v == overlap_slider.value[] || set_close_to!(overlap_slider, v)
    end
    on(schedule_box.stored_string) do s
        s === nothing && return
        try
            set_schedule!(bc, s)
        catch err
            bc.status[] = Controllers._errmsg(err)
        end
    end

    on(add_btn.clicks) do _
        paths = pick_multi_file()
        isempty(paths) || add_files!(bc, paths)
    end
    on(_ -> clear_files!(bc), clear_btn.clicks)
    on(output_btn.clicks) do _
        path = save_file(; filterlist = "jld2")
        isempty(path) || (bc.output_path[] = path)
    end
    on(mask_btn.clicks) do _
        path = pick_file(; filterlist = "png;tif;tiff;bmp")
        isempty(path) && return
        try
            bc.mask[] = load_mask(path)
        catch err
            bc.status[] = Controllers._errmsg(err)
        end
    end
    on(_ -> start!(bc), run_btn.clicks)
    on(_ -> cancel!(bc), cancel_btn.clicks)
    on(explore_btn.clicks) do _
        results = bc.results[]
        results === nothing && return
        display(GLMakie.Screen(), result_explorer(ResultExplorer(results)))
    end

    return fig
end

function _form_summary(bc::BatchRunner)
    return string(join(bc.window_schedule[], "/"), " px, ",
                  round(Int, 100 * bc.overlap_fraction[]), "% overlap, ",
                  bc.correlation_method[],
                  bc.padding[] ? ", padded" : "",
                  bc.apodization[] === :none ? "" : ", $(bc.apodization[])",
                  ", ", bc.subpixel_method[],
                  bc.uncertainty[] ? ", uncertainty" : "")
end
