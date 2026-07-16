# Stereo-batch views: the batch form around a Controllers.StereoBatchRunner,
# and the rig-setup workflow composing two embedded calibration reviews with
# the dewarper-grid options.

"""
    stereo_batch_runner(sbc = StereoBatchRunner(); size = (960, 600)) -> Figure

Open the stereo batch runner. Add each camera's frames, set the dewarpers
(build them from calibrations with [`stereo_calibration`](@ref), or
[`set_dewarpers!`](@ref) from the REPL), pick an effort preset or edit the
manual window schedule, optionally enter the frame interval (`dt` + units —
stereo results are already in world units, so the attached scale converts
displacements to velocities), choose an output file, and run. Cancellation
uses [`run_piv_stereo_sequence`](@ref)'s native between-acquisition
predicate, so the completed prefix always lands in `results`. "view
results" opens the completed acquisitions in the result explorer as soon as
the first one is done, live-appending the rest.
"""
stereo_batch_runner(; kwargs...) = stereo_batch_runner(StereoBatchRunner(); kwargs...)

function stereo_batch_runner(sbc::StereoBatchRunner; size = (960, 600))
    fig = Figure(; size)

    # -- frames column ------------------------------------------------------
    files_col = GridLayout(fig[1, 1]; tellheight = false, valign = :top)
    Label(files_col[1, 1], "frames"; halign = :left, font = :bold)
    files_info = lift(sbc.files1, sbc.files2, sbc.pair_mode, sbc.dewarpers) do f1, f2, _, _
        msg = Controllers.validate(sbc)
        return "cam 1: $(length(f1)) · cam 2: $(length(f2))" *
               (msg === nothing ? "" : "\n" * msg)
    end
    Label(files_col[2, 1], files_info; halign = :left, justification = :left,
          word_wrap = true, width = 180)
    mode_menu = Menu(files_col[3, 1]; tellwidth = false,
                     options = [("paired: 1-2, 3-4, …", :paired),
                                ("chained: 1-2, 2-3, …", :chained)])
    add1_btn = Button(files_col[4, 1]; label = "add cam 1 frames…", tellwidth = false)
    add2_btn = Button(files_col[5, 1]; label = "add cam 2 frames…", tellwidth = false)
    clear_btn = Button(files_col[6, 1]; label = "clear frames", tellwidth = false)
    dw_info = lift(d -> d === nothing ? "dewarpers: not set" :   # Base.size: the
                        "dewarpers: grid $(join(Base.size(d[1].grid), "×"))",
                   sbc.dewarpers)                                # kwarg shadows it
    Label(files_col[7, 1], dw_info; halign = :left, word_wrap = true, width = 180)

    # -- parameters column --------------------------------------------------
    form = GridLayout(fig[1, 2]; tellheight = false, valign = :top)
    Label(form[1, 1:2], "parameters"; halign = :left, font = :bold)
    Label(form[2, 1], "effort"; halign = :left)
    effort_menu = Menu(form[2, 2]; tellwidth = false,
                       options = [("custom", :custom), ("low", :low),
                                  ("medium", :medium), ("high", :high)])
    Label(form[3, 1], "windows"; halign = :left)
    schedule_box = Textbox(form[3, 2];
                           placeholder = join(sbc.window_schedule[], ", "),
                           width = 110)
    Label(form[4, 1], "overlap"; halign = :left)
    overlap_slider = Slider(form[4, 2]; range = 0.0:0.05:0.75,
                            startvalue = sbc.overlap_fraction[])
    unc_toggle = Toggle(form[5, 1]; active = sbc.uncertainty[], halign = :right)
    Label(form[5, 2], "uncertainty"; halign = :left)
    Label(form[6, 1:2], "physical scale (dt only)"; halign = :left, font = :bold)
    Label(form[7, 1], "dt"; halign = :left)
    dt_box = Textbox(form[7, 2]; placeholder = string(sbc.dt[]), width = 110)
    Label(form[8, 1], "length unit"; halign = :left)
    lu_box = Textbox(form[8, 2]; placeholder = sbc.length_unit[], width = 110)
    Label(form[9, 1], "time unit"; halign = :left)
    tu_box = Textbox(form[9, 2]; placeholder = sbc.time_unit[], width = 110)

    # -- run column ----------------------------------------------------------
    run_col = GridLayout(fig[1, 3]; tellheight = false, valign = :top)
    Label(run_col[1, 1], "output"; halign = :left, font = :bold)
    output_obs = lift(p -> isempty(p) ? "in memory only" : basename(p), sbc.output_path)
    Label(run_col[2, 1], output_obs; halign = :left, word_wrap = true, width = 160)
    output_btn = Button(run_col[3, 1]; label = "choose output…", tellwidth = false)
    run_btn = Button(run_col[4, 1]; label = "run", tellwidth = false)
    cancel_btn = Button(run_col[5, 1]; label = "cancel", tellwidth = false)
    progress_obs = lift(sbc.progress, sbc.running) do (done, total), running
        total == 0 ? "" : "$done / $total" * (running ? "…" : "")
    end
    Label(run_col[6, 1], progress_obs; halign = :left)
    Label(run_col[7, 1], sbc.status; halign = :left, justification = :left,
          word_wrap = true, width = 160)
    explore_label = lift(v -> isempty(v) ? "view results" :
                              "view results ($(length(v)))", sbc.completed)
    explore_btn = Button(run_col[8, 1]; label = explore_label, tellwidth = false)

    colsize!(fig.layout, 1, Fixed(190))
    colsize!(fig.layout, 3, Fixed(170))
    colgap!(fig.layout, 30)

    # -- widget <-> controller wiring (guards break notification cycles) ----
    _sync_menu!(mode_menu, sbc.pair_mode)
    _sync_menu!(effort_menu, sbc.effort)
    _sync_toggle!(unc_toggle, sbc.uncertainty)
    on(overlap_slider.value) do v
        v == sbc.overlap_fraction[] || (sbc.overlap_fraction[] = v)
    end
    on(sbc.overlap_fraction) do v
        v == overlap_slider.value[] || set_close_to!(overlap_slider, v)
    end
    on(schedule_box.stored_string) do s
        s === nothing && return
        try
            set_schedule!(sbc, s)
        catch err
            sbc.status[] = Controllers._errmsg(err)
        end
    end
    on(dt_box.stored_string) do s
        s === nothing && return
        try
            Controllers.set_dt!(sbc, s)
        catch err
            sbc.status[] = Controllers._errmsg(err)
        end
    end
    on(s -> s === nothing || (sbc.length_unit[] = s), lu_box.stored_string)
    on(s -> s === nothing || (sbc.time_unit[] = s), tu_box.stored_string)

    on(add1_btn.clicks) do _
        paths = pick_multi_file()
        isempty(paths) || add_files!(sbc, paths; camera = 1)
    end
    on(add2_btn.clicks) do _
        paths = pick_multi_file()
        isempty(paths) || add_files!(sbc, paths; camera = 2)
    end
    on(_ -> clear_files!(sbc), clear_btn.clicks)
    on(output_btn.clicks) do _
        path = save_file(; filterlist = "jld2")
        isempty(path) || (sbc.output_path[] = path)
    end
    on(_ -> start!(sbc), run_btn.clicks)
    on(_ -> cancel!(sbc), cancel_btn.clicks)

    # Live results hand-off, same pattern as the planar batch view.
    live_ex = Ref{Union{Nothing,ResultExplorer}}(nothing)
    on(sbc.completed) do v
        isempty(v) && (live_ex[] = nothing; return)
        ex = live_ex[]
        ex === nothing && return
        while nframes(ex) < length(v)
            push_result!(ex, v[nframes(ex) + 1])
        end
    end
    on(explore_btn.clicks) do _
        v = sbc.completed[]
        isempty(v) && return
        ex = ResultExplorer(copy(v))
        live_ex[] = ex
        display(GLMakie.Screen(), result_explorer(ex))
    end

    return fig
end

"""
    stereo_calibration(cr1, cr2; batch = nothing, size = (1500, 850)) -> Figure

Open the stereo rig setup: the two cameras' [`CalibrationReview`](@ref)s
embedded side by side (plane sliders, model menus, reprojection errors),
the dewarp-grid options (coverage and spacing), and a "build dewarpers"
button running [`build_dewarpers`](@ref) over the two fitted cameras. Pass
a [`StereoBatchRunner`](@ref) as `batch` to install the built pair directly
into its form ([`set_dewarpers!`](@ref)); without one the figure still
reports the built grid in its status line (build programmatically for a
REPL hand-off).
"""
function stereo_calibration(cr1::CalibrationReview, cr2::CalibrationReview;
                            batch = nothing, size = (1500, 850))
    fig = Figure(; size)
    Label(fig[1, 1], "camera 1"; font = :bold, tellwidth = false)
    Label(fig[1, 2], "camera 2"; font = :bold, tellwidth = false)
    calibration_review!(fig[2, 1], cr1)
    calibration_review!(fig[2, 2], cr2)

    opts = GridLayout(fig[3, 1:2]; tellwidth = false)
    Label(opts[1, 1], "coverage"; halign = :left)
    cov_menu = Menu(opts[1, 2]; width = 150,
                    options = [("intersection", :intersection), ("union", :union)])
    Label(opts[1, 3], "spacing"; halign = :left)
    spacing_box = Textbox(opts[1, 4]; placeholder = "auto", width = 100)
    build_btn = Button(opts[1, 5]; label = "build dewarpers")
    status = Label(opts[1, 6], ""; halign = :left, justification = :left,
                   word_wrap = true, width = 320)

    spacing = Ref{Any}(:auto)
    on(spacing_box.stored_string) do s
        s === nothing && return
        t = strip(s)
        if isempty(t) || lowercase(t) == "auto"
            spacing[] = :auto
        else
            v = tryparse(Float64, t)
            (v === nothing || v <= 0) ?
                (status.text[] = "spacing must be a positive number or \"auto\"") :
                (spacing[] = v)
        end
    end
    on(build_btn.clicks) do _
        try
            dw1, dw2 = Controllers.build_dewarpers(cr1, cr2;
                                                   spacing = spacing[],
                                                   coverage = cov_menu.selection[])
            batch === nothing || set_dewarpers!(batch, dw1, dw2)
            status.text[] = "built: grid $(join(Base.size(dw1.grid), "×"))" *
                            (batch === nothing ? "" : ", installed in batch")
        catch err
            status.text[] = Controllers._errmsg(err)
        end
    end
    return fig
end
