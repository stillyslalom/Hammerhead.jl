# Drives the real planar_window through every step, including edits on the
# Prepare sub-pages and the Results tools, and grabs window images with
# `request_grab` (run by test_planar_window.jl in its own process when
# HAMMERHEADGUI_QT_TESTS=true). Exits nonzero on failure.
using Test
using HammerheadGUI
using HammerheadGUI.Controllers
using HammerheadGUI.GLMakie
using HammerheadGUI.Hammerhead
using HammerheadGUI.Hammerhead.SyntheticData: generate_synthetic_piv_pair, linear_flow

imgA, imgB, _, _ = generate_synthetic_piv_pair(linear_flow(3.0, 2.0, 0.0, 0, 0, 0, 0),
                                               (128, 128), 1.0; z_range = (-1.0, 1.0))

@testset "planar_window (Qt)" begin
    wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
    fill_preset!(wf.passes, :low)
    stage = Ref(0)
    seen = Symbol[]
    t_stage = Ref(time())
    t_start = time()
    areas = Symbol[]
    circulation = Ref{Any}(nothing)
    shots = get(ENV, "HAMMERHEADGUI_SHOTS", mktempdir())     # keep the images: set the variable
    mkpath(shots)
    mask_png, scale_png, profile_png, circulation_png =
        (joinpath(shots, f) for f in ("prepare_mask.png", "prepare_scale.png",
                                      "results_profile.png", "results_circulation.png"))
    # a grabbed image lands asynchronously: wait for the file, then a moment
    grabbed(path) = isfile(path) && filesize(path) > 0 && time() - t_stage[] > 1
    next!() = (stage[] += 1; t_stage[] = time())
    HammerheadGUI._TICK_HOOK[] = function (sh)
        w = sh.wf
        ps = w.prepare
        time() - t_start > 240 && return HammerheadGUI.request_close()
        s = stage[]
        if s == 0 && time() - t_stage[] > 1 && frame_size(w.frames) !== nothing
            # Prepare › Preprocess: a step (previewed on a worker), the
            # processed view, and the probe
            set_step!(w, :prepare)
            add_step!(ps.preview, :highpass_filter)
            HammerheadGUI.hh_add_step("invert_image")             # through the bridge
            HammerheadGUI.hh_set_step_option(1, "sigma", "2.5")
            HammerheadGUI.hh_show_processed(true)
            canvas_click!(w, 64.0, 64.0)
            next!()
        elseif s == 1 && ps.preview.probe_result[] !== nothing &&
               ps.preview.processed[] !== ps.preview.image[]
            push!(seen, :preview)
            HammerheadGUI.hh_set_prepare_page("mask")
            for (x, y) in ((20.0, 10.0), (60.0, 10.0), (60.0, 50.0))
                canvas_click!(w, x, y)
            end
            canvas_alt_click!(w)
            next!()
        elseif s == 2 && time() - t_stage[] > 0.5
            HammerheadGUI.request_grab(mask_png); next!()
        elseif s == 3 && grabbed(mask_png)
            set_prepare_page!(w, :roi)
            edit_roi!(w, "9", "120", "5", "124")
            set_prepare_page!(w, :scale)
            canvas_click!(w, 10.0, 20.0); canvas_click!(w, 10.0, 70.0)
            HammerheadGUI.hh_set_scale("separation", "25")
            HammerheadGUI.hh_set_scale("dt", "0.001")
            HammerheadGUI.request_grab(scale_png); next!()
        elseif s == 4 && grabbed(scale_png)
            set_prepare_page!(w, :preprocess)
            HammerheadGUI.hh_estimate_background(2)
            next!()
        elseif s == 5 && !ps.background_running[] && time() - t_stage[] > 0.5
            set_step!(w, :passes); test_pair!(w); next!()
        elseif s == 6 && !w.test.running[]
            set_step!(w, :run); start_run!(w); next!()
        elseif s == 7 && !w.run.running[]
            set_step!(w, :results); next!()
        elseif s == 8 && time() - t_stage[] > 1
            # Results › Profile through the bridge, clicks as the canvas sends them
            ex = w.explorer[]
            HammerheadGUI.hh_result_tool("profile")
            r = current_result(ex)
            ym = (first(r.y) + last(r.y)) / 2
            Controllers.click!(ex, first(r.x) + 5, ym); Controllers.click!(ex, last(r.x) - 5, ym + 10)
            next!()
        elseif s == 9 && w.explorer[].profile_data[] !== nothing && time() - t_stage[] > 1
            occursin("profile of", sh.shown["toolSummary"]) && push!(seen, :profile)  # bridged
            HammerheadGUI.request_grab(profile_png); next!()
        elseif s == 10 && grabbed(profile_png)
            ex = w.explorer[]
            HammerheadGUI.hh_result_tool("circulation")
            r = current_result(ex)
            # lower-right part of the field, clear of the masked corner
            x0, x1 = (first(r.x) + last(r.x)) / 2, last(r.x) - 5
            y0, y1 = (first(r.y) + last(r.y)) / 2, last(r.y) - 5
            for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                Controllers.click!(ex, x, y)
            end
            Controllers.alt_click!(ex)
            circulation[] = ex.circulation_result[]
            push!(seen, ex.tool[])
            HammerheadGUI.request_grab(circulation_png); next!()
        elseif s == 11 && grabbed(circulation_png)
            HammerheadGUI.hh_result_clear_tool()
            next!()
        elseif s == 12 && time() - t_stage[] > 0.5
            HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 13 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 14 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.request_close(); next!()
        end
    end
    try
        planar_window(wf)
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test stage[] == 15
    @test seen == [:preview, :profile, :circulation]
    @test !wf.spawn[]                                  # restored when the window closed
    @test [s.operation for s in wf.preprocessing[]] == [:subtract_background, :highpass_filter, :invert_image]
    @test wf.preprocessing[][2] == PreprocessStep(:highpass_filter; sigma = 2.5)
    @test wf.mask[] == polygon_mask((128, 128), [(20, 10), (60, 10), (60, 50)])
    @test wf.roi[] == ROI(9:120, 5:124)
    @test wf.scale[] == PhysicalScale(0.5, 0.001, "mm", "frame")
    @test wf.test.result[] isa PIVResult
    @test length(wf.run.completed[]) == 2 && nframes(wf.explorer[]) == 2
    @test areas == [:pop, :main]
    @test circulation[] !== nothing && isfinite(circulation[].line)
    @test isempty(wf.explorer[].tool_points[]) && wf.explorer[].tool[] === :circulation
    # the grabbed window images exist and show something (not one color)
    for path in (mask_png, scale_png, profile_png, circulation_png)
        @test isfile(path)
        img = HammerheadGUI.Controllers.FileIO.load(path)
        @test minimum(size(img)) > 100 && length(unique(img)) > 50
    end
    @info "window images" mask_png scale_png profile_png circulation_png
    @test isempty(filter(s -> s isa GLMakie.Screen{HammerheadGUI.QMLMakie.QMLWindow},
                         GLMakie.ALL_SCREENS))
    # the window can be opened again in the same session; this time it runs
    # an ensemble through the bridge
    wf.roi[] = nothing                                 # ensembles take no ROI
    ens_stage = Ref(0)
    ens_text = String[]
    t_start = time()
    HammerheadGUI._TICK_HOOK[] = function (sh)
        w = sh.wf
        time() - t_start > 120 && return HammerheadGUI.request_close()
        if ens_stage[] == 0
            HammerheadGUI.hh_set_mode("ensemble")
            HammerheadGUI.hh_set_output("")
            set_step!(w, :run)
            HammerheadGUI.hh_start_run()
            ens_stage[] = 1
        elseif ens_stage[] == 1 && !w.run.running[]
            push!(ens_text, sh.shown["runStatus"], sh.shown["resultsLabel"])
            ens_stage[] = 2
            HammerheadGUI.request_close()
        end
    end
    try
        @test planar_window(wf) === wf
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test ens_stage[] == 2
    @test ens_text == ["done: ensemble of 2 pairs", "1 result in memory"]
    @test wf.passes.mode[] === :ensemble && only(wf.run.completed[]) isa PIVResult

    # a third session in PTV mode: the Particles page with the detection
    # preview, then a test pair, through the bridge
    particles_png, ptv_test_png = joinpath(shots, "particles.png"), joinpath(shots, "ptv_test.png")
    # inverted frames have dark particles: drop the invert step
    wf.preprocessing[] = filter(s -> s.operation !== :invert_image, wf.preprocessing[])
    ptv_stage = Ref(0)
    ptv_text = String[]
    t_start = time()
    HammerheadGUI._TICK_HOOK[] = function (sh)
        w = sh.wf
        time() - t_start > 120 && return HammerheadGUI.request_close()
        s = ptv_stage[]
        if s == 0
            HammerheadGUI.hh_set_mode("ptv")
            HammerheadGUI.hh_particle_option("search_radius", "4")
            set_step!(w, :passes)
            ptv_stage[] = 1; t_stage[] = time()
        elseif s == 1 && w.particles.detected[] !== nothing && time() - t_stage[] > 1
            push!(ptv_text, sh.step_rows[3].label, sh.shown["ptvSearchRadius"])
            HammerheadGUI.request_grab(particles_png); ptv_stage[] = 2; t_stage[] = time()
        elseif s == 2 && grabbed(particles_png)
            HammerheadGUI.hh_test(); ptv_stage[] = 3; t_stage[] = time()
        elseif s == 3 && !w.test.running[] && w.test.result[] !== nothing && time() - t_stage[] > 1
            push!(ptv_text, first(split(sh.shown["testLines"], '\n')))
            HammerheadGUI.request_grab(ptv_test_png); ptv_stage[] = 4; t_stage[] = time()
        elseif s == 4 && grabbed(ptv_test_png)
            ptv_stage[] = 5
            HammerheadGUI.request_close()
        end
    end
    try
        @test planar_window(wf) === wf
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test ptv_stage[] == 5
    @test ptv_text[1:2] == ["Particles", "4"] && startswith(ptv_text[3], "Particles:")
    @test wf.test.result[] isa PTVResult && length(wf.test.result[].u) > 50
    for path in (particles_png, ptv_test_png)
        img = HammerheadGUI.Controllers.FileIO.load(path)
        @test minimum(size(img)) > 100 && length(unique(img)) > 50
    end

    # a fourth session changes the recording type: the session has frames,
    # so the window asks; declined, then confirmed; a fresh stereo session
    # switches back without asking
    stereo_png = joinpath(shots, "switched_stereo.png")
    sw_stage = Ref(0)
    sw = Any[]
    t_start = time()
    HammerheadGUI._TICK_HOOK[] = function (sh)
        time() - t_start > 120 && return HammerheadGUI.request_close()
        s = sw_stage[]
        if s == 0
            set_step!(sh.wf, :images)
            HammerheadGUI.hh_set_modality("stereo")
            push!(sw, sh.shown["switchQuestion"])
            HammerheadGUI.hh_cancel_switch()
            push!(sw, sh.wf)
            HammerheadGUI.hh_set_modality("stereo")
            HammerheadGUI.hh_confirm_switch()
            sw_stage[] = 1; t_stage[] = time()
        elseif s == 1 && time() - t_stage[] > 1
            push!(sw, sh.wf, sh.shown["modality"], [r.key for r in sh.step_rows], sh.shown["title"])
            HammerheadGUI.request_grab(stereo_png); sw_stage[] = 2
        elseif s == 2 && grabbed(stereo_png)
            HammerheadGUI.hh_set_modality("planar")    # nothing to lose: no question
            push!(sw, sh.shown["switchQuestion"])
            sw_stage[] = 3; t_stage[] = time()
        elseif s == 3 && time() - t_stage[] > 0.5
            push!(sw, length(HammerheadGUI._SHELL[].step_rows))
            sw_stage[] = 4
            HammerheadGUI.request_close()
        end
    end
    local final
    try
        final = hammerhead(wf)
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test sw_stage[] == 4
    @test startswith(sw[1], "Start a new two-camera (stereo) session? This discards 4 frames")
    @test sw[2] === wf                                       # declined: same session
    @test sw[3] isa StereoWorkflow && sw[4] == "stereo"
    @test sw[5] == ["images", "calibration", "prepare", "passes", "run", "results"]
    @test startswith(sw[6], "Hammerhead stereo PIV |")
    @test sw[7] == "" && sw[8] == 5
    @test final isa PlanarWorkflow && final !== wf && isempty(final.frames.files[])
    @test !final.spawn[] && !wf.spawn[]
    img = HammerheadGUI.Controllers.FileIO.load(stereo_png)
    @test minimum(size(img)) > 100 && length(unique(img)) > 20
end
