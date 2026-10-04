# Drives the real stereo_window through calibration (fit, grid, and
# self-calibration), a mask on the dewarped grid, a test pair, a run, and
# Results, and grabs window images with `request_grab` (run by
# test_planar_window.jl in its own process when HAMMERHEADGUI_QT_TESTS=true).
# Exits nonzero on failure. Set HAMMERHEADGUI_SHOTS to keep the images.
using Test
using HammerheadGUI
using HammerheadGUI.Controllers
using HammerheadGUI.GLMakie
using HammerheadGUI.Hammerhead

include(joinpath(@__DIR__, "stereo_fixture.jl"))
fx = stereo_rig(acquisitions = 2)

@testset "stereo_window (Qt)" begin
    wf = StereoWorkflow()
    t_stage = Ref(time())
    t_start = time()
    seen = Symbol[]
    shots = get(ENV, "HAMMERHEADGUI_SHOTS", mktempdir())     # keep the images: set the variable
    mkpath(shots)
    names = ("images", "calibration", "detection", "grid", "selfcal", "prepare_mask",
             "prepare_scale", "test", "results")
    png = Dict(n => joinpath(shots, "stereo_$n.png") for n in names)
    foreach(p -> rm(p; force = true), values(png))
    waited(s) = time() - t_stage[] > s
    # a grabbed image lands asynchronously: wait for the file, then a moment
    grabbed(n) = isfile(png[n]) && filesize(png[n]) > 0 && waited(1)
    grab(n) = HammerheadGUI.request_grab(png[n])
    # Stages: (ready(window shell) -> Bool, act(window shell)); a stage acts on
    # the first tick it is ready, then the next stage waits.
    stages = Any[]
    st!(ready, act) = push!(stages, (ready, act))
    st!(sh -> waited(1) && frame_size(sh.wf.frames1) !== nothing &&
            frame_size(sh.wf.frames2) !== nothing,
        sh -> grab("images"))
    st!(sh -> grabbed("images"), sh -> begin
        # Calibration: plates (in memory), z and detection through the bridge, fit
        cal = sh.wf.calibration
        HammerheadGUI.hh_set_step("calibration")
        for k in 1:2, img in fx.plates[k]
            add_plate!(cal, k, img, 0.0)
        end
        for (i, z) in enumerate(fx.zs), k in 1:2
            HammerheadGUI.hh_set_plate_z(k, i, string(z))
        end
        HammerheadGUI.hh_calibration_option("spacing", "15")
        HammerheadGUI.hh_calibration_option("origin_offset", "30, 7.5")
        HammerheadGUI.hh_calibration_option("grid_spacing", "0.3")
        HammerheadGUI.hh_fit_calibration()
    end)
    st!(sh -> (cal = sh.wf.calibration; cal.dewarpers[] !== nothing && !cal.fitting[] && !cal.building[]),
        sh -> begin
        HammerheadGUI.hh_select_plate(1, 2)
        push!(seen, :fitted)
        occursin("dots · rms", sh.rows["plates1"][2].info) && push!(seen, :plate_info)
    end)
    st!(sh -> waited(0.5), sh -> grab("calibration"))
    st!(sh -> grabbed("calibration"), sh -> (HammerheadGUI.hh_set_calibration_page("detection"); grab("detection")))
    st!(sh -> grabbed("detection"), sh -> (HammerheadGUI.hh_set_calibration_page("grid"); grab("grid")))
    st!(sh -> grabbed("grid"), sh -> begin
        HammerheadGUI.hh_set_calibration_page("selfcal")
        HammerheadGUI.hh_calibration_option("selfcal_pairs", 2)
        HammerheadGUI.hh_start_selfcal()
    end)
    st!(sh -> !sh.wf.calibration.selfcal_running[] && waited(0.5), sh -> begin
        sh.wf.calibration.selfcal[] !== nothing && push!(seen, :selfcal)
        HammerheadGUI.hh_apply_selfcal()
    end)
    st!(sh -> sh.wf.calibration.selfcal_applied[] && waited(0.5), sh -> grab("selfcal"))
    st!(sh -> grabbed("selfcal"), sh -> begin
        # Prepare › Mask on the dewarped grid
        HammerheadGUI.hh_set_step("prepare")
        HammerheadGUI.hh_set_prepare_page("mask")
    end)
    st!(sh -> (w = sh.wf; w.dewarped[] !== nothing && w.prepare.mask[] !== nothing &&
                      w.prepare.mask[].size == grid_size(w)),
        sh -> begin
        for (x, y) in ((20.0, 10.0), (90.0, 10.0), (90.0, 70.0))
            canvas_click!(sh.wf, x, y)
        end
        canvas_alt_click!(sh.wf)
    end)
    st!(sh -> waited(1), sh -> grab("prepare_mask"))
    st!(sh -> grabbed("prepare_mask"), sh -> begin
        HammerheadGUI.hh_set_prepare_page("scale")
        HammerheadGUI.hh_set_scale("dt", "0.001")
        HammerheadGUI.hh_set_scale("time_unit", "s")
    end)
    st!(sh -> waited(0.5), sh -> grab("prepare_scale"))
    st!(sh -> grabbed("prepare_scale"), sh -> begin
        fill_preset!(sh.wf.passes, :low)
        HammerheadGUI.hh_set_step("test")
        HammerheadGUI.hh_test()
    end)
    st!(sh -> !sh.wf.test.running[] && sh.wf.test.result[] !== nothing && waited(1), sh -> grab("test"))
    st!(sh -> grabbed("test"), sh -> (HammerheadGUI.hh_set_step("run"); HammerheadGUI.hh_start_run()))
    st!(sh -> !sh.wf.run.running[] && sh.wf.explorer[] !== nothing, sh -> begin
        HammerheadGUI.hh_set_step("results")
        HammerheadGUI.hh_result_field("w")
    end)
    st!(sh -> waited(1), sh -> begin
        sh.host.content === sh.results.fig && push!(seen, :results_canvas)
        grab("results")
    end)
    st!(sh -> grabbed("results"), sh -> HammerheadGUI.hh_set_camera(2))
    st!(sh -> waited(0.5), sh -> HammerheadGUI.request_close())
    stage = Ref(1)
    HammerheadGUI._TICK_HOOK[] = function (sh)
        time() - t_start > 420 && (@error "stereo window script timed out at stage $(stage[])";
                                   return HammerheadGUI.request_close())
        stage[] > length(stages) && return
        ready, act = stages[stage[]]
        ready(sh) || return
        act(sh)
        stage[] += 1
        t_stage[] = time()
    end
    try
        stereo_window(wf; files1 = fx.files1, files2 = fx.files2)
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test stage[] == length(stages) + 1
    @test seen == [:fitted, :plate_info, :selfcal, :results_canvas]
    @test !wf.spawn[]                                  # restored when the window closed
    cal = wf.calibration
    @test [p.z for p in cal.plates[1][]] == fx.zs
    @test cal.selfcal_applied[] && cal.dewarpers[][1] === cal.selfcal[].dewarpers[1]
    gsz = grid_size(wf)
    @test wf.mask[] == polygon_mask(gsz, [(20, 10), (90, 10), (90, 70)])
    @test wf.scale[] == PhysicalScale(1.0, 0.001, "mm", "s")
    @test wf.test.result[] isa StereoPIVResult
    @test length(wf.run.completed[]) == 2 && nframes(wf.explorer[]) == 2
    @test wf.explorer[].field[] === :w && wf.camera[] == 2
    # the grabbed window images exist and show something (not one colour)
    for n in names
        @test isfile(png[n])
        img = HammerheadGUI.Controllers.FileIO.load(png[n])
        @test minimum(size(img)) > 100 && length(unique(img)) > 50
    end
    @info "stereo window images in $shots"
    @test isempty(filter(s -> s isa GLMakie.Screen{HammerheadGUI.QMLMakie.QMLWindow},
                         GLMakie.ALL_SCREENS))

    # dewarpers built in a script; the window opens again in the same session
    wf2 = StereoWorkflow()
    HammerheadGUI._TICK_HOOK[] = sh -> HammerheadGUI.request_close()
    try
        @test stereo_window(wf2; dewarpers = cal.dewarpers[], files1 = fx.files1[1:2],
                            files2 = fx.files2[1:2]) === wf2
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test wf2.calibration.dewarpers[] === cal.dewarpers[] && length(wf2.frames1.files[]) == 2
end
