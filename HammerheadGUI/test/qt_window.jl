# Drives the real planar_window through every step, including edits on the
# Prepare sub-pages (run by test_planar_window.jl in its own process when
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
            set_prepare_page!(w, :roi)
            edit_roi!(w, "9", "120", "5", "124")
            set_prepare_page!(w, :scale)
            canvas_click!(w, 10.0, 20.0); canvas_click!(w, 10.0, 70.0)
            HammerheadGUI.hh_set_scale("separation", "25")
            HammerheadGUI.hh_set_scale("dt", "0.001")
            set_prepare_page!(w, :preprocess)
            HammerheadGUI.hh_estimate_background(2)
            next!()
        elseif s == 3 && !ps.background_running[] && time() - t_stage[] > 0.5
            set_step!(w, :test); test_pair!(w); next!()
        elseif s == 4 && !w.test.running[]
            set_step!(w, :run); start_run!(w); next!()
        elseif s == 5 && !w.run.running[]
            set_step!(w, :results); next!()
        elseif s == 6 && time() - t_stage[] > 1
            HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 7 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 8 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.request_close(); next!()
        end
    end
    try
        planar_window(wf)
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test stage[] == 9
    @test seen == [:preview]
    @test !wf.spawn[]                                  # restored when the window closed
    @test [s.operation for s in wf.preprocessing[]] == [:subtract_background, :highpass_filter, :invert_image]
    @test wf.preprocessing[][2] == PreprocessStep(:highpass_filter; sigma = 2.5)
    @test wf.mask[] == polygon_mask((128, 128), [(20, 10), (60, 10), (60, 50)])
    @test wf.roi[] == ROI(9:120, 5:124)
    @test wf.scale[] == PhysicalScale(0.5, 0.001, "mm", "frame")
    @test wf.test.result[] isa PIVResult
    @test length(wf.run.completed[]) == 2 && nframes(wf.explorer[]) == 2
    @test areas == [:pop, :main]
    @test isempty(filter(s -> s isa GLMakie.Screen{HammerheadGUI.QMLMakie.QMLWindow},
                         GLMakie.ALL_SCREENS))
    # the window can be opened again in the same session
    HammerheadGUI._TICK_HOOK[] = sh -> HammerheadGUI.request_close()
    try
        @test planar_window(wf) === wf
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
end
