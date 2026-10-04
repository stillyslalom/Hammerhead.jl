# Drives the real planar_window through every step (run by test_planar_window.jl
# in its own process when HAMMERHEADGUI_QT_TESTS=true). Exits nonzero on failure.
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
    t_stage = Ref(time())
    t_start = time()
    areas = Symbol[]
    next!() = (stage[] += 1; t_stage[] = time())
    HammerheadGUI._TICK_HOOK[] = function (sh)
        w = sh.wf
        time() - t_start > 180 && return HammerheadGUI.request_close()
        s = stage[]
        if s == 0 && time() - t_stage[] > 1
            set_step!(w, :test); test_pair!(w); next!()
        elseif s == 1 && !w.test.running[]
            set_step!(w, :run); start_run!(w); next!()
        elseif s == 2 && !w.run.running[]
            set_step!(w, :results); next!()
        elseif s == 3 && time() - t_stage[] > 1
            HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 4 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.toggle_popout!(sh.host); next!()
        elseif s == 5 && sh.host.pending === nothing && time() - t_stage[] > 1
            push!(areas, sh.host.area); HammerheadGUI.request_close(); next!()
        end
    end
    try
        planar_window(wf)
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    @test stage[] == 6
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
