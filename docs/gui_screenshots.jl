# Window screenshots for the GUI how-to and tour (docs/src/assets/gui_window/).
#
# Local only: needs a display and Qt, so neither docs/make.jl nor CI runs it.
# Run from the repository root after changing the window's look:
#
#     julia --project=docs -t 4 docs/gui_screenshots.jl
#
# It opens `planar_window` on the GUI tour's synthetic vortex (with its
# stationary reflection), walks the steps through the controllers on the
# window's tick, and saves the window body with `request_grab`. Look at every
# image afterwards: it should show what the docs say it shows.
#
# Keep GLMakie windows (GLFW) out of this process: a GLFW context and the Qt
# canvases must not share a process.

using Hammerhead
using Hammerhead.SyntheticData
using HammerheadGUI
using HammerheadGUI.Controllers
using HammerheadGUI.Controllers: click!
using Random

const OUT = joinpath(@__DIR__, "src", "assets", "gui_window")
mkpath(OUT)

# The tour's scene: a solid-body vortex and a bright square that does not move.
# Two pairs, so the pair bar and the run have something to step through.
flow(x, y, z, t) = (-0.015 * (y - 128), 0.015 * (x - 128), 0.0)
work = mktempdir()
paths = String[]
for (k, seed) in enumerate((42, 43))
    a, b, _, _ = generate_synthetic_piv_pair(flow, (256, 256), 1.0;
        particle_density = 0.05, background_noise = 0.01, rng = MersenneTwister(seed))
    peak = max(maximum(a), maximum(b))
    for (j, img) in enumerate((a, b))
        img = img ./ peak
        img[12:60, 12:60] .= 1.0
        path = joinpath(work, "frame_$(lpad(2k - 2 + j, 4, '0')).png")
        Hammerhead.FileIO.save(path, Hammerhead.Gray.(img))
        push!(paths, path)
    end
end

shot(name) = joinpath(OUT, name * ".png")
stage = Ref(0)
t_stage = Ref(time())
t_start = time()
waited(s) = time() - t_stage[] > s
grabbed(name) = isfile(shot(name)) && filesize(shot(name)) > 0 && waited(1.5)
function next!()
    stage[] += 1
    t_stage[] = time()
end
function grab!(name)
    rm(shot(name); force = true)
    HammerheadGUI.request_grab(shot(name))
    next!()
end

wf = PlanarWorkflow()
wf.run.output_path[] = joinpath(work, "vectors.jld2")

HammerheadGUI._TICK_HOOK[] = function (sh)
    w = sh.wf
    ps, pp = w.prepare, w.prepare.preview
    time() - t_start > 300 && (@error "screenshot script timed out at stage $(stage[])";
                               return HammerheadGUI.request_close())
    s = stage[]
    if s == 0 && frame_size(w.frames) !== nothing && waited(2)
        grab!("images")
    elseif s == 1 && grabbed("images")
        # Prepare › Preprocess: a highpass filter, the processed view, a probe
        set_step!(w, :prepare)
        add_step!(pp, :highpass_filter)
        ps.show_processed[] = true
        canvas_click!(w, 190.0, 150.0)
        next!()
    elseif s == 2 && pp.probe_result[] !== nothing && pp.processed[] !== pp.image[] && waited(1.5)
        grab!("prepare_preprocess")
    elseif s == 3 && grabbed("prepare_preprocess")
        # Prepare › Mask: a polygon around the reflection
        set_prepare_page!(w, :mask)
        for (x, y) in ((8.0, 8.0), (64.0, 8.0), (64.0, 64.0), (8.0, 64.0))
            canvas_click!(w, x, y)
        end
        canvas_alt_click!(w)
        next!()
    elseif s == 4 && waited(1)
        grab!("prepare_mask")
    elseif s == 5 && grabbed("prepare_mask")
        # Prepare › Scale: typed pixel size and frame interval
        set_prepare_page!(w, :scale)
        edit_scale!(w, :pixel_size, "0.02")
        edit_scale!(w, :length_unit, "mm")
        edit_scale!(w, :dt, "0.001")
        edit_scale!(w, :time_unit, "s")
        next!()
    elseif s == 6 && waited(1)
        grab!("prepare_scale")
    elseif s == 7 && grabbed("prepare_scale")
        set_step!(w, :passes)
        fill_preset!(w.passes, :medium)
        next!()
    elseif s == 8 && waited(1)
        grab!("passes")
    elseif s == 9 && grabbed("passes")
        set_step!(w, :test)
        test_pair!(w)
        next!()
    elseif s == 10 && !w.test.running[] && w.test.result[] !== nothing && waited(1)
        grab!("test_pair")
    elseif s == 11 && grabbed("test_pair")
        set_step!(w, :run)
        start_run!(w)
        next!()
    elseif s == 12 && !w.run.running[] && w.explorer[] !== nothing && waited(1)
        # Results › Profile across the vortex centre
        set_step!(w, :results)
        ex = w.explorer[]
        set_tool!(ex, :profile)
        r = current_result(ex)
        ym = (first(r.y) + last(r.y)) / 2
        click!(ex, first(r.x) + 0.1 * (last(r.x) - first(r.x)), ym)
        click!(ex, last(r.x) - 0.1 * (last(r.x) - first(r.x)), ym)
        next!()
    elseif s == 13 && w.explorer[].profile_data[] !== nothing && waited(1.5)
        grab!("results_profile")
    elseif s == 14 && grabbed("results_profile")
        HammerheadGUI.request_close()
        next!()
    end
end

try
    planar_window(wf; files = paths)
finally
    HammerheadGUI._TICK_HOOK[] = nothing
end

stage[] == 15 || error("the walk-through stopped at stage $(stage[])")
# Qt writes RGBA with light compression; RGB at zlib's best compression is
# lossless and keeps each image well under 300 KiB.
for f in sort(readdir(OUT))
    p = joinpath(OUT, f)
    img = Hammerhead.FileIO.load(p)
    Hammerhead.FileIO.save(p, parentmodule(eltype(img)).RGB.(img);
                           compression_level = 9, compression_strategy = 0)
    println(rpad(f, 28), round(filesize(p) / 1024; digits = 1), " KiB")
end
