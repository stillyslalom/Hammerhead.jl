# Window screenshots for the GUI how-tos and tour (docs/src/assets/gui_window/).
#
# Local only: needs a display and Qt, so neither docs/make.jl nor CI runs it.
# Run from the repository root after changing a window's look:
#
#     julia --project=docs -t 4 docs/gui_screenshots.jl            # both windows
#     julia --project=docs -t 4 docs/gui_screenshots.jl stereo     # one window
#
# It opens `planar_window` on the GUI tour's synthetic vortex (with its
# stationary reflection) and `stereo_window` on the synthetic two-camera rig
# of HammerheadGUI/test/stereo_fixture.jl (plates and frames written as image
# files), walks the steps through the controllers on the window's tick, and
# saves the window body with `request_grab`. Look at every image afterwards:
# it should show what the docs say it shows.
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
const WINDOWS = isempty(ARGS) ? ["planar", "stereo"] : ARGS
const GRABBED = String[]

shot(name) = joinpath(OUT, name * ".png")

# Drive an open window through `stages`, a list of `(ready, act)` pairs: on
# each tick, the current stage acts once `ready(shell)` holds, then the next
# stage waits. `grab!(name)` asks for a window image; `grabbed(name)` holds
# once it has landed (and a moment has passed).
mutable struct Walk
    stages::Vector{Any}
    stage::Int
    t_stage::Float64
    t_start::Float64
end
Walk() = Walk(Any[], 1, time(), time())
st!(w::Walk, ready, act) = push!(w.stages, (ready, act))
waited(w::Walk, s) = time() - w.t_stage > s
grabbed(w::Walk, name) = isfile(shot(name)) && filesize(shot(name)) > 0 && waited(w, 1.5)
function grab!(name)
    rm(shot(name); force = true)
    push!(GRABBED, name)
    HammerheadGUI.request_grab(shot(name))
end

function walk!(open_window, w::Walk; timeout = 420)
    HammerheadGUI._TICK_HOOK[] = function (sh)
        time() - w.t_start > timeout &&
            (@error "screenshot walk timed out at stage $(w.stage)"; return HammerheadGUI.request_close())
        w.stage > length(w.stages) && return
        ready, act = w.stages[w.stage]
        ready(sh) || return
        act(sh)
        w.stage += 1
        w.t_stage = time()
    end
    try
        open_window()
    finally
        HammerheadGUI._TICK_HOOK[] = nothing
    end
    w.stage == length(w.stages) + 1 || error("the walk-through stopped at stage $(w.stage)")
    return
end

# ---------------------------------------------------------------- planar window

function planar_shots()
    # The tour's scene: a solid-body vortex and a bright square that does not
    # move. Two pairs, so the pair bar and the run have something to step through.
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

    wf = PlanarWorkflow()
    wf.run.output_path[] = joinpath(work, "vectors.jld2")
    w = Walk()
    ps, pp = wf.prepare, wf.prepare.preview
    st!(w, sh -> frame_size(wf.frames) !== nothing && waited(w, 2), sh -> grab!("images"))
    st!(w, sh -> grabbed(w, "images"), sh -> begin
        # Prepare › Preprocess: a highpass filter, the processed view, a probe
        set_step!(wf, :prepare)
        add_step!(pp, :highpass_filter)
        ps.show_processed[] = true
        canvas_click!(wf, 190.0, 150.0)
    end)
    st!(w, sh -> pp.probe_result[] !== nothing && pp.processed[] !== pp.image[] && waited(w, 1.5),
        sh -> grab!("prepare_preprocess"))
    st!(w, sh -> grabbed(w, "prepare_preprocess"), sh -> begin
        # Prepare › Mask: a polygon around the reflection
        set_prepare_page!(wf, :mask)
        for (x, y) in ((8.0, 8.0), (64.0, 8.0), (64.0, 64.0), (8.0, 64.0))
            canvas_click!(wf, x, y)
        end
        canvas_alt_click!(wf)
    end)
    st!(w, sh -> waited(w, 1), sh -> grab!("prepare_mask"))
    st!(w, sh -> grabbed(w, "prepare_mask"), sh -> begin
        # Prepare › Scale: typed pixel size and frame interval
        set_prepare_page!(wf, :scale)
        edit_scale!(wf, :pixel_size, "0.02")
        edit_scale!(wf, :length_unit, "mm")
        edit_scale!(wf, :dt, "0.001")
        edit_scale!(wf, :time_unit, "s")
    end)
    st!(w, sh -> waited(w, 1), sh -> grab!("prepare_scale"))
    st!(w, sh -> grabbed(w, "prepare_scale"), sh -> begin
        set_step!(wf, :passes)
        fill_preset!(wf.passes, :medium)
    end)
    st!(w, sh -> waited(w, 1), sh -> grab!("passes"))
    st!(w, sh -> grabbed(w, "passes"), sh -> (set_step!(wf, :test); test_pair!(wf)))
    st!(w, sh -> !wf.test.running[] && wf.test.result[] !== nothing && waited(w, 1),
        sh -> grab!("test_pair"))
    st!(w, sh -> grabbed(w, "test_pair"), sh -> (set_step!(wf, :run); start_run!(wf)))
    st!(w, sh -> !wf.run.running[] && wf.explorer[] !== nothing && waited(w, 1), sh -> begin
        # Results › Profile across the vortex centre
        set_step!(wf, :results)
        ex = wf.explorer[]
        set_tool!(ex, :profile)
        r = current_result(ex)
        ym = (first(r.y) + last(r.y)) / 2
        click!(ex, first(r.x) + 0.1 * (last(r.x) - first(r.x)), ym)
        click!(ex, last(r.x) - 0.1 * (last(r.x) - first(r.x)), ym)
    end)
    st!(w, sh -> wf.explorer[].profile_data[] !== nothing && waited(w, 1.5),
        sh -> grab!("results_profile"))
    st!(w, sh -> grabbed(w, "results_profile"), sh -> HammerheadGUI.request_close())
    walk!(() -> planar_window(wf; files = paths), w; timeout = 300)
end

# ---------------------------------------------------------------- stereo window

# stereo_rig(): two pinhole cameras at ±20° yaw, dot-plate images at
# z = -3/0/3 mm, and particle pairs on a light sheet at z = 0.8 mm (the
# offset self-calibration finds), displaced by (1.0, 0.5) mm.
include(joinpath(pkgdir(HammerheadGUI), "test", "stereo_fixture.jl"))

function stereo_shots()
    fx = stereo_rig(acquisitions = 2)
    work = mktempdir()
    save_gray(path, img) = (Hammerhead.FileIO.save(path, Hammerhead.Gray.(clamp.(img, 0, 1))); path)
    plates = map(1:2) do k
        [save_gray(joinpath(work, "cam$(k)_z$(Int(z)).png"), img)
         for (img, z) in zip(fx.plates[k], fx.zs)]
    end
    frames = map(1:2) do k
        src = k == 1 ? fx.files1 : fx.files2
        peak = maximum(maximum, src)                  # one scale per camera
        [save_gray(joinpath(work, "cam$(k)_$(lpad(i, 4, '0')).png"), img ./ peak)
         for (i, img) in enumerate(src)]
    end

    wf = StereoWorkflow()
    wf.run.output_path[] = joinpath(work, "stereo_vectors.jld2")
    cal = wf.calibration
    w = Walk()
    st!(w, sh -> frame_size(wf.frames1) !== nothing && frame_size(wf.frames2) !== nothing &&
                 waited(w, 2),
        sh -> grab!("stereo_images"))
    st!(w, sh -> grabbed(w, "stereo_images"), sh -> begin
        # Calibration › Plates: three plate images per camera with their z,
        # the dot spacing and origin offset, then Fit cameras
        set_step!(wf, :calibration)
        for k in 1:2, (path, z) in zip(plates[k], fx.zs)
            add_plate!(cal, k, path, z)
        end
        set_calibration_option!(cal, :spacing, "15")
        set_calibration_option!(cal, :origin_offset, "30, 7.5")
        set_calibration_option!(cal, :grid_spacing, "0.3")
        fit_calibration!(cal)
    end)
    st!(w, sh -> cal.dewarpers[] !== nothing && !cal.fitting[] && !cal.building[],
        sh -> HammerheadGUI.hh_select_plate(1, 2))
    st!(w, sh -> waited(w, 1.5), sh -> grab!("stereo_calibration_plates"))
    st!(w, sh -> grabbed(w, "stereo_calibration_plates"), sh -> begin
        HammerheadGUI.hh_set_calibration_page("detection")
    end)
    st!(w, sh -> waited(w, 1), sh -> grab!("stereo_calibration_detection"))
    st!(w, sh -> grabbed(w, "stereo_calibration_detection"), sh -> begin
        # Calibration › Self-calibration on both pairs, then apply it
        HammerheadGUI.hh_set_calibration_page("selfcal")
        set_calibration_option!(cal, :selfcal_pairs, 2)
        start_selfcal!(wf)
    end)
    st!(w, sh -> !cal.selfcal_running[] && cal.selfcal[] !== nothing && waited(w, 0.5),
        sh -> apply_selfcal!(wf))
    st!(w, sh -> cal.selfcal_applied[] && waited(w, 1), sh -> grab!("stereo_calibration_selfcal"))
    st!(w, sh -> grabbed(w, "stereo_calibration_selfcal"), sh -> begin
        # Prepare › Mask on the dewarped grid
        set_step!(wf, :prepare)
        set_prepare_page!(wf, :mask)
    end)
    st!(w, sh -> wf.dewarped[] !== nothing && wf.prepare.mask[] !== nothing &&
                 wf.prepare.mask[].size == grid_size(wf),
        sh -> begin
        rows, cols = grid_size(wf)
        for (x, y) in ((0.62cols, 0.08rows), (0.92cols, 0.08rows), (0.92cols, 0.38rows))
            canvas_click!(wf, x, y)
        end
        canvas_alt_click!(wf)
    end)
    st!(w, sh -> waited(w, 1.5), sh -> grab!("stereo_prepare_mask"))
    st!(w, sh -> grabbed(w, "stereo_prepare_mask"), sh -> begin
        set_prepare_page!(wf, :scale)
        edit_scale!(wf, :dt, "0.001")
        edit_scale!(wf, :time_unit, "s")
        fill_preset!(wf.passes, :medium)
        set_step!(wf, :test)
        test_pair!(wf)
    end)
    st!(w, sh -> !wf.test.running[] && wf.test.result[] !== nothing && waited(w, 1.5),
        sh -> grab!("stereo_test_pair"))
    st!(w, sh -> grabbed(w, "stereo_test_pair"), sh -> (set_step!(wf, :run); start_run!(wf)))
    st!(w, sh -> !wf.run.running[] && wf.explorer[] !== nothing && waited(w, 0.5), sh -> begin
        set_step!(wf, :results)
        set_field!(wf.explorer[], :w)
    end)
    st!(w, sh -> waited(w, 1.5), sh -> grab!("stereo_results"))
    st!(w, sh -> grabbed(w, "stereo_results"), sh -> HammerheadGUI.request_close())
    walk!(() -> stereo_window(wf; files1 = frames[1], files2 = frames[2]), w)
end

"planar" in WINDOWS && planar_shots()
"stereo" in WINDOWS && stereo_shots()

# Qt writes RGBA with light compression; RGB at zlib's best compression is
# lossless and keeps each image well under 300 KiB.
for name in GRABBED
    p = shot(name)
    img = Hammerhead.FileIO.load(p)
    Hammerhead.FileIO.save(p, parentmodule(eltype(img)).RGB.(img);
                           compression_level = 9, compression_strategy = 0)
    println(rpad(name * ".png", 36), round(filesize(p) / 1024; digits = 1), " KiB")
end
