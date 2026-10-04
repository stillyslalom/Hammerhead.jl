# The stereo PIV workflow window's state. It shares the settings, Passes,
# Test pair, Run, and Results machinery with the planar workflow
# (workflow.jl) and adds two synchronized camera frame sets, the Calibration
# step (stereo_calibration.jl), and a Prepare step on the dewarped grid:
# preprocessing applies to the raw frames (as `run_piv_stereo` does), and
# the preview, probe, and mask live on the dewarped grid.

"""
Steps of the stereo workflow window, in order.
"""
const STEREO_WORKFLOW_STEPS = (:images, :calibration, :prepare, :passes, :test, :run, :results)

"""
Sub-pages of the stereo Prepare step, in order (no ROI: mask the dewarped
grid instead).
"""
const STEREO_PREPARE_PAGES = (:preprocess, :mask, :scale)

"""
    StereoWorkflow(; files1 = Any[], files2 = Any[], pair_mode = :paired,
                   dewarpers = nothing, deliver = f -> f())

State of the stereo PIV workflow window (an [`AbstractWorkflow`](@ref)).
`frames1` and `frames2` are the cameras' `FrameSet`s; they share the
pairing rule, the representative pair index, and the shown frame (A/B),
which stay in sync whichever set is edited. `camera` (1 or 2) selects the
camera the viewer shows. `calibration` is the Calibration step
([`StereoCalibration`](@ref)); its `dewarpers` observable holds the two
`ImageDewarper`s every later step uses. `dewarpers = (dw1, dw2)` starts
with dewarpers built in a script (see also [`set_dewarpers!`](@ref)).

The other step controllers and settings are those of every workflow
(`prepare`, `passes`, `test`, `run`, `explorer`; `preprocessing`, `mask`,
`scale`, …). Differences from the planar workflow:

- Prepare ([`STEREO_PREPARE_PAGES`](@ref)): preprocessing applies to the
  raw frames; `prepare.preview.processed`/`processed2` and the probe are
  the shown camera's processed pair *dewarped* onto the grid. `dewarped`
  holds the shown camera's raw pair dewarped, for the "raw" view (without
  dewarpers it is `nothing` and the preview stays in camera pixels;
  `prepare.revision` changes with either). The mask
  is drawn on the dewarped grid ([`out_of_view`](@ref) gives the cameras'
  out-of-view union for shading). The scale has `dt`, `time_unit`, and
  `length_unit` only (`pixel_size = 1`; lengths are world units).
- Changing the dewarpers marks the test stale and sizes the pass presets to
  the dewarped grid.
- Test and run call the stereo `apply_recipe(recipe, pairs1, pairs2, dw1, dw2)`;
  results are `StereoPIVResult`s.

With `spawn[]` (a window sets it), frame loading, previews, probes, the
dewarped view, calibration fits, grid builds, and self-calibration run on
worker tasks and hand their results to `deliver`.
"""
struct StereoWorkflow <: AbstractWorkflow
    frames1::FrameSet
    frames2::FrameSet
    camera::Observable{Int}
    calibration::StereoCalibration
    prepare::PrepareState
    passes::PassesEditor
    test::PairTest
    run::RunState
    step::Observable{Symbol}
    preprocessing::Observable{Vector{PreprocessStep}}
    mask::Observable{Union{Nothing,BitMatrix}}
    scale::Observable{Union{Nothing,PhysicalScale}}
    predictor_smoothing::Observable{Bool}
    mask_threshold::Observable{Float64}
    saved::Observable{Union{Nothing,PIVRecipe}}
    settings_path::Observable{String}
    explorer::Observable{Union{Nothing,ResultExplorer}}
    results_path::Observable{Union{Nothing,String}}
    status::Observable{String}
    dewarped::Observable{Union{Nothing,Tuple{Matrix{Float32},Union{Nothing,Matrix{Float32}}}}}
    deliver::Base.RefValue{Any}
    spawn::Base.RefValue{Bool}
    dewarped_key::Base.RefValue{Any}
    dewarped_generation::Base.RefValue{Int}
end

function StereoWorkflow(; files1 = Any[], files2 = Any[], pair_mode::Symbol = :paired,
                        dewarpers = nothing, deliver = f -> f())
    spawn, deliver_ref = Ref(false), Ref{Any}(deliver)
    runner = _workflow_runner(spawn, deliver_ref)
    fs1 = FrameSet(; files = files1, pair_mode, spawn, deliver = deliver_ref)
    fs2 = FrameSet(; files = files2, pair_mode, spawn, deliver = deliver_ref)
    cal = StereoCalibration(; runner)
    wf = StereoWorkflow(fs1, fs2, Observable(1), cal, PrepareState(; runner), PassesEditor(),
                        PairTest(), RunState(), Observable(:images),
                        Observable(PreprocessStep[]),
                        Observable{Union{Nothing,BitMatrix}}(nothing),
                        Observable{Union{Nothing,PhysicalScale}}(nothing),
                        Observable(true), Observable(0.5),
                        Observable{Union{Nothing,PIVRecipe}}(nothing), Observable(""),
                        Observable{Union{Nothing,ResultExplorer}}(nothing),
                        Observable{Union{Nothing,String}}(nothing), Observable(""),
                        Observable{Union{Nothing,Tuple{Matrix{Float32},Union{Nothing,Matrix{Float32}}}}}(nothing),
                        deliver_ref, spawn, Ref{Any}(nothing), Ref(0))
    _link_frames!(wf)
    on(_ -> _sync_analysis_size!(wf), cal.dewarpers)
    _connect_settings!(wf)
    onany((_...) -> _view_changed!(wf), fs1.files, fs1.pair_mode, fs1.pair, fs1.loaded,
          fs2.files, fs2.pair_mode, fs2.pair, fs2.loaded, wf.camera, cal.dewarpers)
    on(_ -> _bump!(wf.prepare), wf.dewarped)          # the Prepare viewer redraws
    _connect_results!(wf)
    dewarpers === nothing || set_dewarpers!(wf, dewarpers...)
    _sync_analysis_size!(wf)
    _view_changed!(wf)
    return wf
end

function Base.show(io::IO, wf::StereoWorkflow)
    print(io, "StereoWorkflow(", frames_summary(wf), "; ", calibration_summary(wf.calibration),
          "; ", passes_summary(wf.passes), "; step :", wf.step[], ")")
end

workflow_steps(::StereoWorkflow) = STEREO_WORKFLOW_STEPS
prepare_pages(::StereoWorkflow) = STEREO_PREPARE_PAGES

# ---------------------------------------------------------------- frames

# The cameras share the pairing rule, the representative pair, and the
# shown frame: an edit of either set is copied to the other (the equality
# guards end the echo).
function _link_frames!(wf::StereoWorkflow)
    for (src, dst) in ((wf.frames1, wf.frames2), (wf.frames2, wf.frames1))
        on(m -> dst.pair_mode[] == m || (dst.pair_mode[] = m), src.pair_mode)
        on(i -> dst.pair[] == i || (dst.pair[] = i), src.pair)
        on(s -> dst.shown[] == s || (dst.shown[] = s), src.shown)
    end
    return wf
end

"""
    camera_frames(wf::StereoWorkflow, camera) -> FrameSet

Camera `camera`'s (1 or 2) frame set.
"""
camera_frames(wf::StereoWorkflow, k::Integer) = (_check_camera(k); k == 1 ? wf.frames1 : wf.frames2)

"""
    shown_frames(wf::StereoWorkflow) -> FrameSet

The frame set of the camera the viewer shows (`wf.camera`).
"""
shown_frames(wf::StereoWorkflow) = camera_frames(wf, wf.camera[])

"""
    set_camera!(wf::StereoWorkflow, camera)

Show camera 1 or 2 in the viewer.
"""
function set_camera!(wf::StereoWorkflow, k::Integer)
    _check_camera(k)
    wf.camera[] == k || (wf.camera[] = Int(k))
    return wf
end

"""
    add_files!(wf::StereoWorkflow, entries; camera)

Append frames (paths or matrices) to camera `camera`'s list, in
acquisition order. Entry `i` of both cameras must show the same instant.
"""
add_files!(wf::StereoWorkflow, entries; camera::Integer) =
    (add_files!(camera_frames(wf, camera), entries); wf)

"""
    clear_files!(wf::StereoWorkflow; camera = nothing)

Remove the frames of `camera`, or of both cameras.
"""
function clear_files!(wf::StereoWorkflow; camera::Union{Nothing,Integer} = nothing)
    for k in (camera === nothing ? (1, 2) : (Int(camera),))
        clear_files!(camera_frames(wf, k))
    end
    return wf
end

"""
    set_pair_mode!(wf::StereoWorkflow, mode)
    select_pair!(wf::StereoWorkflow, i)
    show_frame!(wf::StereoWorkflow, which)

Pairing rule, representative pair, and shown frame (`:a`/`:b`) for both
cameras.
"""
set_pair_mode!(wf::StereoWorkflow, mode::Symbol) = (set_pair_mode!(wf.frames1, mode); wf)

function select_pair!(wf::StereoWorkflow, i::Integer)
    n = max(npairs(wf.frames1), npairs(wf.frames2))
    i = clamp(Int(i), 1, max(n, 1))
    wf.frames1.pair[] == i || (wf.frames1.pair[] = i)
    return wf
end

show_frame!(wf::StereoWorkflow, which::Symbol) = (show_frame!(wf.frames1, which); wf)

"""
    npairs(wf::StereoWorkflow) -> Int

Number of stereo pairs (the smaller camera's pair count).
"""
npairs(wf::StereoWorkflow) = min(npairs(wf.frames1), npairs(wf.frames2))

"""
    frames_problem(wf::StereoWorkflow) -> Union{Nothing,String}

`nothing` when both cameras' frames form the same number of pairs of
equally sized images (and, with dewarpers, match the frame size each
camera was calibrated for); otherwise a message naming the camera and the
first problem.
"""
function frames_problem(wf::StereoWorkflow)
    for k in (1, 2)
        msg = frames_problem(camera_frames(wf, k))
        msg === nothing || return "camera $k: " * msg
    end
    n1, n2 = npairs(wf.frames1), npairs(wf.frames2)
    n1 == n2 || return "camera 1 has $n1 pair$(n1 == 1 ? "" : "s") but camera 2 has $n2"
    dws = wf.calibration.dewarpers[]
    if dws !== nothing
        for k in (1, 2)
            sz, want = frame_size(camera_frames(wf, k)), dws[k].image_size
            sz === nothing || sz == want ||
                return "camera $k frames are $(sz[2])×$(sz[1]) px but its calibration is for $(want[2])×$(want[1]) px"
        end
    end
    return nothing
end

"""
    frames_summary(wf::StereoWorkflow) -> String

One line: frames per camera, pair count, and image sizes.
"""
function frames_summary(wf::StereoWorkflow)
    n1, n2 = length(wf.frames1.files[]), length(wf.frames2.files[])
    (n1 == 0 && n2 == 0) && return "no frames"
    np = npairs(wf)
    txt = (n1 == n2 ? "$n1 frames per camera" : "$n1 + $n2 frames") *
          " · $np pair" * (np == 1 ? "" : "s")
    s1, s2 = frame_size(wf.frames1), frame_size(wf.frames2)
    px(s) = "$(s[2])×$(s[1]) px"
    if s1 !== nothing && s2 !== nothing
        txt *= s1 == s2 ? " · " * px(s1) : " · " * px(s1) * " / " * px(s2)
    end
    return txt
end

"""
    workflow_problem(wf::StereoWorkflow) -> Union{Nothing,String}

[`frames_problem`](@ref), or the missing dewarpers.
"""
function workflow_problem(wf::StereoWorkflow)
    msg = frames_problem(wf)
    msg === nothing || return msg
    wf.calibration.dewarpers[] === nothing && return "calibrate the cameras first (no dewarpers)"
    return nothing
end

# ---------------------------------------------------------------- calibration

"""
    set_dewarpers!(wf::StereoWorkflow, dw1, dw2)

Use dewarpers built in a script (sharing one `DewarpGrid`) instead of the
Calibration step's fit.
"""
set_dewarpers!(wf::StereoWorkflow, dw1::ImageDewarper, dw2::ImageDewarper) =
    (set_dewarpers!(wf.calibration, dw1, dw2); wf)

"""
    out_of_view(wf::StereoWorkflow) -> Union{Nothing,BitMatrix}

Grid nodes either camera cannot see (`dw1.mask .| dw2.mask`; `true` =
excluded in every analysis), or `nothing` without dewarpers.
"""
function out_of_view(wf::StereoWorkflow)
    dws = wf.calibration.dewarpers[]
    return dws === nothing ? nothing : dws[1].mask .| dws[2].mask
end

"""
    grid_size(wf::StereoWorkflow) -> Union{Nothing,Dims{2}}

Size `(rows, cols)` of the dewarped grid, or `nothing` without dewarpers.
"""
function grid_size(wf::StereoWorkflow)
    dws = wf.calibration.dewarpers[]
    return dws === nothing ? nothing : size(dws[1].grid)
end

_sync_analysis_size!(wf::StereoWorkflow) = (set_analysis_size!(wf.passes, grid_size(wf)); wf)

# The leading pairs' distinct frames, camera by camera (entry k of each
# camera is the same instant).
function _selfcal_frames(wf::StereoWorkflow, n::Integer)
    p1, p2 = frame_pairs(wf.frames1), frame_pairs(wf.frames2)
    f1, f2 = Any[], Any[]
    for (a, b) in zip(p1[1:min(end, n)], p2[1:min(end, n)])
        for j in 1:2
            any(e -> _same_entry(e, a[j]), f1) && continue
            push!(f1, a[j])
            push!(f2, b[j])
        end
    end
    return f1, f2
end

"""
    start_selfcal!(wf::StereoWorkflow; pairs = wf.calibration.selfcal_pairs[],
                   keep_disparity_maps = wf.calibration.keep_disparity_maps[])

Self-calibrate the dewarpers ([`self_calibrate`](@ref Hammerhead.self_calibrate),
Wieneke 2005) on the frames of the first `pairs` pairs (each frame of
camera 1 with the same-instant frame of camera 2), preprocessed with the
current preprocessing and masked with the current mask when it fits the
grid. Runs on the runner (a worker task in a window); the result lands in
`wf.calibration.selfcal` with its summary in `selfcal_status`, and
[`apply_selfcal!`](@ref) then uses the corrected dewarpers.
"""
function start_selfcal!(wf::StereoWorkflow; pairs::Integer = wf.calibration.selfcal_pairs[],
                        keep_disparity_maps::Bool = wf.calibration.keep_disparity_maps[])
    cal = wf.calibration
    cal.selfcal_running[] && return wf
    dws = cal.dewarpers[]
    dws === nothing && (cal.selfcal_status[] = "build the dewarpers first"; return wf)
    msg = frames_problem(wf)
    msg === nothing || (cal.selfcal_status[] = msg; return wf)
    pairs >= 1 || throw(ArgumentError("self-calibration needs at least one pair"))
    f1, f2 = _selfcal_frames(wf, pairs)
    preprocess = recipe_preprocess(wf.preprocessing[])
    m = wf.mask[]
    mask = m !== nothing && size(m) == size(dws[1].grid) ? copy(m) : nothing
    image_type = wf.passes.image_type[]
    g = (cal.selfcal_generation[] += 1)
    cal.selfcal_running[] = true
    n = length(f1)
    cal.selfcal_status[] = "self-calibrating on $n instant" * (n == 1 ? "" : "s") * "…"
    job = () -> self_calibrate(f1, f2, dws[1], dws[2]; keep_disparity_maps, preprocess,
                               mask, image_type)
    cal.runner[](job, out -> _finish_selfcal!(cal, g, dws, out))
    return wf
end

function _finish_selfcal!(cal::StereoCalibration, g::Int, source, out)
    g == cal.selfcal_generation[] || return cal
    cal.selfcal_running[] = false
    if out.err !== nothing
        cal.selfcal_status[] = "self-calibration failed: " * _errmsg(out.err)
        return cal
    end
    cur = cal.dewarpers[]
    if cur === nothing || cur[1] !== source[1] || cur[2] !== source[2]
        cal.selfcal_status[] = "the dewarpers changed during the self-calibration; run it again"
        return cal
    end
    d1, d2, report = out.value
    cal.selfcal[] = (; report, dewarpers = (d1, d2), source)
    cal.selfcal_status[] = selfcal_summary(report)
    return cal
end

"""
    apply_selfcal!(wf::StereoWorkflow) -> Bool

[`apply_selfcal!`](@ref) on the workflow's calibration.
"""
apply_selfcal!(wf::StereoWorkflow) = apply_selfcal!(wf.calibration)

# ---------------------------------------------------------------- recipe

workflow_recipe(wf::StereoWorkflow) =
    PIVRecipe(wf.passes.passes[]; preprocessing = wf.preprocessing[], mask = wf.mask[],
              roi = nothing, scale = wf.scale[], mode = wf.passes.mode[],
              image_type = wf.passes.image_type[],
              predictor_smoothing = wf.predictor_smoothing[],
              mask_threshold = wf.mask_threshold[])

_check_recipe(::StereoWorkflow, r::PIVRecipe) =
    r.roi === nothing ||
    throw(ArgumentError("these settings have an ROI, which stereo analysis does not support; " *
                        "remove it, or mask the dewarped grid instead"))

function _test_inputs(wf::StereoWorkflow)
    dw1, dw2 = wf.calibration.dewarpers[]
    mode = wf.passes.mode[]
    return (_test_pairs(wf.frames1, mode), _test_pairs(wf.frames2, mode), dw1, dw2)
end

function _run_inputs(wf::StereoWorkflow)
    dw1, dw2 = wf.calibration.dewarpers[]
    return (frame_pairs(wf.frames1), frame_pairs(wf.frames2), dw1, dw2)
end

_test_label(wf::StereoWorkflow) = wf.frames1.pair[]

function _inputs_stale(wf::StereoWorkflow)
    wf.passes.mode[] === :sequence && wf.test.pair[] != wf.frames1.pair[] && return true
    inp, dws = wf.test.inputs[], wf.calibration.dewarpers[]
    return inp === nothing || dws === nothing || length(inp) != 4 ||
           inp[3] !== dws[1] || inp[4] !== dws[2]
end

# ---------------------------------------------------------------- step rail

function _step_status(wf::StereoWorkflow, step::Symbol)
    if step === :images
        for fs in (wf.frames1, wf.frames2)
            pair_loading(fs) && return (:busy, "loading pair $(fs.pair[])…")
        end
        msg = frames_problem(wf)
        return msg === nothing ? (:ok, frames_summary(wf)) : (:todo, msg)
    elseif step === :calibration
        cal = wf.calibration
        cal.fitting[] && return (:busy, "fitting the cameras…")
        cal.building[] && return (:busy, "building the dewarp grid…")
        cal.selfcal_running[] && return (:busy, "self-calibrating…")
        cal.dewarpers[] === nothing && return (:todo, calibration_summary(cal))
        fit_stale(cal) &&
            return (:attention, "plates or detection changed since the fit: " * grid_summary(cal))
        return (:ok, calibration_summary(cal))
    elseif step === :prepare
        sz = grid_size(wf)
        m = wf.mask[]
        m === nothing || sz === nothing || size(m) == sz ||
            return (:attention, "the mask is $(size(m, 2))×$(size(m, 1)) px but the dewarped grid is $(sz[2])×$(sz[1]) px")
        status = wf.prepare.preview.status[]
        isempty(status) || return (:attention, status)
        parts = _prepare_parts(wf)
        wf.scale[] === nothing || push!(parts, "scaled")
        return (:ok, isempty(parts) ? "no preprocessing, no mask" : join(parts, " · "))
    end
    throw(ArgumentError("unknown step :$step"))
end

# ---------------------------------------------------------------- Prepare

_has_roi(::StereoWorkflow) = false
_has_scale_tool(::StereoWorkflow) = false
_mask_target(wf::StereoWorkflow) = (grid_size(wf), "the dewarped grid is")
_scale_fields(::StereoWorkflow) = (:dt, :length_unit, :time_unit)
_current_scale(wf::StereoWorkflow) =
    something(wf.scale[], PhysicalScale(1.0, 1.0, wf.calibration.length_unit[], "frame"))

# Dewarp one frame onto the grid (the preview's `post` step).
struct _DewarpFrame{D<:ImageDewarper}
    dw::D
end
(d::_DewarpFrame)(img::AbstractMatrix) = dewarp(d.dw, img)

# The shown camera, its pair, or the dewarpers changed: the preview gets the
# raw pair and the dewarping, the mask editor follows the grid size, and the
# dewarped raw view is recomputed. While a pair loads on a worker,
# everything keeps the previous pair.
function _view_changed!(wf::StereoWorkflow)
    ps = wf.prepare
    sz = grid_size(wf)
    me = ps.mask[]
    if sz === nothing
        me === nothing || _set_editors!(wf, nothing)
    elseif me === nothing || me.size != sz
        _set_editors!(wf, sz)
    end
    dws = wf.calibration.dewarpers[]
    dw = dws === nothing ? nothing : dws[wf.camera[]]
    post = dw === nothing ? nothing : _DewarpFrame(dw)
    fs = shown_frames(wf)
    imgs = try
        pair_images(fs)
    catch
        nothing
    end
    if imgs === nothing
        (pair_loading(fs) || current_pair(fs) !== nothing) && return wf
        _set_preview_input!(ps.preview, nothing, nothing, post)
        _request_dewarped!(wf, nothing, nothing, nothing)
        return wf
    end
    a, b = imgs
    b = size(a) == size(b) ? b : nothing
    _set_preview_input!(ps.preview, a, b, post)
    _request_dewarped!(wf, dw, a, b)
    return wf
end

function _request_dewarped!(wf::StereoWorkflow, dw, a, b)
    k = wf.dewarped_key[]
    k !== nothing && k[1] === dw && k[2] === a && k[3] === b && return wf
    wf.dewarped_key[] = (dw, a, b)
    g = (wf.dewarped_generation[] += 1)
    if dw === nothing || a === nothing
        wf.dewarped[] === nothing || (wf.dewarped[] = nothing)
        return wf
    end
    job = () -> (convert(Matrix{Float32}, dewarp(dw, a)),
                 b === nothing ? nothing : convert(Matrix{Float32}, dewarp(dw, b)))
    apply = function (out)
        g == wf.dewarped_generation[] || return
        wf.dewarped[] = out.err === nothing ? out.value : nothing
    end
    wf.calibration.runner[](job, apply)
    return wf
end

"""
    estimate_background!(wf::StereoWorkflow; kwargs...)

Not available for stereo: a recipe holds one preprocessing list for both
cameras, and each camera has its own background. Sets
`wf.prepare.status` and returns `wf`.
"""
function estimate_background!(wf::StereoWorkflow; kwargs...)
    wf.prepare.status[] = "background subtraction is not available for stereo " *
                          "(the cameras need different backgrounds)"
    return wf
end

# A window closed with jobs in flight: forget them and catch up inline.
function _abandon_jobs!(wf::StereoWorkflow)
    _abandon_load!(wf.frames1)
    _abandon_load!(wf.frames2)
    cal = wf.calibration
    if cal.fitting[]
        cal.fit_generation[] += 1
        cal.fitting[] = false
        cal.fit_status[] = "fit interrupted"
    end
    if cal.building[]
        cal.grid_generation[] += 1
        cal.building[] = false
        cal.grid_status[] = grid_summary(cal)
    end
    if cal.selfcal_running[]
        cal.selfcal_generation[] += 1
        cal.selfcal_running[] = false
        cal.selfcal_status[] = "self-calibration interrupted"
    end
    wf.dewarped_key[] = nothing
    _request_preview!(wf.prepare.preview)
    _view_changed!(wf)
    return wf
end
