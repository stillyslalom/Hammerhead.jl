# The planar PIV workflow window's state: one controller per step, and the
# single place where the processing settings become a core PIVRecipe. The
# settings, test, run, and results functions are shared with the stereo
# workflow (workflow.jl); this file holds the planar hooks.

"""
Steps of the planar workflow window, in order.
"""
const WORKFLOW_STEPS = (:images, :prepare, :passes, :test, :run, :results)

"""
    PlanarWorkflow(; files = Any[], pair_mode = :paired, deliver = f -> f())

State of the planar PIV workflow window (an [`AbstractWorkflow`](@ref)).
Step controllers: `frames` (`FrameSet`), `prepare` (`PrepareState`),
`passes` (`PassesEditor`), `particles` (`ParticleSettings`), `test`
(`PairTest`) and `run` (`RunState`);
`explorer` holds a `ResultExplorer` for the Results step. Preprocessing,
mask, ROI, and physical scale are kept in `preprocessing`, `mask`, `roi`,
and `scale`; the Prepare step's editors read and write them.

`workflow_recipe` assembles the current settings as a core
`PIVRecipe`; `passes.mode` picks the analysis (one of
[`ANALYSIS_MODES`](@ref): PIV per pair or ensemble, or the particle modes
`:ptv` and `:tracking`, whose Passes step — "Particles" on the rail — edits
`particles` and uses the pass schedule as the PIV predictor; a tracking test
follows up to [`TRACKING_TEST_FRAMES`](@ref) frames from the representative
pair, and a tracking run follows every frame in order). On the Passes step
of a particle mode, `particles.detected` previews the detection on the shown
frame. `save_settings` and `load_settings!` store and
open it; a results file written by a run also carries it. `deliver` receives
the observable updates of background jobs (see `start_test!`).

`spawn[]` (`false` by default; a window sets it) moves the remaining work
off the caller's thread too: loading the representative pair, the
preprocessing preview and probe, and the background estimate run on worker
tasks and hand their results to `deliver`. Without it they run inline.
"""
struct PlanarWorkflow <: AbstractWorkflow
    frames::FrameSet
    prepare::PrepareState
    passes::PassesEditor
    particles::ParticleSettings
    test::PairTest
    run::RunState
    step::Observable{Symbol}
    preprocessing::Observable{Vector{PreprocessStep}}
    mask::Observable{Union{Nothing,BitMatrix}}
    roi::Observable{Union{Nothing,ROI}}
    scale::Observable{Union{Nothing,PhysicalScale}}
    predictor_smoothing::Observable{Bool}
    mask_threshold::Observable{Float64}
    saved::Observable{Union{Nothing,PIVRecipe}}
    settings_path::Observable{String}
    explorer::Observable{Union{Nothing,ResultExplorer}}
    results_path::Observable{Union{Nothing,String}}
    status::Observable{String}
    deliver::Base.RefValue{Any}
    spawn::Base.RefValue{Bool}
end

function PlanarWorkflow(; files = Any[], pair_mode::Symbol = :paired, deliver = f -> f())
    spawn, deliver_ref = Ref(false), Ref{Any}(deliver)
    frames = FrameSet(; files, pair_mode, spawn, deliver = deliver_ref)
    prepare = PrepareState(; runner = _workflow_runner(spawn, deliver_ref))
    wf = PlanarWorkflow(frames, prepare, PassesEditor(), ParticleSettings(), PairTest(), RunState(),
                        Observable(:images),
                        Observable(PreprocessStep[]),
                        Observable{Union{Nothing,BitMatrix}}(nothing),
                        Observable{Union{Nothing,ROI}}(nothing),
                        Observable{Union{Nothing,PhysicalScale}}(nothing),
                        Observable(true), Observable(0.5),
                        Observable{Union{Nothing,PIVRecipe}}(nothing), Observable(""),
                        Observable{Union{Nothing,ResultExplorer}}(nothing),
                        Observable{Union{Nothing,String}}(nothing), Observable(""),
                        deliver_ref, spawn)
    onany((_...) -> _sync_analysis_size!(wf), frames.files, frames.pair_mode, frames.loaded, wf.roi)
    _sync_analysis_size!(wf)
    _connect_prepare!(wf)
    _connect_results!(wf)
    pp = prepare.preview
    onany((_...) -> _request_detection!(wf), wf.step, wf.passes.mode, wf.particles.ptv, wf.mask,
          frames.shown, pp.processed)
    return wf
end

function Base.show(io::IO, wf::PlanarWorkflow)
    print(io, "PlanarWorkflow(", frames_summary(wf.frames), "; ", passes_summary(wf.passes),
          "; step :", wf.step[], ")")
end

workflow_steps(::PlanarWorkflow) = WORKFLOW_STEPS
prepare_pages(::PlanarWorkflow) = PREPARE_PAGES
function workflow_problem(wf::PlanarWorkflow)
    msg = frames_problem(wf.frames)
    msg === nothing || return msg
    mode = wf.passes.mode[]
    mode === :ensemble && wf.roi[] !== nothing &&
        return "an ensemble analyzes whole frames: clear the region (Prepare › Region) or use a mask"
    _particle_mode(mode) && wf.roi[] !== nothing &&
        return "particle analysis covers whole frames: clear the region (Prepare › Region) or use a mask"
    return nothing
end

# ---------------------------------------------------------------- detection preview

# On the Passes step of a particle mode, detect particles on the shown frame
# of the representative pair, as the batch will (after preprocessing, inside
# the mask), so the detection settings can be judged on the image.
function _request_detection!(wf::PlanarWorkflow)
    ps = wf.particles
    active = wf.step[] === :passes && _particle_mode(wf.passes.mode[])
    img = active ? _detection_frame(wf) : nothing
    m = wf.mask[]
    key = (img, ps.ptv[], m)
    k = ps.key[]
    k !== nothing && k[1] === key[1] && k[2] == key[2] && k[3] === key[3] && return wf
    ps.key[] = key
    g = (ps.generation[] += 1)
    if img === nothing
        ps.detected[] === nothing || (ps.detected[] = nothing)
        isempty(ps.detect_status[]) || (ps.detect_status[] = "")
        return wf
    end
    params = ps.ptv[]
    mask = m !== nothing && size(m) == size(img) ? m : nothing
    apply = function (out)
        g == ps.generation[] || return
        if out.err === nothing
            ps.detected[] = out.value
            n = length(out.value)
            ps.detect_status[] = "$n particle" * (n == 1 ? "" : "s") * " detected on frame " *
                                 (wf.frames.shown[] === :a ? "A" : "B") * " of the pair"
        else
            ps.detected[] = nothing
            ps.detect_status[] = "detection failed: " * _errmsg(out.err)
        end
    end
    wf.prepare.preview.runner[](() -> detect_particles(img, params; mask), apply)
    return wf
end

# The shown frame after preprocessing (the preview's), or `nothing`.
function _detection_frame(wf::PlanarWorkflow)
    pp = wf.prepare.preview
    return wf.frames.shown[] === :a ? pp.processed[] : pp.processed2[]
end

# Presets are sized to the analyzed region: the ROI, else the frame.
function _sync_analysis_size!(wf::PlanarWorkflow)
    roi = wf.roi[]
    sz = roi !== nothing ? (length(roi.rows), length(roi.cols)) : frame_size(wf.frames)
    set_analysis_size!(wf.passes, sz)
    return wf
end

"""
    workflow_recipe(wf::AbstractWorkflow) -> PIVRecipe

The current settings as a core `PIVRecipe` (a `StereoWorkflow`'s has no ROI).
"""
function workflow_recipe(wf::PlanarWorkflow)
    mode, ps = wf.passes.mode[], wf.particles
    # particle modes analyze whole frames (workflow_problem reports a region)
    PIVRecipe(wf.passes.passes[]; preprocessing = wf.preprocessing[], mask = wf.mask[],
              roi = _particle_mode(mode) ? nothing : wf.roi[], scale = wf.scale[], mode,
              image_type = wf.passes.image_type[],
              predictor_smoothing = wf.predictor_smoothing[],
              mask_threshold = wf.mask_threshold[], ptv = ps.ptv[],
              ptv_predictor = ps.predictor[], min_track_length = ps.min_track_length[],
              max_gap = ps.max_gap[])
end

_load_specific!(wf::PlanarWorkflow, r::PIVRecipe) =
    (wf.roi[] = r.roi; load_particles!(wf.particles, r); wf)

_check_recipe(::PlanarWorkflow, r::PIVRecipe) =
    r.preprocessing isa Tuple &&
    throw(ArgumentError("these settings preprocess each camera separately; open them in the stereo window"))

"""
Frames a tracking test follows: the representative pair's first frame and
the frames after it, up to this many.
"""
const TRACKING_TEST_FRAMES = 10

# The tracking test's frames (see TRACKING_TEST_FRAMES).
function _tracking_test_frames(fs::FrameSet)
    pr = current_pair(fs)
    files = fs.files[]
    pr === nothing && return Any[]
    i = something(findfirst(e -> _same_entry(e, pr[1]), files), 1)
    return files[i:min(end, i + TRACKING_TEST_FRAMES - 1)]
end

function _test_inputs(wf::PlanarWorkflow)
    mode = wf.passes.mode[]
    mode === :tracking && return (_tracking_test_frames(wf.frames),)
    return (_test_pairs(wf.frames, mode === :ensemble ? :ensemble : :sequence),)
end
_run_inputs(wf::PlanarWorkflow) =
    (wf.passes.mode[] === :tracking ? collect(Any, wf.frames.files[]) : frame_pairs(wf.frames),)
_test_label(wf::PlanarWorkflow) = wf.frames.pair[]

function _inputs_stale(wf::PlanarWorkflow)
    mode = wf.passes.mode[]
    if mode === :tracking
        inp = wf.test.inputs[]
        cur = _tracking_test_frames(wf.frames)
        return inp === nothing || length(inp[1]) != length(cur) ||
               !all(((a, b),) -> _same_entry(a, b), zip(inp[1], cur))
    end
    return (mode !== :ensemble && wf.test.pair[] != wf.frames.pair[]) ||
           _test_pairs_changed(wf, 1, wf.frames)
end

function _passes_status(wf::PlanarWorkflow)
    mode = wf.passes.mode[]
    _particle_mode(mode) || return _piv_passes_status(wf)
    ps = wf.particles
    isempty(ps.error[]) || return (:attention, ps.error[])
    isempty(wf.passes.error[]) || return (:attention, wf.passes.error[])
    return (:ok, particles_summary(ps, mode))
end

step_label(wf::PlanarWorkflow, step::Symbol) =
    step === :passes && _particle_mode(wf.passes.mode[]) ? "Particles" : STEP_LABELS[step]

function _step_status(wf::PlanarWorkflow, step::Symbol)
    if step === :images
        pair_loading(wf.frames) && return (:busy, "loading pair $(wf.frames.pair[])…")
        msg = frames_problem(wf.frames)
        return msg === nothing ? (:ok, frames_summary(wf.frames)) : (:todo, msg)
    elseif step === :prepare
        sz = frame_size(wf.frames)
        m = wf.mask[]
        m === nothing || sz === nothing || size(m) == sz ||
            return (:attention, "the mask is $(size(m, 2))×$(size(m, 1)) px but the frames are $(sz[2])×$(sz[1]) px")
        status = wf.prepare.preview.status[]
        isempty(status) || return (:attention, status)
        parts = _prepare_parts(wf)
        wf.roi[] === nothing || push!(parts, "ROI")
        wf.scale[] === nothing || push!(parts, "scaled")
        return (:ok, isempty(parts) ? "full image, no preprocessing" : join(parts, " · "))
    end
    throw(ArgumentError("unknown step :$step"))
end
