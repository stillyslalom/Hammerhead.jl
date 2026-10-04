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
`passes` (`PassesEditor`), `test` (`PairTest`) and `run` (`RunState`);
`explorer` holds a `ResultExplorer` for the Results step. Preprocessing,
mask, ROI, and physical scale are kept in `preprocessing`, `mask`, `roi`,
and `scale`; the Prepare step's editors read and write them.

`workflow_recipe` assembles the current settings as a core
`PIVRecipe`. `save_settings` and `load_settings!` store and
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
    wf = PlanarWorkflow(frames, prepare, PassesEditor(), PairTest(), RunState(), Observable(:images),
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
    wf.passes.mode[] === :ensemble && wf.roi[] !== nothing &&
        return "an ensemble analyzes whole frames: clear the region (Prepare › Region) or use a mask"
    return nothing
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
workflow_recipe(wf::PlanarWorkflow) =
    PIVRecipe(wf.passes.passes[]; preprocessing = wf.preprocessing[], mask = wf.mask[],
              roi = wf.roi[], scale = wf.scale[], mode = wf.passes.mode[],
              image_type = wf.passes.image_type[],
              predictor_smoothing = wf.predictor_smoothing[],
              mask_threshold = wf.mask_threshold[])

_load_region!(wf::PlanarWorkflow, r::PIVRecipe) = (wf.roi[] = r.roi; wf)

_test_inputs(wf::PlanarWorkflow) = (_test_pairs(wf.frames, wf.passes.mode[]),)
_run_inputs(wf::PlanarWorkflow) = (frame_pairs(wf.frames),)
_test_label(wf::PlanarWorkflow) = wf.frames.pair[]
_inputs_stale(wf::PlanarWorkflow) =
    (wf.passes.mode[] === :sequence && wf.test.pair[] != wf.frames.pair[]) ||
    _test_pairs_changed(wf, 1, wf.frames)

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
