# The planar PIV workflow window's state: one controller per step, and the
# single place where the processing settings become a core PIVRecipe.

"""
Steps of a workflow window, in order.
"""
const WORKFLOW_STEPS = (:images, :prepare, :passes, :test, :run, :results)

const STEP_LABELS = Dict(:images => "Images", :prepare => "Prepare", :passes => "Passes",
                         :test => "Test pair", :run => "Run", :results => "Results")

"""
    PlanarWorkflow(; files = Any[], pair_mode = :paired, deliver = f -> f())

State of the planar PIV workflow window. Step controllers:
`frames` (`FrameSet`), `passes` (`PassesEditor`), `test`
(`PairTest`) and `run` (`RunState`); `explorer` holds a
`ResultExplorer` for the Results step. Preprocessing, mask, ROI, and
physical scale are kept in `preprocessing`, `mask`, `roi`, and `scale`.

`workflow_recipe` assembles the current settings as a core
`PIVRecipe`. `save_settings` and `load_settings!` store and
open it; a results file written by a run also carries it. `deliver` receives
the observable updates of background jobs (see `start_test!`).
"""
struct PlanarWorkflow
    frames::FrameSet
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
end

function PlanarWorkflow(; files = Any[], pair_mode::Symbol = :paired, deliver = f -> f())
    frames = FrameSet(; files, pair_mode)
    wf = PlanarWorkflow(frames, PassesEditor(), PairTest(), RunState(), Observable(:images),
                        Observable(PreprocessStep[]),
                        Observable{Union{Nothing,BitMatrix}}(nothing),
                        Observable{Union{Nothing,ROI}}(nothing),
                        Observable{Union{Nothing,PhysicalScale}}(nothing),
                        Observable(true), Observable(0.5),
                        Observable{Union{Nothing,PIVRecipe}}(nothing), Observable(""),
                        Observable{Union{Nothing,ResultExplorer}}(nothing),
                        Observable{Union{Nothing,String}}(nothing), Observable(""),
                        Ref{Any}(deliver))
    onany((_...) -> _sync_analysis_size!(wf), frames.files, frames.pair_mode, wf.roi)
    _sync_analysis_size!(wf)
    # A finished batch becomes the Results step's data.
    on(wf.run.running) do running
        running && return
        out = wf.run.finished_output[]
        if out !== nothing
            open_results!(wf, out)
        elseif !isempty(wf.run.completed[])
            wf.explorer[] = ResultExplorer(collect(wf.run.completed[]))
            wf.results_path[] = nothing
        end
    end
    return wf
end

function Base.show(io::IO, wf::PlanarWorkflow)
    print(io, "PlanarWorkflow(", frames_summary(wf.frames), "; ", passes_summary(wf.passes),
          "; step :", wf.step[], ")")
end

# Presets are sized to the analyzed region: the ROI, else the frame.
function _sync_analysis_size!(wf::PlanarWorkflow)
    roi = wf.roi[]
    sz = roi !== nothing ? (length(roi.rows), length(roi.cols)) : frame_size(wf.frames)
    set_analysis_size!(wf.passes, sz)
    return wf
end

"""
    workflow_recipe(wf::PlanarWorkflow) -> PIVRecipe

The current settings as a core `PIVRecipe`.
"""
workflow_recipe(wf::PlanarWorkflow) =
    PIVRecipe(wf.passes.passes[]; preprocessing = wf.preprocessing[], mask = wf.mask[],
              roi = wf.roi[], scale = wf.scale[], mode = wf.passes.mode[],
              image_type = wf.passes.image_type[],
              predictor_smoothing = wf.predictor_smoothing[],
              mask_threshold = wf.mask_threshold[])

"""
    load_settings!(wf::PlanarWorkflow, recipe_or_path)

Open settings from a `PIVRecipe`, a recipe file, or a results file written
by a run. Every recipe field is kept, so `workflow_recipe(wf)` returns an
equal recipe until something is edited.
"""
load_settings!(wf::PlanarWorkflow, path::AbstractString) =
    (load_settings!(wf, load_recipe(path)); wf.settings_path[] = String(path); wf)

function load_settings!(wf::PlanarWorkflow, r::PIVRecipe)
    wf.preprocessing[] = copy(r.preprocessing)
    wf.mask[] = r.mask === nothing ? nothing : copy(r.mask)
    wf.scale[] = r.scale
    wf.predictor_smoothing[] = r.predictor_smoothing
    wf.mask_threshold[] = r.mask_threshold
    wf.passes.mode[] = r.mode
    set_image_type!(wf.passes, r.image_type)
    wf.roi[] = r.roi                     # resizes a linked preset; replaced next
    load_passes!(wf.passes, r.passes)
    wf.saved[] = workflow_recipe(wf)
    wf.settings_path[] = ""
    wf.status[] = "opened settings: $(length(r.passes)) passes"
    return wf
end

"""
    save_settings(wf::PlanarWorkflow, path) -> path

Save the current settings with `save_recipe`.
"""
function save_settings(wf::PlanarWorkflow, path::AbstractString)
    r = workflow_recipe(wf)
    save_recipe(path, r)
    wf.saved[] = r
    wf.settings_path[] = String(path)
    wf.status[] = "saved settings to $(basename(path))"
    return path
end

"""
    settings_modified(wf::PlanarWorkflow) -> Bool

Whether the settings differ from the last opened or saved recipe (always
`true` before the first save).
"""
settings_modified(wf::PlanarWorkflow) = wf.saved[] === nothing || wf.saved[] != workflow_recipe(wf)

"""
    set_step!(wf::PlanarWorkflow, step)

Show a step (one of `WORKFLOW_STEPS`).
"""
function set_step!(wf::PlanarWorkflow, step::Symbol)
    step in WORKFLOW_STEPS || throw(ArgumentError("unknown step :$step"))
    wf.step[] == step || (wf.step[] = step)
    return wf
end

# Pairs a test analyzes: the representative pair, or the first few pairs for
# an ensemble (one pair says little about an ensemble's yield).
const ENSEMBLE_TEST_PAIRS = 10

function _test_pairs(wf::PlanarWorkflow)
    if wf.passes.mode[] === :ensemble
        prs = frame_pairs(wf.frames)
        return prs[1:min(end, ENSEMBLE_TEST_PAIRS)]
    end
    pr = current_pair(wf.frames)
    return pr === nothing ? Any[] : Any[pr]
end

"""
    test_pair!(wf::PlanarWorkflow; spawn = true)

Analyze the representative pair with the current settings, through the same
`apply_recipe` call the batch uses.
"""
function test_pair!(wf::PlanarWorkflow; spawn::Bool = true)
    msg = frames_problem(wf.frames)
    msg === nothing || (wf.test.status[] = msg; return wf)
    recipe = try
        workflow_recipe(wf)
    catch err
        wf.test.status[] = _errmsg(err)
        return wf
    end
    start_test!(wf.test, recipe, _test_pairs(wf), wf.frames.pair[];
                deliver = wf.deliver[], spawn)
    return wf
end

"""
    test_stale(wf::PlanarWorkflow) -> Bool

Whether the settings or representative pair changed since the last test.
"""
test_stale(wf::PlanarWorkflow) =
    wf.test.recipe[] === nothing || wf.test.recipe[] != workflow_recipe(wf) ||
    (wf.passes.mode[] === :sequence && wf.test.pair[] != wf.frames.pair[])

"""
    start_run!(wf::PlanarWorkflow; spawn = true)

Run the batch on all pairs with a snapshot of the current settings.
"""
function start_run!(wf::PlanarWorkflow; spawn::Bool = true)
    msg = frames_problem(wf.frames)
    msg === nothing || (wf.run.status[] = msg; return wf)
    recipe = try
        workflow_recipe(wf)
    catch err
        wf.run.status[] = _errmsg(err)
        return wf
    end
    start_run!(wf.run, recipe, frame_pairs(wf.frames); deliver = wf.deliver[], spawn)
    return wf
end

cancel_run!(wf::PlanarWorkflow) = cancel_run!(wf.run)

"""
    open_results!(wf::PlanarWorkflow, path)

Browse a results file in the Results step (entries load on demand).
"""
function open_results!(wf::PlanarWorkflow, path::AbstractString)
    ex = try
        ResultExplorer(Hammerhead.ResultFile(path))
    catch err
        wf.status[] = "cannot open results: " * _errmsg(err)
        return wf
    end
    wf.explorer[] = ex
    wf.results_path[] = String(path)
    wf.status[] = "results: $(basename(path))"
    return wf
end

"""
    step_status(wf::PlanarWorkflow, step) -> (state, summary)

`state` is `:todo`, `:ok`, `:attention`, or `:busy`, with a one-line summary
for the step rail.
"""
function step_status(wf::PlanarWorkflow, step::Symbol)
    if step === :images
        msg = frames_problem(wf.frames)
        return msg === nothing ? (:ok, frames_summary(wf.frames)) : (:todo, msg)
    elseif step === :prepare
        parts = String[]
        isempty(wf.preprocessing[]) || push!(parts, "$(length(wf.preprocessing[])) preprocessing steps")
        wf.mask[] === nothing || push!(parts, "mask")
        wf.roi[] === nothing || push!(parts, "ROI")
        wf.scale[] === nothing || push!(parts, "scaled")
        return (:ok, isempty(parts) ? "full image, no preprocessing" : join(parts, " · "))
    elseif step === :passes
        isempty(wf.passes.error[]) || return (:attention, wf.passes.error[])
        return (:ok, passes_summary(wf.passes))
    elseif step === :test
        wf.test.running[] && return (:busy, "testing…")
        s = test_summary(wf.test)
        s === nothing && return (:todo, "not tested")
        txt = @sprintf("%.0f %% valid · %.2f s", 100 * s.valid_fraction, s.seconds)
        return test_stale(wf) ? (:attention, "settings changed since: " * txt) : (:ok, txt)
    elseif step === :run
        wf.run.running[] && return (:busy, "$(wf.run.progress[][1]) of $(wf.run.progress[][2]) pairs")
        return isempty(wf.run.status[]) ? (:todo, "not run") : (:ok, wf.run.status[])
    elseif step === :results
        ex = wf.explorer[]
        ex === nothing && return (:todo, "no results yet")
        return (:ok, wf.results_path[] === nothing ? "$(nframes(ex)) results in memory" :
                     basename(wf.results_path[]))
    end
    throw(ArgumentError("unknown step :$step"))
end
