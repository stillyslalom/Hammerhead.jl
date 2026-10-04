# State and steps shared by the workflow windows (planar and stereo): the
# settings ↔ recipe round trip, the Passes / Test pair / Run / Results steps,
# and the background-job plumbing. Each workflow type supplies a few hooks:
#
#   workflow_steps(wf)          its steps, in order
#   workflow_recipe(wf)         the settings as a core PIVRecipe
#   workflow_problem(wf)        why the frames cannot be analyzed yet, or nothing
#   prepare_pages(wf)           its Prepare sub-pages
#   _check_recipe(wf, r)        reject a recipe the workflow cannot hold
#   _load_specific!(wf, r)      recipe fields only this workflow has (planar: ROI, particles)
#   _test_inputs(wf)            apply_recipe's positional inputs after the recipe,
#   _run_inputs(wf)             for the test and for the batch
#   _test_label(wf)             the representative pair's index
#   _inputs_stale(wf)           whether the test's inputs changed (pair, dewarpers)
#   _step_status(wf, step)      rail status of the steps that are not shared
#   _edited_steps(wf),          the preprocessing list the Prepare preview
#   _set_edited_steps!(wf, s)   edits (stereo: the shown camera's)

"""
    AbstractWorkflow

Supertype of the workflow-window controllers, [`PlanarWorkflow`](@ref) and
[`StereoWorkflow`](@ref). Every workflow has the step controllers `prepare`
(`PrepareState`), `passes` (`PassesEditor`), `test` (`PairTest`), and `run`
(`RunState`); the settings `preprocessing`, `mask`, `scale`,
`predictor_smoothing`, and `mask_threshold`; `step` (the open step, one of
[`workflow_steps`](@ref)); `saved`/`settings_path` (the last opened or saved
recipe); `explorer`/`results_path` (the Results step); `status`; and the
background-job plumbing `deliver` and `spawn`.

The settings, test, run, and results functions ([`workflow_recipe`](@ref),
[`load_settings!`](@ref), [`save_settings`](@ref),
[`settings_modified`](@ref), [`set_step!`](@ref), [`test_pair!`](@ref),
[`test_stale`](@ref), [`start_run!`](@ref), [`cancel_run!`](@ref),
[`open_results!`](@ref), [`step_status`](@ref)) work on any workflow.
"""
abstract type AbstractWorkflow end

const STEP_LABELS = Dict(:images => "Images", :calibration => "Calibration",
                         :prepare => "Prepare", :passes => "Passes",
                         :test => "Test pair", :run => "Run", :results => "Results")

"""
    step_label(wf::AbstractWorkflow, step) -> String

The step's name on the rail (the planar Passes step reads "Particles" in
the particle modes).
"""
step_label(::AbstractWorkflow, step::Symbol) = STEP_LABELS[step]

"""
    workflow_steps(wf::AbstractWorkflow) -> Tuple{Vararg{Symbol}}

The workflow's steps, in order (`WORKFLOW_STEPS` for a `PlanarWorkflow`,
`STEREO_WORKFLOW_STEPS` for a `StereoWorkflow`).
"""
function workflow_steps end

"""
    prepare_pages(wf::AbstractWorkflow) -> Tuple{Vararg{Symbol}}

The workflow's Prepare sub-pages, in order (`PREPARE_PAGES` for a
`PlanarWorkflow`, `STEREO_PREPARE_PAGES` for a `StereoWorkflow`).
"""
function prepare_pages end

"""
    workflow_problem(wf::AbstractWorkflow) -> Union{Nothing,String}

`nothing` when the workflow can test and run, otherwise a message naming
the first missing or inconsistent input.
"""
function workflow_problem end

# Where background computations run: inline, or (with `spawn[]`) on a worker
# whose result is applied through `deliver[]` on the GUI thread.
function _workflow_runner(spawn::Base.RefValue{Bool}, deliver::Base.RefValue{Any})
    return function (job, apply)
        d = deliver[]
        _run_job(() -> (out = _try_job(job); d(() -> apply(out))), spawn[])
        return nothing
    end
end

# A finished batch becomes the Results step's data.
function _connect_results!(wf::AbstractWorkflow)
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

# ---------------------------------------------------------------- settings

_check_recipe(::AbstractWorkflow, ::PIVRecipe) = nothing

_copy_steps(steps::Vector{PreprocessStep}) = copy(steps)
_copy_steps(steps::Tuple) = map(copy, steps)

# The preprocessing list the Prepare preview edits.
_edited_steps(wf::AbstractWorkflow) = wf.preprocessing[]
_set_edited_steps!(wf::AbstractWorkflow, steps) = (wf.preprocessing[] = steps; wf)

_load_specific!(::AbstractWorkflow, ::PIVRecipe) = nothing

"""
    load_settings!(wf::AbstractWorkflow, recipe_or_path)

Open settings from a `PIVRecipe`, a recipe file, or a results file written
by a run. Every recipe field is kept, so `workflow_recipe(wf)` returns an
equal recipe until something is edited. A `StereoWorkflow` rejects a recipe
with an ROI (`ArgumentError`, settings unchanged).
"""
load_settings!(wf::AbstractWorkflow, path::AbstractString) =
    (load_settings!(wf, load_recipe(path)); wf.settings_path[] = String(path); wf)

function load_settings!(wf::AbstractWorkflow, r::PIVRecipe)
    _check_recipe(wf, r)
    wf.preprocessing[] = _copy_steps(r.preprocessing)
    wf.mask[] = r.mask === nothing ? nothing : copy(r.mask)
    wf.scale[] = r.scale
    wf.predictor_smoothing[] = r.predictor_smoothing
    wf.mask_threshold[] = r.mask_threshold
    wf.passes.mode[] = r.mode
    set_image_type!(wf.passes, r.image_type)
    _load_specific!(wf, r)               # may resize a linked preset; replaced next
    load_passes!(wf.passes, r.passes)
    wf.saved[] = workflow_recipe(wf)
    wf.settings_path[] = ""
    wf.status[] = "opened settings: $(length(r.passes)) passes"
    return wf
end

"""
    save_settings(wf::AbstractWorkflow, path) -> path

Save the current settings with `save_recipe`.
"""
function save_settings(wf::AbstractWorkflow, path::AbstractString)
    r = workflow_recipe(wf)
    save_recipe(path, r)
    wf.saved[] = r
    wf.settings_path[] = String(path)
    wf.status[] = "saved settings to $(basename(path))"
    return path
end

"""
    settings_modified(wf::AbstractWorkflow) -> Bool

Whether the settings differ from the last opened or saved recipe (always
`true` before the first save).
"""
settings_modified(wf::AbstractWorkflow) = wf.saved[] === nothing || wf.saved[] != workflow_recipe(wf)

"""
    set_step!(wf::AbstractWorkflow, step)

Show a step (one of [`workflow_steps`](@ref)`(wf)`).
"""
function set_step!(wf::AbstractWorkflow, step::Symbol)
    step in workflow_steps(wf) || throw(ArgumentError("unknown step :$step"))
    wf.step[] == step || (wf.step[] = step)
    return wf
end

# ---------------------------------------------------------------- test and run

# Pairs a test analyzes: the representative pair, or the first few pairs for
# an ensemble (one pair says little about an ensemble's yield).
const ENSEMBLE_TEST_PAIRS = 10

function _test_pairs(fs::FrameSet, mode::Symbol)
    if mode === :ensemble
        prs = frame_pairs(fs)
        return prs[1:min(end, ENSEMBLE_TEST_PAIRS)]
    end
    pr = current_pair(fs)
    return pr === nothing ? Any[] : Any[pr]
end

"""
    test_pair!(wf::AbstractWorkflow; spawn = true)

Analyze the representative pair with the current settings, through the same
`apply_recipe` call the batch uses (the stereo form for a `StereoWorkflow`).
"""
function test_pair!(wf::AbstractWorkflow; spawn::Bool = true)
    msg = workflow_problem(wf)
    msg === nothing || (wf.test.status[] = msg; return wf)
    recipe = try
        workflow_recipe(wf)
    catch err
        wf.test.status[] = _errmsg(err)
        return wf
    end
    start_test!(wf.test, recipe, _test_inputs(wf), _test_label(wf);
                deliver = wf.deliver[], spawn)
    return wf
end

# Whether a test's pairs differ from the pairs a test would analyze now
# (frames are compared by identity, paths by value).
function _pairs_changed(tested, current)
    length(tested) == length(current) || return true
    return !all(((p, q),) -> _same_entry(p[1], q[1]) && _same_entry(p[2], q[2]),
                zip(tested, current))
end

function _test_pairs_changed(wf::AbstractWorkflow, k::Int, fs::FrameSet)
    inp = wf.test.inputs[]
    inp === nothing && return true
    current = try
        _test_pairs(fs, wf.passes.mode[])
    catch
        return true
    end
    return _pairs_changed(inp[k], current)
end

"""
    test_stale(wf::AbstractWorkflow) -> Bool

Whether the settings or the test's inputs (the representative pair, or for
an ensemble the leading pairs; for a stereo workflow also the dewarpers)
changed since the last test.
"""
test_stale(wf::AbstractWorkflow) =
    wf.test.recipe[] === nothing || wf.test.recipe[] != workflow_recipe(wf) || _inputs_stale(wf)

"""
    start_run!(wf::AbstractWorkflow; spawn = true)

Run the batch on all pairs with a snapshot of the current settings: one
result per pair for a `:sequence` recipe, one pooled result for an
`:ensemble` recipe (see [`RunState`](@ref)).
"""
function start_run!(wf::AbstractWorkflow; spawn::Bool = true)
    msg = workflow_problem(wf)
    msg === nothing || (wf.run.status[] = msg; return wf)
    recipe = try
        workflow_recipe(wf)
    catch err
        wf.run.status[] = _errmsg(err)
        return wf
    end
    start_run!(wf.run, recipe, _run_inputs(wf); deliver = wf.deliver[], spawn)
    return wf
end

cancel_run!(wf::AbstractWorkflow) = cancel_run!(wf.run)

"""
    pair_position(wf::AbstractWorkflow) -> (index, count)

What the window's pair bar shows: on the Results step the result being
viewed and the number of results (when results are open), otherwise the
representative pair and the pair count. The two positions are independent.
"""
function pair_position(wf::AbstractWorkflow)
    ex = wf.explorer[]
    wf.step[] === :results && ex !== nothing && return (ex.frame[], nframes(ex))
    return (_representative_pair(wf), npairs(wf))
end


"""
    go_to_pair!(wf::AbstractWorkflow, i)
    step_pair!(wf::AbstractWorkflow, delta)

Move the pair bar (see [`pair_position`](@ref)) to position `i`, or by
`delta`: on the Results step this changes the result shown, elsewhere the
representative pair.
"""
function go_to_pair!(wf::AbstractWorkflow, i::Integer)
    ex = wf.explorer[]
    if wf.step[] === :results && ex !== nothing
        set_frame!(ex, i)
    else
        select_pair!(_pair_target(wf), i)
    end
    return wf
end

step_pair!(wf::AbstractWorkflow, delta::Integer) =
    go_to_pair!(wf, clamp(first(pair_position(wf)) + delta, 1, max(last(pair_position(wf)), 1)))



"""
    open_results!(wf::AbstractWorkflow, path)

Browse a results file in the Results step (entries load on demand).
"""
function open_results!(wf::AbstractWorkflow, path::AbstractString)
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

# ---------------------------------------------------------------- step rail

"""
    step_status(wf::AbstractWorkflow, step) -> (state, summary)

`state` is `:todo`, `:ok`, `:attention`, or `:busy`, with a one-line summary
for the step rail.
"""
function step_status(wf::AbstractWorkflow, step::Symbol)
    step in workflow_steps(wf) || throw(ArgumentError("unknown step :$step"))
    if step === :passes
        return _passes_status(wf)
    elseif step === :test
        wf.test.running[] && return (:busy, "testing…")
        s = test_summary(wf.test)
        s === nothing && return (:todo, "not tested")
        txt = test_brief(s)
        inp = wf.test.inputs[]
        wf.test.recipe[].mode === :ensemble && inp !== nothing &&
            (txt = "ensemble of $(length(first(inp))) pairs: " * txt)
        return test_stale(wf) ? (:attention, "settings changed since: " * txt) : (:ok, txt)
    elseif step === :run
        wf.run.running[] && return (:busy, run_progress(wf.run))
        return isempty(wf.run.status[]) ? (:todo, "not run") : (:ok, wf.run.status[])
    elseif step === :results
        ex = wf.explorer[]
        ex === nothing && return (:todo, "no results yet")
        n = nframes(ex)
        return (:ok, wf.results_path[] === nothing ? "$n result" * (n == 1 ? "" : "s") * " in memory" :
                     basename(wf.results_path[]))
    end
    return _step_status(wf, step)
end

_passes_status(wf::AbstractWorkflow) = _piv_passes_status(wf)

function _piv_passes_status(wf::AbstractWorkflow)
    isempty(wf.passes.error[]) || return (:attention, wf.passes.error[])
    return (:ok, passes_summary(wf.passes))
end

# The Prepare step's summary parts shared by both workflows.
function _prepare_parts(wf::AbstractWorkflow)
    parts = String[]
    pre = wf.preprocessing[]
    if pre isa Tuple
        push!(parts, "preprocessing per camera ($(length(pre[1])) + $(length(pre[2])) steps)")
    else
        n = length(pre)
        n == 0 || push!(parts, "$n preprocessing step" * (n == 1 ? "" : "s"))
    end
    wf.mask[] === nothing || push!(parts, "mask")
    return parts
end
