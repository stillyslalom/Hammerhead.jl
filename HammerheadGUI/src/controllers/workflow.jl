# State and steps shared by the workflow windows (planar and stereo): the
# settings ↔ recipe round trip, the Passes (with the pair test) / Run / Results steps,
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
#   _input_kwargs(wf, which)    apply_recipe keyword inputs for :test / :run (planar: masks)
#   _test_label(wf)             the representative pair's index
#   _inputs_stale(wf)           whether the test's inputs changed (pair, dewarpers)
#   _step_status(wf, step)      rail status of the steps that are not shared
#   _has_pairs(wf)              whether frames forming pairs were added
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
                         :run => "Run", :results => "Results")

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

_input_kwargs(wf::AbstractWorkflow, ::Symbol) = _backend_kw(wf)

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
    _connect_result_images!(wf)
    on(wf.run.running) do running
        running && return
        out = wf.run.finished_output[]
        if out !== nothing
            open_results!(wf, out)
        elseif !isempty(wf.run.completed[])
            wf.results_path[] = nothing
            wf.explorer[] = ResultExplorer(collect(wf.run.completed[]))
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
                deliver = wf.deliver[], spawn, options = _input_kwargs(wf, :test))
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
    wf.test.recipe[] === nothing || wf.test.recipe[] != workflow_recipe(wf) ||
    wf.test.options[] != _input_kwargs(wf, :test) || _inputs_stale(wf)

"""
    run_stale(wf::AbstractWorkflow) -> Bool

Whether the settings or the inputs (frames, mask images, backend; for a
stereo workflow the dewarpers) changed since the last run. `false` before
the first run.
"""
function run_stale(wf::AbstractWorkflow)
    rs = wf.run
    rs.recipe[] === nothing && return false
    rs.recipe[] != workflow_recipe(wf) && return true
    rs.options[] != _input_kwargs(wf, :run) && return true
    current = try
        _run_inputs(wf)
    catch
        return true
    end
    inp = rs.inputs[]
    (inp === nothing || length(inp) != length(current)) && return true
    return !all(((a, b),) -> _same_input(a, b), zip(inp, current))
end

_same_input(a::AbstractVector, b::AbstractVector) =
    length(a) == length(b) && all(((x, y),) -> _same_input(x, y), zip(a, b))
_same_input(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(((x, y),) -> _same_input(x, y), zip(a, b))
_same_input(a, b) = _same_entry(a, b)

# Whether the Results step shows the last run's results (in memory, or the
# file it wrote), rather than a results file opened separately.
_results_from_run(wf::AbstractWorkflow) =
    wf.explorer[] !== nothing && wf.run.recipe[] !== nothing &&
    (wf.results_path[] === nothing || wf.results_path[] == wf.run.finished_output[])

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
    start_run!(wf.run, recipe, _run_inputs(wf); deliver = wf.deliver[], spawn,
               options = _input_kwargs(wf, :run))
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
    wf.results_path[] = String(path)        # before the explorer: its listeners read it
    wf.explorer[] = ex
    wf.status[] = "results: $(basename(path))"
    return wf
end

"""
    save_run_results!(wf::AbstractWorkflow, path) -> path

Write the results of the last run kept in memory (no output file was set)
to `path`, with the settings that produced them, the frame paths of each
pair (when the frames are files), and for stereo the calibration, as a run
with an output file would. The Results step then refers to the file.
"""
function save_run_results!(wf::AbstractWorkflow, path::AbstractString)
    rs = wf.run
    rs.running[] && throw(ArgumentError("wait for the run to finish"))
    isempty(rs.completed[]) && throw(ArgumentError("there are no results in memory to save"))
    results = collect(Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}, rs.completed[])
    inputs = rs.inputs[]
    calibration = inputs !== nothing && length(inputs) == 4 ? (inputs[3], inputs[4]) : nothing
    save_results(path, results; recipe = rs.recipe[], calibration,
                 sources = _run_sources(rs, inputs, length(results)))
    rs.finished_output[] = String(path)
    rs.output_path[] = String(path)
    wf.results_path[] = String(path)
    n = length(results)
    wf.status[] = "saved $n result" * (n == 1 ? "" : "s") * " to $(basename(path))"
    return path
end

# Frame labels per result of an in-memory run (pooled results have none).
function _run_sources(rs::RunState, inputs, n::Int)
    (inputs === nothing || !(rs.mode[] in (:sequence, :ptv))) && return nothing
    return map(1:n) do i
        entries = length(inputs) == 4 ? (inputs[1][i]..., inputs[2][i]...) : Tuple(inputs[1][i])
        all(e -> e isa AbstractString, entries) ? String[e for e in entries] : String[]
    end
end

"""
    results_in_memory(wf::AbstractWorkflow) -> Bool

Whether the last run's results exist only in memory (no output file), so
[`save_run_results!`](@ref) can still write them.
"""
results_in_memory(wf::AbstractWorkflow) =
    !wf.run.running[] && !isempty(wf.run.completed[]) && wf.run.finished_output[] === nothing

# ---------------------------------------------------------------- particle images

# The frame entries (A, B) behind each shown result, or `nothing` when the
# workflow cannot tell (per workflow; stereo results have none).
_result_frame_entries(::AbstractWorkflow) = nothing

# The Results step's particle image: the explorer offers the `:image` field
# when the results map to frame pairs, and the shown frame (A/B) of the
# shown result loads on the runner.
function _connect_result_images!(wf::AbstractWorkflow)
    runner = _workflow_runner(wf.spawn, wf.deliver)
    on(wf.explorer) do ex
        ex === nothing && return
        entries = try
            _result_frame_entries(wf)
        catch
            nothing
        end
        ex.image_available[] = entries !== nothing && any(!isnothing, entries)
        generation = Ref(0)
        function load(_...)
            (ex.field[] === :image && entries !== nothing) || return
            i = ex.frame[]
            e = i <= length(entries) ? entries[i] : nothing
            g = (generation[] += 1)
            if e === nothing
                ex.image[] = nothing
                return
            end
            entry = ex.image_frame[] === :a ? e[1] : e[2]
            job = () -> entry isa AbstractString ? load_image(Float32, entry) : Float32.(entry)
            runner(job, out -> (g == generation[] && (ex.image[] = out.err === nothing ? out.value : nothing)))
        end
        onany(load, ex.frame, ex.field, ex.image_frame)
        load()
    end
    return wf
end

"""
    switch_frame!(wf::AbstractWorkflow, which::Symbol)

Show frame `:a` or `:b`: of the representative pair, or on the Results step
of the particle image under the shown result.
"""
function switch_frame!(wf::AbstractWorkflow, which::Symbol)
    which in (:a, :b) || throw(ArgumentError("frame must be :a or :b, got :$which"))
    ex = wf.explorer[]
    if wf.step[] === :results && ex !== nothing
        ex.image_frame[] == which || (ex.image_frame[] = which)
    else
        show_frame!(_pair_target(wf), which)
    end
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
        state, txt = _until_pairs(wf, _passes_status(wf))
        state === :ok || return (state, txt)
        wf.test.running[] && return (:busy, "testing…")
        s = test_summary(wf.test)
        s === nothing && return (:ok, txt)
        test_stale(wf) && return (:attention, txt * " · settings changed since the test")
        return (:ok, txt * " · " * test_brief(s))
    elseif step === :run
        rs = wf.run
        rs.running[] && return (:busy, run_progress(rs))
        isempty(rs.status[]) && return (:todo, "not run")
        run_stale(wf) && return (:attention, "settings changed since: " * rs.status[])
        ok = startswith(rs.status[], "done")
        return (ok ? :ok : :attention, rs.status[])
    elseif step === :results
        ex = wf.explorer[]
        ex === nothing && return (:todo, "no results yet")
        n = nframes(ex)
        txt = wf.results_path[] === nothing ? "$n result" * (n == 1 ? "" : "s") * " in memory" :
              basename(wf.results_path[])
        _results_from_run(wf) && run_stale(wf) &&
            return (:attention, "settings changed since the run: " * txt)
        return (:ok, txt)
    end
    s = _step_status(wf, step)
    return step === :prepare ? _until_pairs(wf, s) : s
end

# Prepare and Passes settings that are fine stay grey until there are frames
# to apply them to: green means ready to analyze the frames listed on Images.
_until_pairs(wf::AbstractWorkflow, (state, txt)) =
    state === :ok && !_has_pairs(wf) ? (:todo, txt) : (state, txt)

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
