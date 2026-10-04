# The recording type a window session works on: one camera (planar) or two
# (stereo). A window holds one session at a time; changing the type starts a
# fresh session of the other type, after the user confirms what it discards.

"""
Recording types of a window session: `:planar` (one camera, a
[`PlanarWorkflow`](@ref)) and `:stereo` (two cameras, a
[`StereoWorkflow`](@ref)).
"""
const RECORDING_TYPES = (:planar, :stereo)

"""
    recording_type(wf::AbstractWorkflow) -> Symbol
    recording_type(recipe::PIVRecipe) -> Union{Nothing,Symbol}
    recording_type(path) -> Union{Nothing,Symbol}

`:planar` or `:stereo`. A recipe is stereo when it preprocesses each camera
separately and planar when it has a region or a particle mode; otherwise it
fits either (`nothing`). A file is stereo when it holds a camera calibration
(a stereo results file or a saved calibration), else typed by its recipe;
`nothing` when it holds neither.
"""
recording_type(::PlanarWorkflow) = :planar
recording_type(::StereoWorkflow) = :stereo

function recording_type(r::PIVRecipe)
    r.preprocessing isa Tuple && return :stereo
    (r.roi !== nothing || _particle_mode(r.mode)) && return :planar
    return nothing
end

function recording_type(path::AbstractString)
    # a TOML settings file holds no calibration
    calibrated = lowercase(splitext(path)[2]) != ".toml" && isfile(path) && try
        load_calibration(path)
        true
    catch
        false
    end
    calibrated && return :stereo
    recipe = try
        load_recipe(path)
    catch
        return nothing
    end
    return recording_type(recipe)
end

"""
    new_workflow(type::Symbol) -> AbstractWorkflow

A fresh session of recording type `type` (one of [`RECORDING_TYPES`](@ref)).
"""
function new_workflow(type::Symbol)
    type in RECORDING_TYPES ||
        throw(ArgumentError("recording type must be :planar or :stereo, got :$type"))
    return type === :stereo ? StereoWorkflow() : PlanarWorkflow()
end

"""
    unsaved_work(wf::AbstractWorkflow) -> Vector{String}

What starting a fresh session would discard: the frames, settings not saved
to a file, a camera calibration, results kept only in memory, and a test or
run in progress. Empty for an untouched session.
"""
function unsaved_work(wf::AbstractWorkflow)
    lost = String[]
    n = _frame_count(wf)
    n > 0 && push!(lost, "$n frame" * (n == 1 ? "" : "s"))
    unsaved = if wf.saved[] === nothing
        workflow_recipe(wf) != workflow_recipe(new_workflow(recording_type(wf)))
    else
        settings_modified(wf)
    end
    unsaved && push!(lost, "settings not saved to a file")
    _has_calibration(wf) && push!(lost, "the camera calibration")
    results_in_memory(wf) && push!(lost, "results kept only in memory")
    (wf.test.running[] || wf.run.running[]) && push!(lost, "the test or run in progress")
    return lost
end

_frame_count(wf::PlanarWorkflow) = length(wf.frames.files[])
_frame_count(wf::StereoWorkflow) = length(wf.frames1.files[]) + length(wf.frames2.files[])
_has_calibration(::PlanarWorkflow) = false
_has_calibration(wf::StereoWorkflow) =
    wf.calibration.dewarpers[] !== nothing || any(p -> !isempty(p[]), wf.calibration.plates)

"""
    switch_question(wf::AbstractWorkflow, type::Symbol) -> Union{Nothing,String}

The confirmation to show before replacing `wf` with a fresh session of
recording type `type`, or `nothing` when nothing would be lost.
"""
function switch_question(wf::AbstractWorkflow, type::Symbol)
    lost = unsaved_work(wf)
    isempty(lost) && return nothing
    name = type === :stereo ? "two-camera (stereo)" : "one-camera (planar)"
    return "Start a new $name session? This discards " * _join_list(lost) * "."
end

_join_list(items) = length(items) <= 2 ? join(items, " and ") :
                    join(items[1:end-1], ", ") * ", and " * items[end]
