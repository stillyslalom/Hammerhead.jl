# Batch-runner controller: PIVParameters form state plus sequence-execution
# state (file list, output path, progress, cancellation). Framework-free —
# the run itself goes through Hammerhead.run_piv_sequence with a progress
# callback, launched as a cooperative task so a GL render loop stays live.

"""
    BatchCancelled()

Signals cancellation between frame pairs. Completed pairs remain in
`BatchRunner.completed` and in the output file when `output_path` is set.
"""
struct BatchCancelled <: Exception end

"""
    BatchRunner(; kwargs...)

Configure a planar PIV sequence and observe its progress. Inputs, options,
and run state are `Observables`.

# Keyword defaults
- `files = Any[]` (frame paths and/or in-memory matrices), `pair_mode = :paired`
- `effort = :custom` — `:custom` runs the manual window schedule below;
  `:low` / `:medium` / `:high` use [`run_piv_sequence`](@ref)'s effort presets
  and ignore the schedule/option widgets; `:saved` runs the exact pass
  schedule loaded with [`load_settings!`](@ref).
- `window_schedule = [64, 32, 32]`, `overlap_fraction = 0.5`
- `correlation_method = :cross`, `padding = true`, `apodization = :gauss`
  `subpixel_method = :gauss3`,
  `uncertainty = false`
- `pixel_size = 1.0`, `dt = 1.0`, `length_unit = "px"`, `time_unit = "frame"` —
  a [`PhysicalScale`](@ref) is attached to the outputs only when any of these
  differs from its default.
- `output_path = ""` (empty = keep results in memory only), `mask = nothing`
- `roi = nothing` (full image), or a core `ROI` / `(rows, cols)` range tuple.
  Preprocessing receives full frames; the core crops the images and mask,
  retaining original image coordinates in the results.

Drive it with [`add_files!`](@ref), [`set_schedule!`](@ref),
[`set_effort!`](@ref), [`set_preprocess!`](@ref) (an optional per-frame
preprocessing pipeline, e.g. from a [`PreprocessPreview`](@ref)),
[`start!`](@ref) and [`cancel!`](@ref). [`save_settings`](@ref) and
[`load_settings!`](@ref) store and restore the form as a core `PIVRecipe`;
runs with an output file record their recipe there too. Watch
`progress` (`(done, total)`), `status`, `running`, `results` (a
`Vector{PIVResult}` after a completed run, `nothing` before), and
`completed` (finished pairs appended during a run; feed them to a
[`ResultExplorer`](@ref) via
[`push_result!`](@ref) to browse a batch in progress).
"""
struct BatchRunner
    files::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    output_path::Observable{String}
    mask::Observable{Union{Nothing,BitMatrix}}
    roi::Observable{Union{Nothing,ROI}}
    effort::Observable{Symbol}
    window_schedule::Observable{Vector{Int}}
    overlap_fraction::Observable{Float64}
    correlation_method::Observable{Symbol}
    padding::Observable{Bool}
    apodization::Observable{Symbol}
    subpixel_method::Observable{Symbol}
    uncertainty::Observable{Bool}
    pixel_size::Observable{Float64}
    dt::Observable{Float64}
    length_unit::Observable{String}
    time_unit::Observable{String}
    running::Observable{Bool}
    cancel::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    results::Observable{Union{Nothing,Vector{PIVResult}}}
    completed::Observable{Vector{PIVResult}}
    preprocess::Observable{Union{Nothing,Function}}
    preprocess_steps::Observable{Union{Nothing,Vector{PreprocessStep}}}
    saved_passes::Observable{Union{Nothing,Vector{PIVParameters}}}
end

function BatchRunner(; files = Any[], pair_mode::Symbol = :paired,
                     output_path::AbstractString = "", mask = nothing, roi = nothing,
                     effort::Symbol = :custom,
                     window_schedule::AbstractVector{<:Integer} = [64, 32, 32],
                     overlap_fraction::Real = 0.5,
                     correlation_method::Symbol = :cross,
                     padding::Bool = true, apodization::Symbol = :gauss,
                     subpixel_method::Symbol = :gauss3,
                     uncertainty::Bool = false,
                     pixel_size::Real = 1.0, dt::Real = 1.0,
                     length_unit::AbstractString = "px",
                     time_unit::AbstractString = "frame")
    return BatchRunner(Observable{Vector{Any}}(collect(Any, files)),
                       Observable(pair_mode), Observable(String(output_path)),
                       Observable{Union{Nothing,BitMatrix}}(mask === nothing ? nothing : BitMatrix(mask)),
                       Observable{Union{Nothing,ROI}}(_as_roi(roi)),
                       Observable(effort),
                       Observable(collect(Int, window_schedule)),
                       Observable(Float64(overlap_fraction)),
                       Observable(correlation_method), Observable(padding),
                       Observable(apodization), Observable(subpixel_method),
                       Observable(uncertainty),
                       Observable(Float64(pixel_size)), Observable(Float64(dt)),
                       Observable(String(length_unit)), Observable(String(time_unit)),
                       Observable(false), Observable(false),
                       Observable((0, 0)), Observable(""),
                       Observable{Union{Nothing,Vector{PIVResult}}}(nothing),
                       Observable(PIVResult[]),
                       Observable{Union{Nothing,Function}}(nothing),
                       Observable{Union{Nothing,Vector{PreprocessStep}}}(PreprocessStep[]),
                       Observable{Union{Nothing,Vector{PIVParameters}}}(nothing))
end

function Base.show(io::IO, bc::BatchRunner)
    print(io, "BatchRunner($(length(bc.files[])) frames, schedule ",
          bc.window_schedule[], bc.running[] ? ", running)" : ")")
end

"""
    add_files!(bc::BatchRunner, entries)

Append frames (file paths and/or matrices) to the frame list.
"""
function add_files!(bc::BatchRunner, entries)
    append!(bc.files[], entries)
    notify(bc.files)
    return bc
end

"""
    clear_files!(bc::BatchRunner)

Empty the frame list.
"""
clear_files!(bc::BatchRunner) = (empty!(bc.files[]); notify(bc.files); bc)

"""
    set_roi!(bc::BatchRunner, roi)
    set_roi!(bc::BatchRunner, row_first, row_last, col_first, col_last)

Set the core ROI for every batch pair (or `nothing` for the full image).
Accept a `ROI`, range tuple, or inclusive integer/string bounds. Bounds are
checked against the first frame when available; each pair is checked again
by the core during processing. Changing ROI during a run affects the next run.
"""
function set_roi!(bc::BatchRunner, roi)
    rr = _as_roi(roi)
    if rr !== nothing && !isempty(bc.files[])
        frame = first(bc.files[])
        image = frame isa AbstractMatrix ? frame : load_image(frame)
        _check_roi(image, rr)
    end
    bc.roi[] = rr
    return bc
end
set_roi!(bc::BatchRunner, r1, r2, c1, c2) =
    set_roi!(bc, ROI(_roi_index(r1):_roi_index(r2), _roi_index(c1):_roi_index(c2)))

"""
    clear_roi!(bc::BatchRunner)

Reset the next batch run to use the full image.
"""
clear_roi!(bc::BatchRunner) = set_roi!(bc, nothing)

"""
    apply_roi!(bc::BatchRunner, editor::ROIEditor)

Copy the editor's completed selection to the batch. Complete a pending
second corner first; clearing the editor and applying resets the batch ROI.
"""
function apply_roi!(bc::BatchRunner, ed::ROIEditor)
    ed.anchor[] === nothing || throw(ArgumentError("click the opposite ROI corner first"))
    return set_roi!(bc, ed.roi[])
end

"""
    frame_pairs(bc::BatchRunner)

The correlation pairs of the current frame list and pairing mode
(`Hammerhead.image_pairs`; throws on an odd `:paired` frame count).
"""
frame_pairs(bc::BatchRunner) = image_pairs(bc.files[]; mode = bc.pair_mode[])

"""
    parse_schedule(str) -> Vector{Int}

Parse a window-schedule entry: positive integers separated by commas and/or
spaces, e.g. `"64, 32, 32"`. Throws `ArgumentError` on anything else.
"""
function parse_schedule(str::AbstractString)
    tokens = split(str, r"[,\s]+"; keepempty = false)
    isempty(tokens) && throw(ArgumentError("empty window schedule"))
    sizes = Int[]
    for t in tokens
        n = tryparse(Int, t)
        (n === nothing || n <= 0) &&
            throw(ArgumentError("window sizes must be positive integers, got \"$t\""))
        push!(sizes, n)
    end
    return sizes
end

"""
    set_schedule!(bc::BatchRunner, schedule)

Set the multi-pass window schedule from a vector of sizes or a string
(see [`parse_schedule`](@ref)).
"""
set_schedule!(bc::BatchRunner, s::AbstractString) = set_schedule!(bc, parse_schedule(s))
function set_schedule!(bc::BatchRunner, s::AbstractVector{<:Integer})
    isempty(s) && throw(ArgumentError("empty window schedule"))
    bc.window_schedule[] = collect(Int, s)
    return bc
end

const EFFORT_LEVELS = (:custom, :low, :medium, :high, :saved)

"""
    set_effort!(bc::BatchRunner, effort::Symbol)

Set the analysis effort. `:custom` uses the manual window schedule and
option widgets; `:low` / `:medium` / `:high` use [`run_piv_sequence`](@ref)'s
effort presets and ignore the manual schedule; `:saved` uses the schedule from
the last [`load_settings!`](@ref).
"""
function set_effort!(bc::BatchRunner, effort::Symbol)
    effort in EFFORT_LEVELS ||
        throw(ArgumentError("effort must be one of $(EFFORT_LEVELS), got :$effort"))
    bc.effort[] = effort
    return bc
end

# Parse a positive-number textbox entry (pixel size / dt).
function _parse_positive(str::AbstractString, what::AbstractString)
    v = tryparse(Float64, strip(str))
    (v === nothing || !(isfinite(v) && v > 0)) &&
        throw(ArgumentError("$what must be a positive number, got \"$str\""))
    return v
end

"""
    set_pixel_size!(bc::BatchRunner, value)
    set_dt!(bc::BatchRunner, value)

Set the physical `pixel_size` / frame interval `dt` from a positive number or
its string form (see [`set_scale!`](@ref)).
"""
set_pixel_size!(bc::BatchRunner, v::Real) =
    (v > 0 || throw(ArgumentError("pixel_size must be positive")); bc.pixel_size[] = Float64(v); bc)
set_pixel_size!(bc::BatchRunner, s::AbstractString) =
    (bc.pixel_size[] = _parse_positive(s, "pixel size"); bc)
set_dt!(bc::BatchRunner, v::Real) =
    (v > 0 || throw(ArgumentError("dt must be positive")); bc.dt[] = Float64(v); bc)
set_dt!(bc::BatchRunner, s::AbstractString) =
    (bc.dt[] = _parse_positive(s, "dt"); bc)

"""
    set_scale!(bc::BatchRunner; pixel_size, dt, length_unit, time_unit)

Set any of the physical-scale form fields at once; omitted fields are left
unchanged. Positive numbers are required for `pixel_size`/`dt`.
"""
function set_scale!(bc::BatchRunner; pixel_size = nothing, dt = nothing,
                    length_unit = nothing, time_unit = nothing)
    pixel_size === nothing || set_pixel_size!(bc, pixel_size)
    dt === nothing || set_dt!(bc, dt)
    length_unit === nothing || (bc.length_unit[] = String(length_unit))
    time_unit === nothing || (bc.time_unit[] = String(time_unit))
    return bc
end

"""
    set_preprocess!(bc::BatchRunner, pp)

Attach a preprocessing pipeline applied to every frame of the batch: a
[`PreprocessPreview`](@ref) controller (its [`build_preprocess`](@ref)
closure is snapshotted now), a bare function `img -> img′`, or `nothing` to
clear.
"""
function set_preprocess!(bc::BatchRunner, ::Nothing)
    bc.preprocess_steps[] = PreprocessStep[]
    bc.preprocess[] = nothing
    bc
end
# A bare function runs, but cannot be saved with the settings.
function set_preprocess!(bc::BatchRunner, f::Function)
    bc.preprocess_steps[] = nothing
    bc.preprocess[] = f
    bc
end
function set_preprocess!(bc::BatchRunner, pp::PreprocessPreview)
    bc.preprocess_steps[] = preprocess_steps(pp)
    bc.preprocess[] = build_preprocess(pp)
    bc
end

"""
    build_scale(bc::BatchRunner) -> Union{Nothing,PhysicalScale}

The [`PhysicalScale`](@ref) attached to the batch outputs, or `nothing` when
every scale field is at its default (`pixel_size = dt = 1`, units `px`/`frame`)
so the results stay in pixel/frame units.
"""
function build_scale(bc::BatchRunner)
    (bc.pixel_size[] == 1.0 && bc.dt[] == 1.0 &&
     bc.length_unit[] == "px" && bc.time_unit[] == "frame") && return nothing
    return PhysicalScale(bc.pixel_size[], bc.dt[], bc.length_unit[], bc.time_unit[])
end

"""
    build_parameters(bc::BatchRunner) -> Vector{PIVParameters}

The multi-pass schedule for the current form state
(`Hammerhead.multipass_parameters`; propagates `PIVParameters` validation
errors). Only relevant when `effort == :custom`.
"""
build_parameters(bc::BatchRunner) =
    multipass_parameters(bc.window_schedule[];
                         overlap_fraction = bc.overlap_fraction[],
                         correlation_method = bc.correlation_method[],
                         padding = bc.padding[],
                         apodization = bc.apodization[],
                         subpixel_method = bc.subpixel_method[],
                         uncertainty = bc.uncertainty[])

# Image size the passes run on: the ROI, or the first frame.
function _analysis_size(bc::BatchRunner)
    bc.roi[] === nothing || return (length(bc.roi[].rows), length(bc.roi[].cols))
    frame = first(bc.files[])
    return size(frame isa AbstractMatrix ? frame : load_image(frame))
end

function _batch_passes(bc::BatchRunner)
    bc.effort[] === :custom && return build_parameters(bc)
    if bc.effort[] === :saved
        bc.saved_passes[] === nothing && throw(ArgumentError("no saved pass schedule is loaded"))
        return bc.saved_passes[]
    end
    isempty(bc.files[]) && throw(ArgumentError("add frames to size the $(bc.effort[]) preset"))
    return Hammerhead.effort_schedule(bc.effort[]; image_size = _analysis_size(bc))
end

"""
    batch_recipe(bc::BatchRunner) -> PIVRecipe

The current form settings as a core `PIVRecipe`: pass schedule, preprocessing,
mask, ROI and physical scale. Effort presets are expanded for the first
frame's (or ROI's) size. Preprocessing set as a bare function cannot be saved.
"""
function batch_recipe(bc::BatchRunner)
    steps = bc.preprocess_steps[]
    steps === nothing &&
        throw(ArgumentError("custom preprocessing functions cannot be saved; use the preprocess window"))
    PIVRecipe(_batch_passes(bc); preprocessing = steps, mask = bc.mask[],
              roi = bc.roi[], scale = build_scale(bc))
end

"""
    save_settings(bc::BatchRunner, path) -> path

Save the form settings with `Hammerhead.save_recipe`.
"""
save_settings(bc::BatchRunner, path::AbstractString) = save_recipe(path, batch_recipe(bc))

"""
    load_settings!(bc::BatchRunner, recipe_or_path)

Load a `PIVRecipe` (or a file saved with `save_recipe`, or a results file
written by a run) into the form. Its exact pass schedule becomes the `:saved`
effort; mask, ROI, physical scale and preprocessing replace the current ones.
"""
load_settings!(bc::BatchRunner, path::AbstractString) = load_settings!(bc, load_recipe(path))
function load_settings!(bc::BatchRunner, recipe::PIVRecipe)
    recipe.mode === :sequence ||
        throw(ArgumentError("the batch form runs sequences; this recipe is :$(recipe.mode)"))
    bc.saved_passes[] = copy(recipe.passes)
    bc.window_schedule[] = [p.window_size[1] for p in recipe.passes]
    bc.effort[] = :saved
    bc.mask[] = recipe.mask === nothing ? nothing : copy(recipe.mask)
    bc.roi[] = recipe.roi
    sc = recipe.scale
    sc === nothing ?
        set_scale!(bc; pixel_size = 1.0, dt = 1.0, length_unit = "px", time_unit = "frame") :
        set_scale!(bc; pixel_size = sc.pixel_size, dt = sc.dt,
                   length_unit = sc.length_unit, time_unit = sc.time_unit)
    bc.preprocess_steps[] = copy(recipe.preprocessing)
    bc.preprocess[] = recipe_preprocess(recipe)
    nsteps = length(recipe.preprocessing)
    bc.status[] = "loaded settings: $(length(recipe.passes)) passes" *
                  (nsteps == 0 ? "" : ", $nsteps preprocessing steps")
    return bc
end

"""
    validate(bc::BatchRunner) -> Union{Nothing,String}

Return `nothing` when the form has the required inputs and valid settings,
or a message describing the first problem. With an ROI, the first frame is
loaded to check bounds and mask dimensions; later frames are checked by the
core during processing. With a non-`:custom` effort, the manual schedule is
not consulted.
"""
function validate(bc::BatchRunner)
    isempty(bc.files[]) && return "add frames first"
    prs = try
        frame_pairs(bc)
    catch err
        return _errmsg(err)
    end
    isempty(prs) && return "no pairs to process"
    bc.effort[] in EFFORT_LEVELS || return "unknown effort :$(bc.effort[])"
    bc.effort[] === :saved && bc.saved_passes[] === nothing && return "no saved settings loaded"
    if bc.effort[] === :custom
        try
            build_parameters(bc)
        catch err
            return _errmsg(err)
        end
    end
    try
        build_scale(bc)
        if bc.roi[] !== nothing
            frame = first(bc.files[])
            image = frame isa AbstractMatrix ? frame : load_image(frame)
            Hammerhead.roi_views(image, image, bc.mask[], bc.roi[])
            roi_size = (length(bc.roi[].rows), length(bc.roi[].cols))
            for pass in _batch_passes(bc)
                all(pass.search_area_size .<= roi_size) ||
                    return "ROI size $roi_size is smaller than search area $(pass.search_area_size); enlarge the ROI or choose smaller windows"
            end
        end
    catch err
        return _errmsg(err)
    end
    return nothing
end

_errmsg(err) = first(split(sprint(showerror, err), '\n'))

"""
    start!(bc::BatchRunner; async = true)

Validate and start the batch. Return `bc` immediately with `async = true`,
or wait for the run with `async = false`. Follow `running`, `progress`, and
`status`; `completed` receives each finished pair and `results` holds the
finished run. If `output_path` is set, pairs are also written incrementally.
"""
function start!(bc::BatchRunner; async::Bool = true)
    bc.running[] && return bc
    msg = validate(bc)
    msg === nothing || (bc.status[] = msg; return bc)
    # Capture before notifying running or yielding to the asynchronous task.
    # Observers may edit the form immediately when the run starts.
    roi = bc.roi[]
    bc.cancel[] = false
    bc.running[] = true
    async ? errormonitor(@async _run!(bc; roi)) : _run!(bc; roi)
    return bc
end

"""
    cancel!(bc::BatchRunner)

Request cancellation after the pair in flight. Return `bc`. Finished pairs
remain in `completed` and in the output file when one is configured.
"""
cancel!(bc::BatchRunner) = (bc.cancel[] = true; bc)

function _run!(bc::BatchRunner; roi = bc.roi[])
    try
        prs = frame_pairs(bc)
        bc.progress[] = (0, length(prs))
        bc.status[] = "running…"
        bc.completed[] = PIVResult[]   # fresh accumulator for this run
        callback = (i, n) -> begin
            bc.progress[] = (i, n)
            bc.cancel[] && throw(BatchCancelled())
        end
        # Live accumulator: run_piv_sequence stores results on this (serial)
        # task, so the push happens on the task driving the batch and open
        # views can follow along.
        on_result = (i, r) -> (push!(bc.completed[], r); notify(bc.completed))
        output = isempty(bc.output_path[]) ? nothing : bc.output_path[]
        results = if bc.preprocess_steps[] === nothing
            # Custom preprocessing function: run directly, without a recipe.
            run_piv_sequence(prs, _batch_passes(bc); progress = callback, on_result,
                             output, scale = build_scale(bc), preprocess = bc.preprocess[],
                             roi, mask = bc.mask[])
        else
            r = batch_recipe(bc)
            # Use the ROI captured when the run started.
            recipe = PIVRecipe(r.passes; preprocessing = r.preprocessing, mask = r.mask,
                               roi, scale = r.scale)
            apply_recipe(recipe, prs; progress = callback, on_result, output)
        end
        bc.results[] = results
        bc.status[] = "done: $(length(results)) pairs" *
                      (output === nothing ? "" : " → $(basename(output))")
    catch err
        if err isa BatchCancelled
            done, total = bc.progress[]
            bc.status[] = "cancelled after $done of $total pairs"
        else
            bc.status[] = "failed: $(_errmsg(err))"
        end
    finally
        bc.running[] = false
    end
    return bc
end
