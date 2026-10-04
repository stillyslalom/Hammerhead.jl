# Saved processing settings: a recipe captures everything needed to process a
# recording again (passes, preprocessing, mask, ROI, scale), independent of the
# images it is applied to. Recipes are written as TOML text: a settings file
# with its arrays (mask, backgrounds) in image files beside it, or the same
# text embedded in a JLD2 file with the arrays stored next to it.
const RECIPE_FORMAT_VERSION = 3   # 2: per-camera preprocessing, PTV/tracking; 3: TOML text

"""
    PreprocessStep(operation; kwargs...)

A saveable built-in preprocessing operation for a [`PIVRecipe`](@ref).
Operations and their options (defaults shown):

- `:subtract_background` (`background`, a matrix the size of the full image)
- `:intensity_cap` (`n_sigma=2`)
- `:highpass_filter` (`sigma=3`)
- `:clahe` (`tiles=(8, 8)`, `clip_limit=2`, `nbins=256`)
- `:percentile_stretch` (`low=1`, `high=99`)
- `:invert_image`
- `:local_variance_normalize` (`sigma=3`, `epsilon=1e-3`)

A background is copied into the step, so the recipe carries it.
"""
struct PreprocessStep
    operation::Symbol
    options::Dict{String,Any}
end

const _PREPROCESS_DEFAULTS = Dict{Symbol,Dict{String,Any}}(
    :subtract_background => Dict{String,Any}(),
    :intensity_cap => Dict{String,Any}("n_sigma" => 2.0),
    :highpass_filter => Dict{String,Any}("sigma" => 3.0),
    :clahe => Dict{String,Any}("tiles" => [8, 8], "clip_limit" => 2.0, "nbins" => 256),
    :percentile_stretch => Dict{String,Any}("low" => 1.0, "high" => 99.0),
    :invert_image => Dict{String,Any}(),
    :local_variance_normalize => Dict{String,Any}("sigma" => 3.0, "epsilon" => 1e-3))

function PreprocessStep(operation::Symbol; kwargs...)
    haskey(_PREPROCESS_DEFAULTS, operation) ||
        throw(ArgumentError("unsupported preprocessing operation :$operation"))
    options = deepcopy(_PREPROCESS_DEFAULTS[operation])
    allowed = operation === :subtract_background ? ("background",) : keys(options)
    for (key, value) in kwargs
        String(key) in allowed || throw(ArgumentError("unsupported :$operation option $key"))
        options[String(key)] = value
    end
    if operation === :subtract_background
        bg = get(options, "background", nothing)
        bg isa AbstractMatrix{<:Real} ||
            throw(ArgumentError(":subtract_background requires a background matrix"))
        options["background"] = Matrix{Float64}(bg)
    elseif operation === :clahe
        options["tiles"] = Int.(collect(options["tiles"]))
        options["nbins"] = Int(options["nbins"])
        options["clip_limit"] = Float64(options["clip_limit"])
    else
        for (key, value) in options
            options[key] = Float64(value)
        end
    end
    PreprocessStep(operation, options)
end

Base.:(==)(a::PreprocessStep, b::PreprocessStep) =
    a.operation === b.operation && a.options == b.options

function _apply_step(image, step::PreprocessStep)
    o = step.options
    op = step.operation
    op === :subtract_background && return subtract_background(image, o["background"])
    op === :intensity_cap && return intensity_cap(image; n_sigma = o["n_sigma"])
    op === :highpass_filter && return highpass_filter(image; sigma = o["sigma"])
    op === :clahe && return clahe(image; tiles = Tuple(o["tiles"]),
                                  clip_limit = o["clip_limit"], nbins = o["nbins"])
    op === :percentile_stretch && return percentile_stretch(image; low = o["low"], high = o["high"])
    op === :invert_image && return invert_image(image)
    return local_variance_normalize(image; sigma = o["sigma"], epsilon = o["epsilon"])
end

"""
    PIVRecipe(passes; preprocessing=PreprocessStep[], mask=nothing, roi=nothing,
              scale=nothing, mode=:sequence, image_type=Float64,
              predictor_smoothing=true, mask_threshold=0.5,
              ptv=PTVParameters(), ptv_predictor=:piv,
              min_track_length=3, max_gap=0)

Processing settings that can be saved with [`save_recipe`](@ref), reloaded
with [`load_recipe`](@ref), and run on any recording with
[`apply_recipe`](@ref).

- `passes`: one [`PIVParameters`](@ref) or a pass schedule, for example from
  [`multipass_parameters`](@ref) or [`effort_schedule`](@ref).
- `preprocessing`: ordered [`PreprocessStep`](@ref)s applied to every frame.
  For stereo, a tuple `(steps1, steps2)` gives each camera its own list,
  for example to subtract each camera's background.
- `mask`: a static Bool mask the size of the full image (`true` = excluded).
  For stereo, it is on the dewarped grid.
- `roi`: an [`ROI`](@ref) to analyze (planar only).
- `scale`: a [`PhysicalScale`](@ref) attached to the results.
- `mode`: `:sequence` gives one result per pair; `:ensemble` pools all pairs
  into one result with sum-of-correlation; `:ptv` matches particles in each
  pair ([`run_ptv_sequence`](@ref)); `:tracking` links particles through a
  frame sequence ([`track_particles`](@ref)).
- `image_type`: `Float32` or `Float64` processing precision.
- `ptv`: the [`PTVParameters`](@ref) of the `:ptv` and `:tracking` modes.
  `ptv_predictor = :piv` predicts each pair's displacement with `passes`
  (the PIV predictor); `:none` searches around each particle's own position.
- `min_track_length`, `max_gap`: [`track_particles`](@ref) options of the
  `:tracking` mode.

PTV and tracking recipes run planar recordings on the CPU and take no ROI.

The constructor copies its inputs.
"""
struct PIVRecipe
    passes::Vector{PIVParameters}
    preprocessing::Union{Vector{PreprocessStep},NTuple{2,Vector{PreprocessStep}}}
    mask::Union{Nothing,BitMatrix}
    roi::Union{Nothing,ROI}
    scale::Union{Nothing,PhysicalScale}
    mode::Symbol
    image_type::DataType
    predictor_smoothing::Bool
    mask_threshold::Float64
    ptv::PTVParameters
    ptv_predictor::Symbol
    min_track_length::Int
    max_gap::Int
end

const RECIPE_MODES = (:sequence, :ensemble, :ptv, :tracking)

function PIVRecipe(passes::Union{PIVParameters,AbstractVector{PIVParameters}};
                   preprocessing::Union{AbstractVector{PreprocessStep},
                                        NTuple{2,AbstractVector{PreprocessStep}}} = PreprocessStep[],
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                   roi = nothing,
                   scale::Union{Nothing,PhysicalScale} = nothing,
                   mode::Symbol = :sequence,
                   image_type::DataType = Float64,
                   predictor_smoothing::Bool = true,
                   mask_threshold::Real = 0.5,
                   ptv::PTVParameters = PTVParameters(),
                   ptv_predictor::Symbol = :piv,
                   min_track_length::Integer = 3,
                   max_gap::Integer = 0)
    schedule = passes isa PIVParameters ? [passes] : collect(PIVParameters, passes)
    isempty(schedule) && throw(ArgumentError("a recipe needs at least one pass"))
    mode in RECIPE_MODES ||
        throw(ArgumentError("mode must be one of $(join((":$m" for m in RECIPE_MODES), ", ")), got :$mode"))
    ptv_predictor in (:piv, :none) ||
        throw(ArgumentError("ptv_predictor must be :piv or :none, got :$ptv_predictor"))
    min_track_length >= 2 ||
        throw(ArgumentError("min_track_length must be at least 2, got $min_track_length"))
    max_gap >= 0 || throw(ArgumentError("max_gap must be nonnegative, got $max_gap"))
    mode in (:ptv, :tracking) && roi !== nothing &&
        throw(ArgumentError("PTV and tracking recipes do not support an ROI; use a mask instead"))
    image_type in (Float32, Float64) ||
        throw(ArgumentError("image_type must be Float32 or Float64, got $image_type"))
    0 < mask_threshold <= 1 || throw(ArgumentError("mask_threshold must be in (0, 1]"))
    steps = preprocessing isa Tuple ? map(_copy_steps, preprocessing) : _copy_steps(preprocessing)
    PIVRecipe(schedule, steps,
              mask === nothing ? nothing : BitMatrix(mask),
              roi === nothing || roi isa ROI ? roi : ROI(roi),
              scale, mode, image_type, predictor_smoothing, Float64(mask_threshold),
              ptv, ptv_predictor, Int(min_track_length), Int(max_gap))
end

Base.:(==)(a::PIVRecipe, b::PIVRecipe) = _recipe_data(a) == _recipe_data(b)

_copy_steps(steps) = deepcopy(collect(PreprocessStep, steps))

"""
    recipe_preprocess(recipe) -> function or nothing
    recipe_preprocess(steps) -> function or nothing

Return the preprocessing of a recipe, or of an ordered vector of
[`PreprocessStep`](@ref)s, as a single-image function suitable for the
`preprocess` keyword of the sequence drivers. Return `nothing` when there are
no steps. The function returns a new image and leaves its input unchanged.
A recipe with per-camera preprocessing returns a tuple `(f1, f2)`, which the
stereo drivers accept as their `preprocess` keyword.

Use the `steps` form to preview preprocessing before building a recipe: it
applies exactly what [`apply_recipe`](@ref) will. The steps are copied, so
later changes to `steps` do not affect the returned function.
"""
recipe_preprocess(recipe::PIVRecipe) =
    recipe.preprocessing isa Tuple ? map(_preprocess_function, recipe.preprocessing) :
    _preprocess_function(recipe.preprocessing)
recipe_preprocess(steps::AbstractVector{PreprocessStep}) = _preprocess_function(_copy_steps(steps))

function _preprocess_function(steps::Vector{PreprocessStep})
    isempty(steps) && return nothing
    return image -> foldl(_apply_step, steps; init = image)
end

# --- serialization to plain dictionaries ----------------------------------

function _validator_data(v)
    name = v isa PeakRatioValidator ? "peak_ratio" :
           v isa CorrelationMomentValidator ? "correlation_moment" :
           v isa VelocityMagnitudeValidator ? "velocity_magnitude" :
           v isa UniversalOutlierValidator ? "uod" :
           throw(ArgumentError("custom validator $(typeof(v)) cannot be saved in a recipe"))
    Dict{String,Any}("operation" => name,
                     "options" => Dict{String,Any}(String(k) => getfield(v, k) for k in fieldnames(typeof(v))))
end

function _validator_from_data(data)
    name, o = data["operation"], data["options"]
    name == "peak_ratio" && return PeakRatioValidator(o["threshold"])
    name == "correlation_moment" && return CorrelationMomentValidator(o["threshold"])
    name == "velocity_magnitude" && return VelocityMagnitudeValidator(o["min"], o["max"])
    name == "uod" && return UniversalOutlierValidator(o["threshold"];
        neighborhood_size = o["neighborhood_size"], epsilon = o["epsilon"])
    throw(ArgumentError("unknown validator \"$name\" in recipe"))
end

function _pass_data(p::PIVParameters)
    data = Dict{String,Any}()
    for k in fieldnames(PIVParameters)
        value = getfield(p, k)
        data[String(k)] = k === :validation ? [_validator_data(parse_validator(v)) for v in value] :
                          value isa Symbol ? String(value) :
                          value isa Tuple ? collect(value) : value
    end
    data
end

function _pass_from_data(data)
    args = Pair{Symbol,Any}[]
    for k in fieldnames(PIVParameters)
        haskey(data, String(k)) || continue   # newer fields keep their defaults
        value = data[String(k)]
        T = fieldtype(PIVParameters, k)
        value = k === :validation ? Tuple(_validator_from_data(v) for v in value) :
                T === Symbol ? Symbol(value) :
                T <: Tuple ? Tuple(value) : value
        push!(args, k => value)
    end
    PIVParameters(; args...)
end

_steps_data(steps) = [Dict{String,Any}("operation" => String(s.operation),
                                        "options" => deepcopy(s.options)) for s in steps]
_steps_from_data(data) =
    PreprocessStep[PreprocessStep(Symbol(s["operation"]); (Symbol(k) => v for (k, v) in s["options"])...)
               for s in data]

function _recipe_data(r::PIVRecipe)
    pre = r.preprocessing
    Dict{String,Any}(
        "passes" => [_pass_data(p) for p in r.passes],
        (pre isa Tuple ? "camera_preprocessing" => [_steps_data(s) for s in pre] :
                         "preprocessing" => _steps_data(pre)),
        "mask" => r.mask,
        "roi" => r.roi === nothing ? nothing :
                 [first(r.roi.rows), last(r.roi.rows), first(r.roi.cols), last(r.roi.cols)],
        "scale" => r.scale === nothing ? nothing :
                   Dict{String,Any}(String(k) => getfield(r.scale, k) for k in fieldnames(PhysicalScale)),
        "mode" => String(r.mode),
        "image_type" => string(r.image_type),
        "predictor_smoothing" => r.predictor_smoothing,
        "mask_threshold" => r.mask_threshold,
        "ptv" => Dict{String,Any}(String(k) => (v = getfield(r.ptv, k); v isa Symbol ? String(v) : v)
                                 for k in fieldnames(PTVParameters)),
        "ptv_predictor" => String(r.ptv_predictor),
        "min_track_length" => r.min_track_length,
        "max_gap" => r.max_gap)
end

function _recipe_from_data(d)
    steps = haskey(d, "camera_preprocessing") ?
            Tuple(_steps_from_data(s) for s in d["camera_preprocessing"]) :
            _steps_from_data(d["preprocessing"])
    s = get(d, "scale", nothing)
    scale = s === nothing ? nothing :
            PhysicalScale(s["pixel_size"], s["dt"], s["length_unit"], s["time_unit"])
    r = get(d, "roi", nothing)
    roi = r === nothing ? nothing : ROI(r[1]:r[2], r[3]:r[4])
    PIVRecipe([_pass_from_data(p) for p in d["passes"]];
              preprocessing = steps, mask = get(d, "mask", nothing), roi, scale,
              mode = Symbol(d["mode"]),
              image_type = d["image_type"] == "Float32" ? Float32 : Float64,
              predictor_smoothing = d["predictor_smoothing"],
              mask_threshold = d["mask_threshold"],
              ptv = haskey(d, "ptv") ? _ptv_from_data(d["ptv"]) : PTVParameters(),
              ptv_predictor = Symbol(get(d, "ptv_predictor", "piv")),
              min_track_length = get(d, "min_track_length", 3),
              max_gap = get(d, "max_gap", 0))
end

function _ptv_from_data(data)
    args = Pair{Symbol,Any}[]
    for k in fieldnames(PTVParameters)
        haskey(data, String(k)) || continue   # newer fields keep their defaults
        v = data[String(k)]
        push!(args, k => (k === :threshold && v isa AbstractString ? Symbol(v) : v))
    end
    PTVParameters(; args...)
end

# --- TOML text ------------------------------------------------------------------

# The recipe as TOML text. Arrays (the mask, background images) are handed to
# `put(name, array)`, which stores them and returns the reference written in
# their place; `nothing` settings are left out (absent keys load as defaults).
function _recipe_toml(r::PIVRecipe, put)
    d = _recipe_data(r)
    d["recipe_format_version"] = RECIPE_FORMAT_VERSION
    d["hammerhead_version"] = string(pkgversion(@__MODULE__))
    r.mask === nothing || (d["mask"] = put("mask", r.mask))
    nbg = Ref(0)
    function steps_toml(steps, prefix)
        map(steps) do s
            o = Dict{String,Any}("operation" => String(s.operation))
            for (k, v) in s.options
                if v isa AbstractMatrix
                    nbg[] += 1
                    o[k] = put(prefix * "background" * (nbg[] == 1 ? "" : string(nbg[])), v)
                else
                    o[k] = v
                end
            end
            o
        end
    end
    if r.preprocessing isa Tuple
        delete!(d, "camera_preprocessing")
        for (c, steps) in enumerate(r.preprocessing)
            nbg[] = 0
            d["camera$(c)_preprocessing"] = steps_toml(steps, "camera$c.")
        end
    else
        d["preprocessing"] = steps_toml(r.preprocessing, "")
    end
    filter!(kv -> kv.second !== nothing, d)
    io = IOBuffer()
    println(io, "# Hammerhead processing settings (PIVRecipe)")
    TOML.print(io, d; sorted = true, by = _toml_key_order)
    # a blank line before every table header, so repeated passes stand apart
    return replace(String(take!(io)), r"(?<=[^\n])\n(?=\[)" => "\n\n")
end

# Settings in the order a reader looks for them (each key's first rank wins:
# PIV and PTV parameters share some names).
const _TOML_KEY_ORDER = let order = Dict{String,Int}()
    for k in ("recipe_format_version", "hammerhead_version", "mode", "image_type", "mask", "roi",
              "predictor_smoothing", "mask_threshold", "ptv_predictor", "min_track_length",
              "max_gap", "operation", "options", "pixel_size", "dt", "length_unit", "time_unit",
              map(String, fieldnames(PIVParameters))..., map(String, fieldnames(PTVParameters))...)
        get!(order, k, length(order) + 1)
    end
    order
end
_toml_key_order(k) = (get(_TOML_KEY_ORDER, k, typemax(Int)), k)

# A recipe from TOML text; `get(ref, kind)` returns the array a reference
# names (`kind` is `:mask` or `:background`).
function _recipe_from_toml(get, text::AbstractString, source::AbstractString)
    d = TOML.parse(text)
    version = Base.get(d, "recipe_format_version", nothing)
    version == RECIPE_FORMAT_VERSION ||
        throw(ArgumentError("$source has unsupported recipe_format_version $version"))
    haskey(d, "mask") && (d["mask"] = BitMatrix(get(d["mask"], :mask)))
    function steps_data(steps)
        map(steps) do s
            o = Dict{String,Any}(k => v for (k, v) in s if k != "operation")
            haskey(o, "background") && (o["background"] = get(o["background"], :background))
            Dict{String,Any}("operation" => s["operation"], "options" => o)
        end
    end
    if haskey(d, "camera1_preprocessing")
        d["camera_preprocessing"] = [steps_data(Base.get(d, "camera$(c)_preprocessing", []))
                                     for c in 1:2]
    else
        d["preprocessing"] = steps_data(Base.get(d, "preprocessing", []))
    end
    return _recipe_from_data(d)
end

# A recipe inside a JLD2 file (a settings file or a results file): the TOML
# text, with the arrays it names stored under "recipe_arrays/".
function _write_recipe(file, recipe::PIVRecipe)
    file["recipe_format_version"] = RECIPE_FORMAT_VERSION
    file["recipe_toml"] = _recipe_toml(recipe, (name, a) -> (file["recipe_arrays/" * name] = a; name))
    file["recipe_software"] = Dict{String,Any}(
        "hammerhead_version" => string(pkgversion(@__MODULE__)),
        "julia_version" => string(VERSION))
    nothing
end

"""
    save_recipe(path, recipe) -> path

Save a [`PIVRecipe`](@ref). A `.toml` path writes a TOML settings file;
its arrays go in image files beside it, named after it: the mask as
`<name>.mask.png` (white = excluded) and each background as
`<name>.background.tif` (Float64 TIFF; per-camera backgrounds as
`<name>.camera1.background.tif`, …). Keep these files together. Any other
path writes a JLD2 file holding the same TOML text and the arrays. Both
record the Hammerhead version that saved them.
"""
function save_recipe(path::AbstractString, recipe::PIVRecipe)
    if _is_toml(path)
        dir = dirname(abspath(path))
        stem = splitext(basename(path))[1]
        text = _recipe_toml(recipe, function (name, a)
            file = "$stem.$name" * (a isa BitMatrix ? ".png" : ".tif")
            FileIO.save(joinpath(dir, file), Gray.(a))
            file
        end)
        write(path, text)
    else
        jldopen(file -> _write_recipe(file, recipe), path, "w")
    end
    path
end

_is_toml(path) = lowercase(splitext(path)[2]) == ".toml"

"""
    load_recipe(path) -> PIVRecipe

Load a recipe saved with [`save_recipe`](@ref) (a TOML settings file with
its image files, or a JLD2 file), or the recipe that produced a results file
written by [`apply_recipe`](@ref). Recipes saved by earlier versions load too.
"""
function load_recipe(path::AbstractString)
    if _is_toml(path)
        dir = dirname(abspath(path))
        return _recipe_from_toml(read(path, String), path) do ref, kind
            file = joinpath(dir, ref)
            isfile(file) ||
                throw(ArgumentError("$path refers to $ref, which is not beside it"))
            kind === :mask ? load_mask(file) : load_image(Float64, file)
        end
    end
    jldopen(path, "r") do file
        haskey(file, "recipe_toml") &&
            return _recipe_from_toml((ref, _) -> file["recipe_arrays/" * ref], file["recipe_toml"], path)
        haskey(file, "recipe") ||
            throw(ArgumentError("$path contains no Hammerhead recipe"))
        version = file["recipe_format_version"]
        version in (1, 2) ||
            throw(ArgumentError("$path has unsupported recipe_format_version $version"))
        _recipe_from_data(file["recipe"])
    end
end

"""
    recipe_toml(recipe) -> String

The recipe as the TOML text [`save_recipe`](@ref) writes, with each array
replaced by its name (`"mask"`, `"background"`, …): a readable summary of
the settings.
"""
recipe_toml(recipe::PIVRecipe) = _recipe_toml(recipe, (name, _) -> name)

"""
    apply_recipe(recipe, pairs; output=nothing, backend=:cpu, kwargs...)
    apply_recipe(recipe, frames; output=nothing, kwargs...)   # :tracking
    apply_recipe(recipe, pairs1, pairs2, dw1, dw2; output=nothing, backend=:cpu, kwargs...)

Process a recording with a recipe's settings. The first form runs planar
PIV or PTV on image pairs (paths, matrices, or frame references); a
`:tracking` recipe takes the frame sequence instead of pairs; the last form
runs stereo PIV on two synchronized cameras with their
[`ImageDewarper`](@ref)s.

A `:sequence` recipe returns one result per pair, like
[`run_piv_sequence`](@ref) / [`run_piv_stereo_sequence`](@ref); an
`:ensemble` recipe returns one pooled result, like [`run_piv_ensemble`](@ref)
/ [`run_piv_stereo_ensemble`](@ref); a `:ptv` recipe returns one
[`PTVResult`](@ref) per pair, like [`run_ptv_sequence`](@ref), and a
`:tracking` recipe one [`TrackingResult`](@ref), like
[`track_particles`](@ref). When `output` is a file path, results
are saved there and the recipe is stored alongside them, so
`load_recipe(output)` recovers the settings; a stereo run also stores the
cameras and grid, so [`load_calibration`](@ref)`(output)` recovers the
dewarpers.

`masks` adds masks that change from pair to pair (input data, like the
frames; `:sequence` and `:ptv` recipes): one entry per pair — a Bool matrix,
a mask image path (white = excluded, as [`load_mask`](@ref) reads it), or a
tuple of two such entries for frames A and B (unioned) — or a function
`(i, imgA, imgB) -> mask`. Files load as each pair is analyzed. Each pair's
mask is unioned with the recipe's static mask, and mask image paths are
stored with the results ([`load_sources`](@ref)`(output; masks = true)`).
Remaining keywords (such as
`progress`, `on_result`, `collect_results`, or `threaded`) go to the
underlying driver.
"""
function apply_recipe(recipe::PIVRecipe, pairs::AbstractVector;
                      output::Union{Nothing,AbstractString,Function} = nothing,
                      backend::Symbol = :cpu, masks = nothing, kwargs...)
    recipe.preprocessing isa Tuple &&
        throw(ArgumentError("per-camera preprocessing needs a stereo recording"))
    masks === nothing || recipe.mode in (:sequence, :ptv) ||
        throw(ArgumentError("per-pair masks need a :sequence or :ptv recipe; " *
                            "a :$(recipe.mode) recipe takes the recipe's static mask only"))
    masks isa AbstractVector && length(masks) != length(pairs) &&
        throw(DimensionMismatch("masks has $(length(masks)) entries for $(length(pairs)) pairs"))
    mask = _recipe_mask(recipe.mask, masks)
    if recipe.mode in (:ptv, :tracking)
        return _with_mask_sources(output, masks) do
            _apply_ptv(recipe, pairs; output, backend, mask, kwargs...)
        end
    end
    common = (; preprocess = recipe_preprocess(recipe), image_type = recipe.image_type,
              mask, scale = recipe.scale, backend,
              predictor_smoothing = recipe.predictor_smoothing,
              mask_threshold = recipe.mask_threshold)
    if recipe.mode === :ensemble
        recipe.roi === nothing ||
            throw(ArgumentError("ensemble recipes do not support an ROI"))
        output isa Function &&
            throw(ArgumentError("an ensemble result needs a single output path"))
        result = run_piv_ensemble(pairs, recipe.passes; common..., kwargs...)
        output === nothing || _save_with_recipe(output, result, recipe)
        return result
    end
    return _with_mask_sources(output, masks) do
        _with_recipe(output, recipe) do
            run_piv_sequence(pairs, recipe.passes; output, roi = recipe.roi, common..., kwargs...)
        end
    end
end

# Per-pair masks (input data, like the frames) unioned with the recipe's
# static mask: a callback for the sequence drivers that loads mask files as
# each pair is analyzed. An entry is a Bool matrix, a mask image path
# (`load_mask`: white = excluded), or a tuple of two (frames A and B,
# unioned); `nothing` leaves only the static mask.
_recipe_mask(static, ::Nothing) = static
function _recipe_mask(static, masks)
    return function (i, imgA, imgB)
        m = _mask_entry(masks isa Function ? masks(i, imgA, imgB) : masks[i])
        m === nothing && return static === nothing ? falses(size(imgA)) : static
        static === nothing && return m
        size(m) == size(static) ||
            throw(DimensionMismatch("pair $i: mask is $(size(m)) but the recipe's mask is $(size(static))"))
        return BitMatrix(m .| static)
    end
end

_mask_entry(::Nothing) = nothing
_mask_entry(m::AbstractMatrix{Bool}) = m
_mask_entry(path::AbstractString) = load_mask(path)
function _mask_entry(t::Tuple)
    a, b = _mask_entry(t[1]), _mask_entry(t[2])
    a === nothing && return b
    b === nothing && return a
    size(a) == size(b) || throw(DimensionMismatch("the two frames' masks differ in size"))
    return BitMatrix(a .| b)
end
_mask_entry(x) = throw(ArgumentError("a mask entry is a Bool matrix, a mask image path, or a pair of them; got $(typeof(x))"))

# Record mask image paths next to the results (when the masks were files).
function _with_mask_sources(run, output, masks)
    started = time()
    try
        return run()
    finally
        if output isa AbstractString && masks isa AbstractVector && isfile(output) &&
           mtime(output) >= started - 1
            labels = [_mask_labels(e) for e in masks]
            any(!isempty, labels) && jldopen(output, "r+") do file
                for (i, l) in enumerate(labels)
                    isempty(l) || (file[mask_source_key(i)] = l)
                end
            end
        end
    end
end

_mask_labels(e::AbstractString) = [String(e)]
_mask_labels(t::Tuple) = all(x -> x isa AbstractString, t) ? String[String(x) for x in t] : String[]
_mask_labels(_) = String[]

function apply_recipe(recipe::PIVRecipe, pairs1::AbstractVector, pairs2::AbstractVector,
                      dw1::ImageDewarper, dw2::ImageDewarper;
                      output::Union{Nothing,AbstractString,Function} = nothing,
                      backend::Symbol = :cpu, kwargs...)
    recipe.roi === nothing ||
        throw(ArgumentError("stereo recipes do not support an ROI; mask the dewarped grid instead"))
    recipe.mode in (:ptv, :tracking) &&
        throw(ArgumentError("PTV and tracking recipes run on planar recordings, not stereo"))
    common = (; preprocess = recipe_preprocess(recipe), image_type = recipe.image_type,
              mask = recipe.mask, scale = recipe.scale, backend,
              predictor_smoothing = recipe.predictor_smoothing,
              mask_threshold = recipe.mask_threshold)
    if recipe.mode === :ensemble
        output isa Function &&
            throw(ArgumentError("an ensemble result needs a single output path"))
        result = run_piv_stereo_ensemble(pairs1, pairs2, dw1, dw2, recipe.passes;
                                         common..., kwargs...)
        output === nothing || _save_with_recipe(output, result, recipe, (dw1, dw2))
        return result
    end
    return _with_recipe(output, recipe, (dw1, dw2)) do
        run_piv_stereo_sequence(pairs1, pairs2, dw1, dw2, recipe.passes;
                                output, common..., kwargs...)
    end
end

function _apply_ptv(recipe::PIVRecipe, inputs::AbstractVector; output, backend::Symbol, kwargs...)
    backend === :cpu ||
        throw(ArgumentError("PTV and tracking recipes run on the CPU, got backend = :$backend"))
    common = (; preprocess = recipe_preprocess(recipe), image_type = recipe.image_type,
              mask = get(kwargs, :mask, recipe.mask), scale = recipe.scale,
              predictor = recipe.ptv_predictor === :piv ? :piv : nothing,
              piv_passes = recipe.passes)
    if recipe.mode === :tracking
        output isa Function &&
            throw(ArgumentError("a tracking result needs a single output path"))
        result = track_particles(inputs, recipe.ptv; min_track_length = recipe.min_track_length,
                                 max_gap = recipe.max_gap, common..., _without_mask(kwargs)...)
        output === nothing || _save_with_recipe(output, result, recipe)
        return result
    end
    return _with_recipe(output, recipe) do
        run_ptv_sequence(inputs, recipe.ptv; output, common..., _without_mask(kwargs)...)
    end
end

_without_mask(kwargs) = (k => v for (k, v) in pairs(kwargs) if k !== :mask)

# Store the recipe next to the results even when the batch stops early, so a
# partial file still records how it was produced.
function _with_recipe(run, output, recipe, dewarpers = nothing)
    started = time()
    try
        return run()
    finally
        # Skip a stale file the driver never opened (e.g. invalid arguments).
        output isa AbstractString && isfile(output) && mtime(output) >= started - 1 &&
            jldopen(file -> _write_settings(file, recipe, dewarpers), output, "r+")
    end
end

function _save_with_recipe(path, result, recipe, dewarpers = nothing)
    save_results(path, result)
    jldopen(file -> _write_settings(file, recipe, dewarpers), path, "r+")
    nothing
end

function _write_settings(file, recipe, dewarpers)
    _write_recipe(file, recipe)
    dewarpers === nothing || _write_calibration(file, dewarpers)
end

"""
    recipe_diff(a, b) -> Vector{NamedTuple}

List the settings that differ between two recipes as
`(; path, before, after)` entries, with paths such as
`"passes[2].window_size"`. Mask and background arrays are summarized by size.
"""
function recipe_diff(a::PIVRecipe, b::PIVRecipe)
    changes = NamedTuple{(:path, :before, :after),Tuple{String,Any,Any}}[]
    _diff!(changes, "", _recipe_data(a), _recipe_data(b))
    changes
end

_diff_summary(x::AbstractMatrix) = "$(join(size(x), '×')) $(eltype(x)) array"
_diff_summary(x) = x
_diff_summary(::Missing) = missing

function _diff!(changes, path, a, b)
    if a isa AbstractDict && b isa AbstractDict
        for k in sort!(collect(union(keys(a), keys(b))))
            _diff!(changes, isempty(path) ? k : "$path.$k", get(a, k, missing), get(b, k, missing))
        end
    elseif a isa AbstractVector && b isa AbstractVector && !(eltype(a) <: Real && eltype(b) <: Real)
        for i in 1:max(length(a), length(b))
            _diff!(changes, "$path[$i]", get(a, i, missing), get(b, i, missing))
        end
    elseif !isequal(a, b)
        push!(changes, (; path, before = _diff_summary(a), after = _diff_summary(b)))
    end
    changes
end
