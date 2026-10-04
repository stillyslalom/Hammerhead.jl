# Saved processing settings: a recipe captures everything needed to process a
# recording again (passes, preprocessing, mask, ROI, scale), independent of the
# images it is applied to. Recipes are stored as plain JLD2 dictionaries.
const RECIPE_FORMAT_VERSION = 1

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
              predictor_smoothing=true, mask_threshold=0.5)

Processing settings that can be saved with [`save_recipe`](@ref), reloaded
with [`load_recipe`](@ref), and run on any recording with
[`apply_recipe`](@ref).

- `passes`: one [`PIVParameters`](@ref) or a pass schedule, for example from
  [`multipass_parameters`](@ref) or [`effort_schedule`](@ref).
- `preprocessing`: ordered [`PreprocessStep`](@ref)s applied to every frame.
- `mask`: a static Bool mask the size of the full image (`true` = excluded).
  For stereo, it is on the dewarped grid.
- `roi`: an [`ROI`](@ref) to analyze (planar only).
- `scale`: a [`PhysicalScale`](@ref) attached to the results.
- `mode`: `:sequence` gives one result per pair; `:ensemble` pools all pairs
  into one result with sum-of-correlation.
- `image_type`: `Float32` or `Float64` processing precision.

The constructor copies its inputs.
"""
struct PIVRecipe
    passes::Vector{PIVParameters}
    preprocessing::Vector{PreprocessStep}
    mask::Union{Nothing,BitMatrix}
    roi::Union{Nothing,ROI}
    scale::Union{Nothing,PhysicalScale}
    mode::Symbol
    image_type::DataType
    predictor_smoothing::Bool
    mask_threshold::Float64
end

function PIVRecipe(passes::Union{PIVParameters,AbstractVector{PIVParameters}};
                   preprocessing::AbstractVector{PreprocessStep} = PreprocessStep[],
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                   roi = nothing,
                   scale::Union{Nothing,PhysicalScale} = nothing,
                   mode::Symbol = :sequence,
                   image_type::DataType = Float64,
                   predictor_smoothing::Bool = true,
                   mask_threshold::Real = 0.5)
    schedule = passes isa PIVParameters ? [passes] : collect(PIVParameters, passes)
    isempty(schedule) && throw(ArgumentError("a recipe needs at least one pass"))
    mode in (:sequence, :ensemble) ||
        throw(ArgumentError("mode must be :sequence or :ensemble, got :$mode"))
    image_type in (Float32, Float64) ||
        throw(ArgumentError("image_type must be Float32 or Float64, got $image_type"))
    0 < mask_threshold <= 1 || throw(ArgumentError("mask_threshold must be in (0, 1]"))
    PIVRecipe(schedule, deepcopy(collect(preprocessing)),
              mask === nothing ? nothing : BitMatrix(mask),
              roi === nothing || roi isa ROI ? roi : ROI(roi),
              scale, mode, image_type, predictor_smoothing, Float64(mask_threshold))
end

Base.:(==)(a::PIVRecipe, b::PIVRecipe) = _recipe_data(a) == _recipe_data(b)

"""
    recipe_preprocess(recipe) -> function or nothing
    recipe_preprocess(steps) -> function or nothing

Return the preprocessing of a recipe, or of an ordered vector of
[`PreprocessStep`](@ref)s, as a single-image function suitable for the
`preprocess` keyword of the sequence drivers. Return `nothing` when there are
no steps. The function returns a new image and leaves its input unchanged.

Use the `steps` form to preview preprocessing before building a recipe: it
applies exactly what [`apply_recipe`](@ref) will. The steps are copied, so
later changes to `steps` do not affect the returned function.
"""
recipe_preprocess(recipe::PIVRecipe) = _preprocess_function(recipe.preprocessing)
recipe_preprocess(steps::AbstractVector{PreprocessStep}) =
    _preprocess_function(deepcopy(collect(steps)))

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

function _recipe_data(r::PIVRecipe)
    Dict{String,Any}(
        "passes" => [_pass_data(p) for p in r.passes],
        "preprocessing" => [Dict{String,Any}("operation" => String(s.operation),
                                             "options" => deepcopy(s.options)) for s in r.preprocessing],
        "mask" => r.mask,
        "roi" => r.roi === nothing ? nothing :
                 [first(r.roi.rows), last(r.roi.rows), first(r.roi.cols), last(r.roi.cols)],
        "scale" => r.scale === nothing ? nothing :
                   Dict{String,Any}(String(k) => getfield(r.scale, k) for k in fieldnames(PhysicalScale)),
        "mode" => String(r.mode),
        "image_type" => string(r.image_type),
        "predictor_smoothing" => r.predictor_smoothing,
        "mask_threshold" => r.mask_threshold)
end

function _recipe_from_data(d)
    steps = [PreprocessStep(Symbol(s["operation"]); (Symbol(k) => v for (k, v) in s["options"])...)
             for s in d["preprocessing"]]
    s = d["scale"]
    scale = s === nothing ? nothing :
            PhysicalScale(s["pixel_size"], s["dt"], s["length_unit"], s["time_unit"])
    roi = d["roi"] === nothing ? nothing : ROI(d["roi"][1]:d["roi"][2], d["roi"][3]:d["roi"][4])
    PIVRecipe([_pass_from_data(p) for p in d["passes"]];
              preprocessing = steps, mask = d["mask"], roi, scale,
              mode = Symbol(d["mode"]),
              image_type = d["image_type"] == "Float32" ? Float32 : Float64,
              predictor_smoothing = d["predictor_smoothing"],
              mask_threshold = d["mask_threshold"])
end

function _write_recipe(file, recipe::PIVRecipe)
    file["recipe_format_version"] = RECIPE_FORMAT_VERSION
    file["recipe"] = _recipe_data(recipe)
    file["recipe_software"] = Dict{String,Any}(
        "hammerhead_version" => string(pkgversion(@__MODULE__)),
        "julia_version" => string(VERSION))
    nothing
end

"""
    save_recipe(path, recipe) -> path

Save a [`PIVRecipe`](@ref) to a JLD2 file. The file also records the
Hammerhead and Julia versions that saved it.
"""
function save_recipe(path::AbstractString, recipe::PIVRecipe)
    jldopen(file -> _write_recipe(file, recipe), path, "w")
    path
end

"""
    load_recipe(path) -> PIVRecipe

Load a recipe saved with [`save_recipe`](@ref), or the recipe that produced a
results file written by [`apply_recipe`](@ref).
"""
function load_recipe(path::AbstractString)
    jldopen(path, "r") do file
        haskey(file, "recipe") ||
            throw(ArgumentError("$path contains no Hammerhead recipe"))
        version = file["recipe_format_version"]
        version == RECIPE_FORMAT_VERSION ||
            throw(ArgumentError("$path has unsupported recipe_format_version $version"))
        _recipe_from_data(file["recipe"])
    end
end

"""
    apply_recipe(recipe, pairs; output=nothing, backend=:cpu, kwargs...)
    apply_recipe(recipe, pairs1, pairs2, dw1, dw2; output=nothing, backend=:cpu, kwargs...)

Process a recording with a recipe's settings. The first form runs planar
PIV on image pairs (paths, matrices, or frame references); the second runs
stereo PIV on two synchronized cameras with their [`ImageDewarper`](@ref)s.

A `:sequence` recipe returns one result per pair, like
[`run_piv_sequence`](@ref) / [`run_piv_stereo_sequence`](@ref); an
`:ensemble` recipe returns one pooled result, like [`run_piv_ensemble`](@ref)
/ [`run_piv_stereo_ensemble`](@ref). When `output` is a file path, results
are saved there and the recipe is stored alongside them, so
`load_recipe(output)` recovers the settings. Remaining keywords (such as
`progress`, `on_result`, `collect_results`, or `threaded`) go to the
underlying driver.
"""
function apply_recipe(recipe::PIVRecipe, pairs::AbstractVector;
                      output::Union{Nothing,AbstractString,Function} = nothing,
                      backend::Symbol = :cpu, kwargs...)
    common = (; preprocess = recipe_preprocess(recipe), image_type = recipe.image_type,
              mask = recipe.mask, scale = recipe.scale, backend,
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
    return _with_recipe(output, recipe) do
        run_piv_sequence(pairs, recipe.passes; output, roi = recipe.roi, common..., kwargs...)
    end
end

function apply_recipe(recipe::PIVRecipe, pairs1::AbstractVector, pairs2::AbstractVector,
                      dw1::ImageDewarper, dw2::ImageDewarper;
                      output::Union{Nothing,AbstractString,Function} = nothing,
                      backend::Symbol = :cpu, kwargs...)
    recipe.roi === nothing ||
        throw(ArgumentError("stereo recipes do not support an ROI; mask the dewarped grid instead"))
    common = (; preprocess = recipe_preprocess(recipe), image_type = recipe.image_type,
              mask = recipe.mask, scale = recipe.scale, backend,
              predictor_smoothing = recipe.predictor_smoothing,
              mask_threshold = recipe.mask_threshold)
    if recipe.mode === :ensemble
        output isa Function &&
            throw(ArgumentError("an ensemble result needs a single output path"))
        result = run_piv_stereo_ensemble(pairs1, pairs2, dw1, dw2, recipe.passes;
                                         common..., kwargs...)
        output === nothing || _save_with_recipe(output, result, recipe)
        return result
    end
    return _with_recipe(output, recipe) do
        run_piv_stereo_sequence(pairs1, pairs2, dw1, dw2, recipe.passes;
                                output, common..., kwargs...)
    end
end

# Store the recipe next to the results even when the batch stops early, so a
# partial file still records how it was produced.
function _with_recipe(run, output, recipe)
    started = time()
    try
        return run()
    finally
        # Skip a stale file the driver never opened (e.g. invalid arguments).
        output isa AbstractString && isfile(output) && mtime(output) >= started - 1 &&
            jldopen(file -> _write_recipe(file, recipe), output, "r+")
    end
end

function _save_with_recipe(path, result, recipe)
    save_results(path, result)
    jldopen(file -> _write_recipe(file, recipe), path, "r+")
    nothing
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
