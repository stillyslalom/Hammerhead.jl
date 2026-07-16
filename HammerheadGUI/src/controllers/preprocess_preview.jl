# Preprocess-preview controller: an ordered, toggleable preprocessing
# pipeline over a representative image, with live preview state and a
# `build_preprocess` hand-off producing the closure `run_piv_sequence` /
# `run_ptv_sequence` take. Framework-free — views render the raw/processed
# observables and push toggles/params/reorders into the API below.

"""
    PreprocStep(name, enabled, params)

One pipeline entry: a core preprocessing step (`name`), whether it is
applied (`enabled`), and its numeric parameters (`params`, a
`Dict{Symbol,Float64}`). Built by [`PreprocessPreview`](@ref); mutate through
[`enable_step!`](@ref), [`set_step_param!`](@ref), and [`move_step!`](@ref)
so the preview refreshes.
"""
mutable struct PreprocStep
    name::Symbol
    enabled::Bool
    params::Dict{Symbol,Float64}
end

# The supported catalogue: name => (display label, default params). Order
# here is the default pipeline order; `move_step!` reorders per controller.
const PREPROC_CATALOG = [
    :subtract_background => ("background subtraction", Pair{Symbol,Float64}[]),
    :intensity_cap => ("intensity cap", [:n_sigma => 2.0]),
    :highpass_filter => ("highpass filter", [:sigma => 3.0]),
    :clahe => ("CLAHE", [:clip_limit => 2.0]),
    :percentile_stretch => ("percentile stretch", [:low => 1.0, :high => 99.0]),
    :invert_image => ("invert", Pair{Symbol,Float64}[]),
    :local_variance_normalize => ("local variance norm.", [:sigma => 3.0]),
]

preproc_label(name::Symbol) =
    last(PREPROC_CATALOG[findfirst(p -> first(p) === name, PREPROC_CATALOG)])[1]

"""
    PreprocessPreview(image; enabled = Symbol[])
    PreprocessPreview(path::AbstractString; kwargs...)

Controller for the preprocessing-pipeline preview: a representative `image`
(matrix, or a path loaded with `Hammerhead.load_image`), the ordered step
list `steps` (every supported step, disabled unless listed in `enabled`),
an optional `background` for the subtraction step (see
[`set_background!`](@ref)), and the live `processed` observable — the
pipeline applied to `image`, recomputed on every change.

Supported steps, in default order: `:subtract_background`,
`:intensity_cap` (`n_sigma`), `:highpass_filter` (`sigma`), `:clahe`
(`clip_limit`), `:percentile_stretch` (`low`/`high`), `:invert_image`, and
`:local_variance_normalize` (`sigma`) — the core preprocessing set.
[`build_preprocess`](@ref) exports the composed closure for the batch
drivers.
"""
struct PreprocessPreview
    image::Observable{Matrix{Float64}}
    background::Observable{Union{Nothing,Matrix{Float64}}}
    steps::Observable{Vector{PreprocStep}}
    processed::Observable{Matrix{Float64}}
end

function PreprocessPreview(image::AbstractMatrix{<:Real}; enabled = Symbol[])
    steps = [PreprocStep(name, name in enabled, Dict{Symbol,Float64}(defaults))
             for (name, (_, defaults)) in PREPROC_CATALOG]
    img = Matrix{Float64}(image)
    pp = PreprocessPreview(Observable(img),
                           Observable{Union{Nothing,Matrix{Float64}}}(nothing),
                           Observable(steps),
                           Observable(copy(img)))
    recompute = _ -> (pp.processed[] = apply_pipeline(pp, pp.image[]))
    on(recompute, pp.image)
    on(recompute, pp.background)
    on(recompute, pp.steps)
    notify(pp.steps)   # populate `processed` for the initial state
    return pp
end

PreprocessPreview(path::AbstractString; kwargs...) =
    PreprocessPreview(load_image(path); kwargs...)

function Base.show(io::IO, pp::PreprocessPreview)
    n = count(s -> s.enabled, pp.steps[])
    print(io, "PreprocessPreview($(size(pp.image[])) image, $n of ",
          length(pp.steps[]), " steps enabled)")
end

function _step(pp::PreprocessPreview, name::Symbol)
    i = findfirst(s -> s.name === name, pp.steps[])
    i === nothing && throw(ArgumentError("unknown preprocessing step :$name"))
    return pp.steps[][i]
end

"""
    set_image!(pp::PreprocessPreview, image_or_path)

Replace the representative preview image (matrix, or a path loaded with
`Hammerhead.load_image`); the processed preview refreshes.
"""
set_image!(pp::PreprocessPreview, image::AbstractMatrix{<:Real}) =
    (pp.image[] = Matrix{Float64}(image); pp)
set_image!(pp::PreprocessPreview, path::AbstractString) =
    set_image!(pp, load_image(path))

"""
    set_background!(pp::PreprocessPreview, frames; method = :min)
    set_background!(pp::PreprocessPreview, nothing)

Compute the background for the `:subtract_background` step from an iterable
of frames (matrices and/or paths; `Hammerhead.compute_background` with
`method = :min` or `:mean`). Passing `nothing` clears it and disables the
subtraction step.
"""
function set_background!(pp::PreprocessPreview, ::Nothing)
    _step(pp, :subtract_background).enabled = false
    pp.background[] = nothing
    return pp
end

function set_background!(pp::PreprocessPreview, frames; method::Symbol = :min)
    imgs = (f isa AbstractMatrix ? Matrix{Float64}(f) : load_image(String(f))
            for f in frames)
    pp.background[] = compute_background(imgs; method)
    return pp
end

"""
    enable_step!(pp::PreprocessPreview, name::Symbol, on::Bool = true)

Enable or disable a pipeline step. Enabling `:subtract_background` requires
a background (see [`set_background!`](@ref)).
"""
function enable_step!(pp::PreprocessPreview, name::Symbol, on::Bool = true)
    on && name === :subtract_background && pp.background[] === nothing &&
        throw(ArgumentError("compute a background first (set_background!)"))
    s = _step(pp, name)
    s.enabled == on || (s.enabled = on; notify(pp.steps))
    return pp
end

"""
    set_step_param!(pp::PreprocessPreview, name::Symbol, param::Symbol, value)

Set a step parameter from a number or its string form. Invalid values
(unparsable, or rejected by the core preprocessing function when the preview
recomputes) are reverted and rethrown, so the pipeline never sticks in a
broken state.
"""
function set_step_param!(pp::PreprocessPreview, name::Symbol, param::Symbol, value::Real)
    isfinite(value) || throw(ArgumentError("$param must be finite, got $value"))
    s = _step(pp, name)
    haskey(s.params, param) ||
        throw(ArgumentError("step :$name has no parameter :$param"))
    old = s.params[param]
    old == Float64(value) && return pp
    s.params[param] = Float64(value)
    try
        notify(pp.steps)   # recompute validates via the core function
    catch
        s.params[param] = old
        notify(pp.steps)
        rethrow()
    end
    return pp
end

function set_step_param!(pp::PreprocessPreview, name::Symbol, param::Symbol,
                         value::AbstractString)
    v = tryparse(Float64, strip(value))
    v === nothing &&
        throw(ArgumentError("$param must be a number, got \"$value\""))
    return set_step_param!(pp, name, param, v)
end

"""
    move_step!(pp::PreprocessPreview, name::Symbol, offset::Integer)

Move a step by `offset` positions in the pipeline order (negative = earlier),
clamped to the ends. Order matters — e.g. stretching before or after
inversion gives different images.
"""
function move_step!(pp::PreprocessPreview, name::Symbol, offset::Integer)
    steps = pp.steps[]
    i = findfirst(s -> s.name === name, steps)
    i === nothing && throw(ArgumentError("unknown preprocessing step :$name"))
    j = clamp(i + offset, 1, length(steps))
    i == j && return pp
    s = steps[i]
    deleteat!(steps, i)
    insert!(steps, j, s)
    notify(pp.steps)
    return pp
end

# Apply one step in place. `img` is always a private buffer (see
# apply_pipeline), so the mutating core forms are safe and allocation-free.
function _apply_step!(img::Matrix{Float64}, s::PreprocStep, background)
    p = s.params
    if s.name === :subtract_background
        background === nothing &&
            throw(ArgumentError("the :subtract_background step needs a background (set_background!)"))
        subtract_background!(img, background)
    elseif s.name === :intensity_cap
        intensity_cap!(img; n_sigma = p[:n_sigma])
    elseif s.name === :highpass_filter
        highpass_filter!(img; sigma = p[:sigma])
    elseif s.name === :clahe
        clahe!(img; clip_limit = p[:clip_limit])
    elseif s.name === :percentile_stretch
        percentile_stretch!(img; low = p[:low], high = p[:high])
    elseif s.name === :invert_image
        invert_image!(img)
    elseif s.name === :local_variance_normalize
        local_variance_normalize!(img; sigma = p[:sigma])
    else
        throw(ArgumentError("unknown preprocessing step :$(s.name)"))
    end
    return img
end

"""
    apply_pipeline(pp::PreprocessPreview, img) -> Matrix{Float64}

Apply the enabled steps, in order, to a fresh float copy of `img` (the input
is never mutated — same contract as the [`build_preprocess`](@ref) closure).
"""
function apply_pipeline(pp::PreprocessPreview, img::AbstractMatrix{<:Real})
    out = Matrix{Float64}(img)
    bg = pp.background[]
    for s in pp.steps[]
        s.enabled && _apply_step!(out, s, bg)
    end
    return out
end

"""
    build_preprocess(pp::PreprocessPreview) -> Union{Nothing,Function}

The composed pipeline as the `preprocess` closure the batch drivers take
(`run_piv_sequence(...; preprocess = build_preprocess(pp))`), or `nothing`
when no step is enabled. The closure copies each frame before the in-place
steps run, so in-memory matrix pairs are never mutated (frames loaded from
paths pay one extra copy — negligible next to the analysis). The step list
and background are snapshotted at build time: later form edits do not affect
a batch already running.
"""
function build_preprocess(pp::PreprocessPreview)
    steps = [PreprocStep(s.name, true, copy(s.params))
             for s in pp.steps[] if s.enabled]
    isempty(steps) && return nothing
    bg = pp.background[]
    any(s -> s.name === :subtract_background, steps) && bg === nothing &&
        throw(ArgumentError("the :subtract_background step needs a background (set_background!)"))
    return img -> begin
        out = Matrix{Float64}(img)
        for s in steps
            _apply_step!(out, s, bg)
        end
        out
    end
end

"""
    pipeline_summary(pp::PreprocessPreview) -> String

One-line summary of the enabled steps in order (\"no preprocessing\" when
none are enabled).
"""
function pipeline_summary(pp::PreprocessPreview)
    names = [preproc_label(s.name) for s in pp.steps[] if s.enabled]
    return isempty(names) ? "no preprocessing" : join(names, " → ")
end
