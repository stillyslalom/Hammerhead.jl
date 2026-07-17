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
    PreprocessPreview(image; pair = nothing, enabled = Symbol[])
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

With `pair` (the representative frame's correlation partner — matrix or
path, see [`set_pair!`](@ref)) a single-window correlation probe becomes
available: [`click!`](@ref) places it, `probe_window` sizes it (64 px by
default, [`set_probe_window!`](@ref)), and `probe_result` /
[`probe_summary`](@ref) report the window's displacement and peak ratio,
recomputed live as the pipeline changes — a direct readout of what each
preprocessing choice does to the correlation.
"""
struct PreprocessPreview
    image::Observable{Matrix{Float64}}
    background::Observable{Union{Nothing,Matrix{Float64}}}
    steps::Observable{Vector{PreprocStep}}
    processed::Observable{Matrix{Float64}}
    image2::Observable{Union{Nothing,Matrix{Float64}}}
    probe::Observable{Union{Nothing,NTuple{2,Float64}}}
    probe_window::Observable{Int}
    probe_result::Observable{Union{Nothing,NamedTuple}}
end

function PreprocessPreview(image::AbstractMatrix{<:Real}; pair = nothing,
                           enabled = Symbol[])
    steps = [PreprocStep(name, name in enabled, Dict{Symbol,Float64}(defaults))
             for (name, (_, defaults)) in PREPROC_CATALOG]
    img = Matrix{Float64}(image)
    pp = PreprocessPreview(Observable(img),
                           Observable{Union{Nothing,Matrix{Float64}}}(nothing),
                           Observable(steps),
                           Observable(copy(img)),
                           Observable{Union{Nothing,Matrix{Float64}}}(nothing),
                           Observable{Union{Nothing,NTuple{2,Float64}}}(nothing),
                           Observable(64),
                           Observable{Union{Nothing,NamedTuple}}(nothing))
    recompute = _ -> begin
        pp.processed[] = apply_pipeline(pp, pp.image[])
        _update_probe!(pp)
    end
    on(recompute, pp.image)
    on(recompute, pp.background)
    on(recompute, pp.steps)
    on(_ -> _update_probe!(pp), pp.image2)
    on(_ -> _update_probe!(pp), pp.probe)
    on(_ -> _update_probe!(pp), pp.probe_window)
    pair === nothing || set_pair!(pp, pair)
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
    set_pair!(pp::PreprocessPreview, image_or_path)
    set_pair!(pp::PreprocessPreview, nothing)

Set the representative frame's correlation partner (matrix, or a path loaded
with `Hammerhead.load_image`; must match the preview image's size) — this
enables the correlation probe. `nothing` clears it.
"""
set_pair!(pp::PreprocessPreview, ::Nothing) = (pp.image2[] = nothing; pp)
function set_pair!(pp::PreprocessPreview, image::AbstractMatrix{<:Real})
    size(image) == size(pp.image[]) ||
        throw(ArgumentError("pair frame size $(size(image)) does not match the preview image $(size(pp.image[]))"))
    pp.image2[] = Matrix{Float64}(image)
    return pp
end
set_pair!(pp::PreprocessPreview, path::AbstractString) =
    set_pair!(pp, load_image(path))

"""
    click!(pp::PreprocessPreview, x::Real, y::Real)

Place the correlation probe at data-space `(x, y)` on the processed image
(the window is centered there, clamped to stay inside the frame).
"""
click!(pp::PreprocessPreview, x::Real, y::Real) =
    (pp.probe[] = (Float64(x), Float64(y)); pp)

"""
    clear_probe!(pp::PreprocessPreview)

Remove the correlation probe.
"""
clear_probe!(pp::PreprocessPreview) = (pp.probe[] = nothing; pp)

"""
    set_probe_window!(pp::PreprocessPreview, size)

Set the probe's interrogation window size (an even integer ≥ 8, or its
string form).
"""
function set_probe_window!(pp::PreprocessPreview, ws::Integer)
    (ws >= 8 && iseven(ws)) ||
        throw(ArgumentError("probe window must be an even integer ≥ 8, got $ws"))
    pp.probe_window[] = Int(ws)
    return pp
end
function set_probe_window!(pp::PreprocessPreview, s::AbstractString)
    ws = tryparse(Int, strip(s))
    ws === nothing &&
        throw(ArgumentError("probe window must be an even integer ≥ 8, got \"$s\""))
    return set_probe_window!(pp, ws)
end

# Recompute the probe correlation: run the pipeline on both frames (same
# semantics as the batch closure), crop the probe window from each, and run
# a single-window run_piv at the accuracy defaults. The window is clamped to
# stay inside the frame (reported via `clamped`); `probe_result` is
# `nothing` whenever the probe cannot run (no pair, no click, mismatched
# sizes, window larger than the frame) — probe_summary explains which.
function _update_probe!(pp::PreprocessPreview)
    img2 = pp.image2[]
    loc = pp.probe[]
    if img2 === nothing || loc === nothing || size(img2) != size(pp.image[])
        pp.probe_result[] === nothing || (pp.probe_result[] = nothing)
        return pp
    end
    ws = pp.probe_window[]
    nr, nc = size(pp.image[])
    if ws > nr || ws > nc
        pp.probe_result[] === nothing || (pp.probe_result[] = nothing)
        return pp
    end
    x, y = loc
    half = ws ÷ 2
    r0 = clamp(round(Int, y) - half + 1, 1, nr - ws + 1)
    c0 = clamp(round(Int, x) - half + 1, 1, nc - ws + 1)
    clamped = r0 != round(Int, y) - half + 1 || c0 != round(Int, x) - half + 1
    rows, cols = r0:(r0 + ws - 1), c0:(c0 + ws - 1)
    a = pp.processed[][rows, cols]              # frame A is already processed
    b = apply_pipeline(pp, img2)[rows, cols]
    r = run_piv(a, b, PIVParameters(window_size = ws, overlap = (0, 0),
                                    padding = true, apodization = :gauss,
                                    uod_enable = false))
    pp.probe_result[] = (; du = Float64(r.u[1, 1]), dv = Float64(r.v[1, 1]),
                         peak_ratio = Float64(r.peak_ratio[1, 1]),
                         x0 = c0, y0 = r0, window = ws, clamped)
    return pp
end

"""
    probe_summary(pp::PreprocessPreview) -> String

Human-readable state of the correlation probe: the live displacement and
peak ratio when it is placed, otherwise what is missing (pair frame, click,
or a window that fits the frame).
"""
function probe_summary(pp::PreprocessPreview)
    pp.image2[] === nothing && return "load a pair frame to probe the correlation"
    size(pp.image2[]) == size(pp.image[]) || return "pair frame size differs from the preview image"
    pp.probe[] === nothing && return "click the processed image to place the probe"
    res = pp.probe_result[]
    res === nothing && return "probe window ($(pp.probe_window[]) px) does not fit the frame"
    return string("du = ", _fmt(res.du), " px, dv = ", _fmt(res.dv), " px\n",
                  "peak ratio = ", _fmt(res.peak_ratio),
                  res.clamped ? "\n(window clamped to the frame)" : "")
end

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
