# Preprocess-preview controller: an ordered list of core `PreprocessStep`s —
# exactly the preprocessing a recipe runs — previewed on a representative
# pair, with a single-window correlation probe. Framework-free. The preview,
# probe, and background computations are pure functions of inputs captured
# when a request is made, so a window can run them on worker tasks (see the
# `runner` field); stale results are dropped by generation counters.

"""
Preprocessing operations, in the order the add-step menu lists them.
"""
const PREPROCESS_OPERATIONS = (:subtract_background, :intensity_cap, :highpass_filter, :clahe,
                               :percentile_stretch, :invert_image, :local_variance_normalize)

const PREPROCESS_LABELS = Dict(:subtract_background => "background subtraction",
                               :intensity_cap => "intensity cap",
                               :highpass_filter => "highpass filter",
                               :clahe => "CLAHE",
                               :percentile_stretch => "percentile stretch",
                               :invert_image => "invert",
                               :local_variance_normalize => "local variance normalization")

# Editable options per operation, in display order (the background matrix of
# :subtract_background comes from `set_background!`, not from a text field).
const PREPROCESS_OPTIONS = Dict(:subtract_background => String[],
                                :intensity_cap => ["n_sigma"],
                                :highpass_filter => ["sigma"],
                                :clahe => ["tiles", "clip_limit", "nbins"],
                                :percentile_stretch => ["low", "high"],
                                :invert_image => String[],
                                :local_variance_normalize => ["sigma", "epsilon"])

const OPTION_LABELS = Dict("n_sigma" => "cap (σ)", "sigma" => "σ (px)", "tiles" => "tiles",
                           "clip_limit" => "clip limit", "nbins" => "bins",
                           "low" => "low (%)", "high" => "high (%)", "epsilon" => "ε")

"""
    preprocess_label(op::Symbol) -> String

Readable name of a preprocessing operation.
"""
preprocess_label(op::Symbol) = get(PREPROCESS_LABELS, op, String(op))

# Runs `job()` and hands `(; value, err)` to `apply`. The default runs both
# inline; a workflow window runs the job on a worker and `apply` on the GUI
# thread.
_inline_runner(job, apply) = apply(_try_job(job))

"""
    PreprocessPreview(; steps = PreprocessStep[], image = nothing, pair = nothing)
    PreprocessPreview(image::AbstractMatrix; steps = PreprocessStep[], pair = nothing)

Edit and preview an ordered list of core `PreprocessStep`s (`steps`, the
recipe's `preprocessing`). `image` and `pair` are frames A and B of a
representative pair; `processed` and `processed2` hold them after the
steps, computed with `recipe_preprocess(steps)` — exactly what the batch
applies — and refreshed whenever the steps or frames change.

Edit the steps with [`add_step!`](@ref), [`remove_step!`](@ref),
[`move_step!`](@ref), [`set_step_option!`](@ref), [`set_steps!`](@ref), and
[`set_background!`](@ref). Invalid edits throw and leave the steps unchanged.

With both frames, a single-window correlation probe is available:
[`click!`](@ref) places it, `probe_window` sizes it (64 px by default,
[`set_probe_window!`](@ref)), and `probe_result` / [`probe_summary`](@ref)
report the window's displacement and peak ratio on the processed frames,
recomputed as the steps change.

`status` reports a failed preview; `error` and `error_step` hold the
message and step index of an edit rejected through a window (see
`edit_step_option!`). `runner` decides where previews and probes are
computed (inline by default). `post[]` (`nothing` by default) is applied to
each frame after the steps, as part of the preview; a stereo workflow sets
it to the shown camera's dewarping, so `processed`, `processed2`, and the
probe are on the dewarped grid.
"""
struct PreprocessPreview
    steps::Observable{Vector{PreprocessStep}}
    image::Observable{Union{Nothing,Matrix{Float32}}}
    image2::Observable{Union{Nothing,Matrix{Float32}}}
    processed::Observable{Union{Nothing,AbstractMatrix}}
    processed2::Observable{Union{Nothing,AbstractMatrix}}
    probe::Observable{Union{Nothing,NTuple{2,Float64}}}
    probe_window::Observable{Int}
    probe_result::Observable{Union{Nothing,NamedTuple}}
    status::Observable{String}
    error::Observable{String}
    error_step::Observable{Int}
    runner::Base.RefValue{Any}
    post::Base.RefValue{Any}
    generation::Base.RefValue{Int}
    probe_generation::Base.RefValue{Int}
end

function PreprocessPreview(; steps = PreprocessStep[], image = nothing, pair = nothing,
                           runner = _inline_runner)
    pp = PreprocessPreview(Observable(collect(PreprocessStep, steps)),
                           Observable{Union{Nothing,Matrix{Float32}}}(_frame32(image)),
                           Observable{Union{Nothing,Matrix{Float32}}}(_frame32(pair)),
                           Observable{Union{Nothing,AbstractMatrix}}(nothing),
                           Observable{Union{Nothing,AbstractMatrix}}(nothing),
                           Observable{Union{Nothing,NTuple{2,Float64}}}(nothing),
                           Observable(64), Observable{Union{Nothing,NamedTuple}}(nothing),
                           Observable(""), Observable(""), Observable(0),
                           Ref{Any}(runner), Ref{Any}(nothing), Ref(0), Ref(0))
    for obs in (pp.steps, pp.image, pp.image2)
        on(_ -> _request_preview!(pp), obs)
    end
    on(_ -> _request_probe!(pp), pp.probe)
    on(_ -> _request_probe!(pp), pp.probe_window)
    _request_preview!(pp)
    return pp
end

PreprocessPreview(image::AbstractMatrix; kwargs...) = PreprocessPreview(; image, kwargs...)

_frame32(::Nothing) = nothing
_frame32(img::AbstractMatrix{<:Real}) = convert(Matrix{Float32}, img)

function Base.show(io::IO, pp::PreprocessPreview)
    img = pp.image[]
    print(io, "PreprocessPreview(", img === nothing ? "no image" : "$(size(img)) image",
          ", $(length(pp.steps[])) step", length(pp.steps[]) == 1 ? "" : "s", ")")
end

# ---------------------------------------------------------------- frames

"""
    set_image!(pp::PreprocessPreview, image)
    set_pair!(pp::PreprocessPreview, image)
    set_frames!(pp::PreprocessPreview, image, pair)

Set frame A (`set_image!`), frame B (`set_pair!`, which enables the
correlation probe), or both at once (`nothing` clears a frame). Frames are
kept as `Float32` matrices; frame B must match frame A's size.
"""
set_image!(pp::PreprocessPreview, image) = set_frames!(pp, image, pp.image2[])
set_pair!(pp::PreprocessPreview, image) = set_frames!(pp, pp.image[], image)

function set_frames!(pp::PreprocessPreview, a, b)
    a !== nothing && b !== nothing && size(a) != size(b) &&
        throw(ArgumentError("pair frame size $(size(b)) does not match the preview image $(size(a))"))
    a32, b32 = _frame32(a), _frame32(b)
    (a32 === pp.image[] && b32 === pp.image2[]) && return pp
    pp.image2.val = b32
    pp.image[] = a32                      # one notification refreshes both
    return pp
end

# ---------------------------------------------------------------- step edits

# A small fixed image for checking option values: applying a step to it
# runs the core function's own argument validation.
const _TRIAL_IMAGE = [Float32(sin(0.7i) * cos(0.45j) + 0.01i) for i in 1:24, j in 1:24]

function _checked_step(op::Symbol, options)
    step = PreprocessStep(op; options...)
    op === :subtract_background || recipe_preprocess([step])(_TRIAL_IMAGE)
    return step
end

function _parse_option(key::String, value)
    if key == "tiles"
        m = value isa AbstractString ? match(r"^\s*(\d+)\s*(?:(?:[,x×]|\s)\s*(\d+))?\s*$", value) : nothing
        t = value isa AbstractString ?
            (m === nothing ? Any[] : [parse(Int, c) for c in m.captures if c !== nothing]) :
            value isa Integer ? [Int(value)] : collect(value)
        (isempty(t) || length(t) > 2 || any(v -> !(v isa Integer) || v < 1, t)) &&
            throw(ArgumentError("tiles must be one or two positive integers, got \"$value\""))
        return length(t) == 1 ? [Int(t[1]), Int(t[1])] : Int.(t)
    elseif key == "nbins"
        n = value isa AbstractString ? tryparse(Int, strip(value)) :
            value isa Real && isinteger(value) ? Int(value) : nothing
        n === nothing && throw(ArgumentError("bins must be an integer, got \"$value\""))
        return n
    end
    v = value isa AbstractString ? tryparse(Float64, strip(value)) :
        value isa Real ? Float64(value) : nothing
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("$(get(OPTION_LABELS, key, key)) must be a number, got \"$value\""))
    return v
end

function _set_steps!(pp::PreprocessPreview, steps::Vector{PreprocessStep})
    steps == pp.steps[] && return pp
    pp.steps[] = steps
    return pp
end

"""
    add_step!(pp::PreprocessPreview, op::Symbol; options...)

Append a preprocessing step (see `PreprocessStep` for operations and
options). Background subtraction is added by [`set_background!`](@ref).
"""
function add_step!(pp::PreprocessPreview, op::Symbol; options...)
    op === :subtract_background && !haskey(options, :background) &&
        throw(ArgumentError("estimate a background first (it adds the subtraction step)"))
    step = _checked_step(op, options)
    return _set_steps!(pp, [pp.steps[]; step])
end

_clear_error!(pp::PreprocessPreview) =
    (isempty(pp.error[]) || (pp.error[] = ""); pp.error_step[] == 0 || (pp.error_step[] = 0); pp)

function _check_index(pp::PreprocessPreview, i::Integer)
    1 <= i <= length(pp.steps[]) ||
        throw(ArgumentError("no preprocessing step $i (there are $(length(pp.steps[])))"))
    return Int(i)
end

"""
    remove_step!(pp::PreprocessPreview, i)

Remove step `i`.
"""
function remove_step!(pp::PreprocessPreview, i::Integer)
    _check_index(pp, i)
    steps = copy(pp.steps[])
    deleteat!(steps, i)
    _clear_error!(pp)
    return _set_steps!(pp, steps)
end

"""
    move_step!(pp::PreprocessPreview, i, offset)

Move step `i` by `offset` positions (negative = earlier), clamped to the
ends. Order matters — e.g. stretching before or after inversion gives
different images.
"""
function move_step!(pp::PreprocessPreview, i::Integer, offset::Integer)
    _check_index(pp, i)
    steps = copy(pp.steps[])
    j = clamp(i + offset, 1, length(steps))
    i == j && return pp
    s = steps[i]
    deleteat!(steps, i)
    insert!(steps, j, s)
    _clear_error!(pp)
    return _set_steps!(pp, steps)
end

"""
    set_step_option!(pp::PreprocessPreview, i, key, value)

Set option `key` of step `i` from a value or its text form (`tiles` also
accepts `"8, 8"` or `"8x8"`). The new step is validated by the core
(`PreprocessStep` and the operation itself); an invalid value throws and
leaves the steps unchanged.
"""
function set_step_option!(pp::PreprocessPreview, i::Integer, key, value)
    step = pp.steps[][_check_index(pp, i)]
    k = String(key)
    k in PREPROCESS_OPTIONS[step.operation] ||
        throw(ArgumentError("$(preprocess_label(step.operation)) has no option \"$k\""))
    options = Dict{Symbol,Any}(Symbol(o) => v for (o, v) in step.options)
    options[Symbol(k)] = _parse_option(k, value)
    new = _checked_step(step.operation, options)
    new == step && return pp
    steps = copy(pp.steps[])
    steps[i] = new
    return _set_steps!(pp, steps)
end

"""
    set_steps!(pp::PreprocessPreview, steps)

Replace all steps (e.g. with a recipe's `preprocessing`).
"""
set_steps!(pp::PreprocessPreview, steps::AbstractVector{PreprocessStep}) =
    _set_steps!(pp, collect(PreprocessStep, steps))

"""
    estimate_background(frames; method = :min) -> Matrix{Float64}

Background of an iterable of frames (matrices and/or image paths) with core
`compute_background`. Pure; a window runs it on a worker task.
"""
estimate_background(frames; method::Symbol = :min) =
    compute_background((f isa AbstractMatrix ? f : load_image(String(f)) for f in frames);
                       method)

"""
    set_background!(pp::PreprocessPreview, background::AbstractMatrix)
    set_background!(pp::PreprocessPreview, frames; method = :min)
    set_background!(pp::PreprocessPreview, nothing)

Subtract `background` (or the background of `frames`, see
[`estimate_background`](@ref)): replaces the background of an existing
`:subtract_background` step, or inserts one first. `nothing` removes the
subtraction step.
"""
function set_background!(pp::PreprocessPreview, ::Nothing)
    return _set_steps!(pp, filter(s -> s.operation !== :subtract_background, pp.steps[]))
end

function set_background!(pp::PreprocessPreview, bg::AbstractMatrix{<:Real})
    step = PreprocessStep(:subtract_background; background = bg)
    steps = copy(pp.steps[])
    i = findfirst(s -> s.operation === :subtract_background, steps)
    i === nothing ? pushfirst!(steps, step) : (steps[i] = step)
    return _set_steps!(pp, steps)
end

set_background!(pp::PreprocessPreview, frames; method::Symbol = :min) =
    set_background!(pp, estimate_background(frames; method))

"""
    step_options(step::PreprocessStep) -> Vector{Pair{String,String}}

The editable options of a step and their values as text, in display order
(long decimals rounded to 6 significant digits).
"""
function step_options(step::PreprocessStep)
    fmt(v::Union{AbstractVector,Tuple}) = join((x isa Real ? display_number(x) : string(x) for x in v), ", ")
    fmt(v::Real) = display_number(v)
    fmt(v) = string(v)
    return [k => fmt(step.options[k]) for k in PREPROCESS_OPTIONS[step.operation]]
end

"""
    build_preprocess(pp::PreprocessPreview) -> Union{Nothing,Function}
    apply_pipeline(pp::PreprocessPreview, img) -> Matrix

`recipe_preprocess(pp.steps[])` — the batch's preprocessing function, a
snapshot of the current steps (`nothing` without steps) — and its result on
`img` (a copy when there are no steps; the input is never mutated).
"""
build_preprocess(pp::PreprocessPreview) = recipe_preprocess(pp.steps[])

function apply_pipeline(pp::PreprocessPreview, img::AbstractMatrix{<:Real})
    f = build_preprocess(pp)
    return f === nothing ? copy(img) : f(img)
end

"""
    pipeline_summary(pp::PreprocessPreview) -> String

The steps in order (\"no preprocessing\" without steps).
"""
pipeline_summary(pp::PreprocessPreview) = pipeline_summary(pp.steps[])
pipeline_summary(steps::AbstractVector{PreprocessStep}) =
    isempty(steps) ? "no preprocessing" : join((preprocess_label(s.operation) for s in steps), " → ")

# ---------------------------------------------------------------- preview

"""
    preview_frames(steps, a, b; post = nothing) -> (processed_a, processed_b)

Frames `a` and `b` (`b` may be `nothing`) after `steps`, with
`recipe_preprocess`, then `post` (a function of one frame) when given.
Pure; a window runs it on a worker task.
"""
function preview_frames(steps::AbstractVector{PreprocessStep}, a::AbstractMatrix, b; post = nothing)
    f = recipe_preprocess(steps)
    g = f === nothing ? post : post === nothing ? f : post ∘ f
    g === nothing && return (a, b)
    return (g(a), b === nothing ? nothing : g(b))
end

function _request_preview!(pp::PreprocessPreview)
    g = (pp.generation[] += 1)
    a, b, steps, post = pp.image[], pp.image2[], pp.steps[], pp.post[]
    if a === nothing || (isempty(steps) && post === nothing)
        # nothing to compute: the raw frames are the preview
        _finish_preview!(pp, g, (; value = (a, b), err = nothing))
        return pp
    end
    pp.runner[](() -> preview_frames(steps, a, b; post), out -> _finish_preview!(pp, g, out))
    return pp
end

# Frames and post-processing together: one preview request for both.
function _set_preview_input!(pp::PreprocessPreview, a, b, post)
    changed = post !== pp.post[]
    pp.post[] = post
    a32, b32 = _frame32(a), _frame32(b)
    if a32 !== pp.image[] || b32 !== pp.image2[]
        set_frames!(pp, a32, b32)
    elseif changed
        _request_preview!(pp)
    end
    return pp
end

function _finish_preview!(pp::PreprocessPreview, g::Int, out)
    g == pp.generation[] || return pp         # a newer request supersedes this one
    if out.err === nothing
        pp.processed2.val = out.value[2]
        pp.processed[] = out.value[1]
        isempty(pp.status[]) || (pp.status[] = "")
    else
        pp.processed2.val = nothing
        pp.processed[] = nothing
        pp.status[] = "preview failed: " * _errmsg(out.err)
    end
    _request_probe!(pp)
    return pp
end

# ---------------------------------------------------------------- probe

"""
    click!(pp::PreprocessPreview, x::Real, y::Real)

Place the correlation probe at data-space `(x, y)` (the window is centered
there, clamped to stay inside the frame).
"""
click!(pp::PreprocessPreview, x::Real, y::Real) =
    (pp.probe[] = (Float64(x), Float64(y)); pp)

"""
    clear_probe!(pp::PreprocessPreview)

Remove the correlation probe.
"""
clear_probe!(pp::PreprocessPreview) = (pp.probe[] === nothing || (pp.probe[] = nothing); pp)

"""
    set_probe_window!(pp::PreprocessPreview, size)

Set the probe's interrogation window size (an even integer ≥ 8, or its
string form).
"""
function set_probe_window!(pp::PreprocessPreview, ws::Integer)
    (ws >= 8 && iseven(ws)) ||
        throw(ArgumentError("probe window must be an even integer ≥ 8, got $ws"))
    pp.probe_window[] == ws || (pp.probe_window[] = Int(ws))
    return pp
end
function set_probe_window!(pp::PreprocessPreview, s::AbstractString)
    ws = tryparse(Int, strip(s))
    ws === nothing &&
        throw(ArgumentError("probe window must be an even integer ≥ 8, got \"$s\""))
    return set_probe_window!(pp, ws)
end

"""
    probe_rect(loc, window, size) -> Union{Nothing,NamedTuple}

Top-left corner `(x0, y0)` (column, row) of a `window`-pixel probe centered
at `loc = (x, y)` and clamped into an image of `size`, and whether it was
`clamped`; `nothing` when the window does not fit.
"""
function probe_rect(loc::NTuple{2,Real}, ws::Integer, sz::Dims{2})
    nr, nc = sz
    (ws > nr || ws > nc) && return nothing
    x, y = loc
    half = ws ÷ 2
    r = round(Int, y) - half + 1
    c = round(Int, x) - half + 1
    r0 = clamp(r, 1, nr - ws + 1)
    c0 = clamp(c, 1, nc - ws + 1)
    return (; x0 = c0, y0 = r0, window = Int(ws), clamped = r0 != r || c0 != c)
end

"""
    probe_correlation(a, b, loc, window) -> Union{Nothing,NamedTuple}

Correlate one `window`-pixel window of frames `a` and `b` (already
preprocessed) at `loc` with `run_piv` at the accuracy defaults (padding,
Gaussian weighting). Returns `du`, `dv`, `peak_ratio` and the window's
placement (see [`probe_rect`](@ref)), or `nothing` when the window does
not fit. Pure; a window runs it on a worker task.
"""
function probe_correlation(a::AbstractMatrix, b::AbstractMatrix, loc::NTuple{2,Real}, ws::Integer)
    size(a) == size(b) || return nothing
    rect = probe_rect(loc, ws, size(a))
    rect === nothing && return nothing
    rows, cols = rect.y0:(rect.y0 + ws - 1), rect.x0:(rect.x0 + ws - 1)
    r = run_piv(a[rows, cols], b[rows, cols],
                PIVParameters(window_size = ws, overlap = (0, 0), padding = true,
                              apodization = :gauss, uod_enable = false))
    return (; du = Float64(r.u[1, 1]), dv = Float64(r.v[1, 1]),
            peak_ratio = Float64(r.peak_ratio[1, 1]), rect...)
end

function _request_probe!(pp::PreprocessPreview)
    g = (pp.probe_generation[] += 1)
    a, b, loc, ws = pp.processed[], pp.processed2[], pp.probe[], pp.probe_window[]
    if a === nothing || b === nothing || loc === nothing
        pp.probe_result[] === nothing || (pp.probe_result[] = nothing)
        return pp
    end
    pp.runner[](() -> probe_correlation(a, b, loc, ws), out -> _finish_probe!(pp, g, out))
    return pp
end

function _finish_probe!(pp::PreprocessPreview, g::Int, out)
    g == pp.probe_generation[] || return pp
    if out.err === nothing
        pp.probe_result[] = out.value
    else
        pp.probe_result[] = nothing
        pp.status[] = "probe failed: " * _errmsg(out.err)
    end
    return pp
end

"""
    probe_summary(pp::PreprocessPreview) -> String

The probe's displacement and peak ratio when it is placed, otherwise what is
missing (pair frame, click, or a window that fits the frame).
"""
function probe_summary(pp::PreprocessPreview)
    pp.image2[] === nothing && return "load a pair to probe the correlation"
    pp.probe[] === nothing && return "click the image to place the probe"
    res = pp.probe_result[]
    if res === nothing
        sz = size(something(pp.processed[], pp.image[]))
        return pp.probe_window[] > minimum(sz) ?
            "probe window ($(pp.probe_window[]) px) does not fit the frame" : "correlating…"
    end
    return string("du = ", _fmt(res.du), " px, dv = ", _fmt(res.dv), " px\n",
                  "peak ratio = ", _fmt(res.peak_ratio),
                  res.clamped ? "\n(window clamped to the frame)" : "")
end
