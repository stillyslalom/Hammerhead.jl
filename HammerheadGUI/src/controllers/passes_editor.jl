# Pass-schedule editor: an explicit list of PIVParameters. Effort presets
# fill the list; it stays linked to the preset (and refills when the
# analysis size or mode changes) until a pass is edited.

const PASS_FIELDS = (:window, :search, :overlap, :iterations)
const SHARED_OPTIONS = (:correlation, :subpixel, :accuracy, :padding, :apodization, :uod_enable,
                        :uod_threshold, :uod_neighborhood, :min_peak_ratio, :replace_outliers)

"""
    PassesEditor(; image_size = nothing, mode = :sequence, image_type = Float64)

The pass schedule of a workflow. `passes` always holds explicit
`PIVParameters`; `preset` names the effort level the schedule came from
while it is unedited (`nothing` once edited or loaded from a recipe).
`mode` is the analysis mode (one of [`ANALYSIS_MODES`](@ref): `:sequence`,
`:ensemble`, or the particle modes `:ptv`/`:tracking`, where the schedule is
the PIV predictor), `image_type` the processing precision,
and `error` the message from the last rejected edit (empty when none).
"""
struct PassesEditor
    passes::Observable{Vector{PIVParameters}}
    preset::Observable{Union{Nothing,Symbol}}
    mode::Observable{Symbol}
    image_type::Observable{DataType}
    image_size::Observable{Union{Nothing,Dims{2}}}
    error::Observable{String}
end

function PassesEditor(; image_size = nothing, mode::Symbol = :sequence,
                      image_type::DataType = Float64, preset::Symbol = :medium)
    _check_mode(mode)
    image_type in (Float32, Float64) ||
        throw(ArgumentError("image_type must be Float32 or Float64, got $image_type"))
    pe = PassesEditor(Observable(PIVParameters[]), Observable{Union{Nothing,Symbol}}(nothing),
                      Observable(mode), Observable{DataType}(image_type),
                      Observable{Union{Nothing,Dims{2}}}(image_size), Observable(""))
    fill_preset!(pe, preset)
    return pe
end

function Base.show(io::IO, pe::PassesEditor)
    ws = join((p.window_size[1] for p in pe.passes[]), "→")
    print(io, "PassesEditor($ws, ", pe.preset[] === nothing ? "custom" : ":$(pe.preset[])", ")")
end

"""
    fill_preset!(pe::PassesEditor, level)

Replace the schedule with the `:low`, `:medium`, or `:high` effort preset
(`Hammerhead.effort_schedule`), sized to `image_size` and `mode`.
"""
function fill_preset!(pe::PassesEditor, level::Symbol)
    passes = effort_schedule(level; ensemble = pe.mode[] === :ensemble,
                             image_size = pe.image_size[])
    pe.preset[] = level
    pe.error[] = ""
    pe.passes[] = passes
    return pe
end

"""
    set_analysis_size!(pe::PassesEditor, size)

Record the analysis size (ROI or frame). A schedule still linked to a preset
is refilled for the new size.
"""
function set_analysis_size!(pe::PassesEditor, sz::Union{Nothing,Dims{2}})
    sz == pe.image_size[] && return pe
    pe.image_size[] = sz
    pe.preset[] === nothing || fill_preset!(pe, pe.preset[])
    return pe
end

"""
    set_mode!(pe::PassesEditor, mode)

Choose the analysis mode: `:sequence` (one field per pair), `:ensemble`
(one field from all pairs), `:ptv` (particle matches per pair), or
`:tracking` (particle tracks through the frames). A schedule linked to a
preset is refilled, since the ensemble presets differ.
"""
function set_mode!(pe::PassesEditor, mode::Symbol)
    _check_mode(mode)
    mode == pe.mode[] && return pe
    pe.mode[] = mode
    pe.preset[] === nothing || fill_preset!(pe, pe.preset[])
    return pe
end

"""
    set_image_type!(pe::PassesEditor, T)

Process in `Float32` or `Float64`.
"""
function set_image_type!(pe::PassesEditor, T::DataType)
    T in (Float32, Float64) || throw(ArgumentError("image_type must be Float32 or Float64, got $T"))
    T == pe.image_type[] || (pe.image_type[] = T)
    return pe
end

"""
    load_passes!(pe::PassesEditor, passes)

Use an explicit schedule (e.g. from a recipe); unlinks the preset.
"""
function load_passes!(pe::PassesEditor, passes::AbstractVector{PIVParameters})
    isempty(passes) && throw(ArgumentError("a schedule needs at least one pass"))
    pe.preset[] = nothing
    pe.error[] = ""
    pe.passes[] = collect(PIVParameters, passes)
    return pe
end

_check_mode(mode::Symbol) = mode in ANALYSIS_MODES ||
    throw(ArgumentError("mode must be one of $(join((":$m" for m in ANALYSIS_MODES), ", ")), got :$mode"))

# PIVParameters field names are its keyword names: copy with overrides.
function _with(p::PIVParameters; kwargs...)
    PIVParameters(; (f => getfield(p, f) for f in fieldnames(PIVParameters))..., kwargs...)
end

function _try_edit!(pe::PassesEditor, f)
    new = try
        f(copy(pe.passes[]))
    catch err
        err isa ArgumentError || rethrow()
        pe.error[] = _errmsg(err)
        return false
    end
    pe.preset[] = nothing
    pe.error[] = ""
    pe.passes[] = new
    return true
end

"""
    set_pass!(pe::PassesEditor, i, field, value) -> Bool

Edit pass `i`: `field` is `:window` (square window, px), `:search` (search
area, px), `:overlap` (percent of the window), or `:iterations`. Changing the
window keeps the overlap fraction and the search-area margin. Return `false`
and set `error` when the edit is invalid; the schedule is then unchanged.
Any accepted edit unlinks the preset.
"""
function set_pass!(pe::PassesEditor, i::Integer, field::Symbol, value)
    field in PASS_FIELDS ||
        throw(ArgumentError("field must be one of $(join(PASS_FIELDS, ", ")), got :$field"))
    1 <= i <= length(pe.passes[]) || throw(BoundsError(pe.passes[], i))
    return _try_edit!(pe, function (ps)
        p = ps[i]
        if field === :window
            w = Int(value)
            frac = p.overlap[1] / p.window_size[1]
            margin = p.search_area_size[1] - p.window_size[1]
            ps[i] = _with(p; window_size = w, search_area_size = w + margin,
                          overlap = round(Int, frac * w))
        elseif field === :search
            ps[i] = _with(p; search_area_size = Int(value))
        elseif field === :overlap
            pct = Float64(value)
            0 <= pct < 100 || throw(ArgumentError("overlap must be in [0, 100) %"))
            ps[i] = _with(p; overlap = round(Int, pct / 100 * p.window_size[1]))
        else
            ps[i] = _with(p; max_iterations = Int(value))
        end
        ps
    end)
end

"""
    set_option!(pe::PassesEditor, option, value) -> Bool

Set an option on every pass: `:correlation` (`:cross`/`:phase`),
`:subpixel` (`:gauss3`/`:gauss9`/`:gauss2d`), `:padding` (zero padding,
`Bool`), `:apodization` (`true`/`:gauss` for Gaussian weighting,
`false`/`:none`), `:accuracy` (both of these together), the normalized
median test `:uod_enable` (`Bool`), `:uod_threshold`, and
`:uod_neighborhood` (half-width: 1 = 3×3, 2 = 5×5, 3 = 7×7),
`:min_peak_ratio`, or `:replace_outliers`. `:uncertainty` applies to the
final pass only. Invalid values return `false` and set `error`.
"""
function set_option!(pe::PassesEditor, option::Symbol, value)
    option === :uncertainty && return _try_edit!(pe, function (ps)
        ps[end] = _with(ps[end]; uncertainty = Bool(value))
        ps
    end)
    option in SHARED_OPTIONS ||
        throw(ArgumentError("option must be :uncertainty or one of $(join(SHARED_OPTIONS, ", ")), got :$option"))
    kw = try
        option === :correlation ? (; correlation_method = Symbol(value)) :
        option === :subpixel ? (; subpixel_method = Symbol(value)) :
        option === :accuracy ? (; padding = Bool(value), apodization = Bool(value) ? :gauss : :none) :
        option === :padding ? (; padding = Bool(value)) :
        option === :apodization ? (; apodization = value isa Bool ? (value ? :gauss : :none) : Symbol(value)) :
        option === :uod_enable ? (; uod_enable = Bool(value)) :
        option === :uod_threshold ? (; uod_threshold = _option_number(value, "the outlier threshold")) :
        option === :uod_neighborhood ? (; uod_neighborhood = Int(_option_number(value, "the neighborhood"))) :
        option === :min_peak_ratio ? (; min_peak_ratio = _option_number(value, "the minimum peak ratio")) :
        (; replace_outliers = Bool(value))
    catch err
        err isa Union{ArgumentError,InexactError} || rethrow()
        pe.error[] = _errmsg(err)
        return false
    end
    return _try_edit!(pe, ps -> [_with(p; kw...) for p in ps])
end

function _option_number(value, what::AbstractString)
    v = value isa Real ? Float64(value) :
        value isa Union{AbstractString,Symbol} ? tryparse(Float64, strip(String(value))) : nothing
    (v === nothing || !isfinite(v)) && throw(ArgumentError("$what must be a number, got \"$value\""))
    return v
end

"""
    add_pass!(pe::PassesEditor)

Append a copy of the final pass (e.g. to repeat the final window size).
"""
add_pass!(pe::PassesEditor) = _try_edit!(pe, ps -> push!(ps, ps[end]))

"""
    remove_pass!(pe::PassesEditor, i)

Remove pass `i`; the schedule keeps at least one pass.
"""
function remove_pass!(pe::PassesEditor, i::Integer)
    length(pe.passes[]) > 1 || (pe.error[] = "a schedule needs at least one pass"; return false)
    1 <= i <= length(pe.passes[]) || throw(BoundsError(pe.passes[], i))
    return _try_edit!(pe, ps -> deleteat!(ps, i))
end

"""
    pass_rows(pe::PassesEditor) -> Vector{NamedTuple}

Table rows `(; window, search, overlap, iterations, uncertainty)` with the
overlap in percent of the window.
"""
pass_rows(pe::PassesEditor) =
    [(; window = p.window_size[1], search = p.search_area_size[1],
        overlap = round(100 * p.overlap[1] / p.window_size[1]; digits = 1),
        iterations = p.max_iterations, uncertainty = p.uncertainty) for p in pe.passes[]]

"""
    option_value(pe::PassesEditor, option)

Current value of a shared option, read from the final pass.
"""
function option_value(pe::PassesEditor, option::Symbol)
    p = pe.passes[][end]
    option === :correlation && return p.correlation_method
    option === :subpixel && return p.subpixel_method
    option === :accuracy && return p.padding && p.apodization === :gauss
    option === :padding && return p.padding
    option === :apodization && return p.apodization === :gauss
    option === :uod_enable && return p.uod_enable
    option === :uod_neighborhood && return p.uod_neighborhood
    option === :uod_threshold && return p.uod_threshold
    option === :min_peak_ratio && return p.min_peak_ratio
    option === :replace_outliers && return p.replace_outliers
    option === :uncertainty && return p.uncertainty
    throw(ArgumentError("unknown option :$option"))
end

"""
    passes_summary(pe::PassesEditor) -> String

One line, e.g. `"128→64→32 px ×2 · medium preset"`. An ensemble runs each
pass once (the core ensemble driver ignores repeats), so its summary shows
no repeat counts.
"""
function passes_summary(pe::PassesEditor)
    ps = pe.passes[]
    parts = String[]
    for p in ps
        s = string(p.window_size[1])
        p.max_iterations > 1 && pe.mode[] !== :ensemble && (s *= " ×$(p.max_iterations)")
        push!(parts, s)
    end
    txt = join(parts, "→") * " px"
    pe.mode[] === :ensemble && (txt *= " · ensemble")
    _particle_mode(pe.mode[]) && (txt = "PIV predictor " * txt)
    txt *= pe.preset[] === nothing ? " · custom" : " · $(pe.preset[]) preset"
    return txt
end
