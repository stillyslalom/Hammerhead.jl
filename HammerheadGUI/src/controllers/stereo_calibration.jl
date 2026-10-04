# The stereo workflow's Calibration step: plate images per camera, grid
# detection and camera fits (one CalibrationReview per camera), the shared
# dewarp grid and the two ImageDewarpers, and the self-calibration result.
# Fits and grid builds are pure functions of inputs captured when they are
# requested, so a window runs them on worker tasks (`runner`); generation
# counters drop results that a newer request superseded. Self-calibration
# needs the particle frames, so it is started from the workflow
# (stereo_workflow.jl) and only stores its result here.

const CALIBRATION_MODELS = (:soloff, :pinhole)

"""
Options of a `StereoCalibration`, as accepted by
[`set_calibration_option!`](@ref).
"""
const CALIBRATION_OPTIONS = (:spacing, :two_level, :level_separation, :origin_offset, :invert,
                             :orientation, :model, :length_unit, :coverage, :grid_spacing,
                             :grid_z, :margin, :selfcal_pairs, :keep_disparity_maps)

"""
    StereoCalibration(; runner = _inline_runner)

The Calibration step of a [`StereoWorkflow`](@ref) (`wf.calibration`).

Inputs (all `Observable`s):
- `plates[k]` (camera `k` = 1, 2): calibration plate images, each a
  `(; image, z)` named tuple (a path or matrix, and the plate's world z);
  edit with [`add_plate!`](@ref), [`remove_plate!`](@ref),
  [`set_plate_z!`](@ref), [`clear_plates!`](@ref).
- detection (`detect_calibration_grid`): `spacing` (`nothing` until
  entered), `two_level`, `level_separation`, `origin_offset` (`nothing` or
  `(x, y)`), `invert`, `orientation` (`:image`/`:fiducials`); `model`
  (`:soloff`/`:pinhole`); `length_unit`, the world length unit those values
  are entered in (the default length unit of the stereo scale).
- dewarp grid (`common_dewarp_grid`): `coverage` (`:intersection`/`:union`),
  `grid_spacing` (`:auto` or a length), `grid_z`, `margin`.
- self-calibration: `selfcal_pairs` (how many leading pairs' frames it
  uses) and `keep_disparity_maps`.

Edit options with [`set_calibration_option!`](@ref) or
[`edit_calibration_option!`](@ref) (`error` holds the last rejected edit).

Results: [`fit_calibration!`](@ref) detects and fits both cameras into
`reviews[k]` (`CalibrationReview`s; `fitting`, `fit_status`), then builds
the grid; [`build_dewarpers!`](@ref) builds `dewarpers` (`building`,
`grid_status`). A changed `model` refits at once; a changed grid option
rebuilds. [`set_dewarpers!`](@ref) sets dewarpers made elsewhere.
Self-calibration stores `selfcal` (`(; report, dewarpers, source)`,
`selfcal_running`, `selfcal_status`); `selfcal_applied` says whether the
current dewarpers are its correction.
"""
struct StereoCalibration
    plates::NTuple{2,Observable{Vector{Any}}}
    spacing::Observable{Union{Nothing,Float64}}
    two_level::Observable{Bool}
    level_separation::Observable{Float64}
    origin_offset::Observable{Union{Nothing,NTuple{2,Float64}}}
    invert::Observable{Bool}
    orientation::Observable{Symbol}
    model::Observable{Symbol}
    length_unit::Observable{String}
    reviews::NTuple{2,Observable{Union{Nothing,CalibrationReview}}}
    fitting::Observable{Bool}
    fit_status::Observable{String}
    coverage::Observable{Symbol}
    grid_spacing::Observable{Union{Symbol,Float64}}
    grid_z::Observable{Float64}
    margin::Observable{Float64}
    building::Observable{Bool}
    grid_status::Observable{String}
    dewarpers::Observable{Union{Nothing,Tuple{ImageDewarper,ImageDewarper}}}
    selfcal_pairs::Observable{Int}
    keep_disparity_maps::Observable{Bool}
    selfcal_running::Observable{Bool}
    selfcal_status::Observable{String}
    selfcal::Observable{Union{Nothing,NamedTuple}}
    selfcal_applied::Observable{Bool}
    error::Observable{String}
    fitted::Base.RefValue{Any}             # detection inputs of the current reviews
    runner::Base.RefValue{Any}
    fit_generation::Base.RefValue{Int}
    grid_generation::Base.RefValue{Int}
    selfcal_generation::Base.RefValue{Int}
    revision::Base.RefValue{Int}           # counts dewarper changes
end

function StereoCalibration(; runner = _inline_runner)
    cal = StereoCalibration((Observable{Vector{Any}}(Any[]), Observable{Vector{Any}}(Any[])),
                            Observable{Union{Nothing,Float64}}(nothing), Observable(false),
                            Observable(0.0), Observable{Union{Nothing,NTuple{2,Float64}}}(nothing),
                            Observable(false), Observable(:image), Observable(:soloff),
                            Observable("mm"),
                            (Observable{Union{Nothing,CalibrationReview}}(nothing),
                             Observable{Union{Nothing,CalibrationReview}}(nothing)),
                            Observable(false), Observable(""),
                            Observable(:intersection), Observable{Union{Symbol,Float64}}(:auto),
                            Observable(0.0), Observable(0.0), Observable(false), Observable(""),
                            Observable{Union{Nothing,Tuple{ImageDewarper,ImageDewarper}}}(nothing),
                            Observable(5), Observable(false), Observable(false), Observable(""),
                            Observable{Union{Nothing,NamedTuple}}(nothing), Observable(false),
                            Observable(""), Ref{Any}(nothing), Ref{Any}(runner),
                            Ref(0), Ref(0), Ref(0), Ref(0))
    return cal
end

function Base.show(io::IO, cal::StereoCalibration)
    n1, n2 = length(cal.plates[1][]), length(cal.plates[2][])
    print(io, "StereoCalibration($n1 + $n2 plates, :$(cal.model[]), ",
          cal.dewarpers[] === nothing ? "no dewarpers)" : "dewarpers set)")
end

_check_camera(k::Integer) = k in (1, 2) || throw(ArgumentError("camera must be 1 or 2, got $k"))

function _parse_real(value, what::AbstractString)
    v = value isa AbstractString ? tryparse(Float64, strip(value)) :
        value isa Real ? Float64(value) : nothing
    (v === nothing || !isfinite(v)) && throw(ArgumentError("$what must be a number, got \"$value\""))
    return v
end

function _parse_bool(value, what::AbstractString)
    value isa Bool && return value
    if value isa AbstractString
        s = lowercase(strip(value))
        s in ("true", "yes", "1") && return true
        s in ("false", "no", "0") && return false
    end
    throw(ArgumentError("$what must be true or false, got \"$value\""))
end

function _parse_choice(value, choices, what::AbstractString)
    s = value isa Symbol ? value : Symbol(strip(lstrip(strip(String(value)), ':')))
    s in choices || throw(ArgumentError("$what must be one of $(join((":$c" for c in choices), ", ")), got :$s"))
    return s
end

# ---------------------------------------------------------------- plates

"""
    add_plate!(cal::StereoCalibration, camera, image, z)

Add a calibration plate image (path or matrix) for `camera` (1 or 2) at
world position `z` (a number or its text form).
"""
function add_plate!(cal::StereoCalibration, camera::Integer, image, z)
    _check_camera(camera)
    image isa Union{AbstractString,AbstractMatrix{<:Real}} ||
        throw(ArgumentError("a plate image is a path or a matrix"))
    obs = cal.plates[camera]
    push!(obs[], (; image, z = _parse_real(z, "z")))
    notify(obs)
    return cal
end

"""
    remove_plate!(cal::StereoCalibration, camera, i)

Remove plate `i` of `camera`.
"""
function remove_plate!(cal::StereoCalibration, camera::Integer, i::Integer)
    _check_camera(camera)
    obs = cal.plates[camera]
    1 <= i <= length(obs[]) || throw(ArgumentError("camera $camera has no plate $i"))
    deleteat!(obs[], i)
    notify(obs)
    return cal
end

"""
    set_plate_z!(cal::StereoCalibration, camera, i, z)

Set the world z of plate `i` of `camera` (a number or its text form).
"""
function set_plate_z!(cal::StereoCalibration, camera::Integer, i::Integer, z)
    _check_camera(camera)
    obs = cal.plates[camera]
    1 <= i <= length(obs[]) || throw(ArgumentError("camera $camera has no plate $i"))
    v = _parse_real(z, "z")
    obs[][i].z == v && return cal
    obs[][i] = (; image = obs[][i].image, z = v)
    notify(obs)
    return cal
end

"""
    clear_plates!(cal::StereoCalibration, camera = nothing)

Remove the plates of `camera`, or of both cameras.
"""
function clear_plates!(cal::StereoCalibration, camera::Union{Nothing,Integer} = nothing)
    for k in (camera === nothing ? (1, 2) : (_check_camera(camera); (Int(camera),)))
        empty!(cal.plates[k][])
        notify(cal.plates[k])
    end
    return cal
end

# ---------------------------------------------------------------- options

"""
    detect_options(cal::StereoCalibration) -> NamedTuple

The `detect_calibration_grid` keywords of the current detection settings
(throws when the dot spacing is missing).
"""
function detect_options(cal::StereoCalibration)
    cal.spacing[] === nothing && throw(ArgumentError("enter the dot spacing"))
    return (; spacing = cal.spacing[], two_level = cal.two_level[],
            level_separation = cal.level_separation[], origin_offset = cal.origin_offset[],
            invert = cal.invert[], orientation = cal.orientation[])
end

_setobs!(obs, v) = (isequal(obs[], v) || (obs[] = v); nothing)

"""
    set_calibration_option!(cal::StereoCalibration, option, value)

Set one of [`CALIBRATION_OPTIONS`](@ref) from a value or its text form:
`:spacing` (> 0), `:level_separation`, `:grid_z`, `:margin` (numbers);
`:two_level`, `:invert`, `:keep_disparity_maps` (`Bool`); `:origin_offset`
(`nothing`/empty, or `(x, y)` / `"x, y"`); `:orientation`
(`:image`/`:fiducials`); `:model` (`:soloff`/`:pinhole` — refits the
reviews at once); `:length_unit`; `:coverage` (`:intersection`/`:union`);
`:grid_spacing` (`:auto`/`"auto"` or a positive length); `:selfcal_pairs`
(a positive integer). Changing a grid option rebuilds the dewarpers when
both cameras are fitted. Detection options take effect at the next
[`fit_calibration!`](@ref) (see [`fit_stale`](@ref)). Invalid values throw
and change nothing.
"""
function set_calibration_option!(cal::StereoCalibration, option::Symbol, value)
    option in CALIBRATION_OPTIONS ||
        throw(ArgumentError("unknown calibration option :$option"))
    if option === :spacing
        _setobs!(cal.spacing, _parse_positive(value isa Real ? string(value) : String(value), "the dot spacing"))
    elseif option in (:two_level, :invert, :keep_disparity_maps)
        _setobs!(getfield(cal, option), _parse_bool(value, String(option)))
    elseif option === :level_separation
        _setobs!(cal.level_separation, _parse_real(value, "the level separation"))
    elseif option === :origin_offset
        _setobs!(cal.origin_offset, _parse_offset(value))
    elseif option === :orientation
        _setobs!(cal.orientation, _parse_choice(value, (:image, :fiducials), "orientation"))
    elseif option === :model
        m = _parse_choice(value, CALIBRATION_MODELS, "model")
        m == cal.model[] && return cal
        cal.model[] = m
        _refit!(cal)
    elseif option === :length_unit
        u = strip(String(value))
        isempty(u) && throw(ArgumentError("the unit must not be empty"))
        _setobs!(cal.length_unit, String(u))
    elseif option === :selfcal_pairs
        n = value isa Integer ? Int(value) : tryparse(Int, strip(String(value)))
        (n === nothing || n < 1) &&
            throw(ArgumentError("self-calibration needs at least one pair, got \"$value\""))
        _setobs!(cal.selfcal_pairs, n)
    else                                   # grid options
        new = if option === :coverage
            _parse_choice(value, (:intersection, :union), "coverage")
        elseif option === :grid_spacing
            (value === :auto || (value isa AbstractString && lowercase(strip(value)) in ("auto", ":auto", ""))) ?
                :auto : _parse_positive(value isa Real ? string(value) : String(value), "the grid spacing")
        else
            _parse_real(value, option === :grid_z ? "z" : "the margin")
        end
        obs = option === :coverage ? cal.coverage : option === :grid_spacing ? cal.grid_spacing :
              option === :grid_z ? cal.grid_z : cal.margin
        isequal(obs[], new) && return cal
        obs[] = new
        _fitted(cal) && build_dewarpers!(cal)
    end
    return cal
end

function _parse_offset(value)
    value === nothing && return nothing
    if value isa AbstractString
        s = strip(value)
        isempty(s) && return nothing
        parts = split(s, r"[,\s]+"; keepempty = false)
        length(parts) == 2 || throw(ArgumentError("the origin offset is two numbers \"x, y\", got \"$value\""))
        return (_parse_real(parts[1], "the origin offset"), _parse_real(parts[2], "the origin offset"))
    end
    length(value) == 2 || throw(ArgumentError("the origin offset is two numbers (x, y)"))
    return (_parse_real(value[1], "the origin offset"), _parse_real(value[2], "the origin offset"))
end

"""
    edit_calibration_option!(cal::StereoCalibration, option, value) -> Bool

[`set_calibration_option!`](@ref), reporting a rejected value in
`cal.error` (returns `false`) instead of throwing.
"""
function edit_calibration_option!(cal::StereoCalibration, option::Symbol, value)
    try
        set_calibration_option!(cal, option, value)
    catch err
        cal.error[] = _errmsg(err)
        return false
    end
    isempty(cal.error[]) || (cal.error[] = "")
    return true
end

# ---------------------------------------------------------------- fit

_fitted(cal::StereoCalibration) =
    all(k -> (cr = cal.reviews[k][]; cr !== nothing && cr.camera[] !== nothing), (1, 2))

# What a fit depends on, compared by identity for in-memory images.
_entry_key(x) = x isa AbstractString ? String(x) : objectid(x)
_fit_key(cal::StereoCalibration) =
    (map(k -> [(_entry_key(p.image), p.z) for p in cal.plates[k][]], (1, 2)),
     (cal.spacing[], cal.two_level[], cal.level_separation[], cal.origin_offset[], cal.invert[],
      cal.orientation[]))

"""
    fit_stale(cal::StereoCalibration) -> Bool

Whether the plates or detection settings changed since the last fit.
"""
fit_stale(cal::StereoCalibration) = cal.fitted[] !== nothing && cal.fitted[] != _fit_key(cal)

"""
    fit_calibration!(cal::StereoCalibration)

Detect the dot grids on every plate image and fit each camera's model, as
a `CalibrationReview` per camera (`reviews`); then, when both cameras are
fitted (and no dewarpers were set meanwhile), build the dewarpers. Runs on
the `runner` (a worker task in a window); `fitting` and `fit_status`
report progress and per-camera failures.
"""
function fit_calibration!(cal::StereoCalibration)
    for k in (1, 2)
        isempty(cal.plates[k][]) &&
            (cal.fit_status[] = "add calibration images for camera $k"; return cal)
    end
    cal.spacing[] === nothing && (cal.fit_status[] = "enter the dot spacing"; return cal)
    detect = detect_options(cal)
    plates = map(k -> copy(cal.plates[k][]), (1, 2))
    model = cal.model[]
    key = _fit_key(cal)
    rev = cal.revision[]
    g = (cal.fit_generation[] += 1)
    cal.fitting[] = true
    cal.fit_status[] = "detecting the calibration grids…"
    job = () -> map(plates) do ps
        _try_job(() -> CalibrationReview(Any[p.image for p in ps], [p.z for p in ps];
                                         model, detect...))
    end
    cal.runner[](job, out -> _finish_fit!(cal, g, key, rev, out))
    return cal
end

function _finish_fit!(cal::StereoCalibration, g::Int, key, rev::Int, out)
    g == cal.fit_generation[] || return cal
    cal.fitting[] = false
    if out.err !== nothing
        cal.fit_status[] = "fit failed: " * _errmsg(out.err)
        return cal
    end
    msgs = String[]
    for k in (1, 2)
        o = out.value[k]
        if o.err === nothing
            cal.reviews[k][] = o.value
            o.value.camera[] === nothing && push!(msgs, "camera $k: no fit: $(o.value.fit_message[])")
        else
            cal.reviews[k][] = nothing
            push!(msgs, "camera $k: " * _errmsg(o.err))
        end
    end
    cal.fitted[] = key
    cal.fit_status[] = isempty(msgs) ? _fit_summary(cal) : join(msgs, "\n")
    isempty(msgs) && rev == cal.revision[] && build_dewarpers!(cal)
    return cal
end

function _fit_summary(cal::StereoCalibration)
    parts = String[]
    for k in (1, 2)
        cr = cal.reviews[k][]
        cr === nothing && continue
        cam = cr.camera[]
        if cam === nothing
            push!(parts, "camera $k: no fit")
        else
            q = calibration_quality(cam, cr.grids, cr.zs)
            push!(parts, "camera $k: $(nplanes(cr)) planes, rms $(_fmt(q.rms)) px")
        end
    end
    return "$(cal.model[]) fit · " * join(parts, " · ")
end

# A new model refits the existing reviews (no new detection).
function _refit!(cal::StereoCalibration)
    any(k -> cal.reviews[k][] !== nothing, (1, 2)) || return cal
    for k in (1, 2)
        cr = cal.reviews[k][]
        cr === nothing || cr.model[] == cal.model[] || (cr.model[] = cal.model[])
    end
    cal.fit_status[] = _fit_summary(cal)
    _fitted(cal) && build_dewarpers!(cal)
    return cal
end

# ---------------------------------------------------------------- dewarpers

"""
    build_dewarpers!(cal::StereoCalibration)

Build the shared dewarp grid (`common_dewarp_grid` with `coverage`,
`grid_spacing`, `grid_z`, and `margin`; each camera's first plate image
gives its frame size) and the two `ImageDewarper`s from the fitted reviews.
Runs on the `runner`; `building` and `grid_status` report it. The result
replaces `dewarpers`.
"""
function build_dewarpers!(cal::StereoCalibration)
    _fitted(cal) || (cal.grid_status[] = "fit both cameras first"; return cal)
    cr1, cr2 = cal.reviews[1][], cal.reviews[2][]
    cams = (cr1.camera[], cr2.camera[])
    sizes = (size(cr1.images[1]), size(cr2.images[1]))
    z, spacing, coverage, margin = cal.grid_z[], cal.grid_spacing[], cal.coverage[], cal.margin[]
    g = (cal.grid_generation[] += 1)
    cal.building[] = true
    cal.grid_status[] = "building the dewarp grid…"
    job = () -> _dewarper_pair(cams, sizes, z; spacing, coverage, margin)
    apply = function (out)
        g == cal.grid_generation[] || return
        cal.building[] = false
        if out.err === nothing
            _set_dewarpers!(cal, out.value, false)
        else
            cal.grid_status[] = "dewarp grid failed: " * _errmsg(out.err)
        end
    end
    cal.runner[](job, apply)
    return cal
end

"""
    set_dewarpers!(cal::StereoCalibration, dw1, dw2)

Use dewarpers built elsewhere (e.g. in a script); they must share one
`DewarpGrid`. A grid build in flight is dropped.
"""
function set_dewarpers!(cal::StereoCalibration, dw1::ImageDewarper, dw2::ImageDewarper)
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid"))
    cal.grid_generation[] += 1
    cal.building[] && (cal.building[] = false)
    _set_dewarpers!(cal, (dw1, dw2), false)
    return cal
end

function _set_dewarpers!(cal::StereoCalibration, dws::Tuple, selfcal::Bool)
    cal.revision[] += 1
    cal.selfcal_applied[] == selfcal || (cal.selfcal_applied[] = selfcal)
    cal.dewarpers[] = dws
    cal.grid_status[] = grid_summary(cal)
    return cal
end

"""
    grid_summary(cal::StereoCalibration) -> String

One line describing the dewarp grid (size, spacing, z), or what is missing.
"""
function grid_summary(cal::StereoCalibration)
    dws = cal.dewarpers[]
    dws === nothing && return "no dewarp grid"
    grid = dws[1].grid
    ny, nx = size(grid)
    u = cal.length_unit[]
    return "dewarp grid $nx×$ny nodes · spacing $(_fmt(abs(step(grid.x)))) $u · z = $(_fmt(grid.z)) $u"
end

"""
    calibration_summary(cal::StereoCalibration) -> String

One line for the step rail: the dewarp grid, whether it is self-calibrated,
and whether the fit is out of date.
"""
function calibration_summary(cal::StereoCalibration)
    cal.dewarpers[] === nothing && return isempty(cal.fit_status[]) ? "not calibrated" : cal.fit_status[]
    s = grid_summary(cal)
    cal.selfcal_applied[] && (s *= " · self-calibrated")
    return s
end

"""
    apply_selfcal!(cal::StereoCalibration) -> Bool

Replace the dewarpers with the last self-calibration's corrected pair.
Refused (returns `false`, `selfcal_status` says why) when there is no
result or the dewarpers changed since it was computed.
"""
function apply_selfcal!(cal::StereoCalibration)
    s = cal.selfcal[]
    s === nothing && (cal.selfcal_status[] = "run self-calibration first"; return false)
    cur = cal.dewarpers[]
    if cur !== nothing && cur[1] === s.dewarpers[1] && cur[2] === s.dewarpers[2] && cal.selfcal_applied[]
        cal.selfcal_status[] = "the self-calibration is already applied"
        return false
    end
    (cur === nothing || cur[1] !== s.source[1] || cur[2] !== s.source[2]) &&
        (cal.selfcal_status[] = "the dewarpers changed since the self-calibration; run it again"; return false)
    _set_dewarpers!(cal, s.dewarpers, true)
    cal.selfcal_status[] = "applied the self-calibration"
    return true
end
