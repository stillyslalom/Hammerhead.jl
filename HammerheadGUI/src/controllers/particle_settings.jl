# Particle (PTV and tracking) settings of the planar workflow: the core
# PTVParameters, the predictor choice, the tracking options, and a detection
# preview on the representative frame. The pass schedule (PassesEditor)
# serves as the PIV predictor in these modes.

"""
Analysis modes of the planar workflow, as `PIVRecipe` modes: PIV per pair,
PIV ensemble, PTV per pair, and particle tracking through the frames.
"""
const ANALYSIS_MODES = (:sequence, :ensemble, :ptv, :tracking)

"""
Options of a `ParticleSettings`, as accepted by
[`set_particle_option!`](@ref).
"""
const PARTICLE_OPTIONS = (:threshold, :threshold_k, :min_separation, :min_diameter, :max_diameter,
                          :search_radius, :intensity_weight, :diameter_weight, :uod_enable,
                          :uod_threshold, :uod_neighbors, :uod_epsilon, :predictor,
                          :min_track_length, :max_gap)

_particle_mode(mode::Symbol) = mode === :ptv || mode === :tracking

"""
    ParticleSettings()

The particle-analysis settings of a [`PlanarWorkflow`](@ref)
(`wf.particles`), used when the analysis mode is `:ptv` or `:tracking`:
`ptv` (core `PTVParameters`: detection, matching, validation), `predictor`
(`:piv` uses the pass schedule as the PIV predictor, `:none` searches around
each particle's own position), `min_track_length` and `max_gap` (tracking).
Edit them with [`set_particle_option!`](@ref) or
[`edit_particle_option!`](@ref) (`error` holds the last rejected edit).

`detected` holds the particles found on the shown frame of the
representative pair (after preprocessing, inside the mask) with the current
detection settings, and `detect_status` a one-line count; the workflow
refreshes them on the Passes step.
"""
struct ParticleSettings
    ptv::Observable{PTVParameters}
    predictor::Observable{Symbol}
    min_track_length::Observable{Int}
    max_gap::Observable{Int}
    error::Observable{String}
    detected::Observable{Union{Nothing,Particles}}
    detect_status::Observable{String}
    generation::Base.RefValue{Int}
    key::Base.RefValue{Any}
end

ParticleSettings() = ParticleSettings(Observable(PTVParameters()), Observable(:piv), Observable(3),
                                      Observable(0), Observable(""),
                                      Observable{Union{Nothing,Particles}}(nothing), Observable(""),
                                      Ref(0), Ref{Any}(nothing))

Base.show(io::IO, ps::ParticleSettings) =
    print(io, "ParticleSettings(search radius $(ps.ptv[].search_radius) px, predictor :$(ps.predictor[]))")

# PTVParameters field names are its keyword names: copy with overrides.
_with(p::PTVParameters; kwargs...) =
    PTVParameters(; (f => getfield(p, f) for f in fieldnames(PTVParameters))..., kwargs...)

function _parse_int(value, what::AbstractString)
    v = value isa Integer ? Int(value) :
        value isa Real && isinteger(value) ? Int(value) :
        value isa AbstractString ? tryparse(Int, strip(value)) : nothing
    v === nothing && throw(ArgumentError("$what must be a whole number, got \"$value\""))
    return v
end

"""
    set_particle_option!(ps::ParticleSettings, option, value)

Set one of [`PARTICLE_OPTIONS`](@ref) from a value or its text form:
`:threshold` (`:auto`/`"auto"`, or an intensity), the `PTVParameters`
numbers (`:threshold_k`, `:min_separation`, `:min_diameter`,
`:max_diameter`, `:search_radius`, `:intensity_weight`, `:diameter_weight`,
`:uod_threshold`, `:uod_epsilon`; `:uod_neighbors` a whole number),
`:uod_enable` (`Bool`), `:predictor` (`:piv`/`:none`), `:min_track_length`
(at least 2), `:max_gap` (at least 0). Invalid values throw and change
nothing.
"""
function set_particle_option!(ps::ParticleSettings, option::Symbol, value)
    option in PARTICLE_OPTIONS || throw(ArgumentError("unknown particle option :$option"))
    if option === :predictor
        _setobs!(ps.predictor, _parse_choice(value, (:piv, :none), "the predictor"))
    elseif option === :min_track_length
        n = _parse_int(value, "the minimum track length")
        n >= 2 || throw(ArgumentError("a track needs at least 2 frames, got $n"))
        _setobs!(ps.min_track_length, n)
    elseif option === :max_gap
        n = _parse_int(value, "the gap")
        n >= 0 || throw(ArgumentError("the gap must not be negative, got $n"))
        _setobs!(ps.max_gap, n)
    else
        v = if option === :threshold
            (value === :auto || (value isa AbstractString && lowercase(strip(value)) in ("auto", ":auto", ""))) ?
                :auto : _parse_real(value, "the threshold")
        elseif option === :uod_enable
            _parse_bool(value, "the outlier test")
        elseif option === :uod_neighbors
            _parse_int(value, "the neighbour count")
        else
            _parse_real(value, replace(String(option), '_' => ' '))
        end
        new = _with(ps.ptv[]; option => v)
        new == ps.ptv[] || (ps.ptv[] = new)
    end
    return ps
end

"""
    edit_particle_option!(ps::ParticleSettings, option, value) -> Bool

[`set_particle_option!`](@ref), reporting a rejected value in `ps.error`
(returns `false`) instead of throwing.
"""
function edit_particle_option!(ps::ParticleSettings, option::Symbol, value)
    try
        set_particle_option!(ps, option, value)
    catch err
        ps.error[] = _errmsg(err)
        return false
    end
    isempty(ps.error[]) || (ps.error[] = "")
    return true
end

"""
    particle_option(ps::ParticleSettings, option) -> value

The current value of one of [`PARTICLE_OPTIONS`](@ref).
"""
function particle_option(ps::ParticleSettings, option::Symbol)
    option === :predictor && return ps.predictor[]
    option === :min_track_length && return ps.min_track_length[]
    option === :max_gap && return ps.max_gap[]
    option in PARTICLE_OPTIONS || throw(ArgumentError("unknown particle option :$option"))
    return getfield(ps.ptv[], option)
end

"""
    load_particles!(ps::ParticleSettings, recipe::PIVRecipe)

Take the particle settings of a recipe.
"""
function load_particles!(ps::ParticleSettings, r::PIVRecipe)
    ps.ptv[] == r.ptv || (ps.ptv[] = r.ptv)
    _setobs!(ps.predictor, r.ptv_predictor)
    _setobs!(ps.min_track_length, r.min_track_length)
    _setobs!(ps.max_gap, r.max_gap)
    isempty(ps.error[]) || (ps.error[] = "")
    return ps
end

"""
    particles_summary(ps::ParticleSettings, mode) -> String

One line for the step rail, e.g. `"PTV · search 3 px · PIV predictor"`.
"""
function particles_summary(ps::ParticleSettings, mode::Symbol)
    p = ps.ptv[]
    txt = (mode === :tracking ? "tracking" : "PTV") * " · search $(display_number(p.search_radius)) px"
    txt *= ps.predictor[] === :piv ? " · PIV predictor" : " · no predictor"
    mode === :tracking &&
        (txt *= " · tracks ≥ $(ps.min_track_length[])" * (ps.max_gap[] > 0 ? ", gaps ≤ $(ps.max_gap[])" : ""))
    return txt
end
