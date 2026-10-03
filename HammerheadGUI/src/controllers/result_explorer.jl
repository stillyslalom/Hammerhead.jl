# Result-explorer controller: all state and logic for browsing a loaded
# result sequence. Framework-free — views render from the Observables and
# mutate them through the API below, so everything here runs without a GL
# context.

# The explorer browses all four persisted result types. Gridded results
# (PIV/stereo) select a window by CartesianIndex; scattered results (PTV
# particles, tracking trajectories) select by linear index.
const GridResult = Union{PIVResult,StereoPIVResult}
const ScatteredResult = Union{PTVResult,TrackingResult}
const AnyResult = Union{GridResult,ScatteredResult}
const DisplayResult = Union{AnyResult,TimedTrackingResult}
const Selection = Union{Nothing,CartesianIndex{2},Int}

# Dedicated artifacts own one complete trajectory bundle. Keep native vector
# eltypes unchanged: widening AnyResult would break native save_results dispatch.
struct _TimedDisplayResult <: AbstractVector{TimedTrackingResult}
    result::TimedTrackingResult
end
Base.size(::_TimedDisplayResult) = (1,)
Base.IndexStyle(::Type{_TimedDisplayResult}) = IndexLinear()
function Base.getindex(results::_TimedDisplayResult,i::Int)
    checkbounds(results,i)
    Hammerhead._tracking_check(results.result)
    results.result
end
Base.show(io::IO,::_TimedDisplayResult) = print(io,"Timed tracking display (one complete trajectory bundle)")
Base.show(io::IO,::MIME"text/plain",results::_TimedDisplayResult) = show(io,results)

# One physical/display payload, irrespective of recording length. Read/convert
# before replacing the cache, so failed navigation preserves the prior frame.
mutable struct _LazyDisplayResults <: AbstractVector{AnyResult}
    source::Union{Hammerhead.ResultFile,Hammerhead.CheckpointResults}
    index::Int
    result::Union{Nothing,AnyResult}
    inspection::Bool
    companions::NamedTuple
end
_empty_companions(state=:off)=(state=state,history=nothing,diagnostics=nothing,stereo_diagnostics=nothing,display_sha256=nothing)
_LazyDisplayResults(source,index,result)=_LazyDisplayResults(source,index,result,false,_empty_companions())
Base.size(results::_LazyDisplayResults) = size(results.source)
Base.IndexStyle(::Type{_LazyDisplayResults}) = IndexLinear()
Hammerhead._result_protected_paths(results::_LazyDisplayResults)=Hammerhead._result_protected_paths(results.source)
function Base.getindex(results::_LazyDisplayResults, i::Int)
    checkbounds(results, i)
    if results.index != i
        result,companions=_prepare_lazy_frame(results,i,results.inspection)
        _commit_lazy_frame!(results,i,result,companions,results.inspection)
    end
    return results.result::AnyResult
end
function _prepare_lazy_frame(results,i,inspection)
    raw=results.source[i]
    if inspection
        results.source isa Hammerhead.ResultFile || throw(ArgumentError("checkpoint companion inspection is not supported"))
        history=Hammerhead.load_measurement_history(results.source,i)
        diagnostics=Hammerhead.load_execution_diagnostics(results.source,i)
        stereo=Hammerhead.load_stereo_execution_diagnostics(results.source,i)
        if raw isa PIVResult
            stereo===nothing || throw(ArgumentError("stereo companion attached to a planar result"))
            history===nothing || Hammerhead.verify_measurement_history(history,raw)
            state=history===nothing ? :history_missing : :verified
        elseif raw isa StereoPIVResult
            history===nothing && diagnostics===nothing || throw(ArgumentError("planar companion attached to a stereo result"))
            stereo===nothing || Hammerhead.execution_diagnostics_data(stereo;result=raw)
            state=stereo===nothing ? :stereo_missing : :stereo_verified
        else
            history===nothing && diagnostics===nothing && stereo===nothing || throw(ArgumentError("recorded companion attached to an unsupported result kind"))
            state=:unsupported
        end
        result=physical(raw)
        digest=result isa GridResult ? _display_measurement_digest(result) : nothing
        companions=(state=state,history=history,diagnostics=diagnostics,stereo_diagnostics=stereo,display_sha256=digest)
    else
        result=physical(raw)
        companions=_empty_companions()
    end
    result,companions
end
_display_measurement_digest(r::PIVResult)=Hammerhead._history_result_digest(r)
_display_measurement_digest(r::StereoPIVResult)=Hammerhead._stereo_execution_measurement_digest(r)
function _commit_lazy_frame!(results,i,result,companions,inspection)
    results.result=result
    results.index=i
    results.companions=companions
    results.inspection=inspection
    nothing
end
Base.show(io::IO, results::_LazyDisplayResults) =
    print(io, "Lazy display results (", length(results), " indexed, one cached frame)")
Base.show(io::IO, ::MIME"text/plain", results::_LazyDisplayResults) = show(io, results)

"""
    ResultExplorer(results; path = nothing)
    ResultExplorer(result)
    ResultExplorer(path::AbstractString; lazy = false, format = :native)
    ResultExplorer(index::ResultFile)
    ResultExplorer(index::CheckpointResults)

Browse a sequence of `PIVResult`, `StereoPIVResult`, `PTVResult`, or
`TrackingResult` entries; result types may be mixed. View state is held in
`Observables`: `frame` (1-based index into
the sequence), `field` (the displayed scalar field, see
[`available_fields`](@ref)), `show_vectors` / `highlight_outliers` (overlay
toggles), `selection` (the inspected item: a `CartesianIndex` for a
gridded window, a linear `Int` index for a scattered particle/trajectory, or
`nothing`), the colorbar-limit state `color_mode` / `color_min` /
`color_max` (see [`color_limits`](@ref) and [`set_color_limits!`](@ref);
manual overrides persist across frame/field switches until cleared), and
`count` (the current sequence length, notified by [`push_result!`](@ref) so
views can grow their frame slider while a batch appends live), and the
interactive-analysis state `tool` / `tool_points` / `profile_data` /
`circulation_result` (see [`set_tool!`](@ref) and [`click!`](@ref);
planar results only — tool state clears on frame switches).

Results with a [`PhysicalScale`](@ref) are converted through
[`physical`](@ref) for display in physical units. Unscaled results retain
their original units.

A `TimedTrackingResult` is preserved as a wrapper in a dedicated singleton
explorer. Open its dedicated artifact explicitly with `format=:timed_tracking`;
`lazy=true`, mixed timed/native sequences and appending are unsupported. The
complete trajectory bundle must fit memory; its result count is one, not the
number of acquisition samples. Timing/result binding is checked before physical
conversion and on access. Only spatial coordinates are converted; nominal
`scale.dt` does not divide actual-time speeds. Colors are arithmetic observation
means of secant magnitudes, not instantaneous or elapsed-time-weighted speeds.
Nonfinite data and singleton tracks have unavailable speed, never zero or an
average over silently omitted observations. Timed refresh/inspection validates
the whole bundle in O(observations + selected frames); no trusted mutable context
or per-track velocity cache is retained.

The string form loads a saved sequence with `Hammerhead.load_results`.
Use `lazy = true` or pass a [`ResultFile`](@ref) to browse a completed file
while retaining one display result and only the current frame's derived
fields/tool data (plus O(number of results) key metadata). Lazy explorers
cannot append live results. Selected unreadable entries raise an error,
set `status`, and leave the previous frame and its selection displayed.
`set_frame!` and direct `frame[]` assignments both validate before updating
other view state. The index is fixed and does not follow a concurrent writer.
Changing frames resets an unavailable `field` to that result's default and
clears an invalid `selection`.

A `CheckpointResults` index browses a verified fixed committed prefix with the
same one-frame display cache. Open a new index/explorer after resume to see
later commits; this is not a live checkpoint reader.

[`set_companion_inspection!`](@ref) opts into recorded final-sweep history and
execution counts for lazy native files. Verification uses raw measurements
before physical conversion. A failed read/verification preserves the old frame,
selection, mode and bundle. Only the current history packet is retained.
[`companion_summary`](@ref) and [`describe_companion_selection`](@ref) provide
readable processing details; missing/bare/checkpoint/unsupported states remain
explicit. Private packet and physical display mutation checks are O(grid nodes)
on each inspection refresh, not continuous mutation monitoring.
"""
struct ResultExplorer
    results::Union{Vector{AnyResult},_LazyDisplayResults,_TimedDisplayResult}
    path::Union{Nothing,String}
    frame::Observable{Int}
    field::Observable{Symbol}
    show_vectors::Observable{Bool}
    highlight_outliers::Observable{Bool}
    selection::Observable{Selection}
    color_mode::Observable{Symbol}
    color_min::Observable{Union{Nothing,Float64}}
    color_max::Observable{Union{Nothing,Float64}}
    count::Observable{Int}
    tool::Observable{Symbol}
    tool_points::Observable{Vector{NTuple{2,Float64}}}
    profile_data::Observable{Union{Nothing,NamedTuple}}
    circulation_result::Observable{Union{Nothing,NamedTuple}}
    derived_cache::Dict{Int,NamedTuple}
    status::Observable{String}
    companion_enabled::Observable{Bool}
end

function ResultExplorer(results::AbstractVector; path::Union{Nothing,AbstractString} = nothing)
    isempty(results) && throw(ArgumentError("no results to explore"))
    all(r -> r isa AnyResult, results) ||
        throw(ArgumentError("results must be PIVResult, StereoPIVResult, PTVResult, or TrackingResult entries"))
    conv = AnyResult[physical(r) for r in results]
    return _result_explorer(conv, path)
end

function ResultExplorer(result::TimedTrackingResult;path::Union{Nothing,AbstractString}=nothing)
    display = physical(result) # verifies the raw binding before conversion
    tracking_speed_summary(display) # complete arithmetic preflight
    _result_explorer(_TimedDisplayResult(display),path)
end

function ResultExplorer(index::Hammerhead.ResultFile;
                        path::Union{Nothing,AbstractString} = index.path)
    isempty(index) && throw(ArgumentError("no results to explore"))
    return _result_explorer(_LazyDisplayResults(index, 0, nothing), path)
end

function ResultExplorer(index::Hammerhead.CheckpointResults;
                        path::Union{Nothing,AbstractString}=nothing)
    isempty(index) && throw(ArgumentError("no committed checkpoint results to explore"))
    _result_explorer(_LazyDisplayResults(index,0,nothing),path)
end

function _result_explorer(conv, path)
    ex = ResultExplorer(conv,
                        path === nothing ? nothing : String(path),
                        Observable(1), Observable(first(available_fields(conv[1]))),
                        Observable(true), Observable(true),
                        Observable{Selection}(nothing),
                        Observable(:robust),
                        Observable{Union{Nothing,Float64}}(nothing),
                        Observable{Union{Nothing,Float64}}(nothing),
                        Observable(length(conv)),
                        Observable(:inspect),
                        Observable(NTuple{2,Float64}[]),
                        Observable{Union{Nothing,NamedTuple}}(nothing),
                        Observable{Union{Nothing,NamedTuple}}(nothing),
                        Dict{Int,NamedTuple}(), Observable(""),Observable(false))
    last_frame = Ref(1)
    # Read failures must occur before downstream view notifications. A direct
    # observable write has already changed its value: restore silently and
    # throw to stop that failed notification from reaching views.
    on(ex.frame; priority = typemax(Int)) do i
        r = try
            current_result(ex)
        catch err
            ex.frame.val = last_frame[]
            ex.status[] = first(split(sprint(showerror, err), '\n'))
            rethrow()
        end
        changed=i!=last_frame[]
        changed && empty!(ex.derived_cache)
        last_frame[] = i
        ex.status[] = ""
        ex.field[] in available_fields(r) || (ex.field[] = first(available_fields(r)))
        ex.selection[] = _valid_selection(r, ex.selection[])
        # tool state describes one frame's flow: clear it on a frame switch,
        # and revert to :inspect when the new result has no derived analysis
        changed && _reset_tool!(ex)
        r isa PIVResult || ex.tool[] === :inspect || (ex.tool[] = :inspect)
    end
    last_mode=Ref(false)
    on(ex.companion_enabled;priority=typemax(Int)) do enabled
        enabled==last_mode[] && return
        try
            if enabled
                ex.results isa _LazyDisplayResults && ex.results.source isa Hammerhead.ResultFile ||
                    throw(ArgumentError("recorded companion inspection requires a lazy native ResultFile; eager/bare inputs and checkpoints are unsupported"))
                result,companions=_prepare_lazy_frame(ex.results,ex.frame[],true)
                _commit_lazy_frame!(ex.results,ex.frame[],result,companions,true)
                _reset_tool!(ex) # enabling reloads the display; old analysis may reflect edited values
            elseif ex.results isa _LazyDisplayResults
                ex.results.inspection=false
                ex.results.companions=_empty_companions()
            end
        catch err
            ex.companion_enabled.val=last_mode[]
            ex.status[]=first(split(sprint(showerror,err),'\n'))
            rethrow()
        end
        last_mode[]=enabled
        empty!(ex.derived_cache)
        ex.frame[]=ex.frame[] # refresh the same display without retaining the raw payload
    end
    return ex
end

"""
    set_companion_inspection!(ex::ResultExplorer, enabled::Bool=true)

Opt into recorded planar history/execution or stereo camera execution inspection for a lazy native
`ResultFile` explorer. Raw results and companions are read and history binding
verified before physical display conversion/cache replacement. Stereo binding
checks reconstructed and retained camera measurement fields against the raw
result. The physical display has a separate mutation digest; it is not the raw
binding. Failure preserves
the old mode, frame, selection and display/companion bundle, and sets `status`.
Disabling releases the current packet. Bare/eager inputs and checkpoint indexes
have no supported native-companion association. This retains one display result
and one current packet, not a second raw result. Caller-retained packets cost
additional memory. Both setter and direct `companion_enabled[]` writes preflight.
"""
function set_companion_inspection!(ex::ResultExplorer,enabled::Bool=true)
    ex.companion_enabled[]=enabled
    ex
end
function _checked_companions(ex)
    result=current_result(ex)
    c=ex.results.companions
    try
        c.display_sha256===nothing || _display_measurement_digest(result)==c.display_sha256 ||
            throw(ArgumentError("physical display fields changed after companion verification; disable and reenable inspection to reload"))
        c.history===nothing || Hammerhead._history_checked_data(c.history)
        c.stereo_diagnostics===nothing || Hammerhead.execution_diagnostics_data(c.stereo_diagnostics)
    catch err
        ex.status[]=first(split(sprint(showerror,err),'\n'))
        rethrow()
    end
    c
end

"""
    companion_summary(ex::ResultExplorer) -> String

Describe off/missing/unsupported/verified recorded-companion states and actual
pass execution observations. History binds raw measurement content; execution
planar diagnostics v1 bind an entry key, not numerical content or the independent
history UUID. Stereo companions verify raw reconstructed/camera measurement
fields on loading, without verifying calibration or source images. Their
residuals remain dewarped pixels, not world 3C residuals. Stored uncertainty
availability never certifies applicability.
When enabled, integrity/display checks scan/hash the current arrays (O(nodes));
no result reload or full packet copy occurs on inspection.
"""
function companion_summary(ex::ResultExplorer)
    if !(ex.results isa _LazyDisplayResults)
        return "Recorded companions: no indexed native association (eager/bare inputs)."
    elseif !(ex.results.source isa Hammerhead.ResultFile)
        return "Recorded companions: checkpoint inspection is unsupported."
    elseif !ex.companion_enabled[]
        return "Recorded companion inspection: off."
    end
    c=_checked_companions(ex)
    _companion_summary(c)
end
function _companion_summary(c)
    c.state===:unsupported && return "Recorded companions: unsupported for this result kind."
    if c.state in (:stereo_missing,:stereo_verified)
        c.stereo_diagnostics===nothing && return "Stereo execution diagnostics: not recorded.\nStereo per-node measurement history: unavailable."
        d=c.stereo_diagnostics
        lines=["Stereo execution: raw reconstructed and camera measurement binding verified at load.",
            "Current physical display integrity is checked separately."]
        for (role,camera) in enumerate((d.cam1,d.cam2))
            push!(lines,"Camera $role: dewarped pixels (not reconstructed world 3C residuals).")
            _append_execution_passes!(lines,camera;unit="dewarped px")
        end
        push!(lines,"Stereo per-node measurement history: unavailable.",
            "Tolerance outcomes are not measurement validity. Calibration, source inputs, uncertainty applicability, accuracy and coverage are not verified.")
        return join(lines,"\n")
    end
    lines=[c.history===nothing ? "Measurement history: not recorded." : "Measurement history: raw measurement binding verified."]
    if c.diagnostics===nothing
        push!(lines,"Execution diagnostics: not recorded.")
    else
        push!(lines,"Recorded execution counts; not verified against the displayed vector values.")
        _append_execution_passes!(lines,c.diagnostics)
        push!(lines,"Tolerance outcomes are not measurement validity.")
    end
    push!(lines,"History scope: final pass/final sweep only. Uncertainty applicability, accuracy and coverage are not established.")
    join(lines,"\n")
end
function _append_execution_passes!(lines,diagnostics;unit="px")
        for pass in diagnostics.passes
            push!(lines,"Pass $(pass.pass_index): $(pass.executed_iterations)/$(pass.requested_iterations) sweeps; $(replace(String(pass.stop_reason),'_'=>' ')); $(pass.checks) tolerance checks.")
            check=pass.last_check
            push!(lines,check===nothing ? "  Tolerance comparison: not evaluated." :
                "  Last q95 component change: $(check.value_state===:finite ? _fmt(check.value) : check.value_state) $unit; $(check.included_count) contributing nodes (empty support can meet tolerance).")
            residual=pass.residual
            push!(lines,"  Primary residual mean/RMS/max: $(residual.mean_magnitude===nothing ? "unavailable" : _fmt(residual.mean_magnitude))/$(residual.rms_magnitude===nothing ? "unavailable" : _fmt(residual.rms_magnitude))/$(residual.maximum_magnitude===nothing ? "unavailable" : _fmt(residual.maximum_magnitude)) $unit; $(residual.finite_count) finite unmasked nodes.")
        end
    lines
end

"""
    describe_companion_selection(ex::ResultExplorer) -> String

Describe one selected grid node's recorded primary/residual displacement in
pixels, first observed rejection stage, actual alternative/fill/restoration
events, final origin/flag and stored uncertainty numerical status. Use the
ordinary selection panel for final displayed physical units. No history is
inferred from current flags or reconstructed for missing entries. Integrity
checks are O(nodes), but the returned text/scalar accessor retains no arrays.
"""
function describe_companion_selection(ex::ResultExplorer)
    ex.companion_enabled[] || return ""
    ex.results isa _LazyDisplayResults && ex.results.source isa Hammerhead.ResultFile || return ""
    c=_checked_companions(ex)
    _companion_selection(ex,c)
end
function _companion_selection(ex,c)
    c.state in (:stereo_missing,:stereo_verified) && return "Stereo per-node measurement history and world 3C residuals are not recorded.\nUse the ordinary selection panel for displayed vector values."
    c.history===nothing && return ""
    sel=ex.selection[]
    sel isa CartesianIndex{2} || return "Select a grid node to inspect its recorded history."
    node=Hammerhead._history_node(c.history._data,sel) # caller checked packet integrity
    stage=node.rejection_name===nothing ? "none observed" : node.rejection_name
    rank=node.accepted_peak_rank==0 ? "none" : string(node.accepted_peak_rank)
    join(["Recorded node $(Tuple(sel))", "Raw x/y: $(_fmt(node.x)), $(_fmt(node.y)) px",
        "Primary u/v: $(_fmt(node.primary_u)), $(_fmt(node.primary_v)) px",
        "Primary residual u/v: $(_fmt(node.primary_residual_u)), $(_fmt(node.primary_residual_v)) px",
        "First observed rejection: $stage", "Accepted alternative rank: $rank",
        "Median attempted/assigned: $(node.fill_attempted)/$(node.fill_assigned)",
        "Primary restored: $(node.primary_restored)",
        "Final origin: $(replace(node.final_origin,'_'=>' ')); outlier flag: $(node.final_outlier); masked: $(node.masked)",
        "Stored u/v uncertainty: $(replace(node.uncertainty_u_status,'_'=>' '))/$(replace(node.uncertainty_v_status,'_'=>' '))",
        "Numerical availability does not establish applicability."],"\n")
end
function _companion_text(ex)
    if ex.companion_enabled[] && ex.results isa _LazyDisplayResults && ex.results.source isa Hammerhead.ResultFile
        c=_checked_companions(ex)
        summary=_companion_summary(c) # one display hash + one packet hash per refresh
        summary,_companion_selection(ex,c)
    else
        companion_summary(ex),""
    end
end

ResultExplorer(result::AnyResult; kwargs...) = ResultExplorer([result]; kwargs...)
function ResultExplorer(path::AbstractString;lazy::Bool=false,format::Symbol=:native)
    format===:native && return ResultExplorer(load_results(path;lazy);path)
    format===:timed_tracking || throw(ArgumentError("format must be :native or :timed_tracking"))
    lazy && throw(ArgumentError("timed tracking artifacts contain one complete bundle; lazy browsing is unsupported"))
    ResultExplorer(load_timed_tracking(path);path)
end

function Base.show(io::IO, ex::ResultExplorer)
    print(io, "ResultExplorer($(length(ex.results)) frame",
          length(ex.results) == 1 ? "" : "s",
          ", frame $(ex.frame[]), field :$(ex.field[]))")
end

"""
    nframes(ex::ResultExplorer) -> Int

Number of results in the explored sequence.
"""
nframes(ex::ResultExplorer) = length(ex.results)

"""
    current_result(ex::ResultExplorer)

The result at the current frame (one of `PIVResult`, `StereoPIVResult`,
`PTVResult`, `TrackingResult`).
"""
current_result(ex::ResultExplorer) = ex.results[ex.frame[]]

"""
    set_frame!(ex::ResultExplorer, i::Integer)

Move to frame `i`, clamped to `1:nframes(ex)`. A lazy read failure throws,
sets `status`, and preserves the prior frame, selection, and tool state.
"""
function set_frame!(ex::ResultExplorer, i::Integer)
    frame = clamp(i, 1, nframes(ex))
    try
        ex.results[frame] # preflight before notifying views or changing state
    catch err
        ex.status[] = first(split(sprint(showerror, err), '\n'))
        rethrow()
    end
    return ex.frame[] = frame
end

"""
    push_result!(ex::ResultExplorer, r)

Append a result to the explored sequence (routed through [`physical`](@ref)
like the constructor) and notify `ex.count`, so open views extend their frame
slider — this is how a live explorer follows a still-running batch. The
current frame is left unchanged. Lazy file-backed explorers reject appends.
"""
function push_result!(ex::ResultExplorer, r::AnyResult)
    ex.results isa _TimedDisplayResult && throw(ArgumentError("cannot append native results to a timed tracking explorer"))
    ex.results isa _LazyDisplayResults &&
        throw(ArgumentError("cannot append to a lazy ResultFile explorer; use an in-memory explorer for a live batch"))
    push!(ex.results, physical(r))
    ex.count[] = length(ex.results)
    return ex
end
push_result!(::ResultExplorer,::TimedTrackingResult) =
    throw(ArgumentError("timed tracking explorers are dedicated single bundles; appending is unsupported"))

# Whether a stored selection still refers to a valid item of the given result
# (a gridded window index for grids, a linear index for scattered results).
function _valid_selection(r, sel)
    sel === nothing && return nothing
    if r isa GridResult
        return (sel isa CartesianIndex{2} && checkbounds(Bool, r.u, sel)) ? sel : nothing
    elseif r isa PTVResult
        return (sel isa Int && 1 <= sel <= length(r.x)) ? sel : nothing
    else # TrackingResult
        return (sel isa Int && 1 <= sel <= length(_tracking_geometry(r).trajectories)) ? sel : nothing
    end
end

# Derived scalar fields (planar PIVResult only — the core derived.jl API is
# planar-typed; stereo has no gradient methods and PTV/Tracking no grid).
const DERIVED_FIELDS = (:vorticity, :divergence, :strain_rate,
                        :swirling_strength, :q_criterion)

"""
    available_fields(result) -> Vector{Symbol}

Scalar fields displayable for a result. Gridded results (`PIVResult` /
`StereoPIVResult`) offer `:magnitude` (in-plane or three-component magnitude,
in the result's current displacement or velocity units) plus component and
diagnostic fields, with
uncertainty fields included only when estimates are present. Planar results
additionally offer the derived fields `:vorticity`, `:divergence`,
`:strain_rate` (the magnitude `|S|`), `:swirling_strength`, and
`:q_criterion` (computed via `flow_derivatives` on grids with at least two
points per axis). A `PTVResult` offers `[:magnitude, :u, :v,
:match_residual]`; a `TrackingResult` offers `[:speed]` (per-trajectory mean
speed).
"""
function available_fields(r::PIVResult)
    fields = [:magnitude, :u, :v, :peak_ratio, :correlation_moment]
    any(isfinite, r.uncertainty_u) && push!(fields, :uncertainty_u)
    any(isfinite, r.uncertainty_v) && push!(fields, :uncertainty_v)
    length(r.x) >= 2 && length(r.y) >= 2 && append!(fields, DERIVED_FIELDS)
    return fields
end

function available_fields(r::StereoPIVResult)
    fields = [:magnitude, :u, :v, :w]
    any(isfinite, r.uncertainty_u) &&
        append!(fields, (:uncertainty_u, :uncertainty_v, :uncertainty_w))
    return fields
end

available_fields(::PTVResult) = [:magnitude, :u, :v, :match_residual]
available_fields(::TrackingResult) = [:speed]
available_fields(::TimedTrackingResult) = [:speed]
_tracking_geometry(r::TrackingResult) = r
_tracking_geometry(r::TimedTrackingResult) = r.result

# One derived scalar from a precomputed flow_derivatives NamedTuple.
_derived_field(d::NamedTuple, field::Symbol) =
    field === :vorticity ? vorticity(d) :
    field === :divergence ? divergence(d) :
    field === :strain_rate ? strain_rate(d).magnitude :
    field === :swirling_strength ? swirling_strength(d) :
    q_criterion(d)

# Only the current frame's derivatives are cached; frame changes evict them,
# including on eager explorers, so derived data cannot grow with a recording.
_derived(ex::ResultExplorer) =
    get!(() -> flow_derivatives(current_result(ex)), ex.derived_cache, ex.frame[])

"""
    field_values(result, field::Symbol)

Return the scalar field to display. For gridded results this is a `Matrix`:
`:magnitude` is the norm of the available components, derived fields from
[`available_fields`](@ref) use `flow_derivatives`, and other fields come
from the result's matrices. For a `PTVResult` it is a per-particle
`Vector` (`:magnitude`, `:u`, `:v`, `:match_residual`). For a
`TrackingResult` it is a per-trajectory `Vector` of mean speeds (`:speed`).
"""
function field_values(r::GridResult, field::Symbol)
    field === :magnitude &&
        return r isa StereoPIVResult ? hypot.(r.u, r.v, r.w) : hypot.(r.u, r.v)
    field in available_fields(r) ||
        throw(ArgumentError("field :$field is not available for this result"))
    field in DERIVED_FIELDS && return _derived_field(flow_derivatives(r), field)
    return getproperty(r, field)
end

# Explorer-aware variant of field_values: derived fields reuse the cached
# flow_derivatives of the current frame (the view and colorbar both read it).
function current_field_values(ex::ResultExplorer)
    r = current_result(ex)
    field = ex.field[]
    r isa PIVResult && field in DERIVED_FIELDS &&
        return _derived_field(_derived(ex), field)
    return field_values(r, field)
end

function field_values(r::PTVResult, field::Symbol)
    field === :magnitude && return hypot.(r.u, r.v)
    field in available_fields(r) ||
        throw(ArgumentError("field :$field is not available for this result"))
    return getproperty(r, field)
end

function field_values(r::TrackingResult, field::Symbol)
    field === :speed ||
        throw(ArgumentError("field :$field is not available for this result"))
    return [_mean_speed(t, r.scale) for t in r.trajectories]
end
function field_values(r::TimedTrackingResult,field::Symbol)
    field===:speed || throw(ArgumentError("field :$field is not available for this result"))
    tracking_speed_summary(r).speeds
end

const FIELD_NAMES = Dict(
    :magnitude => "|displacement|",
    :u => "u", :v => "v", :w => "w",
    :peak_ratio => "peak ratio",
    :correlation_moment => "correlation moment",
    :match_residual => "match residual",
    :speed => "speed",
    :uncertainty_u => "σu", :uncertainty_v => "σv", :uncertainty_w => "σw",
    :vorticity => "vorticity", :divergence => "divergence",
    :strain_rate => "strain rate |S|", :swirling_strength => "swirling strength",
    :q_criterion => "Q",
)

"""
    field_name(field::Symbol) -> String

Short display name of a scalar field (menu entries).
"""
field_name(field::Symbol) = FIELD_NAMES[field]

# Fallback units when no PhysicalScale is attached: pixels for planar / PTV /
# tracking, "world units" for stereo (world coordinates are already physical).
_fallback_unit(::StereoPIVResult) = "world units"
_fallback_unit(::AnyResult) = "px"

# Position (length) unit and displacement/velocity unit. After `physical`
# conversion a real attached scale leaves labelled unit strings; `nothing`
# means unscaled, so we use the type's fallback.
_length_unit(r::AnyResult) = r.scale === nothing ? _fallback_unit(r) : r.scale.length_unit
_field_unit(r::AnyResult) = r.scale === nothing ? _fallback_unit(r) :
    string(r.scale.length_unit, "/", r.scale.time_unit)
# Time unit for the derived-gradient fields: velocity/length cancels the
# length, so vorticity/divergence/strain are 1/time (Q is 1/time²). The
# `physical` conversion keeps this consistent: positions and displacements
# scale by the same length factor, so the gradients carry exactly 1/dt.
_time_unit(r::AnyResult) = r.scale === nothing ? "frame" : r.scale.time_unit
_length_unit(r::TimedTrackingResult) = r.result.scale===nothing ? "px" : r.result.scale.length_unit
function _field_unit(r::TimedTrackingResult)
    data=Hammerhead._tracking_check(r)
    string(_length_unit(r),"/",something(data["effective_time_unit"],"unknown sample-time unit"))
end
function field_label(r::TimedTrackingResult,field::Symbol)
    field===:speed || throw(ArgumentError("field :$field is not available for this result"))
    "observation-mean secant speed ($(_field_unit(r)))"
end

"""
    field_label(result, field::Symbol) -> String

Display name of a scalar field with the result's units appended (colorbar
label). Displacement/velocity fields and their uncertainties carry the
velocity unit (`length_unit/time_unit`, e.g. `mm/s`, or the `px`/`world units`
fallback when unscaled); a `PTVResult`'s `match_residual` carries the length
unit; the derived gradient fields carry `1/time_unit` (`1/time_unit²` for Q,
`1/frame` when unscaled); dimensionless diagnostics carry no unit.
"""
function field_label(r::AnyResult, field::Symbol)
    field===:magnitude && r.scale!==nothing && return string("speed (",_field_unit(r),")")
    field in (:peak_ratio, :correlation_moment) && return field_name(field)
    field === :match_residual && return string(field_name(field), " (", _length_unit(r), ")")
    field === :q_criterion && return string(field_name(field), " (1/", _time_unit(r), "²)")
    field in DERIVED_FIELDS && return string(field_name(field), " (1/", _time_unit(r), ")")
    return string(field_name(field), " (", _field_unit(r), ")")
end

"""
    set_field!(ex::ResultExplorer, field::Symbol)

Display `field` (must be in `available_fields(current_result(ex))`).
"""
function set_field!(ex::ResultExplorer, field::Symbol)
    field in available_fields(current_result(ex)) ||
        throw(ArgumentError("field :$field is not available for the current result"))
    ex.field[] = field
    return ex
end

# Item validity for the colorbar statistics: masked/outlier grid cells and
# flagged PTV particles are excluded from the robust range (they are exactly
# the values that stretch it); tracking speeds carry no flags.
_flagged(r::GridResult, i) = r.mask[i] || r.outliers[i]
_flagged(r::PTVResult, i) = r.outliers[i]
_flagged(::TrackingResult, i) = false
_flagged(::TimedTrackingResult,i) = false

# Nearest-rank percentile band of (unsorted) values; mutates `vals` by sorting.
function _percentile_band(vals::Vector{Float64}, plo::Real, phi::Real)
    sort!(vals)
    n = length(vals)
    lo = vals[clamp(round(Int, plo * (n - 1)) + 1, 1, n)]
    hi = vals[clamp(round(Int, phi * (n - 1)) + 1, 1, n)]
    return (lo, hi)
end

"""
    color_limits(result, field::Symbol, mode::Symbol = :robust) -> (lo, hi)

Automatic colorbar limits for a displayed field. `:full` is the extrema of
all finite values. `:robust` (the default) is the 2–98% percentile band over
finite values at *valid* items (non-masked, non-outlier grid cells;
non-flagged PTV particles) so a few outliers cannot stretch the color range;
when no valid values exist it falls back to all finite values. A degenerate
range is padded by ±0.5.
"""
color_limits(r::DisplayResult, field::Symbol, mode::Symbol = :robust) =
    _color_limits(r, field_values(r, field), mode)

function _color_limits(r::DisplayResult, data, mode::Symbol)
    mode in (:robust, :full) ||
        throw(ArgumentError("mode must be :robust or :full, got :$mode"))
    vals = Float64[]
    if mode === :robust
        for i in eachindex(data)
            (isfinite(data[i]) && !_flagged(r, i)) || continue
            push!(vals, Float64(data[i]))
        end
    end
    isempty(vals) && (vals = [Float64(v) for v in data if isfinite(v)])
    isempty(vals) && return (0.0, 1.0)
    lo, hi = mode === :robust ? _percentile_band(vals, 0.02, 0.98) : extrema(vals)
    lo == hi && ((lo, hi) = (lo - 0.5, hi + 0.5))
    return (lo, hi)
end

"""
    set_color_mode!(ex::ResultExplorer, mode::Symbol)

Set the automatic colorbar-limit mode: `:robust` (2–98% percentile band over
valid values, the default) or `:full` (extrema). See [`color_limits`](@ref).
"""
function set_color_mode!(ex::ResultExplorer, mode::Symbol)
    mode in (:robust, :full) ||
        throw(ArgumentError("mode must be :robust or :full, got :$mode"))
    ex.color_mode[] = mode
    return ex
end

# One manual colorbar bound: `nothing`, "", or "auto" clears the override;
# numbers and their string forms set it. Throws ArgumentError on junk.
_parse_limit(::Nothing) = nothing
_parse_limit(v::Real) =
    (isfinite(v) || throw(ArgumentError("color limit must be finite, got $v")); Float64(v))
function _parse_limit(s::AbstractString)
    t = strip(s)
    (isempty(t) || lowercase(t) == "auto") && return nothing
    v = tryparse(Float64, t)
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("color limit must be a number or \"auto\", got \"$s\""))
    return v
end

"""
    set_color_limits!(ex::ResultExplorer; min = missing, max = missing)

Manually override the colorbar limits, bound by bound. Each keyword accepts a
number (or its string form) to pin that bound, and `nothing` / `""` /
`"auto"` to clear it back to the automatic [`color_limits`](@ref); `missing`
leaves it unchanged. Manual bounds persist across frame and field switches
until cleared.
"""
function set_color_limits!(ex::ResultExplorer; min = missing, max = missing)
    min === missing || (ex.color_min[] = _parse_limit(min))
    max === missing || (ex.color_max[] = _parse_limit(max))
    return ex
end

"""
    current_color_limits(ex::ResultExplorer) -> (lo, hi)

The colorbar limits in effect for the current frame and field: the automatic
[`color_limits`](@ref) under `ex.color_mode`, with any manual
[`set_color_limits!`](@ref) overrides applied bound-wise (an inverted or
degenerate manual pair is padded to a valid range).
"""
function current_color_limits(ex::ResultExplorer)
    _current_color_limits(ex,current_result(ex),current_field_values(ex))
end
function _current_color_limits(ex,r,data)
    lo, hi = _color_limits(r,data,ex.color_mode[])
    ex.color_min[] === nothing || (lo = ex.color_min[])
    ex.color_max[] === nothing || (hi = ex.color_max[])
    if !(lo < hi)
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5
    end
    return (lo, hi)
end

"""
    select_nearest!(ex::ResultExplorer, x::Real, y::Real)

Select the item nearest to the data-space point `(x, y)` (e.g. a mouse
click) for inspection: the nearest grid node for a gridded result, the
nearest particle for a `PTVResult`, or the nearest trajectory (by vertex) for
a `TrackingResult`.
"""
function select_nearest!(ex::ResultExplorer, x::Real, y::Real)
    ex.selection[] = _nearest(current_result(ex), x, y)
    return ex
end

_nearest(r::GridResult, x, y) =
    CartesianIndex(argmin(i -> abs(r.y[i] - y), eachindex(r.y)),
                   argmin(j -> abs(r.x[j] - x), eachindex(r.x)))

function _nearest(r::PTVResult, x, y)
    isempty(r.x) && return nothing
    return argmin(k -> abs2(r.x[k] - x) + abs2(r.y[k] - y), eachindex(r.x))
end

function _nearest(r::TrackingResult, x, y)
    isempty(r.trajectories) && return nothing
    best, bestd = 1, Inf
    for (k, t) in pairs(r.trajectories), p in eachindex(t.x)
        d = abs2(t.x[p] - x) + abs2(t.y[p] - y)
        d < bestd && ((best, bestd) = (k, d))
    end
    return best
end
function _nearest(r::TimedTrackingResult,x,y)
    best,bestd=nothing,Inf
    for (k,t) in pairs(r.result.trajectories),p in eachindex(t.x)
        isfinite(t.x[p]) && isfinite(t.y[p]) || continue
        # hypot avoids overflowing squared distances for finite coordinates.
        d=hypot(t.x[p]-x,t.y[p]-y)
        d<bestd && ((best,bestd)=(k,d))
    end
    best
end

"""
    clear_selection!(ex::ResultExplorer)

Drop the current selection.
"""
clear_selection!(ex::ResultExplorer) = (ex.selection[] = nothing; ex)

# ---------------------------------------------------------------------------
# Interactive derived-analysis tools (planar PIVResult only)
# ---------------------------------------------------------------------------

const EXPLORER_TOOLS = (:inspect, :profile, :circulation)

# Clear the gesture points and computed outputs (frame switches, tool
# switches, and restarts all funnel through here).
function _reset_tool!(ex::ResultExplorer)
    isempty(ex.tool_points[]) || (empty!(ex.tool_points[]); notify(ex.tool_points))
    ex.profile_data[] === nothing || (ex.profile_data[] = nothing)
    ex.circulation_result[] === nothing || (ex.circulation_result[] = nothing)
    return ex
end

"""
    set_tool!(ex::ResultExplorer, tool::Symbol)

Select the explorer's interaction tool: `:inspect` (click selects the
nearest item — the default), `:profile` (two clicks define a line sampled
with `extract_profile`), or `:circulation` (clicks accumulate a contour,
[`alt_click!`](@ref) closes it and evaluates `circulation`). The analysis
tools need a planar `PIVResult` at the current frame; switching tools (or
frames) clears any in-progress gesture and outputs.
"""
function set_tool!(ex::ResultExplorer, tool::Symbol)
    tool in EXPLORER_TOOLS ||
        throw(ArgumentError("tool must be one of $(EXPLORER_TOOLS), got :$tool"))
    tool === :inspect || current_result(ex) isa PIVResult ||
        throw(ArgumentError("the :$tool tool needs a planar PIVResult at the current frame"))
    _reset_tool!(ex)
    ex.tool[] = tool
    return ex
end

"""
    click!(ex::ResultExplorer, x::Real, y::Real)

Route a click through the active tool: `:inspect` selects the nearest item
([`select_nearest!`](@ref)); `:profile` places a line endpoint (the profile
is computed when the second point lands, a third click starts a new line);
`:circulation` appends a contour vertex.
"""
function click!(ex::ResultExplorer, x::Real, y::Real)
    tool = ex.tool[]
    tool === :inspect && return select_nearest!(ex, x, y)
    pts = ex.tool_points[]
    if tool === :profile
        length(pts) >= 2 && (empty!(pts); ex.profile_data[] = nothing)
        push!(pts, (Float64(x), Float64(y)))
        notify(ex.tool_points)
        length(pts) == 2 && _compute_profile!(ex)
    else # :circulation
        ex.circulation_result[] === nothing ||
            (empty!(pts); ex.circulation_result[] = nothing)   # closed: start anew
        push!(pts, (Float64(x), Float64(y)))
        notify(ex.tool_points)
    end
    return ex
end

"""
    alt_click!(ex::ResultExplorer)

Close the `:circulation` contour (right-click in the view): with at least
three vertices the line-integral `circulation(r, contour)` and the
vorticity-area form `circulation(r; region = contour)` are both evaluated
into `circulation_result`; the area form also reports valid and requested
area and their coverage fraction. An incomplete area retains its partial
integral; zero valid area yields `NaN`. With fewer vertices the gesture is
cancelled. A no-op for the other tools.
"""
function alt_click!(ex::ResultExplorer)
    ex.tool[] === :circulation || return ex
    pts = ex.tool_points[]
    if length(pts) < 3
        _reset_tool!(ex)
        return ex
    end
    r = current_result(ex)
    contour = copy(pts)
    area_report = circulation(r; region = contour, coverage = :report)
    ex.circulation_result[] = (; line = circulation(r, contour),
                               area = area_report.value, contour,
                               area_report.valid_area, area_report.requested_area,
                               area_report.coverage_fraction, area_report.complete)
    return ex
end

"""
    clear_tool!(ex::ResultExplorer)

Clear the active tool's in-progress gesture and computed outputs (the tool
itself stays selected).
"""
clear_tool!(ex::ResultExplorer) = _reset_tool!(ex)

# Sample u/v along the two-point line. The panel always shows u, v, and |V|
# (extract_profile samples the velocity components; the displayed scalar
# field is not resampled — documented simplification).
function _compute_profile!(ex::ResultExplorer)
    r = current_result(ex)
    pts = ex.tool_points[]
    prof = extract_profile(r, pts; n = 100)
    ex.profile_data[] = prof
    return ex
end

# Circulation carries length²/time: px²/frame unscaled, e.g. mm²/s scaled.
_circulation_unit(r::AnyResult) =
    string(_length_unit(r), "²/", _time_unit(r))

"""
    tool_summary(ex::ResultExplorer) -> String

One-line status of the active analysis tool: gesture instructions while
points are being placed, the profile legend once a line is set, or the
computed circulation (line-integral and vorticity-area forms, with units and
area coverage).
"""
function tool_summary(ex::ResultExplorer)
    tool = ex.tool[]
    tool === :inspect && return ""
    r = current_result(ex)
    if tool === :profile
        ex.profile_data[] === nothing &&
            return "profile: click two points to sample a line"
        return "profile along the line: u (blue), v (orange), |V| (black)"
    end
    res = ex.circulation_result[]
    if res === nothing
        n = length(ex.tool_points[])
        return "circulation: click contour vertices ($n placed), right-click to close"
    end
    un = _circulation_unit(r)
    area_text = if res.coverage_fraction == 0
        "Γ (vorticity area): no valid area (0% coverage)"
    elseif !res.complete
        string("Γ (vorticity area, partial) = ", _fmt(res.area), " ", un,
               " (", _fmt(100 * res.coverage_fraction), "% coverage)")
    else
        string("Γ (vorticity area) = ", _fmt(res.area), " ", un)
    end
    return string("Γ (line) = ", _fmt(res.line), " ", un, "\n", area_text)
end

# Data-space point (x, y) marking the current selection, or `nothing` when the
# selection is empty or stale — the view draws a marker there.
function selection_point(r, sel)
    r isa TimedTrackingResult && Hammerhead._tracking_check(r)
    sel = _valid_selection(r, sel)
    sel === nothing && return nothing
    if r isa GridResult
        return (r.x[sel[2]], r.y[sel[1]])
    elseif r isa PTVResult
        return (r.x[sel], r.y[sel])
    else # TrackingResult: mark the trajectory's first point
        t = _tracking_geometry(r).trajectories[sel]
        if r isa TimedTrackingResult
            i=findfirst(p->isfinite(t.x[p]) && isfinite(t.y[p]),eachindex(t.x))
            return i===nothing ? nothing : (t.x[i],t.y[i])
        end
        return (t.x[1], t.y[1])
    end
end

_fmt(v::Real) = isfinite(v) ? @sprintf("%.4g", v) : "—"

_status(r::GridResult, idx) = r.mask[idx]     ? "masked (no measurement)" :
                              r.outliers[idx] ? "outlier (replaced)" : "valid"

"""
    describe_selection(ex::ResultExplorer) -> String

Multi-line summary of the selected item (empty string when nothing is
selected): position, displacement/velocity, diagnostics/uncertainty, and
validation status, all with units.
"""
function describe_selection(ex::ResultExplorer)
    r = current_result(ex)
    sel = _valid_selection(r, ex.selection[])
    sel === nothing && return ""
    return vector_summary(r, sel)
end

function vector_summary(r::PIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["window ($i, $j)",
             "x = $(_fmt(r.x[j])) $lu, y = $(_fmt(r.y[i])) $lu",
             "u = $(_fmt(r.u[idx])) $vu",
             "v = $(_fmt(r.v[idx])) $vu",
             "peak ratio = $(_fmt(r.peak_ratio[idx]))",
             "corr. moment = $(_fmt(r.correlation_moment[idx]))"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_v[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])) $vu, σv = $(_fmt(r.uncertainty_v[idx])) $vu")
    end
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

function vector_summary(r::StereoPIVResult, idx::CartesianIndex{2})
    i, j = Tuple(idx)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["node ($i, $j)",
             "x = $(_fmt(r.x[j])), y = $(_fmt(r.y[i])), z = $(_fmt(r.z)) ($lu)",
             "u = $(_fmt(r.u[idx])) $vu",
             "v = $(_fmt(r.v[idx])) $vu",
             "w = $(_fmt(r.w[idx])) $vu"]
    if isfinite(r.uncertainty_u[idx]) || isfinite(r.uncertainty_w[idx])
        push!(lines, "σu = $(_fmt(r.uncertainty_u[idx])), σv = $(_fmt(r.uncertainty_v[idx])), σw = $(_fmt(r.uncertainty_w[idx])) ($vu)")
    end
    push!(lines, "cam peak ratios: $(_fmt(r.cam1.peak_ratio[idx])) / $(_fmt(r.cam2.peak_ratio[idx]))")
    push!(lines, "status: $(_status(r, idx))")
    return join(lines, "\n")
end

function vector_summary(r::PTVResult, k::Int)
    lu, vu = _length_unit(r), _field_unit(r)
    lines = ["particle $k",
             "x = $(_fmt(r.x[k])) $lu, y = $(_fmt(r.y[k])) $lu",
             "u = $(_fmt(r.u[k])) $vu",
             "v = $(_fmt(r.v[k])) $vu",
             "match residual = $(_fmt(r.match_residual[k])) $lu",
             "status: $(r.outliers[k] ? "flagged (scattered UOD)" : "valid")"]
    return join(lines, "\n")
end

function vector_summary(r::TrackingResult, k::Int)
    t = r.trajectories[k]
    vu = _field_unit(r)
    n = length(t)
    spd = n >= 2 ? "$(_fmt(_mean_speed(t, r.scale))) $vu" : "—"
    lines = ["trajectory $k",
             "start frame $(t.start_frame)",
             "$n point" * (n == 1 ? "" : "s") * ", frames $(first(t.frames))–$(last(t.frames))",
             "gaps: $(trajectory_gap_count(t))",
             "mean speed = $spd"]
    return join(lines, "\n")
end

function vector_summary(r::TimedTrackingResult,k::Int)
    summary=tracking_speed_summary(r)
    t=summary.tracks[k]
    unit=string(summary.length_unit,"/",something(summary.time_unit,"unknown sample-time unit"))
    speed=summary.available[k] ? "$(_fmt(summary.speeds[k])) $unit" :
        "unavailable ($(replace(String(summary.reasons[k]),'_' => ' ')))"
    timeunit=something(summary.time_unit,"unknown unit")
    range=t.first_frame===nothing ? "unavailable" : "$(t.first_frame)–$(t.last_frame)"
    timerange=t.first_time===nothing ? "unavailable" : "$(t.first_time)–$(t.last_time) $timeunit"
    elapsed=t.elapsed===nothing ? "unavailable" : "$(t.elapsed) $timeunit"
    lines=["trajectory $k (whole track)",
        "$(t.observations) observations; selected frames $range",
        "gaps in selected frames: $(t.gaps)",
        "actual time: $timerange",
        "elapsed: $elapsed",
        "observation-mean secant speed:\n$speed"]
    summary.time_unit_provenance=="legacy_scale_same_unit" && push!(lines,"Time unit assumed from scale; acquisition unit unknown.")
    summary.time_unit===nothing && push!(lines,"Sample-time unit unknown; no physical time unit inferred.")
    push!(lines,"Clock: $(something(summary.clock_id,"unknown")); scope: $(replace(summary.source_scope,'_' => ' '))")
    push!(lines,"Secants describe time windows; not instantaneous speed.")
    join(lines,"\n")
end

# Mean speed along a trajectory (physical when `scale` is attached). Zero for
# a single-point track (trajectory_velocities needs ≥ 2 points).
function _mean_speed(t, scale)
    length(t) < 2 && return 0.0
    u, v = trajectory_velocities(t, scale)
    s = 0.0
    for k in eachindex(u)
        s += hypot(u[k], v[k])
    end
    return s / length(u)
end

"""
    trajectory_gap_count(t) -> Int

Number of bridged frame gaps in a trajectory: steps where `t.frames`
increases by more than one (nonconsecutive frames linked across a gap).
"""
trajectory_gap_count(t) = count(>(1), diff(t.frames))

"""
    trajectory_points(t) -> (xs, ys)

Flat polyline coordinates for a trajectory with `NaN` breaks inserted at
frame gaps, so a single `lines!` call renders one polyline per continuous run
(the standard Makie NaN-separation approach).
"""
function trajectory_points(t)
    xs = Float64[]; ys = Float64[]
    for p in eachindex(t.x)
        if p > 1 && t.frames[p] - t.frames[p - 1] > 1
            push!(xs, NaN); push!(ys, NaN)
        end
        push!(xs, t.x[p]); push!(ys, t.y[p])
    end
    return (xs, ys)
end

"""
    vector_data(result) -> NamedTuple

Return `(; x, y, u, v, outlier)` as flat arrays for an arrow overlay.
Gridded results skip nodes whose `u` or `v` is `NaN`; a `PTVResult` skips
matches with nonfinite displacement. The `outlier` flag remains available
for color coding, and `w` is not included for stereo results.
"""
function vector_data(r::GridResult)
    x = Float64[]; y = Float64[]; u = Float64[]; v = Float64[]; outlier = Bool[]
    for j in eachindex(r.x), i in eachindex(r.y)
        (isnan(r.u[i, j]) || isnan(r.v[i, j])) && continue
        push!(x, r.x[j]); push!(y, r.y[i])
        push!(u, r.u[i, j]); push!(v, r.v[i, j])
        push!(outlier, r.outliers[i, j])
    end
    return (; x, y, u, v, outlier)
end

function vector_data(r::PTVResult)
    keep = findall(k -> isfinite(r.u[k]) && isfinite(r.v[k]), eachindex(r.u))
    return (; x = Float64.(r.x[keep]), y = Float64.(r.y[keep]),
            u = Float64.(r.u[keep]), v = Float64.(r.v[keep]),
            outlier = collect(Bool, r.outliers[keep]))
end

"""
    auto_lengthscale(result, data = vector_data(result)) -> Float64

Arrow length scale that keeps the longest displayed vector at ~85% of the
vector spacing. Gridded results use the grid step; scattered (`PTVResult`)
data uses a robust `sqrt(area / n)` spacing proxy.
"""
function auto_lengthscale(r::GridResult, data = vector_data(r))
    dmax = 0.0
    for k in eachindex(data.u)
        dmax = max(dmax, hypot(data.u[k], data.v[k]))
    end
    spacing = min(minimum(abs.(diff(r.x)); init = Inf),
                  minimum(abs.(diff(r.y)); init = Inf))
    (isfinite(spacing) && dmax > 0) || return 1.0
    return 0.85 * spacing / dmax
end

function auto_lengthscale(r::PTVResult, data = vector_data(r))
    n = length(data.x)
    n == 0 && return 1.0
    dmax = 0.0
    for k in eachindex(data.u)
        dmax = max(dmax, hypot(data.u[k], data.v[k]))
    end
    dmax > 0 || return 1.0
    xspan = maximum(data.x) - minimum(data.x)
    yspan = maximum(data.y) - minimum(data.y)
    spacing = sqrt(max(xspan * yspan, 1.0) / n)
    return 0.85 * spacing / dmax
end
