const EXECUTION_DIAGNOSTICS_FORMAT_VERSION = 1

function _reject_execution_diagnostics(kwargs, workflow = "stereo")
    any(k -> haskey(kwargs, k), (:on_diagnostics, :record_diagnostics, :_diagnostics_association)) &&
        throw(ArgumentError("execution diagnostics currently support planar PIV only; $workflow diagnostics are not implemented"))
    _reject_measurement_history(kwargs,workflow)
    _reject_pair_timing(kwargs,workflow)
    nothing
end

"""
    PassDiagnostics

Immutable observations for one planar pass: requested/executed sweeps, actual
tolerance checks and stop reason, plus the final primary-peak residual summary.
Nested values are scalar named tuples, not mutable arrays or dictionaries.
The tolerance condition describes the predictor field after validation/filling;
it is not measurement validity. An empty comparison can satisfy that condition.
"""
struct PassDiagnostics
    pass_index::Int
    requested_iterations::Int
    executed_iterations::Int
    requested_tolerance::Float64
    stop_reason::Symbol
    checks::Int
    last_check::Union{Nothing,NamedTuple}
    residual::NamedTuple
end

"""
    PIVExecutionDiagnostics

Immutable companion observations from one completed planar `run_piv` execution.
`passes` is a tuple of [`PassDiagnostics`](@ref); no images, fields, planes or
sweep traces are retained. `pair_index` identifies the input-sequence position
when known. `association` is absent for generic calls or records verified
recipe/input IDs during experiment replay. Residuals remain processing pixels,
independent of attached physical scales. Result objects and native format 1 are
unchanged; older results have no recoverable execution diagnostics.
`core_source_sha256` records the current on-disk checkout when reporting; run in
a fresh process and keep sources unchanged to relate it to loaded Julia code.
It does not certify loaded modules after editing source files in that process.
"""
struct PIVExecutionDiagnostics
    execution_id::String
    backend::Symbol
    image_type::String
    processing_size::Tuple{Int,Int}
    core_source_sha256::String
    pair_index::Union{Nothing,Int}
    association::Union{Nothing,NamedTuple{(:recipe_id, :input_id),Tuple{String,String}}}
    passes::Tuple{Vararg{PassDiagnostics}}
end

mutable struct _PassObservation
    executed::Int
    checks::Int
    last_check::Union{Nothing,NamedTuple}
    stop_reason::Symbol
end
_PassObservation(maxiter) = _PassObservation(0, 0, nothing, maxiter == 1 ? :single_sweep : :iteration_budget)

function _execution_check!(observer, sweep, value, buffer, mask, tolerance)
    observer.checks += 1
    state = isnan(value) ? :nan : isinf(value) ? :infinite : :finite
    observer.last_check = (sweep = sweep, value_state = state,
        value = state === :finite ? Float64(value) : nothing,
        included_count = length(buffer), finite_count = count(isfinite, buffer),
        infinite_count = count(isinf, buffer), excluded_count = count(!, mask) - length(buffer),
        tolerance_met = value < tolerance)
    value < tolerance && (observer.stop_reason = :tolerance_condition_met)
    nothing
end

function _execution_residual(u, v, mask, predictor_present)
    finite_count = 0
    nonfinite_count = 0
    mean_magnitude = rms_magnitude = maximum_magnitude = 0.0
    for i in eachindex(u, v, mask)
        mask[i] && continue
        magnitude = hypot(Float64(u[i]), Float64(v[i]))
        if !isfinite(u[i]) || !isfinite(v[i]) || !isfinite(magnitude)
            nonfinite_count += 1
            continue
        end
        finite_count += 1
        mean_magnitude += (magnitude - mean_magnitude) / finite_count
        rms_magnitude = hypot(rms_magnitude * sqrt((finite_count - 1) / finite_count), magnitude / sqrt(finite_count))
        maximum_magnitude = max(maximum_magnitude, magnitude)
    end
    (finite_count = finite_count, nonfinite_count = nonfinite_count, masked_count = count(mask),
     mean_magnitude = finite_count == 0 ? nothing : mean_magnitude,
     rms_magnitude = finite_count == 0 ? nothing : rms_magnitude,
     maximum_magnitude = finite_count == 0 ? nothing : maximum_magnitude,
     predictor_present = predictor_present, unit = "px", value_basis = "primary_peak_before_validation")
end

function _execution_finish(reports, backend, T, processing_size)
    PIVExecutionDiagnostics(string(UUIDs.uuid4()), backend, string(T), processing_size,
        _experiment_software()["core_source_sha256"], nothing, nothing, Tuple(reports))
end
function _execution_context(d::PIVExecutionDiagnostics, i, association)
    PIVExecutionDiagnostics(d.execution_id, d.backend, d.image_type, d.processing_size,
        d.core_source_sha256, i, association, d.passes)
end

_execution_primitive(x::NamedTuple) = Dict{String,Any}(String(k) => _execution_primitive(v) for (k, v) in pairs(x))
_execution_primitive(x::Tuple) = [_execution_primitive(v) for v in x]
_execution_primitive(x::Symbol) = String(x)
_execution_primitive(x) = x

"""
    execution_diagnostics_data(diagnostics::PIVExecutionDiagnostics) -> Dict{String,Any}

Return detached version-1 primitive data. Nonfinite convergence values are
represented by `value_state` (`infinite`/`nan`) and `value=nothing`. A missing
check has `last_check=nothing`; an empty check has `included_count=0` and the
actual zero-valued tolerance decision. Residual availability uses `nothing`
statistics when no finite unmasked primary residual exists. No final-vector
peak/filling association, accuracy or convergence-validity claim is made.
"""
function execution_diagnostics_data(d::PIVExecutionDiagnostics)
    data = Dict{String,Any}(String(k) => _execution_primitive(getfield(d, k)) for k in fieldnames(PIVExecutionDiagnostics) if k !== :passes)
    data["passes"] = [Dict{String,Any}(String(k) => _execution_primitive(getfield(p, k)) for k in fieldnames(PassDiagnostics)) for p in d.passes]
    data["diagnostics_format_version"] = EXECUTION_DIAGNOSTICS_FORMAT_VERSION
    _execution_validate(data)
    data
end

_execution_error(message) = throw(ArgumentError(message))
_execution_count(v) = v isa Int && v >= 0
_execution_finite(v) = v isa Real && !(v isa Bool) && isfinite(v) && v >= 0
function _execution_sum(a, b)
    try Base.Checked.checked_add(a, b) catch err
        err isa OverflowError || rethrow()
        _execution_error("execution diagnostic counts overflow")
    end
end
function _execution_validate(data)
    _experiment_keys(data, [String.(fieldnames(PIVExecutionDiagnostics))...; "diagnostics_format_version"], "execution diagnostics")
    data["diagnostics_format_version"] === EXECUTION_DIAGNOSTICS_FORMAT_VERSION || _execution_error("unsupported execution diagnostics version")
    data["execution_id"] isa String && try UUIDs.UUID(data["execution_id"]); true catch; false end || _execution_error("invalid execution ID")
    data["backend"] in ("cpu", "ka", "cuda", "amdgpu") || _execution_error("invalid execution backend")
    data["image_type"] isa String && data["image_type"] in ("Float32", "Float64") || _execution_error("unsupported execution precision")
    shape = data["processing_size"]
    shape isa AbstractVector && length(shape) == 2 && all(v -> v isa Int && v > 0, shape) || _execution_error("invalid processing size")
    pixel_count = try Base.Checked.checked_mul(shape[1], shape[2]) catch err
        err isa OverflowError || rethrow()
        _execution_error("processing size overflows")
    end
    _experiment_hash(data["core_source_sha256"]) || _execution_error("invalid execution source identity")
    data["pair_index"] === nothing || (data["pair_index"] isa Int && data["pair_index"] > 0) || _execution_error("invalid execution pair index")
    association = data["association"]
    if association !== nothing
        _experiment_keys(association, ["recipe_id", "input_id"], "execution association")
        all(_experiment_hash, values(association)) || _execution_error("invalid execution association identities")
        data["pair_index"] !== nothing || _execution_error("associated execution requires pair index")
    end
    passes = data["passes"]
    passes isa AbstractVector && !isempty(passes) || _execution_error("execution diagnostics require passes")
    for (i, pass) in enumerate(passes)
        _experiment_keys(pass, String.(fieldnames(PassDiagnostics)), "pass diagnostics")
        pass["pass_index"] === i || _execution_error("pass indices must be contiguous")
        requested, executed, checks = pass["requested_iterations"], pass["executed_iterations"], pass["checks"]
        _execution_count(requested) && _execution_count(executed) && 1 <= executed <= requested && _execution_count(checks) || _execution_error("invalid pass iteration counts")
        tolerance = pass["requested_tolerance"]
        tolerance isa Real && !(tolerance isa Bool) && !isnan(tolerance) && tolerance >= 0 || _execution_error("invalid requested tolerance")
        stop = pass["stop_reason"]
        stop in ("single_sweep", "iteration_budget", "tolerance_condition_met") || _execution_error("invalid pass stop reason")
        requested == 1 ? (stop == "single_sweep" || _execution_error("invalid single-sweep outcome")) :
            (stop != "single_sweep" || _execution_error("invalid multi-sweep outcome"))
        stop != "tolerance_condition_met" && executed != requested && _execution_error("unchecked early pass termination")
        expected_checks = tolerance == 0 ? 0 : max(0, (stop == "tolerance_condition_met" ? executed : executed - 1) - 1)
        checks == expected_checks || _execution_error("tolerance check count disagrees with executed sweeps")
        stop == "tolerance_condition_met" && !(checks > 0 && 2 <= executed < requested) && _execution_error("invalid tolerance early stop")
        check = pass["last_check"]
        if checks == 0
            check === nothing || _execution_error("unexpected convergence check")
        else
            _experiment_keys(check, ["sweep", "value_state", "value", "included_count", "finite_count", "infinite_count", "excluded_count", "tolerance_met"], "convergence check")
            expected_sweep = stop == "tolerance_condition_met" ? executed : executed - 1
            check["sweep"] === expected_sweep || _execution_error("wrong last checked sweep")
            all(k -> _execution_count(check[k]), ("included_count", "finite_count", "infinite_count", "excluded_count")) || _execution_error("invalid convergence support counts")
            check["included_count"] == _execution_sum(check["finite_count"], check["infinite_count"]) || _execution_error("inconsistent convergence support")
            state = check["value_state"]
            state in ("finite", "infinite", "nan") || _execution_error("invalid convergence value state")
            state == "finite" ? (_execution_finite(check["value"]) || _execution_error("invalid convergence value")) :
                (check["value"] === nothing || _execution_error("nonfinite convergence must omit numeric value"))
            check["included_count"] == 0 && (state != "finite" || check["value"] != 0) && _execution_error("invalid empty comparison value")
            check["tolerance_met"] isa Bool && check["tolerance_met"] == (state == "finite" && check["value"] < tolerance) || _execution_error("invalid tolerance decision")
            check["tolerance_met"] == (stop == "tolerance_condition_met") || _execution_error("stop reason disagrees with tolerance decision")
        end
        residual = pass["residual"]
        _experiment_keys(residual, ["finite_count", "nonfinite_count", "masked_count", "mean_magnitude", "rms_magnitude", "maximum_magnitude", "predictor_present", "unit", "value_basis"], "primary residual")
        all(k -> _execution_count(residual[k]), ("finite_count", "nonfinite_count", "masked_count")) || _execution_error("invalid residual counts")
        _execution_sum(_execution_sum(residual["finite_count"], residual["nonfinite_count"]), residual["masked_count"]) <= pixel_count || _execution_error("residual node counts exceed processing size")
        residual["predictor_present"] isa Bool && residual["unit"] == "px" && residual["value_basis"] == "primary_peak_before_validation" || _execution_error("unsupported residual semantics")
        statistics = (residual["mean_magnitude"], residual["rms_magnitude"], residual["maximum_magnitude"])
        residual["finite_count"] == 0 ? (all(isnothing, statistics) || _execution_error("empty residual summary has values")) :
            (all(_execution_finite, statistics) || _execution_error("invalid residual summary values"))
        if check !== nothing
            _execution_sum(check["included_count"], check["excluded_count"]) == _execution_sum(residual["finite_count"], residual["nonfinite_count"]) || _execution_error("residual/check grid support mismatch")
        end
    end
    data
end

function _execution_decode(data)
    _execution_validate(data)
    passes = Tuple(PassDiagnostics(p["pass_index"], p["requested_iterations"], p["executed_iterations"], Float64(p["requested_tolerance"]),
        Symbol(p["stop_reason"]), p["checks"], p["last_check"] === nothing ? nothing :
        (sweep = p["last_check"]["sweep"], value_state = Symbol(p["last_check"]["value_state"]), value = p["last_check"]["value"],
         included_count = p["last_check"]["included_count"], finite_count = p["last_check"]["finite_count"],
         infinite_count = p["last_check"]["infinite_count"], excluded_count = p["last_check"]["excluded_count"], tolerance_met = p["last_check"]["tolerance_met"]),
        (finite_count = p["residual"]["finite_count"], nonfinite_count = p["residual"]["nonfinite_count"], masked_count = p["residual"]["masked_count"],
         mean_magnitude = p["residual"]["mean_magnitude"], rms_magnitude = p["residual"]["rms_magnitude"], maximum_magnitude = p["residual"]["maximum_magnitude"],
         predictor_present = p["residual"]["predictor_present"], unit = p["residual"]["unit"], value_basis = p["residual"]["value_basis"])) for p in data["passes"])
    association = data["association"] === nothing ? nothing : (recipe_id = data["association"]["recipe_id"], input_id = data["association"]["input_id"])
    PIVExecutionDiagnostics(data["execution_id"], Symbol(data["backend"]), data["image_type"], Tuple(data["processing_size"]),
        data["core_source_sha256"], data["pair_index"], association, passes)
end

_execution_key(key) = "execution_diagnostics/" * last(split(key, '/'))
function _write_execution_diagnostics(file, key, diagnostics)
    data = execution_diagnostics_data(diagnostics)
    if !haskey(file, "execution_diagnostics_format_version")
        file["execution_diagnostics_format_version"] = EXECUTION_DIAGNOSTICS_FORMAT_VERSION
    end
    file[_execution_key(key)] = Dict{String,Any}("result_key" => key, "diagnostics" => data)
    nothing
end

"""
    load_execution_diagnostics(index::ResultFile, i) -> PIVExecutionDiagnostics or nothing
    load_execution_diagnostics(path::AbstractString, i=1)

Read only the indexed native companion metadata, opening/closing the file per
access. Missing metadata means not recorded, not inferred from requested
settings. Validate schema version, exact result-key binding and scalar counts;
unknown or malformed companions throw. Existing native result readers ignore
these optional sibling groups. `pair_index` is the absolute position in the
driver's provided sequence, even in one-result-per-file output. Experiment
association records recipe/input IDs; the run's whole-file hash covers this
companion. No concurrent-writer or individual-vector association is promised.
"""
function load_execution_diagnostics(index::ResultFile, i::Integer)
    checkbounds(index, i)
    _check_result_file(index)
    key = "results/" * index.entry_keys[i]
    diagnostics = jldopen(index.path, "r") do file
        _check_results_format(file, index.path)
        has_version = haskey(file, "execution_diagnostics_format_version")
        if !has_version
            haskey(file, "execution_diagnostics") && _execution_error("execution metadata lacks format version")
            return nothing
        end
        file["execution_diagnostics_format_version"] === EXECUTION_DIAGNOSTICS_FORMAT_VERSION || _execution_error("unsupported execution diagnostics version")
        haskey(file, _execution_key(key)) || return nothing
        entry = file[_execution_key(key)]
        _experiment_keys(entry, ["result_key", "diagnostics"], "execution entry")
        entry["result_key"] == key || _execution_error("execution diagnostics/result key mismatch")
        _execution_decode(entry["diagnostics"])
    end
    _check_result_file(index)
    diagnostics
end
load_execution_diagnostics(path::AbstractString, i::Integer = 1) = load_execution_diagnostics(ResultFile(path), i)

Base.show(io::IO, d::PIVExecutionDiagnostics) = print(io, "PIVExecutionDiagnostics(", length(d.passes), " passes)")
function Base.show(io::IO, ::MIME"text/plain", d::PIVExecutionDiagnostics)
    execution_diagnostics_data(d)
    print(io, "Planar execution: ", d.backend, ", ", d.image_type)
    d.pair_index === nothing || print(io, ", pair ", d.pair_index)
    for pass in d.passes
        print(io, "\nPass ", pass.pass_index, ": ", pass.executed_iterations, '/', pass.requested_iterations,
            " sweeps; ", replace(String(pass.stop_reason), '_' => ' '), "; ", pass.checks, " tolerance checks")
        check = pass.last_check
        if check === nothing
            print(io, "\n  Tolerance comparison: not evaluated")
        else
            print(io, "\n  Last check: sweep ", check.sweep, ", q95 component change ", check.value_state === :finite ? check.value : check.value_state,
                " px, ", check.included_count, " contributing nodes")
            check.included_count == 0 && print(io, " (empty support)")
        end
        pass.stop_reason === :iteration_budget && print(io, "\n  Final budgeted sweep: tolerance not checked")
        r = pass.residual
        print(io, "\n  Primary residual before validation: ", r.finite_count, " finite / ", r.finite_count + r.nonfinite_count,
            " unmasked nodes; mean/RMS/max ", r.mean_magnitude, '/', r.rms_magnitude, '/', r.maximum_magnitude, " px")
    end
    print(io, "\nTolerance outcomes are not measurement validity. Primary residuals are not associated with substituted or filled output vectors.")
end
