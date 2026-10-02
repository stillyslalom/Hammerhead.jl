# Saved quality summaries describe stored fields, not absent measurement history.
const QUALITY_REPORT_FORMAT_VERSION = 1
const _QUALITY_UNAVAILABLE = Dict(
    "rejection_events" => "not_persisted", "replacement_history" => "not_persisted",
    "alternative_peak_history" => "not_persisted", "uncertainty_measurement_association" => "not_persisted",
    "accuracy" => "not_evaluated", "uncertainty_coverage" => "not_evaluated",
    "peak_locking" => "not_evaluated", "recipe_sensitivity" => "not_compared")

"""
    RunQualityReport

Detached aggregate coverage of stored planar/stereo fields. Access its primitive
schema through [`quality_report_data`](@ref); text/plain display shows counts,
fractions and explicitly unavailable diagnostics. Reports retain no result,
recipe, mask, correlation-plane, or uncertainty arrays. Locator metadata may
grow with the known experiment inputs; aggregate counters have fixed size.
"""
struct RunQualityReport
    _data::Dict{String,Any}
    _protected_paths::Tuple{Vararg{String}}
    _identity::String
end

_quality_error(message) = throw(ArgumentError(message))
_quality_components(kind) = kind == "planar" ? ("u", "v") : ("u", "v", "w")
function _quality_counter_names(kind)
    names = ["entries", "nodes", "masked", "unmasked", "outlier_flagged_unmasked",
        "finite_output_unmasked", "unflagged_finite_output_unmasked", "flagged_finite_output_unmasked",
        "uncertainty_requested_entries", "uncertainty_requested_unmasked_nodes",
        "uq_all_available_unmasked", "unflagged_finite_output_with_uq_available",
        "uq_all_unavailable_when_requested_unmasked"]
    for component in _quality_components(kind), state in ("available", "negative_finite", "nonfinite")
        push!(names, "uq_$(component)_$(state)_unmasked")
    end
    names
end
function _quality_metric_specs(kind)
    specs = Dict(
        "masked_fraction" => ("masked", "nodes"),
        "current_outlier_flag_fraction" => ("outlier_flagged_unmasked", "unmasked"),
        "finite_output_fraction" => ("finite_output_unmasked", "unmasked"),
        "unflagged_finite_output_fraction" => ("unflagged_finite_output_unmasked", "unmasked"),
        "stored_uq_all_numerically_available_fraction" => ("uq_all_available_unmasked", "unmasked"),
        "stored_uq_on_unflagged_finite_output_fraction" => ("unflagged_finite_output_with_uq_available", "unflagged_finite_output_unmasked"),
        "stored_uq_unavailable_when_requested_fraction" => ("uq_all_unavailable_when_requested_unmasked", "uncertainty_requested_unmasked_nodes"))
    for component in _quality_components(kind)
        specs["stored_uq_$(component)_numerically_available_fraction"] = ("uq_$(component)_available_unmasked", "unmasked")
    end
    specs
end
function _quality_fractions(counts, kind)
    Dict{String,Any}(name => begin
        numerator, denominator = counts[keys[1]], counts[keys[2]]
        metric = Dict{String,Any}("numerator" => numerator, "denominator" => denominator,
                                  "denominator_count" => keys[2], "unit" => "1", "available" => denominator > 0)
        denominator > 0 && (metric["value"] = numerator / denominator)
        metric
    end for (name, keys) in _quality_metric_specs(kind))
end
_quality_add!(counts, key, n = 1) = (counts[key] = Base.Checked.checked_add(counts[key], n))
function _quality_sum(values...)
    try
        foldl(Base.Checked.checked_add, values; init = 0)
    catch err
        err isa OverflowError || rethrow()
        _quality_error("quality counters overflow")
    end
end

function _quality_update!(groups, result)
    result isa Union{PIVResult,StereoPIVResult} || _quality_error("quality reports support planar/stereo PIV results only, got $(typeof(result))")
    kind = result isa PIVResult ? "planar" : "stereo"
    components = _quality_components(kind)
    fields = Tuple(getproperty(result, Symbol(c)) for c in components)
    uncertainties = Tuple(getproperty(result, Symbol("uncertainty_" * c)) for c in components)
    dimensions = (length(result.y), length(result.x))
    all(a -> size(a) == dimensions, (fields..., uncertainties..., result.mask, result.outliers)) ||
        throw(DimensionMismatch("quality-report fields and flags must match the vector coordinate grid"))
    counts = get!(groups, kind) do
        Dict(name => 0 for name in _quality_counter_names(kind))
    end
    _quality_add!(counts, "entries")
    _quality_add!(counts, "nodes", length(result.u))
    requested = result.parameters.uncertainty
    requested && _quality_add!(counts, "uncertainty_requested_entries")
    for i in eachindex(result.u)
        if result.mask[i]
            _quality_add!(counts, "masked")
            continue
        end
        _quality_add!(counts, "unmasked")
        requested && _quality_add!(counts, "uncertainty_requested_unmasked_nodes")
        flagged = result.outliers[i]
        flagged && _quality_add!(counts, "outlier_flagged_unmasked")
        finite = all(field -> isfinite(field[i]), fields)
        if finite
            _quality_add!(counts, "finite_output_unmasked")
            _quality_add!(counts, flagged ? "flagged_finite_output_unmasked" : "unflagged_finite_output_unmasked")
        end
        available = true
        for (c, uncertainty) in zip(components, uncertainties)
            value = uncertainty[i]
            state = !isfinite(value) ? "nonfinite" : value < 0 ? "negative_finite" : "available"
            _quality_add!(counts, "uq_$(c)_$(state)_unmasked")
            available &= state == "available"
        end
        if available
            _quality_add!(counts, "uq_all_available_unmasked")
            finite && !flagged && _quality_add!(counts, "unflagged_finite_output_with_uq_available")
        elseif requested
            _quality_add!(counts, "uq_all_unavailable_when_requested_unmasked")
        end
    end
    nothing
end

function _quality_validate(data)
    data isa AbstractDict || _quality_error("malformed quality report")
    _experiment_keys(data, ["quality_report_format_version", "generated_at_unix_s", "generator",
        "provenance", "protected_locators", "groups", "unavailable"], "quality report")
    data["quality_report_format_version"] isa Int && data["quality_report_format_version"] == QUALITY_REPORT_FORMAT_VERSION ||
        _quality_error("unsupported quality report format version")
    data["generated_at_unix_s"] isa Real && !(data["generated_at_unix_s"] isa Bool) &&
        isfinite(data["generated_at_unix_s"]) && data["generated_at_unix_s"] >= 0 || _quality_error("invalid report timestamp")
    _experiment_keys(data["generator"], ["julia_version", "hammerhead_version", "core_source_sha256", "value_basis", "weighting"], "quality generator")
    all(v -> v isa String, values(data["generator"])) && data["generator"]["value_basis"] == "stored_arrays" &&
        data["generator"]["weighting"] == "node_weighted" && _experiment_hash(data["generator"]["core_source_sha256"]) || _quality_error("unsupported quality metric basis")
    locators = data["protected_locators"]
    locators isa AbstractVector && all(p -> p isa String && isabspath(p), locators) || _quality_error("invalid protected report locators")
    provenance = data["provenance"]
    provenance isa AbstractDict && haskey(provenance, "association") || _quality_error("invalid quality provenance")
    association = provenance["association"]
    common = haskey(provenance, "source_path") ? ["source_path", "source_sha256", "source_index_entries", "source_selection"] : String[]
    associated = association == "recorded_output_verified"
    association in ("unassociated", "recorded_output_verified") || _quality_error("unsupported quality association")
    extra = associated ? ["recipe_id", "input_id", "run_id", "run_environment_id", "completed_pairs"] : String[]
    _experiment_keys(provenance, ["association"; common; extra], "quality provenance")
    if !isempty(common)
        provenance["source_path"] isa String && isabspath(provenance["source_path"]) &&
            provenance["source_path"] in locators && _experiment_hash(provenance["source_sha256"]) &&
            provenance["source_index_entries"] isa Int && provenance["source_index_entries"] >= 0 &&
            provenance["source_selection"] in ("whole_file", "provided_array") || _quality_error("invalid report source provenance")
    end
    if associated
        !isempty(common) && provenance["source_selection"] == "whole_file" || _quality_error("associated report requires a whole result file")
        all(k -> _experiment_hash(provenance[k]), ("recipe_id", "input_id", "run_environment_id")) || _quality_error("invalid report identities")
        provenance["run_id"] isa String || _quality_error("invalid report run ID")
        try UUIDs.UUID(provenance["run_id"]) catch; _quality_error("invalid report run UUID") end
        provenance["completed_pairs"] isa Int && provenance["completed_pairs"] >= 0 &&
            provenance["completed_pairs"] == provenance["source_index_entries"] || _quality_error("report source/run counts disagree")
    end
    groups = data["groups"]
    groups isa AbstractDict && all(k -> k in ("planar", "stereo"), keys(groups)) || _quality_error("unsupported report result kind")
    total_entries = 0
    for (kind, group) in groups
        _experiment_keys(group, ["counts", "fractions"], "quality group")
        counts = group["counts"]
        _experiment_keys(counts, _quality_counter_names(kind), "quality counters")
        all(v -> v isa Int && v >= 0, values(counts)) || _quality_error("quality counts must be nonnegative integers")
        counts["nodes"] == _quality_sum(counts["masked"], counts["unmasked"]) &&
            counts["outlier_flagged_unmasked"] <= counts["unmasked"] &&
            counts["finite_output_unmasked"] == _quality_sum(counts["unflagged_finite_output_unmasked"], counts["flagged_finite_output_unmasked"]) &&
            counts["unflagged_finite_output_unmasked"] <= counts["unmasked"] - counts["outlier_flagged_unmasked"] &&
            counts["flagged_finite_output_unmasked"] <= counts["outlier_flagged_unmasked"] &&
            counts["uncertainty_requested_entries"] <= counts["entries"] &&
            counts["uncertainty_requested_unmasked_nodes"] <= counts["unmasked"] &&
            counts["uq_all_unavailable_when_requested_unmasked"] <= counts["uncertainty_requested_unmasked_nodes"] &&
            counts["unflagged_finite_output_with_uq_available"] <= min(counts["unflagged_finite_output_unmasked"], counts["uq_all_available_unmasked"]) ||
            _quality_error("inconsistent quality counters")
        for c in _quality_components(kind)
            _quality_sum(counts["uq_$(c)_available_unmasked"], counts["uq_$(c)_negative_finite_unmasked"], counts["uq_$(c)_nonfinite_unmasked"]) == counts["unmasked"] &&
                counts["uq_all_available_unmasked"] <= counts["uq_$(c)_available_unmasked"] || _quality_error("inconsistent uncertainty counters")
        end
        fractions = group["fractions"]
        _experiment_keys(fractions, collect(keys(_quality_metric_specs(kind))), "quality fractions")
        for metric in values(fractions)
            metric isa AbstractDict && haskey(metric, "available") && metric["available"] isa Bool || _quality_error("invalid quality fraction availability")
            all(k -> haskey(metric, k) && metric[k] isa Int, ("numerator", "denominator")) || _quality_error("invalid quality fraction counts")
            haskey(metric, "value") && !(metric["value"] isa AbstractFloat) && _quality_error("invalid quality fraction value")
        end
        isequal(fractions, _quality_fractions(counts, kind)) || _quality_error("quality fractions disagree with their counts or denominators")
        counts["entries"] > 0 || counts["nodes"] == 0 || _quality_error("nonempty grid without result entries")
        total_entries = _quality_sum(total_entries, counts["entries"])
    end
    associated && total_entries != provenance["completed_pairs"] && _quality_error("associated result count mismatch")
    !isempty(common) && provenance["source_selection"] == "whole_file" && total_entries != provenance["source_index_entries"] && _quality_error("whole-file result count mismatch")
    expected_unavailable = Dict(k => Dict("available" => false, "reason_code" => v) for (k, v) in _QUALITY_UNAVAILABLE)
    data["unavailable"] isa AbstractDict && all(v -> v isa AbstractDict && get(v, "available", nothing) === false, values(data["unavailable"])) || _quality_error("invalid unavailable diagnostic availability")
    isequal(data["unavailable"], expected_unavailable) || _quality_error("unsupported or invented quality diagnostics")
    data
end
function _quality_wrap(data)
    _quality_validate(data)
    snapshot = deepcopy(Dict{String,Any}(data))
    RunQualityReport(snapshot, Tuple(snapshot["protected_locators"]), _experiment_digest(snapshot))
end

"""
    quality_report(results) -> RunQualityReport
    quality_report(record::ExperimentRecord, run::ExperimentRun) -> RunQualityReport

Aggregate a finite planar/stereo result iterator, including lazy [`ResultFile`](@ref)
and array views, without retaining results. Planar and stereo groups are separate;
stereo counts describe its reconstructed fields/union flags, not camera histories.
Grid dimensions may vary. Counts/fractions are node-weighted, not frame averages.

Masks use the full-grid denominator; other coverage uses unmasked nodes. Finite
output requires every displacement component finite, including flagged output.
Unflagged-finite output additionally excludes current outlier flags. Stored UQ
is numerically available only when finite and nonnegative, per component/jointly;
finite negative entries have separate counters. Zero denominators have
`available=false` and no value. No physical conversion or amplitude averaging is
performed; all fractions are dimensionless. Numeric UQ availability is not
measurement association, validity, calibrated coverage, or accuracy.

Flagged finite output is NOT a replacement count. Substituted peaks may clear
flags; filling may retain flags. Rejection events/replacements/alternatives and
UQ association are explicitly unavailable; accuracy, coverage, peak locking and
recipe sensitivity are not evaluated.

Generic sequences/files have no asserted experiment association. The record/run
overload validates snapshot/input/run metadata, requires a completed run and
matching result count, and verifies the recorded output SHA-256 before and after
lazy reading. It does not load input images/scripts. A file report also hashes
its source before/after; completed files must remain unchanged, with no concurrent
writer guarantee. Source/input/script/record/result locators known at construction
are protected by [`save_quality_report`](@ref).

Retained metric memory is constant; file indexes retain O(entries) keys and
known experiment locators retain O(inputs/runs) strings. Loading one native
entry includes any saved correlation planes/camera fields. Anonymous iterators
cannot reveal hidden file dependencies; protect those explicitly when saving.
"""
function quality_report(results)
    Base.IteratorSize(typeof(results)) isa Base.IsInfinite && _quality_error("quality reports require a finite sequence")
    groups = Dict{String,Dict{String,Int}}()
    source = _result_file_source(results)
    provenance = Dict{String,Any}("association" => "unassociated")
    locators = unique!(abspath.(_result_protected_paths(results)))
    if source !== nothing
        _check_result_file(source)
        digest = _experiment_file_digest(source.path)
        _check_result_file(source)
        merge!(provenance, Dict("source_path" => source.path, "source_sha256" => digest,
            "source_index_entries" => length(source), "source_selection" => results === source ? "whole_file" : "provided_array"))
        source.path in locators || push!(locators, source.path)
    end
    for result in results
        _quality_update!(groups, result)
    end
    if source !== nothing
        _check_result_file(source)
        _experiment_file_digest(source.path) == provenance["source_sha256"] || _quality_error("result source changed while building quality report")
        _check_result_file(source)
    end
    data = Dict{String,Any}("quality_report_format_version" => QUALITY_REPORT_FORMAT_VERSION,
        "generated_at_unix_s" => time(),
        "generator" => Dict("julia_version" => string(VERSION), "hammerhead_version" => string(Base.pkgversion(Hammerhead)),
                            "core_source_sha256" => _experiment_software()["core_source_sha256"],
                            "value_basis" => "stored_arrays", "weighting" => "node_weighted"),
        "provenance" => provenance, "protected_locators" => locators,
        "groups" => Dict(kind => Dict("counts" => counts, "fractions" => _quality_fractions(counts, kind)) for (kind, counts) in groups),
        "unavailable" => Dict(k => Dict("available" => false, "reason_code" => v) for (k, v) in _QUALITY_UNAVAILABLE))
    _quality_wrap(data)
end

function quality_report(record::ExperimentRecord, run::ExperimentRun)
    _experiment_preflight(record)
    _experiment_validate_environment(record.creation_environment)
    validated = _experiment_run(_experiment_run_data(run), record)
    validated.status === :completed && validated.output_sha256 !== nothing || _quality_error("quality association requires a completed run with output identity")
    source = ResultFile(validated.output)
    length(source) == validated.completed_pairs || _quality_error("completed run/result entry counts disagree")
    _experiment_file_digest(source.path) == validated.output_sha256 || _quality_error("recorded result output changed")
    report = quality_report(source)
    data = quality_report_data(report)
    data["provenance"]["source_sha256"] == validated.output_sha256 || _quality_error("recorded result output changed while building report")
    merge!(data["provenance"], Dict("association" => "recorded_output_verified", "recipe_id" => validated.recipe_id,
        "input_id" => validated.input_id, "run_id" => validated.run_id, "completed_pairs" => validated.completed_pairs,
        "run_environment_id" => _experiment_digest(_experiment_environment_signature(validated.environment))))
    locators = data["protected_locators"]
    append!(locators, [file["path"] for file in record.input_files])
    record.recipe.external_preprocess === nothing || push!(locators, record.recipe.external_preprocess.path)
    append!(locators, record.record_paths)
    append!(locators, [saved.output for saved in record.runs])
    data["protected_locators"] = sort!(unique!(abspath.(locators)))
    _quality_wrap(data)
end

"""
    quality_report_data(report::RunQualityReport) -> Dict{String,Any}

Return a detached primitive schema (strings, integers, booleans, finite floats,
arrays and mappings) for scripts/GUI or language-neutral serialization. Includes
format version, generation/runtime provenance, source/association identities,
protected locators, planar/stereo counts and fractions, and unavailable reasons.
Modifying this dictionary does not alter the report.
"""
function quality_report_data(report::RunQualityReport)
    _experiment_digest(report._data) == report._identity || _quality_error("quality report changed after construction")
    deepcopy(report._data)
end

"""
    save_quality_report(path, report; protected_paths=String[]) -> path

Validate the complete report and serialize it to TOML before opening/replacing
the destination. Refuse normalized-path or filesystem same-file aliases of all
known sources/experiment inputs/scripts/records/results and caller-provided
`protected_paths` (including dependencies hidden by anonymous iterators).
Ordinary report destinations may be overwritten. This is not atomic publication
or concurrent-writer safety; filesystem write errors can leave partial output.
"""
function save_quality_report(path::AbstractString, report::RunQualityReport; protected_paths = String[])
    data = quality_report_data(report)
    _quality_validate(data)
    all(p -> p isa AbstractString, protected_paths) || _quality_error("protected_paths must contain paths")
    protected = [data["protected_locators"]...; abspath.(protected_paths)...]
    any(p -> _experiment_alias(path, p), protected) && _quality_error("quality report destination aliases a protected source, input, script, record, or result")
    io = IOBuffer()
    TOML.print(io, data; sorted = true)
    text = String(take!(io))
    open(path, "w") do output
        write(output, text)
    end
    path
end

"""
    load_quality_report(path) -> RunQualityReport

Read and validate a version-1 TOML quality report without opening any recorded
source/input/script locators. Unknown versions, malformed identities/counters,
invented unsupported diagnostics, and inconsistent fractions are rejected.
Stored provenance records a past verification; loading does not reverify files.
"""
load_quality_report(path::AbstractString) = _quality_wrap(TOML.parsefile(path))

Base.show(io::IO, report::RunQualityReport) = print(io, "RunQualityReport(", join(sort!(collect(keys(report._data["groups"]))), ", "), ")")
function Base.show(io::IO, ::MIME"text/plain", report::RunQualityReport)
    data = quality_report_data(report)
    labels = Dict(
        "masked_fraction" => "Masked nodes",
        "current_outlier_flag_fraction" => "Current outlier flags",
        "finite_output_fraction" => "Finite output",
        "unflagged_finite_output_fraction" => "Unflagged finite output",
        "stored_uq_all_numerically_available_fraction" => "Stored uncertainty available in all components",
        "stored_uq_on_unflagged_finite_output_fraction" => "Stored uncertainty available on unflagged finite output",
        "stored_uq_unavailable_when_requested_fraction" => "Stored uncertainty unavailable when requested")
    for c in ("u", "v", "w")
        labels["stored_uq_$(c)_numerically_available_fraction"] = "Stored $c uncertainty available"
    end
    denominators = Dict("nodes" => "all nodes", "unmasked" => "unmasked nodes",
        "unflagged_finite_output_unmasked" => "unflagged finite output nodes",
        "uncertainty_requested_unmasked_nodes" => "unmasked nodes with uncertainty requested")
    association = data["provenance"]["association"] == "recorded_output_verified" ? "verified recorded output" : "unassociated"
    print(io, "Run quality: stored-array counts (node-weighted)\nExperiment association: ", association)
    for (kind, group) in sort!(collect(data["groups"]); by = first)
        print(io, '\n', uppercasefirst(kind), ": ", group["counts"]["entries"], " entries, ", group["counts"]["nodes"], " nodes")
        for (name, metric) in sort!(collect(group["fractions"]); by = first)
            print(io, "\n  ", labels[name], ": ")
            metric["available"] ? print(io, round(100 * metric["value"]; digits = 2), "%") : print(io, "unavailable (zero denominator)")
            print(io, " [", metric["numerator"], " / ", metric["denominator"], ' ', denominators[metric["denominator_count"]], ']')
        end
        print(io, "\n  Flagged finite output: ", group["counts"]["flagged_finite_output_unmasked"], " (not a replacement count)")
        for c in _quality_components(kind)
            print(io, "\n  Stored $c uncertainty: ", group["counts"]["uq_$(c)_negative_finite_unmasked"],
                " finite negative, ", group["counts"]["uq_$(c)_nonfinite_unmasked"], " nonfinite (unmasked nodes)")
        end
    end
    print(io, "\nStored uncertainty availability means finite and nonnegative; it does not establish measurement association or calibrated coverage.")
    for (name, reason) in sort!(collect(data["unavailable"]); by = first)
        label = name == "uncertainty_measurement_association" ? "Uncertainty measurement association" : uppercasefirst(replace(name, '_' => ' '))
        print(io, "\n", label, ": unavailable (", replace(reason["reason_code"], '_' => ' '), ')')
    end
end
