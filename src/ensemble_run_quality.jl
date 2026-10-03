# Version 4 is an explicit, unassociated whole-file view of recorded pools.
import JLD2: Group
# A ResultFile is structurally immutable, but its public key Vector is not.
# Compare it against disk, then retain an independent key list for this request.
function _quality_whole_file_index(index::ResultFile)
    _check_result_file(index)
    actual = jldopen(index.path, "r") do file
        _check_results_format(file, index.path)
        if haskey(file, "results")
            file["results"] isa Group || _quality_error("whole-file reports require a native results group")
            sort!(collect(String, keys(file["results"])))
        else
            String[]
        end
    end
    _check_result_file(index)
    index.entry_keys == actual || _quality_error("whole-file result index keys disagree with the native source; create a fresh ResultFile")
    ResultFile(index.path, actual, index.file_size, index.modified)
end
const QUALITY_ENSEMBLE_REPORT_FORMAT_VERSION = 4
const _QUALITY_ENSEMBLE_CLASSIFICATION = ("planar_entries_examined", "recorded_ensemble_entries",
    "recorded_planar_iteration_entries", "entries_without_execution_metadata")
const _QUALITY_ENSEMBLE_CONTRIBUTIONS = ("window_opportunities", "masked_window_pairs",
    "source_gated_window_pairs", "accumulated_window_pairs", "finite_zero_planes",
    "finite_flat_nonzero_planes", "finite_nonflat_planes", "nonfinite_planes")
function _quality_ensemble_counter_names()
    ["pair_observations", "passes", "executed_pooling_sweeps", "ignored_requested_iterations",
     "passes_with_positive_ignored_tolerance", "tolerance_checks",
     ["all_pass_$k" for k in _QUALITY_ENSEMBLE_CONTRIBUTIONS]...,
     ["all_pass_$k" for k in ("no_predictor_pairs", "evaluated_pairs", "disabled_nonfinite_source_pairs")]...,
     "all_pass_pair_observations", "final_nodes", "final_masked", "final_unmasked",
     ["final_$k" for k in ("zero_finite_nonzero_count", "some_finite_nonzero_count",
        "all_finite_nonzero_count", "nodes_with_nonfinite_plane")]...,
     "final_primary_finite", "final_primary_nonfinite", "final_uq_requested_entries",
     "final_uq_evaluated_entries", "final_uq_evaluated_unmasked_nodes",
     "final_uq_admitted_window_pair_updates", "final_uq_source_gated_window_pairs",
     ["final_uq_$(component)_$k" for component in ("u", "v") for k in
        ("finite_nonnegative_count", "finite_negative_count", "nonfinite_count")]...]
end
function _quality_ensemble_fractions(classification, counts)
    _quality_make_fractions(merge(counts, classification), Dict(
        "recorded_ensemble_fraction_of_planar_entries" => ("recorded_ensemble_entries", "planar_entries_examined"),
        "accumulated_window_pair_fraction" => ("all_pass_accumulated_window_pairs", "all_pass_window_opportunities"),
        "source_gated_window_pair_fraction" => ("all_pass_source_gated_window_pairs", "all_pass_window_opportunities"),
        "final_primary_finite_fraction" => ("final_primary_finite", "final_unmasked"),
        "final_zero_finite_nonzero_contributor_fraction" => ("final_zero_finite_nonzero_count", "final_unmasked"),
        "final_uq_u_numeric_availability" => ("final_uq_u_finite_nonnegative_count", "final_uq_evaluated_unmasked_nodes"),
        "final_uq_v_numeric_availability" => ("final_uq_v_finite_nonnegative_count", "final_uq_evaluated_unmasked_nodes")))
end
function _quality_ensemble_section(classification, counts, kinds)
    Dict{String,Any}(
        "scope" => "recorded_ensemble_pools_not_expected_workflow_population",
        "classification" => classification,
        "unsupported_entries" => Dict(k => kinds[k] for k in ("stereo", "ptv", "tracking")),
        "aggregation_basis" => "all_pass_window_pair_observations_and_final_pass_node_observations",
        "coordinate_basis" => "planar_processing_pixels_x_columns_y_rows",
        "residual_unit" => "px", "uncertainty_unit" => "px",
        "binding" => "raw_measurement_fields_and_geometry_checked_at_report_generation",
        "verification_time" => "report_generation",
        "source_support_convention" => "original_sampled_4x4_stencil_union_not_nonlocal_bspline_support",
        "primary_support_basis" => "final_pass_primary_peak_before_predictor_addition_and_validation",
        "uncertainty_basis" => "final_pass_pooled_statistics_before_validation_and_availability_cleanup",
        "counts" => counts, "fractions" => _quality_ensemble_fractions(classification, counts))
end
function _quality_check_ensemble_format(source)
    _check_result_file(source)
    jldopen(source.path, "r") do file
        _check_measurement_history_format(file)
        allowed = Set(source.entry_keys)
        for (group, marker, version) in (("execution_diagnostics", "execution_diagnostics_format_version", 1),
                ("stereo_execution_diagnostics", "stereo_execution_diagnostics_format_version", 1),
                ("ensemble_execution_diagnostics", "ensemble_execution_diagnostics_format_version", 1),
                ("measurement_history", "measurement_history_format_version", 1))
            if haskey(file, marker)
                file[marker] === version || _quality_error("unsupported $group version")
            else
                haskey(file, group) && _quality_error("$group metadata lacks version")
            end
            if haskey(file, group)
                file[group] isa Group || _quality_error("$group companions must form a native group")
                all(key -> key in allowed, keys(file[group])) ||
                    _quality_error("$group companion has no native result entry")
            end
        end
    end
    _check_result_file(source)
end
function _quality_ensemble_add_packet!(counts, data)
    _quality_add!(counts, "pair_observations", data["pair_count"])
    for p in data["passes"]
        _quality_add!(counts, "passes")
        _quality_add!(counts, "executed_pooling_sweeps", p["executed_pooling_sweeps"])
        _quality_add!(counts, "ignored_requested_iterations", p["requested_iterations"])
        p["requested_tolerance"] > 0 && _quality_add!(counts, "passes_with_positive_ignored_tolerance")
        _quality_add!(counts, "tolerance_checks", p["tolerance_checks"])
        _quality_add!(counts, "all_pass_pair_observations", p["pair_count"])
        for k in _QUALITY_ENSEMBLE_CONTRIBUTIONS
            _quality_add!(counts, "all_pass_$k", p["contributions"][k])
        end
        for k in ("no_predictor_pairs", "evaluated_pairs", "disabled_nonfinite_source_pairs")
            _quality_add!(counts, "all_pass_$k", p["source_support"][k])
        end
    end
    p = last(data["passes"]); nodes = p["contributor_nodes"]
    _quality_add!(counts, "final_masked", nodes["masked_count"])
    _quality_add!(counts, "final_unmasked", nodes["eligible_count"])
    _quality_add!(counts, "final_nodes", _quality_sum(nodes["masked_count"], nodes["eligible_count"]))
    for k in ("zero_finite_nonzero_count", "some_finite_nonzero_count", "all_finite_nonzero_count", "nodes_with_nonfinite_plane")
        _quality_add!(counts, "final_$k", nodes[k])
    end
    for (k, source) in (("final_primary_finite", "finite_count"), ("final_primary_nonfinite", "nonfinite_count"))
        _quality_add!(counts, k, p["residual"][source])
    end
    uq = p["uncertainty"]
    uq["requested"] && _quality_add!(counts, "final_uq_requested_entries")
    if uq["evaluated"]
        _quality_add!(counts, "final_uq_evaluated_entries")
        _quality_add!(counts, "final_uq_evaluated_unmasked_nodes", nodes["eligible_count"])
        for k in ("admitted_window_pair_updates", "source_gated_window_pairs")
            _quality_add!(counts, "final_uq_$k", uq[k])
        end
        for component in ("u", "v"), k in ("finite_nonnegative_count", "finite_negative_count", "nonfinite_count")
            _quality_add!(counts, "final_uq_$(component)_$k", uq[component][k])
        end
    end
end
function _quality_ensemble_observe!(classification, counts, index, i, result)
    ensemble = load_ensemble_execution_diagnostics(index, i)
    planar = load_execution_diagnostics(index, i)
    stereo = load_stereo_execution_diagnostics(index, i)
    history = load_measurement_history(index, i)
    if result isa PIVResult
        _quality_add!(classification, "planar_entries_examined")
        stereo === nothing || _quality_error("stereo diagnostics attached to planar result")
        if ensemble !== nothing
            planar === history === nothing || _quality_error("ensemble and pair-iteration/history companions cannot describe the same entry")
            data = execution_diagnostics_data(ensemble; result)
            _quality_add!(classification, "recorded_ensemble_entries")
            _quality_ensemble_add_packet!(counts, data)
        elseif planar !== nothing
            _quality_add!(classification, "recorded_planar_iteration_entries")
        else
            _quality_add!(classification, "entries_without_execution_metadata")
        end
    elseif result isa StereoPIVResult
        ensemble === planar === history === nothing || _quality_error("planar/ensemble companions attached to stereo result")
    else
        ensemble === planar === stereo === history === nothing || _quality_error("companions attached to unsupported result kind")
    end
    (; planar, stereo, history)
end
function _quality_validate_ensemble(section, kinds, groups)
    section isa AbstractDict || _quality_error("malformed ensemble report section")
    classification = get(section, "classification", nothing)
    counts = get(section, "counts", nothing)
    _experiment_keys(classification, collect(_QUALITY_ENSEMBLE_CLASSIFICATION), "ensemble classification")
    _experiment_keys(counts, _quality_ensemble_counter_names(), "ensemble report counts")
    all(v -> v isa Int && v >= 0, values(classification)) &&
        all(v -> v isa Int && v >= 0, values(counts)) || _quality_error("ensemble counts must be nonnegative integers")
    c = counts; n = classification["recorded_ensemble_entries"]
    classification["planar_entries_examined"] == kinds["planar"] == _quality_sum((classification[k] for k in
        ("recorded_ensemble_entries", "recorded_planar_iteration_entries", "entries_without_execution_metadata"))...) ||
        _quality_error("ensemble classification partition disagrees")
    c["passes"] >= n && c["pair_observations"] >= n && c["all_pass_pair_observations"] >= c["pair_observations"] &&
        c["all_pass_pair_observations"] >= c["passes"] && c["executed_pooling_sweeps"] == c["passes"] &&
        c["ignored_requested_iterations"] >= c["passes"] && c["passes_with_positive_ignored_tolerance"] <= c["passes"] &&
        c["tolerance_checks"] == 0 || _quality_error("ensemble sweep/request observations disagree")
    c["all_pass_pair_observations"] >= _quality_sum(c["pair_observations"], c["passes"]-n) &&
        (c["passes"] != n || c["all_pass_pair_observations"] == c["pair_observations"]) ||
        _quality_error("ensemble pair observations disagree with pass coverage")
    c["all_pass_window_opportunities"] == _quality_sum((c["all_pass_$k"] for k in
        ("masked_window_pairs", "source_gated_window_pairs", "accumulated_window_pairs"))...) &&
        c["all_pass_accumulated_window_pairs"] == _quality_sum((c["all_pass_$k"] for k in
        ("finite_zero_planes", "finite_flat_nonzero_planes", "finite_nonflat_planes", "nonfinite_planes"))...) &&
        c["all_pass_pair_observations"] == _quality_sum((c["all_pass_$k"] for k in
        ("no_predictor_pairs", "evaluated_pairs", "disabled_nonfinite_source_pairs"))...) ||
        _quality_error("ensemble all-pass contribution partition disagrees")
    c["final_nodes"] == _quality_sum(c["final_masked"], c["final_unmasked"]) &&
        c["final_unmasked"] == _quality_sum((c["final_$k"] for k in
        ("zero_finite_nonzero_count", "some_finite_nonzero_count", "all_finite_nonzero_count"))...) &&
        c["final_unmasked"] == _quality_sum(c["final_primary_finite"], c["final_primary_nonfinite"]) &&
        c["final_nodes_with_nonfinite_plane"] <= c["final_unmasked"] &&
        n <= c["final_nodes"] || _quality_error("ensemble final node partition disagrees")
    c["all_pass_window_opportunities"] >= c["all_pass_pair_observations"] &&
        c["all_pass_masked_window_pairs"] >= c["final_masked"] &&
        _quality_sum(c["all_pass_accumulated_window_pairs"], c["all_pass_source_gated_window_pairs"]) >= c["final_unmasked"] &&
        c["all_pass_nonfinite_planes"] >= c["final_nodes_with_nonfinite_plane"] &&
        _quality_sum(c["all_pass_finite_flat_nonzero_planes"], c["all_pass_finite_nonflat_planes"]) >=
            _quality_sum(c["final_some_finite_nonzero_count"], c["final_all_finite_nonzero_count"]) ||
        _quality_error("ensemble final observations lack all-pass contribution support")
    c["final_uq_evaluated_entries"] <= c["final_uq_requested_entries"] <= n &&
        c["final_uq_evaluated_entries"] <= c["final_uq_evaluated_unmasked_nodes"] <= c["final_unmasked"] &&
        c["final_uq_admitted_window_pair_updates"] <= c["all_pass_accumulated_window_pairs"] &&
        c["final_uq_source_gated_window_pairs"] <= c["all_pass_source_gated_window_pairs"] ||
        _quality_error("ensemble UQ support disagrees")
    for component in ("u", "v")
        _quality_sum((c["final_uq_$(component)_$k"] for k in
            ("finite_nonnegative_count", "finite_negative_count", "nonfinite_count"))...) == c["final_uq_evaluated_unmasked_nodes"] ||
            _quality_error("ensemble UQ component partition disagrees")
    end
    if n == 0
        all(iszero, values(c)) || _quality_error("ensemble observations without packets")
    end
    if c["final_uq_evaluated_entries"] == 0
        all(k -> c[k] == 0, ("final_uq_evaluated_unmasked_nodes", "final_uq_admitted_window_pair_updates",
            "final_uq_source_gated_window_pairs")) || _quality_error("UQ updates without evaluated pools")
    end
    if haskey(groups, "planar")
        stored = groups["planar"]["counts"]
        for (captured, field) in (("final_nodes", "nodes"), ("final_masked", "masked"), ("final_unmasked", "unmasked"))
            c[captured] <= stored[field] || _quality_error("ensemble support exceeds stored planar grids")
            n == kinds["planar"] && c[captured] != stored[field] &&
                _quality_error("fully covered ensemble grids disagree with stored fields")
        end
    end
    expected = _quality_ensemble_section(classification, c, kinds)
    _experiment_keys(section, collect(keys(expected)), "ensemble report section")
    # Exact comparison alone does not distinguish Bool from integer 0/1.
    fractions = section["fractions"]
    fractions isa AbstractDict && all(v -> v isa AbstractDict && get(v, "available", nothing) isa Bool &&
        get(v, "numerator", nothing) isa Int && get(v, "denominator", nothing) isa Int &&
        (!haskey(v, "value") || v["value"] isa AbstractFloat), values(fractions)) ||
        _quality_error("invalid ensemble fractions")
    unsupported = section["unsupported_entries"]
    unsupported isa AbstractDict && all(v -> v isa Int && v >= 0, values(unsupported)) ||
        _quality_error("invalid unsupported ensemble kinds")
    isequal(section, expected) || _quality_error("ensemble scope/binding/fractions disagree")
end
function _quality_show_ensemble(io, section)
    k = section["classification"]; c = section["counts"]
    print(io, "\nRecorded ensemble execution: ", k["recorded_ensemble_entries"], " / ", k["planar_entries_examined"],
        " planar entries; ", k["recorded_planar_iteration_entries"], " ordinary iteration packets; ",
        k["entries_without_execution_metadata"], " entries without execution metadata")
    print(io, "\n  ", c["passes"], " pooled passes/sweeps; ", c["ignored_requested_iterations"],
        " requested iterations ignored; no tolerance checks")
    print(io, "\n  All-pass window-pair observations: ", c["all_pass_accumulated_window_pairs"], " accumulated, ",
        c["all_pass_source_gated_window_pairs"], " source-gated, ", c["all_pass_masked_window_pairs"], " masked")
    print(io, "\n  Final primary support: ", c["final_primary_finite"], " finite, ", c["final_primary_nonfinite"],
        " nonfinite, ", c["final_masked"], " masked; residual/UQ basis is processing pixels")
    print(io, "\n  Raw measurement fields/geometry were checked when generated; loading the report performs no fresh result verification.")
    print(io, "\n  Contribution/update counts are not independent sample sizes. Residual amplitudes, stationarity and uncertainty coverage are unavailable.")
end
