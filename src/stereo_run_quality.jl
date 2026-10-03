"""
    quality_report(record::StereoExperimentRecord, run::ExperimentRun;
        include_measurement_history=false, include_execution_diagnostics=false,
        verify_inputs=false, output=run.output) -> RunQualityReport

Summarize a completed saved stereo run with verified experiment association.
Snapshot the record/run, then verify the native output, frozen recipe geometry,
ordered sources, raw measurement fields and requested companions before and
after streaming the stored-field summary. No PIV or calibration fitting is run.
`verify_inputs=true` additionally checks current input bytes at both boundaries;
the default does not read input images. `output` explicitly locates a relocated
native artifact without changing its historical run locator.

The default preserves report format 1. `include_execution_diagnostics=true`
uses existing format 3 with separate camera pass/support counts and raw camera
binding; it does not infer reconstructed 3C residuals or pool their amplitudes.
Stereo measurement history is unsupported: `include_measurement_history=true`
is refused. Failed or partially published runs are also refused.

Saved provenance describes verification at report generation, not continuing
file validity, calibration accuracy, source authenticity or uncertainty coverage.
Known local input, record and run-output paths are protected when saving the
report. Foreign historical locators are not reinterpreted as local paths.
Metric storage is bounded; native indexes and protected locators grow with the
number of acquisitions and inputs. One native result (including its camera
fields and any saved correlation planes) is loaded at a time.
"""
function quality_report(record::StereoExperimentRecord, run::ExperimentRun;
        include_measurement_history::Bool=false,
        include_execution_diagnostics::Bool=false,
        verify_inputs::Bool=false, output::AbstractString=run.output)
    include_measurement_history &&
        _quality_error("stereo experiment measurement-history reports are unsupported")
    snapshot = deepcopy(record)
    _stereo_record_preflight(snapshot)
    validated = _stereo_run_validate(_experiment_run_data(deepcopy(run)), snapshot)
    validated.status === :completed && validated.output_sha256 !== nothing ||
        _quality_error("quality association requires a completed stereo run with output identity")
    path = _artifact_local_path(output)
    verify_stereo_experiment_run(snapshot, validated;
        verify_results=true, verify_inputs, output=path)
    # Stereo association is checked by its dedicated run verifier, separately
    # from the planar recipe/input fields of individual execution packets.
    source = ResultFile(path)
    report = _quality_report(source, false, include_execution_diagnostics, nothing)
    data = quality_report_data(report)
    data["provenance"]["source_sha256"] == validated.output_sha256 ||
        _quality_error("recorded stereo output changed while building report")
    verify_stereo_experiment_run(snapshot, validated;
        verify_results=true, verify_inputs, output=path)
    merge!(data["provenance"], Dict(
        "association" => "recorded_output_verified",
        "recipe_id" => validated.recipe_id, "input_id" => validated.input_id,
        "run_id" => validated.run_id, "completed_pairs" => validated.completed_pairs,
        "run_environment_id" => _experiment_digest(
            _experiment_environment_signature(validated.environment))))
    locators = String[path, validated.output]
    append!(locators, [file["path"] for file in snapshot.input_files])
    append!(locators, snapshot.record_paths)
    append!(locators, [saved.output for saved in snapshot.runs])
    append!(locators, data["protected_locators"])
    data["protected_locators"] = _artifact_local_protected_paths(locators)
    _quality_wrap(data)
end
