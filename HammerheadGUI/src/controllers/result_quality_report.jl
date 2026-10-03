# Whole-file reports deliberately use the raw native index, never the physical
# display wrapper or a current-frame packet as an ensemble/run association.
function _report_native_index(ex::ResultExplorer)
    ex.results isa _LazyDisplayResults && ex.results.source isa Hammerhead.ResultFile ||
        throw(ArgumentError("whole-file quality reports require a lazy native ResultFile explorer; eager/bare inputs, timed artifacts and checkpoints are unsupported"))
    Hammerhead._quality_whole_file_index(ex.results.source)
end

"""
    explorer_quality_report(ex::ResultExplorer; include_measurement_history=false,
        include_execution_diagnostics=false, include_ensemble_execution_diagnostics=false)

Report the whole completed native file behind a lazy explorer, consuming one raw
payload at a time. Validate the complete sorted native entry mapping and capture
a detached index before scanning; omitted, duplicated or reordered keys are
refused. Displayed physical values,
selected frame and enabled inspection mode do not alter the request. Eager/bare,
timed and checkpoint explorers are refused. The core verifies file identity
before/after scanning; this is a synchronous operation that may pause rendering.
The report is explicitly unassociated with an experiment. Defaults preserve v1;
history/ordinary execution opt into v2/v3; ensemble execution explicitly opts into
v4. Absent metadata cannot identify an ensemble, and current flags cannot supply
missing history. Verification describes report generation, not ongoing validity.
"""
function explorer_quality_report(ex::ResultExplorer;include_measurement_history::Bool=false,
        include_execution_diagnostics::Bool=false,include_ensemble_execution_diagnostics::Bool=false)
    index=_report_native_index(ex)
    quality_report(index;include_measurement_history,include_execution_diagnostics,include_ensemble_execution_diagnostics)
end

"""
    save_explorer_quality_report(path, ex::ResultExplorer; kwargs...) -> RunQualityReport

Generate a whole-file report from the raw native index, then save validated TOML.
Options match [`explorer_quality_report`](@ref). The source and every known core
dependency are protected against output aliases. Failed scans/validation occur
before destination opening; filesystem failures can leave partial output. No
checkpoint/resumability or atomic-publication claim is made.
"""
function save_explorer_quality_report(path::AbstractString,ex::ResultExplorer;kwargs...)
    index=_report_native_index(ex)
    report=quality_report(index;kwargs...)
    save_quality_report(path,report;protected_paths=[index.path])
    report
end

mutable struct _ResultQualityController
    index::Hammerhead.ResultFile
    history::Observable{Bool}
    execution::Observable{Bool}
    ensemble::Observable{Bool}
    running::Observable{Bool}
    report::Observable{Union{Nothing,RunQualityReport}}
    status::Observable{String}
end
_ResultQualityController(ex)=_ResultQualityController(_report_native_index(ex),Observable(false),Observable(false),Observable(false),
    Observable(false),Observable{Union{Nothing,RunQualityReport}}(nothing),Observable("No report generated."))

function _generate_result_quality!(controller::_ResultQualityController;path_picker=nothing)
    controller.running[] && throw(ArgumentError("a quality report request is already active"))
    # Snapshot choices before observers or a file picker can mutate next choices.
    options=(include_measurement_history=controller.history[],include_execution_diagnostics=controller.execution[],
        include_ensemble_execution_diagnostics=controller.ensemble[])
    previous=controller.report[]
    try
        # The caller's ResultFile has a mutable key vector. Check completeness
        # against the native root and detach before observers/dialog callbacks.
        source=Hammerhead._quality_whole_file_index(controller.index)
        controller.running[]=true
        controller.status[]="Generating captured whole-file request; this scan may pause rendering."
        destination=path_picker===nothing ? nothing : path_picker()
        if path_picker!==nothing && (destination===nothing || isempty(destination))
            controller.status[]="Save cancelled; previous report retained."
            return nothing
        end
        report=quality_report(source;options...)
        destination===nothing || save_quality_report(destination,report;protected_paths=[source.path])
        controller.report[]=report
        controller.status[]=destination===nothing ? "Report generated; verification is at generation time." : "Report saved; verification is at generation time."
        report
    catch err
        controller.report.val=previous
        try notify(controller.report) catch end
        try controller.status[]="Report failed; previous report retained: $(sprint(showerror,err))" catch end
        rethrow()
    finally
        try controller.running[]=false catch end
    end
end

function _result_quality_text(controller::_ResultQualityController)
    report=controller.report[]
    report===nothing && return "Status: $(controller.status[])\nWhole native file: $(controller.index.path)\nNo experiment association. Choose optional recorded observations explicitly.\nUnknown/missing execution metadata does not identify an ensemble."
    data=quality_report_data(report);p=data["provenance"]
    "Status: $(controller.status[])\nReported whole file: $(p["source_path"])\nReported file SHA256: $(p["source_sha256"])\nExperiment association: $(p["association"])\nFormat: $(data["quality_report_format_version"])\n"*sprint(show,MIME"text/plain"(),report)
end
