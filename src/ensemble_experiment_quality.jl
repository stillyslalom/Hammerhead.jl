# Associated ensemble reports use their own provenance/count contract. Formats
# 1--4 continue through their original validator without changing their fields.
const QUALITY_ENSEMBLE_EXPERIMENT_REPORT_FORMAT_VERSION = 5
const _QUALITY_ENSEMBLE_RUN_COUNTS = ("input_pairs", "scheduled_passes", "total_contributions",
    "completed_contributions", "completed_pools", "published_results")

function _quality_validate_ensemble_experiment(data)
    pooled=haskey(data,"ensemble_execution_diagnostics")
    _experiment_keys(data,["quality_report_format_version","generated_at_unix_s","generator",
        "provenance","protected_locators","groups","unavailable","entry_kinds",
        (pooled ? ["ensemble_execution_diagnostics"] : String[])...],"associated ensemble report")
    data["quality_report_format_version"] === 5 || _quality_error("unsupported associated ensemble report version")
    p=data["provenance"]
    _experiment_keys(p,["association","workflow","verification_time","source_path","source_sha256",
        "source_index_entries","source_selection","recipe_id","input_id","run_id","run_environment_id",
        _QUALITY_ENSEMBLE_RUN_COUNTS...,"record_diagnostics","input_bytes_checked",
        "raw_measurement_fields_checked","recipe_grid_mask_scale_checked","requested_companions_checked"],
        "ensemble report provenance")
    p["association"]=="recorded_ensemble_output_verified" && p["workflow"]=="planar_ensemble" &&
        p["verification_time"]=="report_generation" && p["source_selection"]=="whole_file" ||
        _quality_error("unsupported ensemble report association or verification scope")
    all(k->_experiment_hash(p[k]),("recipe_id","input_id","run_environment_id")) ||
        _quality_error("invalid ensemble report identities")
    p["run_id"] isa String || _quality_error("invalid ensemble report run ID")
    try UUIDs.UUID(p["run_id"]) catch;_quality_error("invalid ensemble report run UUID") end
    all(k->p[k] isa Int && p[k]>=0,_QUALITY_ENSEMBLE_RUN_COUNTS) &&
        p["input_pairs"]>0 && p["scheduled_passes"]>0 || _quality_error("invalid ensemble report run counts")
    expected=_ensemble_mul(p["input_pairs"],p["scheduled_passes"])
    p["total_contributions"]==p["completed_contributions"]==expected &&
        p["completed_pools"]===1 &&
        p["published_results"]===p["source_index_entries"]===1 ||
        _quality_error("associated ensemble report requires one completed pooled result and all contributions")
    all(k->p[k] isa Bool,("record_diagnostics","input_bytes_checked")) &&
        all(k->p[k]===true,("raw_measurement_fields_checked","recipe_grid_mask_scale_checked","requested_companions_checked")) ||
        _quality_error("invalid ensemble report verification flags")
    kinds=data["entry_kinds"]
    _experiment_keys(kinds,collect(_QUALITY_HISTORY_KINDS),"ensemble report entry kinds")
    all(v->v isa Int,values(kinds)) || _quality_error("ensemble report entry kinds must be integer counts")
    isequal(kinds,Dict("planar"=>1,"stereo"=>0,"ptv"=>0,"tracking"=>0)) ||
        _quality_error("associated ensemble report requires exactly one planar result")
    data["groups"] isa AbstractDict && Set(keys(data["groups"]))==Set(["planar"]) ||
        _quality_error("associated ensemble report requires exactly one planar field group")
    data["unavailable"] isa AbstractDict &&
        all(v->v isa AbstractDict && get(v,"available",nothing)===false,values(data["unavailable"])) &&
        isequal(data["unavailable"],_quality_unavailable(4;include_history=false)) ||
        _quality_error("unsupported ensemble report unavailable diagnostics")

    # Delegate unchanged numerical populations and fractions to the existing
    # unassociated schema. This normalization is private; saved v5 retains the
    # distinct input/contribution/result counts and past verification scope.
    base=deepcopy(data)
    base["quality_report_format_version"]=pooled ? 4 : 1
    base["provenance"]=Dict{String,Any}("association"=>"unassociated",
        (k=>p[k] for k in ("source_path","source_sha256","source_index_entries","source_selection"))...)
    if !pooled
        delete!(base,"entry_kinds")
        base["unavailable"]=_quality_unavailable(1)
    end
    _quality_validate(base)
    if pooled
        s=data["ensemble_execution_diagnostics"];c=s["classification"];n=s["counts"]
        c["recorded_planar_iteration_entries"]===0 &&
            c["recorded_ensemble_entries"]==Int(p["record_diagnostics"]) ||
            _quality_error("ensemble report recording policy disagrees with companion coverage")
        if p["record_diagnostics"]
            n["pair_observations"]==p["input_pairs"] && n["passes"]==p["scheduled_passes"] &&
                n["all_pass_pair_observations"]==p["total_contributions"] ||
                _quality_error("ensemble report run and observed contribution counts disagree")
        end
    end
    data
end

function _quality_show_ensemble_experiment(io,p)
    print(io,"\nEnsemble association: verified recorded output at report generation",
        "\n  ",p["input_pairs"]," ordered input pairs; ",p["scheduled_passes"]," pooling passes; ",
        p["completed_contributions"]," / ",p["total_contributions"]," processed pair contributions",
        "\n  ",p["published_results"]," published pooled result (not one result per input pair)",
        "\n  Raw fields, recipe grid/mask/scale and requested companions checked; current input bytes ",
        p["input_bytes_checked"] ? "checked." : "not checked.",
        "\n  Verification is historical when loading this report; no stationarity or independent-sample claim.")
end

"""
    quality_report(record::EnsembleExperimentRecord, run::EnsembleExperimentRun;
        include_ensemble_execution_diagnostics=true, verify_inputs=false,
        output=run.output) -> RunQualityReport

Generate an associated version-5 report for one completed saved planar ensemble.
Snapshot the record/run and verify output SHA, exact native entry mapping,
recipe grid/mask/scale, raw measurement association, ordered source pairs and
requested companions before and after aggregation. `verify_inputs=true` also
checks current input bytes at both boundaries. No PIV is recomputed. `output`
explicitly locates a relocated local artifact without changing saved locators.

The provenance distinguishes ordered input pairs, scheduled passes, completed
pair contributions and the single published pooled result. It never coerces
these counts to sequence `completed_pairs`. Defaults include the unchanged
format-4 pooled counters; setting `include_ensemble_execution_diagnostics=false`
omits that section while retaining version 5 and all run integrity checks.
Unrecorded diagnostics remain an explicit execution-metadata coverage gap.
History and ordinary iteration sections are unsupported for this overload.

Failed/cancelled attempts are refused. Stored availability and contribution
counts establish neither uncertainty calibration nor stationarity, common
displacement or independent samples. Loading a saved report validates historical
metadata, not current files. Known local inputs, records, historical outputs and
the consumed artifact are protected by [`save_quality_report`](@ref); foreign
historical locators are never reinterpreted in the current working directory.
Only detached scalar metrics survive; verification/aggregation reads one raw
pooled field at a time, including any saved correlation planes.
"""
function quality_report(record::EnsembleExperimentRecord,run::EnsembleExperimentRun;
        include_ensemble_execution_diagnostics::Bool=true,verify_inputs::Bool=false,
        output::AbstractString=run.output,include_measurement_history::Bool=false,
        include_execution_diagnostics::Bool=false)
    include_measurement_history && _quality_error("associated ensemble measurement history is unsupported")
    include_execution_diagnostics && _quality_error("associated ensembles have pooled observations, not ordinary iteration diagnostics")
    snapshot=deepcopy(record);checked=deepcopy(run)
    checked.status===:completed && checked.output_sha256!==nothing ||
        _quality_error("ensemble quality association requires a completed published run")
    path=_artifact_local_path(output)
    verify_ensemble_experiment_run(snapshot,checked;verify_results=true,verify_inputs,output=path)
    source=_quality_whole_file_index(ResultFile(path))
    report=_quality_report(source,false,false,nothing,include_ensemble_execution_diagnostics)
    data=quality_report_data(report)
    data["provenance"]["source_sha256"]==checked.output_sha256 ||
        _quality_error("recorded ensemble output changed while building report")
    verify_ensemble_experiment_run(snapshot,checked;verify_results=true,verify_inputs,output=path)
    data["quality_report_format_version"]=QUALITY_ENSEMBLE_EXPERIMENT_REPORT_FORMAT_VERSION
    data["entry_kinds"]=Dict("planar"=>1,"stereo"=>0,"ptv"=>0,"tracking"=>0)
    data["unavailable"]=_quality_unavailable(4;include_history=false)
    merge!(data["provenance"],Dict{String,Any}(
        "association"=>"recorded_ensemble_output_verified","workflow"=>"planar_ensemble",
        "verification_time"=>"report_generation","recipe_id"=>checked.recipe_id,"input_id"=>checked.input_id,
        "run_id"=>checked.run_id,"run_environment_id"=>_experiment_digest(_experiment_environment_signature(checked.environment)),
        (k=>getfield(checked,Symbol(k)) for k in _QUALITY_ENSEMBLE_RUN_COUNTS)...,
        "record_diagnostics"=>checked.record_diagnostics,"input_bytes_checked"=>verify_inputs,
        "raw_measurement_fields_checked"=>true,"recipe_grid_mask_scale_checked"=>true,"requested_companions_checked"=>true))
    locators=String[path,checked.output]
    append!(locators,[f["path"] for f in snapshot.input_files])
    append!(locators,snapshot.record_paths)
    append!(locators,[saved.output for saved in snapshot.runs])
    append!(locators,data["protected_locators"])
    data["protected_locators"]=_artifact_local_protected_paths(locators)
    _quality_wrap(data)
end
