# Saved quality summaries describe stored fields, not absent measurement history.
const QUALITY_REPORT_FORMAT_VERSION = 1
const QUALITY_HISTORY_REPORT_FORMAT_VERSION = 2
const QUALITY_EXECUTION_REPORT_FORMAT_VERSION = 3
const _QUALITY_UNAVAILABLE = Dict(
    "rejection_events" => "not_persisted", "replacement_history" => "not_persisted",
    "alternative_peak_history" => "not_persisted", "uncertainty_measurement_association" => "not_persisted",
    "accuracy" => "not_evaluated", "uncertainty_coverage" => "not_evaluated",
    "peak_locking" => "not_evaluated", "recipe_sensitivity" => "not_compared")

const _QUALITY_HISTORY_KINDS = ("planar", "stereo", "ptv", "tracking")
const _QUALITY_REJECTION_BUCKETS = ("nonfinite_primary", "implicit_uod", "implicit_peak_ratio",
    "configured_builtin", "configured_custom", "unclassified")
function _quality_unavailable(version;include_history=version==2)
    unavailable=copy(_QUALITY_UNAVAILABLE)
    if include_history
        for name in ("rejection_events","replacement_history","alternative_peak_history")
            delete!(unavailable,name)
        end
        unavailable["earlier_pass_and_sweep_history"]="not_recorded"
        unavailable["uncertainty_measurement_association"]="applicability_not_established"
    end
    version in (3,4) && (unavailable["pooled_execution_residual_amplitudes"]="not_aggregated")
    if version==4
        unavailable["ensemble_independent_sample_size"]="not_established"
        unavailable["ensemble_common_displacement_assumption"]="not_checked"
        unavailable["ensemble_per_node_measurement_history"]="not_recorded"
    end
    Dict(k=>Dict("available"=>false,"reason_code"=>v) for (k,v) in unavailable)
end

const _QUALITY_EXECUTION_ROLES=("planar","cam1","cam2")
const _QUALITY_EXECUTION_STOPS=("single_sweep","iteration_budget","tolerance_condition_met")
function _quality_execution_counter_names()
    ["eligible_entries","recorded_entries","missing_entries","passes","requested_sweeps","executed_sweeps",
     "tolerance_checks",["stop_$name" for name in _QUALITY_EXECUTION_STOPS]...,
     "last_checks_present","last_checks_absent","last_check_finite","last_check_nan","last_check_infinite",
     "last_check_empty","last_check_included","last_check_finite_support","last_check_infinite_support","last_check_excluded",
     "final_primary_nodes","final_primary_masked","final_primary_unmasked","final_primary_finite","final_primary_nonfinite"]
end
_quality_execution_fractions(c)=_quality_make_fractions(c,Dict(
    "recorded_entry_fraction"=>("recorded_entries","eligible_entries"),
    "final_primary_finite_fraction"=>("final_primary_finite","final_primary_unmasked")))
function _quality_execution_group(role,counts)
    Dict{String,Any}("coordinate_basis"=>role=="planar" ? "planar_processing_pixels_x_columns_y_rows" : "dewarped_pixels_x_columns_y_rows",
        "residual_unit"=>"px","binding"=>role=="planar" ? "entry_key_only" : "raw_measurement_fields_checked_at_report_generation",
        "counts"=>counts,"fractions"=>_quality_execution_fractions(counts))
end
function _quality_execution_section(counters,kinds)
    for role in _QUALITY_EXECUTION_ROLES
        counts=counters[role]
        counts["eligible_entries"]=kinds[role=="planar" ? "planar" : "stereo"]
        counts["missing_entries"]=counts["eligible_entries"]-counts["recorded_entries"]
    end
    Dict{String,Any}("scope"=>"recorded_passes_and_final_pass_primary_support",
        "aggregation_basis"=>"entry_and_pass_counts","verification_time"=>"report_generation",
        "last_check_scope"=>"last_recorded_check_per_pass","primary_support_scope"=>"final_pass_before_validation",
        "unsupported_entries"=>Dict(k=>kinds[k] for k in ("ptv","tracking")),
        "groups"=>Dict(role=>_quality_execution_group(role,counters[role]) for role in _QUALITY_EXECUTION_ROLES))
end
function _quality_check_execution_format(source)
    _check_result_file(source)
    jldopen(source.path,"r") do file
        any(key -> haskey(file,key), ("ensemble_execution_diagnostics_format_version",
            "ensemble_execution_diagnostics")) &&
            _quality_error("ensemble pooled execution diagnostics are unsupported by execution-aware quality reports")
        for (group,marker,version) in (("execution_diagnostics","execution_diagnostics_format_version",EXECUTION_DIAGNOSTICS_FORMAT_VERSION),
                ("stereo_execution_diagnostics","stereo_execution_diagnostics_format_version",STEREO_EXECUTION_DIAGNOSTICS_FORMAT_VERSION))
            if haskey(file,marker)
                file[marker]===version || _quality_error("unsupported $group version")
            else
                haskey(file,group) && _quality_error("$group metadata lacks version")
            end
        end
    end
    _check_result_file(source)
    nothing
end
function _quality_execution_passes!(counts,data)
    _quality_add!(counts,"recorded_entries")
    for pass in data["passes"]
        _quality_add!(counts,"passes")
        _quality_add!(counts,"requested_sweeps",pass["requested_iterations"])
        _quality_add!(counts,"executed_sweeps",pass["executed_iterations"])
        _quality_add!(counts,"tolerance_checks",pass["checks"])
        _quality_add!(counts,"stop_"*pass["stop_reason"])
        check=pass["last_check"]
        if check===nothing
            _quality_add!(counts,"last_checks_absent")
        else
            _quality_add!(counts,"last_checks_present")
            _quality_add!(counts,"last_check_"*check["value_state"])
            check["included_count"]==0 && _quality_add!(counts,"last_check_empty")
            for (key,stored) in (("last_check_included","included_count"),("last_check_finite_support","finite_count"),
                    ("last_check_infinite_support","infinite_count"),("last_check_excluded","excluded_count"))
                _quality_add!(counts,key,check[stored])
            end
        end
    end
    residual=last(data["passes"])["residual"]
    finite,nonfinite,masked=residual["finite_count"],residual["nonfinite_count"],residual["masked_count"]
    _quality_add!(counts,"final_primary_nodes",_quality_sum(finite,nonfinite,masked))
    _quality_add!(counts,"final_primary_unmasked",_quality_sum(finite,nonfinite))
    _quality_add!(counts,"final_primary_finite",finite)
    _quality_add!(counts,"final_primary_nonfinite",nonfinite)
    _quality_add!(counts,"final_primary_masked",masked)
    nothing
end
function _quality_execution_update!(counters,index,i,result,expected_association; loaded=nothing)
    planar=loaded===nothing ? load_execution_diagnostics(index,i) : loaded.planar
    stereo=loaded===nothing ? load_stereo_execution_diagnostics(index,i) : loaded.stereo
    if result isa PIVResult
        stereo===nothing || _quality_error("stereo diagnostics attached to planar result")
        if planar!==nothing
            data=execution_diagnostics_data(planar)
            expected_association===nothing || (data["association"]==expected_association && data["pair_index"]===i) ||
                _quality_error("recorded execution recipe/input or pair association disagrees with the selected run")
            _quality_execution_passes!(counters["planar"],data)
        end
    elseif result isa StereoPIVResult
        planar===nothing || _quality_error("planar diagnostics attached to stereo result")
        if stereo!==nothing
            expected_association===nothing || _quality_error("stereo execution diagnostics have no supported experiment recipe/input association")
            data=execution_diagnostics_data(stereo;result)
            for (role,camera) in zip(("cam1","cam2"),data["cameras"])
                _quality_execution_passes!(counters[role],camera["diagnostics"])
            end
        end
    else
        planar===stereo===nothing || _quality_error("execution diagnostics attached to unsupported result kind")
    end
    nothing
end
function _quality_validate_execution(section,kinds,groups)
    _experiment_keys(section,["scope","aggregation_basis","verification_time","last_check_scope","primary_support_scope","unsupported_entries","groups"],"quality execution")
    section["scope"]=="recorded_passes_and_final_pass_primary_support" && section["aggregation_basis"]=="entry_and_pass_counts" &&
        section["verification_time"]=="report_generation" && section["last_check_scope"]=="last_recorded_check_per_pass" &&
        section["primary_support_scope"]=="final_pass_before_validation" || _quality_error("unsupported execution-report scope")
    unsupported=section["unsupported_entries"]
    _experiment_keys(unsupported,["ptv","tracking"],"unsupported execution kinds")
    all(k->unsupported[k] isa Int && unsupported[k]==kinds[k],("ptv","tracking")) || _quality_error("inconsistent unsupported execution counts")
    roles=section["groups"]
    _experiment_keys(roles,collect(_QUALITY_EXECUTION_ROLES),"execution camera groups")
    for role in _QUALITY_EXECUTION_ROLES
        group=roles[role]
        _experiment_keys(group,["coordinate_basis","residual_unit","binding","counts","fractions"],"execution camera group")
        c=group["counts"]
        _experiment_keys(c,_quality_execution_counter_names(),"execution counters")
        all(v->v isa Int && v>=0,values(c)) || _quality_error("execution counts must be nonnegative integers")
        kind=role=="planar" ? "planar" : "stereo"
        c["eligible_entries"]==kinds[kind] && c["eligible_entries"]==_quality_sum(c["recorded_entries"],c["missing_entries"]) || _quality_error("inconsistent execution coverage")
        c["passes"]>=c["recorded_entries"] && c["requested_sweeps"]>=c["executed_sweeps"]>=c["passes"] &&
            c["passes"]==_quality_sum((c["stop_$name"] for name in _QUALITY_EXECUTION_STOPS)...) &&
            c["passes"]==_quality_sum(c["last_checks_present"],c["last_checks_absent"]) &&
            c["last_checks_present"]<=c["tolerance_checks"]<=c["executed_sweeps"]-c["passes"] &&
            c["last_checks_present"]==_quality_sum(c["last_check_finite"],c["last_check_nan"],c["last_check_infinite"]) &&
            c["last_check_empty"]<=c["last_check_finite"] && c["stop_tolerance_condition_met"]<=c["last_check_finite"] &&
            c["last_check_included"]==_quality_sum(c["last_check_finite_support"],c["last_check_infinite_support"]) &&
            c["final_primary_nodes"]==_quality_sum(c["final_primary_masked"],c["final_primary_unmasked"]) &&
            c["final_primary_unmasked"]==_quality_sum(c["final_primary_finite"],c["final_primary_nonfinite"]) &&
            c["recorded_entries"]<=c["final_primary_nodes"] || _quality_error("inconsistent execution pass/support counts")
        c["executed_sweeps"]>=_quality_sum(c["passes"],c["passes"]-c["stop_single_sweep"]) &&
            (c["stop_single_sweep"]!=c["passes"] || c["executed_sweeps"]==c["passes"]) &&
            c["requested_sweeps"]-c["executed_sweeps"]>=c["stop_tolerance_condition_met"] &&
            (c["stop_tolerance_condition_met"]!=0 || c["requested_sweeps"]==c["executed_sweeps"]) &&
            c["last_checks_absent"]>=c["stop_single_sweep"] &&
            c["tolerance_checks"]<=c["executed_sweeps"]-c["passes"]-c["stop_iteration_budget"] ||
            _quality_error("execution budgets/checks disagree with stop outcomes")
        if c["recorded_entries"]==0
            all(k->k in ("eligible_entries","missing_entries") || c[k]==0,keys(c)) || _quality_error("execution observations without recorded entries")
        end
        c["last_checks_present"]==0 && any(k->c[k]!=0,("last_check_included","last_check_excluded")) && _quality_error("support without tolerance checks")
        c["last_check_included"]>=c["last_checks_present"]-c["last_check_empty"] || _quality_error("nonempty tolerance observations require support")
        c["last_check_empty"]==c["last_checks_present"] && c["last_check_included"]!=0 && _quality_error("nonempty support for only empty checks")
        if kind=="stereo" && haskey(groups,kind)
            nodes=groups[kind]["counts"]["nodes"]
            c["final_primary_nodes"]<=nodes && (c["recorded_entries"]!=c["eligible_entries"] || c["final_primary_nodes"]==nodes) || _quality_error("verified camera support disagrees with stereo grid coverage")
        end
        expected=_quality_execution_group(role,c)
        for key in ("coordinate_basis","residual_unit","binding")
            group[key]==expected[key] || _quality_error("unsupported execution coordinate/binding basis")
        end
        fractions=group["fractions"]
        fractions isa AbstractDict && all(v->v isa AbstractDict && get(v,"available",nothing) isa Bool &&
            get(v,"numerator",nothing) isa Int && get(v,"denominator",nothing) isa Int &&
            (!haskey(v,"value") || v["value"] isa AbstractFloat),values(fractions)) || _quality_error("invalid execution fractions")
        isequal(fractions,expected["fractions"]) || _quality_error("execution fractions disagree with covered denominators")
    end
    first,second=roles["cam1"]["counts"],roles["cam2"]["counts"]
    all(k->first[k]==second[k],("eligible_entries","recorded_entries","missing_entries","passes","requested_sweeps","final_primary_nodes")) ||
        _quality_error("camera association/schedule/grid counts disagree")
    nothing
end
function _quality_history_counter_names()
    ["planar_entries","recorded_entries","missing_entries","unsupported_stereo_entries","unsupported_ptv_entries","unsupported_tracking_entries",
     "nodes","masked","unmasked","first_rejected","pre_substitution_flagged","accepted_alternative","fill_attempted","fill_assigned","primary_restored",
     ["origin_$name" for name in ("unavailable","primary","alternative","fill","custom_unclassified")]...,
     ["rejection_$name" for name in _QUALITY_REJECTION_BUCKETS]...]
end
function _quality_history_metric_specs()
    specs=Dict("recorded_entry_fraction"=>("recorded_entries","planar_entries"),"masked_fraction"=>("masked","nodes"))
    for name in ("first_rejected","pre_substitution_flagged","accepted_alternative","fill_attempted","fill_assigned","primary_restored",
                 ("origin_$name" for name in ("unavailable","primary","alternative","fill","custom_unclassified"))...,
                 ("rejection_$name" for name in _QUALITY_REJECTION_BUCKETS)...)
        specs["$(name)_fraction"]=(name,"unmasked")
    end
    specs
end
function _quality_make_fractions(counts,specs)
    Dict{String,Any}(name=>begin
        numerator,denominator=counts[keys[1]],counts[keys[2]]
        metric=Dict{String,Any}("numerator"=>numerator,"denominator"=>denominator,
            "denominator_count"=>keys[2],"unit"=>"1","available"=>denominator>0)
        denominator>0 && (metric["value"]=numerator/denominator)
        metric
    end for (name,keys) in specs)
end
function _quality_rejection_bucket(stage)
    name,kind=stage["name"],stage["kind"]
    if kind=="built_in" && name in ("nonfinite_primary","implicit_uod","implicit_peak_ratio")
        return name
    end
    # Recognize only names actually generated by companion version 1; arbitrary
    # stored labels have no inferred configured-stage semantics.
    matchname=match(r"^validation\[([1-9][0-9]*)\]:([^\s:]+)$",name)
    matchname===nothing && return "unclassified"
    kind=="custom" && return "configured_custom"
    matchname.captures[2] in ("PeakRatioValidator","CorrelationMomentValidator","VelocityMagnitudeValidator","UniversalOutlierValidator") && return "configured_builtin"
    "unclassified"
end
function _quality_history_update!(counts,h,result)
    verify_measurement_history(h,result)
    d=_history_checked_data(h)
    buckets=[_quality_rejection_bucket(stage) for stage in d["rejection_stages"]]
    _quality_add!(counts,"recorded_entries")
    _quality_add!(counts,"nodes",length(d["mask"]))
    for i in eachindex(d["mask"])
        if d["mask"][i]
            _quality_add!(counts,"masked")
            continue
        end
        _quality_add!(counts,"unmasked")
        stage=d["first_rejection_stage"][i]
        if stage>0
            _quality_add!(counts,"first_rejected")
            _quality_add!(counts,"rejection_"*buckets[stage])
        end
        for (key,array) in (("pre_substitution_flagged","pre_substitution_outliers"),("fill_attempted","fill_attempted"),
                            ("fill_assigned","fill_assigned"),("primary_restored","primary_restored"))
            d[array][i] && _quality_add!(counts,key)
        end
        d["accepted_peak_rank"][i]>0 && _quality_add!(counts,"accepted_alternative")
        _quality_add!(counts,"origin_"*d["origin_codes"][Int(d["final_origin"][i])+1])
    end
    nothing
end
function _quality_validate_history(section,kinds,groups)
    _experiment_keys(section,["scope","binding","counts","fractions"],"quality history")
    section["scope"]=="final_pass_final_executed_sweep" && section["binding"]=="raw_measurement_digest_verified" || _quality_error("unsupported quality history scope/binding")
    c=section["counts"]
    _experiment_keys(c,_quality_history_counter_names(),"quality history counters")
    all(v->v isa Int && v>=0,values(c)) || _quality_error("history counts must be nonnegative integers")
    c["planar_entries"]==kinds["planar"] && c["planar_entries"]==_quality_sum(c["recorded_entries"],c["missing_entries"]) || _quality_error("inconsistent history coverage")
    all(kind->c["unsupported_$(kind)_entries"]==kinds[kind],("stereo","ptv","tracking")) || _quality_error("inconsistent unsupported history counts")
    c["nodes"]==_quality_sum(c["masked"],c["unmasked"]) || _quality_error("inconsistent history mask counts")
    c["unmasked"]==_quality_sum((c["origin_$name"] for name in ("unavailable","primary","alternative","fill","custom_unclassified"))...) || _quality_error("inconsistent final origins")
    c["first_rejected"]==_quality_sum((c["rejection_$name"] for name in _QUALITY_REJECTION_BUCKETS)...) || _quality_error("inconsistent rejection categories")
    all(k->c[k]<=c["unmasked"],("first_rejected","pre_substitution_flagged","accepted_alternative","fill_attempted","fill_assigned","primary_restored")) || _quality_error("history events exceed covered nodes")
    c["accepted_alternative"]==c["origin_alternative"] && c["fill_assigned"]<=c["fill_attempted"]<=c["pre_substitution_flagged"] &&
        c["primary_restored"]<=c["fill_attempted"] && c["origin_fill"]<=c["fill_assigned"] &&
        _quality_sum(c["accepted_alternative"],c["fill_attempted"])<=c["pre_substitution_flagged"] || _quality_error("inconsistent history event counts")
    c["recorded_entries"]==0 && c["nodes"]!=0 && _quality_error("history nodes without recorded entries")
    c["recorded_entries"]<=c["nodes"] || _quality_error("recorded history entries require nonempty grids")
    if haskey(groups,"planar")
        p=groups["planar"]["counts"]
        c["nodes"]<=p["nodes"] && c["masked"]<=p["masked"] && c["unmasked"]<=p["unmasked"] || _quality_error("history coverage exceeds planar output")
        c["recorded_entries"]==c["planar_entries"] && (c["nodes"]!=p["nodes"] || c["masked"]!=p["masked"]) && _quality_error("complete history grid counts disagree")
    end
    fractions=section["fractions"]
    fractions isa AbstractDict && all(v->v isa AbstractDict && get(v,"available",nothing) isa Bool &&
        get(v,"numerator",nothing) isa Int && get(v,"denominator",nothing) isa Int &&
        (!haskey(v,"value") || v["value"] isa AbstractFloat),values(fractions)) || _quality_error("invalid history fractions")
    isequal(fractions,_quality_make_fractions(c,_quality_history_metric_specs())) || _quality_error("history fractions disagree with covered denominators")
    nothing
end

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
    _quality_make_fractions(counts,_quality_metric_specs(kind))
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
    version=get(data,"quality_report_format_version",nothing)
    version === 5 && return _quality_validate_ensemble_experiment(data)
    version isa Int && version in (1,2,3,4) || _quality_error("unsupported quality report format version")
    include_history=version==2 || version in (3,4) && haskey(data,"measurement_history")
    include_execution=version==3 || version==4 && haskey(data,"execution_diagnostics")
    extra=version==1 ? String[] : ["entry_kinds";include_history ? ["measurement_history"] : String[];
        include_execution ? ["execution_diagnostics"] : String[];
        version==4 ? ["ensemble_execution_diagnostics"] : String[]]
    _experiment_keys(data, ["quality_report_format_version", "generated_at_unix_s", "generator",
        "provenance", "protected_locators", "groups", "unavailable",extra...], "quality report")
    data["generated_at_unix_s"] isa Real && !(data["generated_at_unix_s"] isa Bool) &&
        isfinite(data["generated_at_unix_s"]) && data["generated_at_unix_s"] >= 0 || _quality_error("invalid report timestamp")
    _experiment_keys(data["generator"], ["julia_version", "hammerhead_version", "core_source_sha256", "value_basis", "weighting"], "quality generator")
    all(v -> v isa String, values(data["generator"])) && data["generator"]["value_basis"] == _quality_value_basis(include_history,include_execution,version==4) &&
        data["generator"]["weighting"] == (version in (3,4) ? "field_nodes_and_execution_observations" : "node_weighted") &&
        _experiment_hash(data["generator"]["core_source_sha256"]) || _quality_error("unsupported quality metric basis")
    locators = data["protected_locators"]
    locators isa AbstractVector && all(p -> p isa String && isabspath(p), locators) || _quality_error("invalid protected report locators")
    provenance = data["provenance"]
    provenance isa AbstractDict && haskey(provenance, "association") || _quality_error("invalid quality provenance")
    association = provenance["association"]
    common = haskey(provenance, "source_path") ? ["source_path", "source_sha256", "source_index_entries", "source_selection"] : String[]
    associated = association == "recorded_output_verified"
    version==4 && associated && _quality_error("ensemble reports have no supported experiment recipe association")
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
    if version in (2,3,4)
        kinds=data["entry_kinds"]
        _experiment_keys(kinds,collect(_QUALITY_HISTORY_KINDS),"quality entry kinds")
        all(v->v isa Int && v>=0,values(kinds)) || _quality_error("invalid entry kind counts")
        all(kind->get(groups,kind,Dict("counts"=>Dict("entries"=>0)))["counts"]["entries"]==kinds[kind],("planar","stereo")) || _quality_error("quality kind/group counts disagree")
        include_history && _quality_validate_history(data["measurement_history"],kinds,groups)
        include_execution && _quality_validate_execution(data["execution_diagnostics"],kinds,groups)
        version==4 && _quality_validate_ensemble(data["ensemble_execution_diagnostics"],kinds,groups)
        if version==4
            classification=data["ensemble_execution_diagnostics"]["classification"]
            include_execution && data["execution_diagnostics"]["groups"]["planar"]["counts"]["recorded_entries"] != classification["recorded_planar_iteration_entries"] &&
                _quality_error("ordinary and ensemble execution classifications disagree")
            include_history && data["measurement_history"]["counts"]["recorded_entries"] > kinds["planar"]-classification["recorded_ensemble_entries"] &&
                _quality_error("history and ensemble packets cannot cover the same entries")
        end
        total_entries=_quality_sum(values(kinds)...)
        !isempty(common) && provenance["source_selection"]=="whole_file" || _quality_error("companion report requires a whole native result source")
    end
    associated && total_entries != provenance["completed_pairs"] && _quality_error("associated result count mismatch")
    !isempty(common) && provenance["source_selection"] == "whole_file" && total_entries != provenance["source_index_entries"] && _quality_error("whole-file result count mismatch")
    expected_unavailable = _quality_unavailable(version;include_history)
    data["unavailable"] isa AbstractDict && all(v -> v isa AbstractDict && get(v, "available", nothing) === false, values(data["unavailable"])) || _quality_error("invalid unavailable diagnostic availability")
    isequal(data["unavailable"], expected_unavailable) || _quality_error("unsupported or invented quality diagnostics")
    data
end
function _quality_wrap(data)
    _quality_validate(data)
    snapshot = deepcopy(Dict{String,Any}(data))
    RunQualityReport(snapshot, Tuple(snapshot["protected_locators"]), _experiment_digest(snapshot))
end
function _quality_value_basis(history,execution,ensemble=false)
    ensemble && return _quality_value_basis(history,execution)*"_and_verified_recorded_ensemble_execution"
    execution ? (history ? "stored_arrays_recorded_execution_and_verified_final_sweep_history" : "stored_arrays_and_recorded_execution") :
        (history ? "stored_arrays_and_verified_final_sweep_history" : "stored_arrays")
end

"""
    quality_report(results; include_measurement_history=false,
                   include_execution_diagnostics=false,
                   include_ensemble_execution_diagnostics=false) -> RunQualityReport
    quality_report(record::ExperimentRecord, run::ExperimentRun;
                   include_measurement_history=false,
                   include_execution_diagnostics=false) -> RunQualityReport

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

By default, format version 1 describes stored fields only. Flagged finite output is NOT a replacement count. Substituted peaks may clear
flags; filling may retain flags. Rejection events/replacements/alternatives and
UQ association are explicitly unavailable; accuracy, coverage, peak locking and
recipe sensitivity are not evaluated.

With `include_measurement_history=true`, require a direct whole-file `ResultFile`
or the verified record/run overload and emit format version 2. Read each raw
result once, verify any recorded final-pass/final-sweep history, and count actual
events separately from final origin. Fractions use history-covered unmasked
nodes; missing planar packets are coverage gaps, not zero events. Explicit kind
counts identify unsupported stereo/PTV/tracking history; PTV/tracking entries
have no numerical quality group in v2. Malformed companions or history attached
to unsupported kinds are refused. Unknown rejection-stage labels count as
unclassified. Bare/converted iterators and array views have no supported checked
entry mapping for this opt-in. Earlier history and UQ applicability remain
unavailable. A present packet in the record/run overload must have matching
recipe/input IDs and absolute pair index; absent or stale association is refused,
whereas a missing packet only reduces coverage. Generic files remain unassociated
even when individual packets name recipe/input IDs.

With `include_execution_diagnostics=true`, require the same direct whole-file
native mapping and emit format version 3, optionally including the existing
history section. Generic aggregation reads each raw payload once. Recorded planar companions retain
entry-key linkage only; stereo companions additionally verify raw measurement
fields and independent geometry against the already loaded result. Associated
planar packets must match recipe/input IDs and absolute pair index. The planar
`ExperimentRecord` overload refuses stereo packets; the dedicated
`StereoExperimentRecord` overload verifies stereo association in additional
streaming passes before and after aggregation. Wrong-kind companions and invalid versions, including markers
in empty files, are refused. Missing metadata is a coverage gap, never inferred
from parameters. PTV/tracking entries have explicit unsupported counts.

Execution counts are entry/pass observations, separately for planar, camera 1
and camera 2: requested/executed sweeps, actual checks, stop reasons, each pass's
last-check state/support and final-pass primary support before validation.
Camera units remain dewarped pixels; no amplitudes are pooled across grids and
no reconstructed 3C residual is inferred. Tolerance outcomes and empty checks
do not establish measurement validity. Saved reports describe checks when
generated, not fresh result/calibration/input verification when loaded.

With `include_ensemble_execution_diagnostics=true`, require a direct whole-file
`ResultFile` and emit format version 4. This opt-in can combine with the two
existing flags; their coverage denominators remain unchanged. The original index
must exactly match the independently sorted native result keys; a detached key
snapshot is used before provenance or populations are recorded. Edited index
vectors (including omissions, duplicates and reordered keys) are refused.
Recorded ensemble
packets verify raw measurement fields and geometry against each already loaded
payload. Planar entries are classified as recorded ensemble, recorded ordinary
iteration, or without execution metadata. Absence does not identify an expected
ensemble workflow. An ensemble packet cannot coexist with ordinary iteration,
stereo, or measurement-history metadata on the same entry. Different entries
may carry different supported companions. Invalid root markers are refused
even in empty files. Experiment association is unsupported for this opt-in.

Ensemble observations separately count all-pass window-pair/source-support
populations and final-pass contributor/primary/UQ support. Each recorded pass
executed exactly one pooled sweep; requested iteration/tolerance settings were
ignored, with no convergence checks. Pair observations may reuse inputs across
pools/passes and are not unique or independent samples. Finite nonzero planes
include flat planes and do not certify a valid measurement. UQ component counts
cover only numerically evaluated final pools before validation/cleanup;
unevaluated pools do not count as evaluated missing estimates. Residual/UQ units
remain processing pixels regardless of result scale. No residual amplitudes,
effective sample size, stationarity, uncertainty applicability or coverage are
aggregated or inferred. Loaded v4 reports describe checks at generation only.

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
quality_report(results;include_measurement_history::Bool=false,include_execution_diagnostics::Bool=false,
        include_ensemble_execution_diagnostics::Bool=false)=
    _quality_report(results,include_measurement_history,include_execution_diagnostics,nothing,include_ensemble_execution_diagnostics)
function _quality_report(results,include_measurement_history,include_execution_diagnostics,expected_association,include_ensemble_execution_diagnostics=false)
    companions=include_measurement_history || include_execution_diagnostics || include_ensemble_execution_diagnostics
    include_ensemble_execution_diagnostics && expected_association!==nothing &&
        _quality_error("ensemble reports have no supported experiment recipe association")
    companions && !(results isa ResultFile) && _quality_error("companion-aware reports require a direct ResultFile or verified record/run; converted wrappers, array views and bare iterators have no checked entry mapping")
    include_ensemble_execution_diagnostics && (results=_quality_whole_file_index(results))
    Base.IteratorSize(typeof(results)) isa Base.IsInfinite && _quality_error("quality reports require a finite sequence")
    groups = Dict{String,Dict{String,Int}}()
    kinds=companions ? Dict(k=>0 for k in _QUALITY_HISTORY_KINDS) : nothing
    history_counts=include_measurement_history ? Dict(k=>0 for k in _quality_history_counter_names()) : nothing
    execution_counts=include_execution_diagnostics ? Dict(role=>Dict(k=>0 for k in _quality_execution_counter_names()) for role in _QUALITY_EXECUTION_ROLES) : nothing
    ensemble_counts=include_ensemble_execution_diagnostics ? Dict(k=>0 for k in _quality_ensemble_counter_names()) : nothing
    ensemble_classification=include_ensemble_execution_diagnostics ? Dict(k=>0 for k in _QUALITY_ENSEMBLE_CLASSIFICATION) : nothing
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
    if include_measurement_history
        jldopen(f->_check_measurement_history_format(f),source.path,"r")
        _check_result_file(source)
    end
    if include_ensemble_execution_diagnostics
        _quality_check_ensemble_format(source)
    elseif include_execution_diagnostics
        _quality_check_execution_format(source)
    end
    for (i,result) in enumerate(results)
        if companions
            kind=result isa PIVResult ? "planar" : result isa StereoPIVResult ? "stereo" : result isa PTVResult ? "ptv" : "tracking"
            _quality_add!(kinds,kind)
        end
        loaded=include_ensemble_execution_diagnostics ? _quality_ensemble_observe!(ensemble_classification,ensemble_counts,results,i,result) : nothing
        if include_measurement_history
            history=loaded===nothing ? load_measurement_history(results,i) : loaded.history # raw payload already loaded once
            if history!==nothing && expected_association!==nothing
                d=_history_checked_data(history)
                d["association"]==expected_association && d["pair_index"]===i || _quality_error("recorded measurement-history recipe/input or pair association disagrees with the selected run")
            end
            if result isa PIVResult
                _quality_add!(history_counts,"planar_entries")
                history===nothing ? _quality_add!(history_counts,"missing_entries") : _quality_history_update!(history_counts,history,result)
            else
                history===nothing || _quality_error("measurement-history companion attached to unsupported $kind result")
                _quality_add!(history_counts,"unsupported_$(kind)_entries")
            end
            history=nothing
        end
        include_execution_diagnostics && _quality_execution_update!(execution_counts,results,i,result,expected_association;loaded)
        (!companions || result isa Union{PIVResult,StereoPIVResult}) && _quality_update!(groups,result)
        result=nothing
        loaded=nothing
    end
    if source !== nothing
        _check_result_file(source)
        _experiment_file_digest(source.path) == provenance["source_sha256"] || _quality_error("result source changed while building quality report")
        _check_result_file(source)
    end
    version=include_ensemble_execution_diagnostics ? QUALITY_ENSEMBLE_REPORT_FORMAT_VERSION :
        include_execution_diagnostics ? QUALITY_EXECUTION_REPORT_FORMAT_VERSION :
        include_measurement_history ? QUALITY_HISTORY_REPORT_FORMAT_VERSION : QUALITY_REPORT_FORMAT_VERSION
    data = Dict{String,Any}("quality_report_format_version" => version,
        "generated_at_unix_s" => time(),
        "generator" => Dict("julia_version" => string(VERSION), "hammerhead_version" => string(Base.pkgversion(Hammerhead)),
                            "core_source_sha256" => _experiment_software()["core_source_sha256"],
                            "value_basis" => _quality_value_basis(include_measurement_history,include_execution_diagnostics,include_ensemble_execution_diagnostics),
                            "weighting" => include_execution_diagnostics || include_ensemble_execution_diagnostics ? "field_nodes_and_execution_observations" : "node_weighted"),
        "provenance" => provenance, "protected_locators" => locators,
        "groups" => Dict(kind => Dict("counts" => counts, "fractions" => _quality_fractions(counts, kind)) for (kind, counts) in groups),
        "unavailable" => _quality_unavailable(version;include_history=include_measurement_history))
    companions && (data["entry_kinds"]=kinds)
    if include_measurement_history
        data["measurement_history"]=Dict("scope"=>"final_pass_final_executed_sweep","binding"=>"raw_measurement_digest_verified",
            "counts"=>history_counts,"fractions"=>_quality_make_fractions(history_counts,_quality_history_metric_specs()))
    end
    include_execution_diagnostics && (data["execution_diagnostics"]=_quality_execution_section(execution_counts,kinds))
    include_ensemble_execution_diagnostics && (data["ensemble_execution_diagnostics"]=_quality_ensemble_section(ensemble_classification,ensemble_counts,kinds))
    _quality_wrap(data)
end

function quality_report(record::ExperimentRecord, run::ExperimentRun;
        include_measurement_history::Bool=false,include_execution_diagnostics::Bool=false,
        include_ensemble_execution_diagnostics::Bool=false)
    include_ensemble_execution_diagnostics && _quality_error("ensemble reports have no supported experiment recipe association; use a direct whole-file ResultFile")
    _experiment_preflight(record)
    _experiment_validate_environment(record.creation_environment)
    validated = _experiment_run(_experiment_run_data(run), record)
    validated.status === :completed && validated.output_sha256 !== nothing || _quality_error("quality association requires a completed run with output identity")
    source = ResultFile(validated.output)
    length(source) == validated.completed_pairs || _quality_error("completed run/result entry counts disagree")
    _experiment_file_digest(source.path) == validated.output_sha256 || _quality_error("recorded result output changed")
    report = _quality_report(source,include_measurement_history,include_execution_diagnostics,
        include_measurement_history || include_execution_diagnostics ? Dict("recipe_id"=>validated.recipe_id,"input_id"=>validated.input_id) : nothing)
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

Read and validate a version-1 or opt-in version-2/3/4/5 TOML quality report without opening any recorded
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
    association = data["quality_report_format_version"] === 5 ? "verified recorded ensemble output" :
        data["provenance"]["association"] == "recorded_output_verified" ? "verified recorded output" : "unassociated"
    print(io, "Run quality: stored-array counts (node-weighted)\nExperiment association: ", association)
    data["quality_report_format_version"] === 5 && _quality_show_ensemble_experiment(io,data["provenance"])
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
    if haskey(data,"measurement_history")
        h=data["measurement_history"]["counts"]
        print(io,"\nRecorded history: final pass/final executed sweep only")
        print(io,"\n  Coverage: ",h["recorded_entries"]," / ",h["planar_entries"]," planar entries; ",h["missing_entries"]," missing")
        print(io,"\n  Unsupported history: ",h["unsupported_stereo_entries"]," stereo, ",h["unsupported_ptv_entries"]," PTV, ",h["unsupported_tracking_entries"]," tracking entries")
        print(io,"\n  Covered nodes: ",h["unmasked"]," unmasked, ",h["masked"]," masked")
        for (key,label) in (("first_rejected","First observed rejections"),("pre_substitution_flagged","Flags before alternatives"),
                            ("accepted_alternative","Accepted alternatives"),("fill_attempted","Median fill attempts"),
                            ("fill_assigned","Median assignments"),("primary_restored","Primary restorations"))
            print(io,"\n  ",label,": ",h[key]," / ",h["unmasked"]," history-covered unmasked nodes")
        end
        print(io,"\n  Final origins: ",h["origin_primary"]," primary, ",h["origin_alternative"]," alternative, ",h["origin_fill"]," fill, ",h["origin_unavailable"]," unavailable, ",h["origin_custom_unclassified"]," custom unclassified")
        print(io,"\n  Unclassified first rejection labels: ",h["rejection_unclassified"])
        print(io,"\n  Missing history entries are coverage gaps, not zero-event measurements. Median assignments may be nonfinite or later restored.")
    end
    if haskey(data,"execution_diagnostics")
        execution=data["execution_diagnostics"]
        print(io,"\nRecorded execution: entry/pass observations, not node-weighted field quality")
        for role in _QUALITY_EXECUTION_ROLES
            group=execution["groups"][role];c=group["counts"]
            print(io,"\n  ",role,": ",c["recorded_entries"]," / ",c["eligible_entries"]," entries; ",c["missing_entries"]," missing; ",group["binding"])
            print(io,"\n    ",c["passes"]," passes; ",c["executed_sweeps"]," / ",c["requested_sweeps"]," sweeps; ",c["tolerance_checks"]," tolerance checks")
            print(io,"\n    Stops: ",c["stop_single_sweep"]," single sweep, ",c["stop_iteration_budget"]," budget, ",c["stop_tolerance_condition_met"]," tolerance condition")
            print(io,"\n    Last checks: ",c["last_checks_present"]," evaluated, ",c["last_checks_absent"]," absent, ",c["last_check_empty"]," empty support")
            print(io,"\n    Final primary support: ",c["final_primary_finite"]," finite, ",c["final_primary_nonfinite"]," nonfinite, ",c["final_primary_masked"]," masked (",group["coordinate_basis"],")")
        end
        print(io,"\n  Unsupported execution kinds: ",execution["unsupported_entries"]["ptv"]," PTV, ",execution["unsupported_entries"]["tracking"]," tracking")
        print(io,"\n  Stereo raw measurement-field binding was checked when generated; loading this report does not freshly verify results.")
        print(io,"\n  Pixel residual amplitudes are not pooled. Tolerance conditions, including empty comparisons, do not establish measurement validity or accuracy.")
    end
    haskey(data,"ensemble_execution_diagnostics") && _quality_show_ensemble(io,data["ensemble_execution_diagnostics"])
    print(io, "\nStored uncertainty availability means finite and nonnegative; it does not establish measurement association or calibrated coverage.")
    for (name, reason) in sort!(collect(data["unavailable"]); by = first)
        label = name == "uncertainty_measurement_association" ? "Uncertainty measurement association" : uppercasefirst(replace(name, '_' => ' '))
        print(io, "\n", label, ": unavailable (", replace(reason["reason_code"], '_' => ' '), ')')
    end
end
