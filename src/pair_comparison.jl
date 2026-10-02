# Controlled, selected-pair reruns. These are sensitivity measurements, not truth.
const PAIR_COMPARISON_FORMAT_VERSION = 1

"""
    RecipePairComparison

Detached comparison of two built-in planar recipes on one verified ordered
image pair. Use [`pair_comparison_data`](@ref) for a copied primitive report.
No image, result, recipe array, or open file is retained. Settings changes and
numerical differences describe sensitivity, not accuracy or uncertainty coverage.
"""
struct RecipePairComparison
    _data::Dict{String,Any}
    _protected_paths::Tuple{Vararg{String}}
    _identity::String
end

_pair_error(message) = throw(ArgumentError(message))

# Tagged settings preserve missing/nothing, tuple shape and symbols in TOML.
function _pair_setting(value)
    kind, payload = if value === missing
        "missing", nothing
    elseif value === nothing
        "nothing", nothing
    elseif value isa RecipeArraySummary
        "array_summary", Dict{String,Any}("size"=>collect(value.size),
            "element_type"=>value.element_type,"sha256"=>value.sha256)
    elseif value isa NamedTuple
        "mapping", Dict{String,Any}(String(k)=>_pair_setting(v) for (k,v) in pairs(value))
    elseif value isa Tuple
        "tuple", [_pair_setting(v) for v in value]
    elseif value isa Symbol
        "symbol", String(value)
    elseif value isa Bool
        "bool", value
    elseif value isa Integer
        "integer", Int(value)
    elseif value isa AbstractFloat
        "float", Float64(value)
    elseif value isa AbstractString
        "string", String(value)
    else
        _pair_error("unsupported recipe comparison setting $(typeof(value))")
    end
    data=Dict{String,Any}("kind"=>kind)
    payload===nothing || (data["value"]=payload)
    data
end

function _pair_check_setting(data)
    data isa AbstractDict && haskey(data,"kind") || _pair_error("malformed comparison setting")
    kind=data["kind"]
    kind isa String || _pair_error("invalid setting kind")
    _experiment_keys(data,kind in ("missing","nothing") ? ["kind"] : ["kind","value"],"comparison setting")
    kind in ("missing","nothing") && return nothing
    value=data["value"]
    if kind=="array_summary"
        _experiment_keys(value,["size","element_type","sha256"],"array summary")
        value["size"] isa AbstractVector && all(v->v isa Int && v>=0,value["size"]) &&
            value["element_type"] isa String && _experiment_hash(value["sha256"]) || _pair_error("invalid array summary")
    elseif kind=="mapping"
        value isa AbstractDict && all(k->k isa String,keys(value)) || _pair_error("invalid setting mapping")
        foreach(_pair_check_setting,values(value))
    elseif kind=="tuple"
        value isa AbstractVector || _pair_error("invalid setting tuple")
        foreach(_pair_check_setting,value)
    elseif kind in ("string","symbol")
        value isa String || _pair_error("invalid setting string")
    elseif kind=="bool"
        value isa Bool || _pair_error("invalid setting boolean")
    elseif kind=="integer"
        value isa Int || _pair_error("invalid setting integer")
    elseif kind=="float"
        value isa AbstractFloat && !isnan(value) || _pair_error("invalid setting float")
    else
        _pair_error("unknown comparison setting kind")
    end
    nothing
end

_pair_scale(scale) = scale===nothing ? Dict{String,Any}("available"=>false) :
    Dict{String,Any}("available"=>true,"pixel_size"=>scale.pixel_size,"dt"=>scale.dt,
        "length_unit"=>scale.length_unit,"time_unit"=>scale.time_unit)

function _pair_check_scale(data)
    data isa AbstractDict && get(data,"available",nothing) isa Bool || _pair_error("invalid comparison scale")
    _experiment_keys(data,data["available"] ? ["available","pixel_size","dt","length_unit","time_unit"] : ["available"],"comparison scale")
    if data["available"]
        all(k->data[k] isa AbstractFloat && isfinite(data[k]) && data[k]>0,("pixel_size","dt")) &&
            all(k->data[k] isa String,("length_unit","time_unit")) || _pair_error("invalid scale factors/labels")
    end
    nothing
end

function _pair_basis(before,after,basis)
    basis in (:pixels,:physical) || _pair_error("basis must be :pixels or :physical")
    basis===:pixels && return (factor=1.0,unit="px",quantity="displacement")
    a,b=before.scale,after.scale
    a!==nothing && b!==nothing && _pair_scale(a)==_pair_scale(b) ||
        _pair_error("physical comparison requires identical scale factors and unit labels on both recipes")
    factor=a.pixel_size/a.dt
    isfinite(factor) && factor>0 || _pair_error("physical comparison conversion factor is not finite and positive")
    (factor=factor,unit=velocity_unit(a),quantity="velocity")
end

# Stable means and scaled sum of squares avoid squaring huge finite samples.
mutable struct _PairMoment
    count::Int
    mean::Float64
    scale::Float64
    ssq::Float64
    failed::Bool
end
_PairMoment() = _PairMoment(0,0.0,0.0,0.0,false)
function _pair_push!(moment::_PairMoment,value)
    moment.count=Base.Checked.checked_add(moment.count,1)
    if !isfinite(value)
        moment.failed=true
        return nothing
    end
    n=moment.count
    delta=value-moment.mean
    moment.mean=isfinite(delta) ? moment.mean+delta/n : moment.mean*((n-1)/n)+value/n
    magnitude=abs(value)
    if magnitude>moment.scale
        moment.ssq=1+moment.ssq*(moment.scale/magnitude)^2
        moment.scale=magnitude
    elseif magnitude>0
        moment.ssq+=(magnitude/moment.scale)^2
    end
    isfinite(moment.mean) || (moment.failed=true)
    nothing
end
_pair_rms(moment) = moment.scale*sqrt(moment.ssq/moment.count)

function _pair_metric(moments,count,empty_reason;vector=false)
    count==0 && return Dict{String,Any}("available"=>false,"count"=>0,"reason_code"=>empty_reason)
    components=Dict{String,Any}(name=>Dict("mean"=>m.mean,"rms"=>_pair_rms(m)) for (name,m) in moments)
    magnitude=vector ? hypot(components["u"]["rms"],components["v"]["rms"]) : 0.0
    if any(m->m.failed,values(moments)) || any(v->!isfinite(v),Iterators.flatten(values(v) for v in values(components))) || !isfinite(magnitude)
        return Dict{String,Any}("available"=>false,"count"=>count,"reason_code"=>"nonfinite_arithmetic")
    end
    data=Dict{String,Any}("available"=>true,"count"=>count,"components"=>components)
    vector && (data["vector_rms_difference"]=magnitude)
    data
end

function _pair_check_grid(result)
    shape=(length(result.y),length(result.x))
    all(axis->!isempty(axis) && all(isfinite,axis) && all(>(0),diff(axis)),(result.x,result.y)) ||
        _pair_error("comparison coordinates must be finite, unique and strictly increasing")
    all(a->size(a)==shape,(result.u,result.v,result.outliers,result.mask,result.uncertainty_u,result.uncertainty_v)) ||
        throw(DimensionMismatch("comparison fields and flags must match the raw coordinate grid"))
    nothing
end

function _pair_intersect(a,b)
    left,right=Int[],Int[]
    i=j=1
    while i<=length(a) && j<=length(b)
        if a[i]==b[j]
            push!(left,i);push!(right,j);i+=1;j+=1
        elseif a[i]<b[j]
            i+=1
        else
            j+=1
        end
    end
    left,right
end

function _pair_native(result)
    groups=Dict{String,Dict{String,Int}}()
    _quality_update!(groups,result)
    counts=groups["planar"]
    Dict{String,Any}("grid"=>Dict("nx"=>length(result.x),"ny"=>length(result.y),
        "x_extent_px"=>[Float64(first(result.x)),Float64(last(result.x))],
        "y_extent_px"=>[Float64(first(result.y)),Float64(last(result.y))]),
        "counts"=>counts,"fractions"=>_quality_fractions(counts,"planar"))
end

function _pair_measure(before,after,factor)
    _pair_check_grid(before);_pair_check_grid(after)
    xa,xb=_pair_intersect(before.x,after.x)
    ya,yb=_pair_intersect(before.y,after.y)
    counters=Dict(k=>0 for k in ("nodes","masked_both","masked_before_only","masked_after_only","unmasked_both",
        "valid_both","valid_before_only","valid_after_only","invalid_both"))
    counters["nodes"]=Base.Checked.checked_mul(length(xa),length(ya))
    uq=Dict(k=>0 for k in ("available_both","available_before_only","available_after_only","unavailable_both"))
    velocity=Dict("u"=>_PairMoment(),"v"=>_PairMoment())
    uncertainty=Dict(k=>_PairMoment() for k in ("before_u","before_v","after_u","after_v","difference_u","difference_v"))
    for (ja,jb) in zip(xa,xb),(ia,ib) in zip(ya,yb)
        a,b=CartesianIndex(ia,ja),CartesianIndex(ib,jb)
        ma,mb=before.mask[a],after.mask[b]
        if ma || mb
            counters[ma && mb ? "masked_both" : ma ? "masked_before_only" : "masked_after_only"]+=1
            continue
        end
        counters["unmasked_both"]+=1
        va,vb=sample_valid(before,a,false),sample_valid(after,b,false)
        counters[va && vb ? "valid_both" : va ? "valid_before_only" : vb ? "valid_after_only" : "invalid_both"]+=1
        va && vb || continue
        _pair_push!(velocity["u"],Float64(after.u[b])*factor-Float64(before.u[a])*factor)
        _pair_push!(velocity["v"],Float64(after.v[b])*factor-Float64(before.v[a])*factor)
        ua=all(v->isfinite(v) && v>=0,(before.uncertainty_u[a],before.uncertainty_v[a]))
        ub=all(v->isfinite(v) && v>=0,(after.uncertainty_u[b],after.uncertainty_v[b]))
        uq[ua && ub ? "available_both" : ua ? "available_before_only" : ub ? "available_after_only" : "unavailable_both"]+=1
        ua && ub || continue
        for (component,av,bv) in (("u",before.uncertainty_u[a],after.uncertainty_u[b]),("v",before.uncertainty_v[a],after.uncertainty_v[b]))
            first_value,last_value=Float64(av)*factor,Float64(bv)*factor
            _pair_push!(uncertainty["before_"*component],first_value)
            _pair_push!(uncertainty["after_"*component],last_value)
            _pair_push!(uncertainty["difference_"*component],last_value-first_value)
        end
    end
    reason=counters["nodes"]==0 ? "no_common_nodes" : "no_joint_valid_nodes"
    Dict{String,Any}("coordinates"=>"exact_original_image_pixels","common_x_nodes"=>length(xa),
        "common_y_nodes"=>length(ya),"counts"=>counters,"uq_counts_on_joint_valid"=>uq,
        "velocity_difference"=>_pair_metric(velocity,counters["valid_both"],reason;vector=true),
        "stored_uncertainty"=>_pair_metric(uncertainty,uq["available_both"],
            counters["valid_both"]==0 ? reason : "no_joint_uq_nodes"))
end

function _pair_check_metric(data,count,components,empty_reason;vector=false)
    data isa AbstractDict && get(data,"available",nothing) isa Bool &&
        get(data,"count",nothing) isa Int && data["count"]==count || _pair_error("invalid comparison metric population")
    if data["available"]
        _experiment_keys(data,vector ? ["available","count","components","vector_rms_difference"] :
            ["available","count","components"],"comparison metric")
        count>0 || _pair_error("empty comparison metric cannot be available")
        _experiment_keys(data["components"],components,"metric components")
        for (name,summary) in data["components"]
            _experiment_keys(summary,["mean","rms"],"component moments")
            all(v->v isa AbstractFloat && isfinite(v),values(summary)) && summary["rms"]>=0 ||
                _pair_error("comparison moments must be finite with nonnegative RMS")
            (abs(summary["mean"])<=summary["rms"] || isapprox(abs(summary["mean"]),summary["rms"];rtol=32eps(Float64))) ||
                _pair_error("comparison RMS is smaller than absolute mean")
            startswith(name,"before_") || startswith(name,"after_") ?
                (summary["mean"]>=0 || _pair_error("stored uncertainty mean must be nonnegative")) : nothing
        end
        if vector
            value=data["vector_rms_difference"]
            value isa AbstractFloat && isfinite(value) && value>=0 &&
                value==hypot(data["components"]["u"]["rms"],data["components"]["v"]["rms"]) ||
                _pair_error("invalid vector RMS difference")
        end
    else
        _experiment_keys(data,["available","count","reason_code"],"unavailable comparison metric")
        data["reason_code"]==(count==0 ? empty_reason : "nonfinite_arithmetic") || _pair_error("invalid metric unavailability reason")
    end
    nothing
end

function _pair_check_native(native,generator,time)
    _experiment_keys(native,["grid","counts","fractions"],"native grid summary")
    grid=native["grid"]
    _experiment_keys(grid,["nx","ny","x_extent_px","y_extent_px"],"native grid")
    all(k->grid[k] isa Int && grid[k]>0,("nx","ny")) || _pair_error("invalid native grid dimensions")
    for (axis,n) in (("x_extent_px",grid["nx"]),("y_extent_px",grid["ny"]))
        values=grid[axis]
        values isa AbstractVector && length(values)==2 && all(v->v isa AbstractFloat && isfinite(v),values) &&
            (n==1 ? values[1]==values[2] : values[1]<values[2]) || _pair_error("invalid native coordinate extent")
    end
    # Reuse the stored-field quality schema rather than diverging flag/UQ rules.
    quality=Dict{String,Any}("quality_report_format_version"=>QUALITY_REPORT_FORMAT_VERSION,
        "generated_at_unix_s"=>time,"generator"=>Dict("julia_version"=>generator["julia_version"],
            "hammerhead_version"=>generator["hammerhead_version"],"core_source_sha256"=>generator["core_source_sha256"],
            "value_basis"=>"stored_arrays","weighting"=>"node_weighted"),
        "provenance"=>Dict("association"=>"unassociated"),"protected_locators"=>String[],
        "groups"=>Dict("planar"=>Dict("counts"=>native["counts"],"fractions"=>native["fractions"])),
        "unavailable"=>Dict(k=>Dict("available"=>false,"reason_code"=>v) for (k,v) in _QUALITY_UNAVAILABLE))
    _quality_validate(quality)
    native["counts"]["entries"]==1 && native["counts"]["nodes"]==Base.Checked.checked_mul(grid["nx"],grid["ny"]) ||
        _pair_error("native counts disagree with the selected result grid")
    nothing
end

function _pair_check_selected(selected)
    _experiment_keys(selected,["pair_index","recipe_id","record_input_id","creation_environment_id","files"],"selected-pair provenance")
    selected["pair_index"] isa Int && selected["pair_index"]>0 &&
        all(k->_experiment_hash(selected[k]),("recipe_id","record_input_id","creation_environment_id")) ||
        _pair_error("invalid selected-pair identities")
    files=selected["files"]
    files isa AbstractVector && length(files)==2 || _pair_error("selected provenance must contain two ordered files")
    for file in files
        _experiment_keys(file,["path","sha256","size_bytes","image_size"],"selected input")
        file["path"] isa String && isabspath(file["path"]) && _experiment_hash(file["sha256"]) &&
            file["size_bytes"] isa Int && file["size_bytes"]>=0 && file["image_size"] isa AbstractVector &&
            length(file["image_size"])==2 && all(v->v isa Int && v>0,file["image_size"]) || _pair_error("invalid selected input descriptor")
    end
    files[1]["image_size"]==files[2]["image_size"] || _pair_error("selected image dimensions differ")
    _experiment_digest(_experiment_input_data(files,[[1,2]]))
end

# TOML has no null: preserve nullable provenance explicitly, including dev
# package tree hashes. Reconstruct the original typed package vector for hashing.
function _pair_environment(environment)
    encoded=deepcopy(environment)
    for key in ("project_path","project_text","manifest_text")
        encoded[key]=_pair_setting(environment[key])
    end
    for package in encoded["packages"],key in ("version","tree_hash")
        package[key]=_pair_setting(package[key])
    end
    encoded
end
function _pair_nullable(data)
    _pair_check_setting(data)
    data["kind"]=="nothing" && return nothing
    data["kind"]=="string" || _pair_error("nullable software provenance must be a string or nothing")
    data["value"]
end
function _pair_restore_environment(encoded)
    _experiment_keys(encoded,["julia_version","hammerhead_version","core_source_sha256","packages","kernel","architecture",
        "julia_threads","fftw_threads","project_path","project_text","manifest_text"],"comparison environment")
    environment=deepcopy(encoded)
    for key in ("project_path","project_text","manifest_text")
        environment[key]=_pair_nullable(encoded[key])
    end
    encoded["packages"] isa AbstractVector || _pair_error("invalid comparison software packages")
    packages=Dict{String,Any}[]
    for original in encoded["packages"]
        _experiment_keys(original,["uuid","name","version","tree_hash"],"comparison software package")
        package=Dict{String,Any}(original)
        for key in ("version","tree_hash");package[key]=_pair_nullable(original[key]);end
        push!(packages,package)
    end
    environment["packages"]=packages
    _experiment_validate_environment(environment)
end

function _pair_validate(data)
    _experiment_keys(data,["pair_comparison_format_version","generated_at_unix_s","generator","actual_environment",
        "provenance","settings_changes","basis","native","common","unavailable","protected_locators"],"pair comparison")
    data["pair_comparison_format_version"]===PAIR_COMPARISON_FORMAT_VERSION || _pair_error("unsupported pair comparison version")
    time=data["generated_at_unix_s"]
    time isa AbstractFloat && isfinite(time) && time>=0 || _pair_error("invalid comparison timestamp")
    generator=data["generator"]
    _experiment_keys(generator,["julia_version","hammerhead_version","core_source_sha256","actual_environment_id"],"comparison generator")
    all(k->generator[k] isa String && !isempty(generator[k]),("julia_version","hammerhead_version")) &&
        all(k->_experiment_hash(generator[k]),("core_source_sha256","actual_environment_id")) || _pair_error("invalid comparison generator")
    environment=_pair_restore_environment(data["actual_environment"])
    generator["actual_environment_id"]==_experiment_digest(_experiment_environment_signature(environment)) &&
        all(k->generator[k]==environment[k],("julia_version","hammerhead_version","core_source_sha256")) ||
        _pair_error("comparison environment identity mismatch")
    locators=data["protected_locators"]
    locators isa AbstractVector && all(p->p isa String && isabspath(p),locators) &&
        length(unique(locators))==length(locators) || _pair_error("invalid comparison protected locators")
    provenance=data["provenance"]
    _experiment_keys(provenance,["verification","ordered_pair_id","allow_environment_change","before","after"],"comparison provenance")
    provenance["verification"]=="selected_ordered_pair_bytes_and_dimensions" &&
        provenance["allow_environment_change"] isa Bool && _experiment_hash(provenance["ordered_pair_id"]) || _pair_error("invalid pair verification")
    for side in ("before","after")
        selected=provenance[side]
        _pair_check_selected(selected)==provenance["ordered_pair_id"] || _pair_error("selected ordered input identities disagree")
        all(f->f["path"] in locators,selected["files"]) || _pair_error("selected input is missing destination protection")
        provenance["allow_environment_change"] || selected["creation_environment_id"]==generator["actual_environment_id"] ||
            _pair_error("strict comparison creation environment differs")
    end
    changes=data["settings_changes"]
    changes isa AbstractVector || _pair_error("invalid comparison settings diff")
    paths=String[]
    for change in changes
        _experiment_keys(change,["path","before","after"],"comparison change")
        change["path"] isa String && !isempty(change["path"]) || _pair_error("invalid recipe change path")
        push!(paths,change["path"])
        _pair_check_setting(change["before"]);_pair_check_setting(change["after"])
    end
    length(unique(paths))==length(paths) || _pair_error("duplicate recipe change paths")
    (provenance["before"]["recipe_id"]==provenance["after"]["recipe_id"])==isempty(changes) ||
        _pair_error("recipe identities disagree with settings-change availability")
    basis=data["basis"]
    _experiment_keys(basis,["mode","quantity","unit","factor","before_scale","after_scale","difference_direction"],"comparison basis")
    basis["mode"] in ("pixels","physical") && basis["unit"] isa String && basis["difference_direction"]=="after_minus_before" &&
        basis["factor"] isa AbstractFloat && isfinite(basis["factor"]) && basis["factor"]>0 || _pair_error("invalid comparison basis")
    _pair_check_scale(basis["before_scale"]);_pair_check_scale(basis["after_scale"])
    if basis["mode"]=="pixels"
        basis["quantity"]=="displacement" && basis["unit"]=="px" && basis["factor"]==1 || _pair_error("invalid pixel comparison units")
    else
        scale=basis["before_scale"]
        scale["available"] && scale==basis["after_scale"] && basis["quantity"]=="velocity" &&
            basis["unit"]==scale["length_unit"]*"/"*scale["time_unit"] && basis["factor"]==scale["pixel_size"]/scale["dt"] ||
            _pair_error("invalid physical comparison scale/units")
    end
    native=data["native"]
    _experiment_keys(native,["before","after"],"native summaries")
    foreach(s->_pair_check_native(native[s],generator,time),("before","after"))
    common=data["common"]
    _experiment_keys(common,["coordinates","common_x_nodes","common_y_nodes","counts","uq_counts_on_joint_valid",
        "velocity_difference","stored_uncertainty"],"common-node comparison")
    common["coordinates"]=="exact_original_image_pixels" && all(k->common[k] isa Int && common[k]>=0,("common_x_nodes","common_y_nodes")) ||
        _pair_error("invalid common coordinates")
    common["common_x_nodes"]<=min(native["before"]["grid"]["nx"],native["after"]["grid"]["nx"]) &&
        common["common_y_nodes"]<=min(native["before"]["grid"]["ny"],native["after"]["grid"]["ny"]) || _pair_error("common grid exceeds native grids")
    c=common["counts"]
    _experiment_keys(c,["nodes","masked_both","masked_before_only","masked_after_only","unmasked_both",
        "valid_both","valid_before_only","valid_after_only","invalid_both"],"common counters")
    all(v->v isa Int && v>=0,values(c)) && c["nodes"]==Base.Checked.checked_mul(common["common_x_nodes"],common["common_y_nodes"]) &&
        c["nodes"]==_quality_sum(c["masked_both"],c["masked_before_only"],c["masked_after_only"],c["unmasked_both"]) &&
        c["unmasked_both"]==_quality_sum(c["valid_both"],c["valid_before_only"],c["valid_after_only"],c["invalid_both"]) ||
        _pair_error("inconsistent common-node populations")
    for (side,only) in (("before","valid_before_only"),("after","valid_after_only"))
        _quality_sum(c["valid_both"],c[only])<=native[side]["counts"]["unflagged_finite_output_unmasked"] ||
            _pair_error("common validity exceeds native validity")
    end
    uq=common["uq_counts_on_joint_valid"]
    _experiment_keys(uq,["available_both","available_before_only","available_after_only","unavailable_both"],"joint-valid UQ populations")
    all(v->v isa Int && v>=0,values(uq)) && _quality_sum(values(uq)...)==c["valid_both"] || _pair_error("inconsistent joint UQ population")
    for (side,only) in (("before","available_before_only"),("after","available_after_only"))
        _quality_sum(uq["available_both"],uq[only])<=native[side]["counts"]["unflagged_finite_output_with_uq_available"] ||
            _pair_error("common uncertainty availability exceeds native availability")
    end
    reason=c["nodes"]==0 ? "no_common_nodes" : "no_joint_valid_nodes"
    _pair_check_metric(common["velocity_difference"],c["valid_both"],["u","v"],reason;vector=true)
    _pair_check_metric(common["stored_uncertainty"],uq["available_both"],
        ["before_u","before_v","after_u","after_v","difference_u","difference_v"],c["valid_both"]==0 ? reason : "no_joint_uq_nodes")
    expected=Dict("accuracy"=>"not_evaluated","uncertainty_coverage"=>"not_evaluated",
        "normalized_difference_by_uncertainty"=>"error_correlation_unknown",
        "rejection_events"=>"not_persisted","replacement_history"=>"not_persisted",
        "alternative_peak_history"=>"not_persisted","uncertainty_measurement_association"=>"not_persisted")
    unavailable=data["unavailable"]
    _experiment_keys(unavailable,collect(keys(expected)),"unavailable diagnostics")
    for (name,reason) in expected
        entry=unavailable[name]
        _experiment_keys(entry,["available","reason_code"],"unavailable diagnostic")
        entry["available"] isa Bool && entry["available"]===false &&
            entry["reason_code"] isa String && entry["reason_code"]==reason || _pair_error("invalid unavailable diagnostic")
    end
    data
end

function _pair_wrap(data)
    _pair_validate(data)
    snapshot=deepcopy(data)
    RecipePairComparison(snapshot,Tuple(snapshot["protected_locators"]),_experiment_digest(snapshot))
end

"""
    pair_comparison_data(report::RecipePairComparison) -> Dict

Return a detached primitive report, including tagged settings changes, verified
selected-pair provenance, native-grid quality summaries and exact-common-node
populations. Access never reopens input files; this is a past verification.
"""
function pair_comparison_data(report::RecipePairComparison)
    _experiment_digest(report._data)==report._identity || _pair_error("comparison snapshot was mutated")
    _pair_validate(report._data)
    deepcopy(report._data)
end

function _pair_run(record,index)
    recipe=record.recipe
    preprocess=_experiment_preprocess(recipe,nothing)
    images=[preprocess(_experiment_load_image(record.input_files[i],recipe.image_type)) for i in record.pairs[index]]
    run_piv(images...,recipe.passes;backend=recipe.backend,threaded=recipe.threaded,
        predictor_smoothing=recipe.predictor_smoothing,mask=recipe.mask,mask_threshold=recipe.mask_threshold,
        uncertainty_backend=recipe.uncertainty_backend,roi=recipe.roi,scale=recipe.scale)
end

function _pair_selected(record,index)
    Dict{String,Any}("pair_index"=>index,"recipe_id"=>recipe_identity(record.recipe),"record_input_id"=>record.input_id,
        "creation_environment_id"=>_experiment_digest(_experiment_environment_signature(record.creation_environment)),
        "files"=>deepcopy(record.input_files[record.pairs[index]]))
end

"""
    compare_recipe_pair(before::ExperimentRecord, after::ExperimentRecord;
                        pair_indices, basis=:pixels, allow_environment_change=false)

Rerun the explicitly selected `(before_index, after_index)` ordered image pair
under both complete built-in planar recipes. Selected SHA-256 byte identities,
file sizes and decoded dimensions must agree; whole-record identities may
differ. Only selected images are opened. Custom scripts/callbacks are refused.
Recipe/record integrity, backends, units and strict creation/software environment
compatibility are checked before processing. An explicit environment override
reruns both recipes under the same recorded current environment. Source hashes
and environment signatures are checked again after computation. No outputs or
run-history records are written; keep source/software files unchanged throughout.

Match finite, unique, increasing raw x/y coordinates by exact numerical value
in the original image frame, including ROI offsets. No interpolation or filling
is performed. Native-grid summaries have separate populations. Paired differences
use only common nodes with finite u/v, no mask and no current outlier flag in
both results. Stored UQ differences use that population further restricted to
all four uncertainty components finite and nonnegative. Zero uncertainty is
available. Current flags/UQ do not reconstruct replacement/alternative history.

`basis=:pixels` compares raw displacement in px and retains differing scales as
metadata, without claiming a common physical velocity. `:physical` requires
identical attached scale factors and unit labels; values/UQ convert in Float64
while node matching remains in raw pixels. Differences are after minus before;
mean and RMS differences describe sensitivity, not accuracy errors. Empty
populations or arithmetic overflow produce explicit unavailable metrics; overflow
does not silently remove nodes from the raw joint-valid denominator. Normalized
uncertainty comparisons/coverage require unavailable error-correlation evidence.
The returned [`RecipePairComparison`](@ref) retains no result or image arrays.
"""
function compare_recipe_pair(before::ExperimentRecord,after::ExperimentRecord;
                             pair_indices,basis::Symbol=:pixels,allow_environment_change::Bool=false)
    (pair_indices isa Tuple || pair_indices isa AbstractVector) && length(pair_indices)==2 &&
        all(i->i isa Integer && !(i isa Bool),pair_indices) || _pair_error("pair_indices must contain two explicit integer indices")
    records=(deepcopy(before),deepcopy(after))
    indices=Int[]
    for (record,index) in zip(records,pair_indices)
        _experiment_preflight(record)
        _experiment_validate_environment(record.creation_environment)
        foreach(run->_experiment_run(_experiment_run_data(run),record),record.runs)
        record.recipe.external_preprocess===nothing || _pair_error("pair comparison supports built-in preprocessing only")
        1<=index<=length(record.pairs) || _pair_error("selected pair index is out of bounds")
        push!(indices,Int(index))
    end
    selection=[_pair_selected(record,index) for (record,index) in zip(records,indices)]
    pair_id=_pair_check_selected(selection[1])
    pair_id==_pair_check_selected(selection[2]) || _pair_error("selected ordered image-pair content/dimensions differ")
    conversion=_pair_basis(records[1].recipe,records[2].recipe,basis)
    environment=_experiment_software()
    signature=_experiment_environment_signature(environment)
    allow_environment_change || all(r->_experiment_environment_signature(r.creation_environment)==signature,records) ||
        _pair_error("comparison software differs from recipe creation; explicitly allow_environment_change to rerun both")
    # Verify all selected locators before either computation; no unrelated image reads.
    for side in selection,file in side["files"]
        isfile(file["path"]) && filesize(file["path"])==file["size_bytes"] &&
            _experiment_file_digest(file["path"])==file["sha256"] || _pair_error("selected comparison input changed or is missing")
    end
    results=[_pair_run(record,index) for (record,index) in zip(records,indices)]
    diff=recipe_diff(records[1].recipe,records[2].recipe)
    protected=String[]
    for record in records
        append!(protected,[f["path"] for f in record.input_files])
        append!(protected,record.record_paths);append!(protected,[r.output for r in record.runs])
    end
    common=_pair_measure(results[1],results[2],conversion.factor)
    natives=Dict(side=>_pair_native(result) for (side,result) in zip(("before","after"),results))
    for side in selection,file in side["files"]
        isfile(file["path"]) && filesize(file["path"])==file["size_bytes"] &&
            _experiment_file_digest(file["path"])==file["sha256"] || _pair_error("selected comparison input changed during computation")
    end
    _experiment_environment_signature(_experiment_software())==signature || _pair_error("comparison software changed during computation")
    unavailable=Dict("accuracy"=>"not_evaluated","uncertainty_coverage"=>"not_evaluated",
        "normalized_difference_by_uncertainty"=>"error_correlation_unknown","rejection_events"=>"not_persisted",
        "replacement_history"=>"not_persisted","alternative_peak_history"=>"not_persisted",
        "uncertainty_measurement_association"=>"not_persisted")
    data=Dict{String,Any}("pair_comparison_format_version"=>PAIR_COMPARISON_FORMAT_VERSION,"generated_at_unix_s"=>time(),
        "generator"=>Dict("julia_version"=>environment["julia_version"],"hammerhead_version"=>environment["hammerhead_version"],
            "core_source_sha256"=>environment["core_source_sha256"],"actual_environment_id"=>_experiment_digest(signature)),
        "actual_environment"=>_pair_environment(environment),
        "provenance"=>Dict("verification"=>"selected_ordered_pair_bytes_and_dimensions","ordered_pair_id"=>pair_id,
            "allow_environment_change"=>allow_environment_change,"before"=>selection[1],"after"=>selection[2]),
        "settings_changes"=>[Dict("path"=>c.path,"before"=>_pair_setting(c.before),"after"=>_pair_setting(c.after)) for c in diff],
        "basis"=>Dict("mode"=>String(basis),"quantity"=>conversion.quantity,"unit"=>conversion.unit,"factor"=>conversion.factor,
            "before_scale"=>_pair_scale(records[1].recipe.scale),"after_scale"=>_pair_scale(records[2].recipe.scale),
            "difference_direction"=>"after_minus_before"),"native"=>natives,"common"=>common,
        "unavailable"=>Dict(k=>Dict("available"=>false,"reason_code"=>v) for (k,v) in unavailable),
        "protected_locators"=>sort!(unique!(abspath.(protected))))
    _pair_wrap(data)
end

function Base.show(io::IO,report::RecipePairComparison)
    data=pair_comparison_data(report)
    print(io,"RecipePairComparison(",data["basis"]["mode"],", ",data["common"]["counts"]["valid_both"]," jointly valid nodes)")
end
function Base.show(io::IO,::MIME"text/plain",report::RecipePairComparison)
    data=pair_comparison_data(report)
    common,basis=data["common"],data["basis"]
    print(io,"Selected-pair recipe comparison [",data["provenance"]["ordered_pair_id"][1:12],"…]\n",
        length(data["settings_changes"])," settings changes; ",basis["quantity"]," differences in ",basis["unit"]," (after minus before)\n",
        common["counts"]["nodes"]," exact common nodes; ",common["counts"]["unmasked_both"]," unmasked in both; ",
        common["counts"]["valid_both"]," finite and unflagged in both")
    metric=common["velocity_difference"]
    if metric["available"]
        for component in ("u","v")
            summary=metric["components"][component]
            print(io,"\n",component,": mean difference ",summary["mean"],", RMS difference ",summary["rms"])
        end
    else
        print(io,"\nVelocity differences unavailable: ",metric["reason_code"])
    end
    print(io,"\nStored UQ jointly available at ",common["uq_counts_on_joint_valid"]["available_both"],
        " jointly valid nodes; accuracy and uncertainty coverage not evaluated")
end

"""
    save_pair_comparison(path, report; protected_paths=[]) -> path

Save a validated version-1 TOML comparison snapshot. Validate/serialize before
opening output; reject same-file aliases of both records' known inputs, saved
records/results, and extra protected paths, including after loading a report.
Ordinary report files may be overwritten. This is not atomic publication or
checkpoint state; an I/O failure can leave a partial report.
"""
function save_pair_comparison(path::AbstractString,report::RecipePairComparison;protected_paths=String[])
    data=pair_comparison_data(report)
    all(p->p isa AbstractString,protected_paths) || _pair_error("protected_paths must contain paths")
    protected=[data["protected_locators"]...;abspath.(protected_paths)...]
    any(p->_experiment_alias(path,p),protected) && _pair_error("comparison output aliases a protected input, record or result")
    io=IOBuffer();TOML.print(io,data;sorted=true);contents=String(take!(io))
    open(path,"w") do output;write(output,contents);end
    path
end

"""
    load_pair_comparison(path) -> RecipePairComparison

Load and validate the independent version-1 TOML schema, identities, population
counts, moments and unavailable reasons. This restores a detached past comparison;
it does not reopen sources, reverify file content or rerun either recipe.
"""
load_pair_comparison(path::AbstractString) = _pair_wrap(TOML.parsefile(path))
