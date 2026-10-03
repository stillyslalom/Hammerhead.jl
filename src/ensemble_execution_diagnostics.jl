const ENSEMBLE_EXECUTION_DIAGNOSTICS_FORMAT_VERSION = 1

"""
    EnsemblePassDiagnostics

Immutable scalar observations for one ensemble pooling pass. Exactly one pooled
sweep executes; requested iteration/tolerance settings are ignored, not checked.
Contribution categories describe numerical plane accumulation, not independently
valid vectors or effective sample size. Primary pooled residuals precede predictor
addition, validation and filling; their unit is processing pixels. UQ observations
describe additive statistics and numerical availability before validation/cleanup,
not applicability, calibrated coverage or final-vector attribution.
"""
struct EnsemblePassDiagnostics
    pass_index::Int
    requested_iterations::Int
    requested_tolerance::Float64
    executed_pooling_sweeps::Int
    tolerance_checks::Int
    stop_reason::Symbol
    image_type::String
    processing_size::Tuple{Int,Int}
    grid::NamedTuple
    predictor_present::Bool
    pair_count::Int
    contributions::NamedTuple
    source_support::NamedTuple
    contributor_nodes::NamedTuple
    residual::NamedTuple
    uncertainty::NamedTuple
end

"""
    EnsemblePIVExecutionDiagnostics

Immutable, array-free observations from a completed planar ensemble execution.
No single-pair, iteration-convergence, stationary-flow or independent-sample
claim is made. Measurement-field binding includes axes, displacement, metrics,
UQ, masks/flags and attached scale; parameters and correlation planes are excluded.
Runtime verification state is not persisted. CPU/KA capture is supported;
vendor-device capture is refused explicitly without numerical fallback.
"""
struct EnsemblePIVExecutionDiagnostics
    execution_id::String
    backend::Symbol
    requested_image_type::String
    core_source_sha256::String
    pair_count::Int
    passes::Tuple{Vararg{EnsemblePassDiagnostics}}
    measurement_sha256::String
    _metadata_sha256::String
    _verification::Symbol
end

# One grid's temporary scratch, independent of the number of input pairs. Rows
# are jobs; columns are zero/flat/nonflat/nonfinite/source-gated contributions.
mutable struct _EnsembleObservation
    counts::Matrix{Int}
    residual_u::Any
    residual_v::Any
    no_predictor_pairs::Int
    evaluated_pairs::Int
    disabled_nonfinite_source_pairs::Int
    pairs::Int
end
function _ensemble_observation(grid,::Type{T},n) where T
    _ensemble_mul(length(grid.grid_mask),n)
    _EnsembleObservation(zeros(Int,length(grid.jobs),5),fill(T(NaN),size(grid.grid_mask)),
        fill(T(NaN),size(grid.grid_mask)),0,0,0,0)
end
_ensemble_add(a,b)=_execution_sum(a,b)
function _ensemble_mul(a,b)
    try Base.Checked.checked_mul(a,b) catch err
        err isa OverflowError || rethrow();_execution_error("ensemble diagnostic counts overflow")
    end
end
function _ensemble_source!(o,context,predictor,gate,jobs)
    o.pairs+=1
    if predictor===nothing
        o.no_predictor_pairs+=1
    elseif context.maps===nothing
        o.disabled_nonfinite_source_pairs+=1
    else
        o.evaluated_pairs+=1
    end
    for (j,(gi,gj,_,_)) in enumerate(jobs)
        _source_informative(gate,gi,gj) || (o.counts[j,5]+=1)
    end
end
function _ensemble_plane_category(R)
    lo=typemax(eltype(R));hi=typemin(eltype(R))
    for value in R
        isfinite(value) || return 4
        lo=min(lo,value);hi=max(hi,value)
    end
    lo==hi ? (lo==0 ? 1 : 2) : 3
end
_ensemble_plane!(o,j,category)=(o.counts[j,category]+=1;nothing)
function _ensemble_residual!(o,gi,gj,u,v)
    o.residual_u[gi,gj]=u;o.residual_v[gi,gj]=v;nothing
end
function _ensemble_component_summary(field,mask)
    nonnegative=negative=nonfinite=0
    for i in eachindex(field,mask)
        mask[i] && continue
        v=field[i]
        if !isfinite(v);nonfinite+=1
        elseif v<0;negative+=1
        else;nonnegative+=1
        end
    end
    (finite_nonnegative_count=nonnegative,finite_negative_count=negative,nonfinite_count=nonfinite)
end
function _ensemble_grid_data(grid,p)
    axis(x)=(first=Float64(first(x)),last=Float64(last(x)),count=length(x))
    (x=axis(grid.x),y=axis(grid.y),window_size=p.window_size,search_area_size=p.search_area_size,overlap=p.overlap)
end
function _ensemble_pass_finish(o,k,p,T,imgsize,grid,predictor,force_replace,uacc,uu,uv)
    totals=Tuple(sum(view(o.counts,:,i)) for i in 1:5)
    masked=count(grid.grid_mask);eligible=length(grid.jobs);n=o.pairs
    accumulated=sum(totals[1:4])
    contributions=(window_opportunities=_ensemble_mul(length(grid.grid_mask),n),
        masked_window_pairs=_ensemble_mul(masked,n),source_gated_window_pairs=totals[5],
        accumulated_window_pairs=accumulated,finite_zero_planes=totals[1],finite_flat_nonzero_planes=totals[2],
        finite_nonflat_planes=totals[3],nonfinite_planes=totals[4])
    # Counts describe nonzero FINITE numerical planes; they do not certify peaks.
    nonzero=Int[o.counts[j,2]+o.counts[j,3] for j in 1:eligible]
    nodes=(eligible_count=eligible,masked_count=masked,zero_finite_nonzero_count=count(==(0),nonzero),
        some_finite_nonzero_count=count(v->0<v<n,nonzero),all_finite_nonzero_count=count(==(n),nonzero),
        minimum_finite_nonzero_contributions=isempty(nonzero) ? nothing : minimum(nonzero),
        maximum_finite_nonzero_contributions=isempty(nonzero) ? nothing : maximum(nonzero),
        nodes_with_nonfinite_plane=count(j->o.counts[j,4]>0,1:eligible))
    support=(no_predictor_pairs=o.no_predictor_pairs,evaluated_pairs=o.evaluated_pairs,
        disabled_nonfinite_source_pairs=o.disabled_nonfinite_source_pairs,
        convention="original_sampled_4x4_stencil_union_not_nonlocal_bspline_support")
    evaluated=uacc!==nothing
    reason=!p.uncertainty ? "disabled" : force_replace ? "intermediate_pass" : eligible==0 ? "no_eligible_windows" : "pooled_statistics"
    uncertainty=(requested=p.uncertainty,evaluated=evaluated,reason=reason,
        admitted_window_pair_updates=evaluated ? accumulated : 0,
        source_gated_window_pairs=evaluated ? totals[5] : 0,
        value_basis="pooled_statistics_before_validation_and_availability_cleanup",unit="px",
        common_displacement_assumption_checked=false,independent_sample_count_available=false,
        u=_ensemble_component_summary(uu,grid.grid_mask),v=_ensemble_component_summary(uv,grid.grid_mask))
    EnsemblePassDiagnostics(k,p.max_iterations,p.convergence_tol,1,0,:pooled_once_iteration_settings_ignored,
        string(T),imgsize,_ensemble_grid_data(grid,p),predictor!==nothing,n,contributions,support,nodes,
        _execution_residual(o.residual_u,o.residual_v,grid.grid_mask,predictor!==nothing),uncertainty)
end

function _ensemble_payload(d)
    data=Dict{String,Any}(String(k)=>_execution_primitive(getfield(d,k)) for k in
        (:execution_id,:backend,:requested_image_type,:core_source_sha256,:pair_count,:measurement_sha256))
    data["passes"]=[Dict{String,Any}(String(k)=>_execution_primitive(getfield(p,k)) for k in fieldnames(EnsemblePassDiagnostics)) for p in d.passes]
    data["ensemble_diagnostics_format_version"]=ENSEMBLE_EXECUTION_DIAGNOSTICS_FORMAT_VERSION
    data
end
_ensemble_sum(values)=foldl(_ensemble_add,values;init=0)
function _ensemble_counts(data,keys,what)
    _experiment_keys(data,keys,what)
    all(k->_execution_count(data[k]),keys) || _execution_error("invalid $what counts")
end
function _ensemble_grid_validate(grid,size,T)
    _experiment_keys(grid,["x","y","window_size","search_area_size","overlap"],"ensemble grid")
    for key in ("window_size","search_area_size","overlap")
        v=grid[key]
        v isa AbstractVector && length(v)==2 && all(x->x isa Int && x>=0,v) || _execution_error("invalid ensemble grid dimensions")
    end
    for (d,key) in enumerate(("y","x"))
        w,s,overlap=grid["window_size"][d],grid["search_area_size"][d],grid["overlap"][d]
        1<=w<=s<=size[d] && 0<=overlap<w && iseven(s-w) || _execution_error("invalid ensemble window geometry")
        margin=div(s-w,2);stride=w-overlap
        n=fld(size[d]-w-2margin,stride)+1
        axis=grid[key];_experiment_keys(axis,["first","last","count"],"ensemble axis")
        axis["count"]===n && all(k->axis[k] isa Float64 && isfinite(axis[k]),("first","last")) || _execution_error("invalid ensemble axis")
        a=Float64(T(1+margin+(w-1)/2));b=Float64(T(1+margin+(n-1)*stride+(w-1)/2))
        axis["first"]==a && axis["last"]==b || _execution_error("ensemble grid attribution differs from pass geometry")
    end
    _ensemble_mul(grid["x"]["count"],grid["y"]["count"])
end
function _ensemble_validate(data)
    _experiment_keys(data,["execution_id","backend","requested_image_type","core_source_sha256","pair_count","passes","measurement_sha256","ensemble_diagnostics_format_version"],"ensemble diagnostics")
    data["ensemble_diagnostics_format_version"]===1 || _execution_error("unsupported ensemble diagnostics version")
    data["execution_id"] isa String && try UUIDs.UUID(data["execution_id"]);true catch;false end || _execution_error("invalid ensemble execution ID")
    data["backend"] in ("cpu","ka") && data["requested_image_type"] in ("Float32","Float64") || _execution_error("unsupported ensemble diagnostic backend/precision")
    all(k->_experiment_hash(data[k]),("core_source_sha256","measurement_sha256")) || _execution_error("invalid ensemble identity")
    n=data["pair_count"];n isa Int && n>0 || _execution_error("invalid ensemble pair count")
    passes=data["passes"];passes isa AbstractVector && !isempty(passes) || _execution_error("missing ensemble passes")
    for (i,p) in enumerate(passes)
        _experiment_keys(p,String.(fieldnames(EnsemblePassDiagnostics)),"ensemble pass")
        p["pass_index"]===i && p["pair_count"]===n && p["requested_iterations"] isa Int && p["requested_iterations"]>0 || _execution_error("invalid ensemble pass association")
        tol=p["requested_tolerance"]
        tol isa Float64 && !isnan(tol) && tol>=0 && p["executed_pooling_sweeps"]===1 && p["tolerance_checks"]===0 &&
            p["stop_reason"]=="pooled_once_iteration_settings_ignored" || _execution_error("ensemble iteration semantics changed")
        p["image_type"] in ("Float32","Float64") && p["predictor_present"] isa Bool &&
            (i!=1 || !p["predictor_present"]) || _execution_error("invalid ensemble predictor/precision")
        shape=p["processing_size"]
        shape isa AbstractVector && length(shape)==2 && all(x->x isa Int && x>0,shape) || _execution_error("invalid ensemble processing size")
        T=p["image_type"]=="Float32" ? Float32 : Float64
        nodes=_ensemble_grid_validate(p["grid"],shape,T)
        c=p["contributions"]
        _ensemble_counts(c,["window_opportunities","masked_window_pairs","source_gated_window_pairs","accumulated_window_pairs","finite_zero_planes","finite_flat_nonzero_planes","finite_nonflat_planes","nonfinite_planes"],"ensemble contribution")
        c["window_opportunities"]==_ensemble_mul(nodes,n) &&
            c["window_opportunities"]==_ensemble_sum(c[k] for k in ("masked_window_pairs","source_gated_window_pairs","accumulated_window_pairs")) &&
            c["accumulated_window_pairs"]==_ensemble_sum(c[k] for k in ("finite_zero_planes","finite_flat_nonzero_planes","finite_nonflat_planes","nonfinite_planes")) || _execution_error("ensemble contribution partition disagrees")
        support=p["source_support"]
        _experiment_keys(support,["no_predictor_pairs","evaluated_pairs","disabled_nonfinite_source_pairs","convention"],"ensemble source support")
        all(k->_execution_count(support[k]),("no_predictor_pairs","evaluated_pairs","disabled_nonfinite_source_pairs")) &&
            _ensemble_sum(support[k] for k in ("no_predictor_pairs","evaluated_pairs","disabled_nonfinite_source_pairs"))==n &&
            support["no_predictor_pairs"]==(p["predictor_present"] ? 0 : n) && support["convention"]=="original_sampled_4x4_stencil_union_not_nonlocal_bspline_support" || _execution_error("invalid source support observations")
        g=p["contributor_nodes"]
        _experiment_keys(g,["eligible_count","masked_count","zero_finite_nonzero_count","some_finite_nonzero_count","all_finite_nonzero_count","minimum_finite_nonzero_contributions","maximum_finite_nonzero_contributions","nodes_with_nonfinite_plane"],"ensemble node contributions")
        all(k->_execution_count(g[k]),("eligible_count","masked_count","zero_finite_nonzero_count","some_finite_nonzero_count","all_finite_nonzero_count","nodes_with_nonfinite_plane")) || _execution_error("invalid node contribution counts")
        eligible=g["eligible_count"]
        _ensemble_add(eligible,g["masked_count"])==nodes && c["masked_window_pairs"]==_ensemble_mul(g["masked_count"],n) &&
            _ensemble_sum(g[k] for k in ("zero_finite_nonzero_count","some_finite_nonzero_count","all_finite_nonzero_count"))==eligible &&
            g["nodes_with_nonfinite_plane"]<=eligible || _execution_error("ensemble node partition disagrees")
        minimum,maximum=g["minimum_finite_nonzero_contributions"],g["maximum_finite_nonzero_contributions"]
        (eligible==0 ? minimum===nothing && maximum===nothing : minimum isa Int && maximum isa Int && 0<=minimum<=maximum<=n) || _execution_error("invalid contributor bounds")
        nonzero=_ensemble_add(c["finite_flat_nonzero_planes"],c["finite_nonflat_planes"])
        if eligible>0
            _ensemble_mul(eligible,minimum)<=nonzero<=_ensemble_mul(eligible,maximum) || _execution_error("contributor range disagrees with totals")
            (g["zero_finite_nonzero_count"]>0)==(minimum==0) && (g["all_finite_nonzero_count"]>0)==(maximum==n) || _execution_error("contributor endpoints disagree")
            lower=_ensemble_add(_ensemble_mul(g["all_finite_nonzero_count"],n),g["some_finite_nonzero_count"])
            upper=_ensemble_add(_ensemble_mul(g["all_finite_nonzero_count"],n),_ensemble_mul(g["some_finite_nonzero_count"],n-1))
            lower<=nonzero<=upper || _execution_error("contributor populations disagree")
        end
        g["nodes_with_nonfinite_plane"]<=c["nonfinite_planes"]<=_ensemble_mul(g["nodes_with_nonfinite_plane"],n) || _execution_error("nonfinite contributor populations disagree")
        c["source_gated_window_pairs"]<=_ensemble_mul(support["evaluated_pairs"],eligible) || _execution_error("source gate skips lack evaluated support")
        r=p["residual"]
        _experiment_keys(r,["finite_count","nonfinite_count","masked_count","mean_magnitude","rms_magnitude","maximum_magnitude","predictor_present","unit","value_basis"],"pooled primary residual")
        all(k->_execution_count(r[k]),("finite_count","nonfinite_count","masked_count")) &&
            _ensemble_add(r["finite_count"],r["nonfinite_count"])==eligible && r["masked_count"]==g["masked_count"] &&
            r["predictor_present"] isa Bool && r["predictor_present"]==p["predictor_present"] && r["unit"]=="px" &&
            r["value_basis"]=="primary_peak_before_validation" || _execution_error("invalid pooled residual semantics/support")
        stats=(r["mean_magnitude"],r["rms_magnitude"],r["maximum_magnitude"])
        (r["finite_count"]==0 ? all(isnothing,stats) : all(_execution_finite,stats)) || _execution_error("invalid pooled residual summary")
        q=p["uncertainty"]
        _experiment_keys(q,["requested","evaluated","reason","admitted_window_pair_updates","source_gated_window_pairs","value_basis","unit","common_displacement_assumption_checked","independent_sample_count_available","u","v"],"pooled UQ")
        q["requested"] isa Bool && q["evaluated"] isa Bool && q["common_displacement_assumption_checked"]===false && q["independent_sample_count_available"]===false &&
            q["value_basis"]=="pooled_statistics_before_validation_and_availability_cleanup" && q["unit"]=="px" || _execution_error("invalid pooled UQ semantics")
        reason=!q["requested"] ? "disabled" : i<length(passes) ? "intermediate_pass" : eligible==0 ? "no_eligible_windows" : "pooled_statistics"
        q["reason"]==reason && q["evaluated"]==(reason=="pooled_statistics") &&
            q["admitted_window_pair_updates"] isa Int && q["admitted_window_pair_updates"]==(q["evaluated"] ? c["accumulated_window_pairs"] : 0) &&
            q["source_gated_window_pairs"] isa Int && q["source_gated_window_pairs"]==(q["evaluated"] ? c["source_gated_window_pairs"] : 0) || _execution_error("invalid pooled UQ treatment")
        for component in (q["u"],q["v"])
            _ensemble_counts(component,["finite_nonnegative_count","finite_negative_count","nonfinite_count"],"UQ component")
            _ensemble_sum(values(component))==eligible || _execution_error("UQ component partition disagrees")
            q["evaluated"] || component["nonfinite_count"]==eligible || _execution_error("unevaluated UQ contains numeric estimates")
        end
    end
    data
end

_ensemble_named(x::AbstractDict)=NamedTuple{Tuple(Symbol.(sort!(collect(keys(x)))))}(Tuple(_ensemble_named(x[k]) for k in sort!(collect(keys(x)))))
_ensemble_named(x::AbstractVector)=Tuple(_ensemble_named(v) for v in x)
_ensemble_named(x)=x
function _ensemble_decode(data,sha,state)
    _ensemble_validate(data)
    _experiment_hash(sha) && _experiment_digest(data)==sha || _execution_error("ensemble diagnostic metadata changed")
    passes=Tuple(EnsemblePassDiagnostics(p["pass_index"],p["requested_iterations"],p["requested_tolerance"],1,0,Symbol(p["stop_reason"]),
        p["image_type"],Tuple(p["processing_size"]),_ensemble_named(p["grid"]),p["predictor_present"],p["pair_count"],
        _ensemble_named(p["contributions"]),_ensemble_named(p["source_support"]),_ensemble_named(p["contributor_nodes"]),
        _ensemble_named(p["residual"]),_ensemble_named(p["uncertainty"])) for p in data["passes"])
    EnsemblePIVExecutionDiagnostics(data["execution_id"],Symbol(data["backend"]),data["requested_image_type"],data["core_source_sha256"],
        data["pair_count"],passes,data["measurement_sha256"],sha,state)
end
function _ensemble_checked(d)
    data=_ensemble_payload(d);_ensemble_validate(data)
    _experiment_digest(data)==d._metadata_sha256 || _execution_error("ensemble diagnostics changed")
    d._verification in (:captured_measurement_fields_verified,:metadata_only,:read_measurement_fields_verified) || _execution_error("invalid runtime inspection state")
    data
end
function _ensemble_check_result(d,result)
    data=_ensemble_checked(d)
    result isa PIVResult || _execution_error("ensemble diagnostics require raw PIVResult")
    p=last(data["passes"]);dims=(p["grid"]["y"]["count"],p["grid"]["x"]["count"])
    T=p["image_type"]=="Float32" ? Float32 : Float64
    result isa PIVResult{T} && length(result.x)==dims[2] && length(result.y)==dims[1] &&
        all(a->size(a)==dims,(result.u,result.v,result.peak_ratio,result.correlation_moment,result.uncertainty_u,result.uncertainty_v,result.mask,result.outliers)) || _execution_error("ensemble result precision/shape differs")
    for (d,key,axis) in ((1,"y",result.y),(2,"x",result.x))
        w,s,overlap=p["grid"]["window_size"][d],p["grid"]["search_area_size"][d],p["grid"]["overlap"][d]
        margin=div(s-w,2)
        all(i->isfinite(axis[i]) && axis[i]==T(1+margin+(i-1)*(w-overlap)+(w-1)/2),eachindex(axis)) || _execution_error("ensemble result attribution differs")
    end
    count(result.mask)==p["contributor_nodes"]["masked_count"] || _execution_error("ensemble final mask support differs")
    _history_result_digest(result)==d.measurement_sha256 || _execution_error("ensemble measurement fields changed")
    nothing
end
function _ensemble_finish(reports,backend,image_type,n,result,source_before)
    _experiment_software()["core_source_sha256"]==source_before || _execution_error("core source changed during ensemble capture")
    d=EnsemblePIVExecutionDiagnostics(string(UUIDs.uuid4()),backend,string(image_type),source_before,n,Tuple(reports),
        _history_result_digest(result),"",:captured_measurement_fields_verified)
    data=_ensemble_payload(d)
    packet=_ensemble_decode(data,_experiment_digest(data),:captured_measurement_fields_verified)
    _ensemble_check_result(packet,result);packet
end

"""
    execution_diagnostics_data(d::EnsemblePIVExecutionDiagnostics; result=nothing)

Return detached scalar ensemble metadata with explicit runtime inspection state.
An optional supplied RAW result verifies measurement fields and geometry without
reading another payload. No result is retained. Pixel residual/UQ observations
are independent of attached scale; parameters/planes and input byte identities
are not verified. Read-time checks do not certify scientific assumptions.
"""
function execution_diagnostics_data(d::EnsemblePIVExecutionDiagnostics;result=nothing)
    data=_ensemble_checked(d);result===nothing || _ensemble_check_result(d,result)
    state=result===nothing ? d._verification : :supplied_measurement_fields_verified
    data["verification"]=Dict{String,Any}("inspection_state"=>String(state),"measurement_field_binding_checked"=>state!==:metadata_only,
        "full_result_serialization"=>false,"source_inputs"=>false,"common_displacement_assumption"=>false)
    data
end
_ensemble_execution_key(key)="ensemble_execution_diagnostics/"*last(split(key,'/'))
function _write_ensemble_execution_diagnostics(file,key,d,result)
    _ensemble_check_result(d,result)
    if haskey(file,"ensemble_execution_diagnostics_format_version")
        file["ensemble_execution_diagnostics_format_version"]===1 || _execution_error("unsupported existing ensemble diagnostics marker")
    else
        file["ensemble_execution_diagnostics_format_version"]=1
    end
    file[_ensemble_execution_key(key)]=Dict{String,Any}("result_key"=>key,"diagnostics"=>_ensemble_checked(d),"diagnostics_sha256"=>d._metadata_sha256)
end

"""
    load_ensemble_execution_diagnostics(index::ResultFile, i; verify_result=false)
    load_ensemble_execution_diagnostics(path, i=1; verify_result=false)

Read a separate version-1 ensemble native companion, or `nothing` when absent.
Default metadata-only access validates schema/counts/digest and result-key linkage.
`verify_result=true` loads/discards one selected raw planar result and checks its
measurement fields and geometry. The packet retains no payload or file handle.
Native format/layouts are unchanged; bare saves cannot invent pooled execution
history. Concurrent writers and calibration/input authenticity are not verified.
"""
function load_ensemble_execution_diagnostics(index::ResultFile,i::Integer;verify_result::Bool=false)
    checkbounds(index,i);_check_result_file(index)
    key="results/"*index.entry_keys[i]
    packet=jldopen(index.path,"r") do file
        _check_results_format(file,index.path)
        if !haskey(file,"ensemble_execution_diagnostics_format_version")
            haskey(file,"ensemble_execution_diagnostics") && _execution_error("ensemble metadata lacks version")
            return nothing
        end
        file["ensemble_execution_diagnostics_format_version"]===1 || _execution_error("unsupported ensemble diagnostics marker")
        haskey(file,_ensemble_execution_key(key)) || return nothing
        entry=file[_ensemble_execution_key(key)]
        _experiment_keys(entry,["result_key","diagnostics","diagnostics_sha256"],"ensemble execution entry")
        entry["result_key"]==key || _execution_error("ensemble diagnostics/result key mismatch")
        _ensemble_decode(entry["diagnostics"],entry["diagnostics_sha256"],:metadata_only)
    end
    if packet!==nothing && verify_result
        result=index[i];_ensemble_check_result(packet,result);result=nothing
        packet=_ensemble_decode(_ensemble_payload(packet),packet._metadata_sha256,:read_measurement_fields_verified)
    end
    _check_result_file(index);packet
end
load_ensemble_execution_diagnostics(path::AbstractString,i::Integer=1;kwargs...)=load_ensemble_execution_diagnostics(ResultFile(path),i;kwargs...)
Base.show(io::IO,d::EnsemblePIVExecutionDiagnostics)=print(io,"EnsemblePIVExecutionDiagnostics(",d.pair_count," pairs, ",length(d.passes)," pooled passes)")
function Base.show(io::IO,::MIME"text/plain",d::EnsemblePIVExecutionDiagnostics)
    _ensemble_checked(d);show(io,d)
    for p in d.passes
        print(io,"\nPass ",p.pass_index,": one pooled sweep; requested ",p.requested_iterations,
            " iterations and tolerance ignored; ",p.contributions.accumulated_window_pairs," accumulated window/pair planes")
    end
end

function _ensemble_output_inputs(pairs)
    paths=String[]
    for pair in pairs,frame in pair
        frame isa AbstractString && push!(paths,_artifact_local_path(frame))
        frame isa FrameRef && frame.source isa TIFFStack && push!(paths,_artifact_local_path(frame.source.path))
    end
    unique(paths)
end
function _ensemble_output_guard(path,inputs)
    destination=_artifact_local_path(path)
    any(source->_artifact_alias(destination,source),inputs) && _execution_error("ensemble output aliases selected input")
    destination
end
function _ensemble_options(pairs,backend,image_type,on_diagnostics,output,record_diagnostics)
    on_diagnostics===nothing || on_diagnostics isa Function || _execution_error("on_diagnostics must be a function or nothing")
    output===nothing || output isa AbstractString || _execution_error("ensemble output must be a path or nothing")
    record_diagnostics isa Bool || _execution_error("record_diagnostics must be Bool")
    record_diagnostics && output===nothing && _execution_error("record_diagnostics requires output")
    capture=on_diagnostics!==nothing || record_diagnostics
    if capture || output!==nothing
        isempty(pairs) && _execution_error("ensemble pairs must not be empty")
        all(p->p isa Union{Tuple,AbstractVector,FramePair} && length(p)==2,pairs) ||
            _execution_error("ensemble capture/output requires two-frame tuple/vector/FramePair containers")
    end
    if capture
        backend in (:cpu,:ka) || _execution_error("ensemble diagnostic capture supports :cpu/:ka; vendor backends are unsupported without fallback")
        image_type in (Float32,Float64) || _execution_error("ensemble diagnostic capture requires Float32/64")
    end
    inputs=output===nothing ? String[] : _ensemble_output_inputs(pairs)
    output===nothing || _ensemble_output_guard(output,inputs)
    (;capture,inputs)
end
