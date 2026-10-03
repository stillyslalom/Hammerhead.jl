# Opt-in bench investigation; deliberately no production estimator changes.
module DiagnosticUncertainty

using Hammerhead, Statistics, TOML, Dates
include("validation_uncertainty.jl")
const U = ValidationUncertainty
const V = U.V
const ROOT = U.ROOT
const SCHEMA = "hammerhead-uncertainty-diagnostic-1"
const MARKER = "Hammerhead uncertainty diagnostic report"
const DEFAULT_SEEDS = (7321, 7322, 7323)
const ZERO_CLASSIFICATIONS = ("zero_covariance", "exact_covariance_cancellation", "negative_variance_clamp",
    "rounded_side_perturbation", "rounded_log_difference", "output_type_underflow")

"""Trace the existing random estimator without changing its threshold or clamp.
Classification explains numerical outcomes, not physical error sources.
"""
function trace_component(stats; output_type::Type{T}=Float64) where {T<:AbstractFloat}
    length(stats) == Hammerhead.UQ_NSTATS || throw(ArgumentError("unexpected statistics layout"))
    s = Float64.(stats)
    C0, Cp, Cm, S00 = s[1:4]
    variance = S00
    threshold = 0.05 * S00
    stopped = false
    rings = Dict{String,Any}[]
    offsets = Dict{String,Any}[]
    for (k, (dr, dc)) in enumerate(Hammerhead.UQ_OFFSETS)
        push!(offsets, Dict("row_offset"=>dr, "column_offset"=>dc, "sum"=>s[4+k]))
    end
    for (radius, ring) in enumerate(Hammerhead.UQ_RINGS)
        peak = maximum(s[4+k] for k in ring)
        included = !stopped && !(peak < threshold)
        stopped |= !included
        contribution = 0.0
        for k in ring
            contribution += 2s[4+k]
            included && (variance += 2s[4+k])
        end
        push!(rings, Dict("radius"=>radius, "maximum"=>peak, "signed_contribution"=>contribution,
            "included"=>included, "variance_after"=>variance))
    end
    delta = sqrt(max(variance, 0.0))
    side = (Cp + Cm) / 2
    lo, hi = side - delta / 2, side + delta / 2
    denom = NaN
    value = NaN
    reason = "positive_sigma"
    if !all(isfinite, s)
        reason = "nonfinite_statistics"
    end
    if C0 > 0 && lo > 0
        denom = 2log(C0) - log(hi) - log(lo)
        denom > 0 && (value = (log(hi) - log(lo)) / (2denom))
    end
    sigma = T(value)
    if reason != "nonfinite_statistics"
        if !isfinite(variance)
            reason = "nonfinite_variance_arithmetic"
        elseif !(C0 > 0)
            reason = "nonpositive_C0"
        elseif !(lo > 0)
            reason = "nonpositive_perturbed_side"
        elseif !(denom > 0) || !isfinite(denom)
            reason = "invalid_peak_curvature"
        elseif !isfinite(sigma)
            reason = "nonfinite_sigma_arithmetic"
        elseif sigma == 0
            reason = variance < 0 ? "negative_variance_clamp" :
                variance == 0 ? (all(iszero,s[4:end]) ? "zero_covariance" : "exact_covariance_cancellation") :
                lo == hi ? "rounded_side_perturbation" :
                value == 0 ? "rounded_log_difference" : "output_type_underflow"
        end
    end
    # Equation 4 uses the unsymmetrized raw side sums. It is not the measured
    # residual of the overlap-normalized/magnitude FFT correlation plane.
    residual = NaN
    if all(x -> isfinite(x) && x > 0, (C0, Cp, Cm))
        d = 2log(C0) - log(Cp) - log(Cm)
        d > 0 && (residual = (log(Cp) - log(Cm)) / (2d))
    end
    Dict{String,Any}("classification"=>reason, "sigma"=>Float64(sigma), "sigma_before_conversion"=>value,
        "C0"=>C0, "Cplus"=>Cp, "Cminus"=>Cm, "S00"=>S00, "threshold"=>threshold,
        "pre_clamp_variance"=>variance, "variance_was_negative"=>variance < 0,
        "sigma_delta_C"=>delta, "perturbed_side_low"=>lo, "perturbed_side_high"=>hi,
        "peak_curvature_denominator"=>denom, "raw_eq4_residual"=>residual,
        "rings"=>rings, "covariance_offsets"=>offsets)
end

# Welford centered moments: overflow invalidates the summary but never drops
# attempted errors. This diagnostic does not silently substitute finite subsets.
mutable struct CenteredMoment
    count::Int
    finite_count::Int
    mean::Float64
    m2::Float64
end
CenteredMoment() = CenteredMoment(0, 0, 0.0, 0.0)
function add!(m::CenteredMoment, value)
    m.count += 1
    x = Float64(value)
    isfinite(x) || return m
    m.finite_count += 1
    delta = x - m.mean
    m.mean += delta / m.finite_count
    m.m2 += delta * (x - m.mean)
    m
end
function merge!(a::CenteredMoment, b::CenteredMoment)
    if b.finite_count > 0
        n = a.finite_count + b.finite_count
        if a.finite_count == 0
            a.mean, a.m2 = b.mean, b.m2
        else
            delta = b.mean - a.mean
            a.mean += delta * (b.finite_count / n)
            a.m2 += b.m2 + delta^2 * (a.finite_count * (b.finite_count / n))
        end
        a.finite_count = n
    end
    a.count += b.count
    a
end
function summary(m::CenteredMoment)
    available = m.count > 0 && m.finite_count == m.count && isfinite(m.mean) &&
        isfinite(m.m2) && m.m2 >= 0
    out = Dict{String,Any}("count"=>m.count, "finite_count"=>m.finite_count, "available"=>available)
    if available
        out["mean"] = m.mean
        out["centered_rms"] = sqrt(m.m2 / m.count)
    else
        out["reason"] = m.count == 0 ? "empty_population" : "nonfinite_centered_arithmetic"
    end
    out
end

function check_passes(passes)
    length(passes) >= 2 || throw(ArgumentError("retained-window audit requires CPU multipass deformation"))
    for p in passes
        p isa PIVParameters || throw(ArgumentError("passes must contain PIVParameters"))
        p.correlation_method === :cross && p.search_area_size == p.window_size ||
            throw(ArgumentError("audit supports cross correlation with equal interrogation/search sizes"))
        p.n_peaks == 1 && !p.replace_outliers || throw(ArgumentError("audit requires primary-only unreplaced output"))
        p.subpixel_method === :gauss3 || throw(ArgumentError("first audit slice requires gauss3"))
        isempty(p.validation) || throw(ArgumentError("custom validator recipes are outside this audit"))
    end
    last(passes).uncertainty || throw(ArgumentError("final uncertainty must be enabled"))
    nothing
end

# Same denominator contract as ValidationUncertainty.metrics, without treating
# requested iterations as a certificate of primary origin. Origin is separately
# checked against the captured final measurement history by audit_scene.
function population(result, truth)
    p = U.Population()
    centered = Dict(k=>CenteredMoment() for k in ("u", "v"))
    subset = Dict(k=>CenteredMoment() for k in ("u", "v"))
    errors = Dict(k=>Float64[] for k in ("u", "v"))
    sigmas = Dict(k=>Float64[] for k in ("u", "v"))
    indices = CartesianIndex{2}[]
    zvalues = Dict(k=>Float64[] for k in ("u", "v"))
    dims = (length(result.y), length(result.x))
    all(a->size(a)==dims, (result.u,result.v,result.mask,result.outliers,
        result.uncertainty_u,result.uncertainty_v)) || throw(ArgumentError("inconsistent result dimensions"))
    all(isfinite,result.x) && all(isfinite,result.y) || throw(ArgumentError("nonfinite coordinates"))
    for index in CartesianIndices(result.u)
        p.counts["grid_nodes"] += 1
        if result.mask[index]
            p.counts["masked"] += 1
            continue
        end
        p.counts["unmasked"] += 1
        finite = isfinite(result.u[index]) && isfinite(result.v[index])
        result.outliers[index] && (p.counts["flagged_unmasked"] += 1)
        !finite && (p.counts["nonfinite_output_unmasked"] += 1)
        key = result.outliers[index] ? (finite ? "flagged_finite" : "flagged_nonfinite") :
            finite ? "primary_valid" : "unflagged_nonfinite"
        p.counts[key] += 1
        key == "primary_valid" || continue
        reference = truth(result.x[index[2]],result.y[index[1]])
        length(reference)==2 && all(isfinite,reference) || throw(ArgumentError("truth must provide finite components"))
        push!(indices,index)
        for (k, output, sigma, ref) in (("u",result.u[index],result.uncertainty_u[index],reference[1]),
                                      ("v",result.v[index],result.uncertainty_v[index],reference[2]))
            error = Float64(output)-Float64(ref)
            push!(errors[k],error); push!(sigmas[k],Float64(sigma))
            U.component!(p.components[k],error,Float64(sigma),zvalues[k])
            add!(centered[k],error)
            isfinite(sigma) && sigma >= 0 && add!(subset[k],error)
        end
    end
    data = U.summary(p)
    for k in ("u","v")
        data["components"][k]["normalized_error_quantiles"] = U.quantiles(zvalues[k],p.components[k].counts["sigma_positive"])
    end
    (; data, p, centered, subset, errors, sigmas, indices)
end

function component_summary(p::U.ComponentPopulation)
    shell = U.Population()
    shell.components["u"] = p
    U.summary(shell)["components"]["u"]
end

function supplementary(errors, sigmas, residuals, raw_residuals)
    length(errors)==length(sigmas)==length(residuals)==length(raw_residuals) ||
        throw(ArgumentError("component populations differ"))
    centered = CenteredMoment()
    for (error,sigma) in zip(errors,sigmas)
        isfinite(sigma) && sigma >= 0 && add!(centered,error)
    end
    center = summary(centered)
    centered_population = U.ComponentPopulation()
    measured_population = U.ComponentPopulation()
    raw_population = U.ComponentPopulation()
    for (e,s,r,b) in zip(errors,sigmas,residuals,raw_residuals)
        stored_available = isfinite(s) && s >= 0
        centered_error = center["available"] ? e-center["mean"] : NaN
        U.component!(centered_population,centered_error,s,Float64[])
        U.component!(measured_population,e,stored_available && isfinite(r) ? hypot(s,r) : NaN,Float64[])
        U.component!(raw_population,e,stored_available && isfinite(b) ? hypot(s,b) : NaN,Float64[])
    end
    Dict("centered_error_diagnostic"=>Dict("scope"=>"same-seed UQ-subset mean removed; in-sample sensitivity, not corrected accuracy",
            "centering"=>center,"metrics"=>component_summary(centered_population)),
        "primary_residual_quadrature_diagnostic"=>Dict("scope"=>"hypot(stored_random_sigma, actual_primary_residual); descriptive only",
            "metrics"=>component_summary(measured_population)),
        "raw_eq4_quadrature_diagnostic"=>Dict("scope"=>"hypot(stored_random_sigma, raw_side_sum_eq4); not FFT-plane residual or production total uncertainty",
            "metrics"=>component_summary(raw_population)))
end

"""Audit one trusted synthetic scene once; no warmup or performance sampling.
The final retained CPU windows and history must reproduce primary measurements.
Returned metadata has no image/result/history payloads.
"""
function audit_scene(condition,seed; passes=U.controlled_passes(condition.window),size=128)
    scene = V.synthetic_scene(;condition.settings...,seed,size)
    audit_pair(scene,condition,seed;passes)
end

# Bench-only supplied-pair entry point. The callback receives owned statistics;
# audit_scene's scientific row and primary population remain unchanged.
function audit_pair(scene,condition,seed;passes=U.controlled_passes(condition.window),on_component=nothing)
    check_passes(passes)
    scene.a isa Matrix{Float64} && scene.b isa Matrix{Float64} && size(scene.a)==size(scene.b) ||
        throw(ArgumentError("audit_pair requires equally sized Float64 CPU matrices"))
    ws = piv_workspace(;backend=:cpu)
    diagnostics, history = Ref{Any}(nothing),Ref{Any}(nothing)
    result = run_piv(scene.a,scene.b,passes;backend=:cpu,threaded=false,workspace=ws,
        on_diagnostics=d->(diagnostics[]=d),on_measurement_history=h->(history[]=h))
    verify_measurement_history(history[],result)
    hd = measurement_history_data(history[])
    observed = population(result,scene.truth)
    ws.warpA isa Matrix && ws.warpB isa Matrix || error("CPU final warped pair is unavailable")
    params = last(passes)
    grid = Hammerhead.pass_grid(Float64,Base.size(scene.a),params,nothing,0.5)
    result.x==grid.x && result.y==grid.y || error("retained-window grid mismatch")
    apod = Hammerhead.apodization_window(Float64,params.window_size,params.apodization)
    scratch = Hammerhead.uncertainty_scratch(Float64,params.window_size)
    stats = zeros(2,Hammerhead.UQ_NSTATS)
    classified = Dict(k=>Dict{String,Int}() for k in ("u","v"))
    negative = Dict(k=>0 for k in ("u","v"))
    sample_counts = Dict(k=>Dict{String,Int}() for k in ("u","v"))
    samples = Dict(k=>Dict{String,Any}[] for k in ("u","v"))
    residuals = Dict(k=>Float64[] for k in ("u","v"))
    raw_residuals = Dict(k=>Float64[] for k in ("u","v"))
    primary_indices = Set(observed.indices)
    reproduction_count = 0
    wr,wc = params.window_size
    # Match the column-major population traversal, independently of job order.
    origins = Dict(CartesianIndex(gi,gj)=>(rs,cs) for (gi,gj,rs,cs) in grid.jobs)
    for index in observed.indices
        hd["origin_codes"][Int(hd["final_origin"][index])+1]=="primary" || error("selected output is not primary")
        hd["primary_u"][index]==result.u[index] && hd["primary_v"][index]==result.v[index] ||
            error("selected output differs from captured primary")
        rs,cs = origins[index]
        A = @view ws.warpA[rs:rs+wr-1,cs:cs+wc-1]
        B = @view ws.warpB[rs:rs+wr-1,cs:cs+wc-1]
        fill!(stats,0)
        Hammerhead.accumulate_uncertainty!(stats,scratch,A,B,nothing,apod)
        for (component,k,stored) in ((1,"u",result.uncertainty_u[index]),(2,"v",result.uncertainty_v[index]))
            s = view(stats,component,:)
            traced = trace_component(s)
            isequal(traced["sigma"],Hammerhead.finalize_uncertainty(Float64,s)) || error("trace differs from production finalizer")
            isequal(traced["sigma"],stored) || error("retained windows do not reproduce stored UQ")
            reason = traced["classification"]
            classified[k][reason] = get(classified[k],reason,0)+1
            traced["variance_was_negative"] && (negative[k]+=1)
            push!(residuals[k],hd["primary_residual_"*k][index])
            push!(raw_residuals[k],traced["raw_eq4_residual"])
            if on_component!==nothing
                on_component((;index,component=k,statistics=copy(s),sigma=stored,
                    error=observed.errors[k][reproduction_count+1],classification=reason,
                    primary_residual=last(residuals[k])))
            end
            if get(sample_counts[k],reason,0)<2
                sample_counts[k][reason] = get(sample_counts[k],reason,0)+1
                push!(samples[k],Dict("row"=>index[1],"column"=>index[2],"x"=>result.x[index[2]],
                    "y"=>result.y[index[1]],"truth_error"=>observed.errors[k][reproduction_count+1],
                    "primary_residual"=>last(residuals[k]),"trace"=>traced))
            end
        end
        reproduction_count += 1
    end
    reproduction_count==length(primary_indices) || error("incomplete primary reproduction")
    detail = Dict{String,Any}()
    for k in ("u","v")
        sum(values(classified[k]);init=0)==length(observed.indices) || error("classification population mismatch")
        sum(get(classified[k],reason,0) for reason in ZERO_CLASSIFICATIONS)==observed.p.components[k].counts["sigma_zero"] ||
            error("zero-sigma classification population mismatch")
        detail[k] = Dict("classification_counts"=>classified[k],"negative_pre_clamp_variance_count"=>negative[k],
            "reproduction_count"=>reproduction_count,"full_error_centered_moments"=>summary(observed.centered[k]),
            "uq_subset_centered_moments"=>summary(observed.subset[k]),
            "residual"=>moment_summary(residuals[k]),"raw_eq4_residual"=>moment_summary(raw_residuals[k]),
            "supplementary"=>supplementary(observed.errors[k],observed.sigmas[k],residuals[k],raw_residuals[k]),
            "examples"=>samples[k],"examples_selection"=>"first two column-major primary nodes per classification; not representative sampling")
    end
    row = Dict{String,Any}("condition"=>condition.id,"seed"=>seed,"inputs"=>scene.spec,
        "recipe"=>V.recipe(passes;on_measurement_history="captured and verified final primary origin",
            workspace="dedicated CPU workspace; retained windows audited before reuse"),
        "full_error_metrics"=>observed.data,"diagnostics"=>diagnostic_data(diagnostics[]),"components"=>detail,
        "reproduction"=>Dict("verified"=>true,"primary_nodes"=>reproduction_count,
            "basis"=>"final executed sweep predictor-deformed CPU windows; not rewarped using returned displacement"))
    (;row,population=observed.p,centered=observed.centered,subset=observed.subset)
end

function diagnostic_data(diagnostics)
    out = U.diagnostic_data(diagnostics)
    for (row,pass) in zip(out["passes"],diagnostics.passes)
        row["requested_tolerance"] = pass.requested_tolerance
        row["last_tolerance_check"] = pass.last_check===nothing ? Dict("available"=>false,"reason"=>"not_checked") :
            Dict{String,Any}("available"=>true,"observation"=>Dict(String(k)=>
                (v===nothing ? "unavailable" : v isa Symbol ? String(v) : v) for (k,v) in pairs(pass.last_check)))
    end
    out
end

function moment_summary(values)
    m = U.Moment()
    foreach(x->U.add!(m,x),values)
    U.summary(m)
end

function conditions()
    [(id="baseline",settings=(;),window=16),(id="baseline_window32",settings=(;),window=32),
     (id="noise_0_03",settings=(;noise=0.03),window=16)]
end
function contrast_cases()
    rows = NamedTuple[(condition=(id="baseline_final_budget3",settings=(;),window=16),
             passes=multipass_parameters([32,16,16];padding=true,apodization=:gauss,n_peaks=1,
                 replace_outliers=false,final=(uncertainty=true,max_iterations=3,convergence_tol=0.0))),
            (condition=(id="baseline_final_tolerance",settings=(;),window=16),
             passes=multipass_parameters([32,16,16];padding=true,apodization=:gauss,n_peaks=1,
                 replace_outliers=false,final=(uncertainty=true,max_iterations=6,convergence_tol=0.001)))]
    for (id,settings) in (("phase_u_integer",(;du=2.0)),("phase_u_half",(;du=2.5)),
                          ("phase_v_quarter",(;dv=-1.25)))
        push!(rows,(condition=(;id,settings,window=16),passes=U.controlled_passes()))
    end
    rows
end
function environment_record()
    record = U.environment_record()
    append!(record["source_files"],V.fixture_identity([@__FILE__]))
    record
end
function run_investigation(;seeds=DEFAULT_SEEDS,contrasts=true)
    length(seeds) in 1:8 && all(s->s isa Integer && !(s isa Bool) && 0<=s<=typemax(Int),seeds) &&
        length(unique(seeds))==length(seeds) || throw(ArgumentError("provide one to eight distinct nonnegative integer seeds"))
    environment = environment_record()
    groups = Dict{String,Any}[]
    for condition in conditions()
        pooled = U.Population()
        centered = Dict(k=>CenteredMoment() for k in ("u","v"))
        subset = Dict(k=>CenteredMoment() for k in ("u","v"))
        rows = Dict{String,Any}[]
        classifications = Dict(k=>Dict{String,Int}() for k in ("u","v"))
        for seed in seeds
            audited = audit_scene(condition,seed)
            U.merge!(pooled,audited.population)
            for k in ("u","v")
                merge!(centered[k],audited.centered[k]);merge!(subset[k],audited.subset[k])
                for (reason,count) in audited.row["components"][k]["classification_counts"]
                    classifications[k][reason]=get(classifications[k],reason,0)+count
                end
            end
            push!(rows,audited.row)
        end
        push!(groups,Dict("condition"=>condition.id,"replicates"=>rows,"full_error_pooled"=>U.summary(pooled),
            "classification_counts"=>classifications,
            "centered_moments_pooled"=>Dict(k=>Dict("full_error"=>summary(centered[k]),"uq_subset"=>summary(subset[k])) for k in ("u","v")),
            "aggregation"=>"pooled nodes, not independent vectors; centered coverage/quantiles remain per seed"))
    end
    contrasts_rows = contrasts ? [audit_scene(c.condition,first(seeds);passes=c.passes).row for c in contrast_cases()] : Dict{String,Any}[]
    stable = U.stable_environment(environment)
    Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),"seed_list"=>collect(seeds),
        "environment"=>environment,"source_and_environment_stable"=>stable,
        "provenance_status"=>stable ? "on-disk source/environment stable; fresh-process execution required" : "source/environment changed; regenerate",
        "groups"=>groups,"contrasts"=>contrasts_rows,
        "conventions"=>Dict("units"=>"px displacement per image pair; variance in correlation-difference squared units",
            "primary_selection"=>"finite unflagged unmasked outputs verified as primary against final measurement history",
            "coverage"=>"original full truth error against stored random sigma; component zeros retained; no floor",
            "centering"=>"supplementary in-sample same-seed UQ-subset centering; never replaces full-error coverage",
            "quadrature"=>"descriptive residual-inclusive alternatives; neither calibrated nor production total uncertainty",
            "window_support"=>"finite physical interrogation support, mean subtraction, apodization of each input, shortened common shift support",
            "diameter"=>"renderer diameter=4sigma; paper size=2sigma; renderer diameter3 equals paper size1.5",
            "timing"=>"no warmup, timing or performance samples",
            "trace"=>"production half-plane offsets, whole-square-ring maximum threshold 0.05*S00, signed whole-ring inclusion, clamp at zero"),
        "limitations"=>["No causal attribution from coverage or bias aggregates.","No production default changes or uncertainty calibration.",
            "Unflagged primary does not prove correct peak or per-node convergence.","Overlapping vectors are correlated; no vector-independence confidence intervals.",
            "Gaussian weighting/overlap gain and finite support differ from paper conventions; not an exact DaVis reproduction.",
            "No GPU hardware, quadrature renderer, independent experimental truth or spatial-resolution study.",
            "Native Julia source hashes describe on-disk files, not module attestation; use a fresh process."])
end

function check_output(directory)
    target = U.resolved_path(directory)
    repository = realpath(ROOT)
    allowed = U.resolved_path(joinpath(ROOT,"bench","profile-output"))
    U.within(target,repository) && !U.within(target,allowed) && throw(ArgumentError("repository output must remain under bench/profile-output"))
    protected = [@__FILE__,joinpath(@__DIR__,"validation_uncertainty.jl"),joinpath(@__DIR__,"validation_scorecard.jl"),
        joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")]
    if Base.active_project()!==nothing
        push!(protected,Base.active_project(),joinpath(dirname(Base.active_project()),"Manifest.toml"))
    end
    for root in ("src","ext","test","docs","reference")
        for (dir,_,files) in walkdir(joinpath(ROOT,root))
            append!(protected,joinpath.(dir,files))
        end
    end
    paths = [joinpath(target,"diagnostic_uncertainty.toml"),joinpath(target,"diagnostic_uncertainty.md")]
    for path in paths
        islink(path) && !ispath(path) && throw(ArgumentError("dangling report symlink"))
        if ispath(path)
            isfile(path) || throw(ArgumentError("output is not a regular file"))
            resolved = realpath(path)
            U.within(resolved,repository) && !U.within(resolved,allowed) && throw(ArgumentError("report aliases protected repository content"))
            any(p->isfile(p) && Base.samefile(path,p),protected) && throw(ArgumentError("report aliases source or fixture"))
            occursin(MARKER,open(readline,path)) || throw(ArgumentError("refusing to overwrite unrelated output"))
        end
    end
    ispath(paths[1]) && ispath(paths[2]) && Base.samefile(paths[1],paths[2]) && throw(ArgumentError("report outputs alias each other"))
    paths
end
function markdown_report(report)
    io = IOBuffer()
    println(io,"<!-- $MARKER -->\n# Uncertainty diagnostic\n\nProvenance: ",report["provenance_status"],".\n")
    println(io,"Full recipes, identities, original populations, traces and separate descriptive alternatives: `diagnostic_uncertainty.toml`.\n")
    println(io,"| Condition | Component | Primary / unmasked | UQ available | Full bias / RMS | Centered RMS (UQ subset) | Full-error 2sigma coverage | Zero classifications |")
    println(io,"|---|---|---:|---:|---:|---:|---:|---|")
    number(x)=string(round(x; sigdigits=6))
    for group in report["groups"],k in ("u","v")
        metrics=group["full_error_pooled"];c=metrics["components"][k]
        moment=c["primary_error"];center=group["centered_moments_pooled"][k]["uq_subset"];coverage=c["coverage"]["2"]
        bias=moment["available"] ? number(moment["mean"])*" / "*number(moment["rms"]) : "unavailable"
        centered=center["available"] ? number(center["centered_rms"]) : "unavailable"
        covered=coverage["available"] ? number(coverage["fraction"])*" ($(coverage["covered_count"])/$(coverage["denominator"]))" : "unavailable"
        counts=group["classification_counts"][k]
        zeros=join([reason*"="*string(n) for (reason,n) in sort!(collect(counts);by=first) if occursin("zero",reason) || occursin("clamp",reason) || occursin("cancellation",reason) || occursin("rounded",reason) || occursin("underflow",reason)],", ")
        println(io,"| $(group["condition"]) | $k | $(metrics["counts"]["primary_valid"])/$(metrics["counts"]["unmasked"]) | $(c["counts"]["uq_available"]) | $bias | $centered | $covered | $zeros |")
    end
    println(io,"\nContrasts (first seed only): ",join([row["condition"] for row in report["contrasts"]],", "),".\n")
    println(io,"Centering and residual quadrature are supplementary sensitivity calculations, not corrected accuracy or calibrated total uncertainty. Original full-error coverage is retained, including zero sigma.\n")
    for limitation in report["limitations"]
        println(io,"- ",limitation)
    end
    String(take!(io))
end
function write_report(directory,report)
    paths=check_output(directory)
    report["schema_version"]==SCHEMA || throw(ArgumentError("unknown diagnostic schema"))
    report["source_and_environment_stable"] || throw(ArgumentError("refusing unstable report; freeze sources and regenerate"))
    io=IOBuffer();println(io,"# $MARKER");TOML.print(io,report;sorted=true)
    toml=String(take!(io));markdown=markdown_report(report)
    mkpath(dirname(first(paths)))
    write(paths[1],toml);write(paths[2],markdown)
    paths
end
function main(args=ARGS)
    seeds=DEFAULT_SEEDS
    contrasts=true
    output=joinpath(ROOT,"bench","profile-output","diagnostic-uncertainty")
    for arg in args
        if arg=="--help"
            println("julia --project=. bench/diagnostic_uncertainty.jl [--seeds=7321,7322,7323] [--no-contrasts] [--output=directory]")
            return nothing
        elseif arg=="--no-contrasts"
            contrasts=false
        elseif startswith(arg,"--seeds=")
            seeds=Tuple(parse.(Int,split(arg[9:end],',')))
        elseif startswith(arg,"--output=")
            output=arg[10:end]
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end
    check_output(output) # refuse protected/unrelated destinations before processing
    report=run_investigation(;seeds,contrasts)
    for path in write_report(output,report)
        println(path)
    end
    report
end
end # module

if abspath(PROGRAM_FILE)==@__FILE__
    DiagnosticUncertainty.main()
end
