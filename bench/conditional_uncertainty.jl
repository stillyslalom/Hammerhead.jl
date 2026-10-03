module ConditionalUncertainty
using Hammerhead, SHA, TOML, Dates, Statistics
include("diagnostic_uncertainty.jl")
const D=DiagnosticUncertainty
const U=D.U
const V=D.V
const ROOT=D.ROOT
const SCHEMA="hammerhead-conditional-uncertainty-1"
const MARKER="Hammerhead conditional uncertainty report v1"
const SCENE_SEEDS=(7321,7322)
const AMPLITUDES=(0.01,0.03)
const REALIZATIONS=6
const BLOCK_WIDTH=5

# SHA domain separation fixes streams independently of loop order or RNG state.
# Hexadecimal seeds are persisted because UInt64 need not fit TOML signed integers.
function stream_seed(scene_seed,amplitude,realization,image)
    image in ("A","B") || throw(ArgumentError("noise image must be A or B"))
    token="hammerhead-centered-uniform-1|$scene_seed|$(bitstring(Float64(amplitude)))|$realization|$image"
    parse(UInt64,bytes2hex(sha256(token))[1:16];base=16)
end
function noisy_scene(clean,amplitude,realization)
    isfinite(amplitude) && amplitude>0 || throw(ArgumentError("noise half width must be positive and finite"))
    realization isa Integer && !(realization isa Bool) && realization>0 || throw(ArgumentError("positive integer realization required"))
    seed=clean.spec["seed"]
    arrays=Matrix{Float64}[]; metadata=Dict{String,Any}()
    for (label,input) in (("A",clean.a),("B",clean.b))
        initial=stream_seed(seed,amplitude,realization,label)
        state=Ref(initial)
        noise=[Float64(amplitude)*(2V.uniform!(state)-1) for _ in eachindex(input)]
        output=input .+ reshape(noise,size(input)) # deliberately no clipping or finite-field recentering
        push!(arrays,output)
        metadata[label]=Dict("stream_seed_hex"=>string(initial;base=16,pad=16),
            "sample_mean"=>mean(noise),"sample_min"=>minimum(noise),"sample_max"=>maximum(noise),
            "negative_image_pixels"=>count(<(0),output),"noise_sha256"=>V.pixel_digest(reshape(noise,size(input))))
    end
    spec=deepcopy(clean.spec)
    spec["image_a_sha256"]=V.pixel_digest(arrays[1]);spec["image_b_sha256"]=V.pixel_digest(arrays[2])
    spec["clean_image_a_sha256"]=clean.spec["image_a_sha256"]
    spec["clean_image_b_sha256"]=clean.spec["image_b_sha256"]
    spec["generator_version"]="fixed-clean-scene-centered-uniform-1"
    spec["noise_uniform_half_width"]=Float64(amplitude)
    spec["noise_clamp"]="none; additive centered-distribution uniform noise, no sample recentering"
    spec["noise_realization"]=Int(realization)
    spec["noise_streams"]=metadata
    spec["noise_theoretical_mean"]=0.0
    spec["noise_theoretical_variance"]=Float64(amplitude)^2/3
    (;a=arrays[1],b=arrays[2],spec,truth=clean.truth)
end

"""All ±4 covariance lags, with pre-specified separable Bartlett width 5.
No production ring cutoff and no clamp. The bound covers summation of supplied
terms only, not covariance accumulation error or model/measurement uncertainty.
"""
function bartlett_variance(stats)
    length(stats)==Hammerhead.UQ_NSTATS || throw(ArgumentError("wrong UQ statistics length"))
    terms=Float64[stats[4]]
    for (i,(dr,dc)) in enumerate(Hammerhead.UQ_OFFSETS)
        push!(terms,2*(1-abs(dr)/BLOCK_WIDTH)*(1-abs(dc)/BLOCK_WIDTH)*stats[4+i])
    end
    value=sum(terms)
    n=length(terms);gamma=n*eps(Float64)/(1-n*eps(Float64))
    bound=gamma*sum(abs,terms)
    (;value,bound,within_summation_bound=isfinite(value) && isfinite(bound) && value<0 && -value<=bound)
end
function bartlett_component(stats)
    variance=bartlett_variance(stats)
    sigma=NaN;reason="nonfinite_statistics"
    if all(isfinite,stats) && !isfinite(variance.value)
        reason="nonfinite_covariance_sum"
    elseif all(isfinite,stats)
        if variance.value<0
            reason=!isfinite(variance.bound) ? "negative_summation_bound_unavailable" :
                variance.within_summation_bound ? "negative_within_summation_bound" : "negative_exceeds_summation_bound"
        else
            C0,Cp,Cm=stats[1:3]
            delta=sqrt(variance.value);center=(Cp+Cm)/2
            lo,hi=center-delta/2,center+delta/2
            if !(C0>0 && lo>0 && isfinite(hi))
                reason="invalid_peak_or_perturbed_side"
            else
                denom=2log(C0)-log(hi)-log(lo)
                if !(denom>0 && isfinite(denom))
                    reason="invalid_peak_curvature"
                else
                    sigma=(log(hi)-log(lo))/(2denom)
                    reason=isfinite(sigma) && sigma>=0 ? (sigma==0 ? "zero_sigma" : "positive_sigma") : "nonfinite_sigma_arithmetic"
                end
            end
        end
    end
    (;sigma,reason,variance)
end

struct Capture
    primary::BitMatrix
    errors::Dict{String,Matrix{Float64}}
    sigmas::Dict{String,Matrix{Float64}}
    bartlett::Dict{String,Matrix{Float64}}
    negative::Dict{String,BitMatrix}
end
function Capture(dims)
    Capture(falses(dims),Dict(k=>fill(NaN,dims) for k in ("u","v")),
        Dict(k=>fill(NaN,dims) for k in ("u","v")),Dict(k=>fill(NaN,dims) for k in ("u","v")),
        Dict(k=>falses(dims) for k in ("u","v")))
end
function audit_capture(scene,id;passes=U.controlled_passes())
    grid=Hammerhead.pass_grid(Float64,size(scene.a),last(passes),nothing,0.5)
    capture=Capture((length(grid.y),length(grid.x)))
    counts=Dict(k=>Dict{String,Int}() for k in ("u","v"))
    bounds=Dict(k=>0 for k in ("u","v"))
    callback=function(obs)
        k=obs.component;index=obs.index
        capture.primary[index]=true
        capture.errors[k][index]=obs.error;capture.sigmas[k][index]=obs.sigma
        capture.negative[k][index]=obs.classification=="negative_variance_clamp"
        comparison=bartlett_component(obs.statistics)
        capture.bartlett[k][index]=comparison.sigma
        counts[k][comparison.reason]=get(counts[k],comparison.reason,0)+1
        comparison.variance.within_summation_bound && (bounds[k]+=1)
    end
    audited=D.audit_pair(scene,(;id,window=16),scene.spec["seed"];passes,on_component=callback)
    for k in ("u","v")
        sum(values(counts[k]);init=0)==count(capture.primary) || error("Bartlett classifications lost nodes")
    end
    candidate=Dict(k=>component_population(capture.errors[k][capture.primary],capture.bartlett[k][capture.primary]) for k in ("u","v"))
    row=audited.row
    row["bartlett_comparator"]=Dict("classification_counts"=>counts,"negative_within_summation_bound_count"=>bounds,
        "full_error_metrics"=>candidate,"scope"=>"same primary windows, fixed ±4 lags, separable Bartlett width5; no clamp or fitted floor")
    (;row,capture)
end
function component_population(errors,sigmas)
    length(errors)==length(sigmas) || throw(ArgumentError("component population lengths differ"))
    population=U.ComponentPopulation();normalized=Float64[]
    for (e,s) in zip(errors,sigmas)
        U.component!(population,e,s,normalized)
    end
    D.component_summary(population)
end
function finite_sigma(s)
    isfinite(s) && s>=0
end
function variance_rms(variances)
    if isempty(variances) || !all(v->isfinite(v) && v>=0,variances)
        return Dict("available"=>false,"reason"=>isempty(variances) ? "empty_variance_population" : "nonfinite_or_negative_variance",
            "denominator"=>length(variances))
    end
    scale=maximum(variances)
    value=scale==0 ? 0.0 : sqrt(scale)*sqrt(sum(v/scale for v in variances)/length(variances))
    isfinite(value) ? Dict("available"=>true,"value"=>value,"denominator"=>length(variances)) :
        Dict("available"=>false,"reason"=>"nonfinite_rms_arithmetic","denominator"=>length(variances))
end

# Complete-case selection is primary-only. UQ availability never chooses the
# full difference or conditional random-error population.
function conditional_summary(clean::Capture,runs::AbstractVector{Capture})
    length(runs)>=2 || throw(ArgumentError("conditional variance needs at least two realizations"))
    dims=size(clean.primary)
    all(r->size(r.primary)==dims,runs) || throw(ArgumentError("capture grids differ"))
    complete=reduce((a,b)->a .& b,getfield.(runs,:primary))
    withclean=complete .& clean.primary
    losses=Dict("grid_nodes"=>length(complete),"noisy_complete_primary"=>count(complete),
        "noisy_incomplete_primary"=>length(complete)-count(complete),"clean_primary"=>count(clean.primary),
        "noisy_complete_and_clean_primary"=>count(withclean),"noisy_complete_lost_clean_primary"=>count(complete .& .!clean.primary),
        "per_realization_primary"=>[count(r.primary) for r in runs],
        "per_realization_primary_lost_to_complete_case"=>[count(r.primary .& .!complete) for r in runs])
    components=Dict{String,Any}()
    for k in ("u","v")
        means=Float64[];variances=Float64[];clean_errors=Float64[];shifts=Float64[]
        arithmetic_unavailable=0;clean_difference_unavailable=0;clean_match_lost_arithmetic=0
        clamp_nodes=0;positive_variance_clamp_nodes=0
        for index in findall(complete)
            errors=[r.errors[k][index] for r in runs]
            m=D.CenteredMoment();foreach(e->D.add!(m,e),errors)
            sm=D.summary(m)
            if !sm["available"]
                arithmetic_unavailable+=1
                withclean[index] && (clean_match_lost_arithmetic+=1)
                continue
            end
            variance=m.m2/(length(runs)-1)
            if !isfinite(variance)
                arithmetic_unavailable+=1
                withclean[index] && (clean_match_lost_arithmetic+=1)
                continue
            end
            push!(means,m.mean);push!(variances,variance)
            any(r->r.negative[k][index],runs) && (clamp_nodes+=1;variance>0 && (positive_variance_clamp_nodes+=1))
            if withclean[index]
                e=clean.errors[k][index];shift=m.mean-e
                if isfinite(e) && isfinite(shift)
                    push!(clean_errors,e);push!(shifts,shift)
                else
                    clean_difference_unavailable+=1
                end
            end
        end
        available=length(means)
        components[k]=Dict("noisy_complete_primary_count"=>count(complete),"conditional_arithmetic_available_count"=>available,
            "conditional_arithmetic_unavailable_count"=>arithmetic_unavailable,
            "conditional_mean_error"=>D.moment_summary(means),"conditional_sample_variance"=>D.moment_summary(variances),
            "conditional_noise_rms"=>variance_rms(variances),
            "clean_error_on_matched_complete_case"=>D.moment_summary(clean_errors),
            "conditional_mean_minus_clean_error"=>D.moment_summary(shifts),
            "clean_difference_arithmetic_unavailable_count"=>clean_difference_unavailable,
            "clean_match_lost_conditional_arithmetic_count"=>clean_match_lost_arithmetic,
            "nodes_with_any_production_negative_clamp"=>clamp_nodes,
            "nodes_with_clamp_and_positive_conditional_variance"=>positive_variance_clamp_nodes,
            "all_realizations_stored_uq_available_count"=>count(i->all(r->finite_sigma(r.sigmas[k][i]),runs),findall(complete)),
            "all_realizations_bartlett_uq_available_count"=>count(i->all(r->finite_sigma(r.bartlett[k][i]),runs),findall(complete)))
    end
    Dict("selection"=>"all noisy realizations primary, independent of sigma; clean comparison uses additional clean-primary intersection",
        "losses"=>losses,"components"=>components,"realizations"=>length(runs),
        "variance_denominator"=>length(runs)-1,
        "interpretation"=>"conditional sample variability at fixed scene; matched clean differences are not a pure systematic-error decomposition")
end
function paired_summary(a::Capture,b::Capture;pair=(1,2))
    size(a.primary)==size(b.primary) || throw(ArgumentError("capture grids differ"))
    common=a.primary .& b.primary
    components=Dict{String,Any}()
    for k in ("u","v")
        errors=[a.errors[k][i]-b.errors[k][i] for i in findall(common)]
        stored=[finite_sigma(a.sigmas[k][i]) && finite_sigma(b.sigmas[k][i]) ? hypot(a.sigmas[k][i],b.sigmas[k][i]) : NaN for i in findall(common)]
        comparison=[finite_sigma(a.bartlett[k][i]) && finite_sigma(b.bartlett[k][i]) ? hypot(a.bartlett[k][i],b.bartlett[k][i]) : NaN for i in findall(common)]
        components[k]=Dict("full_difference_stored_sigma"=>component_population(errors,stored),
            "full_difference_bartlett_sigma"=>component_population(errors,comparison),
            "difference_squared_over_two"=>D.moment_summary([e^2/2 for e in errors]),
            "stored_both_input_uq_available_count"=>count(i->finite_sigma(a.sigmas[k][i]) && finite_sigma(b.sigmas[k][i]),findall(common)),
            "bartlett_both_input_uq_available_count"=>count(i->finite_sigma(a.bartlett[k][i]) && finite_sigma(b.bartlett[k][i]),findall(common)))
    end
    Dict("realizations"=>collect(pair),"grid_nodes"=>length(common),"common_primary"=>count(common),
        "only_first_primary"=>count(a.primary .& .!b.primary),"only_second_primary"=>count(b.primary .& .!a.primary),
        "neither_primary"=>count(.!a.primary .& .!b.primary),"components"=>components,
        "interpretation"=>"disjoint independent noise pairs; fixed conditional mean cancels algebraically, but acceptance can select the pair population")
end

function environment_record()
    record=D.environment_record()
    append!(record["source_files"],V.fixture_identity([@__FILE__]))
    record
end
function run_study(;scene_seeds=SCENE_SEEDS,amplitudes=AMPLITUDES,realizations=REALIZATIONS,size=128)
    !isempty(scene_seeds) && all(s->s isa Integer && !(s isa Bool) && 0<=s<=typemax(Int),scene_seeds) &&
        length(unique(scene_seeds))==length(scene_seeds) || throw(ArgumentError("distinct nonnegative scene seeds required"))
    !isempty(amplitudes) && all(a->isfinite(a) && a>0,amplitudes) && length(unique(amplitudes))==length(amplitudes) ||
        throw(ArgumentError("distinct positive finite amplitudes required"))
    realizations isa Integer && !(realizations isa Bool) && realizations>=2 && iseven(realizations) ||
        throw(ArgumentError("positive even realization count of at least two required"))
    environment=environment_record();groups=Dict{String,Any}[];controls=Dict{String,Any}[]
    for seed in scene_seeds
        clean=V.synthetic_scene(;seed,size)
        control=audit_capture(clean,"baseline");push!(controls,control.row)
        for amplitude in amplitudes
            rows=Dict{String,Any}[];captures=Capture[]
            for realization in 1:realizations
                scene=noisy_scene(clean,amplitude,realization)
                audited=audit_capture(scene,"unclipped_noise_$(amplitude)")
                push!(rows,audited.row);push!(captures,audited.capture)
            end
            push!(groups,Dict("scene_seed"=>seed,"noise_uniform_half_width"=>amplitude,"realizations"=>rows,
                "conditional"=>conditional_summary(control.capture,captures),
                "disjoint_pairs"=>[paired_summary(captures[i],captures[i+1];pair=(i,i+1)) for i in 1:2:realizations]))
        end
    end
    stable=U.stable_environment(environment)
    Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),"environment"=>environment,
        "source_and_environment_stable"=>stable,"provenance_status"=>stable ? "stable on-disk source/environment; fresh process required" : "changed; regenerate",
        "processing_calls"=>length(scene_seeds)*(1+length(amplitudes)*realizations),"scene_seeds"=>collect(scene_seeds),
        "noise_amplitudes"=>collect(amplitudes),"realizations_per_group"=>realizations,"image_size"=>[size,size],
        "clean_controls"=>controls,"groups"=>groups,
        "bartlett"=>Dict("block_width"=>BLOCK_WIDTH,"maximum_lag"=>4,"production_ring_cutoff"=>false,
            "weight"=>"(1-abs(dr)/5)*(1-abs(dc)/5)",
            "identity"=>"sum of squared zero-padded moving5x5 block sums /25 of mean-subtracted smoothed dC field",
            "negative_policy"=>"unavailable, retained classification and summation-only bound; no clamp",
            "selection"=>"fixed before observing data; descriptive comparator, not a production estimator"),
        "limitations"=>["Six realizations and three disjoint pairs per scene/noise level are a small conditional study, not calibration.",
            "Overlapping interrogation nodes are spatially correlated; do not treat nodes as independent replicates.",
            "Complete-case primary selection and sigma availability can depend on noise; losses remain explicit.",
            "Pair cancellation removes a fixed conditional mean algebraically, not renderer/deformation interactions with noise.",
            "Stored sigma and error share image data and may be dependent; quadrature is an independent-noise diagnostic, not a coverage guarantee.",
            "Centered noise means zero in distribution; finite images are neither clipped nor recentered.",
            "All original full-truth-error metrics, zero sigma and unavailable components remain in each run.",
            "Bartlett uses the same finite support and peak formula; sensitivity does not validate a replacement or sigma floor.",
            "No renderer-support/quadrature contrast, performance sampling, GPU study or experimental validation.",
            "Source hashes attest on-disk files, not loaded module code; use a fresh process."])
end

function check_output(directory)
    target=U.resolved_path(directory);repository=realpath(ROOT)
    allowed=U.resolved_path(joinpath(ROOT,"bench","profile-output"))
    U.within(target,repository) && !U.within(target,allowed) && throw(ArgumentError("repository output must stay under bench/profile-output"))
    protected=[@__FILE__,joinpath(@__DIR__,"diagnostic_uncertainty.jl"),joinpath(@__DIR__,"validation_uncertainty.jl"),
        joinpath(@__DIR__,"validation_scorecard.jl"),joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")]
    if Base.active_project()!==nothing
        append!(protected,[Base.active_project(),joinpath(dirname(Base.active_project()),"Manifest.toml")])
    end
    for root in ("src","ext","test","docs","reference"), (dir,_,files) in walkdir(joinpath(ROOT,root))
        append!(protected,joinpath.(dir,files))
    end
    paths=[joinpath(target,"conditional_uncertainty.toml"),joinpath(target,"conditional_uncertainty.md")]
    for path in paths
        islink(path) && !ispath(path) && throw(ArgumentError("dangling output symlink"))
        if ispath(path)
            isfile(path) || throw(ArgumentError("output is not a regular file"))
            resolved=realpath(path)
            U.within(resolved,repository) && !U.within(resolved,allowed) && throw(ArgumentError("output aliases protected repository content"))
            any(p->isfile(p) && Base.samefile(path,p),protected) && throw(ArgumentError("output aliases source or fixture"))
            occursin(MARKER,open(readline,path)) || throw(ArgumentError("refusing unrelated existing output"))
        end
    end
    all(ispath,paths) && Base.samefile(paths...) && throw(ArgumentError("report outputs alias each other"))
    paths
end
function markdown_report(report)
    io=IOBuffer()
    coverage_text(p)=p["available"] ? "$(p["covered_count"])/$(p["denominator"])" :
        "unavailable ($(p["covered_count"])/$(p["denominator"]); $(p["reason"]))"
    println(io,"<!-- $MARKER -->\n# Conditional uncertainty study\n\nProcessing calls: ",report["processing_calls"],". Provenance: ",report["provenance_status"],".\n")
    println(io,"Full-error metrics for every clean/noisy run, identities, stream metadata, selection losses and comparator: `conditional_uncertainty.toml`.\n")
    println(io,"| Scene | Noise half width | Component | Noisy complete primary | Conditional noise RMS | Clamp + positive variance nodes | Paired full / stored UQ / Bartlett UQ | Paired stored 2sigma | Paired Bartlett 2sigma |")
    println(io,"|---|---:|---|---:|---:|---:|---|---|---|")
    for group in report["groups"],k in ("u","v")
        c=group["conditional"]["components"][k];rms=c["conditional_noise_rms"]
        n=rms["available"] ? string(round(rms["value"];sigdigits=6)) : "unavailable"
        pairs=[p["components"][k]["full_difference_stored_sigma"] for p in group["disjoint_pairs"]]
        comparison=[p["components"][k]["full_difference_bartlett_sigma"] for p in group["disjoint_pairs"]]
        populations=join(["$(p["counts"]["primary_valid"])/$(p["counts"]["uq_available"])/$(b["counts"]["uq_available"])" for (p,b) in zip(pairs,comparison)],", ")
        coverage=join([coverage_text(p["coverage"]["2"]) for p in pairs],", ")
        compared=join([coverage_text(p["coverage"]["2"]) for p in comparison],", ")
        println(io,"| $(group["scene_seed"]) | $(group["noise_uniform_half_width"]) | $k | $(c["noisy_complete_primary_count"]) | $n | $(c["nodes_with_clamp_and_positive_conditional_variance"]) | $populations | $coverage | $compared |")
    end
    println(io,"\nEach comma-separated entry follows pairs (1,2), (3,4), (5,6). Coverage entries are covered/available; unavailable populations are explicitly labeled, retaining their denominator.\n")
    println(io,"| Original run | Component | Primary / unmasked | Full-error bias / RMS | Stored UQ / zero sigma | Original full-error 2sigma | Bartlett full-error 2sigma |")
    println(io,"|---|---|---:|---:|---:|---|---|")
    rows=Tuple{String,Dict{String,Any}}[("clean $(r["seed"])",r) for r in report["clean_controls"]]
    for group in report["groups"], (i,row) in enumerate(group["realizations"])
        push!(rows,("$(group["scene_seed"]) noise $(group["noise_uniform_half_width"]) r$i",row))
    end
    for (label,row) in rows,k in ("u","v")
        metrics=row["full_error_metrics"];c=metrics["components"][k];m=c["primary_error"]
        bias=m["available"] ? "$(round(m["mean"];sigdigits=6)) / $(round(m["rms"];sigdigits=6))" : "unavailable"
        original=coverage_text(c["coverage"]["2"])
        compared=coverage_text(row["bartlett_comparator"]["full_error_metrics"][k]["coverage"]["2"])
        println(io,"| $label | $k | $(metrics["counts"]["primary_valid"])/$(metrics["counts"]["unmasked"]) | $bias | $(c["counts"]["uq_available"])/$(c["counts"]["sigma_zero"]) | $original | $compared |")
    end
    println(io,"\nPaired coverage is supplementary and never replaces original full-error coverage. Bartlett uses all ±4 lags with pre-specified width5; no production changes or calibration conclusion.\n")
    for limitation in report["limitations"]
        println(io,"- ",limitation)
    end
    String(take!(io))
end
function write_report(directory,report)
    paths=check_output(directory)
    report["schema_version"]==SCHEMA || throw(ArgumentError("unknown conditional schema"))
    report["source_and_environment_stable"]===true || throw(ArgumentError("refusing unstable report"))
    io=IOBuffer();println(io,"# $MARKER");TOML.print(io,report;sorted=true)
    serialized=String(take!(io));markdown=markdown_report(report)
    mkpath(dirname(first(paths)));write(paths[1],serialized);write(paths[2],markdown)
    paths
end
function main(args=ARGS)
    output=joinpath(ROOT,"bench","profile-output","conditional-uncertainty")
    for arg in args
        if arg=="--help"
            println("julia --project=. bench/conditional_uncertainty.jl [--output=directory] # 26 calls, no timing samples")
            return nothing
        elseif startswith(arg,"--output=")
            output=arg[10:end]
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end
    check_output(output)
    report=run_study()
    foreach(println,write_report(output,report))
    report
end
end # module
if abspath(PROGRAM_FILE)==@__FILE__
    ConditionalUncertainty.main()
end
