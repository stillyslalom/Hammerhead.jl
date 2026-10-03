# Bench-only finite-amplitude response of two specified complete PIV schedules.
module SpatialTransfer
using Hammerhead, LinearAlgebra, Statistics, TOML, Dates
include("validation_uncertainty.jl")
const U=ValidationUncertainty
const V=U.V
const ROOT=U.ROOT
const SCHEMA="hammerhead-spatial-transfer-1"
const MARKER="Hammerhead spatial transfer report v1"
const SEEDS=(7321,7322)
const WAVELENGTHS=(128.0,64.0,32.0)
const TERMINAL_WINDOWS=(16,32)
const AMPLITUDE=0.5
const PHASE=pi/4
const DU=2.25
const DV=-1.5
const DENSITY=0.04
const DIAMETER=3.0
const CONDITION_LIMIT=1e8
const INTERIOR_MARGIN=64

function parameters(window)
    window in TERMINAL_WINDOWS || throw(ArgumentError("terminal window must be 16 or 32"))
    [PIVParameters(;window_size=w,overlap=o,padding=true,apodization=:gauss,
        n_peaks=1,replace_outliers=false,uncertainty=i==4,max_iterations=1)
        for (i,(w,o)) in enumerate(((64,32),(32,16),(window,window-8),(window,window-8)))]
end
function motion(y,wavelength;amplitude=AMPLITUDE,center=128.5)
    DU+amplitude*sin(2pi*((y-center)/wavelength)+PHASE),DV
end
function midpoint_truth(x,y,wavelength;amplitude=AMPLITUDE,center=128.5)
    motion(y-DV/2,wavelength;amplitude,center)
end
function particle_positions(seed,size)
    state=Ref(UInt64(seed));points=zeros(round(Int,DENSITY*size^2),2)
    for i in axes(points,1)
        points[i,1]=1+(size-1)*V.uniform!(state)
        points[i,2]=1+(size-1)*V.uniform!(state)
        V.uniform!(state) # preserve the unchanged generator's zero-dropout draw
    end
    points
end
function scene(seed,wavelength;size=256,amplitude=AMPLITUDE)
    seed isa Integer && !(seed isa Bool) && 0<=seed<=typemax(Int) || throw(ArgumentError("nonnegative integer seed required"))
    size isa Integer && !(size isa Bool) && size>=64 || throw(ArgumentError("image side must be at least 64"))
    isfinite(wavelength) && wavelength>0 && isfinite(amplitude) && amplitude>=0 || throw(ArgumentError("invalid prescribed motion"))
    center=(size+1)/2;points=particle_positions(seed,size)
    if amplitude==0
        original=V.synthetic_scene(;seed,size,density=DENSITY,diameter=DIAMETER,du=DU,dv=DV)
        return (;original...,points)
    end
    a,b=zeros(size,size),zeros(size,size)
    for i in axes(points,1)
        x,y=points[i,1],points[i,2]
        u,v=motion(y,wavelength;amplitude,center)
        Hammerhead.SyntheticData.generate_gaussian_particle!(a,(x,y),DIAMETER)
        Hammerhead.SyntheticData.generate_gaussian_particle!(b,(x+u,y+v),DIAMETER)
    end
    spec=Dict{String,Any}("generator_version"=>"splitmix64-transverse-sine-particles-1",
        "seed"=>seed,"image_size"=>[size,size],"particle_count"=>length(points)÷2,
        "particle_density"=>DENSITY,"diameter_px_4sigma"=>DIAMETER,"particle_intensity"=>1.0,
        "noise_uniform_half_width"=>0.0,"dropout_requested_fraction"=>0.0,"dropout_removed_count"=>0,
        "du_offset_px"=>DU,"dv_px"=>DV,"amplitude_px"=>amplitude,"wavelength_px"=>Float64(wavelength),
        "phase_radians"=>PHASE,"center_row"=>center,"dt"=>1.0,
        "integration"=>"one prescribed launch displacement; not continuous-trajectory integration",
        "particle_positions_sha256"=>V.pixel_digest(points),
        "particle_positions_encoding"=>"Float64 little-endian column-major N-by-2 x,y",
        "pixel_hash_encoding"=>"Float64 little-endian column-major",
        "image_a_sha256"=>V.pixel_digest(a),"image_b_sha256"=>V.pixel_digest(b))
    truth=(x,y)->midpoint_truth(x,y,wavelength;amplitude,center)
    (;a,b,spec,truth,points)
end

function moments(values)
    m=U.Moment();foreach(v->U.add!(m,v),values);U.summary(m)
end

"""Guarded descriptive OLS. Every supplied sample stays in its denominator."""
function harmonic_fit(rows,values,wavelength;amplitude=AMPLITUDE,center=128.5)
    length(rows)==length(values) || throw(ArgumentError("row/value lengths differ"))
    isfinite(wavelength) && wavelength>0 && isfinite(amplitude) && amplitude>=0 && isfinite(center) || throw(ArgumentError("invalid fit reference"))
    n=length(rows)
    out=Dict{String,Any}("available"=>false,"sample_count"=>n,"rank"=>0,
        "condition_available"=>false,"condition_limit"=>CONDITION_LIMIT,
        "rank_threshold"=>"max(N,3)*eps(Float64)*largest_singular_value",
        "normalized_response_available"=>false,"phase_response_available"=>false,
        "model"=>"intercept + sine_coefficient*sin(theta) + cosine_coefficient*cos(theta)",
        "theta"=>"2pi*((y_vector-v/2-center)/wavelength)+pi/4",
        "weighting"=>"equal weight per supplied node; no independence or inferential confidence interval")
    fail(reason)=(out["reason"]=reason;out)
    n>=3 || return fail("too_few_samples")
    yy,zz=Float64.(rows),Float64.(values)
    all(isfinite,yy) && all(isfinite,zz) || return fail("nonfinite_input")
    out["distinct_rows"]=length(unique(yy))
    theta=2pi .* ((yy .- DV/2 .- center) ./ wavelength) .+ PHASE
    all(isfinite,theta) || return fail("nonfinite_phase_arithmetic")
    design=hcat(ones(n),sin.(theta),cos.(theta))
    singular=svdvals(design);threshold=max(n,3)*eps(Float64)*first(singular)
    out["singular_values"]=singular;out["absolute_rank_threshold"]=threshold
    out["rank"]=count(>(threshold),singular)
    out["rank"]==3 || return fail("rank_deficient")
    condition=first(singular)/last(singular)
    if isfinite(condition)
        out["condition_available"]=true;out["condition_number"]=condition
    end
    isfinite(condition) && condition<=CONDITION_LIMIT || return fail("ill_conditioned")
    # Scale data rather than square unscaled values or subtract a huge intercept.
    scale=maximum(abs,zz)
    beta=design\(scale==0 ? zeros(n) : zz./scale)
    all(isfinite,beta) || return fail("nonfinite_solve_arithmetic")
    coefficients=beta.*scale
    all(isfinite,coefficients) || return fail("nonfinite_coefficient_arithmetic")
    residuals=(scale==0 ? zeros(n) : zz./scale-design*beta).*scale
    residual=moments(residuals)
    residual["available"]===true || return fail("nonfinite_residual_arithmetic")
    offset,s,c=coefficients;harmonic=hypot(s,c)
    isfinite(harmonic) || return fail("nonfinite_harmonic_arithmetic")
    out["available"]=true
    out["intercept"]=offset;out["sine_coefficient"]=s;out["cosine_coefficient"]=c
    out["harmonic_amplitude"]=harmonic;out["fit_residual"]=residual
    if amplitude==0
        out["normalization_reason"]="zero_prescribed_amplitude_control"
        out["phase_reason"]="no_input_harmonic_control"
    else
        normalized=(s/amplitude,c/amplitude,harmonic/amplitude)
        if all(isfinite,normalized)
            out["normalized_response_available"]=true
            out["in_phase_gain"],out["quadrature_gain"],out["amplitude_gain"]=normalized
        else
            out["normalization_reason"]="nonfinite_normalization_arithmetic"
        end
        if harmonic>0
            out["phase_response_available"]=true;out["phase_radians"]=atan(c,s)
        else
            out["phase_reason"]="zero_harmonic_phase_undefined"
        end
    end
    out
end

function selected_metrics(result,truth,selection)
    size(selection)==size(result.u) || throw(ArgumentError("selection dimensions differ"))
    # Keep the original estimator and arithmetic/population rules. All fields
    # are borrowed read-only; changing this mask does not modify the result.
    selected=PIVResult(result.x,result.y,result.u,result.v,result.peak_ratio,result.correlation_moment,
        result.uncertainty_u,result.uncertainty_v,result.outliers,result.mask .| .!selection,
        result.parameters,result.correlation_planes,result.scale)
    data=U.metrics(selected,truth).data
    data["analysis_selection"]=Dict("outside_selection_count"=>count(.!selection),
        "original_mask_in_selection_count"=>count(result.mask .& selection),
        "mask_convention"=>"metrics masked count includes original masks plus excluded analysis nodes; stored result unchanged")
    data
end

function response(result,selection,wavelengths;amplitude,center)
    U.check_primary(result.parameters)
    primary=selection .& .!result.mask .& .!result.outliers .& isfinite.(result.u) .& isfinite.(result.v)
    rows=[Float64(result.y[i[1]]) for i in findall(primary)]
    Dict{String,Any}("primary_nodes"=>count(primary),"candidate_nodes"=>count(selection),
        "fits"=>[Dict("wavelength_px"=>Float64(wavelength),
            "u"=>harmonic_fit(rows,result.u[primary],wavelength;amplitude,center),
            "v"=>harmonic_fit(rows,result.v[primary],wavelength;amplitude,center),
            "v_response_convention"=>"v harmonic coefficients divided by prescribed u amplitude describe transverse response, not v-input gain")
            for wavelength in wavelengths])
end
function evaluate(input,window,wavelength;amplitude,interior_margin=INTERIOR_MARGIN)
    passes=parameters(window);U.check_primary(last(passes));diagnostic=Ref{Any}(nothing)
    result=run_piv(input.a,input.b,passes;threaded=false,on_diagnostics=d->(diagnostic[]=d))
    full=trues(size(result.u))
    side=size(input.a,1);interior=[interior_margin<=x<=side-interior_margin && interior_margin<=y<=side-interior_margin
        for y in result.y,x in result.x]
    frequencies=amplitude==0 ? WAVELENGTHS : (Float64(wavelength),)
    center=(side+1)/2
    row=Dict{String,Any}("terminal_window_px"=>window,"inputs"=>input.spec,
        "particle_positions_sha256"=>V.pixel_digest(input.points),
        "particle_positions_encoding"=>"Float64 little-endian column-major N-by-2 x,y",
        "primary_origin"=>"trusted fresh run: actual final parameters n_peaks=1, replace_outliers=false, max_iterations=1, uncertainty=true; exclude masked/flagged/nonfinite components",
        "recipe"=>V.recipe(passes;on_diagnostics="final primary residual summary"),
        "full_error_metrics"=>U.metrics(result,input.truth).data,
        "diagnostics"=>U.diagnostic_data(diagnostic[]),
        "full_grid_response"=>response(result,full,frequencies;amplitude,center),
        "fixed_interior"=>Dict("margin_px"=>interior_margin,"coordinate_bounds"=>[interior_margin,side-interior_margin],
            "full_error_metrics"=>selected_metrics(result,input.truth,interior),
            "response"=>response(result,interior,frequencies;amplitude,center)))
    (;row,result,interior)
end

function common_population(evaluations,input,wavelength;amplitude)
    length(evaluations)==2 || throw(ArgumentError("exactly two schedule results required"))
    coordinates(e)=Dict((Float64(e.result.x[i[2]]),Float64(e.result.y[i[1]]))=>i for i in findall(e.interior))
    foreach(e->U.check_primary(e.result.parameters),evaluations)
    maps=coordinates.(evaluations)
    candidates=sort!(collect(intersect(Set(keys(maps[1])),Set(keys(maps[2])))))
    primary(e,i)=!e.result.mask[i] && !e.result.outliers[i] && isfinite(e.result.u[i]) && isfinite(e.result.v[i])
    common=[xy for xy in candidates if all(primary(e,m[xy]) for (e,m) in zip(evaluations,maps))]
    frequencies=amplitude==0 ? WAVELENGTHS : (Float64(wavelength),)
    rows=Dict{String,Any}[]
    for (e,map) in zip(evaluations,maps)
        chosen=falses(size(e.result.u))
        foreach(xy->(chosen[map[xy]]=true),common)
        nprimary=count(i->primary(e,i),values(map))
        ngeometric=count(xy->primary(e,map[xy]),candidates)
        push!(rows,Dict("terminal_window_px"=>e.row["terminal_window_px"],
            "own_interior_candidates"=>length(map),"interior_candidates_not_shared"=>length(map)-length(candidates),
            "own_interior_primary"=>nprimary,"primary_on_shared_candidates"=>ngeometric,
            "primary_lost_to_common"=>ngeometric-length(common),
            "full_error_metrics"=>selected_metrics(e.result,input.truth,chosen),
            "response"=>response(e.result,chosen,frequencies;amplitude,center=(size(input.a,1)+1)/2)))
    end
    Dict{String,Any}("candidate_coordinates"=>[collect(xy) for xy in candidates],
        "common_primary_coordinates"=>[collect(xy) for xy in common],"candidate_nodes"=>length(candidates),
        "common_primary"=>length(common),"not_common_primary"=>length(candidates)-length(common),
        "selection"=>"exact shared fixed-interior coordinates; primary accepted by both schedules, independent of sigma availability; no interpolation",
        "schedules"=>rows)
end

function environment_record()
    env=U.environment_record();append!(env["source_files"],V.fixture_identity([@__FILE__]));env
end
function run_study(;seeds=SEEDS,size=256,wavelengths=WAVELENGTHS,interior_margin=INTERIOR_MARGIN)
    size>=64 && size isa Integer && !(size isa Bool) || throw(ArgumentError("invalid image size"))
    interior_margin isa Integer && !(interior_margin isa Bool) && 0<=interior_margin<=size/2 || throw(ArgumentError("invalid interior margin"))
    all(w->isfinite(w) && w>0,wavelengths) || throw(ArgumentError("invalid wavelengths"))
    env=environment_record();groups=Dict{String,Any}[];calls=0
    for seed in seeds,(wavelength,amplitude) in ((first(WAVELENGTHS),0.0),[(w,AMPLITUDE) for w in wavelengths]...)
        input=scene(seed,wavelength;size,amplitude)
        evaluations=[evaluate(input,w,wavelength;amplitude,interior_margin) for w in TERMINAL_WINDOWS];calls+=2
        push!(groups,Dict("scene_seed"=>seed,"motion"=>amplitude==0 ? "translation_control" : "sinusoidal_shear",
            "prescribed_amplitude_px"=>amplitude,"wavelength_px"=>Float64(wavelength),
            "schedules"=>[e.row for e in evaluations],
            "common_interior"=>common_population(evaluations,input,wavelength;amplitude)))
    end
    stable=U.stable_environment(env)
    Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),
        "processing_calls"=>calls,"environment"=>env,"source_and_environment_stable"=>stable,
        "provenance_status"=>stable ? "stable on-disk source/environment; fresh process required" : "source/environment drift; regenerate",
        "image_side"=>size,"renderer"=>Dict("function"=>"SyntheticData.generate_gaussian_particle!",
            "diameter_px_4sigma"=>DIAMETER,"gaussian_sigma_px"=>DIAMETER/4,"particle_peak_intensity"=>1.0,
            "sampling"=>"point intensity at integer pixel center; no integration or renormalization",
            "integer_radius"=>ceil(Int,3DIAMETER/4),
            "bbox"=>"max(1,round(Int,center-radius)):min(size,round(Int,center+radius)); square not circular",
            "particle_density"=>DENSITY,"noise"=>0.0,"dropout"=>0.0),
        "truth"=>"prescribed launch u=2.25+A*sin(2pi*(y_launch-center)/lambda+pi/4), v=-1.5; midpoint y_launch=y_vector-v/2; no estimated-vector reference",
        "fit"=>Dict("condition_limit"=>CONDITION_LIMIT,"amplitude_px"=>AMPLITUDE,"phase_radians"=>PHASE,
            "rank_threshold"=>"max(N,3)*eps(Float64)*largest_singular_value; singular values and absolute threshold recorded per computed fit",
            "output_stride_px"=>8,"response_convention"=>"s/A signed in-phase; c/A quadrature; hypot(s,c)/A magnitude; atan(c,s) positive phase lead",
            "v_response_convention"=>"v harmonic coefficients normalized by prescribed u amplitude describe transverse response, not v-input gain",
            "control"=>"A=0 has absolute harmonic leakage coefficients; normalized gain/response phase unavailable"),
        "groups"=>groups,"limitations"=>[
            "Finite-amplitude response of two complete schedules; not a universal linear MTF, isolated window kernel or minimum resolved scale.",
            "Two placements, one phase/amplitude/orientation, three frequencies, clean production rendering only; no experimental or GPU evidence.",
            "Coarse schedules match; last two windows/overlaps differ. Deformation, predictor smoothing and validation remain part of measured response.",
            "Own primary and paired common selections are explicit; rejected/nonfinite nodes remain in yield denominators. Sigma availability never selects fits.",
            "No harmonic-fit correction replaces original full truth-error or stored-sigma coverage. Correlated nodes are not independent replicates.",
            "Finite-frame/support and image interpolation effects remain. Interior analysis does not remove full-grid errors.",
            "No noise realizations, fitted sigma floor, estimator/default change, performance sampling or universal attenuation threshold.",
            "On-disk identity does not attest already loaded code; run in a fresh Julia process with frozen sources."])
end

function check_output(directory)
    target=U.resolved_path(directory);repository=realpath(ROOT);allowed=U.resolved_path(joinpath(ROOT,"bench","profile-output"))
    U.within(target,repository) && !U.within(target,allowed) && throw(ArgumentError("repository output must stay under bench/profile-output"))
    protected=[joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")]
    if Base.active_project()!==nothing
        append!(protected,[Base.active_project(),joinpath(dirname(Base.active_project()),"Manifest.toml")])
    end
    for root in ("src","ext","test","docs","reference","bench"),(dir,_,files) in walkdir(joinpath(ROOT,root))
        root=="bench" && U.within(U.resolved_path(dir),allowed) && continue
        append!(protected,joinpath.(dir,files))
    end
    paths=[joinpath(target,"spatial_transfer.toml"),joinpath(target,"spatial_transfer.md")]
    for path in paths
        islink(path) && !ispath(path) && throw(ArgumentError("dangling report symlink"))
        if ispath(path)
            isfile(path) || throw(ArgumentError("output must be a regular file"))
            resolved=realpath(path)
            U.within(resolved,repository) && !U.within(resolved,allowed) && throw(ArgumentError("output aliases protected repository content"))
            any(p->isfile(p) && Base.samefile(path,p),protected) && throw(ArgumentError("output aliases source/fixture"))
            occursin(MARKER,open(readline,path)) || throw(ArgumentError("refusing unrelated existing output"))
        end
    end
    all(ispath,paths) && Base.samefile(paths...) && throw(ArgumentError("report outputs alias each other"))
    paths
end
function markdown_report(report)
    io=IOBuffer();println(io,"<!-- $MARKER -->\n# Spatial-transfer diagnostic\n\nProcessing calls: $(report["processing_calls"]). $(report["provenance_status"]).\n")
    println(io,"Original full-grid errors, all unavailable/zero UQ, complete recipes/identities and own/interior/common fits: `spatial_transfer.toml`.\n")
    println(io,"| Seed | Motion / wavelength | Window | Component | Primary / unmasked | Full-error bias / RMS px | UQ / zero | Full-error 2sigma |")
    println(io,"|---|---|---:|---|---:|---:|---:|---:|")
    number(x)=string(round(x;sigdigits=6))
    for g in report["groups"],s in g["schedules"],k in ("u","v")
        m=s["full_error_metrics"];c=m["components"][k];err=c["primary_error"];coverage=c["coverage"]["2"]
        text=err["available"] ? "$(number(err["mean"])) / $(number(err["rms"]))" : "unavailable ($(err["reason"]))"
        println(io,"| $(g["scene_seed"]) | $(g["motion"]) / $(g["wavelength_px"]) | $(s["terminal_window_px"]) | $k | $(m["counts"]["primary_valid"])/$(m["counts"]["unmasked"]) | $text | $(c["counts"]["uq_available"])/$(c["counts"]["sigma_zero"]) | $(coverage["covered_count"])/$(coverage["denominator"])$(coverage["available"] ? "" : " unavailable") |")
    end
    println(io,"\n| Seed | Motion / fit wavelength | Window | Component | Common / candidates | Primary lost | UQ on common | Signed / quadrature | Magnitude / phase rad | Intercept / harmonic px | Fit RMS px / condition |")
    println(io,"|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for g in report["groups"],s in g["common_interior"]["schedules"],f in s["response"]["fits"],k in ("u","v")
        fit=f[k];common=g["common_interior"];c=s["full_error_metrics"]["components"][k]
        if fit["available"]
            gains=fit["normalized_response_available"] ? "$(number(fit["in_phase_gain"])) / $(number(fit["quadrature_gain"]))" : "unavailable ($(fit["normalization_reason"]))"
            amp=fit["normalized_response_available"] ? number(fit["amplitude_gain"]) : "unavailable"
            phase=fit["phase_response_available"] ? number(fit["phase_radians"]) : "unavailable"
            coefficient="$(number(fit["intercept"])) / $(number(fit["harmonic_amplitude"]))"
            residual="$(number(fit["fit_residual"]["rms"])) / $(number(fit["condition_number"]))"
        else
            gains="unavailable ($(fit["reason"]); rank $(fit["rank"]))";amp=phase=coefficient=residual="unavailable"
        end
        println(io,"| $(g["scene_seed"]) | $(g["motion"]) / $(f["wavelength_px"]) | $(s["terminal_window_px"]) | $k | $(common["common_primary"])/$(common["candidate_nodes"]) | $(s["primary_lost_to_common"]) | $(c["counts"]["uq_available"])/$(common["common_primary"]) | $gains | $amp / $phase | $coefficient | $residual |")
    end
    println(io,"\nCommon fits select exact shared interior primary nodes independently of sigma availability. v coefficients normalized by prescribed u amplitude describe transverse response, not a v-input gain. Control coefficients are absolute leakage with no input harmonic to normalize. Fits never correct the original full-error table.\n")
    foreach(l->println(io,"- ",l),report["limitations"]);String(take!(io))
end
function write_report(directory,report)
    paths=check_output(directory)
    report["schema_version"]==SCHEMA || throw(ArgumentError("unknown spatial-transfer schema"))
    report["source_and_environment_stable"]===true || throw(ArgumentError("refusing unstable evidence"))
    io=IOBuffer();println(io,"# $MARKER");TOML.print(io,report;sorted=true)
    toml=String(take!(io));markdown=markdown_report(report)
    mkpath(dirname(first(paths)));write(paths[1],toml);write(paths[2],markdown);paths
end
function main(args=ARGS)
    output=joinpath(ROOT,"bench","profile-output","spatial-transfer")
    for arg in args
        if arg=="--help"
            println("julia --project=. bench/spatial_transfer.jl [--output=directory] # fixed 16 CPU calls; no timing samples")
            return nothing
        elseif startswith(arg,"--output=")
            output=arg[10:end]
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end
    check_output(output);report=run_study();foreach(println,write_report(output,report));report
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    SpatialTransfer.main()
end
