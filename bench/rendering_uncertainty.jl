module RenderingUncertainty
using Hammerhead, LinearAlgebra, Statistics, SHA, TOML, Dates
include("diagnostic_uncertainty.jl")
const D=DiagnosticUncertainty
const U=D.U
const V=D.V
const ROOT=D.ROOT
const SCHEMA="hammerhead-rendering-uncertainty-1"
const MARKER="Hammerhead rendering uncertainty report v1"
const SEEDS=(7321,7322)
const SHIFTS=((id="original",du=2.25,dv=-1.5),(id="integer_half_warp",du=2.0,dv=-2.0))
const POLICIES=("production_point","wide_point","wide_area")
const SIGMA=0.75
const QUADRATURE_ORDER=16
const REFERENCE_ORDER=32
const IMAGE_CONVERGENCE_TOL=1e-10
const INTERIOR_MARGIN=16

function quadrature_rule(n::Integer)
    !(n isa Bool) && 2<=n<=64 || throw(ArgumentError("quadrature order must be 2:64"))
    system=SymTridiagonal(zeros(n),[k/sqrt(4.0k^2-1) for k in 1:n-1])
    decomposition=eigen(system)
    (;nodes=decomposition.values,weights=2 .* vec(decomposition.vectors[1,:]).^2)
end
function pixel_integral(pixel,center,sigma,rule)
    sigma>0 && isfinite(sigma) || throw(ArgumentError("positive finite Gaussian sigma required"))
    sum(0.5w*exp(-(pixel+0.5t-center)^2/(2sigma^2)) for (t,w) in zip(rule.nodes,rule.weights))
end
function particle_placements(seed;size=128,density=0.02)
    seed isa Integer && !(seed isa Bool) && 0<=seed<=typemax(Int) || throw(ArgumentError("nonnegative integer seed required"))
    size isa Integer && !(size isa Bool) && size>=32 && isfinite(density) && density>=0 || throw(ArgumentError("invalid particle geometry"))
    state=Ref(UInt64(seed));points=Tuple{Float64,Float64}[]
    for _ in 1:round(Int,density*size^2)
        x,y=1+(size-1)*V.uniform!(state),1+(size-1)*V.uniform!(state)
        V.uniform!(state) # retain the original generator's dropout draw, even at zero dropout
        push!(points,(x,y))
    end
    points
end
function placement_digest(points)
    coordinates=zeros(length(points),2)
    for (i,(x,y)) in enumerate(points)
        coordinates[i,1]=x;coordinates[i,2]=y
    end
    V.pixel_digest(coordinates)
end
function render(points,policy;size=128,shift=(0.0,0.0),rule=quadrature_rule(QUADRATURE_ORDER),full_support=false)
    policy in POLICIES || throw(ArgumentError("unknown renderer"))
    image=zeros(size,size)
    radius=ceil(Int,(policy=="production_point" ? 3 : 6)*SIGMA)
    for (x,y) in points
        cx,cy=x+shift[1],y+shift[2]
        if policy=="production_point" && !full_support
            Hammerhead.SyntheticData.generate_gaussian_particle!(image,(cx,cy),4SIGMA,1.0)
            continue
        end
        columns=full_support ? (1:size) : (max(1,round(Int,cx-radius)):min(size,round(Int,cx+radius)))
        rows=full_support ? (1:size) : (max(1,round(Int,cy-radius)):min(size,round(Int,cy+radius)))
        if policy=="wide_area"
            rp=[pixel_integral(r,cy,SIGMA,rule) for r in rows]
            cp=[pixel_integral(c,cx,SIGMA,rule) for c in columns]
            for (j,c) in enumerate(columns),(i,r) in enumerate(rows)
                image[r,c]+=rp[i]*cp[j]
            end
        else
            # Same point formula and addition order as production on common pixels.
            for c in columns,r in rows
                radius2=(c-cx)^2+(r-cy)^2
                image[r,c]+=exp(-radius2/(2SIGMA^2))
            end
        end
    end
    image
end
function renderer_metadata(policy,points,shift)
    Dict{String,Any}("policy"=>policy,"particle_positions_sha256"=>placement_digest(points),
        "particle_positions_encoding"=>"Float64 little-endian column-major N-by-2 x,y coordinates",
        "continuous_gaussian_sigma"=>SIGMA,"diameter_px_4sigma"=>4SIGMA,"particle_peak_intensity"=>1.0,
        "point_intensity_or_area"=>policy=="wide_area" ? "unit pixel area integral" : "point intensity at pixel center",
        "normalization"=>"none","nominal_support_sigma"=>policy=="production_point" ? 3 : 6,
        "integer_radius"=>ceil(Int,(policy=="production_point" ? 3 : 6)*SIGMA),
        "bounds"=>"max(1,round(Int,center-radius)):min(image_size,round(Int,center+radius)); square bbox, not circular cutoff",
        "displacement"=>[shift.du,shift.dv],"quadrature_order"=>policy=="wide_area" ? QUADRATURE_ORDER : 0)
end
function audit_renderer(scene,id)
    passes=U.controlled_passes()
    grid=Hammerhead.pass_grid(Float64,size(scene.a),last(passes),nothing,0.5)
    dims=(length(grid.y),length(grid.x))
    capture=(primary=falses(dims),errors=Dict(k=>fill(NaN,dims) for k in ("u","v")),
        sigmas=Dict(k=>fill(NaN,dims) for k in ("u","v")),residuals=Dict(k=>fill(NaN,dims) for k in ("u","v")))
    callback=function(obs)
        capture.primary[obs.index]=true
        capture.errors[obs.component][obs.index]=obs.error
        capture.sigmas[obs.component][obs.index]=obs.sigma
        capture.residuals[obs.component][obs.index]=obs.primary_residual
    end
    audited=D.audit_pair(scene,(;id,window=16),scene.spec["seed"];passes,on_component=callback)
    derived=Dict(k=>D.moment_summary(capture.errors[k][capture.primary].-capture.residuals[k][capture.primary]) for k in ("u","v"))
    (;row=audited.row,capture,derived)
end
function component_metrics(errors,sigmas)
    length(errors)==length(sigmas) || throw(ArgumentError("population lengths differ"))
    p=U.ComponentPopulation();normalized=Float64[]
    for (e,s) in zip(errors,sigmas)
        U.component!(p,e,s,normalized)
    end
    D.component_summary(p)
end
function common_summary(captures)
    length(captures)==3 || throw(ArgumentError("three rendering captures required"))
    dims=size(first(captures).primary)
    all(c->size(c.primary)==dims,captures) || throw(ArgumentError("capture grids differ"))
    common=reduce((a,b)->a .& b,getproperty.(captures,:primary))
    rows=Dict{String,Any}[]
    for (policy,c) in zip(POLICIES,captures)
        push!(rows,Dict("policy"=>policy,"original_primary"=>count(c.primary),
            "primary_lost_to_common"=>count(c.primary .& .!common),
            "common_metrics"=>Dict(k=>component_metrics(c.errors[k][common],c.sigmas[k][common]) for k in ("u","v"))))
    end
    differences=Dict{String,Any}[]
    for (i,j,label) in ((1,2,"wide_point_minus_production_point"),(2,3,"wide_area_minus_wide_point"))
        push!(differences,Dict("contrast"=>label,
            "errors"=>Dict(k=>D.moment_summary(captures[j].errors[k][common].-captures[i].errors[k][common]) for k in ("u","v")),
            "interpretation"=>"deterministic renderer contrast on common primary nodes; no independent-noise pairing or sigma quadrature"))
    end
    Dict("grid_nodes"=>length(common),"common_primary"=>count(common),"not_common_primary"=>length(common)-count(common),
        "selection"=>"all three outputs primary, independently of component sigma availability; UQ subset losses retained",
        "renderers"=>rows,"deterministic_differences"=>differences)
end

function image_difference(actual,reference,selection)
    size(actual)==size(reference)==size(selection) || throw(ArgumentError("image comparison dimensions differ"))
    differences=actual[selection].-reference[selection]
    summary=D.moment_summary(differences)
    summary["selected_pixels"]=count(selection)
    summary["maximum_absolute_error"]=isempty(differences) ? "unavailable" : maximum(abs,differences)
    summary
end
function oracle_warp(a,b,du,dv)
    Hammerhead.deform_images(Hammerhead.image_interpolant(a,Float64),Hammerhead.image_interpolant(b,Float64),
        (r,c)->du,(r,c)->dv,size(a),Float64;threaded=false)
end
function oracle_deformation(points,shift,images;size=128,rule=quadrature_rule(QUADRATURE_ORDER))
    all_pixels=trues(size,size);interior=falses(size,size)
    if size>2INTERIOR_MARGIN
        interior[INTERIOR_MARGIN+1:size-INTERIOR_MARGIN,INTERIOR_MARGIN+1:size-INTERIOR_MARGIN].=true
    end
    lanes=Dict{String,Any}[]
    for model in ("wide_point","wide_area")
        fulla=render(points,model;size,rule,full_support=true)
        fullb=render(points,model;size,shift=(shift.du,shift.dv),rule,full_support=true)
        midpoint=render(points,model;size,shift=(shift.du/2,shift.dv/2),rule,full_support=true)
        wa,wb=oracle_warp(fulla,fullb,shift.du,shift.dv)
        cases=Dict{String,Any}[]
        selections=(("full_frame",all_pixels),("fixed_interior",interior))
        for (name,selection) in selections
            push!(cases,Dict("population"=>name,"pixels"=>count(selection),
                "sampled_full_field_warp_a_minus_midpoint"=>image_difference(wa,midpoint,selection),
                "sampled_full_field_warp_b_minus_midpoint"=>image_difference(wb,midpoint,selection),
                "sampled_full_field_warp_a_minus_b"=>image_difference(wa,wb,selection)))
        end
        push!(lanes,Dict("model"=>model,"input_support"=>"all in-frame pixel centers/areas; all particle Gaussian contributions, no particle bbox",
            "comparisons"=>cases,"input_image_a_sha256"=>V.pixel_digest(fulla),"input_image_b_sha256"=>V.pixel_digest(fullb),
            "midpoint_image_sha256"=>V.pixel_digest(midpoint)))
        for policy in (model=="wide_point" ? ("production_point","wide_point") : ("wide_area",))
            a,b=images[policy];wa,wb=oracle_warp(a,b,shift.du,shift.dv)
            cases=Dict{String,Any}[]
            for (name,selection) in selections
                push!(cases,Dict("population"=>name,"pixels"=>count(selection),
                    "source_a_bbox_difference"=>image_difference(a,fulla,selection),
                    "source_b_bbox_difference"=>image_difference(b,fullb,selection),
                    "bounded_warp_a_minus_midpoint"=>image_difference(wa,midpoint,selection),
                    "bounded_warp_b_minus_midpoint"=>image_difference(wb,midpoint,selection),
                    "bounded_warp_a_minus_b"=>image_difference(wa,wb,selection)))
            end
            push!(lanes,Dict("model"=>policy,"input_support"=>"specified rounded particle bbox plus finite frame", "comparisons"=>cases))
        end
    end
    Dict("displacement"=>[shift.du,shift.dv],"convention"=>"A(r-dv/2,c-du/2), B(r+dv/2,c+du/2); analytic midpoint particle centers shifted by +(du/2,dv/2)",
        "scope"=>"known translation diagnostic only; these images/predictors never enter PIV accuracy calls",
        "interior_margin_pixels"=>INTERIOR_MARGIN,"lanes"=>lanes,
        "limitations"=>"finite frame/zero extrapolation and B-spline boundary prefilter affect even full-support sampled fields; interior is a sensitivity subset, not an error certificate")
end
function environment_record()
    environment=D.environment_record()
    append!(environment["source_files"],V.fixture_identity([@__FILE__]))
    environment
end
function run_study(;seeds=SEEDS,size=128,shifts=SHIFTS)
    !isempty(seeds) && length(unique(seeds))==length(seeds) || throw(ArgumentError("distinct scene seeds required"))
    environment=environment_record();rule=quadrature_rule(QUADRATURE_ORDER);reference=quadrature_rule(REFERENCE_ORDER)
    groups=Dict{String,Any}[]
    for seed in seeds
        points=particle_placements(seed;size)
        for shift in shifts
            original=V.synthetic_scene(;seed,size,du=shift.du,dv=shift.dv)
            rows=Dict{String,Any}[];captures=NamedTuple[];images=Dict{String,Tuple{Matrix{Float64},Matrix{Float64}}}()
            for policy in POLICIES
                a=render(points,policy;size,rule)
                b=render(points,policy;size,shift=(shift.du,shift.dv),rule)
                if policy=="production_point"
                    a==original.a && b==original.b || error("production baseline particle/image identity changed")
                end
                convergence=Dict{String,Any}("available"=>false,"reason"=>"point sampling; no quadrature")
                if policy=="wide_area"
                    ra=render(points,policy;size,rule=reference)
                    rb=render(points,policy;size,shift=(shift.du,shift.dv),rule=reference)
                    maximum_difference=max(maximum(abs,a.-ra),maximum(abs,b.-rb))
                    maximum_difference<=IMAGE_CONVERGENCE_TOL || error("pre-specified area integration convergence tolerance failed")
                    convergence=Dict("available"=>true,"reference_order"=>REFERENCE_ORDER,"maximum_image_difference"=>maximum_difference,
                        "absolute_tolerance"=>IMAGE_CONVERGENCE_TOL,"reference_image_a_sha256"=>V.pixel_digest(ra),
                        "reference_image_b_sha256"=>V.pixel_digest(rb),
                        "interpretation"=>"specified numerical refinement check; not a total certified error bound")
                end
                spec=deepcopy(original.spec)
                spec["image_a_sha256"]=V.pixel_digest(a);spec["image_b_sha256"]=V.pixel_digest(b)
                if policy!="production_point"
                    spec["generator_version"]="splitmix64-fixed-placements-rendering-contrast-1"
                    spec["renderer"]=policy=="wide_point" ? "bench point Gaussian; rounded ceil(6sigma) bbox" : "bench unit-area Gaussian integral; rounded ceil(6sigma) bbox"
                end
                id=policy=="production_point" ? (shift.id=="original" ? "baseline" : shift.id) : "$(shift.id)_$policy"
                audited=audit_renderer((;a,b,spec,truth=original.truth),id)
                images[policy]=(a,b);push!(captures,audited.capture)
                push!(rows,Dict("policy"=>policy,"scientific_row"=>audited.row,"renderer"=>renderer_metadata(policy,points,shift),
                    "quadrature_convergence"=>convergence,"algebraically_derived_predictor_error"=>audited.derived,
                    "derived_label"=>"full truth error minus observed primary residual; algebraic quantity, not a newly observed predictor trace",
                    "source_image_flux"=>Dict("a"=>sum(a),"b"=>sum(b),"normalization"=>"none; finite-frame and support losses retained")))
            end
            push!(groups,Dict("scene_seed"=>seed,"shift"=>shift.id,"renderers"=>rows,"common_primary"=>common_summary(captures),
                "oracle_deformation"=>oracle_deformation(points,shift,images;size,rule)))
        end
    end
    stable=U.stable_environment(environment)
    Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),"environment"=>environment,
        "source_and_environment_stable"=>stable,"provenance_status"=>stable ? "stable on-disk source/environment; fresh process required" : "changed; regenerate",
        "processing_calls"=>length(seeds)*length(shifts)*3,"scene_seeds"=>collect(seeds),"image_size"=>[size,size],"groups"=>groups,
        "quadrature"=>Dict("order"=>QUADRATURE_ORDER,"reference_order"=>REFERENCE_ORDER,"nodes"=>rule.nodes,"weights"=>rule.weights,
            "reference_nodes"=>reference.nodes,"reference_weights"=>reference.weights,
            "linear_algebra_backend"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),
            "rule"=>"Gauss-Legendre via stdlib symmetric tridiagonal eigendecomposition; unit pixel integration separable",
            "convergence"=>"16-versus32 numerical agreement and independent tests; no rigorous total floating-point error certificate"),
        "limitations"=>["Twelve clean processing calls diagnose deterministic sensitivity, not random uncertainty calibration.",
            "Renderer diameter3=4sigma; continuous Gaussian sigma=.75; pixel-area integration changes pixel response, without renormalization.",
            "The integer-half-warp shift is an oracle sentinel; actual estimated PIV predictors can still be fractional.",
            "No oracle predictor or oracle image is fed to accuracy PIV calls; original full truth errors and stored sigmas remain unchanged.",
            "Common primary populations and per-component UQ losses are explicit; overlapping nodes are not independent observations.",
            "Finite frame/bbox support and spline prefilter boundaries affect oracle comparisons; interior crops do not erase full-frame errors.",
            "Derived full-error minus primary-residual is algebraic, not a new observed predictor trace.",
            "No noise realizations, fitted floor, estimator/default change, performance sampling, GPU or experimental truth.",
            "On-disk identities do not attest loaded module code; run in a fresh process with frozen sources."])
end

function check_output(directory)
    target=U.resolved_path(directory);repository=realpath(ROOT);allowed=U.resolved_path(joinpath(ROOT,"bench","profile-output"))
    U.within(target,repository) && !U.within(target,allowed) && throw(ArgumentError("repository output must stay under bench/profile-output"))
    protected=[@__FILE__,joinpath(@__DIR__,"diagnostic_uncertainty.jl"),joinpath(@__DIR__,"validation_uncertainty.jl"),
        joinpath(@__DIR__,"validation_scorecard.jl"),joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")]
    if Base.active_project()!==nothing
        append!(protected,[Base.active_project(),joinpath(dirname(Base.active_project()),"Manifest.toml")])
    end
    for root in ("src","ext","test","docs","reference"),(dir,_,files) in walkdir(joinpath(ROOT,root))
        append!(protected,joinpath.(dir,files))
    end
    paths=[joinpath(target,"rendering_uncertainty.toml"),joinpath(target,"rendering_uncertainty.md")]
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
    io=IOBuffer();println(io,"<!-- $MARKER -->\n# Rendering/interpolation uncertainty diagnostic\n\nProcessing calls: ",report["processing_calls"],". Provenance: ",report["provenance_status"],".\n")
    println(io,"Complete recipes/input identities, full-error populations, common losses and separate oracle lane: `rendering_uncertainty.toml`.\n")
    println(io,"| Scene | Shift | Renderer | Component | Primary / unmasked | Full-error bias / RMS | Stored UQ / zero sigma | Full-error 2sigma | Common primary |")
    println(io,"|---|---|---|---|---:|---:|---:|---|---:|")
    for group in report["groups"],row in group["renderers"],k in ("u","v")
        metrics=row["scientific_row"]["full_error_metrics"];c=metrics["components"][k];m=c["primary_error"];p=c["coverage"]["2"]
        bias=m["available"] ? "$(round(m["mean"];sigdigits=6)) / $(round(m["rms"];sigdigits=6))" : "unavailable"
        coverage=p["available"] ? "$(p["covered_count"])/$(p["denominator"])" : "unavailable ($(p["covered_count"])/$(p["denominator"]); $(p["reason"]))"
        println(io,"| $(group["scene_seed"]) | $(group["shift"]) | $(row["policy"]) | $k | $(metrics["counts"]["primary_valid"])/$(metrics["counts"]["unmasked"]) | $bias | $(c["counts"]["uq_available"])/$(c["counts"]["sigma_zero"]) | $coverage | $(group["common_primary"]["common_primary"]) |")
    end
    println(io,"\n| Scene | Shift | Component | Common / grid | Primary losses P/W/A | Common UQ P/W/A | Wide point minus production bias / RMS | Area minus wide point bias / RMS |")
    println(io,"|---|---|---|---:|---|---|---:|---:|")
    moment_text(m)=m["available"] ? "$(round(m["mean"];sigdigits=6)) / $(round(m["rms"];sigdigits=6))" : "unavailable ($(m["finite_count"])/$(m["count"]))"
    for group in report["groups"],k in ("u","v")
        common=group["common_primary"]
        losses=join([string(row["primary_lost_to_common"]) for row in common["renderers"]],"/")
        uq=join([string(row["common_metrics"][k]["counts"]["uq_available"]) for row in common["renderers"]],"/")
        differences=common["deterministic_differences"]
        println(io,"| $(group["scene_seed"]) | $(group["shift"]) | $k | $(common["common_primary"])/$(common["grid_nodes"]) | $losses | $uq | $(moment_text(differences[1]["errors"][k])) | $(moment_text(differences[2]["errors"][k])) |")
    end
    println(io,"\nP/W/A denotes production point, wide point and wide area. Common metrics select primary nodes independently of sigma; each common UQ count has the common-primary denominator. Deterministic renderer differences have no independent-noise sigma quadrature.\n")
    println(io,"\n| Scene | Shift | Oracle model/support | Population | Pixels | Warp A vs midpoint RMS | Warp B vs midpoint RMS |")
    println(io,"|---|---|---|---|---:|---:|---:|")
    for group in report["groups"],lane in group["oracle_deformation"]["lanes"],p in lane["comparisons"]
        full=startswith(lane["input_support"],"all in-frame")
        a=p[full ? "sampled_full_field_warp_a_minus_midpoint" : "bounded_warp_a_minus_midpoint"]
        b=p[full ? "sampled_full_field_warp_b_minus_midpoint" : "bounded_warp_b_minus_midpoint"]
        number(m)=m["available"] ? string(round(m["rms"];sigdigits=6)) : "unavailable"
        println(io,"| $(group["scene_seed"]) | $(group["shift"]) | $(lane["model"]) / $(full ? "full sampled field" : "bbox") | $(p["population"]) | $(p["pixels"]) | $(number(a)) | $(number(b)) |")
    end
    println(io,"\nOracle images/known shifts are separate diagnostic inputs, never substituted into PIV accuracy calls. Full-frame and interior denominators remain explicit. No calibration or estimator-defect inference follows from these contrasts.\n")
    foreach(l->println(io,"- ",l),report["limitations"])
    String(take!(io))
end
function write_report(directory,report)
    paths=check_output(directory)
    report["schema_version"]==SCHEMA || throw(ArgumentError("unknown rendering schema"))
    report["source_and_environment_stable"]===true || throw(ArgumentError("refusing unstable evidence"))
    io=IOBuffer();println(io,"# $MARKER");TOML.print(io,report;sorted=true)
    toml=String(take!(io));markdown=markdown_report(report)
    mkpath(dirname(first(paths)));write(paths[1],toml);write(paths[2],markdown)
    paths
end
function main(args=ARGS)
    output=joinpath(ROOT,"bench","profile-output","rendering-uncertainty")
    for arg in args
        if arg=="--help"
            println("julia --project=. bench/rendering_uncertainty.jl [--output=directory] # fixed12 PIV calls, no timing samples")
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
    RenderingUncertainty.main()
end
