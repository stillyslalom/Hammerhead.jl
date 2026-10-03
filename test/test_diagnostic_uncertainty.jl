using Test, Hammerhead, TOML
using KernelAbstractions: CPU, synchronize
include(joinpath(@__DIR__,"..","bench","diagnostic_uncertainty.jl"))

# Independent reference: enumerate *all* ordered pixel pairs, then bucket
# their separation. No production offset/ring construction or covariance loop.
function diagnostic_covariance_reference(field,max_offset=4)
    mu=sum(Float64.(field))/length(field)
    sums=Dict((dr,dc)=>0.0 for dr in -max_offset:max_offset for dc in -max_offset:max_offset)
    for b in CartesianIndices(field),a in CartesianIndices(field)
        dr,dc=Tuple(b-a)
        haskey(sums,(dr,dc)) || continue
        sums[(dr,dc)]+=(Float64(field[a])-mu)*(Float64(field[b])-mu)
    end
    sums
end
function diagnostic_signal_reference(A,B)
    nr,nc=size(A)
    d=[A[r,c]*B[r,c+1]-A[r,c+1]*B[r,c] for r in 1:nr,c in 1:nc-1]
    smooth=similar(d)
    for p in CartesianIndices(d)
        r,c=Tuple(p)
        smooth[p]=(d[r,max(1,c-1)]+2d[p]+d[r,min(size(d,2),c+1)])/4
    end
    C0=sum(A[r,c]*B[r,c] for c in 1:nc-1 for r in 1:nr)
    Cp=sum(A[r,c]*B[r,c+1] for c in 1:nc-1 for r in 1:nr)
    Cm=sum(A[r,c+1]*B[r,c] for c in 1:nc-1 for r in 1:nr)
    (;C0,Cp,Cm,smooth,covariances=diagnostic_covariance_reference(smooth))
end
function diagnostic_fixture_stats(;C0=10.0,Cp=5.0,Cm=5.0,S00=1.0)
    stats=zeros(Hammerhead.UQ_NSTATS)
    stats[1:4]=[C0,Cp,Cm,S00]
    stats
end
function diagnostic_set_offset!(stats,offset,value)
    i=findfirst(==(offset),Hammerhead.UQ_OFFSETS)
    stats[4+i]=value
    stats
end

@testset "UQ independent covariance and numerical trace" begin
    D=DiagnosticUncertainty
    # Non-square and asymmetric signals catch transpose/support/sign mistakes.
    A=[0.4r+0.2c+sin(r*c) for r in 1:5,c in 1:6]
    B=[0.1r-0.3c+cos(r+2c) for r in 1:5,c in 1:6]
    for (a,b) in ((A,B),(permutedims(A),permutedims(B)))
        reference=diagnostic_signal_reference(a,b)
        scratch=Hammerhead.uncertainty_scratch(Float64,size(a))
        stats=zeros(Hammerhead.UQ_NSTATS)
        Hammerhead.uq_component!(stats,scratch.dC,scratch.dCs,a,b)
        @test stats[1:3] ≈ [reference.C0,reference.Cp,reference.Cm] rtol=1e-14
        @test scratch.dCs[1:size(reference.smooth,1),1:size(reference.smooth,2)] ≈ reference.smooth
        @test stats[4] ≈ reference.covariances[(0,0)] rtol=1e-14
        for (k,(dr,dc)) in enumerate(Hammerhead.UQ_OFFSETS)
            @test stats[4+k] ≈ reference.covariances[(dr,dc)] atol=1e-13
            @test reference.covariances[(dr,dc)] ≈ reference.covariances[(-dr,-dc)] atol=1e-13
        end
        @test stats[4]+2sum(stats[5:end]) ≈ sum(reference.covariances[o] for o in keys(reference.covariances)) atol=1e-12
    end
    # The independent full-offset sum of a fully covered zero-mean field is zero;
    # a truncated covariance estimate need not remain nonnegative.
    field=[1.0 -2.0 3.0; -1.0 4.0 -5.0]
    @test abs(sum(values(diagnostic_covariance_reference(field)))) < 1e-12

    # Check actual KA fill/stat kernels, not just finalized sigmas, on the same
    # signals. CPU gather's mean subtraction/apodization are covered separately.
    for T in (Float64,Float32)
        a=T.(A);b=T.(B)
        CA=reshape(complex.(a),size(a)...,1);CB=reshape(complex.(b),size(b)...,1)
        cache=zeros(T,1,2,6,6);means=zeros(2,1);stats=zeros(2,Hammerhead.UQ_NSTATS,1)
        ka=CPU()
        Hammerhead._ka_uq_fill!(ka)(cache,means,CA,CB,5,6;ndrange=(2,1))
        synchronize(ka)
        Hammerhead._ka_uq_stats!(ka)(stats,means,cache,CA,CB,5,6,1,0,false;ndrange=(2,Hammerhead.UQ_NSTATS,1))
        synchronize(ka)
        scratch=Hammerhead.uncertainty_scratch(T,(5,6))
        for (component,x,y) in ((1,a,b),(2,transpose(a),transpose(b)))
            cpu=zeros(Hammerhead.UQ_NSTATS)
            Hammerhead.uq_component!(cpu,scratch.dC,scratch.dCs,x,y)
            @test stats[component,:,1] == cpu
            @test isequal(D.trace_component(cpu;output_type=T)["sigma"],Float64(Hammerhead.finalize_uncertainty(T,cpu)))
        end
    end
    zero=diagnostic_fixture_stats(S00=0)
    @test D.trace_component(zero)["classification"]=="zero_covariance"
    cancelled=diagnostic_fixture_stats()
    diagnostic_set_offset!(cancelled,(1,0),0.5)
    diagnostic_set_offset!(cancelled,(-1,1),-1.0)
    @test D.trace_component(cancelled)["classification"]=="exact_covariance_cancellation"
    negative=copy(cancelled);diagnostic_set_offset!(negative,(-1,1),-2.0)
    trace=D.trace_component(negative)
    @test trace["classification"]=="negative_variance_clamp" && trace["sigma"]==0
    @test trace["pre_clamp_variance"]==-2 && trace["variance_was_negative"]
    @test trace["rings"][1]["included"] && !trace["rings"][2]["included"]
    # Exact threshold is included; below it stops before any ring contribution.
    for (v,included) in ((prevfloat(0.05),false),(0.05,true),(nextfloat(0.05),true))
        s=diagnostic_fixture_stats();diagnostic_set_offset!(s,(1,0),v)
        @test D.trace_component(s)["rings"][1]["included"]==included
    end
    # A later positive ring cannot restart summation after the first cutoff.
    stopped=diagnostic_fixture_stats();diagnostic_set_offset!(stopped,(4,0),1.0)
    @test all(!r["included"] for r in D.trace_component(stopped)["rings"])
    @test D.trace_component(stopped)["pre_clamp_variance"]==1
    tiny=diagnostic_fixture_stats(S00=1e-40)
    @test D.trace_component(tiny)["classification"]=="rounded_side_perturbation"
    loground=diagnostic_fixture_stats(C0=512,Cp=256,Cm=256,S00=2.0^-88)
    @test D.trace_component(loground)["classification"]=="rounded_log_difference"
    underflow=diagnostic_fixture_stats(S00=1e-20)
    @test D.trace_component(underflow;output_type=Float16)["classification"]=="output_type_underflow"
    @test D.trace_component(underflow)["sigma"]>0
    for (s,reason) in ((diagnostic_fixture_stats(C0=0),"nonpositive_C0"),
                       (diagnostic_fixture_stats(S00=121),"nonpositive_perturbed_side"),
                       (diagnostic_fixture_stats(C0=1),"invalid_peak_curvature"))
        t=D.trace_component(s)
        @test t["classification"]==reason && isnan(t["sigma"])
    end
    nanstats=diagnostic_fixture_stats();nanstats[4]=NaN
    @test D.trace_component(nanstats)["classification"]=="nonfinite_statistics"
    overflow=diagnostic_fixture_stats(S00=floatmax(Float64))
    diagnostic_set_offset!(overflow,(1,0),floatmax(Float64))
    @test D.trace_component(overflow)["classification"]=="nonfinite_variance_arithmetic"
    invalid_clamp=copy(negative);invalid_clamp[1]=0
    @test D.trace_component(invalid_clamp)["classification"]=="nonpositive_C0"
    @test D.trace_component(invalid_clamp)["variance_was_negative"] && isnan(D.trace_component(invalid_clamp)["sigma"])
    @test_throws ArgumentError D.trace_component(zeros(2))
    for s in (zero,cancelled,negative,tiny,underflow,diagnostic_fixture_stats(),nanstats)
        @test isequal(D.trace_component(s)["sigma"],Hammerhead.finalize_uncertainty(Float64,s))
    end
end

@testset "UQ diagnostic denominator and centered contracts" begin
    D=DiagnosticUncertainty
    m=D.CenteredMoment();foreach(x->D.add!(m,x),(1,2,3))
    @test D.summary(m)["centered_rms"] ≈ sqrt(2/3)
    a=D.CenteredMoment();D.add!(a,1)
    b=D.CenteredMoment();D.add!(b,2);D.add!(b,3)
    D.merge!(a,b);@test D.summary(a)==D.summary(m)
    @test !D.summary(D.CenteredMoment())["available"]
    D.add!(m,Inf);@test !D.summary(m)["available"] && D.summary(m)["count"]==4
    alt=D.supplementary([2.0,-2.0,3.0,4.0],[0.0,1.0,NaN,-1.0],[0.0,1.0,0.0,0.0],[NaN,1.0,0.0,0.0])
    c=alt["centered_error_diagnostic"]["metrics"]
    @test c["coverage"]["2"]["denominator"]==2 && c["counts"]["sigma_zero"]==1
    @test c["counts"]["zero_sigma_nonzero_error"]==1
    @test alt["primary_residual_quadrature_diagnostic"]["metrics"]["counts"]["uq_available"]==2
    @test alt["raw_eq4_quadrature_diagnostic"]["metrics"]["counts"]["uq_available"]==1
    @test alt["raw_eq4_quadrature_diagnostic"]["metrics"]["counts"]["primary_valid"]==4
    @test_throws ArgumentError D.supplementary([1.0],Float64[],[0.0],[0.0])
    @test_throws ArgumentError D.check_passes([last(D.U.controlled_passes())])
    bad=multipass_parameters([16,16];n_peaks=2,uncertainty=true)
    @test_throws ArgumentError D.check_passes(bad)
    @test_throws ArgumentError D.run_investigation(seeds=(true,))
    @test_throws ArgumentError D.run_investigation(seeds=(1,1))
    contrasts=D.contrast_cases()
    @test length(contrasts)==5
    @test contrasts[3].condition.settings==(du=2.0,)
    @test contrasts[4].condition.settings==(du=2.5,)
    @test contrasts[5].condition.settings==(dv=-1.25,)
    @test all(c->D.check_passes(c.passes)===nothing,contrasts)
    @test D.main(["--help"])===nothing
    @test_throws ArgumentError D.main(["--samples=2"])
    @test_throws ArgumentError D.check_output(joinpath(D.ROOT,"src"))
end

@testset "Retained CPU windows match final UQ sweep" begin
    D=DiagnosticUncertainty
    condition=(id="test_baseline",settings=(;),window=16)
    base=D.U.controlled_passes()
    fixed=multipass_parameters([32,16,16];padding=true,apodization=:gauss,n_peaks=1,
        replace_outliers=false,final=(uncertainty=true,max_iterations=3,convergence_tol=0.0))
    early=multipass_parameters([32,16,16];padding=true,apodization=:gauss,n_peaks=1,
        replace_outliers=false,final=(uncertainty=true,max_iterations=6,convergence_tol=100.0))
    rows=[D.audit_scene(condition,7321;passes=p,size=64).row for p in (base,fixed,early)]
    @test [row["diagnostics"]["passes"][end]["executed_iterations"] for row in rows]==[1,3,2]
    @test rows[2]["diagnostics"]["passes"][end]["stop_reason"]=="iteration_budget"
    @test rows[3]["diagnostics"]["passes"][end]["stop_reason"]=="tolerance_condition_met"
    @test rows[3]["diagnostics"]["passes"][end]["last_tolerance_check"]["observation"]["tolerance_met"]
    @test !rows[2]["diagnostics"]["passes"][end]["last_tolerance_check"]["available"]
    for row in rows,k in ("u","v")
        c=row["components"][k];full=row["full_error_metrics"]["components"][k]
        @test row["reproduction"]["verified"] && c["reproduction_count"]>0
        @test sum(values(c["classification_counts"]))==full["counts"]["primary_valid"]
        @test c["reproduction_count"]==full["counts"]["primary_valid"]
        @test full["counts"]["primary_valid"]==full["counts"]["uq_available"]+full["counts"]["uq_nonfinite"]+full["counts"]["uq_negative"]
        @test full["coverage"]["2"]["denominator"]==full["counts"]["uq_available"]
        @test sum(get(c["classification_counts"],reason,0) for reason in D.ZERO_CLASSIFICATIONS)==full["counts"]["sigma_zero"]
    end
    scene=D.V.synthetic_scene(seed=7321,size=64)
    result=run_piv(scene.a,scene.b,base;threaded=false)
    @test rows[1]["full_error_metrics"]==D.U.metrics(result,scene.truth).data
    repeated=D.audit_scene(condition,7321;size=64).row
    @test isequal(repeated,rows[1]) # scalar traces deliberately retain NaN
    noise=D.audit_scene((id="test_noise",settings=(;noise=0.03),window=16),7322;size=64).row
    @test noise["reproduction"]["verified"]
    large=D.audit_scene((id="test_window32",settings=(;),window=32),7321;size=96).row
    @test large["reproduction"]["verified"]
    empty=D.audit_scene((id="empty",settings=(;density=0.0),window=16),7321;size=64).row
    @test empty["full_error_metrics"]["counts"]["primary_valid"]==0
    @test isempty(empty["components"]["u"]["classification_counts"])
    # Actual nested examples contain NaN invalid terms, unlike a trivial empty
    # report. Ensure their numerical classifications survive TOML serialization.
    group=Dict("condition"=>"test","replicates"=>[rows[1]],"full_error_pooled"=>rows[1]["full_error_metrics"],
        "classification_counts"=>Dict(k=>rows[1]["components"][k]["classification_counts"] for k in ("u","v")),
        "centered_moments_pooled"=>Dict(k=>Dict("uq_subset"=>rows[1]["components"][k]["uq_subset_centered_moments"]) for k in ("u","v")))
    report=Dict("schema_version"=>D.SCHEMA,"source_and_environment_stable"=>true,
        "provenance_status"=>"test only","groups"=>[group],"contrasts"=>[rows[2]],"limitations"=>["test only"])
    mktempdir() do dir
        paths=D.write_report(dir,report)
        loaded=TOML.parsefile(paths[1])
        @test isequal(loaded["groups"][1]["replicates"][1],rows[1])
        @test isequal(loaded["contrasts"][1],rows[2])
    end
end

@testset "Diagnostic report output preservation" begin
    D=DiagnosticUncertainty
    report=Dict("schema_version"=>D.SCHEMA,"source_and_environment_stable"=>true,
        "provenance_status"=>"test only","groups"=>Any[],"contrasts"=>Any[],"limitations"=>["test only"])
    # Hardlinks require the source volume; use an admitted report directory.
    alias_parent=mkpath(joinpath(D.ROOT,"bench","profile-output"))
    mktempdir(alias_parent) do dir
        paths=D.write_report(joinpath(dir,"report"),report)
        @test TOML.parsefile(paths[1])["schema_version"]==D.SCHEMA
        @test occursin("Original full-error coverage",read(paths[2],String))
        @test D.write_report(joinpath(dir,"report"),report)==paths
        unrelated=joinpath(dir,"unrelated");mkpath(unrelated)
        write(joinpath(unrelated,"diagnostic_uncertainty.md"),"user content")
        @test_throws ArgumentError D.write_report(unrelated,report)
        @test read(joinpath(unrelated,"diagnostic_uncertainty.md"),String)=="user content"
        @test !isfile(joinpath(unrelated,"diagnostic_uncertainty.toml"))
        unstable=copy(report);unstable["source_and_environment_stable"]=false
        @test_throws ArgumentError D.write_report(joinpath(dir,"unstable"),unstable)
        @test !ispath(joinpath(dir,"unstable"))
        aliases=joinpath(dir,"alias");mkpath(aliases)
        source=joinpath(D.ROOT,"bench","diagnostic_uncertainty.jl")
        original=read(source)
        @test D.check_output(aliases) isa Vector
        hardlink(source,joinpath(aliases,"diagnostic_uncertainty.toml"))
        @test Base.samefile(source,joinpath(aliases,"diagnostic_uncertainty.toml"))
        alias_error=try D.write_report(aliases,report); nothing catch error; error end
        @test alias_error isa ArgumentError && occursin("aliases source",sprint(showerror,alias_error))
        @test read(source)==original
        same=joinpath(dir,"same");mkpath(same)
        a=joinpath(same,"diagnostic_uncertainty.toml");write(a,"# $(D.MARKER)\n")
        hardlink(a,joinpath(same,"diagnostic_uncertainty.md"))
        @test_throws ArgumentError D.write_report(same,report)
        project_alias=joinpath(dir,"project-alias");mkpath(project_alias)
        project=joinpath(D.ROOT,"Project.toml");project_bytes=read(project)
        @test D.check_output(project_alias) isa Vector
        hardlink(project,joinpath(project_alias,"diagnostic_uncertainty.toml"))
        @test Base.samefile(project,joinpath(project_alias,"diagnostic_uncertainty.toml"))
        alias_error=try D.write_report(project_alias,report); nothing catch error; error end
        @test alias_error isa ArgumentError && occursin("aliases source",sprint(showerror,alias_error))
        @test read(project)==project_bytes
        dangling=joinpath(dir,"dangling");mkpath(dangling)
        escaped=joinpath(dir,"escaped.toml")
        try
            symlink(escaped,joinpath(dangling,"diagnostic_uncertainty.toml"))
            @test_throws ArgumentError D.write_report(dangling,report)
            @test !ispath(escaped)
        catch error
            error isa Base.IOError || rethrow()
        end
    end
end
