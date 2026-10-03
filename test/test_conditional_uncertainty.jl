using Test, Hammerhead, TOML, Statistics
include(joinpath(@__DIR__,"..","bench","conditional_uncertainty.jl"))
const CU=ConditionalUncertainty

# Independent oracle: enumerate every ordered pixel pair, including boundaries.
function conditional_reference_stats(field)
    z=Float64.(field).-mean(field)
    stats=zeros(Hammerhead.UQ_NSTATS);stats[1:3]=[1000,500,500]
    stats[4]=sum(abs2,z)
    for (k,(dr,dc)) in enumerate(Hammerhead.UQ_OFFSETS)
        stats[4+k]=sum(z[a]*z[b] for a in CartesianIndices(z),b in CartesianIndices(z) if Tuple(b-a)==(dr,dc);init=0.0)
    end
    stats
end
function conditional_block_reference(field)
    z=Float64.(field).-mean(field)
    L=5;nr,nc=size(z)
    total=0.0
    # Origins outside the image must contribute their boundary partial blocks.
    for rs in 2-L:nr,cs in 2-L:nc
        block=0.0
        for r in rs:rs+L-1,c in cs:cs+L-1
            1<=r<=nr && 1<=c<=nc && (block+=z[r,c])
        end
        total+=block^2/L^2
    end
    total
end
function conditional_capture(errors;primary=trues(size(errors)),sigma=ones(size(errors)))
    capture=CU.Capture(size(errors));capture.primary.=primary
    for k in ("u","v")
        capture.errors[k].=errors;capture.sigmas[k].=sigma;capture.bartlett[k].=sigma
    end
    capture
end

@testset "Conditional centered noise streams" begin
    clean=CU.V.synthetic_scene(;seed=7321,size=64)
    before=(copy(clean.a),copy(clean.b),deepcopy(clean.spec))
    noise=CU.noisy_scene(clean,0.03,1);again=CU.noisy_scene(clean,0.03,1)
    @test noise.a==again.a && noise.b==again.b && noise.spec==again.spec
    @test (clean.a,clean.b,clean.spec)==before
    @test all(abs.(noise.a.-clean.a).<=0.03)
    @test all(abs.(noise.b.-clean.b).<=0.03)
    @test any(noise.a.<0) && any(noise.b.<0)
    @test noise.spec["noise_streams"]["A"]["negative_image_pixels"]==count(noise.a.<0)
    @test noise.a.-clean.a != noise.b.-clean.b
    @test noise.spec["noise_theoretical_variance"]==0.03^2/3
    @test noise.spec["clean_image_a_sha256"]==clean.spec["image_a_sha256"]
    @test noise.spec["image_a_sha256"]==CU.V.pixel_digest(noise.a)
    @test length(unique(CU.stream_seed(s,a,r,image) for s in (7321,7322),a in (0.01,0.03),r in 1:6,image in ("A","B")))==48
    @test CU.noisy_scene(clean,0.03,2).a!=noise.a
    @test_throws ArgumentError CU.noisy_scene(clean,0.0,1)
    @test_throws ArgumentError CU.noisy_scene(clean,Inf,1)
    @test_throws ArgumentError CU.noisy_scene(clean,0.03,true)
    @test_throws ArgumentError CU.stream_seed(7321,0.03,1,"C")
end

@testset "Bartlett covariance and boundary block identity" begin
    for field in ([sin(0.71r+1.13c)+0.1r*c for r in 1:7,c in 1:11],
                  [(-1.0)^(r+c)*(r+2c) for r in 1:4,c in 1:9],zeros(3,8),[3.0;;])
        for f in (field,permutedims(field))
            stats=conditional_reference_stats(f)
            candidate=CU.bartlett_variance(stats)
            independent=conditional_block_reference(f)
            @test candidate.value≈independent atol=1e-10 rtol=2e-13
            @test candidate.value>=-candidate.bound
            @test independent>=0
            @test isfinite(CU.bartlett_component(stats).sigma)
        end
    end
    # Verify identity on the *actual* smoothed dC used by production covariance.
    A=[sin(0.37r+0.23c)+0.2cos(r*c) for r in 1:8,c in 1:13]
    B=[cos(0.16r-0.51c)+0.01r*c for r in 1:8,c in 1:13]
    for (a,b) in ((A,B),(transpose(A),transpose(B)))
        scratch=Hammerhead.uncertainty_scratch(Float64,size(a));stats=zeros(Hammerhead.UQ_NSTATS)
        Hammerhead.uq_component!(stats,scratch.dC,scratch.dCs,a,b)
        field=copy(scratch.dCs[1:size(a,1),1:size(a,2)-1])
        @test CU.bartlett_variance(stats).value≈conditional_block_reference(field) atol=1e-10 rtol=2e-13
        @test stats[4:end]≈conditional_reference_stats(field)[4:end] atol=1e-11
    end
    stats=zeros(Hammerhead.UQ_NSTATS);stats[1:4]=[10,5,5,-1e-16]
    @test CU.bartlett_component(stats).reason=="negative_exceeds_summation_bound"
    @test isnan(CU.bartlett_component(stats).sigma)
    stats[4]=1.0;stats[5]=-1/(2*0.8)
    @test CU.bartlett_variance(stats).value≈0 atol=eps(Float64)
    stats[4]=prevfloat(1.0)
    @test CU.bartlett_component(stats).reason=="negative_within_summation_bound"
    @test isnan(CU.bartlett_component(stats).sigma)
    stats[4]=NaN
    @test CU.bartlett_component(stats).reason=="nonfinite_statistics"
    stats[4:end].=floatmax(Float64)
    @test CU.bartlett_component(stats).reason=="nonfinite_covariance_sum"
    @test_throws ArgumentError CU.bartlett_variance(zeros(3))
end

@testset "Conditional populations and disjoint paired differences" begin
    clean=conditional_capture(reshape([0.25,0.5,1.0,2.0],2,2))
    runs=[conditional_capture(reshape([Float64(r),2.0,3.0,4.0],2,2)) for r in 1:6]
    runs[1].primary[2]=false
    runs[2].sigmas["u"][1]=NaN
    runs[3].sigmas["v"][1]=0.0
    runs[4].negative["u"][1]=true
    clean.primary[3]=false
    summary=CU.conditional_summary(clean,runs)
    losses=summary["losses"]
    @test losses["grid_nodes"]==4
    @test losses["noisy_complete_primary"]==3
    @test losses["noisy_incomplete_primary"]==1
    @test losses["noisy_complete_and_clean_primary"]==2
    @test losses["noisy_complete_lost_clean_primary"]==1
    @test losses["per_realization_primary_lost_to_complete_case"]==[0,1,1,1,1,1]
    u=summary["components"]["u"]
    @test u["conditional_sample_variance"]["mean"]≈var(collect(1.0:6))/3
    @test u["conditional_mean_error"]["mean"]≈(3.5+3+4)/3
    @test u["nodes_with_clamp_and_positive_conditional_variance"]==1
    @test u["all_realizations_stored_uq_available_count"]==2
    @test u["all_realizations_bartlett_uq_available_count"]==3
    pair=CU.paired_summary(runs[1],runs[2])
    @test pair["common_primary"]==3
    @test pair["only_second_primary"]==1
    p=pair["components"]["u"]["full_difference_stored_sigma"]
    @test p["counts"]["primary_valid"]==3
    @test p["counts"]["uq_available"]==2
    @test p["primary_error"]["mean"]≈-1/3
    @test p["counts"]["uq_nonfinite"]==1
    @test pair["components"]["u"]["difference_squared_over_two"]["mean"]≈1/6
    # No fitted centering: even identical zero sigmas retain nonzero differences.
    a=conditional_capture(fill(1.0,1,1);sigma=zeros(1,1))
    b=conditional_capture(fill(2.0,1,1);sigma=zeros(1,1))
    p=CU.paired_summary(a,b)["components"]["u"]["full_difference_stored_sigma"]
    @test p["counts"]["zero_sigma_nonzero_error"]==1
    @test p["coverage"]["2"]["covered_count"]==0
    b.primary.=false
    @test CU.paired_summary(a,b)["components"]["u"]["full_difference_stored_sigma"]["coverage"]["2"]["available"]===false
    empty=CU.conditional_summary(a,[b,b])["components"]["u"]
    @test empty["conditional_noise_rms"]["available"]===false
    bad=conditional_capture(fill(Inf,1,1))
    arithmetic=CU.conditional_summary(a,[a,bad])["components"]["u"]
    @test arithmetic["conditional_arithmetic_unavailable_count"]==1
    @test arithmetic["conditional_arithmetic_available_count"]==0
    @test CU.variance_rms([floatmax(Float64),floatmax(Float64)])["value"]≈sqrt(floatmax(Float64))
    @test CU.variance_rms([0.0,0.0])["value"]==0.0
    @test CU.variance_rms([Inf])["available"]===false
    @test CU.variance_rms([-1.0])["available"]===false
    @test CU.variance_rms(Float64[])["available"]===false
    huge=conditional_capture(fill(-floatmax(Float64),1,1))
    large=conditional_capture(fill(floatmax(Float64),1,1))
    @test CU.paired_summary(huge,large)["components"]["u"]["full_difference_stored_sigma"]["counts"]["error_arithmetic_unavailable"]==1
    @test_throws ArgumentError CU.conditional_summary(clean,CU.Capture[])
end

@testset "Supplied-pair audit preserves original scientific rows" begin
    condition=(id="baseline",settings=(;),window=16)
    original=CU.D.audit_scene(condition,7321;size=64)
    scene=CU.V.synthetic_scene(;seed=7321,size=64)
    captured=NamedTuple[]
    pair=CU.D.audit_pair(scene,condition,7321;on_component=x->push!(captured,x))
    @test isequal(original.row,pair.row)
    @test !isempty(captured)
    @test length(captured)==2original.row["reproduction"]["primary_nodes"]
    @test all(x->length(x.statistics)==Hammerhead.UQ_NSTATS,captured)
    @test all(x->isequal(x.sigma,CU.D.trace_component(x.statistics)["sigma"]),captured)
    @test captured[1].statistics!==captured[2].statistics
    traced=CU.audit_capture(scene,"baseline")
    compared=deepcopy(traced.row);delete!(compared,"bartlett_comparator")
    @test isequal(compared,original.row)
    @test count(traced.capture.primary)==original.row["reproduction"]["primary_nodes"]
    @test_throws ArgumentError CU.D.audit_pair((;a=Float32.(scene.a),b=scene.b,spec=scene.spec,truth=scene.truth),condition,7321)
end

@testset "Conditional report and protected destinations" begin
    # Small geometry retains the exact six-realization/pairing protocol.
    report=CU.run_study(;scene_seeds=(7321,),amplitudes=(0.01,),size=64)
    @test report["processing_calls"]==7
    @test length(report["groups"][1]["realizations"])==6
    @test length(report["groups"][1]["disjoint_pairs"])==3
    @test all(r->r["reproduction"]["verified"],report["groups"][1]["realizations"])
    @test report["source_and_environment_stable"]===true
    @test report["bartlett"]["production_ring_cutoff"]===false
    @test_throws ArgumentError CU.run_study(;realizations=3)
    @test_throws ArgumentError CU.run_study(;scene_seeds=(true,))
    @test_throws ArgumentError CU.run_study(;amplitudes=(NaN,))
    @test_throws ArgumentError CU.check_output(joinpath(CU.ROOT,"docs"))
    mktempdir() do directory
        paths=CU.write_report(directory,report)
        @test TOML.parsefile(paths[1])["processing_calls"]==7
        @test occursin("original full-error coverage",read(paths[2],String))
        @test occursin("Paired Bartlett 2sigma",read(paths[2],String))
        @test occursin("Original full-error 2sigma",read(paths[2],String))
        bytes=read.(paths)
        bad=deepcopy(report);bad["source_and_environment_stable"]=false
        @test_throws ArgumentError CU.write_report(directory,bad)
        @test read.(paths)==bytes
        bad=deepcopy(report);bad["schema_version"]="future"
        @test_throws ArgumentError CU.write_report(directory,bad)
        @test read.(paths)==bytes
        write(paths[2],"unrelated data")
        @test_throws ArgumentError CU.write_report(directory,report)
        @test read(paths[2],String)=="unrelated data"
    end
    mktempdir() do directory
        alias=joinpath(directory,"source");mkpath(alias)
        source=joinpath(CU.ROOT,"bench","conditional_uncertainty.jl");bytes=read(source)
        hardlink(source,joinpath(alias,"conditional_uncertainty.toml"))
        @test_throws ArgumentError CU.write_report(alias,report)
        @test read(source)==bytes
        alias=joinpath(directory,"project");mkpath(alias)
        source=joinpath(CU.ROOT,"Project.toml");bytes=read(source)
        hardlink(source,joinpath(alias,"conditional_uncertainty.toml"))
        @test_throws ArgumentError CU.write_report(alias,report)
        @test read(source)==bytes
        alias=joinpath(directory,"same");mkpath(alias)
        a=joinpath(alias,"conditional_uncertainty.toml");write(a,"# $(CU.MARKER)\n")
        hardlink(a,joinpath(alias,"conditional_uncertainty.md"))
        @test_throws ArgumentError CU.write_report(alias,report)
        @test read(a,String)=="# $(CU.MARKER)\n"
        broken=joinpath(directory,"broken");mkpath(broken)
        escaped=joinpath(directory,"missing.toml")
        try
            symlink(escaped,joinpath(broken,"conditional_uncertainty.toml"))
            @test_throws ArgumentError CU.write_report(broken,report)
            @test !ispath(escaped)
        catch error
            error isa Base.IOError || rethrow()
        end
    end
end
