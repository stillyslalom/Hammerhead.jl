using Test, Hammerhead, TOML, Statistics
include(joinpath(@__DIR__,"..","bench","rendering_uncertainty.jl"))
const RU=RenderingUncertainty

# Independent BigFloat composite Simpson integration. Neither eigen rules nor
# the separable renderer are used. This is numerical reference evidence, not
# a rigorous certificate for the Float64 quadrature/exp implementation.
function rendering_simpson_reference(pixel,center,sigma;n=4096)
    setprecision(BigFloat,192) do
        a=BigFloat(pixel)-big(0.5);b=BigFloat(pixel)+big(0.5)
        h=(b-a)/n;c=BigFloat(center);s=BigFloat(sigma)
        f(x)=exp(-(x-c)^2/(2s^2))
        result=f(a)+f(b)
        for i in 1:n-1
            result+=(isodd(i) ? 4 : 2)*f(a+i*h)
        end
        Float64(result*h/3)
    end
end
function rendering_toy_capture(errors;primary=trues(size(errors)),sigma=ones(size(errors)))
    (;primary,errors=Dict(k=>copy(errors) for k in ("u","v")),
        sigmas=Dict(k=>copy(sigma) for k in ("u","v")),residuals=Dict(k=>zeros(size(errors)) for k in ("u","v")))
end

@testset "Pixel-area quadrature independent numerical evidence" begin
    for n in (RU.QUADRATURE_ORDER,RU.REFERENCE_ORDER)
        rule=RU.quadrature_rule(n)
        @test issorted(rule.nodes)
        @test all(abs.(rule.nodes).<1)
        @test all(rule.weights.>0)
        @test sum(rule.weights)≈2 atol=2e-14
        @test rule.nodes≈-reverse(rule.nodes) atol=2e-14
        @test rule.weights≈reverse(rule.weights) atol=2e-14
        for power in 0:2n-1
            exact=isodd(power) ? 0.0 : 2/(power+1)
            @test sum(rule.weights.*rule.nodes.^power)≈exact atol=5e-14
        end
    end
    coarse=RU.quadrature_rule(RU.QUADRATURE_ORDER);fine=RU.quadrature_rule(RU.REFERENCE_ORDER)
    for (pixel,center,sigma) in ((0.0,0.0,0.75),(1.0,0.25,0.75),(0.0,0.5,0.75),
                               (-2.0,0.25,0.75),(4.0,0.1,0.75),(0.0,0.3,0.4))
        a=RU.pixel_integral(pixel,center,sigma,coarse)
        b=RU.pixel_integral(pixel,center,sigma,fine)
        independent=rendering_simpson_reference(pixel,center,sigma)
        @test a≈b atol=1e-13
        @test a≈independent atol=1e-12
        @test 0<=a<=1
    end
    @test_throws ArgumentError RU.quadrature_rule(1)
    @test_throws ArgumentError RU.quadrature_rule(true)
    @test_throws ArgumentError RU.pixel_integral(0,0,0,coarse)
end

@testset "Fixed placement, support, flux, symmetry and translation" begin
    points=RU.particle_placements(7321;size=64)
    @test points==RU.particle_placements(7321;size=64)
    @test RU.placement_digest(points)==RU.placement_digest(copy(points))
    for shift in RU.SHIFTS
        original=RU.V.synthetic_scene(;seed=7321,size=64,du=shift.du,dv=shift.dv)
        @test RU.render(points,"production_point";size=64)==original.a
        @test RU.render(points,"production_point";size=64,shift=(shift.du,shift.dv))==original.b
    end
    isolated=[(32.25,31.75)]
    point=RU.render(isolated,"production_point";size=64)
    wide=RU.render(isolated,"wide_point";size=64)
    @test all(wide.>=point)
    @test wide[point.>0]==point[point.>0]
    @test count(point.>0)==49
    @test count(wide.>0)==121
    meta=RU.renderer_metadata("production_point",isolated,RU.SHIFTS[1])
    @test meta["integer_radius"]==3
    @test meta["nominal_support_sigma"]==3
    @test meta["normalization"]=="none"
    for (x,y) in ((32.0,32.0),(32.25,31.75),(31.5,32.5))
        full=RU.render([(x,y)],"wide_area";size=64,full_support=true)
        @test sum(full)≈2pi*RU.SIGMA^2 rtol=2e-13
        @test all(full.>=0)
        transpose_image=RU.render([(y,x)],"wide_area";size=64,full_support=true)
        @test full≈permutedims(transpose_image) atol=1e-15
    end
    for policy in RU.POLICIES
        original=RU.render([(32.125,31.25)],policy;size=64)
        shifted=RU.render([(32.125,31.25)],policy;size=64,shift=(2.0,-2.0))
        @test original[3:64,1:62]≈shifted[1:62,3:64] atol=3e-15
    end
    boundary=RU.render([(1.0,1.0)],"wide_area";size=64)
    @test sum(boundary)<2pi*RU.SIGMA^2 # frame clipping is retained, never normalized
    @test_throws ArgumentError RU.render(points,"unknown";size=64)
    @test_throws ArgumentError RU.particle_placements(true)
end

@testset "Known translation oracle sign and population limits" begin
    # Affine images interpolate exactly away from boundaries and distinguish
    # all signs and row/column conventions independently of particle rendering.
    du,dv=2.0,-2.0
    a=[2.0c+3.0r for r in 1:64,c in 1:64]
    b=[2.0(c-du)+3.0(r-dv) for r in 1:64,c in 1:64]
    wa,wb=RU.oracle_warp(a,b,du,dv)
    expected=[2.0(c-du/2)+3.0(r-dv/2) for r in 1:64,c in 1:64]
    @test wa[17:48,17:48]≈expected[17:48,17:48] atol=1e-11
    @test wb[17:48,17:48]≈expected[17:48,17:48] atol=1e-11
    @test wa[17:48,17:48]≈wb[17:48,17:48] atol=1e-11
    du,dv=2.25,-1.5
    a=[2.0c+3.0r for r in 1:64,c in 1:64]
    b=[2.0(c-du)+3.0(r-dv) for r in 1:64,c in 1:64]
    wa,wb=RU.oracle_warp(a,b,du,dv)
    expected=[2.0(c-du/2)+3.0(r-dv/2) for r in 1:64,c in 1:64]
    @test wa[17:48,17:48]≈expected[17:48,17:48] atol=1e-8
    @test wb[17:48,17:48]≈expected[17:48,17:48] atol=1e-8
    points=[(31.125,30.25),(40.75,43.125)]
    shift=RU.SHIFTS[2]
    images=Dict(policy=>(RU.render(points,policy;size=64),RU.render(points,policy;size=64,shift=(shift.du,shift.dv))) for policy in RU.POLICIES)
    oracle=RU.oracle_deformation(points,shift,images;size=64)
    @test length(oracle["lanes"])==5
    @test oracle["interior_margin_pixels"]==16
    for lane in oracle["lanes"]
        @test lane["comparisons"][1]["pixels"]==64^2
        @test lane["comparisons"][2]["pixels"]==32^2
        if startswith(lane["input_support"],"all in-frame")
            @test lane["comparisons"][2]["sampled_full_field_warp_a_minus_midpoint"]["maximum_absolute_error"]<1e-13
            @test lane["comparisons"][2]["sampled_full_field_warp_b_minus_midpoint"]["maximum_absolute_error"]<1e-13
        end
    end
    empty=RU.image_difference(zeros(2,2),ones(2,2),falses(2,2))
    @test empty["available"]===false && empty["selected_pixels"]==0
end

@testset "Rendering common primary and UQ denominators" begin
    a=rendering_toy_capture(reshape([1.0,2.0,3.0,4.0],2,2))
    b=rendering_toy_capture(reshape([2.0,3.0,4.0,5.0],2,2))
    c=rendering_toy_capture(reshape([4.0,5.0,6.0,7.0],2,2))
    b.primary[2]=false;c.primary[3]=false
    b.sigmas["u"][1]=NaN;c.sigmas["v"][1]=0.0
    summary=RU.common_summary([a,b,c])
    @test summary["grid_nodes"]==4
    @test summary["common_primary"]==2
    @test summary["not_common_primary"]==2
    @test [r["primary_lost_to_common"] for r in summary["renderers"]]==[2,1,1]
    u=summary["renderers"][2]["common_metrics"]["u"]
    @test u["counts"]["primary_valid"]==2
    @test u["counts"]["uq_available"]==1
    @test u["counts"]["uq_nonfinite"]==1
    v=summary["renderers"][3]["common_metrics"]["v"]
    @test v["counts"]["zero_sigma_nonzero_error"]==1
    @test summary["deterministic_differences"][1]["errors"]["u"]["mean"]==1.0
    @test summary["deterministic_differences"][2]["errors"]["u"]["mean"]==2.0
    c.primary.=false
    @test RU.common_summary([a,b,c])["renderers"][1]["common_metrics"]["u"]["coverage"]["2"]["available"]===false
end

@testset "Rendering baseline exact scientific row and guarded report" begin
    shift=RU.SHIFTS[1]
    # Three small PIV calls use the unchanged recipe; final evidence is held for source freeze.
    report=RU.run_study(;seeds=(7321,),size=64,shifts=(shift,))
    @test report["processing_calls"]==3
    @test report["source_and_environment_stable"]===true
    group=only(report["groups"])
    baseline=group["renderers"][1]["scientific_row"]
    original=RU.D.audit_scene((id="baseline",settings=(;),window=16),7321;size=64).row
    @test isequal(baseline,original)
    @test baseline["inputs"]["generator_version"]=="splitmix64-gaussian-particles-1"
    @test all(r->r["scientific_row"]["inputs"]["generator_version"]=="splitmix64-fixed-placements-rendering-contrast-1",group["renderers"][2:3])
    @test all(r->r["scientific_row"]["reproduction"]["verified"],group["renderers"])
    @test group["renderers"][3]["quadrature_convergence"]["maximum_image_difference"]<=RU.IMAGE_CONVERGENCE_TOL
    @test occursin("algebraic",group["renderers"][1]["derived_label"])
    for renderer in group["renderers"],k in ("u","v")
        @test renderer["algebraically_derived_predictor_error"][k]["count"]==renderer["scientific_row"]["reproduction"]["primary_nodes"]
    end
    @test_throws ArgumentError RU.check_output(joinpath(RU.ROOT,"docs"))
    @test_throws ArgumentError RU.main(["--unrecognized"])
    # Hardlinks require the source volume; use an admitted report directory.
    alias_parent=mkpath(joinpath(RU.ROOT,"bench","profile-output"))
    mktempdir(alias_parent) do dir
        paths=RU.write_report(dir,report)
        loaded=TOML.parsefile(paths[1])
        @test isequal(loaded["groups"][1]["renderers"][1]["scientific_row"],baseline)
        @test occursin("Oracle images/known shifts",read(paths[2],String))
        @test occursin("Common UQ P/W/A",read(paths[2],String))
        bytes=read.(paths)
        bad=deepcopy(report);bad["source_and_environment_stable"]=false
        @test_throws ArgumentError RU.write_report(dir,bad)
        @test read.(paths)==bytes
        bad=deepcopy(report);bad["schema_version"]="future"
        @test_throws ArgumentError RU.write_report(dir,bad)
        @test read.(paths)==bytes
        write(paths[2],"user output")
        @test_throws ArgumentError RU.write_report(dir,report)
        @test read(paths[2],String)=="user output"
        alias=joinpath(dir,"alias");mkdir(alias)
        source=joinpath(RU.ROOT,"bench","rendering_uncertainty.jl");before=read(source)
        @test RU.check_output(alias) isa Vector
        hardlink(source,joinpath(alias,"rendering_uncertainty.toml"))
        @test Base.samefile(source,joinpath(alias,"rendering_uncertainty.toml"))
        alias_error=try RU.write_report(alias,report); nothing catch error; error end
        @test alias_error isa ArgumentError && occursin("aliases source",sprint(showerror,alias_error))
        @test read(source)==before
        same=joinpath(dir,"same");mkdir(same)
        p=joinpath(same,"rendering_uncertainty.toml");write(p,"# $(RU.MARKER)\n")
        hardlink(p,joinpath(same,"rendering_uncertainty.md"))
        @test_throws ArgumentError RU.write_report(same,report)
        @test read(p,String)=="# $(RU.MARKER)\n"
    end
end
