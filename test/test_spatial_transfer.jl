using Test, TOML, LinearAlgebra
include(joinpath(@__DIR__,"..","bench","spatial_transfer.jl"))
const ST=SpatialTransfer

@testset "Spatial-transfer analytic motion and unchanged renderer control" begin
    for wavelength in ST.WAVELENGTHS,y in (1.25,49.75,128.5,253.2)
        u,v=ST.motion(y,wavelength)
        midpoint=(12.5+u/2,y+v/2)
        @test collect(ST.midpoint_truth(midpoint...,wavelength))≈[u,v]
        @test u≈2.25+.5*sin(2pi*(y-128.5)/wavelength+pi/4)
        @test v==-1.5
    end
    control=ST.scene(7321,64;size=128,amplitude=0.0)
    original=ST.V.synthetic_scene(;seed=7321,size=128,density=.04,diameter=3.,du=2.25,dv=-1.5)
    @test control.a==original.a
    @test control.b==original.b
    @test isequal(control.spec,original.spec)
    @test control.truth(3.,4.)==(2.25,-1.5)
    sine=ST.scene(7321,64;size=128)
    @test sine.a==control.a
    @test sine.points==control.points
    @test sine.points==ST.particle_positions(7321,128)
    @test sine.spec["particle_count"]==round(Int,.04*128^2)
    @test sine.spec["particle_positions_sha256"]==ST.V.pixel_digest(sine.points)
    # Independent literal particle rendering verifies the prescribed B positions.
    expected=zeros(128,128)
    for i in axes(sine.points,1)
        x,y=sine.points[i,:]
        u=2.25+.5*sin(2pi*(y-64.5)/64+pi/4)
        Hammerhead.SyntheticData.generate_gaussian_particle!(expected,(x+u,y-1.5),3.)
    end
    @test all(axes(sine.points,1)) do i
        x,y=sine.points[i,:]
        u=2.25+.5*sin(2pi*(y-64.5)/64+pi/4)
        isapprox(sine.truth(x+u/2,y-.75)[1],u)
    end
    @test sine.b==expected
    @test sine.spec["image_b_sha256"]==ST.V.pixel_digest(expected)
    @test_throws ArgumentError ST.scene(true,64)
    @test_throws ArgumentError ST.scene(1,0)
    @test_throws ArgumentError ST.scene(1,64;amplitude=-1)
    @test_throws ArgumentError ST.scene(1,64;size=32)
end

@testset "Spatial-transfer independent harmonic fits and numerical refusal" begin
    rows=collect(64.5:8:184.5)
    for wavelength in ST.WAVELENGTHS
        theta=2pi .* ((rows .+ .75 .- 128.5)./wavelength) .+ pi/4
        values=2.25 .+ .35 .* sin.(theta) .- .2 .* cos.(theta)
        fit=ST.harmonic_fit(rows,values,wavelength)
        @test fit["available"]===true
        @test fit["rank"]==3
        @test fit["absolute_rank_threshold"]==16eps(Float64)*maximum(fit["singular_values"])
        @test fit["condition_number"]≈sqrt(2) atol=1e-14
        @test fit["intercept"]≈2.25 atol=1e-14
        @test fit["sine_coefficient"]≈.35 atol=1e-14
        @test fit["cosine_coefficient"]≈-.2 atol=1e-14
        @test fit["in_phase_gain"]≈.7 atol=1e-14
        @test fit["quadrature_gain"]≈-.4 atol=1e-14
        @test fit["amplitude_gain"]≈hypot(.7,.4) atol=1e-14
        @test fit["phase_radians"]≈atan(-.2,.35) atol=1e-14
        @test fit["fit_residual"]["rms"]<2e-15
        # Orthogonal finite Fourier sums are independent of the QR/SVD fit.
        @test fit["sine_coefficient"]≈2sum(values.*sin.(theta))/length(rows) atol=1e-14
        @test fit["cosine_coefficient"]≈2sum(values.*cos.(theta))/length(rows) atol=1e-14
        chosen=[1,2,4,7,8,12,15]
        subset=ST.harmonic_fit(rows[chosen],values[chosen],wavelength)
        @test subset["available"]===true
        @test subset["sine_coefficient"]≈.35 atol=2e-14
        @test subset["cosine_coefficient"]≈-.2 atol=2e-14
    end
    constant=ST.harmonic_fit(rows,fill(0.,16),64)
    @test constant["available"]===true
    @test constant["amplitude_gain"]==0
    @test constant["phase_response_available"]===false
    @test constant["phase_reason"]=="zero_harmonic_phase_undefined"
    control=ST.harmonic_fit(rows,fill(2.25,16),64;amplitude=0.)
    @test control["available"]===true
    @test control["normalized_response_available"]===false
    @test control["normalization_reason"]=="zero_prescribed_amplitude_control"
    @test control["phase_response_available"]===false
    @test control["harmonic_amplitude"]<2e-15
    @test ST.harmonic_fit(Float64[],Float64[],64)["reason"]=="too_few_samples"
    @test ST.harmonic_fit(rows[1:3],fill(NaN,3),64)["reason"]=="nonfinite_input"
    @test ST.harmonic_fit([1.,1.,1.],[1.,2.,3.],64)["reason"]=="rank_deficient"
    # At the 8 px-grid Nyquist wavelength, sine/cosine columns are dependent.
    @test ST.harmonic_fit(rows,zeros(16),16)["reason"]=="rank_deficient"
    clustered=64.5 .+ collect(0:7).*1e-5
    @test ST.harmonic_fit(clustered,ones(8),128)["reason"]=="ill_conditioned"
    @test ST.harmonic_fit(rows,fill(floatmax(Float64)/4,16),64)["available"]===true
    tiny=ST.harmonic_fit(rows,sin.(2pi .* ((rows .+ .75 .-128.5)./64).+pi/4),64;amplitude=nextfloat(0.))
    @test tiny["available"]===true
    @test tiny["normalized_response_available"]===false
    @test tiny["normalization_reason"]=="nonfinite_normalization_arithmetic"
    @test ST.harmonic_fit(fill(floatmax(Float64),4),ones(4),nextfloat(0.))["reason"]=="nonfinite_phase_arithmetic"
    theta=2pi .* ((rows .+ .75 .-128.5)./64) .+ pi/4
    @test ST.harmonic_fit(rows,floatmax(Float64).*sign.(sin.(theta)),64)["reason"]=="nonfinite_coefficient_arithmetic"
    quadrant_rows=((collect(0:3) .* (pi/2) .- pi/4) .* 64 ./ (2pi)) .+ 128.5 .- .75
    quadrant_values=(.8floatmax(Float64)).*[1.,1.,-1.,-1.]
    @test ST.harmonic_fit(quadrant_rows,quadrant_values,64)["reason"]=="nonfinite_harmonic_arithmetic"
    outlier=fill(floatmax(Float64),16);outlier[1]=-floatmax(Float64)
    @test ST.harmonic_fit(rows,outlier,64)["reason"]=="nonfinite_residual_arithmetic"
    @test_throws ArgumentError ST.harmonic_fit([1.],[1.,2.],64)
    @test_throws ArgumentError ST.harmonic_fit(rows,ones(16),0)
    @test_throws ArgumentError ST.harmonic_fit(rows,ones(16),64;amplitude=-1)
end

function spatial_test_result(x,y,u;v=fill(-1.5,size(u)),sigma=fill(.1,size(u)),outliers=falses(size(u)),mask=falses(size(u)),window=16)
    Hammerhead.PIVResult(Float64.(x),Float64.(y),Float64.(u),Float64.(v),ones(size(u)),ones(size(u)),
        Float64.(sigma),fill(.1,size(u)),outliers,mask,last(ST.parameters(window)))
end
@testset "Spatial-transfer exact grids, common populations and origin guard" begin
    a,b=ST.parameters(16),ST.parameters(32)
    @test ST.V.pass_recipe(a[1])==ST.V.pass_recipe(b[1])
    @test ST.V.pass_recipe(a[2])==ST.V.pass_recipe(b[2])
    @test [p.window_size for p in a]==[(64,64),(32,32),(16,16),(16,16)]
    @test [p.window_size for p in b]==[(64,64),(32,32),(32,32),(32,32)]
    for passes in (a,b)
        @test all(p->p.n_peaks==1 && !p.replace_outliers && p.max_iterations==1,passes)
        @test last(passes).window_size.-last(passes).overlap==(8,8)
        @test last(passes).uncertainty
        @test !any(p->p.uncertainty,passes[1:3])
    end
    grids=[Hammerhead.pass_grid(Float64,(256,256),last(p),nothing,.5) for p in (a,b)]
    coords=[Set((x,y) for x in g.x,y in g.y if 64<=x<=192 && 64<=y<=192) for g in grids]
    @test coords[1]==coords[2]
    @test length(coords[1])==256
    x=[64.5,72.5];y=[64.5,72.5,80.5,88.5]
    u=[ST.midpoint_truth(xx,yy,64)[1] for yy in y,xx in x]
    sigma=fill(.1,4,2);sigma[2,1]=NaN;sigma[3,1]=0
    first_result=spatial_test_result(x,y,u;sigma,outliers=BitMatrix([true false;false false;false false;false false]))
    second_result=spatial_test_result(x,y,u.+.1;outliers=BitMatrix([false false;false true;false false;false false]),window=32)
    evals=[(result=first_result,interior=trues(4,2),row=Dict("terminal_window_px"=>16)),
        (result=second_result,interior=trues(4,2),row=Dict("terminal_window_px"=>32))]
    input=(a=zeros(256,256),truth=(x,y)->ST.midpoint_truth(x,y,64))
    before=copy(first_result.mask)
    common=ST.common_population(evals,input,64;amplitude=.5)
    @test common["candidate_nodes"]==8
    @test common["common_primary"]==6
    @test common["not_common_primary"]==2
    @test all(s->s["own_interior_primary"]==7 && s["primary_lost_to_common"]==1,common["schedules"])
    counts=common["schedules"][1]["full_error_metrics"]["components"]["u"]["counts"]
    @test counts["primary_valid"]==6
    @test counts["uq_available"]==5
    @test counts["uq_nonfinite"]==1
    @test counts["sigma_zero"]==1
    @test common["schedules"][1]["response"]["primary_nodes"]==6
    @test common["schedules"][1]["response"]["fits"][1]["u"]["sample_count"]==6
    @test first_result.mask==before
    first_result.outliers.=true
    empty=ST.common_population(evals,input,64;amplitude=.5)
    @test empty["common_primary"]==0
    @test empty["schedules"][1]["response"]["fits"][1]["u"]["reason"]=="too_few_samples"
    @test empty["schedules"][1]["full_error_metrics"]["components"]["u"]["coverage"]["2"]["available"]===false
    altered=Hammerhead.PIVResult(x,y,u,fill(-1.5,4,2),ones(4,2),ones(4,2),ones(4,2),ones(4,2),falses(4,2),falses(4,2),PIVParameters(;window_size=16,overlap=8,uncertainty=true))
    @test_throws ArgumentError ST.response(altered,trues(4,2),(64,);amplitude=.5,center=128.5)
    @test_throws ArgumentError ST.parameters(24)
end

@testset "Spatial-transfer bounded processing and protected reports" begin
    # Four reduced-size calls: two schedules on the control and one wavelength.
    report=ST.run_study(;seeds=(7321,),size=128,wavelengths=(64.,),interior_margin=32)
    @test report["processing_calls"]==4
    @test report["source_and_environment_stable"]===true
    @test length(report["groups"])==2
    @test report["renderer"]["integer_radius"]==3
    for g in report["groups"]
        @test g["common_interior"]["candidate_nodes"]==64
        @test length(unique(s["particle_positions_sha256"] for s in g["schedules"]))==1
        @test length(unique(s["inputs"]["image_b_sha256"] for s in g["schedules"]))==1
        for s in g["schedules"]
            @test s["recipe"]["passes"][end]["n_peaks"]==1
            @test s["recipe"]["passes"][end]["replace_outliers"]===false
            @test s["fixed_interior"]["response"]["candidate_nodes"]==64
            @test s["full_grid_response"]["primary_nodes"]==s["full_error_metrics"]["counts"]["primary_valid"]
            for k in ("u","v")
                c=s["full_error_metrics"]["components"][k]["counts"]
                @test c["uq_available"]+c["uq_nonfinite"]+c["uq_negative"]==c["primary_valid"]
            end
        end
    end
    @test report["groups"][1]["schedules"][1]["inputs"]["image_a_sha256"]==report["groups"][2]["schedules"][1]["inputs"]["image_a_sha256"]
    @test_throws ArgumentError ST.check_output(joinpath(ST.ROOT,"docs"))
    @test_throws ArgumentError ST.main(["--unknown"])
    @test_throws ArgumentError ST.run_study(;interior_margin=200)
    # Hardlinks require the source volume; use an admitted report directory.
    alias_parent=mkpath(joinpath(ST.ROOT,"bench","profile-output"))
    mktempdir(alias_parent) do dir
        paths=ST.write_report(dir,report)
        loaded=TOML.parsefile(paths[1]);@test isequal(loaded["groups"],report["groups"])
        @test occursin("Full-error 2sigma",read(paths[2],String))
        @test occursin("Signed / quadrature",read(paths[2],String))
        bytes=read.(paths)
        bad=deepcopy(report);bad["source_and_environment_stable"]=false
        @test_throws ArgumentError ST.write_report(dir,bad)
        @test read.(paths)==bytes
        bad=deepcopy(report);bad["schema_version"]="future"
        @test_throws ArgumentError ST.write_report(dir,bad)
        @test read.(paths)==bytes
        write(paths[2],"user file")
        @test_throws ArgumentError ST.write_report(dir,report)
        @test read(paths[2],String)=="user file"
        alias=joinpath(dir,"source-alias");mkdir(alias)
        source=joinpath(ST.ROOT,"bench","spatial_transfer.jl");before=read(source)
        @test ST.check_output(alias) isa Vector
        hardlink(source,joinpath(alias,"spatial_transfer.toml"))
        @test Base.samefile(source,joinpath(alias,"spatial_transfer.toml"))
        alias_error=try ST.write_report(alias,report); nothing catch error; error end
        @test alias_error isa ArgumentError && occursin("aliases source",sprint(showerror,alias_error))
        @test read(source)==before
        same=joinpath(dir,"same");mkdir(same)
        p=joinpath(same,"spatial_transfer.toml");write(p,"# $(ST.MARKER)\n")
        hardlink(p,joinpath(same,"spatial_transfer.md"))
        @test_throws ArgumentError ST.write_report(same,report)
        @test read(p,String)=="# $(ST.MARKER)\n"
    end
end
