using Hammerhead
using Test

@testset "Derived flow and planar polish" begin
    x=collect(0.0:1:4); y=collect(0.0:1:3)
    u=[2xj+3yi for yi in y,xj in x]; v=[-xj+4yi for yi in y,xj in x]
    d=flow_derivatives(x,y,u,v)
    @test all(d.dudx .≈ 2) && all(d.dudy .≈ 3)
    @test all(d.dvdx .≈ -1) && all(d.dvdy .≈ 4)
    @test all(vorticity(d) .≈ -4) && all(divergence(d) .≈ 6)
    @test all(q_criterion(d) .≈ -7)
    @test all(swirling_strength(d) .≈ sqrt(2))

    params = PIVParameters()
    mkfield(xg, yg, uf, vf) = PIVResult(xg, yg, uf, vf, ones(size(uf)),
        zeros(size(uf)), fill(NaN,size(uf)), fill(NaN,size(uf)),
        falses(size(uf)), falses(size(uf)), params)
    xr = collect(0.0:4.0); yr = collect(0.0:4.0)
    rotation = mkfield(xr, yr, [-yy for yy in yr, xx in xr],
                               [xx for yy in yr, xx in xr])
    @test circulation(rotation; region=(0.,4.,0.,4.)) ≈ 32 atol=1e-12
    complete = circulation(rotation; region=(0.,4.,0.,4.), coverage=:report)
    @test complete.value ≈ 32 atol=1e-12
    @test complete.valid_area ≈ 16 atol=1e-12
    @test complete.requested_area ≈ 16 atol=1e-12
    @test complete.coverage_fraction == 1
    @test complete.complete
    @test circulation(rotation; region=(0.25,3.75,0.5,3.5)) ≈ 21 atol=1e-12
    triangle = [(0.,0.),(4.,0.),(0.,4.)]
    @test circulation(rotation; region=triangle) ≈ 16 atol=1e-12
    @test circulation(rotation; region=reverse(triangle)) ≈ 16 atol=1e-12
    concave = [(0.,0.),(4.,0.),(4.,1.),(1.,1.),(1.,4.),(0.,4.)]
    @test circulation(rotation; region=concave) ≈ 14 atol=1e-12
    descending = mkfield(reverse(xr), reverse(yr),
        [-yy for yy in reverse(yr), xx in reverse(xr)],
        [xx for yy in reverse(yr), xx in reverse(xr)])
    profile = extract_profile(descending, [(0.5,0.5),(3.5,3.5)]; n=3)
    @test profile.u ≈ [-0.5,-2.,-3.5]
    @test profile.v ≈ [0.5,2.,3.5]
    @test circulation(descending; region=triangle) ≈ 16 atol=1e-12
    @test circulation(descending; region=reverse(triangle), coverage=:report).coverage_fraction == 1
    shifted_x = 1.0e6 .+ collect(0.0:0.2:4.0)
    shifted_y = -1.0e6 .+ collect(0.0:0.2:4.0)
    shifted = mkfield(shifted_x, shifted_y,
        [-yy for yy in shifted_y, xx in shifted_x],
        [xx for yy in shifted_y, xx in shifted_x])
    shifted_region = (first(shifted_x), last(shifted_x), first(shifted_y), last(shifted_y))
    @test circulation(shifted; region=shifted_region) ≈ 32 atol=1e-8
    @test circulation(shifted; region=shifted_region, coverage=:report).coverage_fraction == 1
    varying = mkfield(xr, yr, zeros(5,5), [xx^2 for yy in yr, xx in xr])
    @test circulation(varying; region=(1.25,2.75,0.5,3.5)) ≈ 18 atol=1e-12
    @test circulation(varying; region=[(1.,1.),(3.,1.),(1.,3.)]) ≈ 20/3 atol=1e-12
    flagged = mkfield(xr, yr, copy(rotation.u), copy(rotation.v))
    flagged.outliers[3,3] = true
    @test_throws ArgumentError circulation(flagged; region=(1.,2.,1.,2.))
    missing_cell = circulation(flagged; region=(1.,2.,1.,2.), coverage=:report)
    @test isnan(missing_cell.value)
    @test missing_cell.valid_area == 0
    @test missing_cell.requested_area ≈ 1
    @test missing_cell.coverage_fraction == 0
    @test !missing_cell.complete
    @test_throws ArgumentError circulation(flagged; region=(0.,4.,0.,4.))
    partial = circulation(flagged; region=(0.,4.,0.,4.), coverage=:report)
    @test partial.value ≈ 24 atol=1e-12
    @test partial.valid_area ≈ 12 atol=1e-12
    @test partial.requested_area ≈ 16 atol=1e-12
    @test partial.coverage_fraction ≈ 0.75 atol=1e-12
    @test !partial.complete
    @test circulation(flagged; region=reverse([(0.,0.),(4.,0.),(4.,4.),(0.,4.)]), coverage=:report).value ≈ 24 atol=1e-12
    @test circulation(flagged; region=(1.,2.,1.,2.),include_invalid=true) ≈ 2
    flagged.mask[3,3] = true
    @test isnan(circulation(flagged; region=(1.,2.,1.,2.),include_invalid=true,coverage=:report).value)
    @test_throws ArgumentError circulation(flagged; region=(1.,2.,1.,2.),include_invalid=true)
    @test_throws ArgumentError circulation(rotation; region=(-1.,1.,0.,1.))
    outside_part = circulation(rotation; region=(-1.,1.,0.,1.), coverage=:report)
    @test outside_part.value ≈ 2 atol=1e-12
    @test outside_part.valid_area ≈ 1 atol=1e-12
    @test outside_part.requested_area ≈ 2 atol=1e-12
    @test outside_part.coverage_fraction ≈ 0.5 atol=1e-12
    @test_throws ArgumentError circulation(rotation; region=(-1e-15,4.,0.,4.))
    tiny_sliver = circulation(rotation; region=(-1e-15,4.,0.,4.), coverage=:report)
    @test !tiny_sliver.complete
    @test tiny_sliver.coverage_fraction ≈ 1 atol=1e-14
    @test_throws ArgumentError circulation(rotation; region=(5.,6.,5.,6.))
    outside_all = circulation(rotation; region=(5.,6.,5.,6.), coverage=:report)
    @test isnan(outside_all.value)
    @test outside_all.valid_area == 0
    @test outside_all.requested_area ≈ 1
    @test outside_all.coverage_fraction == 0
    @test !outside_all.complete
    @test_throws ArgumentError circulation(rotation; region=(0.,4.,0.,4.), coverage=:skip)

    # physical() resets the conversion scale's dt to 1, but not the cadence.
    sequence = [mkfield([0.,1.],[0.,1.],fill(sin(2π*k/10),2,2),zeros(2,2))
                for k in 0:99]
    scaled = [with_scale(r, PhysicalScale(pixel_size=0.02,dt=0.01)) for r in sequence]
    converted = physical.(scaled)
    @test all(r -> r.scale.dt == 1, converted)
    @test_throws ArgumentError result_spectrum(scaled,1,1)
    @test_throws ArgumentError result_spectrum(converted,1,1)
    for samples in (scaled, converted)
        frequency, psd = result_spectrum(samples,1,1; dt=0.01,window=:none)
        @test frequency[argmax(psd)] ≈ 10.0
    end
    @test_throws ArgumentError result_spectrum([sequence[1],mkfield([0.,2.],[0.,1.],ones(2,2),zeros(2,2))],1,1;dt=0.01)

    valid=trues(size(u)); valid[2,3]=false
    dm=flow_derivatives(x,y,u,v; valid)
    @test isnan(dm.dudx[2,3])
    @test dm.dudx[2,2] ≈ 2 # one-sided, does not cross invalid point

    t=planar_calibration((1.0,2.0),(1.0,12.0),10.0)
    @test collect(transform_point(t,(1.0,2.0))) ≈ [0.0,0.0]
    @test collect(transform_point(t,(1.0,12.0))) ≈ [10.0,0.0]
    @test collect(transform_point(inv(t),transform_point(t,(7.0,4.0)))) ≈ [7.0,4.0]
    tr=planar_calibration((0.,0.),(2.,0.),10.; perpendicular_scale=2, reflection=true)
    @test collect(transform_vector(tr,(2.,1.))) ≈ [10.,-2.]

    a=reshape(collect(1.0:100),10,10)
    s=percentile_stretch(a; low=0,high=100)
    @test extrema(s) == (0.0,1.0) && a[1] == 1
    @test invert_image([1 2;3 4]) == [4.0 3.0;2.0 1.0]
    @test all(isfinite,local_variance_normalize(fill(2.0,8,8)))

    # One missing detection can be reacquired, and its frame jump is explicit.
    frames=[zeros(64,64) for _ in 1:4]
    for (k,img) in enumerate(frames), (yy,xx) in ((20.,20.),(44.,44.))
        k == 3 && continue
        for j in 1:64, i in 1:64
            img[i,j] += exp(-((i-(yy+k-1))^2+(j-(xx+k-1))^2)/2)
        end
    end
    tr=track_particles(frames,PTVParameters(search_radius=2,uod_enable=false);
                       predictor=nothing,max_gap=1,min_track_length=3,progress=false)
    @test length(tr.trajectories) == 2
    @test all(t -> t.frames == [1,2,4],tr.trajectories)
    @test all(t -> all(trajectory_velocities(t)[1] .≈ 1),tr.trajectories)
end
