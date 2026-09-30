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
    varying = mkfield(xr, yr, zeros(5,5), [xx^2 for yy in yr, xx in xr])
    @test circulation(varying; region=(1.25,2.75,0.5,3.5)) ≈ 18 atol=1e-12
    @test circulation(varying; region=[(1.,1.),(3.,1.),(1.,3.)]) ≈ 20/3 atol=1e-12
    flagged = mkfield(xr, yr, copy(rotation.u), copy(rotation.v))
    flagged.outliers[3,3] = true
    @test circulation(flagged; region=(1.,2.,1.,2.)) == 0
    @test circulation(flagged; region=(1.,2.,1.,2.),include_invalid=true) ≈ 2
    flagged.mask[3,3] = true
    @test circulation(flagged; region=(1.,2.,1.,2.),include_invalid=true) == 0

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
