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
