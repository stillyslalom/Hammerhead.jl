# Wieneke 2005 disparity self-calibration (Phase 5, slice 4). Reuses
# make_test_camera / calib_world_points from test_calibration.jl and
# generate_gaussian_particle! imported in test_dewarp.jl.

# Render one same-instant image per camera of particles lying on the sheet
# z = a + b·X + c·Y (cameras are calibrated to z = 0, so a nonzero sheet
# produces a disparity between the dewarped views).
function sheet_instant(cams, pts, (a, b, c); image_size = (512, 512), diameter = 6.0)
    map(cams) do cam
        img = zeros(image_size)
        for (X, Y) in pts
            p = world_to_pixel(cam, (X, Y, a + b * X + c * Y))
            generate_gaussian_particle!(img, (p[1], p[2]), diameter)
        end
        img
    end
end

# Frame pair per camera: particles on the sheet displaced by (dx, dy, dz)
# world units between exposures.
function sheet_pair_frames(cams, pts, (a, b, c), (dx, dy, dz);
                           image_size = (512, 512), diameter = 6.0)
    frames = map(cams) do cam
        A = zeros(image_size)
        B = zeros(image_size)
        for (X, Y) in pts
            Z = a + b * X + c * Y
            pa = world_to_pixel(cam, (X, Y, Z))
            pb = world_to_pixel(cam, (X + dx, Y + dy, Z + dz))
            generate_gaussian_particle!(A, (pa[1], pa[2]), diameter)
            generate_gaussian_particle!(B, (pb[1], pb[2]), diameter)
        end
        (A, B)
    end
    return frames[1][1], frames[1][2], frames[2][1], frames[2][2]
end

@testset "TransformedCamera and rigid world transforms" begin
    cam = make_test_camera(yaw_deg = 15.0)
    θ = deg2rad(1.2)
    R = [cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)]
    t = [0.4, -0.3, 0.7]
    wpts = ((3.0, -4.0, 0.5), (-10.0, 8.0, -1.0), (0.0, 0.0, 0.0))

    # A pinhole absorbs the transform exactly into its projection matrix.
    pc = Hammerhead.apply_world_transform(cam, R, t)
    @test pc isa PinholeCamera
    for w in wpts
        @test world_to_pixel(pc, w) ≈ world_to_pixel(cam, R * collect(w) + t) atol = 1e-9
    end

    # Soloff has no exact closure under rotation and gets the wrapper.
    world = calib_world_points()
    sol = calibrate_camera([world_to_pixel(cam, w) for w in world], world)
    tc = Hammerhead.apply_world_transform(sol, R, t)
    @test tc isa TransformedCamera{SoloffCamera}
    for w in wpts
        @test world_to_pixel(tc, w) ≈ world_to_pixel(sol, R * collect(w) + t) atol = 1e-9
        # Newton inversion of the composed map round-trips.
        @test pixel_to_world(tc, world_to_pixel(tc, w), w[3]) ≈ collect(w) atol = 1e-5
    end

    # Wrapping a wrapper collapses into a single composed transform.
    tc2 = Hammerhead.apply_world_transform(tc, R, t)
    @test tc2 isa TransformedCamera{SoloffCamera}
    @test tc2.cam === sol
    w = (2.0, 3.0, -0.5)
    @test world_to_pixel(tc2, w) ≈
          world_to_pixel(sol, R * (R * collect(w) + t) + t) atol = 1e-9

    @test_throws ArgumentError TransformedCamera(cam, zeros(2, 2), t)
    @test_throws ArgumentError TransformedCamera(cam, 2 .* R, t)     # not a rotation
    @test_throws ArgumentError TransformedCamera(cam, R, [1.0, 2.0])
end

@testset "self_calibrate: translated sheet (pinhole)" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(11)
    plane = (0.8, 0.0, 0.0)
    frames = [sheet_instant(cams, [(56 * rand(rng) - 28, 56 * rand(rng) - 28)
                                   for _ in 1:400], plane) for _ in 1:3]
    frames1 = [f[1] for f in frames]
    frames2 = [f[2] for f in frames]

    dw1c, dw2c, report = self_calibrate(frames1, frames2, dws[1], dws[2])

    # Initial disparity: the sheet offset appears as ∓a·tan20° per camera,
    # so |d| = 2·a·tan20° / step ≈ 2.91 dewarped px.
    d0 = 2 * plane[1] * tan(deg2rad(20.0)) / step(grid.x)
    @test report.passes[1].disparity_rms ≈ d0 rtol = 0.05
    # The first plane fit recovers the sheet.
    pl = report.passes[1].plane
    @test pl !== nothing
    @test pl.a ≈ plane[1] atol = 0.02
    @test abs(pl.b) < 2e-3 && abs(pl.c) < 2e-3
    @test report.passes[1].triangulation_rms < 0.25
    @test report.converged
    @test report.passes[end].disparity_rms < 0.05
    @test report.passes[end].plane === nothing
    # Corrected pinholes stay pinholes.
    @test dw1c.cam isa PinholeCamera && dw2c.cam isa PinholeCamera

    # The corrected frame's z = 0 plane coincides with the physical sheet.
    for (X, Y) in ((0.0, 0.0), (15.0, -10.0), (-20.0, 18.0))
        w = report.R * [X, Y, 0.0] + report.t
        @test w[3] ≈ plane[1] atol = 0.02
    end

    # Independent verification: the corrected dewarpers align the two views
    # to a small fraction of a pixel.
    p64 = Hammerhead.default_selfcal_parameters(size(grid))
    r = run_piv(dewarp(dw1c, frames1[1]), dewarp(dw2c, frames2[1]), p64;
                mask = dw1c.mask .| dw2c.mask)
    sel = .!r.mask .& .!r.outliers
    @test abs(median(r.u[sel])) < 0.05
    @test abs(median(r.v[sel])) < 0.05
end

@testset "self_calibrate: tilted sheet and stereo reconstruction" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(13)
    plane = (0.5, 0.012, -0.008)
    frames = [sheet_instant(cams, [(56 * rand(rng) - 28, 56 * rand(rng) - 28)
                                   for _ in 1:400], plane) for _ in 1:3]
    frames1 = [f[1] for f in frames]
    frames2 = [f[2] for f in frames]

    dw1c, dw2c, report = self_calibrate(frames1, frames2, dws[1], dws[2])
    @test report.converged
    @test report.passes[end].disparity_rms < 0.05
    @test 1 <= count(p -> p.plane !== nothing, report.passes) <= 3

    # The corrected frame's z = 0 plane lies on the tilted sheet everywhere.
    for (X, Y) in ((0.0, 0.0), (18.0, 12.0), (-15.0, 20.0), (10.0, -22.0))
        w = report.R * [X, Y, 0.0] + report.t
        @test w[3] ≈ plane[1] + plane[2] * w[1] + plane[3] * w[2] atol = 0.02
    end

    # A known world displacement of particles on the sheet reconstructs
    # without bias through the corrected dewarpers.
    truth = (0.3, -0.2, 0.25)
    pts = [(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:350]
    A1, B1, A2, B2 = sheet_pair_frames(cams, pts, plane, truth)
    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss)
    stereo = run_piv_stereo(A1, B1, A2, B2, dw1c, dw2c, params)
    sel = .!stereo.mask .& .!stereo.outliers
    @test median(stereo.u[sel]) ≈ truth[1] atol = 0.03
    @test median(stereo.v[sel]) ≈ truth[2] atol = 0.03
    @test median(stereo.w[sel]) ≈ truth[3] atol = 0.05
end

@testset "self_calibrate: fitted Soloff cameras" begin
    pin = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    world = calib_world_points()
    cams = map(c -> calibrate_camera([world_to_pixel(c, w) for w in world], world), pin)
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(29)
    plane = (0.6, 0.01, 0.0)
    # Particles are rendered through the true (pinhole) optics; the fitted
    # Soloff models are what gets corrected.
    frames = [sheet_instant(pin, [(56 * rand(rng) - 28, 56 * rand(rng) - 28)
                                  for _ in 1:400], plane) for _ in 1:3]
    frames1 = [f[1] for f in frames]
    frames2 = [f[2] for f in frames]

    dw1c, dw2c, report = self_calibrate(frames1, frames2, dws[1], dws[2])
    @test report.converged
    @test dw1c.cam isa TransformedCamera{SoloffCamera}
    @test dw2c.cam isa TransformedCamera{SoloffCamera}
    for (X, Y) in ((0.0, 0.0), (18.0, 12.0), (-15.0, 20.0))
        w = report.R * [X, Y, 0.0] + report.t
        @test w[3] ≈ plane[1] + plane[2] * w[1] + plane[3] * w[2] atol = 0.03
    end
end

@testset "self_calibrate: report options, short-circuit, errors" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -20.0:0.25:20.0, y = -20.0:0.25:20.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(41)
    pts = [(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:600]
    I1, I2 = sheet_instant(cams, pts, (0.4, 0.0, 0.0))

    # Single-matrix convenience method; disparity maps kept on request.
    dw1c, dw2c, rep = self_calibrate(I1, I2, dws[1], dws[2]; keep_disparity_maps = true)
    @test rep.converged
    @test length(rep.disparity_maps) == length(rep.passes)
    @test rep.disparity_maps[1] isa PIVResult
    @test rep.disparity_maps[1].parameters.window_size ==
          Hammerhead.default_selfcal_parameters(size(grid)).window_size

    # Perfect alignment: converges on the first measurement, applies no
    # correction, and hands back the input dewarpers untouched.
    J1, J2 = sheet_instant(cams, pts, (0.0, 0.0, 0.0))
    a1, a2, rep0 = self_calibrate(J1, J2, dws[1], dws[2])
    @test rep0.converged
    @test length(rep0.passes) == 1
    @test rep0.passes[1].plane === nothing
    @test isnan(rep0.passes[1].triangulation_rms)
    @test a1 === dws[1] && a2 === dws[2]
    @test rep0.R == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test rep0.t == zeros(3)
    @test isempty(rep0.disparity_maps)

    # Argument validation.
    grid2 = DewarpGrid(x = -20.0:0.25:20.0, y = -20.0:0.25:19.75)
    dw_other = ImageDewarper(cams[2], grid2, (512, 512))
    @test_throws ArgumentError self_calibrate(I1, I2, dws[1], dw_other)
    @test_throws ArgumentError self_calibrate([I1], [I2, I2], dws[1], dws[2])
    @test_throws ArgumentError self_calibrate(Matrix{Float64}[], Matrix{Float64}[],
                                              dws[1], dws[2])
    @test_throws ArgumentError self_calibrate(I1, I2, dws[1], dws[2]; iterations = 0)
    @test_throws ArgumentError self_calibrate(I1, I2, dws[1], dws[2]; tol = -1.0)
    @test_throws ArgumentError self_calibrate(I1, I2, dws[1], dws[2];
                                              max_triangulation_error = 0.0)
    @test_throws DimensionMismatch self_calibrate(I1, I2, dws[1], dws[2];
                                                  mask = falses(3, 3))
    @test_throws DimensionMismatch self_calibrate(I1[1:100, :], I2, dws[1], dws[2])
end
