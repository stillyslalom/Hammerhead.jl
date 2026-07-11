# Stereo 3C reconstruction (Phase 5, slice 3). Reuses make_test_camera /
# calib_world_points from test_calibration.jl and generate_gaussian_particle!
# imported in test_dewarp.jl.

# Render one stereo frame pair: world-plane particles at z = 0 displaced by
# (dx, dy, dz) world units, projected through each camera.
function stereo_frames(cams, pts, (dx, dy, dz); image_size = (512, 512), diameter = 6.0)
    frames = map(cams) do cam
        A = zeros(image_size)
        B = zeros(image_size)
        for (X, Y) in pts
            pa = world_to_pixel(cam, (X, Y, 0.0))
            pb = world_to_pixel(cam, (X + dx, Y + dy, dz))
            generate_gaussian_particle!(A, (pa[1], pa[2]), diameter)
            generate_gaussian_particle!(B, (pb[1], pb[2]), diameter)
        end
        (A, B)
    end
    return frames[1][1], frames[1][2], frames[2][1], frames[2][2]
end

@testset "Stereo: viewing-ray slopes" begin
    cam = make_test_camera(yaw_deg = 20.0)
    # At the world origin the in-plane ray drift per unit Z equals the yaw.
    t0 = Hammerhead.ray_slopes(cam, 0.0, 0.0, 0.0, 0.25)
    @test t0[1] ≈ tan(deg2rad(20.0)) atol = 1e-6
    @test t0[2] ≈ 0.0 atol = 1e-6
    # Off-axis it matches the exact pinhole ray from pixel_to_world.
    for (X, Y) in ((12.0, -8.0), (-20.0, 15.0))
        p = world_to_pixel(cam, (X, Y, 0.0))
        exact = (pixel_to_world(cam, p, 1.0) - pixel_to_world(cam, p, -1.0)) / 2
        t = Hammerhead.ray_slopes(cam, X, Y, 0.0, 0.25)
        @test t[1] ≈ exact[1] atol = 1e-6
        @test t[2] ≈ exact[2] atol = 1e-6
    end
end

@testset "Stereo: reconstruction algebra on exact measurements" begin
    cams = (make_test_camera(yaw_deg = -18.0), make_test_camera(yaw_deg = 27.0))
    grid = DewarpGrid(x = -10.0:0.5:10.0, y = -10.0:0.5:10.0)
    sx, sy = step(grid.x), step(grid.y)
    δ = max(abs(sx), abs(sy))
    Xw, Yw = 3.0, -4.0
    dx, dy, dz = 0.31, -0.22, 0.17
    σpx = 0.05

    # Synthesize the exact per-camera 2C measurements at one grid point.
    results = map(cams) do cam
        t = Hammerhead.ray_slopes(cam, Xw, Yw, grid.z, δ)
        u = (dx - dz * t[1]) / sx
        v = (dy - dz * t[2]) / sy
        PIVResult([(Xw - first(grid.x)) / sx + 1], [(Yw - first(grid.y)) / sy + 1],
                  fill(u, 1, 1), fill(v, 1, 1), ones(1, 1), ones(1, 1),
                  fill(σpx, 1, 1), fill(σpx, 1, 1), falses(1, 1), falses(1, 1),
                  PIVParameters())
    end

    # The 4-equation LSQ is consistent, so it must return the truth exactly.
    s = Hammerhead.reconstruct_stereo(results[1], results[2], cams[1], cams[2], grid)
    @test s.x[1] ≈ Xw atol = 1e-9
    @test s.y[1] ≈ Yw atol = 1e-9
    @test s.u[1, 1] ≈ dx atol = 1e-9
    @test s.v[1, 1] ≈ dy atol = 1e-9
    @test s.w[1, 1] ≈ dz atol = 1e-9

    # σ propagation on a symmetric ±20° rig at the on-axis point, where the
    # analytic answer is σ_u = σ/√2, σ_v = σ/√2, σ_w = σ/(√2 tan20°).
    sym = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    results0 = map(sym) do cam
        PIVResult([(0.0 - first(grid.x)) / sx + 1], [(0.0 - first(grid.y)) / sy + 1],
                  zeros(1, 1), zeros(1, 1), ones(1, 1), ones(1, 1),
                  fill(σpx, 1, 1), fill(σpx, 1, 1), falses(1, 1), falses(1, 1),
                  PIVParameters())
    end
    s0 = Hammerhead.reconstruct_stereo(results0[1], results0[2], sym[1], sym[2], grid)
    σmm = σpx * sx
    @test s0.uncertainty_u[1, 1] ≈ σmm / sqrt(2) atol = 1e-5
    @test s0.uncertainty_v[1, 1] ≈ σmm / sqrt(2) atol = 1e-5
    @test s0.uncertainty_w[1, 1] ≈ σmm / (sqrt(2) * tan(deg2rad(20.0))) atol = 1e-5
end

@testset "run_piv_stereo: 3C reconstruction" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(17)
    pts = [(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:350]
    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss)
    truth = (0.5, -0.3, 0.4)   # mm
    A1, B1, A2, B2 = stereo_frames(cams, pts, truth)

    @testset "known 3D displacement (pinhole)" begin
        stereo = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params)
        @test stereo isa StereoPIVResult{Float64}
        @test size(stereo.u) == (length(stereo.y), length(stereo.x))
        @test stereo.z == 0.0
        # The vector grid's world coordinates derive from the window centers.
        @test stereo.x ≈ first(grid.x) .+ (stereo.cam1.x .- 1) .* step(grid.x)
        @test stereo.y ≈ first(grid.y) .+ (stereo.cam1.y .- 1) .* step(grid.y)
        # A symmetric rig reconstructs all three components without bias.
        @test median(stereo.u) ≈ truth[1] atol = 0.02
        @test median(stereo.v) ≈ truth[2] atol = 0.02
        @test median(stereo.w) ≈ truth[3] atol = 0.03
        # Flags: union of the cameras', masked ≠ outlier, no mask here.
        @test stereo.outliers == (stereo.cam1.outliers .| stereo.cam2.outliers)
        @test !any(stereo.mask)
        @test sum(stereo.outliers) <= 0.1 * length(stereo.u)
        # Uncertainty fields stay NaN when the parameter is off.
        @test all(isnan, stereo.uncertainty_u)
        @test all(isnan, stereo.uncertainty_w)
        # Per-camera 2C results are retained (dewarped-pixel units).
        @test stereo.cam1 isa PIVResult{Float64}
        @test stereo.parameters === stereo.cam1.parameters
    end

    @testset "fitted Soloff cameras" begin
        world = calib_world_points()
        fits = map(cams) do cam
            calibrate_camera([world_to_pixel(cam, w) for w in world], world)
        end
        dws_fit = map(fit -> ImageDewarper(fit, grid, (512, 512)), fits)
        stereo = run_piv_stereo(A1, B1, A2, B2, dws_fit[1], dws_fit[2], params)
        @test median(stereo.u) ≈ truth[1] atol = 0.03
        @test median(stereo.v) ≈ truth[2] atol = 0.03
        @test median(stereo.w) ≈ truth[3] atol = 0.05
    end

    @testset "pure out-of-plane displacement" begin
        dz = 0.5
        C1, D1, C2, D2 = stereo_frames(cams, pts, (0.0, 0.0, dz))
        stereo = run_piv_stereo(C1, D1, C2, D2, dws[1], dws[2], params)
        # The cameras see equal and opposite in-plane displacement
        # (±dz·tan20° = ±0.91 dewarped px) ...
        m1, m2 = median(stereo.cam1.u), median(stereo.cam2.u)
        @test m1 * m2 < 0
        @test abs(m1) > 0.5 && abs(m2) > 0.5
        # ... which reconstructs to pure w.
        @test median(stereo.u) ≈ 0.0 atol = 0.02
        @test median(stereo.v) ≈ 0.0 atol = 0.02
        @test median(stereo.w) ≈ dz atol = 0.03
    end
end

@testset "run_piv_stereo: uncertainty propagation" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(23)
    # Dense seeding: the 16-px final windows need several particles each for
    # the outlier rate to stay low at this noise level.
    pts = [(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:1200]
    truth = (0.3, -0.2, 0.25)
    A1, B1, A2, B2 = stereo_frames(cams, pts, truth)
    for img in (A1, B1, A2, B2)
        img .+= 0.05 .* randn(rng, size(img))
    end
    passes = multipass_parameters([32, 16, 16]; padding = true, apodization = :gauss,
                                  uncertainty = true)
    stereo = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], passes)

    sel = .!stereo.outliers .& isfinite.(stereo.uncertainty_w)
    @test count(sel) > 0.7 * length(stereo.u)
    @test median(stereo.w[.!stereo.outliers]) ≈ truth[3] atol = 0.05
    # w is the noise-amplified component: for a ±20° rig, σw/σu = 1/tan20°
    # ≈ 2.75 — the w-sensitivity scales with the camera separation angle.
    ratio = median(stereo.uncertainty_w[sel]) / median(stereo.uncertainty_u[sel])
    @test 2.0 < ratio < 3.6
    # In-plane σ combines the two cameras: ≈ per-camera σ (in mm) / √2.
    ratio_u = median(stereo.uncertainty_u[sel]) /
              (median(stereo.cam1.uncertainty_u[sel]) * step(grid.x))
    @test 0.5 < ratio_u < 0.9
end

@testset "run_piv_stereo: mask union and out-of-view regions" begin
    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    # Grid wider than either camera's field of view (±36.6 mm).
    grid = DewarpGrid(x = -60.0:0.4:60.0, y = -60.0:0.4:60.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(41)
    pts = [(110 * rand(rng) - 55, 110 * rand(rng) - 55) for _ in 1:800]
    truth = (0.4, 0.2, 0.0)
    A1, B1, A2, B2 = stereo_frames(cams, pts, truth)
    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss)

    stereo = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params)
    @test any(stereo.mask) && !all(stereo.mask)
    @test all(isnan, stereo.u[stereo.mask])
    @test all(isnan, stereo.w[stereo.mask])
    @test !any(stereo.outliers .& stereo.mask)   # masked cells are never outliers
    valid = .!stereo.mask .& .!stereo.outliers
    @test median(stereo.u[valid]) ≈ truth[1] atol = 0.03
    @test median(stereo.v[valid]) ≈ truth[2] atol = 0.03
    @test median(stereo.w[valid]) ≈ truth[3] atol = 0.04

    # A user mask joins the dewarpers' out-of-view union.
    um = falses(size(grid))
    um[:, 1:(size(grid)[2] ÷ 2)] .= true
    stereo_m = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params; mask = um)
    @test sum(stereo_m.mask) > sum(stereo.mask)
end

@testset "run_piv_stereo: degenerate rig, errors, save/load, Float32" begin
    cams = (make_test_camera(yaw_deg = -15.0), make_test_camera(yaw_deg = 15.0))
    grid = DewarpGrid(x = -8.0:0.4:8.0, y = -8.0:0.4:8.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    rng = MersenneTwister(5)
    pts = [(20 * rand(rng) - 10, 20 * rand(rng) - 10) for _ in 1:60]
    truth = (0.2, -0.1, 0.15)
    A1, B1, A2, B2 = stereo_frames(cams, pts, truth)
    params = PIVParameters(window_size = 16, overlap = 8,
                           padding = true, apodization = :gauss)
    stereo = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params)
    @test median(stereo.w) ≈ truth[3] atol = 0.05

    # Identical cameras: parallel viewing rays carry no depth information.
    degen = run_piv_stereo(A1, B1, A1, B1, dws[1], dws[1], params)
    @test all(isnan, degen.w)
    @test all(isnan, degen.u)

    # Precision follows the images.
    s32 = run_piv_stereo(Float32.(A1), Float32.(B1), Float32.(A2), Float32.(B2),
                         dws[1], dws[2], params)
    @test s32 isa StereoPIVResult{Float32}
    @test median(s32.w) ≈ truth[3] atol = 0.05

    # Effort schedules work on the stereo driver too, using the dewarped grid
    # size to clamp the pyramid when needed.
    slow = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2]; effort = :low)
    @test slow isa StereoPIVResult{Float64}
    @test slow.parameters.window_size == (32, 32)
    @test_throws ArgumentError run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2],
                                              params; effort = :low)

    # Argument validation.
    grid2 = DewarpGrid(x = -8.0:0.4:8.0, y = -8.0:0.4:7.6)
    dw_other = ImageDewarper(cams[2], grid2, (512, 512))
    @test_throws ArgumentError run_piv_stereo(A1, B1, A2, B2, dws[1], dw_other, params)
    @test_throws DimensionMismatch run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2],
                                                  params; mask = falses(3, 3))
    @test_throws DimensionMismatch run_piv_stereo(A1[1:256, :], B1[1:256, :], A2, B2,
                                                  dws[1], dws[2], params)

    # JLD2 roundtrip, including a mixed PIVResult/StereoPIVResult file.
    mktempdir() do dir
        path = joinpath(dir, "stereo.jld2")
        save_results(path, stereo)
        loaded = load_results(path)
        @test length(loaded) == 1
        s = loaded[1]
        @test s isa StereoPIVResult{Float64}
        @test s.x == stereo.x && s.y == stereo.y && s.z == stereo.z
        @test isequal(s.u, stereo.u) && isequal(s.v, stereo.v) && isequal(s.w, stereo.w)
        @test s.outliers == stereo.outliers && s.mask == stereo.mask
        @test isequal(s.cam1.u, stereo.cam1.u)
        @test s.parameters.window_size == stereo.parameters.window_size

        mixed = Union{PIVResult{Float64},StereoPIVResult{Float64}}[stereo.cam1, stereo]
        save_results(path, mixed)
        back = load_results(path)
        @test back[1] isa PIVResult{Float64}
        @test back[2] isa StereoPIVResult{Float64}
    end
end
