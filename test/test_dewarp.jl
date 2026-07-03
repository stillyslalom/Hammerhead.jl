# Image dewarping onto a common world-plane grid (Phase 5, slice 2).
# Reuses make_test_camera / calib_world_points from test_calibration.jl.

using Hammerhead.SyntheticData: generate_gaussian_particle!

# Intensity-weighted centroid of `img` in a box of half-width `radius` around
# `(r0, c0)`, returned as (row, col).
function blob_centroid(img, r0, c0; radius = 8)
    rs = max(1, round(Int, r0) - radius):min(size(img, 1), round(Int, r0) + radius)
    cs = max(1, round(Int, c0) - radius):min(size(img, 2), round(Int, c0) + radius)
    m = sr = sc = 0.0
    for c in cs, r in rs
        w = img[r, c]
        m += w
        sr += w * r
        sc += w * c
    end
    return (sr / m, sc / m)
end

# Dewarped pixel position of a world point on `grid`, as (row, col).
grid_position(grid, X, Y) = ((Y - first(grid.y)) / step(grid.y) + 1,
                             (X - first(grid.x)) / step(grid.x) + 1)

@testset "DewarpGrid" begin
    g = DewarpGrid(x = -30.0:0.5:30.0, y = -20.0:0.5:20.0, z = 1.5)
    @test size(g) == (81, 121)
    @test step(g.x) ≈ 0.5 && step(g.y) ≈ 0.5
    @test first(g.x) == -30.0 && last(g.x) == 30.0
    @test g.z == 1.5
    @test DewarpGrid(x = 0.0:1.0:10.0, y = 0.0:1.0:10.0).z == 0.0

    # Descending ranges are allowed (display-oriented +Y up).
    gd = DewarpGrid(x = -5.0:0.5:5.0, y = 5.0:-0.5:-5.0)
    @test step(gd.y) ≈ -0.5

    @test_throws ArgumentError DewarpGrid(x = 0.0:1.0:0.0, y = 0.0:1.0:10.0)
    @test_throws ArgumentError DewarpGrid(x = 0.0:1.0:10.0, y = 0.0:1.0:0.0)
    @test_throws ArgumentError DewarpGrid(x = LinRange(0.0, Inf, 5), y = 0.0:1.0:10.0)
    @test_throws ArgumentError DewarpGrid(x = 0.0:1.0:10.0, y = 0.0:1.0:10.0, z = NaN)
end

@testset "ImageDewarper: coordinate map" begin
    cam = make_test_camera(yaw_deg = 20.0)
    grid = DewarpGrid(x = -30.0:0.25:30.0, y = -30.0:0.25:30.0)
    dw = ImageDewarper(cam, grid, (512, 512))
    @test size(dw.rows) == size(dw.cols) == size(dw.mask) == size(grid)
    @test !any(dw.mask)  # whole grid inside the camera's view

    # Every node's stored source coordinate is the exact forward projection,
    # in the right orientation (rows ↔ y, cols ↔ x).
    for (i, j) in ((1, 1), (121, 121), (241, 1), (60, 200))
        p = world_to_pixel(cam, (grid.x[j], grid.y[i], grid.z))
        @test dw.cols[i, j] == p[1] && dw.rows[i, j] == p[2]
        # ... and back-projects onto the grid node.
        w = pixel_to_world(cam, (dw.cols[i, j], dw.rows[i, j]), grid.z)
        @test w[1] ≈ grid.x[j] atol = 1e-9
        @test w[2] ≈ grid.y[i] atol = 1e-9
    end

    @test_throws ArgumentError ImageDewarper(cam, grid, (0, 512))
end

@testset "Dewarp roundtrip: projected particles land on their world positions" begin
    cam = make_test_camera(yaw_deg = 20.0)
    grid = DewarpGrid(x = -30.0:0.25:30.0, y = -30.0:0.25:30.0)
    dw = ImageDewarper(cam, grid, (512, 512))

    world_pts = [(0.0, 0.0), (12.3, -7.7), (-20.6, 15.1), (24.9, 24.2)]
    img = zeros(512, 512)
    for (X, Y) in world_pts
        p = world_to_pixel(cam, (X, Y, 0.0))
        generate_gaussian_particle!(img, (p[1], p[2]), 8.0)
    end
    out = dewarp(dw, img)

    # Each particle's centroid in the dewarped image sits on its world
    # position to well under 0.1 dewarped px (= 0.025 mm here).
    for (X, Y) in world_pts
        r0, c0 = grid_position(grid, X, Y)
        r, c = blob_centroid(out, r0, c0)
        @test r ≈ r0 atol = 0.05
        @test c ≈ c0 atol = 0.05
    end
end

@testset "Dewarped calibration target: dots on a regular mm grid" begin
    cam = make_test_camera(yaw_deg = 25.0)
    img = render_calibration_target(cam, (512, 512); spacing = 15.0)
    grid = DewarpGrid(x = -30.0:0.25:30.0, y = -30.0:0.25:30.0)
    dw = ImageDewarper(cam, grid, (512, 512))
    out = dewarp(dw, img)

    # In the dewarped image the (perspective-distorted) dots become circles
    # again, centered at exact multiples of the 15 mm spacing.
    for (X, Y) in ((0.0, 0.0), (15.0, 0.0), (0.0, -15.0), (-15.0, 15.0), (15.0, -15.0))
        r0, c0 = grid_position(grid, X, Y)
        r, c = blob_centroid(out, r0, c0; radius = 12)
        @test r ≈ r0 atol = 0.1
        @test c ≈ c0 atol = 0.1
    end
end

@testset "Stereo alignment: two dewarped views of the same plane" begin
    cams = (make_test_camera(yaw_deg = -15.0), make_test_camera(yaw_deg = 25.0))
    grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)

    # World-plane particles at z = 0, rendered through each camera.
    rng = MersenneTwister(31)
    pts = [(56.0 * rand(rng) - 28.0, 56.0 * rand(rng) - 28.0) for _ in 1:350]
    imgs = map(cams) do cam
        img = zeros(512, 512)
        for (X, Y) in pts
            p = world_to_pixel(cam, (X, Y, 0.0))
            generate_gaussian_particle!(img, (p[1], p[2]), 6.0)
        end
        img
    end

    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss)

    # Dewarped through the true cameras, the two views align to a small
    # fraction of a pixel: the residual displacement field is ~zero.
    outs = map((cam, img) -> dewarp(ImageDewarper(cam, grid, (512, 512)), img),
               cams, imgs)
    result = run_piv(outs[1], outs[2], params)
    @test abs(median(result.u)) < 0.05
    @test abs(median(result.v)) < 0.05
    @test sum(result.outliers) <= 0.05 * length(result.u)

    # Same through fitted Soloff cameras (the realistic path: calibrate from
    # point pairs, then dewarp with the fit).
    world = calib_world_points()
    fits = map(cams) do cam
        calibrate_camera([world_to_pixel(cam, w) for w in world], world)
    end
    outs_fit = map((cam, img) -> dewarp(ImageDewarper(cam, grid, (512, 512)), img),
                   fits, imgs)
    result_fit = run_piv(outs_fit[1], outs_fit[2], params)
    @test abs(median(result_fit.u)) < 0.05
    @test abs(median(result_fit.v)) < 0.05
end

@testset "Dewarp validity mask and fill" begin
    cam = make_test_camera()
    # Grid wider than the camera's field of view (±36.6 mm at 512 px, f/dist = 7).
    grid = DewarpGrid(x = -60.0:0.5:60.0, y = -60.0:0.5:60.0)
    dw = ImageDewarper(cam, grid, (512, 512))
    @test dw.mask isa BitMatrix
    @test any(dw.mask) && !all(dw.mask)
    # The world origin is in view; the grid corners are not.
    r0, c0 = grid_position(grid, 0.0, 0.0)
    @test !dw.mask[round(Int, r0), round(Int, c0)]
    @test dw.mask[1, 1] && dw.mask[end, end]

    img = ones(512, 512)
    out = dewarp(dw, img)
    @test all(iszero, out[dw.mask])        # out-of-view nodes are zero-filled
    @test all(out[.!dw.mask] .≈ 1.0)       # in-view nodes resample the image
end

@testset "Dewarp precision and in-place reuse" begin
    cam = make_test_camera(yaw_deg = 10.0)
    grid = DewarpGrid(x = -20.0:0.5:20.0, y = -20.0:0.5:20.0)
    dw = ImageDewarper(cam, grid, (512, 512))

    rng = MersenneTwister(7)
    img = rand(rng, 512, 512)

    # Output eltype follows the image.
    out64 = dewarp(dw, img)
    @test out64 isa Matrix{Float64}
    out32 = dewarp(dw, Float32.(img))
    @test out32 isa Matrix{Float32}
    @test maximum(abs, out32 .- out64) < 1e-3

    # In-place form reuses the buffer and matches the allocating form.
    buf = similar(out64)
    @test dewarp!(buf, dw, img) === buf
    @test buf == out64

    @test_throws DimensionMismatch dewarp(dw, img[1:256, :])
    @test_throws DimensionMismatch dewarp!(zeros(3, 3), dw, img)
end
