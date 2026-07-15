# Camera calibration models and dot-grid target detection (Phase 5, slice 1).
# The synthetic fixture: a known pinhole camera renders a calibration plate,
# detection + fitting must recover mappings that agree with the true camera.

"""
Pinhole camera looking at the world origin from `dist` along its optical
axis, rotated `yaw_deg` about the world Y axis. The negative row focal term
makes world +Y point up in the image (decreasing row).
"""
function make_test_camera(; f = 3500.0, cx = 256.0, cy = 256.0, yaw_deg = 0.0, dist = 500.0)
    th = deg2rad(yaw_deg)
    R = [cos(th) 0.0 -sin(th); 0.0 1.0 0.0; sin(th) 0.0 cos(th)]
    C = R' * [0.0, 0.0, -dist]          # camera center in world coordinates
    K = [f 0.0 cx; 0.0 -f cy; 0.0 0.0 1.0]
    return PinholeCamera(K, R, -R * C)
end

# World test points spanning the calibration volume.
calib_world_points(; step = 15.0, zs = (-3.0, 0.0, 3.0)) =
    [(x, y, z) for x in -45.0:step:45.0 for y in -45.0:step:45.0 for z in zs]

@testset "Pinhole camera model" begin
    @test_throws ArgumentError PinholeCamera(zeros(3, 3))
    @test_throws ArgumentError PinholeCamera(zeros(2, 2), zeros(3, 3), zeros(3))
    @test_throws ArgumentError PinholeCamera(zeros(3, 3), zeros(3, 3), zeros(2))
    @test_throws ArgumentError PinholeCamera([1.0 0 0 0; 0 1 0 0; 0 0 0 1])  # zero third-row rotation

    cam = make_test_camera(yaw_deg = 25.0)
    # Third row of P is normalized to a unit direction.
    @test hypot(cam.P[3, 1], cam.P[3, 2], cam.P[3, 3]) ≈ 1.0

    # world_to_pixel matches the homogeneous projection by hand.
    w = (10.0, -5.0, 1.5)
    h = cam.P * [w..., 1.0]
    @test world_to_pixel(cam, w) ≈ [h[1] / h[3], h[2] / h[3]]

    # pixel_to_world is the exact inverse on the given plane.
    for w in ((0.0, 0.0, 0.0), (20.0, 15.0, -2.5), (-31.0, 4.0, 3.0))
        p = world_to_pixel(cam, w)
        @test pixel_to_world(cam, p, w[3]) ≈ [w...] atol = 1e-9
    end
end

@testset "Pinhole DLT fit" begin
    cam = make_test_camera(yaw_deg = 25.0, f = 4000.0)
    world = calib_world_points()
    pixels = [world_to_pixel(cam, w) for w in world]

    fit = calibrate_camera(pixels, world; model = :pinhole)
    q = calibration_quality(fit, pixels, world)
    @test q.rms < 1e-9
    @test q.n == length(world)
    # Held-out point (not in the fit set).
    w = (7.3, -12.9, 1.2)
    @test world_to_pixel(fit, w) ≈ world_to_pixel(cam, w) atol = 1e-6

    # Noisy pixels: the fit averages the noise down.
    rng = MersenneTwister(4)
    noisy = [p .+ 0.05 .* randn(rng, 2) for p in pixels]
    fitn = calibrate_camera(noisy, world; model = :pinhole)
    @test calibration_quality(fitn, noisy, world).rms < 0.1
    @test norm(world_to_pixel(fitn, w) - world_to_pixel(cam, w)) < 0.1

    @test_throws ArgumentError calibrate_camera(pixels[1:5], world[1:5]; model = :pinhole)
    @test_throws ArgumentError calibrate_camera(pixels, world[1:9]; model = :pinhole)
    @test_throws ArgumentError calibrate_camera(pixels, world; model = :dlt)
    # A single plane is degenerate for the pinhole model.
    planar = [(x, y, 0.0) for x in -45.0:15.0:45.0 for y in -45.0:15.0:45.0]
    planar_px = [world_to_pixel(cam, w) for w in planar]
    @test_throws ArgumentError calibrate_camera(planar_px, planar; model = :pinhole)
end

@testset "Soloff polynomial fit" begin
    cam = make_test_camera(yaw_deg = 25.0, f = 4000.0)
    world = calib_world_points()
    # Radially distorted camera: beyond the pinhole model, within Soloff's.
    distort(p) = (d = p .- (256.0, 256.0); p .+ 5e-8 * (d[1]^2 + d[2]^2) .* d)
    pixels = [distort(world_to_pixel(cam, w)) for w in world]

    fit = calibrate_camera(pixels, world)          # :soloff is the default
    @test fit isa SoloffCamera
    q = calibration_quality(fit, pixels, world)
    # The distortion (~2 px at the corners) is cubic in *pixel* space, so a
    # world-space cubic absorbs most but not all of it.
    @test q.rms < 0.1
    # The pinhole model cannot absorb the distortion.
    fitp = calibrate_camera(pixels, world; model = :pinhole)
    @test calibration_quality(fitp, pixels, world).rms > 5 * q.rms

    # Newton inversion round-trips through the polynomial.
    for w in ((0.0, 0.0, 0.0), (20.0, 15.0, -2.5), (-31.0, 4.0, 3.0))
        p = world_to_pixel(fit, w)
        @test pixel_to_world(fit, p, w[3]) ≈ [w...] atol = 1e-8
    end

    @test_throws ArgumentError calibrate_camera(pixels[1:18], world[1:18])
    two_planes = [w for w in world if w[3] != 0.0]
    two_px = [distort(world_to_pixel(cam, w)) for w in two_planes]
    @test_throws ArgumentError calibrate_camera(two_px, two_planes)

    errs = reprojection_errors(fit, pixels, world)
    @test length(errs) == length(world)
    @test q.max ≈ maximum(errs)
end

@testset "Target detection: single-level plate" begin
    cam = make_test_camera()
    img = render_calibration_target(cam, (512, 512); spacing = 15.0,
                                    marker_square = (-30.0, -7.5),
                                    marker_triangle = (-15.0, -7.5))
    @test all(0.0 .<= img .<= 1.0)

    grid = detect_calibration_grid(img; spacing = 15.0, origin_offset = (30.0, 7.5))
    @test length(grid.pixels) >= 20
    @test all(iszero, grid.level)
    @test all(iseven(i[1]) && iseven(i[2]) for i in grid.indices)  # single level
    @test grid.square !== nothing && grid.triangle !== nothing

    # Every detected dot's claimed world position projects back onto its
    # measured centroid through the true camera: indexing, orientation, and
    # origin are all correct, and centroids are subpixel-exact.
    px, wd = calibration_points(grid, 0.0)
    errs = [norm(world_to_pixel(cam, wd[i]) - px[i]) for i in eachindex(px)]
    @test maximum(errs) < 1e-6

    # The origin dot is the one 30 mm right of / 7.5 mm above the square.
    io = findfirst(==((0, 0)), grid.indices)
    @test io !== nothing
    @test grid.pixels[io] ≈ world_to_pixel(cam, (0.0, 0.0, 0.0)) atol = 1e-6

    # Fiducial-oriented indexing is invariant to camera roll. A 180-degree
    # sensor roll used to reverse both image-anchored axes.
    gf = detect_calibration_grid(img; spacing = 15.0,
                                 origin_offset = (30.0, 7.5),
                                 orientation = :fiducials)
    rolled = reverse(img; dims = (1, 2))
    gr = detect_calibration_grid(rolled; spacing = 15.0,
                                 origin_offset = (30.0, 7.5),
                                 orientation = :fiducials)
    for (p, idx) in zip(gf.pixels, gf.indices)
        pr = [size(img, 2) + 1 - p[1], size(img, 1) + 1 - p[2]]
        j = argmin([norm(q - pr) for q in gr.pixels])
        @test gr.indices[j] == idx
    end

    # Detection survives sensor noise with subpixel accuracy.
    rng = MersenneTwister(9)
    noisy = clamp.(img .+ 0.02 .* randn(rng, size(img)...), 0.0, 1.0)
    gn = detect_calibration_grid(noisy; spacing = 15.0, origin_offset = (30.0, 7.5))
    pxn, wdn = calibration_points(gn, 0.0)
    errsn = [norm(world_to_pixel(cam, wdn[i]) - pxn[i]) for i in eachindex(pxn)]
    @test length(pxn) >= 20
    @test maximum(errsn) < 0.1

    # Without markers, detection still indexes a lattice (relative frame).
    plain = render_calibration_target(cam, (512, 512); spacing = 15.0)
    gp = detect_calibration_grid(plain; spacing = 15.0)
    @test gp.square === nothing && gp.triangle === nothing
    @test any(==((0, 0)), gp.indices)
    @test_throws ArgumentError detect_calibration_grid(plain; spacing = 15.0,
                                                       orientation = :fiducials)
    # ... but an origin_offset without a marker is an error.
    @test_throws ArgumentError detect_calibration_grid(plain; spacing = 15.0,
                                                       origin_offset = (30.0, 7.5))

    @test_throws ArgumentError detect_calibration_grid(img; spacing = -1.0)
    @test_throws ArgumentError detect_calibration_grid(img; spacing = 15.0,
                                                       two_level = true)
    @test_throws ArgumentError detect_calibration_grid(img; spacing = 15.0,
                                                       origin_level = :middle)
    @test_throws ArgumentError detect_calibration_grid(zeros(64, 64); spacing = 15.0)
end

@testset "Target detection: two-level plate, stereo pair" begin
    # The slice-1 stereo rig: two cameras, a 4E-style two-level plate at
    # three Z positions, full detect → calibrate per camera.
    cams = (make_test_camera(yaw_deg = 0.0), make_test_camera(yaw_deg = 25.0))
    zs = [-3.0, 0.0, 3.0]
    fits = SoloffCamera[]
    for cam in cams
        grids = CalibrationGrid[]
        for z in zs
            img = render_calibration_target(cam, (512, 512); spacing = 15.0, z = z,
                                            two_level = true, level_separation = 3.0,
                                            marker_square = (-30.0, -7.5),
                                            marker_triangle = (-15.0, -7.5))
            g = detect_calibration_grid(img; spacing = 15.0, two_level = true,
                                        level_separation = 3.0,
                                        origin_offset = (30.0, 7.5))
            @test length(g.pixels) >= 40
            # Both levels present, distinguished by index parity.
            @test 0.3 <= sum(g.level) / length(g.level) <= 0.7
            @test all(g.level[i] == (isodd(g.indices[i][1]) ? 1 : 0)
                      for i in eachindex(g.level))
            push!(grids, g)
        end
        # Detected world positions project through the *true* camera onto the
        # measured centroids (level Z offsets included). The tolerance is the
        # perspective centroid bias of the tilted circular dots, not noise.
        pxs = reduce(vcat, (calibration_points(g, z)[1] for (g, z) in zip(grids, zs)))
        wds = reduce(vcat, (calibration_points(g, z)[2] for (g, z) in zip(grids, zs)))
        errs = [norm(world_to_pixel(cam, wds[i]) - pxs[i]) for i in eachindex(pxs)]
        @test maximum(errs) < 0.15

        fit = calibrate_camera(grids, zs)
        @test calibration_quality(fit, pxs, wds).rms < 0.1
        # Grid/zs convenience method sees the same point set.
        @test calibration_quality(fit, grids, zs) == calibration_quality(fit, pxs, wds)
        @test_throws ArgumentError calibration_quality(fit, grids, zs[1:2])
        fitp = calibrate_camera(grids, zs; model = :pinhole)
        @test calibration_quality(fitp, pxs, wds).rms < 0.1
        push!(fits, fit)

        @test_throws ArgumentError calibrate_camera(grids, zs[1:2])
    end

    # Stereo consistency: both fitted cameras agree with their true cameras
    # on shared world points — the marker-anchored frame is common.
    for w in ((0.0, 0.0, 0.0), (15.0, 7.5, 1.0), (-22.5, -15.0, -2.0))
        for (fit, cam) in zip(fits, cams)
            @test norm(world_to_pixel(fit, w) - world_to_pixel(cam, w)) < 0.15
        end
    end
end
