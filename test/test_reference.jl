using Hammerhead
using Test
using Statistics
using TiffImages

# PIV Challenge (2001) case A: tip vortex behind a transport aircraft model,
# with loss of seeding in the vortex core — see reference_images/A/readmeA.txt.
@testset "Reference Images (PIV Challenge A)" begin
    dir = joinpath(@__DIR__, "reference_images", "A")
    imgA = Float64.(TiffImages.load(joinpath(dir, "A001_1.tif")))
    imgB = Float64.(TiffImages.load(joinpath(dir, "A001_2.tif")))
    @test size(imgA) == size(imgB)

    passes = multipass_parameters([64, 32]; padding = true, apodization = :gauss)
    result = run_piv(imgA, imgB, passes)

    @test all(isfinite, result.u)
    @test all(isfinite, result.v)
    # The reference analysis uses 32 px windows with no offset, so in-plane
    # displacements stay within the quarter-window rule.
    valid_u = result.u[.!result.outliers]
    valid_v = result.v[.!result.outliers]
    @test maximum(hypot.(valid_u, valid_v)) < 16
    # Seeding loss is confined to the vortex core; most of the field validates.
    @test count(result.outliers) / length(result.outliers) < 0.25
end

# PIV Challenge (2014) case E subset: stereo vortex ring, cameras 1 and 3,
# three calibration planes and one frame pair — see
# reference_images/E/readmeE.txt. Real data without ground truth, so these
# are smoke-level bounds on the full Phase 5 chain (detection → Soloff →
# dewarp → self-calibration → 3C reconstruction), not accuracy claims.
@testset "Reference Images (PIV Challenge E, stereo)" begin
    dir = joinpath(@__DIR__, "reference_images", "E")
    zs = [-3.0, 0.0, 3.0]
    detect(cam, k) = detect_calibration_grid(
        load_image(joinpath(dir, "E_camera_$(cam)_z_$(k).png"));
        spacing = 15.0, two_level = true, level_separation = 3.0,
        origin_offset = (30.0, 7.5))

    grids1 = [detect(1, k) for k in (1, 4, 7)]
    grids3 = [detect(3, k) for k in (1, 4, 7)]
    @test all(g -> length(g.pixels) == 61, grids1)   # full grid, every plane
    @test all(g -> length(g.pixels) >= 20, grids3)   # tighter view, edge dots drift out
    @test all(g -> g.square !== nothing, [grids1; grids3])

    cam1 = calibrate_camera(grids1, zs)
    cam3 = calibrate_camera(grids3, zs)
    # ~0.65 / ~1.1 px RMS, dominated by the plate's dot-position tolerance.
    @test calibration_quality(cam1, grids1, zs).rms < 1.0
    @test calibration_quality(cam3, grids3, zs).rms < 1.6

    # Shared grid over the stereo overlap at camera 1's raw pixel scale.
    grid = DewarpGrid(x = -36.0:0.0836:22.0, y = -32.5:0.0836:29.0)
    dw1 = ImageDewarper(cam1, grid, (1024, 1024))
    dw3 = ImageDewarper(cam3, grid, (1024, 1024))
    @test count(dw1.mask .| dw3.mask) / length(dw1.mask) < 0.1

    f1 = [joinpath(dir, "E_camera_1_frame_000$(k).png") for k in (50, 51)]
    f3 = [joinpath(dir, "E_camera_3_frame_000$(k).png") for k in (50, 51)]
    dw1c, dw3c, report = self_calibrate(f1, f3, dw1, dw3)
    # The sheet sits ~0.7 mm behind the plate: the first pass must see the
    # misregistration and the first correction must remove nearly all of it.
    @test report.passes[1].disparity_rms > 1.5
    @test -1.2 < report.passes[1].plane.a < -0.2
    @test report.passes[end].disparity_rms < 0.8   # sheet-thickness noise floor
    @test report.passes[1].triangulation_rms < 0.3

    A1, B1 = load_image.(f1)
    A3, B3 = load_image.(f3)
    passes = multipass_parameters([64, 32, 32]; padding = true,
                                  apodization = :gauss, uncertainty = true)
    stereo = run_piv_stereo(A1, B1, A3, B3, dw1c, dw3c, passes)
    sel = .!(stereo.mask .| stereo.outliers)
    @test count(sel) / length(sel) > 0.9
    @test count(stereo.outliers .& .!stereo.mask) / count(.!stereo.mask) < 0.05
    # Displacements stay physical (|d| ≈ 0.22 mm max on this pair)...
    @test maximum(hypot.(stereo.u[sel], stereo.v[sel], stereo.w[sel])) < 0.5
    # ...and the uncertainties land in the measured decade, with the
    # out-of-plane component worse than in-plane (shallow-angle geometry).
    σu = median(filter(isfinite, stereo.uncertainty_u[sel]))
    σw = median(filter(isfinite, stereo.uncertainty_w[sel]))
    @test 0.0005 < σu < 0.01
    @test σu < σw < 0.05
end
