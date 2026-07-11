@testset "CPU backend plumbing" begin
    same_piv(a, b) =
        isequal(a.x, b.x) &&
        isequal(a.y, b.y) &&
        isequal(a.u, b.u) &&
        isequal(a.v, b.v) &&
        isequal(a.peak_ratio, b.peak_ratio) &&
        isequal(a.correlation_moment, b.correlation_moment) &&
        isequal(a.uncertainty_u, b.uncertainty_u) &&
        isequal(a.uncertainty_v, b.uncertainty_v) &&
        a.outliers == b.outliers &&
        a.mask == b.mask &&
        a.parameters == b.parameters &&
        isequal(a.correlation_planes, b.correlation_planes) &&
        a.scale == b.scale

    same_stereo(a, b) =
        isequal(a.x, b.x) &&
        isequal(a.y, b.y) &&
        a.z == b.z &&
        isequal(a.u, b.u) &&
        isequal(a.v, b.v) &&
        isequal(a.w, b.w) &&
        isequal(a.uncertainty_u, b.uncertainty_u) &&
        isequal(a.uncertainty_v, b.uncertainty_v) &&
        isequal(a.uncertainty_w, b.uncertainty_w) &&
        a.outliers == b.outliers &&
        a.mask == b.mask &&
        same_piv(a.cam1, b.cam1) &&
        same_piv(a.cam2, b.cam2) &&
        a.parameters == b.parameters &&
        a.scale == b.scale

    @test !(:CPUBackend in names(Hammerhead; all = true))
    @test !(:AbstractHammerheadBackend in names(Hammerhead; all = true))
    @test !(:_CPUBackend in names(Hammerhead))
    backend = :cpu
    cpu_backend = Hammerhead._resolve_backend(backend)
    @test cpu_backend === Hammerhead._DEFAULT_BACKEND
    @test Hammerhead._supports_fft(cpu_backend)
    @test !Hammerhead._supports_batched_fft(cpu_backend)
    @test Hammerhead._supports_fp64(cpu_backend)
    @test !Hammerhead._supports_unified_memory(cpu_backend)
    @test_throws ArgumentError Hammerhead._resolve_backend(:gpu)
    @test_throws ArgumentError piv_workspace(; backend = :gpu)

    rng = MersenneTwister(20260711)
    imgA = zeros(128, 128)
    imgB = zeros(128, 128)
    for _ in 1:450
        p = (rand(rng) * 148 - 10, rand(rng) * 148 - 10)
        add_particle!(imgA, p, 3.0)
        add_particle!(imgB, (p[1] + 2.0, p[2] - 1.25), 3.0)
    end
    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss)
    pairs = [(imgA, imgB), (imgB, imgA)]

    r_default = run_piv(imgA, imgB, params; threaded = false)
    r_backend = run_piv(imgA, imgB, params; backend, threaded = false)
    @test same_piv(r_backend, r_default)
    @test_throws ArgumentError run_piv(imgA, imgB, params; backend = :gpu,
                                       threaded = false)

    seq_default = run_piv_sequence(pairs, params; progress = false, threaded = false)
    seq_backend = run_piv_sequence(pairs, params; backend, progress = false,
                                   threaded = false)
    @test all(same_piv.(seq_backend, seq_default))

    ens_default = run_piv_ensemble(pairs, params; progress = false, threaded = false)
    ens_backend = run_piv_ensemble(pairs, params; backend, progress = false,
                                   threaded = false)
    @test same_piv(ens_backend, ens_default)

    cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
    grid = DewarpGrid(x = -20.0:0.25:20.0, y = -20.0:0.25:20.0)
    dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
    pts = [(44 * rand(rng) - 22, 44 * rand(rng) - 22) for _ in 1:220]
    A1, B1, A2, B2 = stereo_frames(cams, pts, (0.35, -0.2, 0.25))
    stereo_params = PIVParameters(window_size = 32, overlap = 16,
                                  padding = true, apodization = :gauss)
    stereo_default = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2],
                                    stereo_params; threaded = false)
    stereo_backend = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2],
                                    stereo_params; backend, threaded = false)
    @test same_stereo(stereo_backend, stereo_default)
end
