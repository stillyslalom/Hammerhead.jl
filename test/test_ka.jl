using KernelAbstractions   # loading the trigger package activates HammerheadKAExt

@testset "KA backend (KernelAbstractions CPU)" begin
    @test Base.get_extension(Hammerhead, :HammerheadKAExt) !== nothing
    kab = Hammerhead._resolve_backend(:ka)
    @test kab isa Hammerhead._AbstractHammerheadBackend
    # Device backends run the whole grid as one logical batch, not host-thread
    # fan-out.
    @test Hammerhead._engine_nchunks(kab, 8) == 1

    rng = MersenneTwister(20260712)
    image_size = (256, 256)
    positions = [(rand(rng) * 276 - 10, rand(rng) * 276 - 10) for _ in 1:1500]
    dv, du = 2.0, -1.25
    imgA, imgB = particle_pair(image_size, positions, dv, du)
    params = PIVParameters(window_size = 32, overlap = 16, padding = true, apodization = :gauss)

    r_cpu = run_piv(imgA, imgB, params; threaded = false)
    r_ka = run_piv(imgA, imgB, params; backend = :ka, threaded = false)
    valid = .!r_cpu.outliers .& .!r_cpu.mask

    # Equivalence to the CPU FFTW path (scientific tolerance, not bitwise — FFT
    # order may differ on other platforms/backends) and ground-truth recovery.
    @test maximum(abs.(r_ka.u[valid] .- r_cpu.u[valid])) < 1e-3
    @test maximum(abs.(r_ka.v[valid] .- r_cpu.v[valid])) < 1e-3
    @test maximum(abs.(r_ka.correlation_moment[valid] .- r_cpu.correlation_moment[valid])) < 1e-3
    @test isapprox(median(r_ka.u[valid]), du; atol = 0.05)
    @test isapprox(median(r_ka.v[valid]), dv; atol = 0.05)

    # Multipass deformation with the :gauss3-alternatives path exercised.
    schedule = Hammerhead.multipass_parameters([64, 32]; padding = true, apodization = :gauss)
    m_cpu = run_piv(imgA, imgB, schedule; threaded = false)
    m_ka = run_piv(imgA, imgB, schedule; backend = :ka, threaded = false)
    mvalid = .!m_cpu.outliers .& .!m_cpu.mask
    @test maximum(abs.(m_ka.u[mvalid] .- m_cpu.u[mvalid])) < 1e-3
    @test maximum(abs.(m_ka.v[mvalid] .- m_cpu.v[mvalid])) < 1e-3

    # Partially/fully masked interrogation windows.
    mask = falses(image_size)
    mask[100:160, 100:160] .= true
    k_cpu = run_piv(imgA, imgB, params; mask, threaded = false)
    k_ka = run_piv(imgA, imgB, params; backend = :ka, mask, threaded = false)
    @test k_ka.mask == k_cpu.mask
    kvalid = .!k_cpu.outliers .& .!k_cpu.mask
    @test maximum(abs.(k_ka.u[kvalid] .- k_cpu.u[kvalid])) < 1e-3
    @test all(isnan, k_ka.u[k_ka.mask])

    # The engine collapses to one batch regardless of `threaded`, so the two
    # paths are identical.
    r_ka_t = run_piv(imgA, imgB, params; backend = :ka, threaded = true)
    @test isequal(r_ka_t.u, r_ka.u) && isequal(r_ka_t.v, r_ka.v)

    # Out-of-scope options report a clear error rather than silently differing.
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(correlation_method = :phase); backend = :ka)
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(subpixel_method = :gauss2d); backend = :ka)
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(uncertainty = true, max_iterations = 2); backend = :ka)
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(keep_correlation_planes = true); backend = :ka)

    # A workspace built for a different backend is rejected.
    @test_throws ArgumentError run_piv(imgA, imgB, params;
        backend = :ka, workspace = piv_workspace())
    # A :ka workspace is accepted and matches the workspace-free path.
    ws = piv_workspace(; backend = :ka)
    r_ka_ws = run_piv(imgA, imgB, schedule; backend = :ka, workspace = ws, threaded = false)
    @test isequal(r_ka_ws.u, m_ka.u) && isequal(r_ka_ws.v, m_ka.v)
end
