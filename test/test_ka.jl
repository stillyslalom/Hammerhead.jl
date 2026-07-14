@testset "KA backend (KernelAbstractions CPU)" begin
    # :ka is built into the core package — no trigger packages needed.
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

    @testset "cooperative regional-max peak finding" begin
        # Exercise the analysis kernel directly on crafted planes so ordering,
        # plateau ties, non-positive secondaries, and NaN handling are checked
        # independently of FFT round-off.
        function ka_regional_peaks(R, K = 4)
            T = eltype(R)
            nr, nc = size(R)
            Rt = zeros(T, nr, nc, 2) # plane-major batch plus trailing pad
            copyto!(view(Rt, :, :, 1), R)
            vals = zeros(T, K, 1)
            locs = zeros(Int32, 2, K, 1)
            out = zeros(T, 5 + 2 * (K - 1), 1)
            dev = Hammerhead.CPU()
            Hammerhead._ka_analyze!(dev, Hammerhead._KA_TPW)(
                out, vals, locs, Rt, true, false, K, nr, nc,
                Val(Hammerhead._KA_TPW); ndrange = Hammerhead._KA_TPW)
            Hammerhead.KernelAbstractions.synchronize(dev)
            found = round(Int, out[5, 1])
            return [(value = vals[m, 1],
                     location = (Int(locs[1, m, 1]), Int(locs[2, m, 1])))
                    for m in 1:found]
        end

        nearby = zeros(16, 16)
        nearby[8, 8] = 1.0
        nearby[8, 9] = 0.1
        nearby[8, 10] = 0.8 # distinct maximum inside the exclusion radius
        nearby[5, 3] = 0.6
        nearby[12, 13] = 0.6 # equal value: earlier column-major point wins
        @test ka_regional_peaks(nearby) ==
              find_peaks(nearby, 4; peak_finder = :regionalmax)

        plateau = ones(7, 9)
        @test ka_regional_peaks(plateau) ==
              find_peaks(plateau, 4; peak_finder = :regionalmax)

        nonpositive = fill(-2.0, 8, 8)
        nonpositive[3, 4] = -1.0
        @test ka_regional_peaks(nonpositive) ==
              find_peaks(nonpositive, 4; peak_finder = :regionalmax)

        allnan = fill(NaN, 8, 8)
        @test isempty(ka_regional_peaks(allnan))
    end

    # End-to-end production option: parameter plumbing plus batched FFT and
    # cooperative regional-max analysis must remain equivalent to the CPU.
    rparams = PIVParameters(window_size = 32, overlap = 16, padding = true,
                            apodization = :gauss, peak_finder = :regionalmax)
    rr_cpu = run_piv(imgA, imgB, rparams; threaded = false)
    rr_ka = run_piv(imgA, imgB, rparams; backend = :ka, threaded = false)
    rrvalid = .!rr_cpu.outliers .& .!rr_cpu.mask
    @test maximum(abs.(rr_ka.u[rrvalid] .- rr_cpu.u[rrvalid])) < 1e-3
    @test maximum(abs.(rr_ka.v[rrvalid] .- rr_cpu.v[rrvalid])) < 1e-3
    @test maximum(abs.(rr_ka.peak_ratio[rrvalid] .- rr_cpu.peak_ratio[rrvalid])) < 1e-3

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

    # Filtered phase correlation uses the same robust normalized spectrum as
    # the CPU path, including padded FFTs and zero-energy frequency bins.
    phaseparams = PIVParameters(window_size = 32, overlap = 16, padding = true,
                                apodization = :gauss, correlation_method = :phase)
    phase_cpu = run_piv(imgA, imgB, phaseparams; threaded = false)
    phase_ka = run_piv(imgA, imgB, phaseparams; backend = :ka, threaded = false)
    phasevalid = .!phase_cpu.outliers .& .!phase_cpu.mask
    @test maximum(abs.(phase_ka.u[phasevalid] .- phase_cpu.u[phasevalid])) < 1e-3
    @test maximum(abs.(phase_ka.v[phasevalid] .- phase_cpu.v[phasevalid])) < 1e-3
    @test all(isfinite, phase_ka.u[phasevalid])

    # Zero-energy bins remain zero instead of producing NaN/Inf. The second
    # batch slice confirms that the 2D filter repeats over the batched layout.
    phase_CA = zeros(ComplexF64, 4, 6, 2)
    phase_CB = zeros(ComplexF64, 4, 6, 2)
    phase_W = reshape(collect(1.0:24.0), 4, 6)
    phase_dev = Hammerhead.CPU()
    Hammerhead._ka_phasepower!(phase_dev)(phase_CA, phase_CB, phase_W,
        length(phase_W); ndrange = length(phase_CA))
    Hammerhead.KernelAbstractions.synchronize(phase_dev)
    @test all(iszero, phase_CA)

    # A shared workspace must not reuse a cross engine for phase correlation.
    phase_ws = piv_workspace(; backend = :ka)
    cross_ws_result = run_piv(imgA, imgB, params; backend = :ka,
                              workspace = phase_ws, threaded = false)
    phase_ws_result = run_piv(imgA, imgB, phaseparams; backend = :ka,
                              workspace = phase_ws, threaded = false)
    @test isequal(cross_ws_result.u, r_ka.u)
    @test isequal(phase_ws_result.u, phase_ka.u)
    @test length(phase_ws.engines) == 2

    # Out-of-scope options report a clear error rather than silently differing.
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(subpixel_method = :gauss2d); backend = :ka)
    @test Hammerhead._supports_fp64(kab)
    uqparams = PIVParameters(window_size = 32, overlap = 16, padding = true,
                             apodization = :gauss, uncertainty = true,
                             max_iterations = 2)
    uqA, uqB = imgA[1:96, 1:96], imgB[1:96, 1:96]
    uq_cpu = run_piv(uqA, uqB, uqparams; threaded = false)
    uq_ka = run_piv(uqA, uqB, uqparams; backend = :ka, threaded = false)
    uq_hybrid = run_piv(uqA, uqB, uqparams; backend = :ka,
                        uncertainty_backend = :cpu, threaded = true)
    uqvalid = isfinite.(uq_cpu.uncertainty_u) .& isfinite.(uq_ka.uncertainty_u)
    vqvalid = isfinite.(uq_cpu.uncertainty_v) .& isfinite.(uq_ka.uncertainty_v)
    @test any(uqvalid) && any(vqvalid)
    @test maximum(abs.(uq_ka.uncertainty_u[uqvalid] .-
                       uq_cpu.uncertainty_u[uqvalid])) < 1e-10
    @test maximum(abs.(uq_ka.uncertainty_v[vqvalid] .-
                       uq_cpu.uncertainty_v[vqvalid])) < 1e-10
    hqvalid = isfinite.(uq_cpu.uncertainty_u) .& isfinite.(uq_hybrid.uncertainty_u)
    hvvalid = isfinite.(uq_cpu.uncertainty_v) .& isfinite.(uq_hybrid.uncertainty_v)
    @test isequal(uq_hybrid.u, uq_ka.u) && isequal(uq_hybrid.v, uq_ka.v)
    @test maximum(abs.(uq_hybrid.uncertainty_u[hqvalid] .-
                       uq_cpu.uncertainty_u[hqvalid])) < 1e-10
    @test maximum(abs.(uq_hybrid.uncertainty_v[hvvalid] .-
                       uq_cpu.uncertainty_v[hvvalid])) < 1e-10
    @test_throws ArgumentError run_piv(uqA, uqB, uqparams; backend = :ka,
                                       uncertainty_backend = :automatic)

    config_bench = benchmark_piv_configurations(uqA, uqB, uqparams;
                                                backends = (:cpu, :ka),
                                                samples = 1, warmup = false)
    @test [r.configuration for r in config_bench] == [:cpu, :device, :hybrid]
    @test all(r.seconds > 0 && isfinite(r.speedup) for r in config_bench)
    @test config_bench[1].max_vector_delta == 0
    @test config_bench[1].max_uncertainty_delta == 0
    @test_throws ArgumentError run_piv(imgA, imgB,
        PIVParameters(keep_correlation_planes = true); backend = :ka)

    # A workspace built for a different backend is rejected.
    @test_throws ArgumentError run_piv(imgA, imgB, params;
        backend = :ka, workspace = piv_workspace())
    # A :ka workspace is accepted and matches the workspace-free path.
    ws = piv_workspace(; backend = :ka)
    r_ka_ws = run_piv(imgA, imgB, schedule; backend = :ka, workspace = ws, threaded = false)
    @test isequal(r_ka_ws.u, m_ka.u) && isequal(r_ka_ws.v, m_ka.v)
    # The workspace pools engines per window configuration (one per pass of
    # the 64/32 schedule) plus the deform context (staged coefficients + warp
    # buffers); a second run reuses the identical objects — and with them the
    # batch buffers and FFT plans — with identical results.
    @test length(ws.engines) == 3
    @test haskey(ws.engines, (:ka_deform, Float64, size(imgA)))
    engine_ids = Dict(k => map(objectid, v) for (k, v) in ws.engines)
    r_ka_ws2 = run_piv(imgA, imgB, schedule; backend = :ka, workspace = ws, threaded = false)
    @test isequal(r_ka_ws2.u, m_ka.u) && isequal(r_ka_ws2.v, m_ka.v)
    @test length(ws.engines) == 3
    @test all(map(objectid, ws.engines[k]) == engine_ids[k] for k in keys(engine_ids))

    @testset "ensemble on :ka" begin
        # Same flow in every pair (the ensemble assumption); different particle
        # sets so the summed planes actually pool information across pairs.
        pos2 = [(rand(rng) * 276 - 10, rand(rng) * 276 - 10) for _ in 1:1500]
        imgA2, imgB2 = particle_pair(image_size, pos2, dv, du)
        pairs = [(imgA, imgB), (imgA2, imgB2)]

        e_cpu = run_piv_ensemble(pairs, params; progress = false, threaded = false)
        e_ka = run_piv_ensemble(pairs, params; backend = :ka, progress = false,
                                threaded = false)
        evalid = .!e_cpu.outliers .& .!e_cpu.mask
        @test maximum(abs.(e_ka.u[evalid] .- e_cpu.u[evalid])) < 1e-3
        @test maximum(abs.(e_ka.v[evalid] .- e_cpu.v[evalid])) < 1e-3
        @test maximum(abs.(e_ka.peak_ratio[evalid] .- e_cpu.peak_ratio[evalid])) < 1e-3
        @test maximum(abs.(e_ka.correlation_moment[evalid] .-
                           e_cpu.correlation_moment[evalid])) < 1e-3
        @test isapprox(median(e_ka.u[evalid]), du; atol = 0.05)
        @test isapprox(median(e_ka.v[evalid]), dv; atol = 0.05)

        # Phase spectra are normalized before their planes enter the same
        # device-resident ensemble accumulator.
        pe_cpu = run_piv_ensemble(pairs, phaseparams; progress = false,
                                  threaded = false)
        pe_ka = run_piv_ensemble(pairs, phaseparams; backend = :ka,
                                 progress = false, threaded = false)
        pevalid = .!pe_cpu.outliers .& .!pe_cpu.mask
        @test maximum(abs.(pe_ka.u[pevalid] .- pe_cpu.u[pevalid])) < 1e-3
        @test maximum(abs.(pe_ka.v[pevalid] .- pe_cpu.v[pevalid])) < 1e-3
        @test all(isfinite, pe_ka.u[pevalid])

        # Multi-pass ensemble: the shared-predictor deformation path plus the
        # predictor-relative alternative peaks in the device analyze/scatter.
        me_cpu = run_piv_ensemble(pairs, schedule; progress = false, threaded = false)
        me_ka = run_piv_ensemble(pairs, schedule; backend = :ka, progress = false,
                                 threaded = false)
        mevalid = .!me_cpu.outliers .& .!me_cpu.mask
        @test maximum(abs.(me_ka.u[mevalid] .- me_cpu.u[mevalid])) < 1e-3
        @test maximum(abs.(me_ka.v[mevalid] .- me_cpu.v[mevalid])) < 1e-3

        # Masked windows accumulate identically to the CPU path.
        ke_cpu = run_piv_ensemble(pairs, params; mask, progress = false,
                                  threaded = false)
        ke_ka = run_piv_ensemble(pairs, params; backend = :ka, mask,
                                 progress = false, threaded = false)
        @test ke_ka.mask == ke_cpu.mask
        kevalid = .!ke_cpu.outliers .& .!ke_cpu.mask
        @test maximum(abs.(ke_ka.u[kevalid] .- ke_cpu.u[kevalid])) < 1e-3
        @test all(isnan, ke_ka.u[ke_ka.mask])

        # The engine collapses to one batch regardless of `threaded`.
        e_ka_t = run_piv_ensemble(pairs, params; backend = :ka, progress = false,
                                  threaded = true)
        @test isequal(e_ka_t.u, e_ka.u) && isequal(e_ka_t.v, e_ka.v)

        # Correlation-statistics UQ stays in Float64 on the backend and pools
        # per-window statistics across pairs before returning final scalars.
        uqparams = PIVParameters(window_size = 32, overlap = 16, padding = true,
                                 apodization = :gauss, uncertainty = true)
        smallpairs = [(imgA[1:96, 1:96], imgB[1:96, 1:96]),
                      (imgA2[1:96, 1:96], imgB2[1:96, 1:96])]
        uq_cpu = run_piv_ensemble(smallpairs, uqparams; progress = false,
                                  threaded = false)
        uq_ka = run_piv_ensemble(smallpairs, uqparams; backend = :ka,
                                 progress = false, threaded = false)
        uqvalid = isfinite.(uq_cpu.uncertainty_u) .& isfinite.(uq_ka.uncertainty_u)
        vqvalid = isfinite.(uq_cpu.uncertainty_v) .& isfinite.(uq_ka.uncertainty_v)
        @test any(uqvalid) && any(vqvalid)
        @test maximum(abs.(uq_ka.uncertainty_u[uqvalid] .-
                           uq_cpu.uncertainty_u[uqvalid])) < 1e-10
        @test maximum(abs.(uq_ka.uncertainty_v[vqvalid] .-
                           uq_cpu.uncertainty_v[vqvalid])) < 1e-10

        # Remaining out-of-scope option is still rejected up front.
        @test_throws ArgumentError run_piv_ensemble(pairs,
            PIVParameters(keep_correlation_planes = true); backend = :ka,
            progress = false)
    end

    @testset "device deformation kernel" begin
        # Direct comparison of the portable deformation kernel against the CPU
        # cubic-B-spline path (Interpolations.jl) on a smooth predictor: same
        # interpolation model, different evaluation order, so agreement is
        # floating-point-level, not bitwise.
        small = imgA[1:96, 1:128]
        smallB = imgB[1:96, 1:128]
        itpA = Hammerhead.image_interpolant(small, Float64)
        itpB = Hammerhead.image_interpolant(smallB, Float64)
        py = [16.5, 48.5, 80.5]
        px = [16.5, 48.5, 80.5, 112.5]
        pu = [0.8 * sin(0.05 * yi) + 0.3 * cos(0.04 * xj) for yi in py, xj in px]
        pv = [0.5 * cos(0.03 * yi) - 0.4 * sin(0.06 * xj) for yi in py, xj in px]
        pred = (x = px, y = py, u = pu, v = pv)
        gx = [20.5, 60.5, 100.5]
        gy = [20.5, 50.5]
        wA_cpu, wB_cpu, u_cpu, v_cpu = Hammerhead.apply_predictor(small, smallB,
            itpA, itpB, pred, gx, gy, Float64; threaded = false)
        wA_ka, wB_ka, u_ka, v_ka = Hammerhead.apply_predictor(kab, small, smallB,
            itpA, itpB, pred, gx, gy, Float64; threaded = false)
        # Pass-grid predictor values use the identical host evaluation.
        @test isequal(u_ka, u_cpu) && isequal(v_ka, v_cpu)
        @test maximum(abs.(wA_ka .- wA_cpu)) < 1e-12
        @test maximum(abs.(wB_ka .- wB_cpu)) < 1e-12
        # No predictor: pass-through, like the CPU path.
        pA, pB, z_u, z_v = Hammerhead.apply_predictor(kab, small, smallB,
            nothing, nothing, nothing, gx, gy, Float64)
        @test pA === small && pB === smallB && all(iszero, z_u) && all(iszero, z_v)
        # The staged-context path the drivers (and the device extensions) use:
        # coefficients copied into the context once, warp output in the
        # context's resident buffers — values identical to the direct path
        # (same kernel, same inputs).
        ctx = Hammerhead._deform_context(kab, nothing, itpA, itpB,
                                         size(small), Float64)
        wA_ctx, wB_ctx, u_ctx, v_ctx = Hammerhead.apply_predictor(kab, small,
            smallB, itpA, itpB, pred, gx, gy, Float64; ctx)
        @test wA_ctx === ctx.warpA && wB_ctx === ctx.warpB
        @test isequal(wA_ctx, wA_ka) && isequal(wB_ctx, wB_ka)
        @test isequal(u_ctx, u_ka) && isequal(v_ctx, v_ka)
    end

    @testset "stereo on :ka" begin
        # The stereo driver forwards the backend to its per-camera run_piv
        # calls (dewarping and 3C reconstruction stay on the CPU).
        cams = (make_test_camera(yaw_deg = -20.0), make_test_camera(yaw_deg = 20.0))
        dgrid = DewarpGrid(x = -20.0:0.25:20.0, y = -20.0:0.25:20.0)
        dws = map(cam -> ImageDewarper(cam, dgrid, (512, 512)), cams)
        pts = [(44 * rand(rng) - 22, 44 * rand(rng) - 22) for _ in 1:220]
        A1, B1, A2, B2 = stereo_frames(cams, pts, (0.35, -0.2, 0.25))
        s_cpu = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params;
                               threaded = false)
        s_ka = run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2], params;
                              backend = :ka, threaded = false)
        svalid = .!s_cpu.outliers .& .!s_cpu.mask .&
                 isfinite.(s_cpu.u) .& isfinite.(s_ka.u)
        @test any(svalid)
        @test maximum(abs.(s_ka.u[svalid] .- s_cpu.u[svalid])) < 1e-3
        @test maximum(abs.(s_ka.v[svalid] .- s_cpu.v[svalid])) < 1e-3
        @test maximum(abs.(s_ka.w[svalid] .- s_cpu.w[svalid])) < 1e-3
    end
end
