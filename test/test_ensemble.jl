using Hammerhead
using Test
using Random
using Statistics

@testset "Ensemble & statistics" begin
    @testset "run_piv_ensemble on low-SNR pairs" begin
        rng = MersenneTwister(31)
        n = 96
        dv, du = -2.0, 1.5
        pairs = map(1:24) do _
            positions = [(rand(rng) * (n + 20) - 10, rand(rng) * (n + 20) - 10) for _ in 1:120]
            imgA, imgB = particle_pair((n, n), positions, dv, du)
            imgA .+= 1.5 .* rand(rng, n, n)  # heavy uncorrelated noise
            imgB .+= 1.5 .* rand(rng, n, n)
            (imgA, imgB)
        end
        params = PIVParameters(window_size = 32, overlap = 16)

        # Plain circular correlation keeps its ~0.15 px bias toward zero;
        # tolerances mirror the single-pair suite.
        ens = run_piv_ensemble(pairs, params; progress = false)
        @test ens isa PIVResult{Float64}
        @test all(abs.(ens.u .- du) .< 0.5)
        @test all(abs.(ens.v .- dv) .< 0.5)
        @test median(ens.u) ≈ du atol = 0.25
        @test median(ens.v) ≈ dv atol = 0.25
        @test sum(ens.outliers) <= 2

        # The ensemble is at least as accurate as a single noisy pair.
        single = run_piv(pairs[1]..., params)
        err(r) = median(abs.(filter(isfinite, r.u) .- du))
        @test err(ens) <= err(single)

        # Threaded matches serial exactly.
        e_ser = run_piv_ensemble(pairs, params; progress = false, threaded = false)
        e_thr = run_piv_ensemble(pairs, params; progress = false, threaded = true)
        @test isequal(e_ser.u, e_thr.u)
        @test isequal(e_ser.peak_ratio, e_thr.peak_ratio)

        # Multi-pass ensemble with a mask: masked windows stay NaN, and the
        # symmetric deformation cancels the circular-correlation bias.
        mask = falses(n, n)
        mask[1:32, 1:32] .= true
        multi = run_piv_ensemble(pairs, multipass_parameters([64, 32]);
                                 mask, progress = false)
        @test any(multi.mask)
        @test all(isnan, multi.u[multi.mask])
        @test median(multi.u[.!multi.mask]) ≈ du atol = 0.15
        @test median(multi.v[.!multi.mask]) ≈ dv atol = 0.15
        @test all(abs.(multi.u[.!multi.mask] .- du) .< 0.5)

        @test_throws ArgumentError run_piv_ensemble([], params; progress = false)
    end

    @testset "field_statistics" begin
        params = PIVParameters()
        x = [1.0, 2.0]
        y = [1.0, 2.0, 3.0]
        mk(uval, vval; out = falses(3, 2), msk = falses(3, 2)) =
            PIVResult(x, y, fill(uval, 3, 2), fill(vval, 3, 2),
                      ones(3, 2), zeros(3, 2), fill(NaN, 3, 2), fill(NaN, 3, 2),
                      out, msk, params)
        r1 = mk(2.0, 1.0)
        r2 = mk(4.0, -1.0)
        s = field_statistics([r1, r2])
        @test s.mean_u == fill(3.0, 3, 2)
        @test s.mean_v == fill(0.0, 3, 2)
        @test s.rms_u ≈ fill(1.0, 3, 2)
        @test s.rms_v ≈ fill(1.0, 3, 2)
        @test s.reynolds_uv ≈ fill(-1.0, 3, 2)
        @test s.count == fill(2, 3, 2)

        # Outliers excluded by default, included on request.
        out2 = falses(3, 2)
        out2[1, 1] = true
        r2o = mk(4.0, -1.0; out = out2)
        s2 = field_statistics([r1, r2o])
        @test s2.count[1, 1] == 1
        @test s2.mean_u[1, 1] == 2.0 && s2.rms_u[1, 1] == 0.0
        @test field_statistics([r1, r2o]; include_invalid = true).mean_u[1, 1] == 3.0

        # Masked cells (NaN) have no samples.
        msk = falses(3, 2)
        msk[2, 2] = true
        um = fill(2.0, 3, 2)
        um[2, 2] = NaN
        rm = PIVResult(x, y, um, fill(1.0, 3, 2), ones(3, 2), zeros(3, 2),
                       fill(NaN, 3, 2), fill(NaN, 3, 2), falses(3, 2), msk, params)
        sm = field_statistics([rm, rm])
        @test isnan(sm.mean_u[2, 2]) && sm.count[2, 2] == 0

        r_other = PIVResult([5.0, 6.0], y, zeros(3, 2), zeros(3, 2), ones(3, 2),
                            zeros(3, 2), fill(NaN, 3, 2), fill(NaN, 3, 2),
                            falses(3, 2), falses(3, 2), params)
        @test_throws ArgumentError field_statistics([r1, r_other])
        @test_throws ArgumentError field_statistics(PIVResult[])
    end

    @testset "validate_temporal!" begin
        params = PIVParameters()
        rng = MersenneTwister(5)
        x = collect(1.0:4.0)
        y = collect(1.0:4.0)
        results = [PIVResult(x, y, 3.0 .+ 0.05 .* randn(rng, 4, 4),
                             0.05 .* randn(rng, 4, 4), ones(4, 4), zeros(4, 4),
                             fill(NaN, 4, 4), fill(NaN, 4, 4),
                             falses(4, 4), falses(4, 4), params) for _ in 1:10]
        results[5].u[2, 3] = 50.0  # temporally inconsistent, spatially plausible
        validate_temporal!(results)
        @test results[5].outliers[2, 3]
        @test sum(sum(r.outliers) for r in results) == 1
        @test_throws ArgumentError validate_temporal!(results; threshold = 0)
    end

    @testset "power_spectrum" begin
        dt = 0.01
        f0 = 10.0
        A = 2.0
        t = (0:99) .* dt
        sig = A .* sin.(2π .* f0 .* t)
        f, psd = power_spectrum(sig; dt, window = :none)
        @test f[argmax(psd)] ≈ f0
        @test sum(psd) * (f[2] - f[1]) ≈ A^2 / 2 rtol = 0.02  # Parseval
        fh, psdh = power_spectrum(sig; dt)  # Hann default
        @test fh[argmax(psdh)] ≈ f0
        @test_throws ArgumentError power_spectrum(sig; dt = 0)
        @test_throws ArgumentError power_spectrum(sig; window = :hamming)
        @test_throws ArgumentError power_spectrum([1.0])
    end
end
