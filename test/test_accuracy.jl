using Hammerhead
using Test
using Random
using Statistics

@testset "Accuracy tools (Phase 4)" begin
    @testset "subpixel_gauss9" begin
        # Rotated anisotropic Gaussian peak: the 2D regression is exact,
        # independent 1D fits are biased by the cross term.
        r0, c0 = 17.3, 15.6
        R = [exp(-(0.20 * (i - r0)^2 + 0.16 * (i - r0) * (j - c0) + 0.10 * (j - c0)^2))
             for i in 1:33, j in 1:33]
        peakloc = Tuple(argmax(R))
        ref9 = Hammerhead.subpixel_gauss9(R, peakloc)
        ref3 = Hammerhead.subpixel_gauss3(R, peakloc)
        err9 = hypot(ref9[1] - r0, ref9[2] - c0)
        err3 = hypot(ref3[1] - r0, ref3[2] - c0)
        @test err9 < 1e-8          # exact for a Gaussian
        @test err9 < err3          # beats the 1D fits on a rotated peak

        # Edge and degenerate cases fall back to gauss3.
        @test Hammerhead.subpixel_gauss9(R, (1, 5)) ==
              Hammerhead.subpixel_gauss3(R, (1, 5))
        flat = ones(9, 9)
        @test Hammerhead.subpixel_gauss9(flat, (5, 5)) ==
              Hammerhead.subpixel_gauss3(flat, (5, 5))

        # End-to-end through correlate and run_piv.
        imgA, imgB = particle_pair((64, 64), [(32.0, 32.0)], 2.7, -1.4)
        c = CrossCorrelator{Float64}((64, 64))
        res = correlate(c, imgA, imgB; subpixel = :gauss9)
        @test res.du ≈ -1.4 atol = 0.1
        @test res.dv ≈ 2.7 atol = 0.1
        @test PIVParameters(subpixel_method = :gauss9).subpixel_method === :gauss9
        @test_throws ArgumentError PIVParameters(subpixel_method = :gauss5)
    end

    @testset "smoothn" begin
        rng = MersenneTwister(41)
        truth = [sin(2π * i / 16) * cos(2π * j / 16) for i in 1:32, j in 1:32]
        noisy = truth .+ 0.2 .* randn(rng, 32, 32)

        z, s = smoothn(noisy)
        @test s > 0
        @test sqrt(mean((z .- truth) .^ 2)) < 0.5 * sqrt(mean((noisy .- truth) .^ 2))

        # Fixed s: more smoothing, less variance; constants pass through.
        z1, _ = smoothn(noisy; s = 0.1)
        z2, _ = smoothn(noisy; s = 100.0)
        @test var(z2) < var(z1) < var(noisy)
        zc, _ = smoothn(fill(3.0, 16, 16))
        @test zc ≈ fill(3.0, 16, 16) atol = 1e-8

        # Non-finite entries are filled from the smooth surface.
        holey = copy(noisy)
        holey[10, 12] = NaN
        zh, _ = smoothn(holey)
        @test isfinite(zh[10, 12])
        @test abs(zh[10, 12] - truth[10, 12]) < 0.5

        # Zero weight excludes a corrupted sample explicitly.
        spiked = copy(noisy)
        spiked[20, 20] += 30.0
        w = ones(32, 32)
        w[20, 20] = 0.0
        zw, _ = smoothn(spiked; weights = w)
        @test abs(zw[20, 20] - truth[20, 20]) < 0.5

        # robust = true resists the spike without explicit weights.
        zr, _ = smoothn(spiked; robust = true)
        zp, _ = smoothn(spiked)
        @test abs(zr[20, 20] - truth[20, 20]) < abs(zp[20, 20] - truth[20, 20])

        @test_throws ArgumentError smoothn(noisy; s = 0)
        @test_throws ArgumentError smoothn(noisy; weights = ones(4, 4))
        @test_throws ArgumentError smoothn(fill(NaN, 4, 4))
    end

    @testset "error_statistics" begin
        params = PIVParameters()
        x = collect(1.0:4.0)
        y = collect(1.0:3.0)
        u_true = [0.1 * xj + 0.2 * yi for yi in y, xj in x]
        v_true = [0.3 * xj - 0.1 * yi for yi in y, xj in x]
        out = falses(3, 4)
        out[2, 2] = true
        r = PIVResult(x, y, u_true .+ 0.5, v_true .- 0.25, ones(3, 4),
                      zeros(3, 4), out, falses(3, 4), params)

        es = error_statistics(r, (xj, yi) -> 0.1 * xj + 0.2 * yi,
                              (xj, yi) -> 0.3 * xj - 0.1 * yi)
        @test es.n == 11                       # outlier excluded
        @test es.bias_u ≈ 0.5 && es.bias_v ≈ -0.25
        @test es.rms_u ≈ 0.5 && es.rms_v ≈ 0.25
        @test isnan(es.err_u[2, 2])            # excluded cells are NaN
        @test es.err_u[1, 1] ≈ 0.5

        es_arr = error_statistics(r, u_true, v_true)   # array references
        @test es_arr.bias_u ≈ 0.5
        @test error_statistics(r, u_true, v_true; include_invalid = true).n == 12

        @test_throws ArgumentError error_statistics(r, zeros(2, 2), zeros(2, 2))
    end
end
