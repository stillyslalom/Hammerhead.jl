using Hammerhead
using Test
using Random
using Statistics

@testset "Masking" begin
    @testset "polygon_mask" begin
        m = polygon_mask((20, 20), [(5, 5), (15, 5), (15, 15), (5, 15)])
        @test m isa BitMatrix && size(m) == (20, 20)
        @test m[10, 10] && m[6, 6]
        @test !m[2, 2] && !m[10, 18] && !m[18, 10]
        @test 80 <= count(m) <= 130  # ≈ 11×11 square, modulo edge rasterization

        # Triangle: right half above the diagonal is outside.
        tri = polygon_mask((20, 20), [(1, 1), (19, 19), (1, 19)])
        @test tri[15, 5] && !tri[5, 15]

        # Vertices may exceed the image; the mask is clipped.
        big = polygon_mask((10, 10), [(-5, -5), (30, -5), (30, 30), (-5, 30)])
        @test all(big)

        @test_throws ArgumentError polygon_mask((20, 20), [(1, 1), (2, 2)])
    end

    @testset "automatic and morphological masks" begin
        seed = falses(9, 9); seed[5, 5] = true
        grown = grow_mask(seed, 2)
        @test count(grown) == 13
        @test shrink_mask(grown, 2) == seed
        @test grow_mask(seed, 0) == seed
        @test_throws ArgumentError grow_mask(seed, -1)

        img = zeros(16, 16); img[4:6, 7:9] .= 10
        bright = automatic_mask(img; threshold = 5.0)
        @test bright == (img .>= 5)
        edge = automatic_mask(img; method = :edge, threshold = 1.0)
        @test any(edge) && !edge[1, 1]
        pair = automatic_mask(img, reverse(img; dims = 1); threshold = 5.0)
        @test pair == ((img .>= 5) .| (reverse(img; dims = 1) .>= 5))
        @test_throws DimensionMismatch automatic_mask(img, zeros(2, 2))
        @test_throws ArgumentError automatic_mask(img; method = :magic)
    end

    @testset "universal_outlier_detection exclude" begin
        rng = MersenneTwister(7)
        u = 3.0 .+ 0.05 .* randn(rng, 8, 8)
        v = 0.05 .* randn(rng, 8, 8)
        exclude = falses(8, 8)
        exclude[3:4, 3:4] .= true
        u[exclude] .= NaN  # as in masked windows
        v[exclude] .= NaN
        flagged = universal_outlier_detection(u, v, 2.0; exclude)
        @test !any(flagged)  # NaN cells skipped and never poison neighbor medians
        @test_throws ArgumentError universal_outlier_detection(u, v, 2.0;
                                                               exclude = falses(4, 4))
    end

    @testset "masked correlate" begin
        rng = MersenneTwister(13)
        dv, du = 2.2, -1.7
        imgA, imgB = particle_pair((64, 64), [(40.0, 40.0), (20.0, 44.0)], dv, du)
        # Corrupt a corner (uncorrelated "reflection" noise in both frames).
        imgA[1:24, 1:24] .= 3 .* rand(rng, 24, 24)
        imgB[1:24, 1:24] .= 3 .* rand(rng, 24, 24)
        m = falses(64, 64)
        m[1:24, 1:24] .= true

        c = CrossCorrelator{Float64}((64, 64))
        res = correlate(c, imgA, imgB; mask = m)
        @test res.du ≈ du atol = 0.1
        @test res.dv ≈ dv atol = 0.1
        @test_throws DimensionMismatch correlate(c, imgA, imgB; mask = falses(32, 32))
    end

    @testset "run_piv with mask" begin
        rng = MersenneTwister(17)
        n = 128
        dv, du = 3.0, 2.0
        positions = [(rand(rng) * (n + 20) - 10, rand(rng) * (n + 20) - 10) for _ in 1:250]
        imgA, imgB = particle_pair((n, n), positions, dv, du)
        # A central "model" region with uncorrelated garbage in both frames.
        region = (49:80, 49:80)
        imgA[region...] .= 3 .* rand(rng, 32, 32)
        imgB[region...] .= 3 .* rand(rng, 32, 32)
        mask = falses(n, n)
        mask[region...] .= true

        params = PIVParameters(window_size = 32, overlap = 16)
        result = run_piv(imgA, imgB, params; mask)

        @test any(result.mask) && !all(result.mask)
        @test all(isnan, result.u[result.mask])
        @test all(isnan, result.peak_ratio[result.mask])
        @test !any(result.outliers .& result.mask)   # masked ≠ outlier
        # The unmasked field is unaffected by the garbage region.
        good_u = result.u[.!result.mask]
        @test median(good_u) ≈ du atol = 0.25
        @test median(result.v[.!result.mask]) ≈ dv atol = 0.25
        @test count(abs.(good_u .- du) .< 0.5) >= 0.85 * length(good_u)

        # mask_threshold = 1 keeps every window that has any valid pixel.
        result_all = run_piv(imgA, imgB, params; mask, mask_threshold = 1.0)
        @test count(result_all.mask) < count(result.mask)

        # Multi-pass: predictor filling keeps deformation finite; masked cells
        # stay NaN in the final result.
        multi = run_piv(imgA, imgB, multipass_parameters([64, 32]); mask)
        @test any(multi.mask)
        @test all(isnan, multi.u[multi.mask])
        @test all(isfinite, multi.u[.!multi.mask])
        @test median(multi.u[.!multi.mask]) ≈ du atol = 0.25

        # Float32 pipeline with mask.
        result32 = run_piv(Float32.(imgA), Float32.(imgB), params; mask)
        @test result32 isa PIVResult{Float32}
        @test all(isnan, result32.u[result32.mask])

        # Threaded matches serial with a mask.
        r_ser = run_piv(imgA, imgB, params; mask, threaded = false)
        r_thr = run_piv(imgA, imgB, params; mask, threaded = true)
        @test isequal(r_ser.u, r_thr.u)  # isequal: NaN == NaN at masked cells
        @test r_ser.mask == r_thr.mask

        # No mask: result.mask is all-false.
        @test !any(run_piv(imgA, imgB, params).mask)

        @test_throws DimensionMismatch run_piv(imgA, imgB, params; mask = falses(4, 4))
        @test_throws ArgumentError run_piv(imgA, imgB, params; mask, mask_threshold = 0.0)
        @test_throws ArgumentError run_piv(imgA, imgB, params; mask, mask_threshold = 1.5)
    end
end
