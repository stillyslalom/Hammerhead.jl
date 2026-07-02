using Hammerhead
using Test
using Random
using Statistics
using JLD2: jldopen

# Wieneke (2015) correlation-statistics uncertainty. The sweep below
# reproduces the paper's own validation (its figure 6): on uniform flow with
# increasing image noise, the predicted σ must track the measured RMS error —
# right magnitude and monotone trend. Comparisons run over valid (non-outlier)
# vectors and use the median of the predictions: the paper assumes outliers
# are removed before comparison, and near-outlier windows legitimately report
# huge σ (that is how the method flags them), which would dominate an RMS.
@testset "Uncertainty quantification (Wieneke 2015)" begin
    image_size = (160, 160)
    dv, du = 0.3, 0.6
    passes = multipass_parameters([32, 32]; padding = true, apodization = :gauss,
                                  uncertainty = true)

    function noisy_pair(rng, noise)
        positions = [(rand(rng) * 180 - 10, rand(rng) * 180 - 10) for _ in 1:420]
        imgA, imgB = particle_pair(image_size, positions, dv, du)
        if noise > 0
            imgA .+= noise .* randn(rng, image_size...)
            imgB .+= noise .* randn(rng, image_size...)
        end
        return imgA, imgB
    end

    @testset "Predicted σ tracks measured RMS over a noise sweep" begin
        rng = MersenneTwister(1701)
        med_pred = Float64[]
        for noise in (0.02, 0.05, 0.1, 0.2)
            imgA, imgB = noisy_pair(rng, noise)
            r = run_piv(imgA, imgB, passes)
            valid = .!(r.outliers .| r.mask)
            es = error_statistics(r, (x, y) -> du, (x, y) -> dv)
            rms_meas = sqrt((es.rms_u^2 + es.rms_v^2) / 2)
            σ = [s for (s, ok) in zip(r.uncertainty_u, valid) if ok && isfinite(s)]
            σv = [s for (s, ok) in zip(r.uncertainty_v, valid) if ok && isfinite(s)]
            # Essentially every valid window must yield an estimate here.
            @test length(σ) >= 0.98 * count(valid)
            @test all(>=(0), σ)
            # Right magnitude: within a factor ~2 of the measured error.
            @test 0.4 * rms_meas < median(σ) < 2 * rms_meas
            @test 0.4 * rms_meas < median(σv) < 2 * rms_meas
            push!(med_pred, median(σ))
        end
        # Monotone trend across the sweep.
        @test issorted(med_pred)
        @test med_pred[end] > 4 * med_pred[1]
    end

    @testset "Clean images give small σ; threaded ≡ serial" begin
        rng = MersenneTwister(1702)
        imgA, imgB = noisy_pair(rng, 0.0)
        r = run_piv(imgA, imgB, passes)
        σ = filter(isfinite, r.uncertainty_u)
        @test !isempty(σ)
        @test median(σ) < 0.02

        r_ser = run_piv(imgA, imgB, passes; threaded = false)
        r_thr = run_piv(imgA, imgB, passes; threaded = true)
        @test isequal(r_ser.uncertainty_u, r_thr.uncertainty_u)
        @test isequal(r_ser.uncertainty_v, r_thr.uncertainty_v)
    end

    @testset "Disabled by default: NaN fields" begin
        imgA, imgB = particle_pair((64, 64), [(32.0, 32.0), (16.0, 48.0)], dv, du)
        r = run_piv(imgA, imgB, PIVParameters(window_size = 32, overlap = 16))
        @test size(r.uncertainty_u) == size(r.u)
        @test all(isnan, r.uncertainty_u)
        @test all(isnan, r.uncertainty_v)
    end

    @testset "Float32 pipeline" begin
        rng = MersenneTwister(1703)
        imgA, imgB = noisy_pair(rng, 0.05)
        r64 = run_piv(imgA, imgB, passes)
        r32 = run_piv(Float32.(imgA), Float32.(imgB), passes)
        @test r32 isa PIVResult{Float32}
        @test eltype(r32.uncertainty_u) == Float32
        m32 = median(filter(isfinite, r32.uncertainty_u))
        m64 = median(filter(isfinite, r64.uncertainty_u))
        @test m32 > 0
        @test m32 ≈ m64 rtol = 0.15
    end

    @testset "Masked windows stay NaN" begin
        rng = MersenneTwister(1704)
        imgA, imgB = noisy_pair(rng, 0.05)
        mask = falses(image_size)
        mask[1:48, 1:48] .= true
        r = run_piv(imgA, imgB, passes; mask)
        @test any(r.mask)
        @test all(isnan, r.uncertainty_u[r.mask])
        @test all(isfinite, r.uncertainty_u[.!r.mask])
    end

    @testset "Ensemble pooling shrinks σ" begin
        rng = MersenneTwister(1705)
        pairs = [noisy_pair(rng, 0.15) for _ in 1:6]
        params = PIVParameters(window_size = 32, overlap = 16, padding = true,
                               apodization = :gauss, uncertainty = true)
        single = run_piv(pairs[1]..., [params, params])
        ens = run_piv_ensemble(pairs, [params, params]; progress = false)
        m_single = median(filter(isfinite, single.uncertainty_u))
        m_ens = median(filter(isfinite, ens.uncertainty_u))
        @test m_ens > 0
        # Pooling 6 pairs should shrink σ by ≈ √6; assert a loose fraction.
        @test m_ens < 0.7 * m_single

        ens_off = run_piv_ensemble(pairs[1:2], PIVParameters(window_size = 32, overlap = 16);
                                   progress = false)
        @test all(isnan, ens_off.uncertainty_u)
    end

    @testset "JLD2 round-trip" begin
        rng = MersenneTwister(1706)
        imgA, imgB = noisy_pair(rng, 0.05)
        r = run_piv(imgA, imgB, passes)
        mktempdir() do dir
            path = joinpath(dir, "unc.jld2")
            save_results(path, r)
            loaded = load_results(path)[1]
            @test isequal(loaded.uncertainty_u, r.uncertainty_u)
            @test isequal(loaded.uncertainty_v, r.uncertainty_v)
            @test loaded.parameters.uncertainty
        end
    end
end
