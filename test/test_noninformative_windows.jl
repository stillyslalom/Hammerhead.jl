using Hammerhead
using Test
using Statistics

function noninformative_ka_analysis(R, finder)
    T = eltype(R)
    nr, nc = size(R)
    K = 3
    Rt = reshape(copy(R), nr, nc, 1)
    vals = fill(T(99), K, 1)
    locs = fill(Int32(4), 2, K, 1)
    out = fill(T(NaN), 5 + 2 * (K - 1), 1)
    dev = Hammerhead.CPU()
    Hammerhead._ka_analyze!(dev, Hammerhead._KA_TPW)(out, vals, locs,
        Rt, finder === :regionalmax, false, K, nr, nc,
        Val(Hammerhead._KA_TPW); ndrange = Hammerhead._KA_TPW)
    Hammerhead.KernelAbstractions.synchronize(dev)
    return (; du = out[1], dv = out[2], ratio = out[3], moment = out[4],
            found = round(Int, out[5]))
end

function noninformative_texture(::Type{T}; background = zero(T)) where {T}
    A = fill(T(background), 64, 64)
    B = copy(A)
    for j in 6:7:62, i in 5:7:61
        # Deterministic asymmetric positions avoid a periodic tie plane.
        x = j + 0.17 * mod(i, 3)
        y = i + 0.23 * mod(j, 4)
        Hammerhead.SyntheticData.generate_gaussian_particle!(A, (x, y), 3.0)
        Hammerhead.SyntheticData.generate_gaussian_particle!(B, (x + 1.25, y - 0.75), 3.0)
    end
    return A, B
end

@testset "Non-informative correlation measurements" begin
    @testset "Raw planes and stale scratch" begin
        for T in (Float32, Float64), finder in (:regionalmax, :exclusion)
            p = PIVParameters(window_size = 16, overlap = 8, peak_finder = finder)
            badplanes = [zeros(T, 8, 8), ones(T, 8, 8), fill(T(-1), 8, 8),
                         fill(T(NaN), 8, 8), fill(T(Inf), 8, 8)]
            mixed = zeros(T, 8, 8); mixed[4, 4] = 1; mixed[1, 1] = T(NaN)
            push!(badplanes, mixed)
            nonpositive = fill(T(-2), 8, 8); nonpositive[4, 4] = T(-1)
            push!(badplanes, nonpositive)
            for R in badplanes
                vals = fill(T(99), 3); locs = fill((4, 4), 3)
                cpu = Hammerhead.analyze_plane!(vals, locs, R, p)
                ka = noninformative_ka_analysis(R, finder)
                @test cpu.found == ka.found == 0
                @test all(isnan, (cpu.du, cpu.dv, cpu.ratio, cpu.moment))
                @test all(isnan, (ka.du, ka.dv, ka.ratio, ka.moment))
                raw = Hammerhead.locate_displacement(R, :gauss3)
                @test raw.peakloc == (0, 0) && isnan(raw.du) && isnan(raw.dv)
                @test vals == fill(T(99), 3) && locs == fill((4, 4), 3)
            end
            # A weak but distinct singleton peak remains a measurement. Its
            # undefined moment (legacy diagnostic floor) does not invalidate it.
            R = zeros(T, 8, 8); R[5, 6] = T(1e-15)
            cpu = Hammerhead.analyze_plane!(zeros(T, 3), fill((1, 1), 3), R, p)
            ka = noninformative_ka_analysis(R, finder)
            @test cpu.found == ka.found == 1
            @test cpu.du == ka.du == 1 && cpu.dv == ka.dv == 0
            @test cpu.ratio == ka.ratio == Inf
        end
        @test length(find_peaks(ones(8, 8), 3)) == 1
        @test length(find_peaks(zeros(8, 8), 3)) == 1
    end

    @testset "Exact constant centering and missing frames" begin
        for T in (Float32, Float64), backend in (:cpu, :ka)
            A, B = noninformative_texture(T)
            for (method, padding, apod) in ((:cross, true, :gauss), (:phase, false, :none))
                p = PIVParameters(window_size = 16, overlap = 8,
                    correlation_method = method, padding = padding, apodization = apod,
                    uod_enable = false, replace_outliers = false, n_peaks = 1)
                for (X, Y) in ((zeros(T, 64, 64), zeros(T, 64, 64)),
                               (fill(T(0.1), 64, 64), fill(T(0.1), 64, 64)),
                               (A, fill(T(0.1), 64, 64)))
                    r = run_piv(X, Y, p; backend, threaded = false)
                    @test all(r.outliers) && !any(r.mask)
                    @test all(isnan, r.u) && all(isnan, r.v)
                    @test all(isnan, r.peak_ratio)
                end
            end
            # A nonconstant masked pixel must not affect valid-pixel centering.
            C = fill(T(0.1), 64, 64); mask = falses(64, 64)
            C[1:8:end, 1:8:end] .= T(100)
            mask[1:8:end, 1:8:end] .= true
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, uod_enable = false, replace_outliers = false)
            r = run_piv(C, C, p; mask, backend, threaded = false)
            @test all(r.outliers) && !any(r.mask)
            @test all(isnan, r.u)
            fully_masked = run_piv(C, C, p; mask = trues(64, 64), backend, threaded = false)
            @test all(fully_masked.mask) && !any(fully_masked.outliers)
            @test all(isnan, fully_masked.u)

            # Center frame A and a larger frame-B search area independently.
            if backend === :cpu
                pwide = PIVParameters(window_size = 16, search_area_size = 32,
                    overlap = 8, padding = true, apodization = :gauss,
                    uod_enable = false, replace_outliers = false)
                wide = run_piv(fill(T(0.1), 64, 64), B, pwide; threaded = false)
                @test all(wide.outliers) && all(isnan, wide.u)
            end
        end
    end

    @testset "Texture and scale independence" begin
        for T in (Float32, Float64)
            A, B = noninformative_texture(T; background = T(0.1))
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, replace_outliers = false)
            cpu = run_piv(A, B, p; threaded = false)
            for backend in (:cpu, :ka)
                r = run_piv(A, B, p; backend, threaded = false)
                small = run_piv(T(1e-6) .* A, T(1e-6) .* B, p; backend, threaded = false)
                valid = .!r.outliers .& .!small.outliers
                @test count(valid) > length(valid) ÷ 2
                @test abs(median(r.u[valid]) - 1.25) < 0.2
                @test abs(median(r.v[valid]) + 0.75) < 0.2
                @test maximum(abs.(small.u[valid] .- r.u[valid])) < 2e-3
                @test maximum(abs.(r.u[valid] .- cpu.u[valid])) < 2e-3
            end
            # An image may contain both exactly blank and textured regions;
            # replacement retains flags even when it fills missing vectors.
            A[:, 1:32] .= T(0.1); B[:, 1:32] .= T(0.1)
            for backend in (:cpu, :ka)
                r = run_piv(A, B, p; backend, threaded = false)
                blank = r.x .<= 24
                @test all(r.outliers[:, blank]) && all(isnan, r.u[:, blank])
                @test any(.!r.outliers[:, r.x .> 40])
                filled = run_piv(A, B, PIVParameters(window_size = 16, overlap = 8,
                    padding = true, apodization = :gauss); backend, threaded = false)
                @test all(filled.outliers[:, blank]) && all(isfinite, filled.u[:, blank])
            end
        end
    end

    @testset "Multipass, iteration, ensemble, and predictor conditioning" begin
        for T in (Float32, Float64), backend in (:cpu, :ka)
            A = fill(T(0.1), 64, 64)
            passes = multipass_parameters([32, 16]; padding = true, apodization = :gauss,
                uod_enable = false, replace_outliers = false)
            for r in (run_piv(A, A, passes; backend, threaded = false),
                      run_piv(A, A, PIVParameters(window_size = 16, overlap = 8,
                          max_iterations = 3, convergence_tol = 0, padding = true,
                          apodization = :gauss, uod_enable = false,
                          replace_outliers = false); backend, threaded = false),
                      run_piv_ensemble([(A, A), (A, A)], passes; backend, threaded = false))
                @test all(r.outliers) && all(isnan, r.u) && all(isnan, r.v)
                @test Hammerhead.build_predictor(r, true) === nothing
            end
            # Empty samples contribute zero planes rather than poisoning an
            # otherwise informative ensemble measurement.
            X, Y = noninformative_texture(T)
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, replace_outliers = false)
            ref = run_piv(X, Y, p; backend, threaded = false)
            pooled = run_piv_ensemble([(A, A), (X, Y)], p; backend, threaded = false)
            valid = .!ref.outliers .& .!pooled.outliers
            @test count(valid) > length(valid) ÷ 2
            @test maximum(abs.(pooled.u[valid] .- ref.u[valid])) < 2e-3
        end
        p = PIVParameters(window_size = 16, overlap = 8, replace_outliers = false)
        shape = (5, 5)
        r = PIVResult(collect(1.0:5), collect(1.0:5), fill(2.0, shape), fill(-1.0, shape),
            fill(Inf, shape), zeros(shape), fill(NaN, shape), fill(NaN, shape),
            falses(shape), falses(shape), p)
        r.u[3, 3] = NaN; r.v[3, 3] = Inf
        r.u[1, 1] = 100
        apply_validator!(r, UniversalOutlierValidator(2.0; neighborhood_size = 1))
        @test r.outliers[3, 3] && r.outliers[1, 1]
        @test count(r.outliers) == 2 # nonfinite donor did not hide neighboring validation
        masked = deepcopy(r); masked.outliers .= false; masked.mask[3, 3] = true
        apply_validator!(masked, UniversalOutlierValidator(2.0; neighborhood_size = 1))
        @test !masked.outliers[3, 3] && masked.outliers[1, 1]
        pred = Hammerhead.build_predictor(r, false)
        @test all(isfinite, pred.u) && all(isfinite, pred.v)
        @test pred.u[3, 3] == 2 && pred.v[3, 3] == -1
        @test isnan(r.u[3, 3]) && isinf(r.v[3, 3]) # measurements stay untouched
        # A valid singleton peak's Inf ratio is accepted at the default cutoff.
        clean = deepcopy(r); clean.u .= 0; clean.v .= 0; clean.outliers .= false
        Hammerhead.validate_and_replace!(clean, p, false)
        @test !any(clean.outliers)
    end

    @testset "Original stencil rejects deformation texture" begin
        # Warped intensities retain spline roundoff; original-stencil evidence
        # prevents it becoming a measurement in the blank interior.
        for T in (Float32, Float64), backend in (:cpu, :ka)
            A, B = noninformative_texture(T; background = T(0.1))
            A[:, 1:32] .= T(0.1); B[:, 1:32] .= T(0.1)
            passes = multipass_parameters([32, 16]; padding = true,
                apodization = :gauss, replace_outliers = false)
            coarse = run_piv(A, B, passes[1]; backend, threaded = false)
            @test all(coarse.outliers[:, 1])
            pred = Hammerhead.build_predictor(coarse, true)
            @test pred !== nothing && maximum(abs, pred.u) > T(0.1)
            itpA = Hammerhead.image_interpolant(A, T)
            itpB = Hammerhead.image_interpolant(B, T)
            wa, _, _, _ = Hammerhead.apply_predictor(Hammerhead._resolve_backend(backend),
                A, B, itpA, itpB, pred, coarse.x, coarse.y, T; threaded = false)
            # x = 16.5, y = 24.5: all resampling support remains away from
            # the image boundary and the moving texture beginning at column 33.
            @test all(==(T(0.1)), A[17:32, 9:24])
            @test extrema(wa[17:32, 9:24])[1] < extrema(wa[17:32, 9:24])[2]
            final = run_piv(A, B, passes; backend, threaded = false)
            rows = (final.y .>= 24) .& (final.y .<= 41)
            col = final.x .== T(16.5)
            @test !any(final.mask[rows, col])
            @test all(final.outliers[rows, col])
            @test all(isnan, final.u[rows, col]) && all(isnan, final.peak_ratio[rows, col])
        end
    end

    @testset "Unavailable uncertainties and masked flags" begin
        for (T, backend, iterations, uqbackend) in
            ((Float32, :cpu, 1, :same), (Float64, :cpu, 2, :same),
             (Float32, :ka, 1, :same), (Float64, :ka, 2, :same),
             (Float64, :ka, 2, :cpu))
            A = fill(T(0.1), 64, 64)
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, uncertainty = true, max_iterations = iterations)
            r = run_piv(A, A, p; backend, uncertainty_backend = uqbackend, threaded = false)
            @test all(r.outliers) && all(isnan, r.uncertainty_u) && all(isnan, r.uncertainty_v)
        end
        for backend in (:cpu, :ka)
            A, B = noninformative_texture(Float64)
            Z = fill(0.1, 64, 64)
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, uncertainty = true, replace_outliers = false)
            empty = run_piv_ensemble([(Z, Z), (Z, Z)], p; backend, threaded = false)
            @test all(empty.outliers) && all(isnan, empty.uncertainty_u)
            ref = run_piv(A, B, p; backend, threaded = false)
            pooled = run_piv_ensemble([(Z, Z), (A, B)], p; backend, threaded = false)
            finite = isfinite.(ref.uncertainty_u) .& isfinite.(pooled.uncertainty_u)
            @test any(finite)
            @test maximum(abs.(ref.uncertainty_u[finite] .- pooled.uncertainty_u[finite])) < 1e-12
        end
    end
end
