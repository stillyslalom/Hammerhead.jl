using Hammerhead
using Test

function original_test_predictor(::Type{T}, n; u = 1.25, v = -0.75) where {T}
    return (x = T[1, n], y = T[1, n], u = fill(T(u), 2, 2), v = fill(T(v), 2, 2))
end

function original_test_contrast(map, predictor, rs, cs, wr, wc, mask = nothing, sign = -1)
    itpu = Hammerhead.predictor_interpolant(predictor.y, predictor.x, predictor.u)
    itpv = Hammerhead.predictor_interpolant(predictor.y, predictor.x, predictor.v)
    return Hammerhead._original_window_contrast(map, rs, cs, wr, wc, mask,
                                                predictor, itpu, itpv, sign)
end

Base.@noinline function original_support_release_probe()
    A, B = noninformative_texture(Float64; background = 0.1)
    A[:, 1:32] .= 0.1; B[:, 1:32] .= 0.1
    ws = piv_workspace()
    run_piv(A, B, multipass_parameters([32, 16]; padding = true, apodization = :gauss);
            workspace = ws, threaded = false)
    return WeakRef(A), WeakRef(B), ws
end

@testset "Original source stencil information" begin
    @testset "Compact offsets, precision, masks, and lazy lifetime" begin
        for T in (Float32, Float64)
            A = fill(T(0.1), 16, 16)
            map = Hammerhead._original_stencil_map(A, nothing, T, nothing, :A)
            @test eltype(map.codes) === UInt8 && sizeof(map.codes) == 15^2
            code, value = Hammerhead._original_stencil_value(map, T(5.5), T(5.5))
            @test code == 1 && value === T(0.1)
            @test Hammerhead._original_stencil_value(map, T(-1), T(5))[1] == 0
            @test Hammerhead._original_stencil_value(map, T(16), T(16))[2] == T(0.1)
            # Every possible first eligible offset, including clipped corners.
            for offset in 1:16
                mask = trues(16, 16)
                mask[4 + (offset - 1) % 4, 4 + (offset - 1) ÷ 4] = false
                masked = Hammerhead._original_stencil_map(A, mask, T, nothing, :A)
                @test Hammerhead._original_stencil_value(masked, T(5.5), T(5.5)) == (UInt8(offset), T(0.1))
            end
            masked = Hammerhead._original_stencil_map(A, trues(16, 16), T, nothing, :A)
            @test Hammerhead._original_stencil_value(masked, T(5.5), T(5.5))[1] == 0
            varying = T[isodd(i + j) ? 1 : 2 for i in 1:16, j in 1:16]
            dense = Hammerhead._original_stencil_map(varying, nothing, T, nothing, :A)
            @test dense.codes === nothing # no full image allocation retained
            @test Hammerhead._original_stencil_value(dense, T(5), T(5))[1] == 17
            weak = copy(A); weak[6, 6] = nextfloat(weak[6, 6])
            weakmap = Hammerhead._original_stencil_map(weak, nothing, T, nothing, :A)
            @test Hammerhead._original_stencil_value(weakmap, T(5), T(5))[1] == 17

            ws = piv_workspace()
            p = PIVParameters(window_size = 8, overlap = 4)
            grid = Hammerhead.pass_grid(T, size(A), p, nothing, 0.5)
            context = Hammerhead._original_support_context(A, A, nothing, T, ws)
            @test Hammerhead._original_source_gate(context, nothing, grid, p, nothing) === nothing
            @test !context.prepared && isempty(ws.engines)
            pred = original_test_predictor(T, 16)
            gate = Hammerhead._original_source_gate(context, pred, grid, p, nothing)
            @test context.prepared && !any(gate)
            @test length(ws.engines) == 2
            @test all(v -> v[1] isa Matrix{UInt8}, values(ws.engines)) # no input payload in pool
            oldcodes = context.maps[1].codes
            # Fully rewrite cached metadata when source/mask changes.
            B = copy(A); B[6, 6] = T(0.2)
            next = Hammerhead._original_support_context(B, B, nothing, T, ws)
            fresh = Hammerhead._original_source_gate(next, pred, grid, p, nothing)
            @test any(fresh) && next.maps[1].codes === oldcodes
            @test fresh == Hammerhead._original_source_gate(
                Hammerhead._original_support_context(B, B, nothing, T, nothing), pred, grid, p, nothing)
            corrupt = copy(A); corrupt[1] = T(NaN)
            invalid = Hammerhead._original_support_context(corrupt, A, nothing, T, nothing)
            @test Hammerhead._original_source_gate(invalid, pred, grid, p, nothing) === nothing
            @test invalid.prepared && invalid.maps === nothing
        end
        A = fill(0.1, 16, 16); A[6, 6] = nextfloat(A[6, 6])
        converted = Hammerhead._original_stencil_map(A, nothing, Float32, nothing, :A)
        @test Hammerhead._original_stencil_value(converted, 5, 5)[1] != 17
        wa, wb, ws = original_support_release_probe()
        GC.gc(true); GC.gc(true)
        @test wa.value === nothing && wb.value === nothing
        @test haskey(ws.engines, (:original_stencil, :A))
    end

    @testset "Different constant levels and boundary evidence" begin
        A = fill(0.1, 16, 16); A[:, 9:16] .= 0.2
        map = Hammerhead._original_stencil_map(A, nothing, Float64, nothing, :A)
        pred = (x = [6.0, 7.0], y = [6.0, 7.0],
                u = [-5.0 11.0; -5.0 11.0], v = zeros(2, 2))
        @test original_test_contrast(map, pred, 6, 6, 1, 2, nothing, 1)
        constant = fill(0.1, 16, 16)
        map = Hammerhead._original_stencil_map(constant, nothing, Float64, nothing, :A)
        shifted = original_test_predictor(Float64, 16; u = 6, v = 0)
        @test !original_test_contrast(map, shifted, 1, 1, 8, 8) # partial zero extrapolation
        outside = original_test_predictor(Float64, 16; u = 100, v = 0)
        @test !original_test_contrast(map, outside, 1, 1, 8, 8)
        # Originally excluded bright intensity cannot donate provenance.
        constant[6, 6] = 100; mask = falses(16, 16); mask[6, 6] = true
        map = Hammerhead._original_stencil_map(constant, mask, Float64, nothing, :A)
        @test !original_test_contrast(map, original_test_predictor(Float64, 16), 1, 1, 8, 8, mask)
    end

    @testset "Independent search footprint" begin
        A = fill(0.1, 64, 64); B = copy(A)
        A[16, 16] = 0.2; B[3, 3] = 0.2
        pred = original_test_predictor(Float64, 64; u = 0.25, v = 0.25)
        p = PIVParameters(window_size = 16, search_area_size = 32, overlap = 8)
        grid = Hammerhead.pass_grid(Float64, size(A), p, nothing, 0.5)
        context = Hammerhead._original_support_context(A, B, nothing, Float64, nothing)
        gate = Hammerhead._original_source_gate(context, pred, grid, p, nothing)
        @test gate[1, 1] # B search includes contrast outside its central window
        @test !original_test_contrast(context.maps[2], pred, 9, 9, 16, 16, nothing, 1)
    end

    @testset "Repeated passes and uncertainty" begin
        for T in (Float32, Float64), backend in (:cpu, :ka)
            A, B = noninformative_texture(T; background = T(0.1))
            A[:, 1:32] .= T(0.1); B[:, 1:32] .= T(0.1)
            p = multipass_parameters([32, 16]; padding = true, apodization = :gauss,
                replace_outliers = false, final = (; uncertainty = true, max_iterations = 2))
            r = run_piv(A, B, p; backend, threaded = false)
            interior = (r.y .>= 24) .& (r.y .<= 41)
            col = r.x .== T(16.5)
            @test all(r.outliers[interior, col]) && all(isnan, r.u[interior, col])
            @test all(isnan, r.uncertainty_u[interior, col]) && all(isnan, r.uncertainty_v[interior, col])
            threaded = run_piv(A, B, p; backend, threaded = true)
            @test isequal(r.u, threaded.u) && r.outliers == threaded.outliers
            fillpasses = multipass_parameters([32, 16]; padding = true,
                apodization = :gauss, final = (; uncertainty = true, max_iterations = 2))
            filled = run_piv(A, B, fillpasses; backend, threaded = false)
            @test all(filled.outliers[interior, col]) && all(isfinite, filled.u[interior, col])
            @test all(isnan, filled.peak_ratio[interior, col]) && all(isnan, filled.uncertainty_u[interior, col])
            if backend === :ka && T === Float64
                hybrid = run_piv(A, B, p; backend, uncertainty_backend = :cpu, threaded = false)
                @test all(hybrid.outliers[interior, col]) && all(isnan, hybrid.uncertainty_u[interior, col])
            end
        end
    end

    @testset "Mixed ensemble contribution and pooled uncertainty" begin
        for T in (Float32, Float64), backend in (:cpu, :ka)
            A, B = noninformative_texture(T)
            Z = fill(T(0.1), 64, 64)
            pred = original_test_predictor(T, 64)
            p = PIVParameters(window_size = 16, overlap = 8, padding = true,
                apodization = :gauss, replace_outliers = false, uncertainty = true)
            function analyze(pairs)
                Hammerhead.ensemble_pass(pairs, p, pred; threaded = false,
                    mask = nothing, mask_threshold = 0.5, preprocess = nothing,
                    image_type = T, force_replace = false,
                    meter = Hammerhead.Progress(length(pairs); enabled = false),
                    workspace = piv_workspace(; backend), backend = Hammerhead._resolve_backend(backend))
            end
            empty = analyze([(Z, Z), (Z, Z)])
            @test all(empty.outliers) && all(isnan, empty.u) && all(isnan, empty.uncertainty_u)
            ref = analyze([(A, B)])
            mixed = analyze([(Z, Z), (A, B)])
            @test isequal(ref.u, mixed.u) && ref.outliers == mixed.outliers
            @test isequal(ref.uncertainty_u, mixed.uncertainty_u)
            weak = analyze([(T(1e-6) .* A, T(1e-6) .* B)])
            valid = .!ref.outliers .& .!weak.outliers
            @test count(valid) > length(valid) ÷ 2
            @test maximum(abs.(ref.u[valid] .- weak.u[valid])) < 2e-3
        end
    end

    @testset "Skip marker and stale alternatives" begin
        T = Float64; dev = Hammerhead.CPU()
        A = [i + j for i in 1:8, j in 1:8]
        origins = [0 1; 1 1]
        meansA = fill(99.0, 2); meansB = copy(meansA)
        CA = fill(ComplexF64(99), 8, 8, 2); CB = copy(CA)
        mask = falses(8, 8)
        Hammerhead._ka_window_means!(dev)(meansA, meansB, A, A, origins, mask, true, 8, 8; ndrange = 2)
        Hammerhead._ka_gather!(dev)(CA, CB, A, A, origins, ones(8, 8), meansA, meansB, mask, true; ndrange = (8, 8, 2))
        Hammerhead.KernelAbstractions.synchronize(dev)
        @test meansA[1] == meansB[1] == 0
        @test all(iszero, CA[:, :, 1]) && all(iszero, CB[:, :, 1])
        @test any(!iszero, CA[:, :, 2])
        p = PIVParameters(window_size = 8, overlap = 4)
        for backend in (:cpu, :ka)
            engine = backend === :cpu ? Hammerhead._make_correlation_engine(p, T) :
                Hammerhead._make_ka_engine(p, T)
            u = fill(99.0, 1, 1); v = copy(u); ratio = copy(u); moment = copy(u)
            altu = fill(99.0, 1, 1, 2); altv = copy(altu)
            planes = backend === :cpu ? Matrix{Union{Nothing,Matrix{T}}}(undef, 1, 1) : nothing
            Hammerhead.process_windows!(u, v, ratio, moment, altu, altv, nothing, nothing,
                [(1, 1, 1, 1)], A, A, p, engine, nothing, planes; source_gate = falses(1, 1))
            @test all(isnan, u) && all(isnan, ratio)
            @test all(isnan, altu) && all(isnan, altv)
            planes === nothing || @test all(iszero, planes[1, 1])
        end
    end
end
