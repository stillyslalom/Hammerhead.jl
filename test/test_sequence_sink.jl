using Hammerhead
using Test
using Random

@testset "Non-collecting sequence drivers" begin
    rng = MersenneTwister(419)
    a = rand(rng, 64, 64)
    b = circshift(a, (1, 2))
    pairs = [(a, b), (a, b), (a, b)]
    params = PIVParameters(window_size = 16, overlap = 8, padding = true)
    scale = PhysicalScale(0.2, 0.5, "mm", "s")

    samefield(actual, expected) =
        isequal(actual.u, expected.u) && isequal(actual.v, expected.v) &&
        isequal(actual.outliers, expected.outliers) &&
        actual.scale.pixel_size == expected.scale.pixel_size &&
        actual.scale.dt == expected.scale.dt

    @testset "Planar persistence and callback order" begin
        expected = run_piv_sequence(pairs, params; scale, progress = false)
        mktempdir() do dir
            events = Tuple{Symbol,Int}[]
            path = joinpath(dir, "stream.jld2")
            returned = run_piv_sequence(pairs, params; scale,
                collect_results = false, output = path,
                on_result = (i, r) -> begin
                    @test samefield(r, expected[i])
                    push!(events, (:result, i))
                end,
                progress = (i, n) -> begin
                    @test n == 3
                    push!(events, (:progress, i))
                end)
            @test returned === nothing
            @test events == [(:result, 1), (:progress, 1), (:result, 2),
                             (:progress, 2), (:result, 3), (:progress, 3)]
            saved = load_results(path)
            @test length(saved) == length(expected)
            @test all(samefield.(saved, expected))

            # The effort overload consumes the option at the driver boundary.
            @test run_piv_sequence(pairs[1:1]; effort = :low,
                collect_results = false, progress = false,
                output = (i, pair) -> joinpath(dir, "pair-$i.jld2")) === nothing
            @test length(load_results(joinpath(dir, "pair-1.jld2"))) == 1

            # An on_result failure precedes persistence of that result.
            failed_path = joinpath(dir, "failed.jld2")
            @test_throws ErrorException run_piv_sequence(pairs, params;
                collect_results = false, output = failed_path, progress = false,
                on_result = (i, r) -> (i == 2 && error("consumer stopped")))
            @test length(load_results(failed_path)) == 1
        end
    end

    @testset "PTV per-pair sink" begin
        ptvparams = PTVParameters()
        expected = run_ptv_sequence(pairs, ptvparams;
            predictor = nothing, scale, progress = false)
        mktempdir() do dir
            seen = Int[]
            @test run_ptv_sequence(pairs, ptvparams; predictor = nothing, scale,
                collect_results = false, progress = false,
                output = (i, pair) -> joinpath(dir, "ptv-$i.jld2"),
                on_result = (i, r) -> begin
                    @test samefield(r, expected[i])
                    push!(seen, i)
                end) === nothing
            @test seen == [1, 2, 3]
            @test all(i -> samefield(only(load_results(joinpath(dir, "ptv-$i.jld2"))),
                                    expected[i]), 1:3)
        end
    end

    @testset "Stereo collection, effort, and cancellation" begin
        # Two cameras map z=0 identically but have distinct viewing directions.
        cams = [PinholeCamera([1.0 0.0 tilt 0.0;
                              0.0 1.0 0.0 0.0;
                              0.0 0.0 0.001 1.0]) for tilt in (-0.2, 0.2)]
        grid = DewarpGrid(x = 1.0:64.0, y = 1.0:64.0)
        dw1, dw2 = [ImageDewarper(cam, grid, (64, 64)) for cam in cams]
        acquisitions = [(a, b, a, b) for _ in 1:3]
        expected = run_piv_stereo_sequence(acquisitions, dw1, dw2, params;
            scale, progress = false)
        mktempdir() do dir
            path = joinpath(dir, "stereo.jld2")
            seen = Int[]
            @test run_piv_stereo_sequence(pairs, pairs, dw1, dw2, params;
                scale, collect_results = false, output = path, progress = false,
                on_result = (i, r) -> begin
                    @test samefield(r, expected[i])
                    @test isequal(r.w, expected[i].w)
                    push!(seen, i)
                end) === nothing
            @test seen == [1, 2, 3]
            @test all(samefield.(load_results(path), expected))

            cancelled_path = joinpath(dir, "canceled.jld2")
            completed = Ref(0)
            @test run_piv_stereo_sequence(acquisitions, dw1, dw2;
                effort = :low, collect_results = false, output = cancelled_path,
                progress = false, on_result = (i, r) -> (completed[] = i),
                cancel = () -> completed[] == 1) === nothing
            @test completed[] == 1
            @test length(load_results(cancelled_path)) == 1

            @test run_piv_stereo_sequence(pairs[1:1], pairs[1:1], dw1, dw2;
                effort = :low, collect_results = false, progress = false) === nothing
        end
    end

    @testset "Driver does not retain delivered results" begin
        references = WeakRef[]
        # Exercise the generic driver with isolated buffers: numerical workspaces
        # cannot obscure whether the sequence itself retains earlier outputs.
        process = function (imgA, imgB, i, pair, mask, scale)
            if i > 2
                GC.gc()
                @test references[i - 2].value === nothing
            end
            fill(UInt8(i), 1024)
        end
        @test Hammerhead._run_sequence(process, Vector{UInt8}, fill((a, b), 6);
            collect_results = false, progress = false,
            on_result = (i, r) -> push!(references, WeakRef(r))) === nothing
        GC.gc()
        @test all(w -> w.value === nothing, references)
    end
end
