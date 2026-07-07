using Hammerhead
using Test
using FileIO: save
using ImageCore: Gray, N0f8, N0f16
using JLD2: jldopen
using Random
using Statistics

@testset "File I/O and batch processing" begin
    @testset "load_image" begin
        mktempdir() do dir
            rng = MersenneTwister(5)
            data = rand(rng, 32, 24)

            path16 = joinpath(dir, "img16.tif")
            save(path16, Gray{N0f16}.(data))
            img16 = load_image(path16)
            @test img16 isa Matrix{Float64}
            @test size(img16) == (32, 24)
            @test maximum(abs.(img16 .- data)) <= 1 / 65535  # 16-bit quantization

            path8 = joinpath(dir, "img8.png")
            save(path8, Gray{N0f8}.(data))
            img8 = load_image(path8)
            @test maximum(abs.(img8 .- data)) <= 1 / 255  # 8-bit quantization

            img32 = load_image(Float32, path16)
            @test img32 isa Matrix{Float32}
            @test img32 ≈ img16

            @test_throws ArgumentError load_image(joinpath(dir, "missing.tif"))
        end
    end

    @testset "image_pairs" begin
        files = ["a", "b", "c", "d"]
        @test image_pairs(files) == [("a", "b"), ("c", "d")]
        @test image_pairs(files; mode = :chained) == [("a", "b"), ("b", "c"), ("c", "d")]
        @test_throws ArgumentError image_pairs(files[1:3])           # odd count
        @test_throws ArgumentError image_pairs(["a"]; mode = :chained)
        @test_throws ArgumentError image_pairs(files; mode = :fancy)
    end

    @testset "save_results / load_results roundtrip" begin
        imgA, imgB = particle_pair((64, 64), [(32.0, 32.0), (16.0, 48.0)], 2.0, 1.0)
        params = PIVParameters(window_size = 32, overlap = 16)
        result = run_piv(imgA, imgB, params)
        mktempdir() do dir
            path = joinpath(dir, "result.jld2")
            @test save_results(path, result) == path
            loaded = load_results(path)
            @test length(loaded) == 1
            @test loaded[1].u == result.u
            @test loaded[1].v == result.v
            @test loaded[1].x == result.x && loaded[1].y == result.y
            @test loaded[1].peak_ratio == result.peak_ratio
            @test loaded[1].outliers == result.outliers
            @test loaded[1].parameters.window_size == (32, 32)

            save_results(path, [result, result])
            @test length(load_results(path)) == 2

            # Float32 results roundtrip with their precision intact.
            result32 = run_piv(Float32.(imgA), Float32.(imgB), params)
            save_results(path, result32)
            @test load_results(path)[1] isa PIVResult{Float32}

            other = joinpath(dir, "other.jld2")
            jldopen(f -> (f["foo"] = 1), other, "w")
            @test_throws ArgumentError load_results(other)
        end
    end

    @testset "run_piv_sequence" begin
        rng = MersenneTwister(9)
        positions = [(rand(rng) * 148 - 10, rand(rng) * 148 - 10) for _ in 1:250]
        imgA, imgB = particle_pair((128, 128), positions, 2.0, 3.0)
        params = PIVParameters(window_size = 32, overlap = 16)

        # In-memory matrix pairs match a direct run_piv call.
        direct = run_piv(imgA, imgB, params)
        res = run_piv_sequence([(imgA, imgB), (imgA, imgB)], params; progress = false)
        @test length(res) == 2
        @test res[1].u == direct.u && res[1].v == direct.v
        @test res[2].u == res[1].u

        # Preprocessing hook: correlation is intensity-scale invariant.
        res_pp = run_piv_sequence([(imgA, imgB)], params;
                                  preprocess = img -> 2 .* img, progress = false)
        @test res_pp[1].u ≈ direct.u atol = 1e-6

        # Progress callback: called once per pair, in order.
        calls = Tuple{Int,Int}[]
        run_piv_sequence([(imgA, imgB), (imgA, imgB)], params;
                         progress = (i, n) -> push!(calls, (i, n)))
        @test calls == [(1, 2), (2, 2)]

        # Throwing from the callback aborts the batch but keeps the pairs
        # finished before the abort in the incremental output.
        mktempdir() do dir
            out = joinpath(dir, "aborted.jld2")
            struct_free_abort = ErrorException("stop")
            @test_throws ErrorException run_piv_sequence(
                [(imgA, imgB), (imgA, imgB), (imgA, imgB)], params;
                output = out, progress = (i, n) -> i == 2 && throw(struct_free_abort))
            kept = load_results(out)
            @test length(kept) == 2
            @test kept[1].u == direct.u
        end

        mktempdir() do dir
            # File pairs with incrementally persisted output.
            frames = [imgA, imgB, imgA, imgB]
            files = [joinpath(dir, "frame$i.tif") for i in 1:4]
            for (f, img) in zip(files, frames)
                save(f, Gray{N0f16}.(clamp.(img, 0, 1)))
            end
            out = joinpath(dir, "results.jld2")
            res_f = run_piv_sequence(image_pairs(files), params;
                                     output = out, progress = false)
            @test length(res_f) == 2
            @test median(res_f[1].u) ≈ 3.0 atol = 0.25
            @test median(res_f[1].v) ≈ 2.0 atol = 0.25
            loaded = load_results(out)
            @test length(loaded) == 2
            @test loaded[1].u == res_f[1].u
            @test loaded[2].v == res_f[2].v
            jldopen(out, "r") do f
                @test f["sources/000001"] == [files[1], files[2]]
                @test f["sources/000002"] == [files[3], files[4]]
            end

            # image_type = Float32 runs the whole pipeline in single precision.
            res_32 = run_piv_sequence(image_pairs(files), params;
                                      image_type = Float32, progress = false)
            @test res_32[1] isa PIVResult{Float32}
        end

        @test_throws ArgumentError run_piv_sequence([], params)
        # Failures identify the offending pair and propagate.
        @test_throws ArgumentError run_piv_sequence([(imgA, "missing.tif")], params;
                                                    progress = false)
        @test_throws ArgumentError run_piv_sequence([(imgA, :nope)], params;
                                                    progress = false)
    end
end
