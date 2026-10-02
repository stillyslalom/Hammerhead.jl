using Hammerhead
using Test
using Random

@testset "Non-collecting batch to statistics and lazy replay" begin
    image = rand(MersenneTwister(607), 48, 48)
    frames = [image, circshift(image, (1, 2)), image, circshift(image, (2, 3))]
    source = FrameSource(4, i -> frames[i]; timestamps = [0.0, 0.5, 2.0, 3.0])
    pairs = image_pairs(source; mode = :paired)
    params = PIVParameters(window_size = 16, overlap = 8, padding = true)
    scale = PhysicalScale(0.2, 1.0, "mm", "s")
    mask = falses(48, 48)
    mask[1:20, 1:20] .= true

    mktempdir() do dir
        path = joinpath(dir, "stream.jld2")
        online = FieldStatisticsAccumulator()
        @test run_piv_sequence(pairs, params; scale, mask, output = path,
            progress = false, collect_results = false,
            on_result = (_, r) -> update_statistics!(online, physical(r))) === nothing

        saved = load_results(path; lazy = true)
        @test saved isa ResultFile
        @test length(saved) == 2
        @test saved[1].scale.dt == 0.5
        @test saved[2].scale.dt == 1.0
        replayed = FieldStatisticsAccumulator()
        for r in saved
            update_statistics!(replayed, physical(r))
        end
        actual = field_statistics(online)
        @test isequal(actual, field_statistics(replayed))
        @test isequal(actual, field_statistics(physical.(load_results(path))))
        @test any(==(0), actual.count)
        @test any(==(2), actual.count)
        @test actual.x == physical(saved[1]).x
    end
end
