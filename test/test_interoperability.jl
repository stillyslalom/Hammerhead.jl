using Test
using Hammerhead
using JLD2
using Random
using FileIO: save
using ImageCore: Gray, N0f8

@testset "Interoperability, frame sources, and ROI" begin
    @testset "lazy sources and flexible pairing" begin
        calls = Int[]
        source = FrameSource(5, i -> (push!(calls, i); fill(Float64(i), 32, 32));
                             timestamps = [0.0, 0.1, 0.25, 0.5, 0.9])
        pairs = image_pairs(source; mode=:chained, stride=2, offset=1, deltas=(1, 2))
        @test isempty(calls)                    # pair construction stays lazy
        @test [(p.first.index, p.second.index) for p in pairs] == [(2,3),(4,5),(2,4)]
        @test pairs[1].dt == 0.15
        @test Hammerhead.load_frame(pairs[1][1], Float64) == fill(2.0, 32, 32)
        @test calls == [2]

        files = collect('a':'h')
        @test image_pairs(files; mode=:chained, stride=2, offset=1, deltas=2) ==
              [('b','d'), ('d','f'), ('f','h')]

        mktempdir() do dir
            path = joinpath(dir, "stack.tif")
            stack = cat(fill(Gray{N0f8}(0.25), 8, 9),
                        fill(Gray{N0f8}(0.75), 8, 9); dims=3)
            save(path, stack)
            tif = TIFFStack(path; image_type=Float32)
            @test length(tif) == 2
            @test tif[2] isa Matrix{Float32}
            @test tif[2][1,1] ≈ 0.75 atol=1/255
        end
    end

    @testset "dynamic pair masks and ROI coordinates" begin
        imgA = rand(MersenneTwister(41), 64,64); imgB = copy(imgA)
        p = PIVParameters(window_size=16, overlap=8)
        left = falses(64,64); left[:,1:20] .= true
        right = falses(64,64); right[:,45:end] .= true
        seq = run_piv_sequence([(imgA,imgB)], p;
            mask=(i,a,b)->(left,right), roi=ROI(9:56,9:56), progress=false)
        @test first(seq[1].x) >= 9 && first(seq[1].y) >= 9
        @test any(seq[1].mask)
        direct = run_piv(imgA,imgB,p; roi=(9:56,9:56))
        @test direct.x == seq[1].x && direct.y == seq[1].y

        source = FrameSource(2, i -> i == 1 ? imgA : imgB; timestamps=[1.0,1.25])
        fmasks = [left, right]
        timed = run_piv_sequence(image_pairs(source; mode=:chained), p;
            mask=fmasks, scale=PhysicalScale(pixel_size=0.1, dt=1.0,
                length_unit="mm", time_unit="s"), progress=false)
        @test timed[1].scale.dt == 0.25
        @test any(timed[1].mask)
    end

    @testset "table, VTK, and tracking persistence" begin
        imgA = rand(MersenneTwister(42),48,48); imgB = copy(imgA)
        r = run_piv(imgA,imgB,PIVParameters(window_size=16,overlap=8))
        mktempdir() do dir
            csv = export_table(joinpath(dir,"field.csv"), r; frame_id="7", source_a="a.tif")
            lines = readlines(csv)
            @test split(lines[1], ',') == collect(TABLE_COLUMNS)
            @test occursin(TABLE_SCHEMA_VERSION, lines[2])
            vtk = export_vtk(joinpath(dir,"field.vtk"), r)
            text = read(vtk,String)
            @test occursin("DATASET STRUCTURED_GRID", text)
            @test occursin("VECTORS velocity", text)

            tr = TrackingResult([Trajectory(1,[1.0,2.0],[3.0,4.0])], 2, PTVParameters())
            path = save_results(joinpath(dir,"tracks.jld2"), tr)
            loaded = load_results(path)[1]
            @test loaded isa TrackingResult
            @test loaded.trajectories[1].x == [1.0,2.0]
        end
    end
end
