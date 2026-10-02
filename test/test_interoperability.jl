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

    @testset "effort presets size their windows to the ROI" begin
        # A singleton predictor axis is constant, while the other keeps its
        # linear interpolation and flat extrapolation.
        for T in (Float32, Float64)
            row = Hammerhead.predictor_interpolant(T[5], T[3, 7], T[2 6])
            col = Hammerhead.predictor_interpolant(T[3, 7], T[5], T[2; 6;;])
            point = Hammerhead.predictor_interpolant(T[5], T[5], fill(T(3), 1, 1))
            @test row(-20, 5) == row(5, 5) == row(20, 5) == T(4)
            @test row(5, -20) == T(2) && row(5, 20) == T(6)
            @test col(5, -20) == col(5, 5) == col(5, 20) == T(4)
            @test col(-20, 5) == T(2) && col(20, 5) == T(6)
            @test all(point(y, x) == T(3) for y in (-20, 5, 20), x in (-20, 5, 20))
        end
        a = rand(MersenneTwister(418), 96, 96)
        b = circshift(a, (1, 2))
        rr = ROI(9:56, 17:56)
        mask = falses(size(a)); mask[rr.rows, 17:24] .= true
        for effort in (:low, :medium, :high)
            expected = run_piv(a[rr.rows, rr.cols], b[rr.rows, rr.cols];
                effort, mask = mask[rr.rows, rr.cols])
            actual = run_piv(a, b; effort, roi = (rr.rows, rr.cols), mask)
            @test actual.x == expected.x .+ first(rr.cols) .- 1
            @test actual.y == expected.y .+ first(rr.rows) .- 1
            @test isequal(actual.u, expected.u)
            @test isequal(actual.v, expected.v)
            @test actual.mask == expected.mask
            @test actual.parameters.window_size == expected.parameters.window_size
        end
        ka = run_piv(a, b; effort = :medium, roi = rr, backend = :ka)
        cpu = run_piv(a, b; effort = :medium, roi = rr)
        @test ka.u ≈ cpu.u atol = 1e-10
        @test ka.v ≈ cpu.v atol = 1e-10
        @test_throws BoundsError run_piv(a, b; effort = :medium, roi = ROI(1:97, 1:96))
        # Explicit schedules still reject a window exceeding the selected region.
        @test_throws ArgumentError run_piv(a, b, PIVParameters(window_size = 64);
                                          roi = rr)
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
            function vtk_units(path)
                lines = readlines(path)
                start = findfirst(==("FIELD FieldData 2"), lines)
                @test start !== nothing
                @test startswith(lines[start + 5], "POINT_DATA ")
                units = Dict{String,String}()
                for k in (start + 1, start + 3)
                    name, ncomp, ntuple, dtype = split(lines[k])
                    @test ncomp == "1" && dtype == "unsigned_char"
                    bytes = parse.(UInt8, split(lines[k + 1]))
                    @test length(bytes) == parse(Int, ntuple)
                    units[name] = String(bytes)
                end
                units
            end
            @test vtk_units(vtk) == Dict("coordinate_unit" => "px",
                                         "component_unit" => "px/frame")
            scaled = with_scale(r, PhysicalScale(pixel_size=0.1, dt=0.25,
                length_unit="µm", time_unit="s"))
            scaled_vtk = export_vtk(joinpath(dir,"scaled.vtk"), scaled)
            @test vtk_units(scaled_vtk) == Dict("coordinate_unit" => "µm",
                                                "component_unit" => "µm/s")

            tr = TrackingResult([Trajectory(1,[1.0,2.0],[3.0,4.0])], 2, PTVParameters())
            path = save_results(joinpath(dir,"tracks.jld2"), tr)
            loaded = load_results(path)[1]
            @test loaded isa TrackingResult
            @test loaded.trajectories[1].x == [1.0,2.0]
        end
    end
end
