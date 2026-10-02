using Test
using Hammerhead

# Independent CSV reader for round-trip assertions, including quoted UTF-8
# labels, embedded delimiters/newlines, and doubled quotes. No CSV dependency.
function tracking_export_rows(path)
    chars = collect(read(path, String))
    records = Vector{String}[]
    record = String[]
    field = IOBuffer()
    quoted = false
    i = 1
    while i <= length(chars)
        c = chars[i]
        if c == '"'
            if quoted && i < length(chars) && chars[i + 1] == '"'
                print(field, '"')
                i += 1
            else
                quoted = !quoted
            end
        elseif !quoted && c == ','
            push!(record, String(take!(field)))
        elseif !quoted && c == '\n'
            push!(record, String(take!(field)))
            push!(records, record)
            record = String[]
        elseif !quoted && c == '\r'
            # Accept CRLF as well as the writer's LF.
        else
            print(field, c)
        end
        i += 1
    end
    @test !quoted
    @test isempty(record) && position(field) == 0
    columns = first(records)
    @test columns == collect(TABLE_COLUMNS)
    @test length(unique(columns)) == length(columns)
    @test all(row -> length(row) == length(columns), records[2:end])
    [Dict(zip(columns, row)) for row in records[2:end]]
end

@testset "Tracking table export" begin
    params = PTVParameters()
    gapped = Trajectory{Float64}(2, [10.0, 14.0, 26.0], [20.0, 18.0, 12.0], [2, 3, 6])
    empty_track = Trajectory{Float64}(1, Float64[], Float64[], Int[])
    singleton = Trajectory{Float64}(4, [7.0], [9.0], [4])
    other = Trajectory(1, [1.0, 3.0], [5.0, 6.0])
    result = TrackingResult([gapped, empty_track, singleton, other], 6, params)

    mktempdir() do dir
        path = joinpath(dir, "tracks.csv")
        @test export_table(path, result) == path
        rows = tracking_export_rows(path)
        @test length(rows) == 6  # only observations, no rows for missing frames
        @test all(row -> row["schema_version"] == TABLE_SCHEMA_VERSION &&
                         row["result_type"] == "tracking", rows)
        @test parse.(Int, getindex.(rows, "point_id")) == collect(1:6)
        @test parse.(Int, getindex.(rows, "trajectory_id")) == [1, 1, 1, 3, 4, 4]
        @test parse.(Int, getindex.(rows, "observation_id")) == [1, 2, 3, 1, 1, 2]
        @test parse.(Int, getindex.(rows, "frame_index")) == [2, 3, 6, 4, 1, 2]
        @test parse.(Int, getindex.(rows, "gap_before")) == [0, 0, 2, 0, 0, 0]
        @test parse.(Float64, getindex.(rows, "elapsed_time")) == [1, 2, 5, 3, 0, 1]
        @test all(row -> row["time_provenance"] == "frame_index", rows)
        @test all(row -> row["length_unit"] == "px" && row["time_unit"] == "frame" &&
                         row["velocity_unit"] == "px/frame", rows)
        @test all(row -> row["position_valid"] == "true", rows)
        @test getindex.(rows, "velocity_valid") == ["true", "true", "true", "false", "true", "true"]
        @test rows[4]["u"] == rows[4]["v"] == ""
        @test parse.(Float64, getindex.(rows[1:3], "u")) == [4, 4, 4]
        @test parse.(Float64, getindex.(rows[1:3], "v")) == [-2, -2, -2]
        unused = ("i", "j", "z", "w", "masked", "outlier", "peak_ratio",
                  "correlation_moment", "uncertainty_u", "uncertainty_v", "uncertainty_w",
                  "match_residual", "index_a", "index_b")
        @test all(row -> all(key -> row[key] == "", unused), rows)

        # Unequal slopes across unequal frame intervals exercise the existing
        # central secant convention, rather than averaging adjacent velocities.
        curved = Trajectory{Float64}(2, [10.0, 14.0, 23.0], [0.0, 2.0, 14.0], [2, 3, 6])
        export_table(path, TrackingResult([curved], 6, params))
        curved_rows = tracking_export_rows(path)
        @test parse.(Float64, getindex.(curved_rows, "u")) == [4, 3.25, 3]
        @test parse.(Float64, getindex.(curved_rows, "v")) == [2, 3.5, 4]

        # A non-Julia reader can reconstruct each observed trajectory by ID.
        for id in (1, 3, 4)
            selected = filter(row -> parse(Int, row["trajectory_id"]) == id, rows)
            restored = Trajectory{Float64}(parse(Int, first(selected)["frame_index"]),
                parse.(Float64, getindex.(selected, "x")),
                parse.(Float64, getindex.(selected, "y")),
                parse.(Int, getindex.(selected, "frame_index")))
            expected = result.trajectories[id]
            @test restored.x == expected.x && restored.y == expected.y &&
                  restored.frames == expected.frames && restored.start_frame == expected.start_frame
        end

        @testset "physical scale and quoted labels" begin
            scale = PhysicalScale(pixel_size=0.25, dt=0.5,
                                  length_unit="µm, \"lab\"", time_unit="s")
            scaled = with_scale(result, scale)
            labels = (frame_id="run, \"two\"", source_a="α\nimage.tif", source_b="B\r\nimage.tif")
            export_table(path, scaled; labels...)
            raw_text = read(path, String)
            physical_rows = tracking_export_rows(path)
            @test parse.(Float64, getindex.(physical_rows[1:3], "x")) == gapped.x .* 0.25
            @test parse.(Float64, getindex.(physical_rows[1:3], "y")) == gapped.y .* 0.25
            @test parse.(Float64, getindex.(physical_rows[1:3], "u")) == [2, 2, 2]
            @test parse.(Float64, getindex.(physical_rows[1:3], "v")) == [-1, -1, -1]
            @test parse.(Float64, getindex.(physical_rows, "elapsed_time")) == [0.5, 1, 2.5, 1.5, 0, 0.5]
            @test all(row -> row["length_unit"] == scale.length_unit &&
                             row["time_unit"] == "s" && row["velocity_unit"] == scale.length_unit * "/s" &&
                             row["time_provenance"] == "physical_scale", physical_rows)
            @test all(row -> all(key -> row[String(key)] == labels[key], keys(labels)), physical_rows)
            export_table(path, physical(scaled); labels...)
            @test read(path, String) == raw_text  # no double conversion; dt survives
            @test gapped.x == [10, 14, 26] && result.scale === nothing
            for (pixel_size, dt) in ((1.0, 0.25), (0.25, 1.0), (1.0, 1.0))
                s = PhysicalScale(pixel_size=pixel_size, dt=dt, length_unit="mm", time_unit="s")
                export_table(path, with_scale(result, s))
                one = read(path, String)
                export_table(path, physical(with_scale(result, s)))
                @test read(path, String) == one
            end
        end

        @testset "numerical validity and degenerate results" begin
            invalid = Trajectory(1, [0.0, NaN, 4.0], [0.0, 2.0, 4.0])
            export_table(path, TrackingResult([invalid], 3, params))
            invalid_rows = tracking_export_rows(path)
            @test getindex.(invalid_rows, "position_valid") == ["true", "false", "true"]
            @test all(row -> row["velocity_valid"] == "false", invalid_rows)
            @test isnan(parse(Float64, invalid_rows[2]["x"]))
            @test parse(Float64, invalid_rows[2]["u"]) == 2  # finite stencil, invalid current position
            inf_track = Trajectory(1, [1.0, Inf], [0.0, 1.0])
            export_table(path, TrackingResult([inf_track], 2, params))
            @test getindex.(tracking_export_rows(path), "position_valid") == ["true", "false"]
            for empty_result in (TrackingResult(Trajectory{Float64}[], 0, params),
                                 TrackingResult([empty_track], 6, params),
                                 with_scale(TrackingResult(Trajectory{Float64}[], 6, params),
                                            PhysicalScale(pixel_size=0.25, dt=0.5)))
                export_table(path, empty_result)
                @test isempty(tracking_export_rows(path))
                @test read(path, String) == join(TABLE_COLUMNS, ',') * "\n"
            end
            t32 = Trajectory{Float32}(1, Float32[1, 5], Float32[2, 8], [1, 3])
            export_table(path, with_scale(TrackingResult([t32], 3, params),
                PhysicalScale(pixel_size=0.5, dt=0.25, length_unit="mm", time_unit="s")))
            @test parse.(Float64, getindex.(tracking_export_rows(path), "u")) == [4, 4]
        end

        @testset "malformed trajectories preserve destination" begin
            bad_tracks = [
                Trajectory{Float64}(1, [1.0, 2.0], [1.0], [1, 2]),
                Trajectory{Float64}(1, [1.0, 2.0], [1.0, 2.0], [1]),
                Trajectory{Float64}(1, [1.0, 2.0], [1.0, 2.0], [1, 1]),
                Trajectory{Float64}(2, [1.0, 2.0], [1.0, 2.0], [2, 1]),
                Trajectory{Float64}(1, [1.0], [1.0], [0]),
                Trajectory{Float64}(3, [1.0], [1.0], [3]),
                Trajectory{Float64}(2, [1.0], [1.0], [1]),
            ]
            for bad in bad_tracks
                write(path, "existing archive")
                @test_throws ArgumentError export_table(path, TrackingResult([bad], 2, params))
                @test read(path, String) == "existing archive"
            end
            @test_throws ArgumentError export_table(path, TrackingResult(Trajectory{Float64}[], -1, params))
        end

        @testset "existing result schema stays compatible" begin
            old_columns = ("schema_version", "result_type", "frame_id", "source_a", "source_b",
                "point_id", "i", "j", "x", "y", "z", "u", "v", "w", "masked", "outlier",
                "peak_ratio", "correlation_moment", "uncertainty_u", "uncertainty_v", "uncertainty_w",
                "match_residual", "index_a", "index_b", "length_unit", "time_unit", "velocity_unit")
            @test TABLE_SCHEMA_VERSION == "hammerhead-table-1"
            @test TABLE_COLUMNS[1:length(old_columns)] == old_columns
            a = fill(2.0, 1, 1)
            piv = PIVResult([10.0], [20.0], a, -a, a, a, a, a,
                            trues(1, 1), falses(1, 1), PIVParameters())
            stereo = StereoPIVResult(piv.x, piv.y, 3.0, piv.u, piv.v, a,
                a, a, a, piv.outliers, piv.mask, piv, piv, piv.parameters)
            particles = detect_particles(zeros(16, 16), params)
            ptv = PTVResult([10.0], [20.0], [2.0], [-2.0], [0.25], trues(1),
                            [1], [2], particles, particles, params)
            for (r, kind) in ((piv, "planar"), (stereo, "stereo"), (ptv, "ptv"))
                export_table(path, r; frame_id="pair-1")
                row = only(tracking_export_rows(path))
                @test row["result_type"] == kind && row["frame_id"] == "pair-1"
                @test row["x"] == "10.0" && row["y"] == "20.0" &&
                      row["u"] == "2.0" && row["v"] == "-2.0" && row["outlier"] == "true"
                @test all(key -> row[key] == "", TABLE_COLUMNS[length(old_columns) + 1:end])
                if kind == "ptv"
                    @test row["match_residual"] == "0.25" && row["index_a"] == "1" && row["index_b"] == "2"
                else
                    @test row["i"] == row["j"] == "1" && row["masked"] == "false"
                    @test row["z"] == (kind == "stereo" ? "3.0" : "")
                end
            end
        end
    end
end
