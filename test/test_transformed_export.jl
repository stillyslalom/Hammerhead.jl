using Test
using Hammerhead
using LinearAlgebra

function transformed_export_fixture(::Type{T}=Float64) where {T}
    x, y = T[10, 20, 30], T[40, 50]
    u = T[1 2 3; 4 5 6]
    v = T[7 8 9; 10 11 12]
    su, sv = fill(T(0.3), 2, 3), fill(T(0.4), 2, 3)
    mask = falses(2, 3); mask[2,3] = true
    flags = falses(2, 3); flags[1,2] = true
    u[2,3] = v[2,3] = su[2,3] = sv[2,3] = T(NaN)
    PIVResult(x, y, u, v, fill(T(2), 2, 3), fill(T(0.2), 2, 3),
              su, sv, flags, mask, PIVParameters())
end

# These test labels contain no delimiters; quoting/UTF-8 round trips are
# already exercised independently in test_tracking_export.jl.
function transformed_export_csv(path)
    records = split.(readlines(path), ',')
    columns = first(records)
    @test columns == collect(TABLE_COLUMNS)
    @test all(row -> length(row) == length(columns), records[2:end])
    [Dict(zip(columns, strip.(row, '"'))) for row in records[2:end]]
end

function transformed_export_vtk(path)
    lines = readlines(path)
    start = findfirst(line -> startswith(line, "POINTS "), lines)
    n = parse(Int, split(lines[start])[2])
    points = [parse.(Float64, split(line)) for line in lines[start+1:start+n]]
    start = findfirst(==("VECTORS velocity double"), lines)
    vectors = [parse.(Float64, split(line)) for line in lines[start+1:start+n]]
    scalars = Dict{String,Vector{Float64}}()
    units = Dict{String,String}()
    for (i, line) in enumerate(lines)
        if startswith(line, "SCALARS ")
            scalars[split(line)[2]] = parse.(Float64, lines[i+2:i+1+n])
        elseif endswith(line, "unsigned_char")
            name, _, count, _ = split(line)
            bytes = parse.(UInt8, split(lines[i+1]))
            @test length(bytes) == parse(Int, count)
            units[name] = String(bytes)
        end
    end
    (; points, vectors, scalars, units)
end

@testset "Planar transformed table and VTK export" begin
    r = transformed_export_fixture()
    original = deepcopy(r)
    mktempdir() do dir
        csv, vtk = joinpath(dir, "grid.csv"), joinpath(dir, "grid.vtk")
        @testset "affine point/vector basis and grid topology" begin
            transforms = [
                PlanarTransform([1.0 0; 0 1], [-10.0, -40.0]),
                PlanarTransform([0.0 1; -1 0], [4.0, -3.0]),
                PlanarTransform([2.0 0; 0 -3], [7.0, 9.0]),
                planar_calibration((10.0, 40.0), (20.0, 50.0), 5.0;
                    origin=(3.0, 4.0), reflection=true, perpendicular_scale=0.2),
            ]
            for transform in transforms, interval in (nothing, 0.25)
                time_kwargs = interval === nothing ? (;) : (; dt=interval, time_unit="s")
                kwargs = (; transform, length_unit="mm", uncertainty_assumption=:independent,
                            time_kwargs...)
                @test export_table(csv, r; kwargs...) == csv
                @test export_vtk(vtk, r; kwargs...) == vtk
                rows = transformed_export_csv(csv)
                data = transformed_export_vtk(vtk)
                @test length(rows) == length(data.points) == length(data.vectors) == 6
                @test occursin("DIMENSIONS 3 2 1", read(vtk, String))
                divisor = interval === nothing ? 1 : interval
                A, b = transform.matrix, transform.offset
                # CSV retains column-major point IDs; VTK uses x-fast order.
                for row in rows
                    i, j = parse(Int, row["i"]), parse(Int, row["j"])
                    xy = A * [r.x[j], r.y[i]] + b
                    uv = A * [r.u[i,j], r.v[i,j]] / divisor
                    sigmas = [hypot(A[k,1] * r.uncertainty_u[i,j],
                                    A[k,2] * r.uncertainty_v[i,j]) / divisor for k in 1:2]
                    @test parse(Int, row["point_id"]) == i + (j - 1) * length(r.y)
                    @test parse.(Float64, [row["x"], row["y"]]) ≈ xy
                    @test isequal(parse.(Float64, [row["u"], row["v"]]), collect(uv))
                    @test parse(Bool, row["masked"]) == r.mask[i,j]
                    @test parse(Bool, row["outlier"]) == r.outliers[i,j]
                    @test row["peak_ratio"] == string(r.peak_ratio[i,j]) &&
                          row["correlation_moment"] == string(r.correlation_moment[i,j])
                    exported_sigma = parse.(Float64, [row["uncertainty_u"], row["uncertainty_v"]])
                    @test all(k -> isequal(exported_sigma[k], sigmas[k]) ||
                                   isapprox(exported_sigma[k], sigmas[k]), 1:2)
                    vtk_id = j + (i - 1) * length(r.x)
                    @test data.points[vtk_id][1:2] ≈ xy
                    @test data.points[vtk_id][3] == 0
                    @test isequal(data.vectors[vtk_id][1:2], collect(uv))
                    @test data.vectors[vtk_id][3] == 0
                    @test data.scalars["masked"][vtk_id] == r.mask[i,j]
                    @test data.scalars["outlier"][vtk_id] == r.outliers[i,j]
                    @test isequal(data.scalars["uncertainty_u"][vtk_id], exported_sigma[1])
                    @test isequal(data.scalars["uncertainty_v"][vtk_id], exported_sigma[2])
                    @test row["length_unit"] == "mm"
                    @test row["time_unit"] == (interval === nothing ? "frame" : "s")
                    @test row["velocity_unit"] == (interval === nothing ? "mm/frame" : "mm/s")
                    @test all(key -> row[key] == "", TABLE_COLUMNS[28:end])
                end
                @test data.units == Dict("coordinate_unit" => "mm",
                    "component_unit" => (interval === nothing ? "mm/frame" : "mm/s"))
            end
        end

        @testset "uncertainty assumptions and unavailable marginals" begin
            diagonal = PlanarTransform([2.0 0; 0 -3], zeros(2))
            permutation = PlanarTransform([0.0 -2; 3 0], zeros(2))
            mixed = PlanarTransform([0.6 0.8; -0.8 0.6], zeros(2))
            for transform in (diagonal, permutation)
                export_table(csv, r; transform, length_unit="mm")
                rows = transformed_export_csv(csv)
                export_table(csv, r; transform, length_unit="mm", uncertainty_assumption=:independent)
                @test transformed_export_csv(csv) == rows  # no covariance assumption needed
            end
            export_table(csv, r; transform=mixed, length_unit="mm")
            rows = transformed_export_csv(csv)
            @test all(row -> row["uncertainty_u"] == row["uncertainty_v"] == "NaN", rows)
            export_vtk(vtk, r; transform=mixed, length_unit="mm")
            data = transformed_export_vtk(vtk)
            @test all(isnan, data.scalars["uncertainty_u"])
            @test all(isnan, data.scalars["uncertainty_v"])
            unavailable = transformed_export_fixture()
            unavailable.uncertainty_v[1,1] = NaN
            export_table(csv, unavailable; transform=diagonal, length_unit="mm")
            firstrow = first(transformed_export_csv(csv))
            @test parse(Float64, firstrow["uncertainty_u"]) == 0.6
            @test firstrow["uncertainty_v"] == "NaN"
            export_table(csv, unavailable; transform=permutation, length_unit="mm")
            firstrow = first(transformed_export_csv(csv))
            @test firstrow["uncertainty_u"] == "NaN"
            @test parse(Float64, firstrow["uncertainty_v"]) ≈ 0.9
            export_table(csv, unavailable; transform=mixed, length_unit="mm",
                         uncertainty_assumption=:independent)
            @test first(transformed_export_csv(csv))["uncertainty_u"] == "NaN"
            unavailable.uncertainty_u[1,1] = -0.3
            export_table(csv, unavailable; transform=diagonal, length_unit="mm")
            @test first(transformed_export_csv(csv))["uncertainty_u"] == "NaN"
        end

        @testset "precision and zero-node grids" begin
            r32 = transformed_export_fixture(Float32)
            transform = PlanarTransform(Float32[2 0; 0 -3], Float32[4, 5])
            _, _, _, _, grid = Hammerhead._export_grid(r32; transform, length_unit="mm")
            @test eltype(grid.x) == eltype(grid.u) == eltype(grid.su) == Float32
            # Integer constructor inputs produce a floating calibration; export
            # promotes the Float32 measurements to that calibration precision.
            integer_transform = PlanarTransform([2 0; 0 -3], [4, 5])
            _, _, _, _, promoted = Hammerhead._export_grid(r32;
                transform=integer_transform, length_unit="mm", dt=0.5, time_unit="s")
            @test eltype(promoted.x) == eltype(promoted.u) == eltype(promoted.su) == Float64
            @test promoted.x[1,1] == 24 && promoted.y[1,1] == -115
            @test promoted.u[1,1] == 4 && promoted.v[1,1] == -42
            empty = PIVResult(Float64[], Float64[], zeros(0,0), zeros(0,0), zeros(0,0),
                zeros(0,0), zeros(0,0), zeros(0,0), falses(0,0), falses(0,0), PIVParameters())
            export_table(csv, empty; transform, length_unit="mm")
            @test isempty(transformed_export_csv(csv))
        end

        @testset "refusal precedes destination replacement" begin
            transform = PlanarTransform([2.0 0; 0 -3], zeros(2))
            bad_kwargs = [
                (; transform), (; transform, length_unit=""),
                (; transform, length_unit="mm", dt=0.5),
                (; transform, length_unit="mm", time_unit="s"),
                (; transform, length_unit="mm", dt=0.0, time_unit="s"),
                (; transform, length_unit="mm", dt=Inf, time_unit="s"),
                (; transform, length_unit="mm", dt=-0.5, time_unit="s"),
                (; transform, length_unit="mm", dt=0.5, time_unit=""),
                (; transform, length_unit="mm", uncertainty_assumption=:correlated),
                (; transform="affine", length_unit="mm"),
                (; transform=PlanarTransform(zeros(2,2), zeros(2)), length_unit="mm"),
                (; transform=PlanarTransform([NaN 0; 0 1], zeros(2)), length_unit="mm"),
                (; transform=PlanarTransform([1.0 0; 0 1], [Inf, 0]), length_unit="mm"),
                (; length_unit="mm"), (; dt=0.5), (; time_unit="s"),
                (; uncertainty_assumption=:independent),
            ]
            for writer in (export_table, export_vtk), kwargs in bad_kwargs
                path = writer === export_table ? csv : vtk
                write(path, "existing export")
                @test_throws ArgumentError writer(path, r; kwargs...)
                @test read(path, String) == "existing export"
            end
            scale = PhysicalScale(pixel_size=0.25, dt=0.5, length_unit="mm", time_unit="s")
            for scaled in (with_scale(r, scale), physical(with_scale(r, scale)),
                           with_scale(r, PhysicalScale()))
                for writer in (export_table, export_vtk)
                    path = writer === export_table ? csv : vtk
                    write(path, "existing export")
                    @test_throws ArgumentError writer(path, scaled; transform, length_unit="mm")
                    @test read(path, String) == "existing export"
                end
            end
            stereo = StereoPIVResult(r.x, r.y, 0.0, r.u, r.v, r.u, r.uncertainty_u,
                r.uncertainty_v, r.uncertainty_u, r.outliers, r.mask, r, r, r.parameters)
            for writer in (export_table, export_vtk)
                write(csv, "existing export")
                @test_throws ArgumentError writer(csv, stereo; transform, length_unit="mm")
                @test read(csv, String) == "existing export"
            end
            write(csv, "existing export")
            @test_throws ArgumentError export_table(csv, TrackingResult(Trajectory{Float64}[],
                0, PTVParameters()); transform, length_unit="mm")
            @test read(csv, String) == "existing export"
            particles = detect_particles(zeros(8,8), PTVParameters())
            ptv = PTVResult(Float64[], Float64[], Float64[], Float64[], Float64[], falses(0),
                Int[], Int[], particles, particles, PTVParameters())
            @test_throws ArgumentError export_table(csv, ptv; transform, length_unit="mm")
            @test read(csv, String) == "existing export"
        end

        @testset "default and physical export behavior" begin
            for writer in (export_table, export_vtk)
                path = writer === export_table ? csv : vtk
                writer(path, r)
                baseline = read(path, String)
                writer(path, r; transform=nothing)
                @test read(path, String) == baseline
                s = PhysicalScale(pixel_size=0.25, dt=0.5, length_unit="mm", time_unit="s")
                writer(path, with_scale(r, s))
                scaled = read(path, String)
                writer(path, physical(with_scale(r, s)))
                @test read(path, String) == scaled
                writer(path, with_scale(with_scale(r, s), nothing);
                       transform=PlanarTransform([0.25 0; 0 0.25], zeros(2)),
                       length_unit="mm", dt=0.5, time_unit="s")
                # Same numeric fields and units as scalar physical calibration.
                @test read(path, String) == scaled
            end
            @test isequal(r.u, original.u) && isequal(r.v, original.v) &&
                  isequal(r.uncertainty_u, original.uncertainty_u) && r.x == original.x &&
                  r.y == original.y && r.outliers == original.outliers && r.mask == original.mask
        end
    end
end
