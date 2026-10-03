using Test
using Hammerhead
using JLD2

# A dataset that really throws upon deserialization detects accidental bulk
# reads, including array display; an unsupported integer alone cannot do that.
struct LazyUnreadableEntry
    value::Int
end
const lazy_unreadable_reads = Ref(0)
JLD2.writeas(::Type{LazyUnreadableEntry}) = Int
JLD2.wconvert(::Type{Int}, entry::LazyUnreadableEntry) = entry.value
function JLD2.rconvert(::Type{LazyUnreadableEntry}, ::Int)
    lazy_unreadable_reads[] += 1
    error("deliberately unreadable result entry")
end

function lazy_test_grid(value::T) where {T<:AbstractFloat}
    n = 3
    return PIVResult(T[8.5, 16.5, 24.5], T[8.5, 16.5, 24.5],
                     fill(value, n, n), fill(-value, n, n),
                     ones(T, n, n), ones(T, n, n),
                     fill(T(NaN), n, n), fill(T(NaN), n, n),
                     falses(n, n), falses(n, n),
                     PIVParameters(window_size = 16, overlap = 8))
end

@testset "Indexed native result loading" begin
    grid = lazy_test_grid(2.0f0)
    scale = PhysicalScale(0.2, 0.5, "mm", "s")
    stereo = StereoPIVResult(copy(grid.x), copy(grid.y), 0.0f0,
                              grid.u, grid.v, copy(grid.u),
                              grid.uncertainty_u, grid.uncertainty_v, copy(grid.uncertainty_u),
                              grid.outliers, grid.mask, grid, grid, grid.parameters)
    particles = Particles([10.0, 20.0], [10.0, 20.0], ones(2), fill(3.0, 2))
    ptv = PTVResult(particles.x, particles.y, ones(2), zeros(2), zeros(2),
                    falses(2), [1, 2], [1, 2], particles, particles, PTVParameters())
    tracks = TrackingResult([Trajectory(1, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0])],
                             3, PTVParameters())
    entries = Hammerhead.SavedResult[with_scale(grid, scale), with_scale(stereo, scale),
                                     with_scale(ptv, scale), with_scale(tracks, scale)]

    mktempdir() do dir
        path = joinpath(dir, "mixed.jld2")
        save_results(path, entries)
        eager = load_results(path)
        index = ResultFile(path)
        @test load_results(path; lazy = true) isa ResultFile
        @test size(index) == (4,) && length(index) == 4
        @test index isa AbstractVector{Hammerhead.SavedResult}
        @test index.path == abspath(path)
        @test index.entry_keys == ["000001", "000002", "000003", "000004"]
        for i in (4, 1, 3, 2)
            result = index[i]
            @test typeof(result) == typeof(eager[i])
            @test result.scale.pixel_size == scale.pixel_size
            @test result.scale.dt == scale.dt
            @test result.scale.length_unit == "mm"
            if result isa TrackingResult
                @test result.trajectories[1].x == eager[i].trajectories[1].x
                @test result.trajectories[1].frames == eager[i].trajectories[1].frames
            else
                @test result.x == eager[i].x && result.y == eager[i].y
                @test isequal(result.u, eager[i].u) && isequal(result.v, eager[i].v)
            end
        end
        @test index[Int32(1)] isa PIVResult{Float32}
        @test_throws BoundsError index[0]
        @test_throws BoundsError index[5]
        @test length(collect(index)) == 4
        first_read = index[1]
        first_read.u .= -99
        @test all(==(2.0f0), index[1].u) # no payload retained between reads
        # The lazy AbstractVector is accepted by save_results. Guard the source
        # before opening a writer, including normal views/reshape wrappers.
        before_copy = read(path)
        @test_throws ArgumentError save_results(path, index)
        # A relative alias must be relative to the source volume. CI may
        # check out on D: while the temporary directory lives on C:.
        cd(dir) do
            relative_alias=relpath(path)
            @test Base.samefile(relative_alias,path)
            @test_throws ArgumentError save_results(relative_alias, index)
        end
        @test_throws ArgumentError save_results(path, view(index, 1:2))
        @test_throws ArgumentError save_results(path, view(view(index, :), [1, 3]))
        @test_throws ArgumentError save_results(path, reshape(index, (4,)))
        @test_throws ArgumentError save_results(path, PermutedDimsArray(index, (1,)))
        @test read(path) == before_copy
        @test index[1].u == eager[1].u
        copy_path = joinpath(dir, "copy.jld2")
        @test save_results(copy_path, index) == copy_path
        @test length(load_results(copy_path)) == 4
        @test ResultFile(copy_path)[4].trajectories[1].x == tracks.trajectories[1].x
        alias_path = joinpath(dir, "source-hardlink.jld2")
        linked = try
            Base.hardlink(path, alias_path)
            true
        catch
            false # filesystems without hardlink support still check aliases above
        end
        if linked
            @test Base.samefile(path, alias_path)
            @test_throws ArgumentError save_results(alias_path, index)
            @test_throws ArgumentError save_results(alias_path, view(index, 2:3))
            @test read(path) == before_copy
        else
            @test_skip false
        end
        # No handle remains open on Windows after constructor/access.
        renamed = joinpath(dir, "renamed.jld2")
        mv(path, renamed)
        mv(renamed, path)
        @test index[1].x == grid.x

        # Sorted noncontiguous and nonnumeric keys match eager saved ordering.
        unordered = joinpath(dir, "unordered.jld2")
        jldopen(unordered, "w") do f
            f["format_version"] = Hammerhead.RESULTS_FORMAT_VERSION
            f["results/000010"] = lazy_test_grid(10.0)
            f["results/000002"] = lazy_test_grid(2.0)
            f["results/label"] = lazy_test_grid(30.0)
        end
        unordered_index = ResultFile(unordered)
        @test unordered_index.entry_keys == ["000002", "000010", "label"]
        @test [r.u[1] for r in unordered_index] == [r.u[1] for r in load_results(unordered)]

        empty_path = joinpath(dir, "empty.jld2")
        save_results(empty_path, PIVResult[])
        @test isempty(ResultFile(empty_path))
        @test isempty(collect(load_results(empty_path; lazy = true)))

        invalid = joinpath(dir, "invalid.jld2")
        jldopen(f -> (f["format_version"] = 999), invalid, "w")
        @test_throws ArgumentError ResultFile(invalid)
        @test_throws ArgumentError load_results(invalid; lazy = true)
        @test_throws ArgumentError load_results(invalid)
        jldopen(f -> (f["other"] = 1), invalid, "w")
        @test_throws ArgumentError ResultFile(invalid)
        @test_throws ArgumentError load_results(invalid)

        unreadable = joinpath(dir, "unreadable.jld2")
        jldopen(unreadable, "w") do f
            f["format_version"] = Hammerhead.RESULTS_FORMAT_VERSION
            f["results/000001"] = grid
            f["results/000002"] = LazyUnreadableEntry(7)
            f["results/000003"] = ptv
            f["results/000004"] = 123
        end
        lazy_unreadable_reads[] = 0
        lazy_index = ResultFile(unreadable)
        @test length(lazy_index) == 4
        @test occursin("4 results", sprint(show, lazy_index))
        @test occursin("4 results", sprint(show, MIME"text/plain"(), lazy_index))
        @test lazy_index[1] isa PIVResult
        @test lazy_index[3] isa PTVResult
        @test lazy_unreadable_reads[] == 0
        @test_throws ErrorException lazy_index[2]
        @test lazy_unreadable_reads[] == 1
        @test_throws ArgumentError lazy_index[4]
        @test lazy_index[3] isa PTVResult # recover to another readable entry
        # A subset copy remains lazy: it must never read the poison entry.
        subset_path = joinpath(dir, "selected-copy.jld2")
        save_results(subset_path, view(lazy_index, [1, 3]))
        @test length(load_results(subset_path)) == 2
        @test load_results(subset_path)[2] isa PTVResult
        @test lazy_unreadable_reads[] == 1
        @test_throws ErrorException load_results(unreadable)
        @test lazy_unreadable_reads[] == 2

        # A completed-file index rejects detectable replacement/appends.
        changed_index = ResultFile(path)
        jldopen(path, "a+") do f
            f["results/000005"] = lazy_test_grid(5.0)
        end
        @test length(changed_index) == 4
        @test_throws ArgumentError changed_index[1]
        @test length(ResultFile(path)) == 5
    end
end
