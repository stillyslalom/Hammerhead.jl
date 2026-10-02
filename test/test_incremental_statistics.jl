# Self-contained result fixtures, including deliberately malformed dimensions
# used to verify that compatibility checks precede any accumulator mutation.
function incremental_planar(u::Matrix{T}, v::Matrix{T};
        x = T.(1:size(u, 2)), y = T.(1:size(u, 1)),
        mask = falses(size(u)), outliers = falses(size(u)), scale = nothing) where {T<:AbstractFloat}
    dims = size(u)
    PIVResult{T}(collect(T, x), collect(T, y), u, v, ones(T, dims), zeros(T, dims),
        fill(T(NaN), dims), fill(T(NaN), dims), outliers, mask, PIVParameters(),
        nothing, scale)
end

function incremental_stereo(u::Matrix{T}, v::Matrix{T}, w::Matrix{T};
        x = T.(1:size(u, 2)), y = T.(1:size(u, 1)), z = 0.0,
        mask = falses(size(u)), outliers = falses(size(u)), scale = nothing) where {T<:AbstractFloat}
    cam = incremental_planar(copy(u), copy(v); x, y)
    StereoPIVResult{T}(collect(T, x), collect(T, y), z, u, v, w,
        fill(T(NaN), size(u)), fill(T(NaN), size(u)), fill(T(NaN), size(u)),
        outliers, mask, cam, cam, PIVParameters(), scale)
end

Base.@noinline function incremental_release_probe()
    result = incremental_stereo(ones(2, 3), fill(2.0, 2, 3), fill(3.0, 2, 3))
    refs = (WeakRef(result), WeakRef(result.u), WeakRef(result.x))
    acc = FieldStatisticsAccumulator()
    update_statistics!(acc, result)
    return acc, refs
end

@testset "Incremental field statistics" begin
    @testset "Batch equivalence, validity, and independent snapshots" begin
        for stereo in (false, true), include_invalid in (false, true)
            results = map(1:4) do k
                u = fill(Float64(k), 2, 3)
                v = fill(-2.0 * k, 2, 3)
                w = fill(3.0 * k, 2, 3)
                mask = falses(2, 3); mask[2, 3] = true
                outliers = falses(2, 3); outliers[1, 1] = iseven(k)
                k == 1 && (u[1, 2] = NaN)
                k == 2 && (v[1, 2] = Inf)
                k == 3 && (w[2, 1] = -Inf)
                stereo ? incremental_stereo(u, v, w; mask, outliers) :
                         incremental_planar(u, v; mask, outliers)
            end
            acc = FieldStatisticsAccumulator(; include_invalid)
            @test_throws ArgumentError field_statistics(acc)
            for (i, r) in enumerate(results)
                @test update_statistics!(acc, r) === acc
                expected = field_statistics(results[1:i]; include_invalid)
                @test isequal(field_statistics(acc), expected)
                @test acc.nsamples == i
            end
            stats = field_statistics(acc)
            @test stats.count[1, 1] == (include_invalid ? 4 : 2)
            @test stats.count[1, 2] == 2
            @test stats.count[2, 1] == (stereo ? 3 : 4)
            @test stats.count[2, 3] == 0 && isnan(stats.mean_u[2, 3])
            @test isnan(stats.rms_u[2, 3])
            untouched = deepcopy(stats)
            for value in values(stats)
                value isa AbstractArray || continue
                value[1] = 123
            end
            @test isequal(field_statistics(acc), untouched)
            @test acc.count !== field_statistics(acc).count
            @test acc.x !== field_statistics(acc).x
        end

        empty_node = incremental_planar(fill(NaN, 2, 3), ones(2, 3))
        acc = FieldStatisticsAccumulator()
        update_statistics!(acc, empty_node)
        @test acc.nsamples == 1 && all(iszero, field_statistics(acc).count)
        @test all(isnan, field_statistics(acc).mean_v)

        # Mixed precisions are accepted when coordinate values and units agree;
        # snapshot coordinates preserve the first field's precision.
        f32 = incremental_planar(fill(1.0f0, 2, 3), fill(2.0f0, 2, 3))
        f64 = incremental_planar(fill(3.0, 2, 3), fill(4.0, 2, 3))
        acc = FieldStatisticsAccumulator()
        update_statistics!(acc, f32); update_statistics!(acc, f64)
        @test eltype(field_statistics(acc).x) === Float32
        @test eltype(field_statistics(acc).mean_u) === Float64
        @test all(==(2.0), field_statistics(acc).mean_u)
    end

    @testset "Stable population moments with large offsets" begin
        acc = FieldStatisticsAccumulator()
        for delta in (-3.0, -1.0, 1.0, 3.0)
            update_statistics!(acc, incremental_stereo(fill(1e12 + delta, 2, 3),
                fill(-1e12 - 2delta, 2, 3), fill(1e12 + 3delta, 2, 3)))
        end
        stats = field_statistics(acc)
        @test stats.mean_u == fill(1e12, 2, 3)
        @test stats.mean_v == fill(-1e12, 2, 3)
        @test stats.mean_w == fill(1e12, 2, 3)
        @test stats.reynolds_uu == fill(5.0, 2, 3)
        @test stats.reynolds_vv == fill(20.0, 2, 3)
        @test stats.reynolds_ww == fill(45.0, 2, 3)
        @test stats.reynolds_uv == fill(-10.0, 2, 3)
        @test stats.reynolds_uw == fill(15.0, 2, 3)
        @test stats.reynolds_vw == fill(-30.0, 2, 3)
        @test stats.rms_u ≈ fill(sqrt(5.0), 2, 3)
        @test stats.rms_w ≈ fill(sqrt(45.0), 2, 3)
    end

    @testset "Compatibility failures leave state unchanged" begin
        base = incremental_planar(ones(2, 3), fill(2.0, 2, 3))
        acc = FieldStatisticsAccumulator()
        update_statistics!(acc, base)
        before = field_statistics(acc)
        bad_results = (
            incremental_stereo(ones(2, 3), ones(2, 3), ones(2, 3)),
            incremental_planar(ones(2, 3), ones(2, 3); x = [2., 3., 4.]),
            incremental_planar(ones(2, 3), ones(2, 3); y = [2., 3.]),
            incremental_planar(ones(2, 3), ones(1, 3)),
            incremental_planar(ones(2, 3), ones(2, 3); mask = falses(1, 3)),
            incremental_planar(ones(2, 3), ones(2, 3); outliers = falses(1, 3)),
            incremental_planar(ones(2, 3), ones(2, 3); x = [1., NaN, 3.]),
            incremental_planar(ones(2, 3), ones(2, 3); x = [1., 2.]),
            with_scale(base, PhysicalScale()))
        for r in bad_results
            @test_throws ArgumentError update_statistics!(acc, r)
            @test isequal(field_statistics(acc), before)
            @test acc.nsamples == 1
            first_acc = FieldStatisticsAccumulator()
            # Kind/grid mismatches may be valid as first results; malformed
            # dimensions/nonfinite coordinates must leave even fresh state empty.
            if r === bad_results[4] || r === bad_results[5] || r === bad_results[6] ||
               r === bad_results[7] || r === bad_results[8]
                @test_throws ArgumentError update_statistics!(first_acc, r)
                @test first_acc.nsamples == 0 && first_acc.kind === :unset
                @test_throws ArgumentError field_statistics(first_acc)
            end
        end
        # Coordinates are copied during initialization, rather than aliasing
        # the first result's mutable vectors.
        base.x[1] = 50.0
        @test isequal(field_statistics(acc), before)
        @test_throws ArgumentError update_statistics!(acc, base)

        stereo = incremental_stereo(ones(2, 3), ones(2, 3), ones(2, 3))
        sa = FieldStatisticsAccumulator()
        update_statistics!(sa, stereo)
        sbefore = field_statistics(sa)
        for bad in (incremental_stereo(ones(2, 3), ones(2, 3), ones(2, 3); z = 1.0),
                    incremental_stereo(ones(2, 3), ones(2, 3), ones(1, 3)),
                    incremental_stereo(ones(2, 3), ones(2, 3), ones(2, 3); z = NaN),
                    incremental_planar(ones(2, 3), ones(2, 3)))
            @test_throws ArgumentError update_statistics!(sa, bad)
            @test isequal(field_statistics(sa), sbefore)
            @test sa.nsamples == 1
        end
    end

    @testset "Stored units and physical conversion" begin
        raw = incremental_planar(ones(2, 3), fill(2.0, 2, 3))
        scale = PhysicalScale(pixel_size = 0.02, dt = 0.001, length_unit = "mm", time_unit = "s")
        scaled = with_scale(raw, scale)
        acc = FieldStatisticsAccumulator()
        update_statistics!(acc, scaled)
        @test field_statistics(acc).mean_u == raw.u # attaching never converts
        update_statistics!(acc, with_scale(raw, PhysicalScale(0.02, 0.001, "mm", "s")))
        before = field_statistics(acc)
        for other_scale in (nothing,
                PhysicalScale(0.04, 0.001, "mm", "s"),
                PhysicalScale(0.02, 0.002, "mm", "s"),
                PhysicalScale(0.02, 0.001, "m", "s"),
                PhysicalScale(0.02, 0.001, "mm", "ms"))
            @test_throws ArgumentError update_statistics!(acc, with_scale(raw, other_scale))
            @test isequal(field_statistics(acc), before)
            @test acc.nsamples == 2
        end
        # Different pair delays become compatible after explicit conversion.
        velocity_acc = FieldStatisticsAccumulator()
        a = physical(scaled)
        b = physical(with_scale(raw, PhysicalScale(0.02, 0.002, "mm", "s")))
        update_statistics!(velocity_acc, a); update_statistics!(velocity_acc, b)
        @test isequal(field_statistics(velocity_acc), field_statistics([a, b]))
        @test field_statistics(velocity_acc).mean_u ≈ fill(15.0, 2, 3)
        @test field_statistics(velocity_acc).x ≈ raw.x .* 0.02
    end

    @testset "Fixed memory and input release" begin
        # Separate producer scope avoids test-local bindings/closures retaining
        # an input independently of the accumulator.
        acc, refs = incremental_release_probe()
        GC.gc(true)
        @test all(ref -> ref.value === nothing, refs)
        initial_bytes = Base.summarysize(acc)
        result = incremental_stereo(ones(2, 3), fill(2.0, 2, 3), fill(3.0, 2, 3))
        for _ in 1:2000
            update_statistics!(acc, result)
        end
        @test Base.summarysize(acc) == initial_bytes
        @test acc.nsamples == 2001 && all(==(2001), field_statistics(acc).count)
    end

    @testset "Non-collecting planar and stereo sequence callbacks" begin
        image = rand(MersenneTwister(149), 40, 40)
        pairs = [(image, image) for _ in 1:3]
        params = PIVParameters(window_size = 16, overlap = 8, uod_enable = false)
        acc = FieldStatisticsAccumulator()
        @test run_piv_sequence(pairs, params; progress = false, collect_results = false,
            on_result = (i, r) -> update_statistics!(acc, r)) === nothing
        collected = run_piv_sequence(pairs, params; progress = false)
        @test isequal(field_statistics(acc), field_statistics(collected))
        @test acc.nsamples == 3

        grid = DewarpGrid(x = 2.0:1.0:33.0, y = 2.0:1.0:33.0)
        cams = map((-8.0, 8.0)) do slope
            PinholeCamera([64.0 0.0 slope 0.0; 0.0 64.0 0.0 0.0; 0.0 0.0 1.0 64.0])
        end
        dw1, dw2 = map(cam -> ImageDewarper(cam, grid, size(image)), cams)
        sa = FieldStatisticsAccumulator()
        @test run_piv_stereo_sequence(pairs, pairs, dw1, dw2, params;
            progress = false, collect_results = false,
            on_result = (i, r) -> update_statistics!(sa, r)) === nothing
        stereo = run_piv_stereo_sequence(pairs, pairs, dw1, dw2, params; progress = false)
        @test isequal(field_statistics(sa), field_statistics(stereo))
        @test sa.nsamples == 3
    end
end
