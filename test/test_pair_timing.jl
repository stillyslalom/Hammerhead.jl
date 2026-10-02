using Test, Hammerhead, JLD2

timing_exact(value) = value === nothing ? nothing : parse(BigInt, value["numerator"]) // parse(BigInt, value["denominator"])
function timing_image()
    [Float64(mod(17i + 31j + 7i*j, 251)) / 250 for i in 1:32, j in 1:32]
end
timing_params() = PIVParameters(window_size = 16, overlap = 8, padding = true, uod_enable = false)
function timing_source(timestamps; labels = string.(1:length(timestamps)), kwargs...)
    loads = Int[]
    source = FrameSource(length(timestamps), i -> (push!(loads, i); timing_image()); timestamps, labels, kwargs...)
    (; source, loads)
end
function timing_no_payloads(value)
    value isa AbstractDict && return all(timing_no_payloads, values(value))
    value isa AbstractArray && return ndims(value) == 1 && all(timing_no_payloads, value)
    value === nothing || value isa Union{AbstractString,Number}
end
struct TimingCustomPair
    first
    second
    dt
end
Base.length(::TimingCustomPair) = 2
Base.getindex(p::TimingCustomPair, i::Int) = i == 1 ? p.first : p.second
Base.@noinline function timing_release_probe()
    source = FrameSource(6, i -> timing_image(); timestamps = collect(0:5))
    result_refs, packet_refs = WeakRef[], WeakRef[]
    returned = run_piv_sequence(image_pairs(source), timing_params(); collect_results = false, progress = false,
        on_result = (i, result) -> push!(result_refs, WeakRef(result.u)),
        on_pair_timing = (i, packet) -> push!(packet_refs, WeakRef(packet._data)))
    (; returned, result_refs, packet_refs)
end

@testset "Exact pair timing metadata and preflight" begin
    loader = i -> timing_image()
    old = FrameSource(2, loader, [0, 1], ["", "B"])
    typed = FrameSource{typeof(loader),Vector{Int},Vector{String}}(2, loader, [0, 1], ["A", "B"])
    @test old.source_id === nothing && typed.frame_ids === nothing
    @test FrameSource(0, loader; frame_ids = String[]).frame_ids isa Vector{String}
    @test FrameSource(2, loader; frame_ids = ["A", "B"]).frame_ids == ["A", "B"]
    @test_throws ArgumentError FrameSource(1, loader; frame_ids = [nothing])
    @test_throws DimensionMismatch FrameSource(2, loader; frame_ids = ["A"])
    @test_throws ArgumentError FrameSource(2, loader; source_id = "")

    epoch = typemax(Int64) - 7
    source, loads = timing_source([epoch, epoch + 3]; source_id = "camera-x", frame_ids = ["exp-1", "exp-2"], time_unit = "ns", clock_id = "clock-x")
    pairs = image_pairs(source)
    snapshot = only(Hammerhead._timing_preflight(pairs, nothing, 0.0, sqrt(eps(Float64))))
    @test isempty(loads)
    @test snapshot["frames"][1]["frame_index"] == 1 && snapshot["frames"][2]["frame_id"] == "exp-2"
    @test snapshot["frames"][1]["source_identity"] == "opaque_provided"
    @test timing_exact(snapshot["frames"][1]["timestamp"]) == epoch
    @test timing_exact(snapshot["observed_delay"]) == 3
    @test timing_exact(snapshot["sample_time"]) == BigInt(epoch) + 3 // 2
    @test snapshot["sample_time"]["denominator"] == "2"
    @test snapshot["timestamp_status"] == snapshot["clock_status"] == snapshot["time_unit_status"] == "complete"
    @test snapshot["pair_time_unit"] == "ns" && snapshot["pair_clock_id"] == "clock-x"
    @test snapshot["effective_delay_provenance"] == "declared_pair_dt_metadata_only" && !snapshot["scale_applied"]
    @test timing_no_payloads(snapshot)
    for value in (Int8(-2), UInt128(12), big(10)^50, Float16(0.5), Float32(0.1), 0.1, 3 // 7)
        @test Hammerhead._timing_decode(Hammerhead._timing_number(value)) == Rational{BigInt}(value)
    end
    for value in (true, NaN, Inf, BigFloat(0.5))
        @test_throws ArgumentError Hammerhead._timing_number(value)
    end
    # Safe-promoted arithmetic can describe a delay wider than machine integers.
    wide, _ = timing_source([typemin(Int64), typemax(Int64)])
    dt = BigInt(typemax(Int64)) - BigInt(typemin(Int64))
    pair = FramePair(FrameRef(wide, 1), FrameRef(wide, 2), dt)
    @test timing_exact(only(Hammerhead._timing_preflight([pair], nothing, 0.0, 0.0))["observed_delay"]) == dt
    @test timing_exact(only(Hammerhead._timing_preflight([pair], nothing, 0.0, 0.0))["sample_time"]) == -1 // 2
    # Existing image_pairs subtraction can wrap; capture refuses that declared delay.
    @test_throws ArgumentError Hammerhead._timing_preflight(image_pairs(wide), nothing, 0.0, 0.0)

    floating, _ = timing_source([0.1, 0.4])
    @test only(Hammerhead._timing_preflight(image_pairs(floating), nothing, 0.0, sqrt(eps(Float64))))["effective_agrees_with_observed"]
    @test_throws ArgumentError Hammerhead._timing_preflight(image_pairs(floating), nothing, 0.0, 0.0)
    within = FramePair(FrameRef(source, 1), FrameRef(source, 2), 3.000001)
    @test timing_exact(only(Hammerhead._timing_preflight([within], nothing, 1e-5, 0.0))["observed_delay"]) == 3
    @test_throws ArgumentError Hammerhead._timing_preflight([within], nothing, 0.0, 0.0)

    partial, _ = timing_source([0, nothing])
    p = only(Hammerhead._timing_preflight(image_pairs(partial), PhysicalScale(dt = 4), 0.0, 0.0))
    @test p["timestamp_status"] == "partial" && p["sample_time"] === nothing && p["observed_delay"] === nothing
    @test timing_exact(p["effective_delay"]) == 4 && p["effective_delay_provenance"] == "supplied_scale_dt"
    @test p["pair_time_unit"] === nothing && p["pair_clock_id"] === nothing
    a, _ = timing_source([0]; time_unit = "s", clock_id = "clock")
    b, _ = timing_source([1])
    unknown = only(Hammerhead._timing_preflight([(FrameRef(a, 1), FrameRef(b, 1))], nothing, 0.0, 0.0))
    @test unknown["time_unit_status"] == unknown["clock_status"] == "partial"
    @test unknown["pair_time_unit"] === nothing && unknown["pair_clock_id"] === nothing
    tuple_scale = only(Hammerhead._timing_preflight([(FrameRef(a, 1), FrameRef(b, 1))], PhysicalScale(dt = 2, time_unit = "s"), 0.0, 0.0))
    @test timing_exact(tuple_scale["observed_delay"]) == 1 && timing_exact(tuple_scale["effective_delay"]) == 2
    @test tuple_scale["effective_delay_provenance"] == "supplied_scale_dt" && !tuple_scale["effective_agrees_with_observed"]
    @test_throws ArgumentError Hammerhead._timing_preflight([(FrameRef(a, 1), FrameRef(b, 1))], PhysicalScale(time_unit = "ms"), 0.0, 0.0)
    c, _ = timing_source([1]; time_unit = "ms", clock_id = "other")
    @test_throws ArgumentError Hammerhead._timing_preflight([(FrameRef(a, 1), FrameRef(c, 1))], nothing, 0.0, 0.0)
    @test_throws ArgumentError Hammerhead._timing_freeze_pairs([TimingCustomPair(timing_image(), timing_image(), 1.0)])

    mktempdir() do directory
        output = joinpath(directory, "protected.jld2"); write(output, "unchanged")
        for ts in ([0, 1, 3, 2], [0, 1, 3, 3], [0, 1, NaN, 4], Any[0, 1, true, 4])
            bad, loaded = timing_source(ts)
            @test_throws ArgumentError run_piv_sequence(image_pairs(bad), timing_params(); output, record_pair_timing = true, progress = false)
            @test isempty(loaded) && read(output, String) == "unchanged"
        end
        for tolerance in ((-1.0, 0.0), (0.0, Inf), (true, 0.0))
            @test_throws ArgumentError run_piv_sequence(pairs, timing_params(); output, record_pair_timing = true,
                timing_atol = tolerance[1], timing_rtol = tolerance[2], progress = false)
            @test isempty(loads) && read(output, String) == "unchanged"
        end
    end
end

@testset "Pair timing sequence snapshots and persistence" begin
    mktempdir() do directory
        source, loads = timing_source([0, 1, 2, 4]; labels = ["a", "b", "c", "d"],
            source_id = "camera", frame_ids = ["f1", "f2", "f3", "f4"], time_unit = "s", clock_id = "clock")
        pairs = image_pairs(source)
        captured = PairTiming[]
        output = joinpath(directory, "timed.jld2")
        result = run_piv_sequence(pairs, timing_params(); output, record_pair_timing = true,
            on_pair_timing = (i, packet) -> begin
                push!(captured, packet)
                i == 1 && (source.timestamps[3] = 200; source.labels[3] = "rewritten"; source.frame_ids[3] = "changed")
            end, scale = PhysicalScale(pixel_size = 2, dt = 99, length_unit = "mm", time_unit = "s"), progress = false)
        @test length(result) == length(captured) == 2
        @test result[1].scale.dt == 1 && result[2].scale.dt == 2
        data = pair_timing_data(captured[2])
        @test timing_exact(data["frames"][1]["timestamp"]) == 2 && data["frames"][1]["label"] == "c"
        @test data["frames"][1]["frame_id"] == "f3"
        @test timing_exact(data["sample_time"]) == 3
        @test data["effective_delay_provenance"] == "pair_dt_scale_override"
        @test jldopen(f -> f["sources/000002"], output, "r") == ["c", "d"]
        @test jldopen(f -> f["pair_timing_format_version"], output, "r") === 1
        index = ResultFile(output)
        @test pair_timing_data(load_pair_timing(index, 2)) == data
        @test pair_timing_data(load_pair_timing(output, 2; verify_result = true)) == data
        @test timing_no_payloads(data)
        @test occursin("PairTiming", sprint(show, captured[2]))
        copy = pair_timing_data(captured[1]); copy["frames"][1]["label"] = "edited"
        @test pair_timing_data(captured[1])["frames"][1]["label"] == "a"
        forged = deepcopy(captured[1]); forged._data["scale_applied"] = 0
        @test_throws ArgumentError pair_timing_data(forged)
        malformed = pair_timing_data(captured[1]); malformed["scale"]["dt"] = 7.0
        forged = PairTiming(malformed, Hammerhead._experiment_digest(malformed))
        @test_throws ArgumentError pair_timing_data(forged)
        bare = joinpath(directory, "bare.jld2"); save_results(bare, result)
        @test load_pair_timing(bare, 1) === nothing

        # Callback edits to selected inner/outer containers cannot redirect later frames.
        frozen_source, loaded = timing_source(collect(0:5))
        inner = [[FrameRef(frozen_source, i), FrameRef(frozen_source, i+1)] for i in (1, 3, 5)]
        observed = PairTiming[]
        selected = Int[]
        run_piv_sequence(inner, timing_params(); collect_results = false, progress = false,
            output = (i, pair) -> (push!(selected, pair[1].index); @test pair isa Tuple; joinpath(directory, "frozen$i.jld2")),
            on_pair_timing = (i, packet) -> begin
                push!(observed, packet)
                i == 1 && (inner[3][1] = FrameRef(frozen_source, 1); inner[3][2] = FrameRef(frozen_source, 2))
            end)
        @test pair_timing_data(observed[3])["frames"][1]["frame_index"] == 5
        @test 5 in loaded && 6 in loaded
        @test selected == [1, 3, 5]

        per_source, _ = timing_source([0, 1, 2, 3])
        per_files = [joinpath(directory, "pair$i.jld2") for i in 1:2]
        returned = run_piv_sequence(image_pairs(per_source); effort = :low, collect_results = false,
            output = (i, pair) -> per_files[i], record_pair_timing = true, progress = false)
        @test returned === nothing
        @test pair_timing_data(load_pair_timing(per_files[2]; verify_result = true))["input_sequence_index"] == 2
        @test length(load_results(per_files[2])) == 1

        mutated = joinpath(directory, "mutated.jld2")
        @test_throws ArgumentError run_piv_sequence(image_pairs(per_source), timing_params(); output = mutated,
            record_pair_timing = true, progress = false, on_pair_timing = (i, p) -> (p._data["frames"][1]["label"] = "bad"))
        @test isempty(load_results(mutated))
        @test_throws ArgumentError run_piv_sequence(image_pairs(per_source), timing_params(); output = mutated,
            record_pair_timing = true, progress = false, on_result = (i, r) -> (r.u[1] += 1))
        @test isempty(load_results(mutated))
        @test_throws ArgumentError run_piv_sequence(image_pairs(per_source), timing_params(); record_pair_timing = true, progress = false)

        # Metadata-only reader can inspect a packet beside an unreadable result.
        bad_payload = joinpath(directory, "bad-payload.jld2")
        entry = jldopen(f -> f["pair_timing/000001"], output, "r")
        jldopen(bad_payload, "w") do f
            f["format_version"] = 1; f["results/000001"] = "not a result"
            f["pair_timing_format_version"] = 1; f["pair_timing/000001"] = entry
        end
        @test load_pair_timing(bad_payload) isa PairTiming
        @test_throws ArgumentError load_pair_timing(bad_payload; verify_result = true)
        @test_throws BoundsError load_pair_timing(index, 3)
        # Malformed native packets are rejected independently of checksum forgery.
        malformed_path = joinpath(directory, "malformed.jld2")
        for mutation in (
                e -> (e["result_key"] = "results/000002"),
                e -> (e["timing"]["pair_timing_format_version"] = 2),
                e -> (e["timing"]["scale_applied"] = 1),
                e -> (e["timing"]["sample_time"]["numerator"] = "0"),
                e -> (e["timing"]["frames"][1]["source_identity"] = "verified_content_hash"))
            bad_entry = deepcopy(entry); mutation(bad_entry)
            bad_entry["timing_sha256"] = Hammerhead._experiment_digest(bad_entry["timing"])
            jldopen(malformed_path, "w") do f
                f["format_version"] = 1; f["results/000001"] = result[1]
                f["pair_timing_format_version"] = 1; f["pair_timing/000001"] = bad_entry
            end
            @test_throws ArgumentError load_pair_timing(malformed_path)
        end
        for version in (2, 1.0, true)
            jldopen(malformed_path, "w") do f
                f["format_version"] = 1; f["results/000001"] = result[1]
                f["pair_timing_format_version"] = version; f["pair_timing/000001"] = entry
            end
            @test_throws ArgumentError load_pair_timing(malformed_path)
        end
        jldopen(malformed_path, "w") do f
            f["format_version"] = 1; f["results/000001"] = result[1]; f["pair_timing/000001"] = entry
        end
        @test_throws ArgumentError load_pair_timing(malformed_path)
        guarded = joinpath(directory, "guarded-output.jld2"); write(guarded, "preserved")
        bound_result = Ref{Any}(nothing)
        @test_throws ArgumentError run_piv_sequence(image_pairs(per_source), timing_params(); progress = false,
            on_pair_timing = (i, p) -> nothing, on_result = (i, r) -> (bound_result[] = r),
            output = (i, pair) -> (bound_result[].u[1] += 1; guarded))
        @test read(guarded, String) == "preserved"
        @test_throws ArgumentError run_piv_sequence(image_pairs(per_source), timing_params(); progress = false,
            on_measurement_history = (i, h) -> nothing, on_result = (i, r) -> (bound_result[] = r),
            output = (i, pair) -> (bound_result[].u[1] += 1; guarded))
        @test read(guarded, String) == "preserved"
        failed = joinpath(directory, "loader-failed.jld2")
        broken = FrameSource(2, i -> error("intentional loader failure"); timestamps = [0, 1])
        @test_throws ErrorException run_piv_sequence(image_pairs(broken), timing_params(); output = failed, record_pair_timing = true, progress = false)
        @test jldopen(f -> f["pair_timing_format_version"], failed, "r") === 1
        @test isempty(load_results(failed))
    end
    probe = timing_release_probe(); GC.gc(true)
    @test probe.returned === nothing
    @test all(r -> r.value === nothing, probe.result_refs)
    @test all(r -> r.value === nothing, probe.packet_refs)
end

@testset "Pair timing unsupported workflow preflight" begin
    mktempdir() do directory
        source, loads = timing_source([0, 1])
        pairs = image_pairs(source)
        output = joinpath(directory, "protected.jld2"); write(output, "preserved")
        @test_throws ArgumentError run_ptv_sequence(pairs; record_pair_timing = true, output, progress = false)
        @test_throws MethodError run_piv_ensemble(pairs, timing_params(); record_pair_timing = true, progress = false)
        @test_throws ArgumentError run_piv_ensemble(pairs; effort = :low, record_pair_timing = true, progress = false)
        grid = DewarpGrid(x = 2.0:1.0:25.0, y = 2.0:1.0:25.0)
        cameras = [PinholeCamera([64.0 0.0 slope 0.0; 0.0 64.0 0.0 0.0; 0.0 0.0 1.0 64.0]) for slope in (-8.0, 8.0)]
        dw1, dw2 = [ImageDewarper(cam, grid, (32,32)) for cam in cameras]
        @test_throws ArgumentError run_piv_stereo_sequence(pairs, pairs, dw1, dw2, timing_params(); record_pair_timing = true, output, progress = false)
        @test isempty(loads) && read(output, String) == "preserved"
        # Fixed replay/checkpoint keyword signatures reject before executing inputs.
        fixture = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
        record = ExperimentRecord([(joinpath(fixture,"A001_1.tif"), joinpath(fixture,"A001_2.tif"))], PIVRecipe(timing_params()))
        @test_throws MethodError replay_experiment(record; output, record_pair_timing = true)
        @test read(output, String) == "preserved"
        checkpoint = create_checkpoint(joinpath(directory, "checkpoint"), record; output_dir = joinpath(directory, "parts"))
        @test_throws MethodError resume_checkpoint!(checkpoint; record_pair_timing = true)
        @test isempty(readdir(joinpath(directory, "parts")))
    end
end
