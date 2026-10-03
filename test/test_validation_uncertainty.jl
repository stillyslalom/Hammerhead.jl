using Test, Hammerhead, TOML
include(joinpath(@__DIR__, "..", "bench", "validation_uncertainty.jl"))
struct UncertaintyTestCustomValidator <: Hammerhead.PIVValidator end

function uncertainty_test_result(u, v, su, sv; mask = falses(size(u)), flags = falses(size(u)),
        parameters = PIVParameters(window_size = 4, overlap = 2, n_peaks = 1,
            replace_outliers = false, uncertainty = true))
    PIVResult(collect(1.0:size(u, 2)), collect(1.0:size(u, 1)), u, v,
        ones(size(u)), ones(size(u)), su, sv, BitMatrix(flags), BitMatrix(mask), parameters)
end
function uncertainty_no_payloads(value)
    value isa AbstractDict && return all(uncertainty_no_payloads, values(value))
    value isa AbstractArray && return ndims(value) == 1 && all(uncertainty_no_payloads, value)
    value isa Union{Number,AbstractString}
end
Base.@noinline function uncertainty_transient_result()
    r = uncertainty_test_result(ones(2, 2), zeros(2, 2), ones(2, 2), ones(2, 2))
    refs = (WeakRef(r.u), WeakRef(r.mask), WeakRef(r.uncertainty_u))
    m = ValidationUncertainty.metrics(r, (x, y) -> (0.0, 0.0))
    (; refs, population = m.population, data = m.data)
end

@testset "Synthetic uncertainty metric contracts" begin
    U = ValidationUncertainty
    row(v) = reshape(Float64.(v), 1, :)
    truth = (x, y) -> (0.0, 0.0)
    result = uncertainty_test_result(row([0, 2, -3, 4, 5, 6, 7, 8]), zeros(1, 8),
        row([0, 0, 1, 2, NaN, -1, Inf, 1]), ones(1, 8))
    measured = U.metrics(result, truth)
    data = measured.data; saved_data = deepcopy(data)
    u = data["components"]["u"]; v = data["components"]["v"]
    c = u["counts"]
    @test data["counts"]["primary_valid"] == 8 && data["valid_yield"]["fraction"] == 1
    @test c["primary_valid"] == c["uq_available"] + c["uq_nonfinite"] + c["uq_negative"] == 8
    @test c["uq_available"] == c["sigma_zero"] + c["sigma_positive"] == 5
    @test c["uq_nonfinite"] == 2 && c["uq_negative"] == 1
    @test c["sigma_zero"] == 2 && c["sigma_positive"] == 3 && c["sigma_above_0_3"] == 3
    @test c["zero_sigma_zero_error"] == c["zero_sigma_nonzero_error"] == 1
    @test c["zero_sigma_error_arithmetic_unavailable"] == 0
    @test u["primary_error"]["mean"] ≈ 29 / 8
    @test u["primary_error"]["rms"] ≈ sqrt(203 / 8)
    @test u["uq_subset_error"]["count"] == 5 && u["uq_subset_error"]["mean"] ≈ 11 / 5
    @test u["uq_subset_error"]["rms"] ≈ sqrt(93 / 5)
    @test u["coverage"]["1"]["fraction"] == 1 / 5
    @test u["coverage"]["2"]["fraction"] == 2 / 5
    @test u["normalized_error"]["count"] == 3
    @test u["normalized_error"]["mean"] ≈ 7 / 3
    @test u["normalized_error"]["rms"] ≈ sqrt(77 / 3)
    @test u["normalized_error_quantiles"]["q50"] == 2
    @test v["counts"]["uq_available"] == 8 && v["coverage"]["1"]["fraction"] == 1
    @test uncertainty_no_payloads(data)

    # Exhaustive primary partition; the two marginal rejection counts overlap.
    excluded = uncertainty_test_result(row([1, 999, NaN, NaN, 999]), zeros(1, 5), ones(1, 5), ones(1, 5);
        mask = reshape(Bool[0, 1, 0, 0, 0], 1, :), flags = reshape(Bool[0, 0, 0, 1, 1], 1, :))
    calls = Ref(0)
    m = U.metrics(excluded, (x, y) -> (calls[] += 1; (0.0, 0.0))).data
    cc = m["counts"]
    @test calls[] == 1
    @test cc["grid_nodes"] == cc["masked"] + cc["unmasked"] == 5
    @test cc["unmasked"] == cc["primary_valid"] + cc["flagged_finite"] + cc["flagged_nonfinite"] + cc["unflagged_nonfinite"] == 4
    @test cc["flagged_unmasked"] == 2 && cc["nonfinite_output_unmasked"] == 2
    @test m["valid_yield"]["fraction"] == 1 / 4

    # Subtraction overflow retains finite measured/truth nodes and all denominators.
    huge = floatmax(Float64)
    overflow = uncertainty_test_result(row([huge, 1]), zeros(1, 2), ones(1, 2), ones(1, 2))
    o = U.metrics(overflow, (x, y) -> (-huge, 0.0)).data
    ou = o["components"]["u"]
    @test o["counts"]["primary_valid"] == 2
    @test !ou["primary_error"]["available"] && ou["primary_error"]["count"] == 2
    @test ou["primary_error"]["finite_count"] == 1
    @test !ou["coverage"]["1"]["available"] && ou["coverage"]["1"]["denominator"] == 2
    @test ou["counts"]["error_arithmetic_unavailable"] == ou["counts"]["uq_error_arithmetic_unavailable"] == 1
    @test ou["normalized_error"]["count"] == 2 && !ou["normalized_error"]["available"]
    @test !ou["normalized_error_quantiles"]["available"]
    @test o["components"]["v"]["coverage"]["1"]["fraction"] == 1

    # Finite-error quotient overflow is certainly uncovered, but z moments are unavailable.
    tiny = uncertainty_test_result(ones(1, 1), zeros(1, 1), fill(nextfloat(0.0), 1, 1), ones(1, 1))
    t = U.metrics(tiny, truth).data["components"]["u"]
    @test t["coverage"]["2"]["fraction"] == 0 && t["coverage"]["2"]["denominator"] == 1
    @test t["counts"]["normalized_arithmetic_unavailable"] == 1
    @test !t["normalized_error"]["available"]
    enormous = uncertainty_test_result(fill(huge, 1, 1), zeros(1, 1), fill(huge, 1, 1), ones(1, 1))
    @test U.metrics(enormous, truth).data["components"]["u"]["coverage"]["2"]["fraction"] == 1
    zero_overflow = uncertainty_test_result(fill(huge, 1, 1), zeros(1, 1), zeros(1, 1), ones(1, 1))
    @test U.metrics(zero_overflow, (x, y) -> (-huge, 0.0)).data["components"]["u"]["counts"]["zero_sigma_error_arithmetic_unavailable"] == 1

    @test_throws ArgumentError U.metrics(result, (x, y) -> (NaN, 0.0))
    custom = PIVParameters(window_size = 4, overlap = 2, n_peaks = 1,
        replace_outliers = false, uncertainty = true, validation = (UncertaintyTestCustomValidator(),))
    @test_throws ArgumentError U.metrics(uncertainty_test_result(ones(1, 1), zeros(1, 1), ones(1, 1), ones(1, 1); parameters = custom), truth)
    for overrides in ((n_peaks = 3,), (replace_outliers = true,), (max_iterations = 2,), (uncertainty = false,))
        kwargs = merge((; window_size = 4, overlap = 2, n_peaks = 1, replace_outliers = false, uncertainty = true), overrides)
        p = PIVParameters(; kwargs...)
        @test_throws ArgumentError U.metrics(uncertainty_test_result(ones(1, 1), zeros(1, 1), ones(1, 1), ones(1, 1); parameters = p), truth)
    end
    invalid_dims = uncertainty_test_result(ones(1, 2), zeros(1, 2), ones(1, 1), ones(1, 2))
    @test_throws ArgumentError U.metrics(invalid_dims, truth)
    all_masked = uncertainty_test_result(ones(1, 1), zeros(1, 1), ones(1, 1), ones(1, 1); mask = trues(1, 1))
    @test !U.metrics(all_masked, truth).data["valid_yield"]["available"]
    @test U.metrics(all_masked, truth).data["components"]["u"]["primary_error"]["reason"] == "empty_population"

    pool = U.Population()
    U.merge!(pool, measured.population); U.merge!(pool, measured.population)
    pooled = U.summary(pool)
    @test pooled["counts"]["primary_valid"] == 16
    @test pooled["components"]["u"]["normalized_error"]["mean"] ≈ u["normalized_error"]["mean"]
    @test pooled["components"]["u"]["primary_error"]["rms"] ≈ u["primary_error"]["rms"]
    @test !haskey(pooled["components"]["u"], "normalized_error_quantiles")
    @test measured.data == saved_data # merging never alters the per-seed snapshot
    single = U.metrics(uncertainty_test_result(fill(20.0, 1, 1), zeros(1, 1), ones(1, 1), ones(1, 1)), truth)
    U.merge!(pool, single.population)
    @test U.summary(pool)["components"]["u"]["primary_error"]["mean"] ≈ (58 + 20) / 17
    stable = U.Moment(); U.add!(stable, -1e300); U.add!(stable, 1e300)
    @test U.summary(stable)["mean"] == 0 && U.summary(stable)["rms"] ≈ 1e300
    first_moment = U.Moment(); U.add!(first_moment, 1e300)
    second_moment = U.Moment(); U.add!(second_moment, -1e300)
    U.merge!(first_moment, second_moment)
    @test U.summary(first_moment) == U.summary(stable)
    @test U.quantiles([-1e308, 1e308], 2)["q50"] == 0
    weak = uncertainty_transient_result(); GC.gc(true)
    @test all(r -> r.value === nothing, weak.refs)
    @test U.summary(weak.population)["counts"]["primary_valid"] == 4

    @test length(U.conditions()) == 4 && length(U.conditions(true)) == 12
    @test length(U.DEFAULT_SEEDS) == 3 && length(U.EXPANDED_SEEDS) == 8
    passes = U.controlled_passes()
    @test last(passes).n_peaks == 1 && !last(passes).replace_outliers && last(passes).max_iterations == 1
    @test last(passes).uncertainty && !first(passes).uncertainty
    @test_throws ArgumentError U.main(["--unknown"])
    @test_throws ArgumentError U.main(["--samples=0"])
    @test_throws ArgumentError U.main(["--samples=11"])
    @test U.main(["--help"]) === nothing
    @test_throws ArgumentError U.check_output(joinpath(U.ROOT, "test", "reference_images"))
end

@testset "Uncertainty report output contracts" begin
    U = ValidationUncertainty
    r = uncertainty_test_result(ones(1, 2), zeros(1, 2), ones(1, 2), ones(1, 2))
    pooled = U.summary(U.metrics(r, (x, y) -> (0.0, 0.0)).population)
    empty_result = uncertainty_test_result(fill(NaN, 1, 2), fill(NaN, 1, 2), fill(NaN, 1, 2), fill(NaN, 1, 2))
    empty = Dict("status" => "no_valid_measurements", "metrics" => U.metrics(empty_result, (x, y) -> (0.0, 0.0)).data)
    report = Dict("schema_version" => U.SCHEMA, "provenance_status" => "test only", "groups" => [Dict("condition" => "test", "pooled" => pooled)],
        "empty_failure" => empty, "limitations" => ["test only"], "unsupported" => Dict("known_motion" => "not supplied"))
    alias_parent = mkpath(joinpath(U.ROOT, "bench", "profile-output"))
    mktempdir(alias_parent) do directory
        paths = U.write_report(joinpath(directory, "report"), report)
        loaded = TOML.parsefile(first(paths))
        @test loaded["groups"][1]["pooled"] == pooled
        @test occursin("without a Gaussian", read(paths[2], String))
        @test U.write_report(joinpath(directory, "report"), report) == paths
        unrelated = joinpath(directory, "unrelated"); mkpath(unrelated)
        write(joinpath(unrelated, "uncertainty.md"), "user content")
        @test_throws ArgumentError U.write_report(unrelated, report)
        @test !isfile(joinpath(unrelated, "uncertainty.toml"))
        @test read(joinpath(unrelated, "uncertainty.md"), String) == "user content"
        alias_dir = joinpath(directory, "alias"); mkpath(alias_dir)
        link = joinpath(alias_dir, "uncertainty.md")
        source = joinpath(U.ROOT, "bench", "validation_scorecard.jl")
        saved = read(source)
        @test U.check_output(alias_dir) isa Vector
        hardlink(source, link)
        @test Base.samefile(source, link)
        error = try U.write_report(alias_dir, report); nothing catch caught; caught end
        @test error isa ArgumentError && occursin("aliases source", sprint(showerror, error))
        @test read(source) == saved
        dangling_dir = joinpath(directory, "dangling"); mkpath(dangling_dir)
        dangling = joinpath(dangling_dir, "uncertainty.toml")
        escaped = joinpath(directory, "escaped.toml")
        try
            symlink(escaped, dangling)
            @test_throws ArgumentError U.write_report(dangling_dir, report)
            @test !ispath(escaped)
        catch error
            error isa Base.IOError || rethrow()
        end
        outside = joinpath(directory, "directory-link")
        try
            symlink(joinpath(U.ROOT, "test", "reference_images"), outside; dir_target = true)
            @test_throws ArgumentError U.check_output(outside)
        catch error
            error isa Base.IOError || rethrow()
        end
    end
end

if get(ENV, "HAMMERHEAD_UQ_HELPERS_ONLY", "0") != "1"
    @testset "Seeded uncertainty evaluation and report" begin
        U = ValidationUncertainty
        condition = (id = "test_noise", settings = (; noise = 0.03), window = 16)
        a = U.evaluate(condition, 7321; size = 64)
        repeated = U.evaluate(condition, 7321; size = 64, samples = 2)
        b = U.evaluate(condition, 7322; size = 64)
        @test a.row["inputs"] == repeated.row["inputs"]
        @test a.row["recipe"] == repeated.row["recipe"]
        @test a.row["metrics"] == repeated.row["metrics"]
        @test a.row["diagnostics"] == repeated.row["diagnostics"]
        @test a.row["inputs"]["image_a_sha256"] != b.row["inputs"]["image_a_sha256"]
        @test a.row["metrics"]["counts"]["primary_valid"] > 0
        @test a.row["metrics"]["components"]["u"]["primary_error"]["rms"] < 0.25
        @test a.row["diagnostics"]["passes"][end]["tolerance_checks"] == 0
        @test a.row["diagnostics"]["passes"][end]["residual"]["value_basis"] == "primary_peak_before_validation"
        @test occursin("not proven", a.row["diagnostics"]["assumption_status"])
        @test length(repeated.row["performance"]["runtime_seconds_samples"]) == 2
        @test all(>=(0), repeated.row["performance"]["julia_allocated_bytes_samples"])
        @test occursin("NOT peak", a.row["performance"]["allocation_definition"])
        pool = U.Population(); U.merge!(pool, a.population); U.merge!(pool, b.population)
        pooled = U.summary(pool)
        @test pooled["counts"]["primary_valid"] == a.row["metrics"]["counts"]["primary_valid"] + b.row["metrics"]["counts"]["primary_valid"]
        @test uncertainty_no_payloads(a.row) && uncertainty_no_payloads(pooled)
        empty = U.evaluate((id = "empty", settings = (; density = 0.0), window = 16), 7321; size = 64).row
        @test empty["status"] == "no_valid_measurements" && empty["metrics"]["counts"]["primary_valid"] == 0
        @test !empty["metrics"]["components"]["u"]["coverage"]["1"]["available"]
    end
end
