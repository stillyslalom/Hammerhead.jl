include(joinpath(@__DIR__, "..", "bench", "validation_scorecard.jl"))

@testset "Validation scorecard contracts" begin
    V = ValidationScorecard
    make_result(u, v; mask = falses(size(u)), outliers = falses(size(u))) =
        PIVResult(collect(1.0:size(u, 2)), collect(1.0:size(u, 1)), u, v,
            ones(size(u)), ones(size(u)), fill(NaN, size(u)), fill(NaN, size(u)),
            outliers, mask, PIVParameters())
    mask = falses(2, 2); mask[2, 1] = true
    flags = falses(2, 2); flags[1, 2] = true
    result = make_result([1.0 999.0; 100.0 NaN], zeros(2, 2); mask, outliers = flags)
    m = V.metrics(result; truth = (x, y) -> (0.0, 0.0))
    @test m["eligible_count"] == 3 && m["valid_count"] == 1
    @test m["valid_yield"] == 1 / 3
    @test m["outlier_count_unmasked"] == 1 && m["nonfinite_count_unmasked"] == 1
    @test m["bias"]["u"] == 1.0 && m["rms_error"]["u"] == 1.0
    @test occursin("replacement", m["selection"])
    population = V.metrics(make_result(reshape([-1.0, 3.0], 1, 2), zeros(1, 2));
                           truth = (x, y) -> (0.0, 0.0))
    @test population["bias"]["u"] == 1.0
    @test population["rms_error"]["u"] ≈ sqrt(5.0) # error RMS includes bias
    smoke = V.metrics(result)
    @test !haskey(smoke, "bias") && !haskey(smoke, "rms_error")
    @test occursin("no displacement truth", smoke["error_status"])
    no_valid = V.metrics(make_result(fill(NaN, 2, 2), zeros(2, 2));
                         truth = (x, y) -> (0.0, 0.0))
    @test no_valid["valid_count"] == 0 && no_valid["valid_yield"] == 0.0
    @test !haskey(no_valid, "bias") && occursin("no valid vectors", no_valid["error_status"])
    @test_throws ArgumentError V.metrics(result; truth = (x, y) -> (NaN, 0.0))

    scene = V.synthetic_scene(size = 64, shear = 0.025, noise = 0.01, dropout = 0.2)
    repeated = V.synthetic_scene(size = 64, shear = 0.025, noise = 0.01, dropout = 0.2)
    @test scene.a == repeated.a && scene.b == repeated.b
    @test scene.spec == repeated.spec
    @test scene.spec["image_a_sha256"] == V.pixel_digest(scene.a)
    @test length(scene.spec["image_a_sha256"]) == 64
    @test scene.spec["image_a_sha256"] != V.synthetic_scene(size = 64, seed = 7322).spec["image_a_sha256"]
    # Analytic inverse uses constant truth dv, never the measured u/v.
    y_launch = 27.0
    y_midpoint = y_launch - 1.5 / 2
    @test scene.truth(19.0, y_midpoint) == (2.25 + 0.025 * (y_launch - 32.5), -1.5)
    @test scene.truth(19.0, y_midpoint) != (2.25 + 0.025 * (y_midpoint - 32.5), -1.5)

    passes = multipass_parameters([32, 16, 16]; padding = true, apodization = :gauss)
    recipe = V.recipe(passes; roi = [1, 64, 1, 64])
    @test length(recipe["passes"]) == 3
    @test Set(keys(recipe["passes"][1])) == Set(String.(fieldnames(PIVParameters)))
    @test recipe["passes"][1]["window_size"] == [32, 32]
    @test recipe["passes"][3]["apodization"] == "gauss"
    @test recipe["roi"] == [1, 64, 1, 64] && recipe["backend"] == "cpu"

    row = V.evaluate(() -> V.synthetic_case("test_translation", (; size = 64)); samples = 2)
    repeated_row = V.evaluate(() -> V.synthetic_case("test_translation", (; size = 64)); samples = 1)
    @test row["inputs"] == repeated_row["inputs"]
    @test row["recipe"] == repeated_row["recipe"]
    @test row["metrics"] == repeated_row["metrics"]
    @test row["category"] == "synthetic_ground_truth" && row["status"] == "measured"
    @test row["metrics"]["valid_count"] > 0
    @test row["metrics"]["rms_error"]["u"] < 0.25
    perf = row["performance"]
    @test length(perf["runtime_seconds_samples"]) == 2
    @test all(>=(0), perf["julia_allocated_bytes_samples"])
    @test perf["runtime_seconds_min"] <= perf["runtime_seconds_median"] <= perf["runtime_seconds_max"]
    @test occursin("NOT peak", perf["allocation_definition"])
    @test occursin("excludes generation/loading/calibration", perf["scope"])
    @test_throws ArgumentError V.evaluate(() -> error("must not prepare"); samples = 0)

    mktempdir() do dir
        identity_path = joinpath(dir, "identity.txt")
        write(identity_path, "abc")
        @test V.file_digest(identity_path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        report = Dict("schema_version" => V.SCHEMA, "cases" => [row], "limitations" => ["test only"])
        out = joinpath(dir, "reports")
        paths = V.write_report(out, report)
        restored = V.TOML.parsefile(paths[1])
        @test restored["cases"][1]["recipe"] == row["recipe"]
        @test restored["cases"][1]["metrics"] == row["metrics"]
        @test occursin("not peak memory", read(paths[2], String))
        @test V.write_report(out, report) == paths # owned reports may be regenerated
        unrelated = joinpath(dir, "unrelated")
        mkpath(unrelated)
        write(joinpath(unrelated, "scorecard.md"), "user content")
        @test_throws ArgumentError V.write_report(unrelated, report)
        @test read(joinpath(unrelated, "scorecard.md"), String) == "user content"
        @test !isfile(joinpath(unrelated, "scorecard.toml"))
    end
    @test_throws ArgumentError V.write_report(joinpath(V.ROOT, "test", "reference_images"), Dict())
    @test_throws ArgumentError V.main(["--unknown"])
    @test_throws ArgumentError V.main(["--samples=0"])
    @test V.main(["--help"]) === nothing
end
