using Test, Hammerhead, TOML, UUIDs

function quality_fixture(::Type{T} = Float64; masked = false, uncertainty = true) where T
    fields(values) = reshape(T.(values), 2, 3)
    mask = masked ? trues(2, 3) : BitMatrix(reshape(Bool[0, 0, 1, 0, 0, 0], 2, 3))
    PIVResult(T[1, 2, 3], T[1, 2], fields([1, 2, NaN, 3, Inf, 4]),
        fields([1, 2, NaN, 3, 5, 6]), ones(T, 2, 3), ones(T, 2, 3),
        fields([0.1, -0.2, NaN, 0.3, Inf, 0]), fields([0.1, 0.2, NaN, NaN, 0.4, 0.5]),
        BitMatrix(reshape(Bool[0, 1, 0, 1, 0, 0], 2, 3)), mask,
        PIVParameters(; window_size = 4, overlap = 2, uncertainty))
end
quality_groups(report) = quality_report_data(report)["groups"]

struct QualityOneShot
    used::Base.RefValue{Bool}
    released::Base.RefValue{Bool}
    previous::Base.RefValue{Any}
end
Base.IteratorSize(::Type{QualityOneShot}) = Base.SizeUnknown()
function Base.iterate(iterator::QualityOneShot, i = 1)
    if i == 1
        iterator.used[] && error("iterator cannot restart")
        iterator.used[] = true
    elseif i > 1
        GC.gc()
        # A for-loop can keep its current entry until the next iterate returns.
        # Older entries must already have been released during traversal.
        iterator.released[] &= all(reference -> reference.value === nothing,
                                  iterator.previous[][1:end-1])
    end
    i > 4 && return nothing
    result = quality_fixture()
    push!(iterator.previous[], WeakRef(result.u))
    result, i + 1
end

@testset "Saved run quality reports" begin
    fixture = quality_fixture()
    @testset "Measured counters, selections and dimensionless fractions" begin
        original = deepcopy(fixture)
        report = quality_report([fixture])
        group = quality_groups(report)["planar"]
        counts, fractions = group["counts"], group["fractions"]
        @test counts["entries"] == 1 && counts["nodes"] == 6
        @test counts["masked"] == 1 && counts["unmasked"] == 5
        @test counts["outlier_flagged_unmasked"] == 2
        @test counts["finite_output_unmasked"] == 4
        @test counts["unflagged_finite_output_unmasked"] == 2
        @test counts["flagged_finite_output_unmasked"] == 2
        @test counts["uq_u_available_unmasked"] == 3
        @test counts["uq_u_negative_finite_unmasked"] == 1
        @test counts["uq_u_nonfinite_unmasked"] == 1
        @test counts["uq_v_available_unmasked"] == 4
        @test counts["uq_all_available_unmasked"] == 2
        @test counts["unflagged_finite_output_with_uq_available"] == 2
        @test fractions["masked_fraction"]["value"] == 1 / 6
        @test fractions["current_outlier_flag_fraction"]["value"] == 2 / 5
        @test fractions["finite_output_fraction"]["value"] == 4 / 5
        @test fractions["unflagged_finite_output_fraction"]["value"] == 2 / 5
        @test fractions["stored_uq_all_numerically_available_fraction"]["value"] == 2 / 5
        @test fractions["stored_uq_on_unflagged_finite_output_fraction"]["value"] == 1
        @test fractions["stored_uq_unavailable_when_requested_fraction"]["value"] == 3 / 5
        @test all(metric -> metric["unit"] == "1", values(fractions))
        @test all(k -> isequal(getfield(fixture, k), getfield(original, k)), fieldnames(typeof(fixture)))
        scaled = with_scale(fixture, PhysicalScale(pixel_size = 0.02, dt = 0.001, length_unit = "mm", time_unit = "s"))
        @test quality_groups(quality_report([scaled])) == quality_groups(report)
        @test quality_groups(quality_report([quality_fixture(Float32)])) == quality_groups(report)
        detached = quality_report_data(report)
        detached["groups"]["planar"]["counts"]["nodes"] = 100
        @test quality_groups(report)["planar"]["counts"]["nodes"] == 6
        @test all(value -> !value["available"], values(quality_report_data(report)["unavailable"]))
        text = sprint(show, MIME"text/plain"(), report)
        @test occursin("not a replacement count", text)
        @test occursin("Uncertainty measurement association: unavailable (not persisted)", text)
        @test occursin("Accuracy: unavailable (not evaluated)", text)
        @test occursin("Current outlier flags: 40.0% [2 / 5 unmasked nodes]", text)
        @test occursin("Stored u uncertainty: 1 finite negative, 1 nonfinite", text)
        @test occursin("Replacement history: unavailable (not persisted)", text)
        @test occursin("node-weighted", text)
        @test occursin("RunQualityReport", sprint(show, report))
    end

    @testset "Node weighting, stereo, zero denominators and input rejection" begin
        small = PIVResult([1.], [1.], ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1),
                          zeros(1, 1), zeros(1, 1), falses(1, 1), falses(1, 1), fixture.parameters)
        weighted = quality_groups(quality_report([fixture, small]))["planar"]
        @test weighted["fractions"]["masked_fraction"]["value"] == 1 / 7
        @test weighted["fractions"]["finite_output_fraction"]["value"] == 5 / 6
        stereo = StereoPIVResult(copy(fixture.x), copy(fixture.y), 0., fixture.u, fixture.v,
            reshape([1., 2., NaN, 3., 4., NaN], 2, 3), fixture.uncertainty_u, fixture.uncertainty_v,
            reshape([0., 0., NaN, -0.1, 0., 0.], 2, 3), fixture.outliers, fixture.mask,
            fixture, fixture, fixture.parameters)
        mixed = quality_groups(quality_report([fixture, stereo]))
        @test Set(keys(mixed)) == Set(["planar", "stereo"])
        @test mixed["stereo"]["counts"]["nodes"] == 6
        @test mixed["stereo"]["counts"]["finite_output_unmasked"] == 3
        @test mixed["stereo"]["counts"]["uq_w_negative_finite_unmasked"] == 1
        @test mixed["stereo"]["fractions"]["stored_uq_w_numerically_available_fraction"]["value"] == 4 / 5
        all_masked = quality_groups(quality_report([quality_fixture(; masked = true)]))["planar"]
        @test all_masked["fractions"]["masked_fraction"]["value"] == 1
        unavailable = all_masked["fractions"]["finite_output_fraction"]
        @test !unavailable["available"] && !haskey(unavailable, "value")
        disabled = quality_groups(quality_report([quality_fixture(; uncertainty = false)]))["planar"]
        @test disabled["counts"]["uncertainty_requested_entries"] == 0
        @test !disabled["fractions"]["stored_uq_unavailable_when_requested_fraction"]["available"]
        @test disabled["counts"]["uq_all_available_unmasked"] == 2 # numeric fields, not parameter-derived validity
        @test isempty(quality_groups(quality_report(PIVResult[])))
        bad = PIVResult(fixture.x, fixture.y, fixture.u, fixture.v, fixture.peak_ratio,
            fixture.correlation_moment, fixture.uncertainty_u, zeros(1, 1), fixture.outliers,
            fixture.mask, fixture.parameters)
        @test_throws DimensionMismatch quality_report([bad])
        @test_throws ArgumentError quality_report(["not a result"])
        @test_throws ArgumentError quality_report(Iterators.repeated(fixture))
        oneshot = QualityOneShot(Ref(false), Ref(true), Ref{Any}(WeakRef[]))
        @test quality_groups(quality_report(oneshot))["planar"]["counts"]["entries"] == 4
        @test oneshot.released[]
        GC.gc()
        @test all(reference -> reference.value === nothing, oneshot.previous[])
    end

    mktempdir() do dir
        @testset "Lazy files, bounded reports and safe TOML roundtrip" begin
            source = joinpath(dir, "results.jld2")
            save_results(source, fill(fixture, 12))
            index = ResultFile(source)
            report = quality_report(index)
            provenance = quality_report_data(report)["provenance"]
            @test provenance["association"] == "unassociated"
            @test provenance["source_selection"] == "whole_file"
            @test provenance["source_sha256"] == Hammerhead._experiment_file_digest(source)
            @test quality_groups(report)["planar"]["counts"]["entries"] == 12
            @test Base.summarysize(report) < 20000
            subset = quality_report(view(index, 2:2:8))
            @test quality_groups(subset)["planar"]["counts"]["entries"] == 4
            @test quality_report_data(subset)["provenance"]["source_selection"] == "provided_array"
            output = joinpath(dir, "quality.toml")
            @test save_quality_report(output, report) == output
            loaded = load_quality_report(output)
            @test quality_report_data(loaded) == quality_report_data(report)
            @test TOML.parsefile(output)["quality_report_format_version"] == 1
            original_bytes = read(source)
            @test_throws ArgumentError save_quality_report(source, report)
            @test_throws ArgumentError save_quality_report(source, subset)
            @test_throws ArgumentError save_quality_report(source, loaded)
            # Positional construction cannot omit validated metadata's paths.
            data = quality_report_data(report)
            manually_wrapped = RunQualityReport(data, (), Hammerhead._experiment_digest(data))
            @test_throws ArgumentError save_quality_report(source, manually_wrapped)
            @test read(source) == original_bytes
            alias = joinpath(dir, "source-alias.jld2")
            linked = try hardlink(source, alias); true catch; false end
            if linked
                @test_throws ArgumentError save_quality_report(alias, report)
                @test read(source) == original_bytes
            end
            anonymous = quality_report((result for result in index))
            @test !haskey(quality_report_data(anonymous)["provenance"], "source_path")
            @test_throws ArgumentError save_quality_report(source, anonymous; protected_paths = [source])
            save_results(source, [fixture])
            @test_throws ArgumentError quality_report(index)
        end

        @testset "Malformed reports reject before destination replacement" begin
            data = quality_report_data(quality_report([fixture]))
            malformed = joinpath(dir, "malformed.toml")
            function rejected(change)
                candidate = deepcopy(data)
                change(candidate)
                open(malformed, "w") do io
                    TOML.print(io, candidate)
                end
                @test_throws ArgumentError load_quality_report(malformed)
            end
            rejected(d -> d["quality_report_format_version"] = 99)
            rejected(d -> d["quality_report_format_version"] = true)
            rejected(d -> d["generated_at_unix_s"] = true)
            rejected(d -> d["generated_at_unix_s"] = Inf)
            rejected(d -> d["generator"]["core_source_sha256"] = "bad")
            rejected(d -> d["groups"]["planar"]["counts"]["nodes"] = true)
            rejected(d -> d["groups"]["planar"]["fractions"]["masked_fraction"]["value"] = 0.5)
            rejected(d -> d["groups"]["planar"]["fractions"]["masked_fraction"]["numerator"] = true)
            rejected(d -> d["groups"]["planar"]["fractions"]["masked_fraction"]["available"] = 1)
            rejected(d -> d["unavailable"]["replacement_history"]["available"] = true)
            rejected(d -> d["unavailable"]["replacement_history"]["available"] = 0)
            rejected(d -> d["groups"]["planar"]["counts"]["masked"] = typemax(Int))
            rejected(d -> begin
                c = d["groups"]["planar"]["counts"]
                c["uq_u_available_unmasked"] = typemax(Int)
                c["uq_u_negative_finite_unmasked"] = typemax(Int)
                c["uq_u_nonfinite_unmasked"] = 7
            end)
            report = quality_report([fixture])
            sentinel = joinpath(dir, "preserved.toml"); write(sentinel, "preserved")
            report._data["groups"]["planar"]["counts"]["nodes"] = 99
            @test_throws ArgumentError save_quality_report(sentinel, report)
            @test read(sentinel, String) == "preserved"
        end

        @testset "Verified experiment/run association and protected dependencies" begin
            fixture_directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
            files = sort(filter(p -> endswith(lowercase(p), ".tif"), readdir(fixture_directory; join = true)))
            script_path = joinpath(dir, "prepare.jl"); write(script_path, "identity(image)\n")
            recipe = PIVRecipe(fixture.parameters; external_preprocess = ScriptReference(script_path; entrypoint = "identity(image)"))
            record = ExperimentRecord([(files[1], files[2])], recipe)
            output = joinpath(dir, "associated-results.jld2"); save_results(output, [fixture])
            now = time()
            run = ExperimentRun(string(uuid4()), recipe_identity(recipe), record.input_id, now, now,
                :completed, 1, abspath(output), Hammerhead._experiment_file_digest(output),
                deepcopy(record.creation_environment), nothing)
            push!(record.runs, run)
            record_path = joinpath(dir, "experiment.jld2"); save_experiment(record_path, record)
            report = quality_report(record, run)
            data = quality_report_data(report)
            @test data["provenance"]["association"] == "recorded_output_verified"
            @test data["provenance"]["recipe_id"] == recipe_identity(recipe)
            @test data["provenance"]["input_id"] == record.input_id
            @test data["provenance"]["run_id"] == run.run_id
            @test length(data["provenance"]["run_environment_id"]) == 64
            for path in (files[1], files[2], script_path, record_path, output)
                original_bytes = read(path)
                @test_throws ArgumentError save_quality_report(path, report)
                @test read(path) == original_bytes
            end
            @test save_quality_report(joinpath(dir, "associated-quality.toml"), report) isa String
            # Reporting persisted output does not reopen/evaluate the script.
            rm(script_path)
            @test quality_groups(quality_report(record, run)) == quality_groups(report)
            mismatched = ExperimentRun(run.run_id, repeat("a", 64), run.input_id, now, now,
                :completed, 1, run.output, run.output_sha256, run.environment, nothing)
            @test_throws ArgumentError quality_report(record, mismatched)
            failed = ExperimentRun(run.run_id, run.recipe_id, run.input_id, now, now,
                :failed, 1, run.output, run.output_sha256, run.environment, "failure")
            @test_throws ArgumentError quality_report(record, failed)
            bad_input = deepcopy(record); bad_input.pairs[1] = [2, 1]
            @test_throws ArgumentError quality_report(bad_input, run)
            save_results(output, [fixture, fixture])
            @test_throws ArgumentError quality_report(record, run)
            save_results(output, [quality_fixture(Float32)])
            @test_throws ArgumentError quality_report(record, run)
        end
    end
end
