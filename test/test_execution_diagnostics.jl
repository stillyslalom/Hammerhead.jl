using Test, Hammerhead, Random, JLD2

function execution_same_result(a, b)
    all(k -> isequal(getfield(a, k), getfield(b, k)),
        (:x, :y, :u, :v, :peak_ratio, :correlation_moment, :uncertainty_u,
         :uncertainty_v, :outliers, :mask, :correlation_planes))
end

@testset "Actual planar execution diagnostics" begin
    rng = MersenneTwister(831)
    A = rand(rng, 48, 48); B = circshift(A, (1, 2))
    base = (; window_size = 16, overlap = 8, padding = true, uod_enable = false,
             validation = (), replace_outliers = false)
    @testset "Counts, stopping and numerical equivalence" begin
        for T in (Float32, Float64), backend in (:cpu, :ka), maxiter in (1, 2, 4)
            p = PIVParameters(; base..., max_iterations = maxiter, convergence_tol = 1e6, uncertainty = true)
            a, b = T.(A), T.(B)
            plain = run_piv(a, b, p; backend, threaded = false)
            captured = Ref{Any}(nothing); calls = Ref(0)
            measured = run_piv(a, b, p; backend, threaded = false,
                on_diagnostics = d -> (captured[] = d; calls[] += 1))
            @test calls[] == 1
            @test execution_same_result(plain, measured)
            d = captured[]; pass = only(d.passes)
            @test d.backend == backend && d.image_type == string(T)
            @test d.processing_size == (48, 48) && d.pair_index === nothing && d.association === nothing
            @test pass.requested_iterations == maxiter
            @test pass.executed_iterations == (maxiter == 4 ? 2 : maxiter)
            @test pass.checks == (maxiter == 4 ? 1 : 0)
            @test pass.stop_reason == (maxiter == 1 ? :single_sweep : maxiter == 2 ? :iteration_budget : :tolerance_condition_met)
            @test pass.residual.finite_count + pass.residual.nonfinite_count + pass.residual.masked_count == length(measured.u)
            @test pass.residual.unit == "px" && pass.residual.value_basis == "primary_peak_before_validation"
            @test isimmutable(d) && d.passes isa Tuple && pass.residual isa NamedTuple
            data = execution_diagnostics_data(d); data["passes"][1]["executed_iterations"] = 999
            @test pass.executed_iterations != 999
            @test occursin("not measurement validity", sprint(show, MIME"text/plain"(), d))
        end
        for tol in (0., 1e-30)
            p = PIVParameters(; base..., max_iterations = 3, convergence_tol = tol)
            d = Ref{Any}(nothing)
            run_piv(A, rand(rng, size(A)...), p; threaded = false, on_diagnostics = x -> (d[] = x))
            pass = only(d[].passes)
            @test pass.executed_iterations == 3 && pass.stop_reason === :iteration_budget
            @test pass.checks == (tol == 0 ? 0 : 1)
            @test tol == 0 ? pass.last_check === nothing : pass.last_check.sweep == 2
        end
        infinite_tolerance = Ref{Any}(nothing)
        run_piv(A, B, PIVParameters(; base..., max_iterations = 4, convergence_tol = Inf);
            on_diagnostics = x -> (infinite_tolerance[] = x))
        @test only(infinite_tolerance[].passes).requested_tolerance == Inf
        @test only(infinite_tolerance[].passes).executed_iterations == 2
        @test execution_diagnostics_data(infinite_tolerance[])["passes"][1]["requested_tolerance"] == Inf
        p = PIVParameters(; base..., max_iterations = 4, convergence_tol = 1.)
        d = Ref{Any}(nothing)
        flat = run_piv(zeros(48, 48), zeros(48, 48), p; threaded = false, on_diagnostics = x -> (d[] = x))
        pass = only(d[].passes)
        @test pass.executed_iterations == 2 && pass.stop_reason === :tolerance_condition_met
        @test pass.last_check.included_count == 0 && pass.last_check.value == 0
        @test pass.last_check.tolerance_met && all(flat.outliers)
        @test pass.residual.finite_count == 0 && pass.residual.mean_magnitude === nothing
        @test occursin("empty support", sprint(show, MIME"text/plain"(), d[]))
        # Exact existing nonfinite rules: unchanged Inf differences are omitted;
        # finite/nonfinite transitions contribute Inf; all-NaN patterns omit.
        u = reshape([Inf, NaN, 1., 2.], 2, 2); v = zeros(2, 2)
        previous = reshape([Inf, NaN, NaN, 1.], 2, 2)
        buffer = Float64[]; mask = falses(2, 2)
        value = Hammerhead.field_change(buffer, u, v, previous, v, mask)
        observer = Hammerhead._PassObservation(4)
        Hammerhead._execution_check!(observer, 2, value, buffer, mask, 1.)
        @test observer.last_check.included_count == 2
        @test observer.last_check.finite_count == 1 && observer.last_check.infinite_count == 1
        @test observer.last_check.excluded_count == 2 && !observer.last_check.tolerance_met
        @test observer.last_check.value_state in (:infinite, :nan)
    end

    @testset "Raw primary residual, masks, ROI and scale" begin
        analytic = Hammerhead._execution_residual(reshape([0., 3., 5., NaN, 99., 1.], 2, 3),
            reshape([0., 4., 12., 2., 99., 2.], 2, 3),
            BitMatrix(reshape([false, false, false, false, true, true], 2, 3)), false)
        @test analytic.finite_count == 3 && analytic.nonfinite_count == 1 && analytic.masked_count == 2
        @test analytic.mean_magnitude == 6
        @test analytic.rms_magnitude ≈ sqrt((0 + 25 + 169) / 3)
        @test analytic.maximum_magnitude == 13
        p = PIVParameters(; base...)
        d = Ref{Any}(nothing)
        raw = run_piv(A, B, p; threaded = false, on_diagnostics = x -> (d[] = x))
        expected = Hammerhead._execution_residual(raw.u, raw.v, raw.mask, false)
        @test only(d[].passes).residual == expected
        magnitudes = sort(vec(hypot.(raw.u, raw.v)))
        rejected = PIVParameters(; base..., validation = (:velocity_magnitude => (max = magnitudes[cld(length(magnitudes), 2)],),), replace_outliers = true)
        bad = run_piv(A, B, rejected; threaded = false, on_diagnostics = x -> (d[] = x))
        @test any(bad.outliers) && !all(bad.outliers)
        @test !isequal(bad.u, raw.u) || !isequal(bad.v, raw.v)
        @test only(d[].passes).residual == expected
        @test only(d[].passes).residual.finite_count > 0 # not a summary of filled output
        mask = falses(48, 48); mask[1:20, 1:20] .= true
        scaled = run_piv(A, B, p; mask, roi = ROI(3:46, 3:46), threaded = false,
            scale = PhysicalScale(pixel_size = 0.02, dt = 0.001), on_diagnostics = x -> (d[] = x))
        @test d[].processing_size == (44, 44)
        @test only(d[].passes).residual.masked_count == count(scaled.mask)
        @test only(d[].passes).residual.unit == "px"
        @test Base.summarysize(d[]) < 10000
        @test_throws ErrorException run_piv(A, B, p; threaded = false, on_diagnostics = _ -> error("consumer failed"))
        calls = Ref(0)
        @test_throws DimensionMismatch run_piv(A, B[1:40, :], p; on_diagnostics = _ -> (calls[] += 1))
        @test calls[] == 0
        passes = multipass_parameters([24, 16]; uod_enable = false, final = (; max_iterations = 2))
        run_piv(A, B, passes; threaded = false, on_diagnostics = x -> (d[] = x))
        @test length(d[].passes) == 2 && [x.pass_index for x in d[].passes] == [1, 2]
        @test [x.executed_iterations for x in d[].passes] == [1, 2]
        # Ensemble capture has a separate pooled-sweep contract, exercised in
        # test_ensemble_execution_diagnostics.jl.
    end

    mktempdir() do directory
        @testset "Unsupported stereo associations/history fail before I/O" begin
            camera = PinholeCamera([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 1.])
            dewarper = ImageDewarper(camera, DewarpGrid(x = 1.:8., y = 1.:8.), (8, 8))
            p = PIVParameters(window_size = 4, overlap = 2)
            pairs = [("missing-A.png", "missing-B.png")]
            acquisitions = [("missing-A.png", "missing-B.png", "missing-C.png", "missing-D.png")]
            protected = joinpath(directory, "stereo-preserved.jld2"); write(protected, "preserved")
            for keywords in ((; _diagnostics_association = :unsupported), (; on_measurement_history = identity))
                for call in (
                    () -> run_piv_stereo(A, B, A, B, dewarper, dewarper, p; keywords...),
                    () -> run_piv_stereo(A, B, A, B, dewarper, dewarper; effort = :low, keywords...),
                    () -> run_piv_stereo_sequence(pairs, pairs, dewarper, dewarper, p; output = protected, keywords...),
                    () -> run_piv_stereo_sequence(pairs, pairs, dewarper, dewarper; effort = :low, output = protected, keywords...),
                    () -> run_piv_stereo_sequence(acquisitions, dewarper, dewarper, p; output = protected, keywords...),
                    () -> run_piv_stereo_sequence(acquisitions, dewarper, dewarper; effort = :low, output = protected, keywords...),
                    () -> run_piv_stereo_ensemble(pairs, pairs, dewarper, dewarper, p; keywords...),
                    () -> run_piv_stereo_ensemble(pairs, pairs, dewarper, dewarper; effort = :low, keywords...))
                    @test_throws ArgumentError call()
                    @test read(protected, String) == "preserved"
                end
            end
            for keywords in ((;on_diagnostics=identity),(;record_diagnostics=true))
                @test_throws ArgumentError run_piv_stereo_ensemble(pairs,pairs,dewarper,dewarper,p;keywords...)
                @test read(protected,String)=="preserved"
            end
        end
        @testset "Sequence ordering, persistence, lifetime and failures" begin
            p = PIVParameters(; base..., max_iterations = 2)
            pairs = fill((A, B), 4)
            output = joinpath(directory, "sequence.jld2")
            events = Tuple{Symbol,Int}[]; payloads = WeakRef[]
            @test run_piv_sequence(pairs, p; output, record_diagnostics = true, collect_results = false, progress = false,
                on_diagnostics = (i, d) -> (push!(events, (:diagnostics, i)); @test d.pair_index == i),
                on_result = (i, r) -> (push!(events, (:result, i)); push!(payloads, WeakRef(r.u)))) === nothing
            @test events == [(kind, i) for i in 1:4 for kind in (:diagnostics, :result)]
            index = ResultFile(output)
            @test length(index) == 4
            for i in 1:4
                d = load_execution_diagnostics(index, i)
                @test d.pair_index == i && only(d.passes).executed_iterations == 2
                @test d.association === nothing
                @test execution_diagnostics_data(Hammerhead._execution_decode(execution_diagnostics_data(d))) == execution_diagnostics_data(d)
            end
            GC.gc(); @test all(ref -> ref.value === nothing, payloads)
            @test_throws BoundsError load_execution_diagnostics(index, 5)
            legacy = joinpath(directory, "legacy.jld2"); save_results(legacy, run_piv(A, B, p))
            @test load_execution_diagnostics(legacy) === nothing
            copied = joinpath(directory, "copy.jld2"); save_results(copied, index)
            @test load_execution_diagnostics(copied) === nothing
            perpair = (i, pair) -> joinpath(directory, "pair-$i.jld2")
            run_piv_sequence(pairs, p; output = perpair, record_diagnostics = true, collect_results = false, progress = false)
            @test load_execution_diagnostics(perpair(4, nothing)).pair_index == 4
            attempted = Ref(false)
            @test_throws ArgumentError run_piv_sequence(pairs, p; record_diagnostics = true, preprocess = x -> (attempted[] = true; x))
            @test !attempted[]
            unsupported_output = joinpath(directory, "unsupported.jld2"); write(unsupported_output, "preserved")
            @test_throws ArgumentError run_ptv_sequence(pairs; output = unsupported_output, on_diagnostics = (_, _) -> nothing)
            @test_throws ArgumentError run_ptv_sequence(pairs; output = unsupported_output, record_diagnostics = true)
            @test read(unsupported_output, String) == "preserved"
            @test_throws ArgumentError run_piv_sequence(pairs; record_diagnostics = true)
            failed_output = joinpath(directory, "failed.jld2")
            error_seen = try
                run_piv_sequence(pairs, p; output = failed_output, record_diagnostics = true, collect_results = false, progress = false,
                    on_diagnostics = (i, _) -> i == 2 && error("stop pair2"))
                nothing
            catch err; err end
            @test error_seen isa ErrorException && occursin("stop pair2", sprint(showerror, error_seen))
            @test length(ResultFile(failed_output)) == 1 && load_execution_diagnostics(failed_output).pair_index == 1
            first_failed = joinpath(directory, "first-failed.jld2")
            @test_throws ErrorException run_piv_sequence(pairs, p; output = first_failed, record_diagnostics = true,
                progress = false, on_diagnostics = (_, _) -> error("first pair"))
            @test isempty(ResultFile(first_failed))
            @test jldopen(f -> f["execution_diagnostics_format_version"], first_failed) == 1
        end

        @testset "Native companion schema and exact key binding" begin
            p = PIVParameters(; base...); d = Ref{Any}(nothing)
            r = run_piv(A, B, p; on_diagnostics = x -> (d[] = x))
            data = execution_diagnostics_data(d[])
            path = joinpath(directory, "schema.jld2")
            function write_companion(change; version = 1, entry_key = "results/000007", declared_key = entry_key, unread_result = false)
                candidate = deepcopy(data); change(candidate)
                jldopen(path, "w") do file
                    file["format_version"] = 1
                    file[entry_key] = unread_result ? "unread unsupported payload" : r
                    file["execution_diagnostics_format_version"] = version
                    file[Hammerhead._execution_key(entry_key)] = Dict("result_key" => declared_key, "diagnostics" => candidate)
                end
            end
            write_companion(identity; unread_result = true)
            @test load_execution_diagnostics(path).execution_id == d[].execution_id # metadata-only read
            @test_throws ArgumentError ResultFile(path)[1]
            for change in (x -> x["diagnostics_format_version"] = true,
                           x -> x["pair_index"] = true,
                           x -> x["passes"][1]["executed_iterations"] = true,
                           x -> x["passes"][1]["checks"] = 1,
                           x -> x["passes"][1]["residual"]["masked_count"] = -1,
                           x -> begin
                               summary = x["passes"][1]["residual"]
                               summary["finite_count"] = typemax(Int)
                               summary["nonfinite_count"] = typemax(Int)
                               summary["masked_count"] = 7
                           end,
                           x -> x["passes"][1]["residual"]["masked_count"] = prod(size(A)),
                           x -> x["passes"][1]["residual"]["mean_magnitude"] = NaN,
                           x -> x["core_source_sha256"] = "bad")
                write_companion(change); @test_throws ArgumentError load_execution_diagnostics(path)
            end
            write_companion(identity; version = 2); @test_throws ArgumentError load_execution_diagnostics(path)
            @test ResultFile(path)[1] isa PIVResult # result reader ignores companion version
            write_companion(identity; declared_key = "results/000001"); @test_throws ArgumentError load_execution_diagnostics(path)
            jldopen(path, "w") do file
                file["format_version"] = 1; file["results/000007"] = r
                file["execution_diagnostics/000007"] = Dict("result_key" => "results/000007", "diagnostics" => data)
            end
            @test_throws ArgumentError load_execution_diagnostics(path)
            stale = ResultFile(path); save_results(path, r)
            @test_throws ArgumentError load_execution_diagnostics(stale, 1)
        end

        @testset "Replay diagnostics are observational, not recipe changes" begin
            fixtures = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
            files = sort(filter(p -> endswith(lowercase(p), ".tif"), readdir(fixtures; join = true)))
            recipe = PIVRecipe(PIVParameters(; base..., max_iterations = 2); roi = ROI(1:48, 1:48), image_type = Float32, threaded = false)
            record = ExperimentRecord([(files[1], files[2])], recipe)
            initial_id = recipe_identity(recipe)
            output = joinpath(directory, "replay.jld2"); delivered = Ref{Any}(nothing)
            run = replay_experiment(record; output, record_diagnostics = true,
                on_diagnostics = (i, d) -> (delivered[] = d; @test i == 1))
            @test run.status === :completed && run.recipe_id == initial_id && recipe_identity(recipe) == initial_id
            d = load_execution_diagnostics(output)
            @test d.association == (recipe_id = initial_id, input_id = record.input_id)
            @test execution_diagnostics_data(d) == execution_diagnostics_data(delivered[])
            @test run.output_sha256 == Hammerhead._experiment_file_digest(output)
            @test only(d.passes).executed_iterations == 2 && only(d.passes).checks == 0
        end
    end
end
