using Test
using Hammerhead

comparison_rows(diff) = [(path = c.path, before = c.before, after = c.after) for c in diff]
comparison_changes(diff) = Dict(c.path => c for c in diff)

@testset "Scientific recipe revision comparison" begin
    p32 = PIVParameters()
    p16 = PIVParameters(window_size = 16, overlap = 8)
    baseline = PIVRecipe(p32)

    @testset "Identity verification and report interface" begin
        equivalent = PIVRecipe(p32)
        diff = recipe_diff(baseline, equivalent)
        @test diff isa RecipeDiff
        @test isempty(diff) && length(diff) == 0
        @test collect(diff) == RecipeChange[]
        @test diff.before_id == diff.after_id == recipe_identity(baseline)
        @test occursin("No scientific recipe changes", sprint(show, MIME"text/plain"(), diff))
        @test sprint(show, diff) == "RecipeDiff(0 changes)"
        changed = PIVRecipe(p32; threaded = true)
        one = recipe_diff(baseline, changed)
        @test length(one) == 1
        @test one[1] isa RecipeChange
        @test one[end] === one[1]
        @test firstindex(one) == lastindex(one) == 1
        @test comparison_rows(one) == [(path = "threaded", before = false, after = true)]
        @test occursin("threaded: false -> true", sprint(show, MIME"text/plain"(), one))
        @test occursin("threaded: false -> true", sprint(show, MIME"text/plain"(), one[1]))
        @test_throws BoundsError one[2]

        # Identical/stale IDs never bypass snapshot checks on either side.
        mutated = PIVRecipe(p32)
        push!(mutated.passes, p16)
        @test_throws ArgumentError recipe_diff(mutated, mutated)
        @test_throws ArgumentError recipe_diff(baseline, mutated)
        @test_throws ArgumentError recipe_diff(mutated, baseline)
        altered_mask = PIVRecipe(p32; mask = falses(8, 8))
        altered_mask.mask[2, 3] = true
        @test_throws ArgumentError recipe_diff(altered_mask, altered_mask)
        altered_background = PIVRecipe(p32; preprocessing = [PreprocessStep(:subtract_background; background = zeros(8, 8))])
        altered_background.preprocessing[1].options["background"][2, 3] = 1
        @test_throws ArgumentError recipe_diff(altered_background, baseline)
    end

    @testset "Full settings and readable form values" begin
        changed_pass = PIVParameters(window_size = (64, 48), search_area_size = (80, 64),
            overlap = (8, 12), correlation_method = :phase, padding = true,
            apodization = :gauss, subpixel_method = :none, n_peaks = 5,
            peak_finder = :exclusion, uncertainty = true, uod_enable = false,
            uod_threshold = 3, uod_neighborhood = 1, min_peak_ratio = 0.4,
            validation = (:velocity_magnitude => (min = 1, max = Inf),),
            replace_outliers = false, max_iterations = 2, convergence_tol = 0.1,
            keep_correlation_planes = true)
        before = PIVRecipe(p32; roi = ROI(1:32, 2:33), scale = PhysicalScale())
        after = PIVRecipe(changed_pass; roi = ROI(3:34, 4:35),
            scale = PhysicalScale(pixel_size = 0.02, dt = 0.001, length_unit = "mm", time_unit = "s"),
            backend = :ka, image_type = Float32, threaded = true,
            predictor_smoothing = false, mask_threshold = 0.25, uncertainty_backend = :cpu)
        rows = comparison_changes(recipe_diff(before, after))
        for key in fieldnames(PIVParameters)
            @test any(path -> path == "passes[1].$key" || startswith(path, "passes[1].$key["), keys(rows))
        end
        @test rows["passes[1].window_size"].before == (32, 32)
        @test rows["passes[1].window_size"].after == (64, 48)
        @test rows["passes[1].search_area_size"].after == (80, 64)
        @test rows["passes[1].overlap"].after == (8, 12)
        @test rows["roi.rows"].before == (1, 32)
        @test rows["roi.rows"].after == (3, 34)
        @test rows["roi.cols"].after == (4, 35)
        @test rows["backend"].after === :ka
        @test rows["image_type"].after == "Float32"
        @test rows["threaded"].after === true
        @test rows["predictor_smoothing"].after === false
        @test rows["mask_threshold"].after == 0.25
        @test rows["uncertainty_backend"].after === :cpu
        @test rows["scale.pixel_size"].after == 0.02
        @test rows["scale.dt"].after == 0.001
        @test rows["scale.length_unit"].after == "mm"
        @test rows["scale.time_unit"].after == "s"
        @test rows["passes[1].validation[1]"].before === missing
        @test rows["passes[1].validation[1]"].after.options.max === Inf
        optional = comparison_changes(recipe_diff(baseline, PIVRecipe(p32; roi = ROI(1:32, 1:32), scale = PhysicalScale())))
        @test optional["roi"].before === nothing
        @test optional["roi"].after == (cols = (1, 32), rows = (1, 32))
        @test optional["scale"].before === nothing
        @test optional["scale"].after.pixel_size == 1
    end

    @testset "Ordered passes, preprocessing, validators and deterministic paths" begin
        ordered = PIVRecipe([p32, p16])
        swapped = PIVRecipe([p16, p32])
        paths = [c.path for c in recipe_diff(ordered, swapped)]
        @test "passes[1].window_size" in paths
        @test "passes[2].window_size" in paths
        appended = recipe_diff(PIVRecipe(p32), ordered)
        @test length(appended) == 1
        @test appended[1].path == "passes[2]"
        @test appended[1].before === missing
        @test appended[1].after.window_size == (16, 16)
        @test recipe_diff(ordered, PIVRecipe(p32))[1].after === missing

        steps = [PreprocessStep(:highpass_filter; sigma = 3), PreprocessStep(:clahe; tiles = (8, 8))]
        reordered = [steps[2], steps[1]]
        before = PIVRecipe(p32; preprocessing = steps)
        after = PIVRecipe(p32; preprocessing = reordered)
        rows = comparison_changes(recipe_diff(before, after))
        @test rows["preprocessing[1].operation"].before === :highpass_filter
        @test rows["preprocessing[1].operation"].after === :clahe
        @test rows["preprocessing[1].options.sigma"].after === missing
        @test rows["preprocessing[1].options.tiles"].before === missing
        @test rows["preprocessing[1].options.tiles"].after == (8, 8)
        tiles = comparison_changes(recipe_diff(PIVRecipe(p32; preprocessing = [steps[2]]),
            PIVRecipe(p32; preprocessing = [PreprocessStep(:clahe; tiles = (4, 6))])))
        @test tiles["preprocessing[1].options.tiles"].before == (8, 8)
        @test tiles["preprocessing[1].options.tiles"].after == (4, 6)
        validators = (:peak_ratio => 1.2, :uod => (threshold = 3.0,))
        validator_diff = comparison_changes(recipe_diff(PIVRecipe(PIVParameters(validation = validators)),
            PIVRecipe(PIVParameters(validation = reverse(validators)))))
        @test validator_diff["passes[1].validation[1].operation"].before == "peak_ratio"
        @test validator_diff["passes[1].validation[1].operation"].after == "uod"
        @test validator_diff["passes[1].validation[2].operation"].after == "peak_ratio"
        @test isequal(comparison_rows(recipe_diff(before, after)), comparison_rows(recipe_diff(before, after)))
        @test recipe_diff(before, after) == recipe_diff(before, after)
        @test sprint(show, MIME"text/plain"(), recipe_diff(before, after)) == sprint(show, MIME"text/plain"(), recipe_diff(before, after))
        @test recipe_identity(before) == before.recipe_id
        @test recipe_identity(after) == after.recipe_id
        @test [s.operation for s in before.preprocessing] == [:highpass_filter, :clahe]
    end

    @testset "Embedded summaries retain no payloads" begin
        background = reshape(Float32.(1:128^2), 128, 128)
        altered = copy(background); altered[17, 31] += 1
        mask = falses(128, 128); mask[17, 31] = true
        before = PIVRecipe(p32; mask = falses(128, 128),
            preprocessing = [PreprocessStep(:subtract_background; background)])
        after = PIVRecipe(p32; mask,
            preprocessing = [PreprocessStep(:subtract_background; background = altered)])
        rows = comparison_changes(recipe_diff(before, after))
        bg = rows["preprocessing[1].options.background"]
        @test bg.before isa RecipeArraySummary && bg.after isa RecipeArraySummary
        @test bg.before.size == (128, 128)
        @test bg.before.element_type == "Float32"
        @test length(bg.before.sha256) == 64
        @test bg.before.sha256 == Hammerhead._experiment_digest(background)
        @test bg.before.sha256 != bg.after.sha256
        @test rows["mask"].after.element_type == "Bool"
        @test rows["mask"].before.sha256 != rows["mask"].after.sha256
        @test Base.summarysize(recipe_diff(before, after)) < 8192
        @test recipe_diff(before, after) == recipe_diff(before, after)
        @test hash(recipe_diff(before, after)) == hash(recipe_diff(before, after))
        @test length(sprint(show, MIME"text/plain"(), recipe_diff(before, after))) < 1000
        @test occursin("Float32 array (128, 128) sha256=", sprint(show, bg.before))
        @test sprint(show, MIME"text/plain"(), bg.before) == sprint(show, bg.before)
        @test isempty(recipe_diff(before, deepcopy(before)))

        # Same values with changed shape/precision/order are different content.
        for variant in (Float64.(background), reshape(copy(vec(background)), 64, 256), permutedims(background))
            comparison = recipe_diff(before, PIVRecipe(p32; mask = before.mask,
                preprocessing = [PreprocessStep(:subtract_background; background = variant)]))
            @test length(comparison) == 1
            @test comparison[1].before.sha256 != comparison[1].after.sha256
        end
        appended = recipe_diff(baseline, after)
        step = only(c for c in appended if c.path == "preprocessing[1]")
        @test step.before === missing
        @test step.after.options.background isa RecipeArraySummary
        @test Base.summarysize(appended) < 8192
        removed = recipe_diff(after, baseline)
        @test only(c for c in removed if c.path == "preprocessing[1]").before.options.background isa RecipeArraySummary
        @test comparison_changes(removed)["mask"].after === nothing
        @test background == reshape(Float32.(1:128^2), 128, 128)
        @test recipe_identity(before) == before.recipe_id
        @test recipe_identity(after) == after.recipe_id
    end

    @testset "External content and entrypoints, not script locators" begin
        hash_a, hash_b = repeat("a", 64), repeat("b", 64)
        script = ScriptReference("missing/first-location.jl", hash_a, "prepare(image)")
        relocated = ScriptReference("another/missing-location.jl", hash_a, "prepare(image)")
        before = PIVRecipe(p32; external_preprocess = script)
        @test isempty(recipe_diff(before, PIVRecipe(p32; external_preprocess = relocated)))
        changed = PIVRecipe(p32; external_preprocess = ScriptReference("missing/other.jl", hash_b, "changed(image)"))
        rows = comparison_changes(recipe_diff(before, changed))
        @test Set(keys(rows)) == Set(["external_preprocess.entrypoint", "external_preprocess.sha256"])
        @test rows["external_preprocess.sha256"].before == hash_a
        @test rows["external_preprocess.sha256"].after == hash_b
        @test rows["external_preprocess.entrypoint"].after == "changed(image)"
        added = recipe_diff(baseline, before)
        @test added[1].path == "external_preprocess"
        @test added[1].before === nothing
        @test added[1].after == (entrypoint = "prepare(image)", sha256 = hash_a)
        @test !occursin("missing/", sprint(show, MIME"text/plain"(), added))
        @test script.path == "missing/first-location.jl"
    end
end
