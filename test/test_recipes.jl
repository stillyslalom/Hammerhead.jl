using Hammerhead
using Test
using Random
using JLD2

@testset "Recipes" begin
    rng = MersenneTwister(7)
    n = 96
    pairs = map(1:3) do _
        positions = [(rand(rng) * n, rand(rng) * n) for _ in 1:150]
        particle_pair((n, n), positions, 1.0, -1.5)
    end
    passes = multipass_parameters([32, 16]; overlap_fraction = 0.5)
    mask = falses(n, n); mask[1:8, 1:8] .= true
    background = fill(0.01, n, n)
    recipe = PIVRecipe(passes;
        preprocessing = [PreprocessStep(:subtract_background; background),
                         PreprocessStep(:highpass_filter; sigma = 4)],
        mask, roi = ROI(5:92, 3:94), scale = PhysicalScale(0.02, 1e-3, "mm", "s"),
        image_type = Float32)

    @testset "preprocessing steps" begin
        step = PreprocessStep(:clahe; tiles = (4, 4))
        @test step.options == Dict("tiles" => [4, 4], "clip_limit" => 2.0, "nbins" => 256)
        @test_throws ArgumentError PreprocessStep(:blur)
        @test_throws ArgumentError PreprocessStep(:highpass_filter; radius = 2)
        @test_throws ArgumentError PreprocessStep(:subtract_background)
        img = Float32.(pairs[1][1])
        @test recipe_preprocess(recipe)(img) ≈ highpass_filter(subtract_background(img, background); sigma = 4)
        @test recipe_preprocess(PIVRecipe(passes)) === nothing
        # the steps form applies exactly the recipe's preprocessing, on a snapshot
        @test recipe_preprocess(recipe.preprocessing)(img) == recipe_preprocess(recipe)(img)
        @test recipe_preprocess(PreprocessStep[]) === nothing
        editable = [PreprocessStep(:highpass_filter; sigma = 2)]
        preview = recipe_preprocess(editable)
        editable[1].options["sigma"] = 8.0
        original = copy(img)
        @test preview(img) == highpass_filter(img; sigma = 2)
        @test img == original
    end

    @testset "save and load" begin
        mktempdir() do dir
            path = save_recipe(joinpath(dir, "recipe.jld2"), recipe)
            loaded = load_recipe(path)
            @test loaded == recipe
            @test loaded.passes == recipe.passes
            @test loaded.mask == mask && loaded.roi == recipe.roi && loaded.scale == recipe.scale
            @test loaded.image_type === Float32
            save_results(joinpath(dir, "r.jld2"), run_piv(pairs[1]...))
            @test_throws ArgumentError load_recipe(joinpath(dir, "r.jld2"))
        end
        validated = PIVRecipe(PIVParameters(validation = (:peak_ratio => 1.3, :velocity_magnitude => (max = 5.0,))))
        mktempdir() do dir
            @test load_recipe(save_recipe(joinpath(dir, "v.jld2"), validated)) == validated
        end
    end

    @testset "apply to a sequence" begin
        mktempdir() do dir
            out = joinpath(dir, "results.jld2")
            results = apply_recipe(recipe, pairs; output = out, progress = false)
            direct = run_piv_sequence(pairs, passes; progress = false,
                preprocess = recipe_preprocess(recipe), image_type = Float32,
                mask, roi = recipe.roi, scale = recipe.scale)
            @test length(results) == 3
            @test all(isequal(r.u, d.u) && isequal(r.v, d.v) for (r, d) in zip(results, direct))
            @test results[1].scale == recipe.scale
            @test load_recipe(out) == recipe
            @test isequal(load_results(out)[2].u, results[2].u)
        end
    end

    @testset "apply as an ensemble" begin
        ens = PIVRecipe(passes; mode = :ensemble)
        mktempdir() do dir
            out = joinpath(dir, "ensemble.jld2")
            result = apply_recipe(ens, pairs; output = out, progress = false)
            @test isequal(result.u, run_piv_ensemble(pairs, passes; progress = false).u)
            @test load_recipe(out) == ens
            @test isequal(only(load_results(out)).u, result.u)
        end
        @test_throws ArgumentError apply_recipe(PIVRecipe(passes; mode = :ensemble, roi = ROI(1:64, 1:64)), pairs)
        @test_throws ArgumentError PIVRecipe(passes; mode = :stereo)
    end

    @testset "recipe_diff" begin
        @test isempty(recipe_diff(recipe, recipe))
        changed = PIVRecipe(multipass_parameters([32, 24]; overlap_fraction = 0.5);
                            preprocessing = recipe.preprocessing, mask = nothing,
                            roi = recipe.roi, scale = recipe.scale, image_type = Float32)
        paths = [c.path for c in recipe_diff(recipe, changed)]
        @test "passes[2].window_size" in paths
        @test "mask" in paths
        mask_change = only(c for c in recipe_diff(recipe, changed) if c.path == "mask")
        @test mask_change.before == "96×96 Bool array" && mask_change.after === nothing
    end
end
