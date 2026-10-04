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
            # format version 1 (before per-camera preprocessing) still loads
            jldopen(joinpath(dir, "v1.jld2"), "w") do f
                f["recipe_format_version"] = 1
                f["recipe"] = Hammerhead._recipe_data(recipe)
            end
            @test load_recipe(joinpath(dir, "v1.jld2")) == recipe
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

    @testset "save_results with settings and sources" begin
        mktempdir() do dir
            rs = apply_recipe(recipe, pairs; progress = false)
            out = save_results(joinpath(dir, "later.jld2"), rs; recipe,
                               sources = [["a$i.tif", "b$i.tif"] for i in 1:3])
            @test load_recipe(out) == recipe
            @test isequal(load_results(out)[3].u, rs[3].u)
            @test load_sources(out) == [["a$i.tif", "b$i.tif"] for i in 1:3]
            @test_throws DimensionMismatch save_results(joinpath(dir, "bad.jld2"), rs;
                                                        sources = [["a"]])
            # sequence drivers store frame paths; in-memory frames store none
            paths = String[]
            for (k, (a, b)) in enumerate(pairs[1:2])
                for (img, tag) in ((a, "a"), (b, "b"))
                    p = joinpath(dir, "f$(k)$(tag).png")
                    Hammerhead.FileIO.save(p, Hammerhead.Gray.(img ./ maximum(img)))
                    push!(paths, p)
                end
            end
            seq = joinpath(dir, "seq.jld2")
            run_piv_sequence(image_pairs(paths), passes; output = seq, progress = false)
            @test load_sources(seq) == [paths[1:2], paths[3:4]]
            mem = joinpath(dir, "mem.jld2")
            run_piv_sequence(pairs[1:2], passes; output = mem, progress = false)
            @test load_sources(mem) == [String[], String[]]
        end
    end

    @testset "PTV and tracking recipes" begin
        ptv = PTVParameters(search_radius = 4)
        pr = PIVRecipe(passes; mode = :ptv, ptv, mask)
        mktempdir() do dir
            out = joinpath(dir, "ptv.jld2")
            res = apply_recipe(pr, pairs; output = out, progress = false)
            direct = run_ptv_sequence(pairs, ptv; mask, piv_passes = passes, progress = false)
            @test length(res) == 3 && all(isequal(a.u, b.u) && isequal(a.x, b.x) for (a, b) in zip(res, direct))
            @test count(!, res[1].outliers) > 50
            @test load_recipe(out) == pr && load_recipe(out).ptv == ptv
            @test isequal(load_results(out)[2].v, res[2].v)
        end
        # tracking takes the frame sequence; preprocessing applies to each frame
        trng = MersenneTwister(3)
        positions = [(8 + 80 * rand(trng), 8 + 80 * rand(trng)) for _ in 1:80]
        frames = map(0:3) do k
            img = zeros(n, n)
            foreach(p -> add_particle!(img, (p[1] + 1.0k, p[2] - 1.5k), 3.0), positions)
            img
        end
        tr = PIVRecipe(passes; mode = :tracking, ptv, min_track_length = 4, max_gap = 1,
                       preprocessing = [PreprocessStep(:intensity_cap)])
        mktempdir() do dir
            out = joinpath(dir, "tracks.jld2")
            result = apply_recipe(tr, frames; output = out, progress = false)
            direct = track_particles(frames, ptv; piv_passes = passes, min_track_length = 4,
                                     max_gap = 1, preprocess = recipe_preprocess(tr), progress = false)
            @test length(result.trajectories) == length(direct.trajectories) > 5
            @test result.trajectories[1].x == direct.trajectories[1].x
            @test load_recipe(out) == tr && load_recipe(out).max_gap == 1
            @test only(load_results(out)) isa TrackingResult
            @test_throws ArgumentError apply_recipe(tr, frames; output = i -> "$i.jld2", progress = false)
        end
        # no predictor: a pure nearest-neighbor search
        nopred = PIVRecipe(passes; mode = :ptv, ptv = PTVParameters(search_radius = 3), ptv_predictor = :none)
        @test isequal(only(apply_recipe(nopred, pairs[1:1]; progress = false)).u,
                      run_ptv(pairs[1]..., nopred.ptv; predictor = nothing).u)
        paths = [c.path for c in recipe_diff(pr, PIVRecipe(passes; mode = :ptv, mask,
                                                              ptv = PTVParameters(search_radius = 5)))]
        @test paths == ["ptv.search_radius"]
        @test_throws ArgumentError PIVRecipe(passes; mode = :ptv, roi = ROI(1:50, 1:50))
        @test_throws ArgumentError PIVRecipe(passes; ptv_predictor = :field)
        @test_throws ArgumentError PIVRecipe(passes; min_track_length = 1)
        @test_throws ArgumentError PIVRecipe(passes; max_gap = -1)
        @test_throws ArgumentError apply_recipe(pr, pairs; backend = :ka)
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
