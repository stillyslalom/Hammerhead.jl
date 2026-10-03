using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

# This oracle composes the standalone public core operations. It deliberately
# never calls the saved-recipe preprocessing closure or the GUI preview helper.
function preprocessing_parity_oracle(image, steps)
    value = copy(image)
    for step in steps
        o = step.options
        if step.operation === :subtract_background
            value = subtract_background(value, o["background"])
        elseif step.operation === :intensity_cap
            value = intensity_cap(value; n_sigma=o["n_sigma"])
        elseif step.operation === :highpass_filter
            value = highpass_filter(value; sigma=o["sigma"])
        elseif step.operation === :clahe
            value = clahe(value; tiles=Tuple(o["tiles"]), nbins=o["nbins"], clip_limit=o["clip_limit"])
        elseif step.operation === :percentile_stretch
            value = percentile_stretch(value; low=o["low"], high=o["high"])
        elseif step.operation === :invert_image
            value = invert_image(value)
        elseif step.operation === :local_variance_normalize
            value = local_variance_normalize(value; sigma=o["sigma"], epsilon=o["epsilon"])
        else
            error("unhandled oracle operation: $(step.operation)")
        end
    end
    value
end

# Test-owned path dispatch changes bytes AFTER actual decoding. No ordinary
# AbstractString loader method or production preview API is replaced.
struct PreprocessingParityDecodePath <: AbstractString
    path::String
    mutated::Base.RefValue{Bool}
end
Base.String(p::PreprocessingParityDecodePath) = p.path
Base.ncodeunits(p::PreprocessingParityDecodePath) = ncodeunits(p.path)
Base.codeunit(p::PreprocessingParityDecodePath, i::Integer) = codeunit(p.path, i)
Base.codeunit(::Type{PreprocessingParityDecodePath}) = UInt8
Base.isvalid(p::PreprocessingParityDecodePath, i::Integer) = isvalid(p.path, i)
Base.iterate(p::PreprocessingParityDecodePath) = iterate(p.path)
Base.iterate(p::PreprocessingParityDecodePath, state::Integer) = iterate(p.path, state)
Base.cconvert(::Type{Cstring}, p::PreprocessingParityDecodePath) = p.path
function HammerheadGUI.Hammerhead.load_image(::Type{T}, p::PreprocessingParityDecodePath) where {T<:AbstractFloat}
    image = load_image(T, p.path)
    bytes = read(p.path)
    bytes[end] ⊻= 0x01
    write(p.path, bytes)
    p.mutated[] = true
    image
end

function preprocessing_parity_files(dir)
    paths = [joinpath(dir, "frame $i.png") for i in 1:3]
    for (i, path) in enumerate(paths)
        image = [mod(31r + 19c + 7i + div(r*c, 3), 251) / 250
                 for r in 1:48, c in 1:64]
        save(path, Gray{N0f8}.(image))
    end
    paths
end

function preprocessing_parity_recipe(T, steps; external_preprocess=nothing)
    mask = falses(48, 64)
    mask[17:21, 24:29] .= true
    PIVRecipe([PIVParameters(window_size=8, search_area_size=12, overlap=4)];
        preprocessing=steps, external_preprocess, image_type=T, threaded=false,
        roi=ROI(9:40, 13:52), mask)
end

@testset "Preprocessing byte checks before and after decoding" begin
    H = HammerheadGUI.Hammerhead
    mktempdir() do dir
        paths = preprocessing_parity_files(dir)
        record = ExperimentRecord([(paths[1], paths[2])], preprocessing_parity_recipe(Float32, PreprocessStep[]))
        descriptor = deepcopy(record.input_files[1])
        original = read(paths[1])
        changed = copy(original); changed[end] ⊻= 0x01
        write(paths[1], changed)
        @test_throws ArgumentError H._experiment_load_image(descriptor, Float32)
        write(paths[1], original)
        mutated = Ref(false)
        descriptor["path"] = PreprocessingParityDecodePath(paths[1], mutated)
        @test_throws ArgumentError H._experiment_load_image(descriptor, Float32)
        @test mutated[] # actual decode returned; post-decode digest refused it
    end
end

@testset "Saved recipe image preview independent composition" begin
    mktempdir() do dir
        paths = preprocessing_parity_files(dir)
        ordered_pairs = [(paths[2], paths[1]), (paths[1], paths[1]), (paths[2], paths[3])]
        for T in (Float32, Float64)
            background_type = T === Float32 ? Float64 : Float32
            background = background_type.([0.05 + mod(r + 2c, 7) / 1000 for r in 1:48, c in 1:64])
            singles = [PreprocessStep(:subtract_background; background),
                PreprocessStep(:intensity_cap; n_sigma=0.8),
                PreprocessStep(:highpass_filter; sigma=1.3),
                PreprocessStep(:clahe; tiles=(3, 5), nbins=37, clip_limit=2.4),
                PreprocessStep(:percentile_stretch; low=7, high=89),
                PreprocessStep(:invert_image),
                PreprocessStep(:local_variance_normalize; sigma=1.7, epsilon=0.017)]
            for step in singles
                recipe = preprocessing_parity_recipe(T, [step])
                record = ExperimentRecord(ordered_pairs, recipe)
                rc = RecipeRevisionController(record)
                pc = RecipeImagePreviewController()
                preview_recipe_images!(pc, rc; pair_index=1, async=false)
                @test pc.state[] === :completed
                @test !pc.running[] && pc.task[] === nothing
                bundle = pc.bundle[]
                @test bundle !== nothing
                raw_a, raw_b = load_image(T, paths[2]), load_image(T, paths[1])
                @test bundle.raw_a == raw_a && bundle.raw_b == raw_b
                @test bundle.processed_a == preprocessing_parity_oracle(raw_a, [step])
                @test bundle.processed_b == preprocessing_parity_oracle(raw_b, [step])
                @test eltype(bundle.processed_a) === T && eltype(bundle.processed_b) === T
                @test size(bundle.processed_a) == size(bundle.raw_a) == (48, 64)
                @test bundle.recipe_id == recipe_identity(recipe)
                @test bundle.source_recipe_id == recipe_identity(record.recipe)
                @test bundle.input_id == record.input_id && bundle.pair_index == 1
                @test bundle.input_paths == [paths[2], paths[1]]
                @test bundle.input_descriptors == record.input_files[record.pairs[1]]
                @test bundle.mask == recipe.mask && bundle.mask !== recipe.mask
                @test bundle.roi.rows == recipe.roi.rows && bundle.roi.cols == recipe.roi.cols
                @test bundle.raw_a !== bundle.processed_a
                @test recipe.preprocessing[1].options == step.options
                # Conditioning ignores mask values: even masked pixels preserve
                # the same standalone-operation result as every other pixel.
                @test bundle.processed_a[recipe.mask] == preprocessing_parity_oracle(raw_a, [step])[recipe.mask]
                if step.operation in (:clahe, :highpass_filter, :local_variance_normalize)
                    rows, cols = recipe.roi.rows, recipe.roi.cols
                    cropped_first = preprocessing_parity_oracle(raw_a[rows, cols], [step])
                    @test bundle.processed_a[rows, cols] != cropped_first
                end
            end
            # Duplicates are ordered executable steps, not a toggle set.
            chain = [singles[1], singles[6], singles[2], singles[6], singles[7]]
            record = ExperimentRecord(ordered_pairs, preprocessing_parity_recipe(T, chain))
            rc = RecipeRevisionController(record); pc = RecipeImagePreviewController()
            preview_recipe_images!(pc, rc; pair_index=3, async=false)
            original_bundle = pc.bundle[]
            @test original_bundle.processed_a == preprocessing_parity_oracle(load_image(T, paths[2]), chain)
            @test original_bundle.processed_b == preprocessing_parity_oracle(load_image(T, paths[3]), chain)
            @test original_bundle.pair_index == 3 && original_bundle.input_paths == [paths[2], paths[3]]
            move_revision_preprocess!(rc, 1, 2)
            reordered = revision_recipe(rc)
            preview_recipe_images!(pc, rc; pair_index=3, async=false)
            @test pc.bundle[].processed_a == preprocessing_parity_oracle(load_image(T, paths[2]), reordered.preprocessing)
            @test pc.bundle[].processed_a != original_bundle.processed_a
            @test pc.bundle[].recipe_id != original_bundle.recipe_id
            @test all(a.operation == b.operation && a.options == b.options
                      for (a, b) in zip(record.recipe.preprocessing, chain))
            @test record.input_id == pc.bundle[].input_id
        end
    end
end

@testset "Captured pair and recipe; failed preview retains prior bundle" begin
    mktempdir() do dir
        paths = preprocessing_parity_files(dir)
        recipe = preprocessing_parity_recipe(Float32, [PreprocessStep(:intensity_cap; n_sigma=0.7)])
        record = ExperimentRecord([(paths[2], paths[1]), (paths[1], paths[3])], recipe)
        rc = RecipeRevisionController(record); pc = RecipeImagePreviewController()
        captured_id = recipe_identity(recipe)
        subscription = on(pc.running) do running
            running || return
            set_revision_preprocess!(rc, 1, Dict(:n_sigma=>"3.9"))
            reverse!(rc.original.pairs[1])
        end
        preview_recipe_images!(pc, rc; pair_index=1, async=false)
        off(subscription)
        @test pc.state[] === :completed
        bundle = pc.bundle[]
        @test bundle.recipe_id == captured_id && bundle.input_paths == [paths[2], paths[1]]
        @test bundle.processed_a == intensity_cap(load_image(Float32, paths[2]); n_sigma=0.7)
        # A later valid request has its own new capture, not the former identities.
        reverse!(rc.original.pairs[1])
        preview_recipe_images!(pc, rc; pair_index=2, async=false)
        @test pc.state[] === :completed && pc.bundle[].recipe_id != captured_id
        @test pc.bundle[].input_paths == [paths[1], paths[3]]
        prior = pc.bundle[]
        original = read(paths[1]); changed = copy(original); changed[end] ⊻= 0x01
        write(paths[1], changed)
        preview_recipe_images!(pc, rc; pair_index=2, async=false)
        @test pc.state[] === :failed && pc.error[] isa ArgumentError
        @test pc.bundle[] === prior && !pc.running[]
        write(paths[1], original)
        rm(paths[3])
        preview_recipe_images!(pc, rc; pair_index=2, async=false)
        @test pc.state[] === :failed && pc.bundle[] === prior && !pc.running[]
        # Missing acquisitions do not prevent metadata-only recipe inspection.
        apply_recipe_revision!(rc; async=false)
        @test rc.state[] === :completed && revision_recipe(rc).recipe_id == pc.bundle[].recipe_id
    end
end

@testset "External scripts inspect/save but refuse pixel preview" begin
    mktempdir() do dir
        paths = preprocessing_parity_files(dir)
        script_path = joinpath(dir, "never-evaluate.jl")
        write(script_path, "error(\"numerical preview must never include this script\")")
        script = ScriptReference(script_path; entrypoint="caller_supplied_function")
        recipe = preprocessing_parity_recipe(Float64, [PreprocessStep(:invert_image)]; external_preprocess=script)
        record = ExperimentRecord([(paths[1], paths[2])], recipe)
        rc = RecipeRevisionController(record); pc = RecipeImagePreviewController()
        apply_recipe_revision!(rc; async=false)
        @test rc.state[] === :completed && revision_recipe(rc).external_preprocess.sha256 == script.sha256
        destination = joinpath(dir, "script-revision.jld2")
        save_recipe_revision!(rc, destination; async=false)
        @test rc.state[] === :completed && isfile(destination)
        @test rc.saved_record[].input_id == record.input_id && isempty(rc.saved_record[].runs)
        preview_recipe_images!(pc, rc; async=false)
        @test pc.state[] === :failed && pc.error[] isa ArgumentError && pc.bundle[] === nothing
        @test occursin("script", lowercase(sprint(showerror, pc.error[])))
        @test rc.saved_record[] !== nothing && revision_recipe(rc).recipe_id == recipe.recipe_id
    end
end
