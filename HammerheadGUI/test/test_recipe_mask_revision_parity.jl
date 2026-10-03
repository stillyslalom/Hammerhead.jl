using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8
using Observables: on, off

function mask_parity_files(directory)
    a = [mod(37r + 19c + div(r*c, 3), 251)/250 for r in 1:56, c in 1:72]
    paths = [joinpath(directory, "mask frame $i.png") for i in 1:2]
    for (i, path) in enumerate(paths)
        save(path, Gray{N0f8}.(circshift(a, (i-1, 2(i-1)))))
    end
    paths
end

function mask_parity_seed()
    bits = falses(56, 72)
    bits[8:42, 13:25] .= true
    bits[2:4, 60:66] .= true
    bits[49:54, 63:70] .= true
    bits
end

function mask_parity_recipe(T, backend; mask=mask_parity_seed(), script=nothing)
    B = T === Float32 ? Float64 : Float32
    background = B.([.03 + mod(r+2c, 9)/1000 for r in 1:56, c in 1:72])
    passes = [PIVParameters(window_size=16, search_area_size=backend === :ka ? 16 : 20,
                  overlap=8, padding=true, uod_enable=false, n_peaks=2,
                  replace_outliers=false),
              PIVParameters(window_size=8, search_area_size=backend === :ka ? 8 : 12,
                  overlap=4, padding=true, uod_enable=false, uncertainty=true,
                  validation=(:peak_ratio=>1.0,), replace_outliers=false)]
    PIVRecipe(passes; image_type=T, backend, threaded=false,
        predictor_smoothing=false, uncertainty_backend=:cpu, mask_threshold=.37,
        mask, roi=ROI(7:50, 11:66), scale=PhysicalScale(.025, .01, "mm", "s"),
        preprocessing=[PreprocessStep(:subtract_background; background),
                       PreprocessStep(:highpass_filter; sigma=1.6)],
        external_preprocess=script)
end

# The numerical oracle conditions complete images with standalone public core
# operations. Neither a revision preview nor a saved preprocessing closure is
# used to create the expected input or mask.
function mask_parity_condition(image, recipe)
    background = recipe.preprocessing[1].options["background"]
    sigma = recipe.preprocessing[2].options["sigma"]
    highpass_filter(subtract_background(image, background); sigma)
end

function mask_parity_direct(a, b, recipe; roi=recipe.roi, mask=recipe.mask)
    run_piv(a, b, recipe.passes; backend=recipe.backend,
        uncertainty_backend=recipe.uncertainty_backend, threaded=recipe.threaded,
        predictor_smoothing=recipe.predictor_smoothing, mask_threshold=recipe.mask_threshold,
        roi, mask, scale=recipe.scale)
end

mask_parity_fields(a, b) = all(key -> isequal(getfield(a, key), getfield(b, key)),
    (:x, :y, :u, :v, :peak_ratio, :correlation_moment,
     :uncertainty_u, :uncertainty_v, :outliers, :mask, :correlation_planes))

@testset "Mask revisions retain exact bits and distinguish absence" begin
    H = HammerheadGUI.Hammerhead
    mktempdir() do directory
        paths = mask_parity_files(directory)
        seed = mask_parity_seed()
        record = ExperimentRecord([(paths[2], paths[1]), (paths[1], paths[2])],
                                  mask_parity_recipe(Float32, :cpu))
        old_output = joinpath(directory, "earlier-output.jld2")
        write(old_output, "unpublished earlier attempt")
        push!(record.runs, ExperimentRun("b58f888f-9e9b-4e4d-ad85-5a8c4d5c4085",
            record.recipe.recipe_id, record.input_id, 1., 2., :failed, 0,
            old_output, nothing, deepcopy(record.creation_environment), "prior failure"))
        original = H._experiment_recipe_data(record.recipe)
        rc = RecipeRevisionController(record)
        @test rc.mask_draft[].enabled && rc.mask_draft[].raster == seed
        @test rc.mask_draft[].raster !== record.recipe.mask
        @test revision_recipe(rc).recipe_id == record.recipe.recipe_id
        @test isempty(revision_diff(rc))
        replacement = copy(seed); replacement[20:35, 45:52] .= true
        set_revision_mask!(rc; raster=replacement)
        replacement .= false
        candidate = revision_recipe(rc)
        @test candidate.mask[25, 48] && candidate.mask[3, 63] && !candidate.mask[3, 3]
        @test size(candidate.mask) == (56, 72) && candidate.mask isa BitMatrix
        data = H._experiment_recipe_data(candidate)
        @test all(k -> isequal(data[k], original[k]), setdiff(collect(keys(data)), ["mask"]))
        @test candidate.mask_threshold == .37
        @test H._experiment_recipe_data(record.recipe) == original
        fresh = revision_record(rc)
        @test fresh.pairs == record.pairs && fresh.input_id == record.input_id
        @test fresh.input_files == record.input_files && isempty(fresh.runs)
        @test isempty(fresh.record_paths)
        @test length(record.runs) == 1 && length(rc.original.runs) == 1
        enabled_id = candidate.recipe_id
        retained = copy(rc.mask_draft[].raster)
        set_revision_mask!(rc; enabled=false)
        @test revision_recipe(rc).mask === nothing && rc.mask_draft[].raster == retained
        absent_id = revision_recipe(rc).recipe_id
        set_revision_mask!(rc; enabled=true)
        @test revision_recipe(rc).recipe_id == enabled_id
        set_revision_mask!(rc; raster=falses(56, 72))
        @test revision_recipe(rc).recipe_id != absent_id
        @test !any(revision_recipe(rc).mask)
        reset_revision_mask!(rc)
        @test revision_recipe(rc).recipe_id == record.recipe.recipe_id
        @test rc.mask_draft[].raster == seed
        for imported in (nothing, falses(56, 72))
            original_record = ExperimentRecord([(paths[1], paths[2])],
                mask_parity_recipe(Float64, :cpu; mask=imported))
            c = RecipeRevisionController(original_record)
            set_revision_mask!(c; raster=seed, enabled=true)
            reset_revision_mask!(c)
            @test isequal(revision_recipe(c).mask, imported)
            @test c.mask_draft[].enabled === (imported !== nothing)
        end
    end
end

@testset "Seeded masks use ordered exclusion and hole operations" begin
    seed = mask_parity_seed()
    untouched = copy(seed)
    exclusion = [(40., 15.), (55., 15.), (55., 40.), (40., 40.)]
    hole = [(17., 18.), (48., 18.), (48., 31.), (17., 31.)]
    later = [(21., 22.), (24., 22.), (24., 26.), (21., 26.)]
    # An independent core rasterization of each polygon plus explicit algebra,
    # rather than the production editor's combined-raster helper as oracle.
    expected = copy(seed)
    expected .|= polygon_mask(size(seed), exclusion)
    expected .&= .!polygon_mask(size(seed), hole)
    expected .|= polygon_mask(size(seed), later)
    editor = MaskEditor(zeros(size(seed)); raster=seed,
        polygons=[exclusion, hole, later], holes=[false, true, false])
    @test polygon_mask(editor) == expected
    @test seed == untouched && editor.raster[] !== seed
    @test !expected[20, 18] && seed[20, 18] # Hole removes imported exclusion.
    @test expected[23, 22]                # Later exclusion overrides the hole.
    reordered = MaskEditor(zeros(size(seed)); raster=seed,
        polygons=[exclusion, later, hole], holes=[false, false, true])
    @test polygon_mask(reordered) != expected
    grow_mask!(editor, 2)
    expected = grow_mask(expected, 2)
    @test polygon_mask(editor) == expected && isempty(editor.polygons[])
    shrink_mask!(editor, 1)
    @test polygon_mask(editor) == shrink_mask(expected, 1)
    clear_polygons!(editor)
    @test !any(polygon_mask(editor))
    @test_throws ArgumentError MaskEditor(zeros(56, 72); raster=falses(72, 56))
    @test_throws ArgumentError MaskEditor(zeros(56, 72); raster=zeros(56, 72))
end

@testset "Revised full-image masks match direct ROI replay" begin
    H = HammerheadGUI.Hammerhead
    mktempdir() do directory
        paths = mask_parity_files(directory)
        for T in (Float32, Float64), backend in (:cpu, :ka)
            record = ExperimentRecord([(paths[2], paths[1]), (paths[1], paths[2])],
                                      mask_parity_recipe(T, backend))
            original = H._experiment_recipe_data(record.recipe)
            rc = RecipeRevisionController(record)
            replacement = mask_parity_seed()
            replacement[25:37, 45:52] .= true
            set_revision_mask!(rc; raster=replacement)
            destination = joinpath(directory, "revision-$T-$backend.jld2")
            save_recipe_revision!(rc, destination; async=false)
            @test rc.state[] === :completed
            revised = load_experiment(destination)
            recipe = revised.recipe
            @test isempty(revised.runs) && revised.input_id == record.input_id
            @test revised.pairs == record.pairs && recipe.mask == replacement
            @test size(recipe.mask) == (56, 72) && recipe.mask_threshold == .37
            data = H._experiment_recipe_data(recipe)
            @test all(k -> isequal(data[k], original[k]), setdiff(collect(keys(data)), ["mask"]))
            output = joinpath(directory, "replayed-$T-$backend.jld2")
            run = replay_experiment(revised; output)
            @test run.status === :completed && run.completed_pairs == 2
            index = ResultFile(output)
            for (i, pair) in enumerate(revised.pairs)
                a, b = (load_image(T, revised.input_files[k]["path"]) for k in pair)
                ca, cb = mask_parity_condition(a, recipe), mask_parity_condition(b, recipe)
                raw = index[i]
                direct = mask_parity_direct(ca, cb, recipe)
                @test raw isa PIVResult{T} && mask_parity_fields(raw, direct)
                @test any(raw.mask) && any(.!raw.mask) && any(isfinite, raw.u[.!raw.mask])
                roi = recipe.roi
                local_result = mask_parity_direct(ca[roi.rows, roi.cols], cb[roi.rows, roi.cols], recipe;
                    roi=nothing, mask=replacement[roi.rows, roi.cols])
                @test raw.x == local_result.x .+ T(first(roi.cols)-1)
                @test raw.y == local_result.y .+ T(first(roi.rows)-1)
                @test isequal(raw.u, local_result.u) && isequal(raw.mask, local_result.mask)
                @test ca[roi.rows, roi.cols] != highpass_filter(
                    subtract_background(a[roi.rows, roi.cols],
                        recipe.preprocessing[1].options["background"][roi.rows, roi.cols]); sigma=1.6)
                @test raw.scale.pixel_size == .025 && raw.scale.dt == .01
                @test recipe.mask == replacement && record.recipe.mask == mask_parity_seed()
            end
        end
    end
end

@testset "Mask import embeds thresholded full-frame bits" begin
    mktempdir() do directory
        paths = mask_parity_files(directory)
        record = ExperimentRecord([(paths[1], paths[2])], mask_parity_recipe(Float32, :cpu))
        rc = RecipeRevisionController(record)
        replacement_path = joinpath(directory, "replacement.png")
        values = [mod(7r+11c, 5)/4 for r in 1:56, c in 1:72]
        save(replacement_path, Gray{N0f8}.(values))
        expected = load_mask(replacement_path; threshold=.6, invert=true)
        load_revision_mask!(rc, replacement_path; threshold=.6, invert=true, async=false)
        @test rc.state[] === :completed && rc.mask_draft[].raster == expected
        @test rc.mask_draft[].enabled && revision_recipe(rc).mask_threshold == .37
        @test HammerheadGUI.Hammerhead._artifact_local_path(replacement_path) in rc.protected_paths[]
        before = read(replacement_path)
        save_recipe_revision!(rc, replacement_path; async=false)
        @test rc.state[] === :failed && read(replacement_path) == before
        # The accepted raster is embedded, not a new external file dependency.
        rm(replacement_path)
        destination = joinpath(directory, "embedded-mask.jld2")
        save_recipe_revision!(rc, destination; async=false)
        @test rc.state[] === :completed && load_experiment(destination).recipe.mask == expected
        previous = copy(rc.mask_draft[].raster)
        bad_path = joinpath(directory, "cropped-mask.png")
        save(bad_path, Gray{N0f8}.(zeros(44, 56)))
        load_revision_mask!(rc, bad_path; async=false)
        @test rc.state[] === :failed && rc.mask_draft[].raster == previous
        set_revision_mask!(rc; raster=falses(44, 56))
        @test_throws ArgumentError revision_recipe(rc)
        @test_throws ArgumentError set_revision_mask!(rc; raster=zeros(56, 72))
    end
end

@testset "Raw mask reference captures identity and never executes preprocessing scripts" begin
    mktempdir() do directory
        paths = mask_parity_files(directory)
        marker = joinpath(directory, "script-was-run")
        script_path = joinpath(directory, "never-run.jl")
        # Even evaluating the script, before its named entrypoint, would fail.
        write(script_path, "write(" * repr(marker) * ", \"ran\"); error(\"raw-reference loading evaluated a script\")")
        script = ScriptReference(script_path; entrypoint="never_called")
        source = ExperimentRecord([(paths[2], paths[1]), (paths[1], paths[2])],
            mask_parity_recipe(Float32, :cpu; script))
        rc = RecipeRevisionController(source)
        mc = RecipeMaskReferenceController()
        captured = revision_recipe(rc)
        observer = on(mc.running) do active
            active || return
            set_revision_preprocess!(rc, 2, Dict(:sigma=>"2.2"))
        end
        load_recipe_mask_reference!(mc, rc; pair_index=1, frame=:a, async=false)
        off(observer)
        bundle = mc.bundle[]
        @test mc.state[] === :completed && bundle !== nothing && !mc.running[]
        @test bundle.raw_image isa Matrix{Float32} && size(bundle.raw_image) == (56, 72)
        descriptor = source.input_files[source.pairs[1][1]]
        @test bundle.input_descriptor == descriptor && bundle.input_descriptor !== descriptor
        @test bundle.input_id == source.input_id && bundle.source_recipe_id == source.recipe.recipe_id
        @test bundle.pair_index == 1 && bundle.frame === :a && bundle.recipe_id == captured.recipe_id
        @test bundle.recipe_id != revision_recipe(rc).recipe_id
        @test bundle.raw_image == load_image(Float32, descriptor["path"])
        @test bundle.raw_image != mask_parity_condition(bundle.raw_image, captured)
        @test polygon_mask(bundle.editor) == mask_parity_seed() && isempty(bundle.editor.polygons[])
        @test !ispath(marker) && isfile(script_path)
        # Apply composes newer non-mask settings, while leaving loading identity
        # and original-coordinate ROI tied to the displayed reference.
        polygon = [(40., 15.), (55., 15.), (55., 40.), (40., 40.)]
        for (x, y) in polygon; add_vertex!(bundle.editor, x, y); end
        @test close_active!(bundle.editor)
        expected = mask_parity_seed() .| polygon_mask((56, 72), polygon)
        apply_revision_mask!(rc, mc; async=false)
        @test rc.state[] === :completed && revision_recipe(rc).mask == expected
        @test revision_recipe(rc).preprocessing[2].options["sigma"] == 2.2
        @test mc.bundle[] === bundle && bundle.recipe_id == captured.recipe_id
        @test revision_recipe(rc).external_preprocess.sha256 == script.sha256
        destination = joinpath(directory, "script-mask-revision.jld2")
        save_recipe_revision!(rc, destination; async=false)
        @test rc.state[] === :completed && isempty(load_experiment(destination).runs)
        @test load_experiment(destination).recipe.mask == expected
        # Changed acquisition bytes refuse a new raw reference and a new record.
        bytes = read(paths[2]); changed = copy(bytes); changed[end] ⊻= 0x01
        write(paths[2], changed)
        load_recipe_mask_reference!(mc, rc; pair_index=1, frame=:a, async=false)
        @test mc.state[] === :failed && mc.bundle[] === bundle
        save_recipe_revision!(rc, joinpath(directory, "changed-input.jld2"); async=false)
        @test rc.state[] === :failed && !ispath(joinpath(directory, "changed-input.jld2"))
        write(paths[2], bytes)
        rm(paths[2])
        load_recipe_mask_reference!(mc, rc; pair_index=1, frame=:a, async=false)
        @test mc.state[] === :failed && mc.bundle[] === bundle
        # Offline draft inspection is still possible; it neither decodes nor runs scripts.
        @test revision_recipe(rc).mask == expected
    end
end

@testset "Mask editing refuses stale sessions and unfinished drawing" begin
    mktempdir() do directory
        paths = mask_parity_files(directory)
        source = ExperimentRecord([(paths[1], paths[2])], mask_parity_recipe(Float64, :cpu))
        rc = RecipeRevisionController(source); mc = RecipeMaskReferenceController()
        load_recipe_mask_reference!(mc, rc; frame=:b, async=false)
        bundle = mc.bundle[]
        add_vertex!(bundle.editor, 40, 20)
        @test_throws ArgumentError apply_revision_mask!(rc, mc; async=false)
        @test rc.mask_draft[].raster == mask_parity_seed()
        undo_vertex!(bundle.editor)
        changed = mask_parity_seed(); changed[10, 40] = true
        set_revision_mask!(rc; raster=changed)
        @test_throws ArgumentError apply_revision_mask!(rc, mc; async=false)
        reset_revision_mask!(rc)
        apply_revision_mask!(rc, mc; async=false)
        @test rc.state[] === :completed && revision_recipe(rc).mask == mask_parity_seed()
        other = ExperimentRecord([(paths[2], paths[1])], mask_parity_recipe(Float64, :cpu))
        @test_throws ArgumentError apply_revision_mask!(RecipeRevisionController(other), mc; async=false)
        # Clearing explicitly creates an enabled all-false raster, not nothing.
        clear_polygons!(bundle.editor)
        apply_revision_mask!(rc, mc; async=false)
        @test rc.state[] === :completed && rc.mask_draft[].enabled
        @test !any(revision_recipe(rc).mask) && revision_recipe(rc).mask !== nothing
        reset_revision_mask!(rc)
        @test rc.mask_draft[].raster == mask_parity_seed()
    end
end
