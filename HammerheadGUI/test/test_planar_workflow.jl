# Planar workflow window controllers (no GL, no Qt). Uses the `imgA`/`imgB`
# synthetic pair from runtests.jl (uniform flow u = 3, v = 2 px).

@testset "Planar workflow controllers (no GL)" begin
    C = HammerheadGUI.Controllers
    frames = Any[imgA, imgB, imgA, imgB, imgA, imgB]

    @testset "FrameSet" begin
        fs = FrameSet()
        @test frames_problem(fs) == "add frames to analyze"
        @test frames_summary(fs) == "no frames" && npairs(fs) == 0
        add_files!(fs, frames)
        @test npairs(fs) == 3 && frames_problem(fs) === nothing
        @test frames_summary(fs) == "6 frames · 3 pairs · 128×128 px"
        set_pair_mode!(fs, :chained)
        @test npairs(fs) == 5
        select_pair!(fs, 9)
        @test fs.pair[] == 5                       # clamped
        set_pair_mode!(fs, :paired)
        @test fs.pair[] == 3                       # re-clamped on fewer pairs
        @test pair_images(fs)[1] == Float32.(imgA) && shown_image(fs) == Float32.(imgA)
        show_frame!(fs, :b)
        @test shown_image(fs) == Float32.(imgB)
        add_files!(fs, Any[imgA])
        @test occursin("even", frames_problem(fs)) || occursin("odd", frames_problem(fs))
        add_files!(fs, Any[rand(64, 64)])
        select_pair!(fs, 4)
        @test occursin("different sizes", frames_problem(fs))
        clear_files!(fs)
        @test npairs(fs) == 0 && fs.pair[] == 1
    end

    @testset "PassesEditor" begin
        pe = PassesEditor(image_size = (128, 128))
        @test pe.preset[] === :medium
        @test pe.passes[] == effort_schedule(:medium; image_size = (128, 128))
        fill_preset!(pe, :high)
        @test pe.passes[] == effort_schedule(:high; image_size = (128, 128))
        set_analysis_size!(pe, (48, 200))          # linked preset follows the size
        @test pe.passes[] == effort_schedule(:high; image_size = (48, 200))
        set_mode!(pe, :ensemble)
        @test pe.passes[] == effort_schedule(:high; ensemble = true, image_size = (48, 200))
        set_mode!(pe, :sequence)
        set_analysis_size!(pe, (256, 256))

        n = length(pe.passes[])
        @test set_pass!(pe, n, :window, 24)
        @test pe.preset[] === nothing               # edits unlink the preset
        p = pe.passes[][n]
        @test p.window_size == (24, 24) && p.search_area_size == (24, 24)
        @test p.overlap == (12, 12)                 # 50 % kept
        @test set_pass!(pe, n, :overlap, 75) && pe.passes[][n].overlap == (18, 18)
        @test set_pass!(pe, n, :iterations, 3) && pe.passes[][n].max_iterations == 3
        before = copy(pe.passes[])
        @test !set_pass!(pe, n, :search, 25)        # odd margin: rejected
        @test !isempty(pe.error[]) && pe.passes[] == before
        @test set_pass!(pe, n, :search, 32) && isempty(pe.error[])
        @test set_pass!(pe, n, :window, 16) && pe.passes[][n].search_area_size == (24, 24)
        set_analysis_size!(pe, (64, 64))            # unlinked: schedule unchanged
        @test pe.passes[][n].window_size == (16, 16)

        @test set_option!(pe, :correlation, :phase)
        @test all(q -> q.correlation_method === :phase, pe.passes[])
        @test set_option!(pe, :accuracy, false)
        @test option_value(pe, :accuracy) == false && all(q -> q.apodization === :none, pe.passes[])
        # padding and Gaussian weighting are separate options
        @test set_option!(pe, :padding, true) && option_value(pe, :padding)
        @test !option_value(pe, :apodization) && !option_value(pe, :accuracy)
        @test set_option!(pe, :apodization, true) && option_value(pe, :accuracy)
        @test set_option!(pe, :apodization, false) && all(q -> q.apodization === :none && q.padding, pe.passes[])
        # validation and replacement
        @test set_option!(pe, :uod_enable, false) && all(q -> !q.uod_enable, pe.passes[])
        @test set_option!(pe, :uod_threshold, Symbol("2.5")) && option_value(pe, :uod_threshold) == 2.5
        @test set_option!(pe, :uod_neighborhood, 1) && option_value(pe, :uod_neighborhood) == 1
        @test set_option!(pe, :min_peak_ratio, Symbol("1.3")) && option_value(pe, :min_peak_ratio) == 1.3
        @test set_option!(pe, :replace_outliers, false) && !option_value(pe, :replace_outliers)
        @test set_option!(pe, :image_interpolation, :linear) &&
              all(q -> q.image_interpolation === :linear, pe.passes[])
        @test set_option!(pe, :predictor_interpolation, :cubic) &&
              option_value(pe, :predictor_interpolation) === :cubic
        @test !set_option!(pe, :image_interpolation, :quintic) && occursin("quintic", pe.error[])
        before = pe.passes[]
        @test !set_option!(pe, :uod_threshold, Symbol("abc")) && occursin("number", pe.error[])
        @test !set_option!(pe, :uod_neighborhood, 0) && pe.passes[] == before
        @test set_option!(pe, :uncertainty, true)
        @test pe.passes[][end].uncertainty && !pe.passes[][1].uncertainty
        @test !set_option!(pe, :subpixel, :nope)
        @test add_pass!(pe) && length(pe.passes[]) == n + 1
        @test remove_pass!(pe, 1) && length(pe.passes[]) == n
        while length(pe.passes[]) > 1
            remove_pass!(pe, 1)
        end
        @test !remove_pass!(pe, 1)
        rows = pass_rows(pe)
        @test rows[1].window == 16 && rows[1].overlap == 75.0
        @test occursin("custom", passes_summary(pe))
    end

    @testset "frames by folder and pattern" begin
        mktempdir() do dir
            for name in ("run_2.tif", "run_10.tif", "run_1.tif", "other.png")
                touch(joinpath(dir, name))
            end
            @test basename.(matching_files(dir, "run_*.tif")) == ["run_1.tif", "run_2.tif", "run_10.tif"]
            @test basename.(matching_files(dir, "run_?.tif")) == ["run_1.tif", "run_2.tif"]
            @test_throws ArgumentError matching_files(joinpath(dir, "missing"), "*")
            fs = FrameSet()
            set_frame_pattern!(fs, dir, "*.png")
            @test basename.(fs.pattern_matches[]) == ["other.png"] && isempty(fs.pattern_error[])
            set_frame_pattern!(fs, dir, "*.jpg")
            @test isempty(fs.pattern_matches[]) && occursin("no files", fs.pattern_error[])
            @test_throws ArgumentError add_matching!(fs)
            set_frame_pattern!(fs, joinpath(dir, "missing"), "*.tif")
            @test occursin("no folder", fs.pattern_error[])
            set_frame_pattern!(fs, dir, "run_*.tif")
            @test add_matching!(fs) == 3 && basename.(fs.files[]) == ["run_1.tif", "run_2.tif", "run_10.tif"]
        end
        @test infer_pattern(joinpath("d", "A001_1.tif"), joinpath("d", "A001_2.tif")) == ("d", "A*_*.tif")
        @test infer_pattern("cam1_00001.tif", "cam1_00002.tif")[2] == "cam1_*.tif"
        @test infer_pattern("run_0001_a.tif", "run_0001_b.tif")[2] == "run_*_*.tif"
        @test infer_pattern("x.tif", "longer.tif")[2] == "*.tif"
    end

    @testset "correlation probe on Passes; ruler image" begin
        wf = PlanarWorkflow(; files = frames)
        set_step!(wf, :passes)
        pp = wf.prepare.preview
        @test passes_probe_available(wf)
        @test canvas_click!(wf, 64.0, 64.0)
        @test pp.probe[] !== nothing && pp.probe_window[] == last(wf.passes.passes[]).window_size[1]
        @test pp.probe_result[] !== nothing
        @test canvas_alt_click!(wf) && pp.probe[] === nothing
        set_mode!(wf.passes, :ptv)
        @test !passes_probe_available(wf) && !canvas_click!(wf, 64.0, 64.0)
        # the scale measured on a separate ruler image
        set_step!(wf, :prepare); set_prepare_page!(wf, :scale)
        ruler = zeros(50, 80)
        load_ruler!(wf, ruler)
        @test wf.prepare.ruler[] == Float32.(ruler) && wf.prepare.ruler_name[] == "image"
        canvas_click!(wf, 10.0, 10.0); canvas_click!(wf, 70.0, 10.0)
        @test edit_scale!(wf, :separation, "30") && wf.scale[].pixel_size ≈ 0.5
        clear_ruler!(wf)
        @test wf.prepare.ruler[] === nothing && isempty(wf.prepare.scale[].points[])
        @test wf.scale[].pixel_size ≈ 0.5                       # the measured scale stays
    end

    @testset "run and results go stale; GPU backend" begin
        wf = PlanarWorkflow(; files = frames)
        fill_preset!(wf.passes, :low)
        @test step_status(wf, :run) == (:todo, "not run") && !run_stale(wf)
        start_run!(wf; spawn = false)
        @test step_status(wf, :run)[1] === :ok && step_status(wf, :results)[1] === :ok
        set_option!(wf.passes, :correlation, :phase)
        @test run_stale(wf)
        @test step_status(wf, :run)[1] === :attention && step_status(wf, :results)[1] === :attention
        set_option!(wf.passes, :correlation, :cross)
        @test !run_stale(wf) && step_status(wf, :run)[1] === :ok
        add_files!(wf.frames, frames[1:2])                       # a new pair
        @test run_stale(wf) && step_status(wf, :results)[1] === :attention

        # the hardware-free :ka backend stands in for a GPU
        wf = PlanarWorkflow(; files = frames)
        fill_preset!(wf.passes, :low)
        @test_throws ArgumentError set_backend!(wf.passes, :nonexistent)
        set_backend!(wf.passes, :ka)
        test_pair!(wf; spawn = false)
        direct = only(apply_recipe(workflow_recipe(wf), [current_pair(wf.frames)];
                                   backend = :ka, progress = false))
        @test isequal(wf.test.result[].u, direct.u) && !test_stale(wf)
        start_run!(wf; spawn = false)
        @test wf.run.options[] == (; backend = :ka) && !run_stale(wf)
        set_backend!(wf.passes, :cpu)
        @test test_stale(wf) && run_stale(wf)
        set_backend!(wf.passes, :ka)
        set_option!(wf.passes, :subpixel, :gauss2d)
        @test occursin("on the GPU", workflow_problem(wf))
        @test occursin("switch the GPU off", workflow_problem(wf))
        @test gpu_problem(wf) == "the 2-D Gaussian subpixel fit is CPU-only"
        set_mode!(wf.passes, :ptv)                               # particles: CPU regardless
        @test workflow_problem(wf) === nothing
        use_gpu!(wf, false)
        @test wf.passes.backend[] === :cpu
        set_mode!(wf.passes, :sequence)
        # on the CPU, :ka answers for the GPU backends (the switch greys out)
        @test gpu_problem(wf) == "the 2-D Gaussian subpixel fit is CPU-only"
        set_option!(wf.passes, :subpixel, :gauss3)
        @test gpu_problem(wf) === nothing
        set_option!(wf.passes, :image_interpolation, :linear)
        @test gpu_problem(wf) == "linear image interpolation is CPU-only"
        if isempty(gpu_packages())
            use_gpu!(wf, true; spawn = false)
            @test occursin("no GPU package", wf.passes.gpu_status[]) && wf.passes.backend[] === :cpu
        end
    end

    @testset "pair bar: representative pair and shown result" begin
        wf = PlanarWorkflow(; files = frames)
        @test pair_position(wf) == (1, 3)
        step_pair!(wf, 1)
        @test wf.frames.pair[] == 2
        step_pair!(wf, 5)
        @test pair_position(wf) == (3, 3)
        start_run!(wf; spawn = false)
        set_step!(wf, :results)
        @test pair_position(wf) == (1, 3)              # results: independent of the pair
        go_to_pair!(wf, 2)
        @test wf.explorer[].frame[] == 2 && wf.frames.pair[] == 3
        step_pair!(wf, -1)
        @test pair_position(wf) == (1, 3)
        set_step!(wf, :images)
        @test pair_position(wf) == (3, 3)
    end

    @testset "PlanarWorkflow recipe round trip" begin
        wf = PlanarWorkflow(files = frames)
        @test wf.passes.image_size[] == (128, 128)
        @test settings_modified(wf)
        r = workflow_recipe(wf)
        @test r.passes == effort_schedule(:medium; image_size = (128, 128))

        mask = falses(128, 128); mask[1:10, 1:10] .= true
        loaded = PIVRecipe(multipass_parameters([48, 24]; final = (max_iterations = 2,));
                           preprocessing = [PreprocessStep(:highpass_filter; sigma = 5)],
                           mask, roi = ROI(9:120, 5:124),
                           scale = PhysicalScale(0.01, 1e-3, "mm", "s"),
                           image_type = Float32, predictor_smoothing = false,
                           mask_threshold = 0.25)
        load_settings!(wf, loaded)
        @test workflow_recipe(wf) == loaded         # every field survives
        @test !settings_modified(wf)
        @test wf.passes.preset[] === nothing
        mktempdir() do dir
            path = save_settings(wf, joinpath(dir, "s.toml"))
            wf2 = PlanarWorkflow(files = frames)
            load_settings!(wf2, path)
            @test workflow_recipe(wf2) == loaded && wf2.settings_path[] == path
        end
        set_pass!(wf.passes, 1, :window, 64)
        @test settings_modified(wf)
        # ROI changes resize a linked preset to the region
        wf3 = PlanarWorkflow(files = frames)
        wf3.roi[] = ROI(1:64, 1:96)
        @test wf3.passes.passes[] == effort_schedule(:medium; image_size = (64, 96))
    end

    @testset "Prepare and Passes stay grey until frames form pairs" begin
        wf = PlanarWorkflow()
        @test step_status(wf, :prepare) == (:todo, "full image, no preprocessing")
        @test step_status(wf, :passes)[1] === :todo
        add_files!(wf.frames, frames)
        @test step_status(wf, :prepare)[1] === :ok && step_status(wf, :passes)[1] === :ok
        sw = StereoWorkflow()
        @test step_status(sw, :prepare)[1] === :todo && step_status(sw, :passes)[1] === :todo
    end

    @testset "test pair matches apply_recipe; staleness" begin
        wf = PlanarWorkflow(files = frames)
        fill_preset!(wf.passes, :low)
        @test step_status(wf, :passes) == (:ok, passes_summary(wf.passes))     # not tested
        test_pair!(wf; spawn = false)
        res = wf.test.result[]
        @test res isa PIVResult
        direct = apply_recipe(workflow_recipe(wf), [current_pair(wf.frames)]; progress = false)[1]
        @test isequal(res.u, direct.u) && isequal(res.v, direct.v)
        s = test_summary(wf.test)
        @test !test_stale(wf) &&
              step_status(wf, :passes) == (:ok, passes_summary(wf.passes) * " · " * test_brief(s))
        @test s.valid > 0 && s.valid_fraction > 0.9
        @test median(filter(!isnan, res.u)) ≈ 3.0 atol = 0.3
        @test any(l -> startswith(l, "Valid vectors"), summary_lines(s))
        set_option!(wf.passes, :correlation, :phase)
        @test test_stale(wf) && step_status(wf, :passes) ==
              (:attention, passes_summary(wf.passes) * " · settings changed since the test")
        test_pair!(wf; spawn = false)
        @test !test_stale(wf) && wf.test.previous[] !== nothing
        @test any(l -> occursin("pts)", l), summary_lines(test_summary(wf.test); previous = wf.test.previous[]))
        select_pair!(wf.frames, 2)
        @test test_stale(wf)
        # deliver: background updates are handed to the shell's queue
        queue = Function[]
        wf.deliver[] = f -> push!(queue, f)
        test_pair!(wf; spawn = false)
        @test wf.test.running[] && length(queue) == 1
        foreach(f -> f(), queue)
        @test !wf.test.running[] && !test_stale(wf)
        @test step_status(wf, :images) == (:ok, "6 frames · 3 pairs · 128×128 px")
        @test frames_problem(PlanarWorkflow().frames) !== nothing
        empty_wf = PlanarWorkflow()
        test_pair!(empty_wf; spawn = false)
        @test empty_wf.test.status[] == "add frames to analyze"
    end

    @testset "batch run, results, cancellation" begin
        wf = PlanarWorkflow(files = frames)
        fill_preset!(wf.passes, :low)
        mktempdir() do dir
            out = joinpath(dir, "run.jld2")
            wf.run.output_path[] = out
            start_run!(wf; spawn = false)
            @test !wf.run.running[] && length(wf.run.completed[]) == 3
            @test wf.run.progress[] == (3, 3) && startswith(wf.run.status[], "done: 3 pairs")
            @test wf.results_path[] == out && nframes(wf.explorer[]) == 3
            @test load_recipe(out) == workflow_recipe(wf)
            @test step_status(wf, :results) == (:ok, "run.jld2")
            # a results file opens its recipe as settings
            wf2 = PlanarWorkflow(files = frames)
            load_settings!(wf2, out)
            @test workflow_recipe(wf2) == workflow_recipe(wf)

            # cancel after the first pair: finished pairs stay
            wf.run.output_path[] = joinpath(dir, "cancel.jld2")
            on(wf.run.completed) do c
                length(c) == 1 && cancel_run!(wf)
            end
            start_run!(wf; spawn = false)
            @test occursin("canceled after", wf.run.status[]) && length(wf.run.completed[]) < 3
            @test nframes(wf.explorer[]) == length(wf.run.completed[])
        end
        # in-memory run (no output file)
        wf.run.output_path[] = ""
        empty!(wf.run.completed.listeners)
        start_run!(wf; spawn = false)
        @test wf.results_path[] === nothing && nframes(wf.explorer[]) == 3
        # threaded test with an immediate deliver still completes
        test_pair!(wf)
        t0 = time()
        while wf.test.running[] && time() - t0 < 120
            sleep(0.05)
        end
        @test wf.test.result[] isa PIVResult
    end

    @testset "per-frame mask images" begin
        wf = PlanarWorkflow(files = frames)
        fill_preset!(wf.passes, :low)
        mktempdir() do dir
            paths = map(1:6) do k
                m = falses(128, 128)
                k == 4 && (m[:, 1:48] .= true)           # frame 4: the second pair's frame B
                p = joinpath(dir, "mask_$k.png")
                C.FileIO.save(p, C.Gray.(m))
                p
            end
            add_files!(wf.frame_masks, paths[1:5])
            @test occursin("5 mask images for 6 frames", workflow_problem(wf))
            add_files!(wf.frame_masks, paths[6:6])
            @test workflow_problem(wf) === nothing
            @test representative_mask(wf) == falses(128, 128)
            select_pair!(wf.frames, 2)
            @test wf.frame_masks.pair[] == 2 && count(representative_mask(wf)) == 128 * 48
            test_pair!(wf; spawn = false)
            res = wf.test.result[]
            direct = only(apply_recipe(workflow_recipe(wf), [current_pair(wf.frames)];
                                       masks = [(paths[3], paths[4])], progress = false))
            @test isequal(res.u, direct.u) && any(res.mask) && !test_stale(wf)
            select_pair!(wf.frames, 1)
            test_pair!(wf; spawn = false)
            @test !any(wf.test.result[].mask)
            out = joinpath(dir, "masked.jld2")
            wf.run.output_path[] = out
            start_run!(wf; spawn = false)
            @test [any(r.mask) for r in wf.run.completed[]] == [false, true, false]
            @test load_sources(out; masks = true)[2] == paths[3:4]
            # pair-by-pair modes only
            set_mode!(wf.passes, :ensemble)
            @test occursin("pair by pair", workflow_problem(wf))
            set_mode!(wf.passes, :sequence)
            clear_files!(wf.frame_masks)
            @test workflow_problem(wf) === nothing && test_stale(wf)
        end
    end

    @testset "recording type and what a new session discards" begin
        @test recording_type(PlanarWorkflow()) === :planar && recording_type(StereoWorkflow()) === :stereo
        p = multipass_parameters([32, 16])
        @test recording_type(PIVRecipe(p)) === nothing
        @test recording_type(PIVRecipe(p; roi = ROI(1:64, 1:64))) === :planar
        @test recording_type(PIVRecipe(p; mode = :ptv)) === :planar
        @test recording_type(PIVRecipe(p; preprocessing = (PreprocessStep[], PreprocessStep[]))) === :stereo
        mktempdir() do dir
            path = save_recipe(joinpath(dir, "s.toml"), PIVRecipe(p; mode = :tracking))
            @test recording_type(path) === :planar
            @test recording_type(joinpath(dir, "missing.toml")) === nothing
        end
        @test new_workflow(:stereo) isa StereoWorkflow
        @test_throws ArgumentError new_workflow(:tomo)

        wf = PlanarWorkflow()
        @test isempty(unsaved_work(wf)) && switch_question(wf, :stereo) === nothing
        add_files!(wf.frames, frames)
        set_option!(wf.passes, :correlation, :phase)
        @test unsaved_work(wf) == ["6 frames", "settings not saved to a file"]
        q = switch_question(wf, :stereo)
        @test q == "Start a new two-camera (stereo) session? This discards 6 frames and " *
                   "settings not saved to a file."
        mktempdir() do dir
            save_settings(wf, joinpath(dir, "s.toml"))
            @test unsaved_work(wf) == ["6 frames"]
        end
        start_run!(wf; spawn = false)
        @test "results kept only in memory" in unsaved_work(wf)
        @test "the camera calibration" ∉ unsaved_work(StereoWorkflow())
    end

    @testset "ensemble: test, run, progress, cancellation" begin
        wf = PlanarWorkflow(files = frames)
        fill_preset!(wf.passes, :low)
        set_mode!(wf.passes, :ensemble)
        npass = length(wf.passes.passes[])
        # an ensemble runs each pass once: the summary shows no repeats
        set_pass!(wf.passes, 1, :iterations, 3)
        @test !occursin("×", passes_summary(wf.passes)) && occursin("ensemble", passes_summary(wf.passes))
        test_pair!(wf; spawn = false)
        @test wf.test.result[] isa PIVResult
        @test length(only(wf.test.inputs[])) == 3 && step_status(wf, :passes)[1] === :ok
        select_pair!(wf.frames, 2)                     # the ensemble ignores the pair
        @test !test_stale(wf)
        add_files!(wf.frames, Any[imgA, imgB])         # but not the frames
        @test test_stale(wf)
        clear_files!(wf.frames)
        add_files!(wf.frames, frames)
        wf.roi[] = ROI(1:64, 1:64)                    # an ensemble takes no ROI
        @test occursin("clear the region", workflow_problem(wf))
        start_run!(wf; spawn = false)
        @test occursin("clear the region", wf.run.status[]) && isempty(wf.run.completed[])
        wf.roi[] = nothing
        fill_preset!(wf.passes, :low)
        mktempdir() do dir
            out = joinpath(dir, "ensemble.jld2")
            wf.run.output_path[] = out
            texts = String[]
            on(_ -> push!(texts, run_progress(wf.run)), wf.run.progress)
            start_run!(wf; spawn = false)
            @test !wf.run.running[] && length(wf.run.completed[]) == 1
            @test wf.run.status[] == "done: ensemble of 3 pairs → ensemble.jld2"
            @test wf.run.progress[] == (3npass, 3npass)
            @test "ensemble of 3 pairs · pass 1 of $npass · 1 of 3 pairs" in texts
            @test last(texts) == "ensemble of 3 pairs · pass $npass of $npass · 3 of 3 pairs"
            @test wf.results_path[] == out && nframes(wf.explorer[]) == 1
            @test step_status(wf, :results) == (:ok, "ensemble.jld2")
            @test load_recipe(out) == workflow_recipe(wf)
            direct = apply_recipe(workflow_recipe(wf), frame_pairs(wf.frames); progress = false)
            saved = only(load_results(out))
            @test saved isa PIVResult && isequal(saved.u, direct.u) && isequal(saved.v, direct.v)
            @test median(filter(!isnan, saved.u)) ≈ 3.0 atol = 0.3

            # canceling stops after the pair in flight and keeps no result
            empty!(wf.run.progress.listeners)
            canceled = joinpath(dir, "canceled.jld2")
            wf.run.output_path[] = canceled
            on(p -> p[1] == 1 && cancel_run!(wf), wf.run.progress)
            start_run!(wf; spawn = false)
            @test wf.run.status[] == "canceled; an ensemble keeps no partial result"
            @test isempty(wf.run.completed[]) && !isfile(canceled)
            @test wf.run.progress[][1] == 1
            @test wf.results_path[] == out              # the previous results stay
        end
        # in memory: the explorer holds the one result
        empty!(wf.run.progress.listeners)
        wf.run.output_path[] = ""
        start_run!(wf; spawn = false)
        @test wf.results_path[] === nothing && nframes(wf.explorer[]) == 1
        @test step_status(wf, :results) == (:ok, "1 result in memory")
        @test wf.run.status[] == "done: ensemble of 3 pairs"

        # progress text for a stereo ensemble (two cameras, three passes)
        rs = RunState()
        rs.mode[] = :ensemble; rs.pairs[] = 10; rs.cameras[] = 2
        rs.progress[] = (25, 60)
        @test run_progress(rs) == "ensemble of 10 pairs · camera 1 · pass 3 of 3 · 5 of 10 pairs"
        rs.progress[] = (35, 60)
        @test run_progress(rs) == "ensemble of 10 pairs · camera 2 · pass 1 of 3 · 5 of 10 pairs"
        rs.mode[] = :sequence; rs.progress[] = (4, 10)
        @test run_progress(rs) == "4 of 10 pairs"
    end

    @testset "particle modes: PTV" begin
        wf = PlanarWorkflow(; files = frames)
        pt = wf.particles
        set_step!(wf, :passes)
        @test step_label(wf, :passes) == "Passes" && pt.detected[] === nothing
        set_mode!(wf.passes, :ptv)
        @test step_label(wf, :passes) == "Particles"
        # the detection preview follows the settings
        @test pt.detected[] isa Particles && length(pt.detected[]) > 100
        @test occursin("detected on frame A", pt.detect_status[])
        n0 = length(pt.detected[])
        @test edit_particle_option!(pt, :threshold, string(0.6 * maximum(imgA)))
        @test 0 < length(pt.detected[]) < n0
        @test !edit_particle_option!(pt, :min_diameter, "20") && !isempty(pt.error[])
        @test !edit_particle_option!(pt, :predictor, "field")
        @test step_status(wf, :passes)[1] === :attention
        @test edit_particle_option!(pt, :threshold_k, 5) && isempty(pt.error[])
        @test edit_particle_option!(pt, :search_radius, "3") && edit_particle_option!(pt, :threshold, "auto")
        @test step_status(wf, :passes) == (:ok, particles_summary(pt, :ptv))
        @test particle_option(pt, :search_radius) == 3
        set_step!(wf, :images)
        @test pt.detected[] === nothing                       # only on the Passes step
        r = workflow_recipe(wf)
        @test r.mode === :ptv && r.ptv.search_radius == 3 && r.ptv_predictor === :piv
        # the test matches the pair exactly as the batch will
        test_pair!(wf; spawn = false)
        res = wf.test.result[]
        @test res isa PTVResult
        @test isequal(res.u, only(apply_recipe(r, [current_pair(wf.frames)]; progress = false)).u)
        s = test_summary(wf.test)
        @test s.kind === :ptv && s.matches > 50 && s.valid_fraction > 0.8
        @test median(res.u[.!res.outliers]) ≈ 3.0 atol = 0.2
        @test startswith(test_brief(s), "$(s.matches) matches")
        @test any(l -> startswith(l, "Matches:"), summary_lines(s))
        # the hint about the search radius depends on the predictor
        @test !any(l -> occursin("approach the search radius", l),
                   summary_lines(merge(s, (; displacement = 3.9, search_radius = 4.0))))
        @test any(l -> occursin("approach the search radius", l),
                  summary_lines(merge(s, (; predictor = :none, displacement = 3.9, search_radius = 4.0))))
        @test any(l -> occursin("far from the PIV prediction", l),
                  summary_lines(merge(s, (; residual = 2.5, search_radius = 4.0))))
        @test !test_stale(wf)
        edit_particle_option!(pt, :search_radius, 4)
        @test test_stale(wf)
        # a region blocks particle runs; the recipe never carries one
        wf.roi[] = ROI(1:64, 1:64)
        @test occursin("whole frames", workflow_problem(wf)) && workflow_recipe(wf).roi === nothing
        wf.roi[] = nothing
        mktempdir() do dir
            out = joinpath(dir, "ptv.jld2")
            wf.run.output_path[] = out
            start_run!(wf; spawn = false)
            @test length(wf.run.completed[]) == 3 && all(x -> x isa PTVResult, wf.run.completed[])
            @test wf.run.status[] == "done: 3 pairs → ptv.jld2"
            @test load_recipe(out) == workflow_recipe(wf)
            @test current_result(wf.explorer[]) isa PTVResult
            # settings round trip keeps the particle settings
            p = save_settings(wf, joinpath(dir, "settings.jld2"))
            w2 = PlanarWorkflow()
            load_settings!(w2, p)
            @test workflow_recipe(w2) == workflow_recipe(wf) && w2.particles.ptv[].search_radius == 4
            @test_throws ArgumentError load_settings!(StereoWorkflow(), p)
        end
    end

    @testset "particle modes: tracking" begin
        # a time-resolved recording: particles moving (1.0, 0.5) px per frame
        lcg = Ref(0x9e3779b97f4a7c15)
        uniform() = (lcg[] = lcg[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (lcg[] >> 11) / 2.0^53)
        pts = [(8 + 112 * uniform(), 8 + 112 * uniform()) for _ in 1:150]
        gauss! = Hammerhead.SyntheticData.generate_gaussian_particle!
        seq = map(0:11) do k
            img = zeros(128, 128)
            foreach(p -> gauss!(img, (p[1] + 1.0k, p[2] + 0.5k), 3.0), pts)
            img
        end
        tw = PlanarWorkflow(; files = seq, pair_mode = :chained)
        set_mode!(tw.passes, :tracking)
        set_particle_option!(tw.particles, :min_track_length, 4)
        set_particle_option!(tw.particles, :predictor, :none)
        @test occursin("tracks ≥ 4", step_status(tw, :passes)[2])
        test_pair!(tw; spawn = false)
        tr = tw.test.result[]
        @test tr isa TrackingResult && tr.n_frames == TRACKING_TEST_FRAMES
        @test length(tr.trajectories) > 50
        s = test_summary(tw.test)
        @test s.kind === :tracking && s.longest == TRACKING_TEST_FRAMES
        @test occursin("tracks through $(TRACKING_TEST_FRAMES) frames", test_brief(s))
        @test !test_stale(tw)
        select_pair!(tw.frames, 2)                # the test follows the representative pair
        @test test_stale(tw)
        start_run!(tw; spawn = false)
        full = only(tw.run.completed[])
        @test full isa TrackingResult && full.n_frames == 12
        @test occursin("tracks through 12 frames", tw.run.status[])
        @test run_progress(tw.run) == "tracking: frame step 11 of 11"
        @test current_result(tw.explorer[]) isa TrackingResult
        @test_throws ArgumentError load_settings!(StereoWorkflow(), workflow_recipe(tw))
    end
end
