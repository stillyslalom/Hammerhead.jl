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
            path = save_settings(wf, joinpath(dir, "s.jld2"))
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

    @testset "test pair matches apply_recipe; staleness" begin
        wf = PlanarWorkflow(files = frames)
        fill_preset!(wf.passes, :low)
        @test step_status(wf, :test) == (:todo, "not tested")
        test_pair!(wf; spawn = false)
        res = wf.test.result[]
        @test res isa PIVResult
        direct = apply_recipe(workflow_recipe(wf), [current_pair(wf.frames)]; progress = false)[1]
        @test isequal(res.u, direct.u) && isequal(res.v, direct.v)
        @test !test_stale(wf) && step_status(wf, :test)[1] === :ok
        s = test_summary(wf.test)
        @test s.valid > 0 && s.valid_fraction > 0.9
        @test median(filter(!isnan, res.u)) ≈ 3.0 atol = 0.3
        @test any(l -> startswith(l, "Valid vectors"), summary_lines(s))
        set_option!(wf.passes, :correlation, :phase)
        @test test_stale(wf) && step_status(wf, :test)[1] === :attention
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
            @test occursin("cancelled after", wf.run.status[]) && length(wf.run.completed[]) < 3
            @test nframes(wf.explorer[]) == length(wf.run.completed[])
        end
        # in-memory run (no output file)
        wf.run.output_path[] = ""
        empty!(wf.run.completed.listeners)
        start_run!(wf; spawn = false)
        @test wf.results_path[] === nothing && nframes(wf.explorer[]) == 3
        wf.passes.mode[] = :ensemble
        start_run!(wf; spawn = false)
        @test occursin("ensemble", wf.run.status[])
        # threaded test with an immediate deliver still completes
        wf.passes.mode[] = :sequence
        test_pair!(wf)
        t0 = time()
        while wf.test.running[] && time() - t0 < 120
            sleep(0.05)
        end
        @test wf.test.result[] isa PIVResult
    end
end
