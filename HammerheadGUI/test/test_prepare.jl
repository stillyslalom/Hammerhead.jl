# The workflow's Prepare step (no GL, no Qt): editor ↔ settings sync, canvas
# gesture dispatch, and off-thread loading/previews with a manually drained
# deliver queue. Uses the `imgA`/`imgB` synthetic pair from runtests.jl.

@testset "Prepare step controllers (no GL)" begin
    C = HammerheadGUI.Controllers
    frames = Any[imgA, imgB, imgA, imgB]
    A32 = Float32.(imgA)

    @testset "editors follow the frames" begin
        wf = PlanarWorkflow()
        ps = wf.prepare
        @test ps.mask[] === nothing && ps.roi[] === nothing && ps.scale[] === nothing
        @test !canvas_click!(set_step!(wf, :prepare), 10.0, 10.0)  # no frame: not used
        add_files!(wf.frames, frames)
        @test ps.mask[].size == ps.roi[].size == ps.scale[].size == (128, 128)
        @test ps.preview.image[] == A32 && ps.preview.image2[] == Float32.(imgB)
        me = ps.mask[]
        select_pair!(wf.frames, 2)                   # same size: editors kept
        @test ps.mask[] === me
        clear_files!(wf.frames)
        @test ps.mask[] === nothing && ps.preview.image[] === nothing
        add_files!(wf.frames, Any[imgA[1:64, :], imgB[1:64, :]])
        @test ps.mask[].size == (64, 128)
    end

    @testset "editors write the settings; settings reseed the editors" begin
        wf = PlanarWorkflow(files = frames)
        ps = wf.prepare
        set_step!(wf, :prepare)
        rev = ps.revision[]

        # preprocessing: preview steps are the recipe's preprocessing
        pp = ps.preview
        add_step!(pp, :highpass_filter)
        add_step!(pp, :clahe)
        @test edit_step_option!(wf, 2, "tiles", "4, 4")
        @test wf.preprocessing[] == pp.steps[] && wf.preprocessing[][2].options["tiles"] == [4, 4]
        @test pp.processed[] == recipe_preprocess(wf.preprocessing[])(A32)
        @test !edit_step_option!(wf, 1, "sigma", "-2")                 # rejected inline
        @test pp.error_step[] == 1 && occursin("sigma", pp.error[])
        @test wf.preprocessing[][1] == PreprocessStep(:highpass_filter)
        @test edit_step_option!(wf, 1, "sigma", "2") && isempty(pp.error[])
        @test ps.revision[] > rev

        # mask: committed polygons become the mask (in-progress ones do not)
        set_prepare_page!(wf, :mask)
        for (x, y) in ((20.0, 10.0), (60.0, 10.0), (60.0, 50.0))
            @test canvas_click!(wf, x, y)
        end
        @test wf.mask[] === nothing
        @test canvas_alt_click!(wf)                                     # close
        expected = polygon_mask((128, 128), [(20, 10), (60, 10), (60, 50)])
        @test wf.mask[] == expected
        @test canvas_click!(wf, 50.0, 20.0) && ps.mask[].selected[] == 1  # select
        @test canvas_key!(wf, :delete) && wf.mask[] === nothing
        @test !canvas_key!(wf, :delete) && !canvas_alt_click!(wf)       # nothing to use

        # ROI: two clicks; numeric bounds with inline errors
        set_prepare_page!(wf, :roi)
        @test canvas_click!(wf, 10.0, 20.0) && wf.roi[] === nothing
        @test canvas_alt_click!(wf) && ps.roi[].anchor[] === nothing    # cancel corner
        canvas_click!(wf, 10.0, 20.0); canvas_click!(wf, 100.0, 110.0)
        @test wf.roi[] == ROI(20:110, 10:100)
        @test wf.passes.image_size[] == (91, 91)                       # presets follow the ROI
        @test !edit_roi!(wf, "1", "200", "1", "10") && occursin("rows", ps.roi_error[])
        @test wf.roi[] == ROI(20:110, 10:100)
        @test edit_roi!(wf, "9", "120", "5", "124") && isempty(ps.roi_error[])
        @test wf.roi[] == ROI(9:120, 5:124)

        # scale: typed fields and a measured line
        set_prepare_page!(wf, :scale)
        @test edit_scale!(wf, :dt, "0.001") && wf.scale[] == PhysicalScale(1.0, 0.001, "px", "frame")
        @test edit_scale!(wf, :time_unit, "s")
        @test !edit_scale!(wf, :pixel_size, "-1") && !isempty(ps.scale_error[])
        canvas_click!(wf, 10.0, 20.0); canvas_click!(wf, 10.0, 70.0)    # 50 px
        @test edit_scale!(wf, :separation, "25") && isempty(ps.scale_error[])
        @test wf.scale[] == PhysicalScale(0.5, 0.001, "mm", "s")
        @test edit_scale!(wf, :length_unit, "cm") && wf.scale[].length_unit == "cm"
        @test ps.scale[].length_unit[] == "cm"
        @test canvas_key!(wf, :backspace) && length(ps.scale[].points[]) == 1
        @test wf.scale[].pixel_size == 0.5                              # kept
        clear_scale!(wf)
        @test wf.scale[] === nothing && isempty(ps.scale[].points[])
        @test_throws ArgumentError set_scale_field!(wf, :speed, 1)
        # readable scale text; long decimals shortened for text fields
        @test scale_description(nothing) == "no scale: results stay in pixels and frames"
        @test scale_description(PhysicalScale(1 / 170, 1.0, "mm", "frame")) ==
              "0.005882 mm per pixel · 1 frame between exposures"
        @test scale_description(PhysicalScale(0.02, 2.0, "mm", "frame")) ==
              "0.02 mm per pixel · 2 frames between exposures"
        @test scale_description(PhysicalScale(0.5, 1e-3, "mm", "s")) ==
              "0.5 mm per pixel · 0.001 s between exposures"
        @test C.display_number(1 / 170) == "0.00588235" && C.display_number(2.5) == "2.5"
        @test C.display_number(3.0) == "3.0" && C.display_number(7) == "7"
        @test C.step_options(PreprocessStep(:highpass_filter; sigma = 1 / 3)) == ["sigma" => "0.333333"]

        # a recipe opened from outside reseeds every editor, and nothing
        # writes back: the recipe survives unchanged
        mask = falses(128, 128); mask[1:10, 1:10] .= true
        recipe = PIVRecipe(multipass_parameters([48, 24]);
                           preprocessing = [PreprocessStep(:clahe; tiles = (2, 3), nbins = 64),
                                            PreprocessStep(:subtract_background; background = imgA)],
                           mask, roi = ROI(9:120, 5:124),
                           scale = PhysicalScale(0.01, 1e-3, "mm", "s"))
        load_settings!(wf, recipe)
        @test workflow_recipe(wf) == recipe && !settings_modified(wf)
        @test pp.steps[] == recipe.preprocessing
        @test ps.mask[].raster[] == mask && isempty(ps.mask[].polygons[])
        @test ps.roi[].roi[] == recipe.roi
        @test ps.scale[].length_unit[] == "mm" && ps.scale[].dt[] == 1e-3
        @test pp.processed[] == recipe_preprocess(recipe)(A32)

        # new polygons go on top of a loaded mask
        set_prepare_page!(wf, :mask)
        for (x, y) in ((100.0, 100.0), (120.0, 100.0), (120.0, 120.0)); canvas_click!(wf, x, y); end
        canvas_alt_click!(wf)
        @test wf.mask[][5, 5] && wf.mask[][105, 110] && settings_modified(wf)
        load_settings!(wf, recipe)
        @test wf.mask[] == mask && polygon_mask(ps.mask[]) == mask

        # saving and reopening keeps everything, including CLAHE tiles
        mktempdir() do dir
            path = save_settings(wf, joinpath(dir, "prep.jld2"))
            wf2 = PlanarWorkflow(files = frames)
            load_settings!(wf2, path)
            @test workflow_recipe(wf2) == recipe
            @test wf2.prepare.preview.steps[][1].options["tiles"] == [2, 3]
            # mask images round-trip through the mask page's file actions
            mpath = save_mask_file(wf2, joinpath(dir, "mask.png"))
            wf3 = PlanarWorkflow(files = frames)
            load_mask_file!(wf3, mpath)
            @test wf3.mask[] == mask && wf3.prepare.mask[].raster[] == mask
            FileIO = C.FileIO
            FileIO.save(joinpath(dir, "small.png"), C.Gray.(falses(8, 8)))
            @test_throws DimensionMismatch load_mask_file!(wf3, joinpath(dir, "small.png"))
        end

        # a mask of another size stays in the settings and is flagged
        wf.mask[] = falses(64, 64)
        @test !has_mask(ps.mask[]) && step_status(wf, :prepare)[1] === :attention
        wf.mask[] = nothing
        @test step_status(wf, :prepare)[1] === :ok
    end

    @testset "gesture dispatch" begin
        wf = PlanarWorkflow(files = frames)
        ps = wf.prepare
        @test !canvas_click!(wf, 64.0, 64.0)                  # Images step: not used
        set_step!(wf, :prepare)
        @test_throws ArgumentError set_prepare_page!(wf, :nope)
        # Preprocess: the click places the probe (window at the accuracy defaults)
        @test canvas_click!(wf, 64.0, 64.0)
        res = ps.preview.probe_result[]
        @test res.du ≈ 3.0 atol = 0.3
        @test res.dv ≈ 2.0 atol = 0.3
        @test !canvas_click!(wf, NaN, 1.0)
        @test canvas_key!(wf, :escape) && ps.preview.probe[] === nothing
        @test !canvas_alt_click!(wf)
        # Mask: Backspace undoes a vertex, Escape cancels; leaving the page
        # drops an unfinished polygon
        set_prepare_page!(wf, :mask)
        canvas_click!(wf, 5.0, 5.0); canvas_click!(wf, 25.0, 5.0)
        @test canvas_key!(wf, :backspace) && length(ps.mask[].active[]) == 1
        @test canvas_key!(wf, :escape) && isempty(ps.mask[].active[])
        @test !canvas_key!(wf, :escape)
        canvas_click!(wf, 5.0, 5.0)
        set_prepare_page!(wf, :roi)
        @test isempty(ps.mask[].active[])
        # ROI: a pending corner is cancelled by Escape and by leaving the page
        canvas_click!(wf, 5.0, 5.0)
        @test canvas_key!(wf, :escape) && ps.roi[].anchor[] === nothing
        canvas_click!(wf, 5.0, 5.0)
        set_prepare_page!(wf, :scale)
        @test ps.roi[].anchor[] === nothing && wf.roi[] === nothing
        # Scale: clicks place points, right-click drops the line
        canvas_click!(wf, 5.0, 5.0)
        @test canvas_alt_click!(wf) && isempty(ps.scale[].points[])
        @test !canvas_alt_click!(wf)
        # Other steps keep their clicks
        set_step!(wf, :passes)
        @test !canvas_click!(wf, 5.0, 5.0) && !canvas_alt_click!(wf) && !canvas_key!(wf, :escape)
    end

    @testset "background estimate" begin
        wf = PlanarWorkflow(files = frames)
        ps = wf.prepare
        add_step!(ps.preview, :highpass_filter)
        estimate_background!(wf; frames = 3)
        @test !ps.background_running[] && occursin("3 frames", ps.status[])
        bg = wf.preprocessing[][1]
        @test bg.operation === :subtract_background
        @test bg.options["background"] == min.(imgA, imgB)
        @test wf.preprocessing[][2] == PreprocessStep(:highpass_filter)
        empty = PlanarWorkflow()
        estimate_background!(empty)
        @test empty.prepare.status[] == "add frames first"
    end

    @testset "window mode: work runs off the calling thread" begin
        # Deliveries are queued (as the window's tick would) and drained by hand.
        queue = Any[]
        lk = ReentrantLock()
        wf = PlanarWorkflow(deliver = f -> lock(() -> push!(queue, f), lk))
        wf.spawn[] = true
        drain!() = foreach(f -> f(), lock(() -> splice!(queue, 1:length(queue)), lk))
        waitfor(n) = timedwait(() -> lock(() -> length(queue) >= n, lk), 60) === :ok

        dir = mktempdir()
        paths = [joinpath(dir, "frame$i.png") for i in 1:4]
        for (p, im) in zip(paths, (imgA, imgB, imgA, imgB))
            C.FileIO.save(p, C.Gray.(clamp.(im ./ maximum(im), 0, 1)))
        end
        add_files!(wf.frames, paths)
        # nothing was read on this thread: the pair is loading
        @test isempty(wf.frames.cache) && pair_loading(wf.frames)
        @test frames_problem(wf.frames) == "loading pair 1…"
        @test step_status(wf, :images) == (:busy, "loading pair 1…")
        @test frame_size(wf.frames) === nothing && frames_summary(wf.frames) == "4 frames · 2 pairs"
        test_pair!(wf)                                        # refuses while loading
        @test wf.test.status[] == "loading pair 1…" && !wf.test.running[]
        # a newer pair makes the first load stale
        select_pair!(wf.frames, 2)
        @test waitfor(2)
        drain!()
        @test !pair_loading(wf.frames) && wf.frames.loaded[] == 1
        @test haskey(wf.frames.cache, paths[3]) && !haskey(wf.frames.cache, paths[1])
        @test frame_size(wf.frames) == (128, 128) && wf.prepare.mask[].size == (128, 128)
        @test wf.passes.image_size[] == (128, 128)
        @test frames_problem(wf.frames) === nothing
        a = wf.frames.cache[paths[3]]

        # previews and probes are computed on workers and land through deliver
        pp = wf.prepare.preview
        @test pp.processed[] === a                            # no steps: the raw frame
        add_step!(pp, :highpass_filter)
        @test pp.processed[] === a                            # old preview until delivery
        add_step!(pp, :invert_image)
        @test waitfor(2)
        drain!()                                              # first result is stale
        @test pp.processed[] == recipe_preprocess(pp.steps[])(a)
        set_step!(wf, :prepare)
        canvas_click!(wf, 64.0, 64.0)
        @test pp.probe_result[] === nothing
        @test waitfor(1)
        drain!()
        @test pp.probe_result[] !== nothing
        clear_probe!(pp)

        # background estimate
        estimate_background!(wf; frames = 2)
        @test wf.prepare.background_running[]
        @test waitfor(1)
        drain!()
        @test !wf.prepare.background_running[]
        @test first(wf.preprocessing[]).operation === :subtract_background
        @test waitfor(1)                                      # the preview follows
        drain!()
        @test pp.processed[] == recipe_preprocess(pp.steps[])(a)

        # an unreadable pair reports its error and is not requested again
        add_files!(wf.frames, [joinpath(dir, "missing1.png"), joinpath(dir, "missing2.png")])
        select_pair!(wf.frames, 3)
        @test waitfor(1)
        drain!()
        @test occursin("cannot read pair 3", frames_problem(wf.frames)) && !pair_loading(wf.frames)
        @test step_status(wf, :images)[1] === :todo
        frames_problem(wf.frames)
        @test lock(() -> isempty(queue), lk)

        # a closed window abandons pending jobs and catches up inline
        select_pair!(wf.frames, 1)
        @test pair_loading(wf.frames)
        wf.spawn[] = false
        C._abandon_jobs!(wf)
        @test !pair_loading(wf.frames) && frame_size(wf.frames) == (128, 128)
        @test pp.image[] === wf.frames.cache[paths[1]]
    end
end
