# Stereo workflow controllers (no GL, no Qt): calibration → dewarpers,
# synchronized camera frames, the stereo recipe, test pair and batch run
# through the stereo apply_recipe, Prepare on the dewarped grid, and
# self-calibration.
#
# Fixture: two pinhole cameras at ±20° yaw (the core make_test_camera
# recipe), dot-plate images at z = -3/0/3 mm, and particle pairs on a light
# sheet offset to z = 0.8 mm, displaced by (1.0, 0.5) mm. Particle positions
# come from an inline 64-bit LCG (not an RNG stream, so the scenes are
# identical on every Julia version; a low-discrepancy sequence is too
# lattice-like for single-window correlation).

@testset "Stereo workflow controllers (no GL)" begin
    C = HammerheadGUI.Controllers
    gauss! = Hammerhead.SyntheticData.generate_gaussian_particle!

    function rig_camera(θdeg)
        θ = deg2rad(θdeg)
        R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
        camC = R' * [0.0, 0.0, -500.0]
        K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
        return PinholeCamera(K, R, -R * camC)
    end
    cams = (rig_camera(20.0), rig_camera(-20.0))
    zs = [-3.0, 0.0, 3.0]
    plates = map(cams) do cam
        [render_calibration_target(cam, (512, 512); spacing = 15.0, z = z,
                                   marker_square = (-30.0, -7.5), marker_triangle = (-15.0, -7.5))
         for z in zs]
    end
    sheet, disp = 0.8, (1.0, 0.5)
    # uniform points in ±35 mm from Knuth's MMIX LCG
    lcg = Ref(0x2545f4914f6cdd1d)
    uniform() = (lcg[] = lcg[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (lcg[] >> 11) / 2.0^53)
    points(n) = [(70 * uniform() - 35, 70 * uniform() - 35) for _ in 1:n]
    function stereo_pair(pts)
        map(cams) do cam
            A, B = zeros(512, 512), zeros(512, 512)
            for (X, Y) in pts
                pa = world_to_pixel(cam, (X, Y, sheet))
                pb = world_to_pixel(cam, (X + disp[1], Y + disp[2], sheet))
                gauss!(A, (pa[1], pa[2]), 5.0)
                gauss!(B, (pb[1], pb[2]), 5.0)
            end
            (A, B)
        end
    end
    acquisitions = [stereo_pair(points(1500)) for _ in 1:3]
    files1 = Any[f for a in acquisitions for f in a[1]]
    files2 = Any[f for a in acquisitions for f in a[2]]

    function add_plates!(cal)
        for k in 1:2, (img, z) in zip(plates[k], zs)
            add_plate!(cal, k, img, z)
        end
        set_calibration_option!(cal, :spacing, "15")
        set_calibration_option!(cal, :origin_offset, "30, 7.5")
        set_calibration_option!(cal, :grid_spacing, 0.3)
        return cal
    end

    # One calibrated workflow shared by the testsets below.
    wf = StereoWorkflow(; files1, files2)
    cal = wf.calibration
    dws = nothing

    @testset "calibration → dewarpers (inline)" begin
        @test step_status(wf, :calibration) == (:todo, "not calibrated")
        @test workflow_problem(wf) == "calibrate the cameras first (no dewarpers)"
        fit_calibration!(cal)
        @test cal.fit_status[] == "add calibration images for camera 1"
        add_plate!(cal, 1, plates[1][1], "-3")
        @test cal.plates[1][][1].z == -3.0
        clear_plates!(cal)
        @test isempty(cal.plates[1][]) && isempty(cal.plates[2][])
        for k in 1:2, (img, z) in zip(plates[k], zs)
            add_plate!(cal, k, img, z)
        end
        fit_calibration!(cal)
        @test cal.fit_status[] == "enter the dot spacing"
        @test !edit_calibration_option!(cal, :spacing, "-1") && occursin("spacing", cal.error[])
        @test !edit_calibration_option!(cal, :origin_offset, "30") && cal.origin_offset[] === nothing
        @test !edit_calibration_option!(cal, :model, :bundle)
        @test_throws ArgumentError set_calibration_option!(cal, :nope, 1)
        add_plates!(cal)
        @test edit_calibration_option!(cal, :spacing, 15) && isempty(cal.error[])
        @test cal.origin_offset[] == (30.0, 7.5) && cal.grid_spacing[] == 0.3

        fit_calibration!(cal)                                # fit, then build the grid
        @test !cal.fitting[] && !cal.building[]
        @test all(k -> cal.reviews[k][].camera[] isa SoloffCamera, 1:2)
        @test occursin("soloff fit", cal.fit_status[])
        dws = cal.dewarpers[]
        @test dws !== nothing && dws[1].grid == dws[2].grid
        @test abs(step(dws[1].grid.x)) ≈ 0.3 rtol = 0.01
        @test grid_size(wf) == size(dws[1].grid)
        @test wf.passes.image_size[] == grid_size(wf)       # presets follow the grid
        @test workflow_problem(wf) === nothing
        @test step_status(wf, :calibration)[1] === :ok
        @test occursin("dewarp grid", calibration_summary(cal))
        @test !fit_stale(cal)

        # a grid option rebuilds; the model refits the reviews and rebuilds
        set_calibration_option!(cal, :grid_spacing, "0.6")
        @test cal.dewarpers[] !== dws && size(cal.dewarpers[][1].grid) != size(dws[1].grid)
        @test wf.passes.image_size[] == grid_size(wf)
        set_calibration_option!(cal, :model, "pinhole")
        @test all(k -> cal.reviews[k][].camera[] isa PinholeCamera, 1:2)
        # detection edits mark the fit stale until the next fit
        set_calibration_option!(cal, :invert, "false")
        @test !fit_stale(cal)
        set_calibration_option!(cal, :spacing, 15.5)
        @test fit_stale(cal) && step_status(wf, :calibration)[1] === :attention
        set_calibration_option!(cal, :spacing, 15)
        @test !fit_stale(cal)
        set_calibration_option!(cal, :model, :soloff)
        set_calibration_option!(cal, :grid_spacing, 0.3)
        dws = cal.dewarpers[]
        @test abs(step(dws[1].grid.x)) ≈ 0.3 rtol = 0.01
    end

    @testset "async calibration: stale results are dropped" begin
        queue = Any[]
        lk = ReentrantLock()
        aw = StereoWorkflow(deliver = f -> lock(() -> push!(queue, f), lk))
        aw.spawn[] = true
        drain!() = foreach(f -> f(), lock(() -> splice!(queue, 1:length(queue)), lk))
        waitfor(n) = timedwait(() -> lock(() -> length(queue) >= n, lk), 120) === :ok
        acal = add_plates!(aw.calibration)
        fit_calibration!(acal)
        set_calibration_option!(acal, :model, :pinhole)
        fit_calibration!(acal)                               # supersedes the first fit
        @test acal.fitting[] && all(k -> acal.reviews[k][] === nothing, 1:2)
        @test waitfor(2)
        drain!()                                             # the first fit is dropped
        @test !acal.fitting[] && all(k -> acal.reviews[k][].camera[] isa PinholeCamera, 1:2)
        @test acal.building[] && acal.dewarpers[] === nothing  # the grid builds next
        @test waitfor(1)
        drain!()
        @test !acal.building[] && acal.dewarpers[] !== nothing
        # dewarpers set during a build win over the build
        build_dewarpers!(acal)
        set_dewarpers!(aw, dws...)
        @test !acal.building[]
        @test waitfor(1)
        drain!()
        @test acal.dewarpers[][1] === dws[1] && grid_size(aw) == size(dws[1].grid)
        # frames load on workers; then the dewarped preview and raw view follow
        add_files!(aw, files1[1:2]; camera = 1)
        add_files!(aw, files2[1:2]; camera = 2)
        @test pair_loading(aw.frames1) && pair_loading(aw.frames2)
        @test frames_problem(aw) == "camera 1: loading pair 1…"
        @test step_status(aw, :images)[1] === :busy
        pp = aw.prepare.preview
        t0 = time()
        while (aw.dewarped[] === nothing || pp.processed[] === nothing ||
               size(pp.processed[]) != size(dws[1].grid)) && time() - t0 < 120
            waitfor(1)
            drain!()
        end
        @test frames_problem(aw) === nothing && aw.frames1.loaded[] == 1
        @test pp.processed[] == dewarp(dws[1], Float32.(files1[1]))
        @test aw.dewarped[][2] == dewarp(dws[1], Float32.(files1[2]))
        # a closed window forgets pending jobs
        fit_calibration!(acal)
        aw.spawn[] = false
        C._abandon_jobs!(aw)
        @test !acal.fitting[]
        @test waitfor(1)
        drain!()
        @test acal.dewarpers[][1] === dws[1]
    end

    @testset "set_dewarpers! and the dewarpers keyword" begin
        sw = StereoWorkflow(; files1, files2, dewarpers = dws)
        @test sw.calibration.dewarpers[] == dws && workflow_problem(sw) === nothing
        @test sw.passes.image_size[] == size(dws[1].grid)
        @test sw.prepare.mask[].size == size(dws[1].grid)
        other = ImageDewarper(cams[2], DewarpGrid(x = -8.0:0.5:8.0, y = 8.0:-0.5:-8.0), (512, 512))
        @test_throws ArgumentError set_dewarpers!(sw, dws[1], other)
        @test sw.calibration.dewarpers[] == dws
        @test out_of_view(sw) == (dws[1].mask .| dws[2].mask)
        @test out_of_view(StereoWorkflow()) === nothing
    end

    @testset "camera frames stay in sync" begin
        sw = StereoWorkflow(; files1, files2, dewarpers = dws)
        f1, f2 = sw.frames1, sw.frames2
        @test frames_problem(sw) === nothing
        @test frames_summary(sw) == "6 frames per camera · 3 pairs · 512×512 px"
        @test step_status(sw, :images) == (:ok, frames_summary(sw))
        select_pair!(f2, 3)
        @test f1.pair[] == 3 && current_pair(f1)[1] === files1[5]
        select_pair!(sw, 2)
        @test f2.pair[] == 2
        set_pair_mode!(f1, :chained)
        @test f2.pair_mode[] === :chained && npairs(sw) == 5
        set_pair_mode!(sw, :paired)
        @test f1.pair_mode[] === f2.pair_mode[] === :paired
        show_frame!(f2, :b)
        @test f1.shown[] === :b
        show_frame!(sw, :a)
        # the viewer's camera switches the preview frames
        pp = sw.prepare.preview
        @test pp.image[] == Float32.(files1[3])
        set_camera!(sw, 2)
        @test shown_frames(sw) === f2 && pp.image[] == Float32.(files2[3])
        @test size(pp.processed[]) == size(dws[1].grid)      # dewarped
        @test_throws ArgumentError set_camera!(sw, 3)
        # per-camera problems
        clear_files!(sw; camera = 2)
        @test frames_problem(sw) == "camera 2: add frames to analyze"
        @test step_status(sw, :images)[1] === :todo
        add_files!(sw, files2[1:4]; camera = 2)
        @test frames_problem(sw) == "camera 1 has 3 pairs but camera 2 has 2"
        clear_files!(sw)
        add_files!(sw, Any[zeros(256, 256), zeros(256, 256)]; camera = 1)
        add_files!(sw, Any[zeros(256, 256), zeros(256, 256)]; camera = 2)
        @test frames_problem(sw) == "camera 1 frames are 256×256 px but its calibration is for 512×512 px"
        test_pair!(sw; spawn = false)
        @test sw.test.status[] == frames_problem(sw)
    end

    @testset "recipe round trip; ROI recipes are rejected" begin
        sw = StereoWorkflow(; files1, files2, dewarpers = dws)
        @test settings_modified(sw)
        @test workflow_recipe(sw).roi === nothing
        gsz = size(dws[1].grid)
        mask = falses(gsz); mask[1:20, 1:30] .= true
        loaded = PIVRecipe(multipass_parameters([48, 24]; final = (max_iterations = 2,));
                           preprocessing = [PreprocessStep(:highpass_filter; sigma = 5)],
                           mask, scale = PhysicalScale(1.0, 1e-3, "mm", "s"),
                           image_type = Float32, predictor_smoothing = false,
                           mask_threshold = 0.25)
        load_settings!(sw, loaded)
        @test workflow_recipe(sw) == loaded && !settings_modified(sw)
        @test sw.prepare.mask[].raster[] == mask
        mktempdir() do dir
            path = save_settings(sw, joinpath(dir, "stereo.jld2"))
            sw2 = StereoWorkflow(; dewarpers = dws)
            load_settings!(sw2, path)
            @test workflow_recipe(sw2) == loaded && sw2.settings_path[] == path
        end
        with_roi = PIVRecipe(multipass_parameters([32]); roi = ROI(1:100, 1:100))
        @test_throws ArgumentError load_settings!(sw, with_roi)
        @test workflow_recipe(sw) == loaded                  # unchanged
    end

    @testset "test pair through the stereo apply_recipe" begin
        sw = StereoWorkflow(; files1, files2)
        test_pair!(sw; spawn = false)
        @test sw.test.status[] == "calibrate the cameras first (no dewarpers)"
        set_dewarpers!(sw, dws...)
        fill_preset!(sw.passes, :low)
        test_pair!(sw; spawn = false)
        r = sw.test.result[]
        @test r isa StereoPIVResult
        direct = apply_recipe(workflow_recipe(sw), [current_pair(sw.frames1)],
                              [current_pair(sw.frames2)], dws...; progress = false)[1]
        @test isequal(r.u, direct.u) && isequal(r.v, direct.v) && isequal(r.w, direct.w)
        good = .!(r.mask .| r.outliers)
        @test median(r.u[good]) ≈ disp[1] atol = 0.1          # world units (mm)
        @test median(r.v[good]) ≈ disp[2] atol = 0.1
        @test !test_stale(sw) && step_status(sw, :test)[1] === :ok
        s = test_summary(sw.test)
        @test s.valid_fraction > 0.9 && s.sigma_unit == "world units"
        @test s.max_displacement > 2                         # dewarped px
        @test any(l -> startswith(l, "Valid vectors"), summary_lines(s))
        select_pair!(sw, 2)
        @test test_stale(sw)
        select_pair!(sw, 1)
        @test !test_stale(sw)
        # new dewarpers make the test stale
        set_dewarpers!(sw, ImageDewarper(dws[1].cam, dws[1].grid, (512, 512)), dws[2])
        @test test_stale(sw)
    end

    @testset "batch run with cancellation" begin
        sw = StereoWorkflow(; files1, files2, dewarpers = dws)
        fill_preset!(sw.passes, :low)
        sw.scale[] = PhysicalScale(1.0, 1e-3, "mm", "s")
        mktempdir() do dir
            out = joinpath(dir, "stereo_run.jld2")
            sw.run.output_path[] = out
            start_run!(sw; spawn = false)
            @test !sw.run.running[] && length(sw.run.completed[]) == 3
            @test startswith(sw.run.status[], "done: 3 pairs")
            @test sw.results_path[] == out && nframes(sw.explorer[]) == 3
            @test current_result(sw.explorer[]) isa StereoPIVResult
            @test load_recipe(out) == workflow_recipe(sw)
            @test sw.run.completed[][1].scale.time_unit == "s"

            sw.run.output_path[] = joinpath(dir, "cancel.jld2")
            on(c -> length(c) == 1 && cancel_run!(sw), sw.run.completed)
            start_run!(sw; spawn = false)
            @test occursin("cancelled after", sw.run.status[]) && length(sw.run.completed[]) < 3
            @test nframes(sw.explorer[]) == length(sw.run.completed[])
            @test length(load_results(joinpath(dir, "cancel.jld2"))) == length(sw.run.completed[])
        end
    end

    @testset "Prepare on the dewarped grid" begin
        sw = StereoWorkflow(; files1, files2)
        ps, pp = sw.prepare, sw.prepare.preview
        @test ps.mask[] === nothing && ps.roi[] === nothing && ps.scale[] === nothing
        @test pp.image[] == Float32.(files1[1])               # raw frames, camera space
        @test size(pp.processed[]) == (512, 512) && sw.dewarped[] === nothing
        set_dewarpers!(sw, dws...)
        gsz = size(dws[1].grid)
        @test ps.mask[].size == gsz && ps.roi[] === nothing && ps.scale[] === nothing
        # preprocessing applies to the raw frames, then the pair is dewarped
        add_step!(pp, :highpass_filter)
        @test sw.preprocessing[] == pp.steps[]
        @test pp.processed[] == dewarp(dws[1], recipe_preprocess(sw.preprocessing[])(Float32.(files1[1])))
        @test sw.dewarped[][1] == dewarp(dws[1], Float32.(files1[1])) && size(sw.dewarped[][2]) == gsz
        set_step!(sw, :prepare)
        @test canvas_click!(sw, 120.0, 120.0)                 # probe on the dewarped pair
        res = pp.probe_result[]
        @test res.du ≈ disp[1] / abs(step(dws[1].grid.x)) atol = 0.3
        @test res.dv ≈ -disp[2] / abs(step(dws[1].grid.y)) atol = 0.3   # descending y
        set_camera!(sw, 2)
        @test pp.processed[] == dewarp(dws[2], recipe_preprocess(sw.preprocessing[])(Float32.(files2[1])))
        @test sw.dewarped[][1] == dewarp(dws[2], Float32.(files2[1]))
        @test pp.probe_result[].du ≈ res.du atol = 0.3
        # mask: drawn in grid coordinates
        @test prepare_pages(sw) == STEREO_PREPARE_PAGES
        @test_throws ArgumentError set_prepare_page!(sw, :roi)
        set_prepare_page!(sw, :mask)
        for (x, y) in ((20.0, 10.0), (60.0, 10.0), (60.0, 50.0))
            @test canvas_click!(sw, x, y)
        end
        @test canvas_alt_click!(sw)
        @test sw.mask[] == polygon_mask(gsz, [(20, 10), (60, 10), (60, 50)])
        @test step_status(sw, :prepare) == (:ok, "1 preprocessing step · mask")
        mktempdir() do dir
            path = save_mask_file(sw, joinpath(dir, "mask.png"))
            sw.mask[] = nothing
            load_mask_file!(sw, path)
            @test sw.mask[] == polygon_mask(gsz, [(20, 10), (60, 10), (60, 50)])
            C.FileIO.save(joinpath(dir, "small.png"), C.Gray.(falses(8, 8)))
            @test_throws DimensionMismatch load_mask_file!(sw, joinpath(dir, "small.png"))
        end
        # scale: dt and units only; lengths are world units
        set_prepare_page!(sw, :scale)
        @test !canvas_click!(sw, 10.0, 10.0)
        @test edit_scale!(sw, :dt, "0.001") && sw.scale[] == PhysicalScale(1.0, 0.001, "mm", "frame")
        @test edit_scale!(sw, :time_unit, "s") && sw.scale[].time_unit == "s"
        @test !edit_scale!(sw, :pixel_size, "2") && occursin("world units", ps.scale_error[])
        @test !edit_scale!(sw, :separation, "2")
        clear_scale!(sw)
        @test sw.scale[] === nothing
        estimate_background!(sw)
        @test occursin("not available for stereo", ps.status[])
        # another grid: the mask editor follows, the old mask is flagged
        coarse = (ImageDewarper(dws[1].cam, DewarpGrid(x = -20.0:0.5:20.0, y = 20.0:-0.5:-20.0), (512, 512)),
                  ImageDewarper(dws[2].cam, DewarpGrid(x = -20.0:0.5:20.0, y = 20.0:-0.5:-20.0), (512, 512)))
        set_dewarpers!(sw, coarse...)
        @test ps.mask[].size == (81, 81) && !has_mask(ps.mask[])
        @test step_status(sw, :prepare)[1] === :attention
        @test size(pp.processed[]) == (81, 81)
    end

    @testset "self-calibration finds the offset sheet" begin
        sw = StereoWorkflow(; files1, files2, dewarpers = dws)
        scal = sw.calibration
        @test !apply_selfcal!(sw) && scal.selfcal_status[] == "run self-calibration first"
        fill_preset!(sw.passes, :low)
        test_pair!(sw; spawn = false)
        start_selfcal!(sw; pairs = 2)
        @test !scal.selfcal_running[]
        s = scal.selfcal[]
        @test s !== nothing && s.source[1] === dws[1]
        @test s.report.passes[1].plane.a ≈ sheet atol = 0.05
        @test s.report.converged && occursin("pass 1", scal.selfcal_status[])
        @test !test_stale(sw)
        @test apply_selfcal!(sw) && scal.selfcal_applied[]
        @test scal.dewarpers[][1] === s.dewarpers[1]
        @test test_stale(sw)
        @test occursin("self-calibrated", step_status(sw, :calibration)[2])
        @test !apply_selfcal!(sw) && occursin("already applied", scal.selfcal_status[])
        # dewarpers replaced meanwhile: the old result no longer applies
        set_dewarpers!(sw, ImageDewarper(dws[1].cam, dws[1].grid, (512, 512)), dws[2])
        @test !scal.selfcal_applied[]
        @test !apply_selfcal!(sw) && occursin("changed", scal.selfcal_status[])
        @test step_status(sw, :calibration)[1] === :ok
    end

    @testset "shared steps and labels" begin
        @test workflow_steps(wf) == STEREO_WORKFLOW_STEPS
        @test all(st -> haskey(C.STEP_LABELS, st), STEREO_WORKFLOW_STEPS)
        @test_throws ArgumentError set_step!(wf, :roi)
        set_step!(wf, :calibration)
        @test wf.step[] === :calibration
        @test !canvas_click!(wf, 10.0, 10.0)
        for st in STEREO_WORKFLOW_STEPS
            @test step_status(wf, st)[1] in (:todo, :ok, :attention, :busy)
        end
        @test workflow_steps(PlanarWorkflow()) == WORKFLOW_STEPS
        @test sprint(show, wf) isa String
    end
end
