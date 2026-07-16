using Test
using HammerheadGUI
using HammerheadGUI.GLMakie
using HammerheadGUI.Hammerhead
using HammerheadGUI.Hammerhead.SyntheticData: generate_synthetic_piv_pair, linear_flow
using Statistics: median

# Shared fixtures: a uniform-flow synthetic pair (realistic particle images,
# so the Wieneke uncertainty estimates come out finite) analyzed with and
# without uncertainty, plus a StereoPIVResult assembled from the 2C fields.
# NOTE: the direct StereoPIVResult constructor call below breaks when fields
# are added to the type — same caveat as the core test suite.
imgA, imgB, _, _ = generate_synthetic_piv_pair(linear_flow(3.0, 2.0, 0.0, 0, 0, 0, 0),
                                               (128, 128), 1.0; z_range = (-1.0, 1.0))
const r_unc = run_piv(imgA, imgB, multipass_parameters([32, 32]; uncertainty = true))
const r_plain = run_piv(imgB, imgA, PIVParameters(window_size = 32))
const r_stereo = StereoPIVResult(collect(r_unc.x), collect(r_unc.y), 0.0,
                                 r_unc.u, r_unc.v, 0.5 .* r_unc.u,
                                 r_unc.uncertainty_u, r_unc.uncertainty_v,
                                 2.0 .* r_unc.uncertainty_u,
                                 r_unc.outliers, r_unc.mask,
                                 r_unc, r_plain, r_unc.parameters)

# Scattered-result fixtures constructed directly for determinism (one flagged
# PTV particle; one trajectory with a bridged frame gap). Same caveat as the
# StereoPIVResult above: positional constructors break when fields are added.
const px_a = Particles([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0],
                       [1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0])
const r_ptv = PTVResult([10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0],
                        [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0],
                        [0.2, 0.1, 0.3, 5.0],
                        BitVector([false, false, false, true]),
                        [1, 2, 3, 4], [1, 2, 3, 4], px_a, px_a, PTVParameters())
const r_track = TrackingResult(
    [Trajectory{Float64}(1, [10.0, 12.0, 14.0, 16.0], [10.0, 11.0, 12.0, 13.0], [1, 2, 3, 4]),
     Trajectory{Float64}(1, [50.0, 52.0, 54.0], [50.0, 51.0, 52.0], [1, 2, 4])],
    4, PTVParameters())

@testset "HammerheadGUI.jl" begin
    @testset "Offscreen GL rendering" begin
        GLMakie.activate!()
        fig = Figure(size = (400, 300))
        ax = Axis(fig[1, 1])
        heatmap!(ax, rand(32, 32))
        img = colorbuffer(fig; px_per_unit = 1)
        @test size(img) == (300, 400)
    end

    @testset "Core pipeline + Makie extension smoke test" begin
        @test median(filter(!isnan, r_unc.u)) ≈ 3.0 atol = 0.3
        @test median(filter(!isnan, r_unc.v)) ≈ 2.0 atol = 0.3

        # GLMakie loads Makie, so Hammerhead's Makie extension must be
        # active and renderable through the GL backend.
        fig = plot_vector_field(r_unc)
        @test fig isa Figure
        img = colorbuffer(fig; px_per_unit = 1)
        @test !isempty(img)
    end

    @testset "Controllers are framework-free" begin
        for makie_name in (:Figure, :Axis, :heatmap!, :GLMakie, :Makie)
            @test !isdefined(HammerheadGUI.Controllers, makie_name)
        end
    end

    @testset "ResultExplorer controller (no GL)" begin
        ex = ResultExplorer([r_unc, r_plain])
        @test nframes(ex) == 2
        @test ex.frame[] == 1
        @test ex.field[] == :magnitude
        @test current_result(ex) === r_unc

        # field inventory: uncertainty entries only where estimates exist
        @test available_fields(r_unc) ==
              [:magnitude, :u, :v, :peak_ratio, :correlation_moment,
               :uncertainty_u, :uncertainty_v]
        @test available_fields(r_plain) ==
              [:magnitude, :u, :v, :peak_ratio, :correlation_moment]
        @test :w in available_fields(r_stereo)

        @test field_values(r_unc, :magnitude) ≈ hypot.(r_unc.u, r_unc.v) nans = true
        @test field_values(r_unc, :peak_ratio) === r_unc.peak_ratio
        @test field_values(r_stereo, :magnitude) ≈
              hypot.(r_stereo.u, r_stereo.v, r_stereo.w) nans = true
        @test_throws ArgumentError field_values(r_plain, :uncertainty_u)
        @test_throws ArgumentError field_values(r_unc, :parameters)

        # frame change resets an unavailable field and clamps out-of-range
        set_field!(ex, :uncertainty_u)
        set_frame!(ex, 2)
        @test ex.field[] == :magnitude
        @test_throws ArgumentError set_field!(ex, :uncertainty_u)
        set_frame!(ex, 99)
        @test ex.frame[] == 2

        # selection: nearest grid node, info text, clearing
        select_nearest!(ex, r_plain.x[2] + 1.0, r_plain.y[3] - 1.0)
        @test ex.selection[] == CartesianIndex(3, 2)
        info = describe_selection(ex)
        @test occursin("u = ", info) && occursin("status:", info)
        clear_selection!(ex)
        @test describe_selection(ex) == ""

        # stereo summaries include the third component and camera diagnostics
        exs = ResultExplorer(r_stereo)
        select_nearest!(exs, r_stereo.x[1], r_stereo.y[1])
        sinfo = describe_selection(exs)
        @test occursin("w = ", sinfo) && occursin("cam peak ratios", sinfo)

        # arrow-overlay data: flat, NaN-free, matching lengths
        d = HammerheadGUI.Controllers.vector_data(r_unc)
        @test length(d.x) == length(d.u) == length(d.outlier)
        @test all(isfinite, d.u) && all(isfinite, d.v)
        @test HammerheadGUI.Controllers.auto_lengthscale(r_unc) > 0

        # file round-trip through load_results
        path = joinpath(mktempdir(), "results.jld2")
        save_results(path, [r_unc, r_plain])
        ex2 = ResultExplorer(path)
        @test nframes(ex2) == 2
        @test ex2.path == path

        @test_throws ArgumentError ResultExplorer(PIVResult[])
    end

    @testset "ResultExplorer physical units (no GL)" begin
        C = HammerheadGUI.Controllers

        # unscaled fallbacks: px for planar, world units for stereo
        @test C.field_label(r_plain, :u) == "u (px)"
        @test C.field_label(r_plain, :peak_ratio) == "peak ratio"
        @test C.field_label(r_stereo, :w) == "w (world units)"

        # a real scale attached -> physical units in every label + summary
        scale = PhysicalScale(20.0, 0.5, "mm", "s")
        exs = ResultExplorer(with_scale(r_unc, scale))
        rs = current_result(exs)
        @test rs.scale !== nothing
        @test C.field_label(rs, :u) == "u (mm/s)"
        @test C.field_label(rs, :magnitude) == "|displacement| (mm/s)"
        @test C.field_label(rs, :uncertainty_u) == "σu (mm/s)"
        @test C.field_label(rs, :peak_ratio) == "peak ratio"
        select_nearest!(exs, rs.x[3], rs.y[3])
        info = describe_selection(exs)
        @test occursin("mm/s", info) && occursin("mm", info)

        # PTV match_residual is a length; velocity fields carry the velocity unit
        exp = ResultExplorer(with_scale(r_ptv, scale))
        rp = current_result(exp)
        @test C.field_label(rp, :match_residual) == "match residual (mm)"
        @test C.field_label(rp, :u) == "u (mm/s)"

        # tracking keeps dt through physical; speed carries the velocity unit
        ext = ResultExplorer(with_scale(r_track, scale))
        rt = current_result(ext)
        @test C.field_label(rt, :speed) == "speed (mm/s)"
        select_nearest!(ext, rt.trajectories[1].x[1], rt.trajectories[1].y[1])
        @test occursin("mm/s", describe_selection(ext))
    end

    @testset "ResultExplorer PTV + tracking (no GL)" begin
        C = HammerheadGUI.Controllers

        exp = ResultExplorer(r_ptv)
        @test nframes(exp) == 1
        @test exp.field[] == :magnitude
        @test available_fields(r_ptv) == [:magnitude, :u, :v, :match_residual]
        @test field_values(r_ptv, :magnitude) ≈ hypot.(r_ptv.u, r_ptv.v)
        @test field_values(r_ptv, :u) === r_ptv.u
        @test_throws ArgumentError field_values(r_ptv, :peak_ratio)

        # 2-D nearest particle (not per-axis), inspection wording is "flagged"
        select_nearest!(exp, 21.0, 19.0)
        @test exp.selection[] == 2
        select_nearest!(exp, 41.0, 39.0)
        @test exp.selection[] == 4
        info = describe_selection(exp)
        @test occursin("particle 4", info) && occursin("flagged", info)
        @test !occursin("replaced", info)

        # vector_data flat + finite; scattered auto_lengthscale positive
        d = C.vector_data(r_ptv)
        @test length(d.x) == length(d.u) == length(d.outlier) == 4
        @test all(isfinite, d.u) && all(isfinite, d.v)
        @test C.auto_lengthscale(r_ptv) > 0

        # tracking: fields, gap counting, NaN-separated polylines, selection
        ext = ResultExplorer(r_track)
        @test available_fields(r_track) == [:speed]
        @test C.trajectory_gap_count(r_track.trajectories[1]) == 0
        @test C.trajectory_gap_count(r_track.trajectories[2]) == 1
        xs, ys = C.trajectory_points(r_track.trajectories[2])
        @test count(isnan, xs) == 1 && length(xs) == 4   # 3 points + 1 gap break
        speeds = field_values(r_track, :speed)
        @test length(speeds) == 2 && all(>(0), speeds)
        select_nearest!(ext, 51.0, 50.0)
        @test ext.selection[] == 2
        s = describe_selection(ext)
        @test occursin("trajectory 2", s) && occursin("gaps: 1", s)

        # selection type follows the result: a stale CartesianIndex is dropped
        # for scattered, and a stale Int for grid, on a mixed sequence
        exm = ResultExplorer([r_unc, r_ptv])
        select_nearest!(exm, r_unc.x[1], r_unc.y[1])
        @test exm.selection[] isa CartesianIndex
        set_frame!(exm, 2)
        @test exm.selection[] === nothing
        select_nearest!(exm, 10.0, 10.0)
        @test exm.selection[] isa Int
        set_frame!(exm, 1)
        @test exm.selection[] === nothing
    end

    @testset "mixed-type JLD2 round-trip" begin
        path = joinpath(mktempdir(), "mixed.jld2")
        entries = Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}[
            r_unc, r_stereo, r_ptv, r_track]
        save_results(path, entries)
        loaded = load_results(path)
        @test length(loaded) == 4
        ex = ResultExplorer(path)
        @test nframes(ex) == 4
        @test current_result(ex) isa PIVResult
        set_frame!(ex, 2); @test current_result(ex) isa StereoPIVResult
        set_frame!(ex, 3); @test current_result(ex) isa PTVResult
        set_frame!(ex, 4); @test current_result(ex) isa TrackingResult
    end

    @testset "result_explorer view (offscreen)" begin
        ex = ResultExplorer([r_unc, r_plain])
        fig = result_explorer(ex; size = (900, 650))
        # copy: colorbuffer may return the screen's reused framebuffer
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (650, 900)

        # drive the view through the controller and re-render
        set_field!(ex, :peak_ratio)
        set_frame!(ex, 2)
        select_nearest!(ex, r_plain.x[2], r_plain.y[2])
        ex.show_vectors[] = false
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test size(img2) == size(img1)
        @test img2 != img1

        # stereo view renders too
        figs = result_explorer(r_stereo)
        @test !isempty(colorbuffer(figs; px_per_unit = 1))

        # PTV explorer: renders and re-renders on a field/toggle change
        exp = ResultExplorer(r_ptv)
        figp = result_explorer(exp; size = (900, 650))
        imgp1 = copy(colorbuffer(figp; px_per_unit = 1))
        @test size(imgp1) == (650, 900)
        set_field!(exp, :match_residual)
        exp.show_vectors[] = false
        imgp2 = colorbuffer(figp; px_per_unit = 1)
        @test imgp2 != imgp1

        # tracking explorer: polylines render and re-render on selection
        ext = ResultExplorer(r_track)
        figt = result_explorer(ext; size = (900, 650))
        imgt1 = copy(colorbuffer(figt; px_per_unit = 1))
        @test size(imgt1) == (650, 900)
        select_nearest!(ext, 51.0, 50.0)
        @test colorbuffer(figt; px_per_unit = 1) != imgt1
    end

    @testset "MaskEditor controller (no GL)" begin
        me = MaskEditor(imgA)

        # drawing gestures: empty-background click starts a polygon,
        # subsequent clicks add vertices, alt-click commits
        HammerheadGUI.Controllers.click!(me, 20.0, 10.0)
        HammerheadGUI.Controllers.click!(me, 60.0, 10.0)
        @test length(me.active[]) == 2
        HammerheadGUI.Controllers.alt_click!(me)          # < 3 vertices: cancel
        @test isempty(me.active[]) && isempty(me.polygons[])
        for (x, y) in ((20.0, 10.0), (60.0, 10.0), (60.0, 50.0), (20.0, 50.0))
            HammerheadGUI.Controllers.click!(me, x, y)
        end
        undo_vertex!(me)                                  # drop and re-add a corner
        @test length(me.active[]) == 3
        add_vertex!(me, 20.0, 50.0)
        @test close_active!(me)
        @test length(me.polygons[]) == 1 && isempty(me.active[])

        # the committed rectangle matches polygon_mask directly
        m = polygon_mask(me)
        @test m == polygon_mask(size(imgA), [(20, 10), (60, 10), (60, 50), (20, 50)])
        @test m[30, 40] && !m[80, 40]

        # selection: click inside selects, alt-click deselects, delete removes
        HammerheadGUI.Controllers.click!(me, 40.0, 30.0)
        @test me.selected[] == 1
        @test isempty(me.active[])                        # selecting ≠ drawing
        HammerheadGUI.Controllers.alt_click!(me)
        @test me.selected[] === nothing
        HammerheadGUI.Controllers.click!(me, 40.0, 30.0)
        delete_selected!(me)
        @test isempty(me.polygons[]) && me.selected[] === nothing
        @test !any(polygon_mask(me))

        # seeded polygons, union masks, clear
        seeded = MaskEditor(imgA; polygons = [[(5, 5), (15, 5), (15, 15), (5, 15)],
                                              [(30, 30), (40, 30), (40, 40), (30, 40)]])
        @test HammerheadGUI.Controllers.polygon_at(seeded, 10.0, 10.0) == 1
        @test HammerheadGUI.Controllers.polygon_at(seeded, 100.0, 100.0) === nothing
        ms = polygon_mask(seeded)
        @test ms[10, 10] && ms[35, 35] && !ms[25, 25]

        # Holes subtract from earlier exclusion polygons, and morphology can
        # then add or remove a pixel safety margin.
        holed = MaskEditor(imgA;
            polygons = [[(5, 5), (30, 5), (30, 30), (5, 30)],
                        [(12, 12), (22, 12), (22, 22), (12, 22)]],
            holes = [false, true])
        mh = polygon_mask(holed)
        @test mh[8, 8] && !mh[16, 16]
        n0 = count(mh)
        grow_mask!(holed, 1)
        @test count(polygon_mask(holed)) > n0
        shrink_mask!(holed, 1)
        @test count(polygon_mask(holed)) < count(Hammerhead.grow_mask(mh, 1))

        drawn_hole = MaskEditor(imgA; polygons = [[(5, 5), (30, 5), (30, 30), (5, 30)]])
        begin_hole!(drawn_hole)
        for p in ((12, 12), (22, 12), (22, 22), (12, 22))
            add_vertex!(drawn_hole, p...)
        end
        close_active!(drawn_hole)
        @test !polygon_mask(drawn_hole)[16, 16]
        clear_polygons!(seeded)
        @test isempty(seeded.polygons[])
        @test_throws ArgumentError MaskEditor(imgA; polygons = [[(0, 0), (1, 1)]])

        # save_mask round-trips through load_mask
        me2 = MaskEditor(imgA; polygons = [[(20, 10), (60, 10), (60, 50), (20, 50)]])
        path = joinpath(mktempdir(), "mask.png")
        save_mask(me2, path)
        @test load_mask(path) == polygon_mask(me2)
    end

    @testset "mask_editor view (offscreen)" begin
        me = MaskEditor(imgA; polygons = [[(20, 10), (60, 10), (60, 50), (20, 50)]])
        fig = mask_editor(me; size = (900, 650))
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (650, 900)

        # drive editing through the controller and re-render
        HammerheadGUI.Controllers.click!(me, 40.0, 30.0)  # select
        me.show_mask[] = true
        for (x, y) in ((80.0, 80.0), (110.0, 80.0), (110.0, 110.0))
            add_vertex!(me, x, y)
        end
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test size(img2) == size(img1)
        @test img2 != img1
    end

    @testset "BatchRunner controller (no GL)" begin
        C = HammerheadGUI.Controllers

        @test C.parse_schedule("64, 32 32") == [64, 32, 32]
        @test_throws ArgumentError C.parse_schedule("64, nope")
        @test_throws ArgumentError C.parse_schedule(" ")
        @test_throws ArgumentError C.parse_schedule("0, 32")

        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                         window_schedule = [32, 32])
        @test C.validate(bc) === nothing
        @test length(C.frame_pairs(bc)) == 2
        bc.pair_mode[] = :chained
        @test length(C.frame_pairs(bc)) == 3
        bc.pair_mode[] = :paired

        ps = C.build_parameters(bc)
        @test length(ps) == 2
        @test ps[end].window_size == (32, 32) && ps[end].overlap == (16, 16)
        @test ps[end].padding && ps[end].apodization == :gauss
        set_schedule!(bc, "48 24")
        @test bc.window_schedule[] == [48, 24]

        @test C.validate(BatchRunner()) == "add frames first"
        odd = BatchRunner(files = Any[imgA, imgB, imgA])
        @test occursin("even number", C.validate(odd))

        # synchronous run: progress trace, results, incremental output
        out = joinpath(mktempdir(), "batch.jld2")
        bc2 = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                          window_schedule = [32], output_path = out,
                          padding = false, apodization = :none)
        seen = Tuple{Int,Int}[]
        on(p -> push!(seen, p), bc2.progress)
        start!(bc2; async = false)
        @test !bc2.running[]
        @test bc2.results[] !== nothing && length(bc2.results[]) == 2
        @test seen[end] == (2, 2)
        @test occursin("done", bc2.status[])
        @test length(load_results(out)) == 2

        # cancellation after the first pair keeps it in the output
        out2 = joinpath(mktempdir(), "cancelled.jld2")
        bc3 = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                          window_schedule = [32], output_path = out2,
                          padding = false, apodization = :none)
        on(p -> p[1] == 1 && cancel!(bc3), bc3.progress)
        start!(bc3; async = false)
        @test occursin("cancelled", bc3.status[])
        @test bc3.results[] === nothing
        @test length(load_results(out2)) == 1

        # the mask forwards into run_piv
        m = falses(size(imgA)); m[1:48, 1:48] .= true
        bc4 = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                          mask = m, padding = false, apodization = :none)
        start!(bc4; async = false)
        @test any(bc4.results[][1].mask)

        # effort presets bypass the manual schedule
        @test_throws ArgumentError set_effort!(BatchRunner(), :turbo)
        bce = BatchRunner(files = Any[imgA, imgB], effort = :low)
        @test C.validate(bce) === nothing
        start!(bce; async = false)
        @test bce.results[] !== nothing && length(bce.results[]) == 1
        @test occursin("done", bce.status[])

        # physical scale plumbs into the outputs; default is no scale
        @test C.build_scale(BatchRunner(files = Any[imgA, imgB])) === nothing
        @test_throws ArgumentError C.set_pixel_size!(BatchRunner(), "-1")
        @test_throws ArgumentError C.set_pixel_size!(BatchRunner(), "abc")
        bcs = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                          padding = false, apodization = :none,
                          pixel_size = 20.0, dt = 0.5,
                          length_unit = "mm", time_unit = "s")
        @test C.build_scale(bcs) isa PhysicalScale
        start!(bcs; async = false)
        sc = bcs.results[][1].scale
        @test sc !== nothing && sc.length_unit == "mm" && sc.time_unit == "s"
        @test sc.pixel_size == 20.0 && sc.dt == 0.5
    end

    @testset "batch_runner view (offscreen)" begin
        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                         window_schedule = [32],
                         padding = false, apodization = :none)
        fig = batch_runner(bc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (640, 960)

        set_schedule!(bc, "64 32")   # form summary label updates
        bc.uncertainty[] = true      # toggle syncs back into the widget
        set_effort!(bc, :medium)     # effort menu + inactive-schedule summary
        set_scale!(bc; pixel_size = 5.0, length_unit = "mm")  # scale summary line
        start!(bc; async = false)    # status + progress labels update
        @test occursin("done", bc.status[])
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test size(img2) == size(img1)
        @test img2 != img1
    end

    @testset "CalibrationReview controller (no GL)" begin
        C = HammerheadGUI.Controllers

        # synthetic plate through a known pinhole rig (recipe from the core
        # test suite's make_test_camera fixture)
        θ = deg2rad(20.0)
        R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
        camC = R' * [0.0, 0.0, -500.0]
        K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
        cam = PinholeCamera(K, R, -R * camC)
        zs = [-3.0, 0.0, 3.0]
        plates = [render_calibration_target(cam, (512, 512); spacing = 15.0,
                                            z = z,
                                            marker_square = (-30.0, -7.5),
                                            marker_triangle = (-15.0, -7.5))
                  for z in zs]

        cr = CalibrationReview(plates, zs;
                               spacing = 15.0, origin_offset = (30.0, 7.5))
        @test nplanes(cr) == 3
        @test cr.camera[] isa SoloffCamera
        @test cr.fit_message[] == ""
        pe = C.plane_errors(cr)
        @test length(pe.errors) == length(pe.pixels)
        @test maximum(pe.errors) < 0.1   # synthetic plates: subpixel-exact
        @test occursin("rms", C.plane_summary(cr))
        @test occursin("3 planes", C.fit_summary(cr))
        set_plane!(cr, 99)
        @test cr.plane[] == 3

        # switching the model refits
        cr.model[] = :pinhole
        @test cr.camera[] isa PinholeCamera
        @test maximum(C.plane_errors(cr).errors) < 0.1

        # soloff needs 3 planes: 2-plane review reports the failed fit
        cr2 = CalibrationReview(plates[1:2], zs[1:2];
                                spacing = 15.0, origin_offset = (30.0, 7.5))
        @test cr2.camera[] === nothing
        @test !isempty(cr2.fit_message[])
        @test C.plane_errors(cr2) === nothing
        @test occursin("no fit", C.fit_summary(cr2))

        @test_throws ArgumentError CalibrationReview(plates, zs[1:2]; spacing = 15.0)

        # self-calibration summary (direct SelfCalPass/Report construction —
        # breaks when the report types gain fields, like the core fixtures)
        passes = [Hammerhead.SelfCalPass(2.85, 2.74, 1200, 0.10,
                                         (a = -0.674, b = 0.001, c = -0.004)),
                  Hammerhead.SelfCalPass(0.46, -0.03, 1200, 0.09, nothing)]
        report = SelfCalibrationReport(passes, false, 0.05,
                                       [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                       [0.0, 0.0, 0.674], [r_unc, r_plain])
        s = C.selfcal_summary(report)
        @test occursin("pass 1", s) && occursin("plane a = -0.674", s)
        @test occursin("no correction", s)
        @test occursin("not converged", s)
        @test occursin("shift 0.674", s)
    end

    @testset "calibration & selfcal views (offscreen)" begin
        θ = deg2rad(20.0)
        R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
        camC = R' * [0.0, 0.0, -500.0]
        K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
        cam = PinholeCamera(K, R, -R * camC)
        zs = [-3.0, 0.0, 3.0]
        plates = [render_calibration_target(cam, (512, 512); spacing = 15.0,
                                            z = z,
                                            marker_square = (-30.0, -7.5),
                                            marker_triangle = (-15.0, -7.5))
                  for z in zs]
        cr = CalibrationReview(plates, zs;
                               spacing = 15.0, origin_offset = (30.0, 7.5))
        fig = calibration_review(cr)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (720, 1000)
        set_plane!(cr, 2)
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test img2 != img1

        passes = [Hammerhead.SelfCalPass(2.85, 2.74, 1200, 0.10,
                                         (a = -0.674, b = 0.001, c = -0.004)),
                  Hammerhead.SelfCalPass(0.46, -0.03, 1200, 0.09, nothing)]
        with_maps = SelfCalibrationReport(passes, false, 0.05,
                                          [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                          [0.0, 0.0, 0.674], [r_unc, r_plain])
        figm = selfcal_review(with_maps)
        @test !isempty(colorbuffer(figm; px_per_unit = 1))
        without = SelfCalibrationReport(passes, false, 0.05,
                                        [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                        [0.0, 0.0, 0.674], PIVResult[])
        fign = selfcal_review(without)
        @test !isempty(colorbuffer(fign; px_per_unit = 1))
    end
end
