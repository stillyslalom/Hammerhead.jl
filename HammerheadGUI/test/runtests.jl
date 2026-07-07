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
    end

    @testset "batch_runner view (offscreen)" begin
        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                         window_schedule = [32],
                         padding = false, apodization = :none)
        fig = batch_runner(bc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (520, 900)

        set_schedule!(bc, "64 32")   # form summary label updates
        bc.uncertainty[] = true      # toggle syncs back into the widget
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
