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
    include("test_experiments.jl")
    include("test_workflow_layout.jl")
    include("test_experiment_replay_progress.jl")
    include("test_recipe_comparison.jl")
    include("test_recipe_revision.jl")
    include("test_recipe_revision_view.jl")
    include("test_tracking_timing_explorer.jl")
    include("test_checkpoints.jl")
    include("test_companions.jl")
    include("test_stereo_companions.jl")
    include("test_ensemble_companions.jl")
    include("test_stereo_experiments.jl")
    include("test_ensemble_experiments.jl")
    include("test_derivative_support.jl")
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

        # field inventory: uncertainty entries only where estimates exist,
        # derived fields on every planar grid with >= 2 points per axis
        derived = collect(HammerheadGUI.Controllers.DERIVED_FIELDS)
        @test available_fields(r_unc) ==
              vcat([:magnitude, :u, :v, :peak_ratio, :correlation_moment,
                    :uncertainty_u, :uncertainty_v], derived)
        @test available_fields(r_plain) ==
              vcat([:magnitude, :u, :v, :peak_ratio, :correlation_moment], derived)
        @test :w in available_fields(r_stereo)
        @test !any(in(available_fields(r_stereo)), derived)   # planar only

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
        @test C.field_label(rs, :magnitude) == "speed (mm/s)"
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

    @testset "ResultExplorer color limits (no GL)" begin
        C = HammerheadGUI.Controllers

        # 5x5 grid with a huge injected outlier and one masked NaN cell
        u = Float64.(reshape(1:25, 5, 5))
        outl = falses(5, 5); outl[3, 3] = true; u[3, 3] = 1000.0
        mask = falses(5, 5); mask[1, 1] = true
        u[1, 1] = NaN
        rz = PIVResult(collect(1.0:5.0), collect(1.0:5.0), u, zeros(5, 5),
                       ones(5, 5), ones(5, 5), fill(NaN, 5, 5), fill(NaN, 5, 5),
                       outl, mask, PIVParameters(window_size = 16, overlap = (8, 8)))

        # robust excludes the flagged outlier (and the masked cell); full keeps it
        lo, hi = C.color_limits(rz, :u)
        @test hi < 1000.0 && lo >= 2.0
        @test C.color_limits(rz, :u, :full)[2] == 1000.0
        @test_throws ArgumentError C.color_limits(rz, :u, :nope)

        # all-flagged fallback: percentiles over all finite values
        rf = PIVResult(collect(1.0:5.0), collect(1.0:5.0), u, zeros(5, 5),
                       ones(5, 5), ones(5, 5), fill(NaN, 5, 5), fill(NaN, 5, 5),
                       trues(5, 5), falses(5, 5),
                       PIVParameters(window_size = 16, overlap = (8, 8)))
        lof, hif = C.color_limits(rf, :u)
        @test isfinite(lof) && isfinite(hif) && lof < hif

        # PTV: the flagged particle's huge residual is excluded under :robust
        @test C.color_limits(r_ptv, :match_residual)[2] < 5.0
        @test C.color_limits(r_ptv, :match_residual, :full)[2] == 5.0

        # manual overrides win bound-wise and persist across frame changes
        ex = ResultExplorer([rz, rz])
        @test current_color_limits(ex) == C.color_limits(rz, :u, :robust)
        set_color_limits!(ex; min = "0", max = 10.0)
        @test current_color_limits(ex) == (0.0, 10.0)
        set_frame!(ex, 2)
        set_field!(ex, :v)
        @test current_color_limits(ex) == (0.0, 10.0)
        set_color_limits!(ex; max = "auto")   # clear one bound
        @test current_color_limits(ex)[1] == 0.0
        @test current_color_limits(ex)[2] != 10.0
        set_color_limits!(ex; min = nothing)
        set_color_mode!(ex, :full)
        @test ex.color_mode[] == :full
        @test_throws ArgumentError set_color_mode!(ex, :nope)
        @test_throws ArgumentError set_color_limits!(ex; min = "junk")
        @test_throws ArgumentError set_color_limits!(ex; max = NaN)

        # inverted manual pair is padded to a valid range
        set_color_limits!(ex; min = 5.0, max = 1.0)
        loi, hii = current_color_limits(ex)
        @test loi < hii
    end

    @testset "ResultExplorer derived fields + analysis tools (no GL)" begin
        C = HammerheadGUI.Controllers

        # solid-body rotation about the grid center: vorticity = 2Ω exactly,
        # circulation over any enclosed contour = 2Ω * area
        n, Ω, ctr = 21, 0.05, 10.0
        xs = collect(0.0:20.0)
        u = [-Ω * (y - ctr) for y in xs, x in xs]
        v = [Ω * (x - ctr) for y in xs, x in xs]
        rot = PIVResult(xs, copy(xs), u, v, ones(n, n), ones(n, n),
                        fill(NaN, n, n), fill(NaN, n, n), falses(n, n),
                        falses(n, n), PIVParameters(window_size = 16, overlap = (8, 8)))

        # derived fields match the direct core calls (hand-checked value)
        ω = field_values(rot, :vorticity)
        @test all(isapprox.(ω, 2Ω; atol = 1e-12))
        d = flow_derivatives(rot)
        @test isequal(field_values(rot, :divergence), divergence(d))
        @test isequal(field_values(rot, :strain_rate), strain_rate(d).magnitude)
        @test isequal(field_values(rot, :swirling_strength), swirling_strength(d))
        @test isequal(field_values(rot, :q_criterion), q_criterion(d))
        @test isequal(field_values(r_unc, :vorticity),
                      vorticity(flow_derivatives(r_unc)))

        # units: gradients are 1/time — the physical conversion (positions
        # and velocities share the length factor) leaves exactly 1/dt
        @test C.field_label(rot, :vorticity) == "vorticity (1/frame)"
        @test C.field_label(rot, :q_criterion) == "Q (1/frame²)"
        exs = ResultExplorer(with_scale(rot, PhysicalScale(2.0, 0.5, "mm", "s")))
        rs = current_result(exs)
        @test C.field_label(rs, :vorticity) == "vorticity (1/s)"
        @test C.field_label(rs, :q_criterion) == "Q (1/s²)"
        set_field!(exs, :vorticity)
        ωs = C.current_field_values(exs)
        @test all(isapprox.(ωs, 2Ω / 0.5; atol = 1e-12))   # 1/dt scaling
        @test C._derived(exs) === C._derived(exs)          # cached per frame
        lo, hi = current_color_limits(exs)
        @test isfinite(lo) && isfinite(hi)

        # profile tool: two clicks sample the line; a third starts over
        ex = ResultExplorer(rot)
        @test ex.tool[] == :inspect
        C.click!(ex, xs[3] + 0.2, xs[4] - 0.2)             # :inspect → selection
        @test ex.selection[] == CartesianIndex(4, 3)
        set_tool!(ex, :profile)
        @test occursin("two points", tool_summary(ex))
        C.click!(ex, 5.0, 10.0)
        @test ex.profile_data[] === nothing
        C.click!(ex, 15.0, 10.0)
        pd = ex.profile_data[]
        @test pd !== nothing && length(pd.s) == 100
        @test pd.u[1] ≈ 0.0 atol = 1e-12                   # u = -Ω(y-yc) = 0 on y = 10
        @test pd.v[1] ≈ -5Ω atol = 1e-12                   # v = Ω(x-yc) at x = 5
        @test pd.v[end] ≈ 5Ω atol = 1e-12
        @test occursin("u (blue)", tool_summary(ex))
        C.click!(ex, 0.0, 0.0)                             # restart
        @test length(ex.tool_points[]) == 1 && ex.profile_data[] === nothing

        # circulation tool: polygon + alt-click close, both estimators
        set_tool!(ex, :circulation)
        @test isempty(ex.tool_points[])
        for p in ((5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0))
            C.click!(ex, p...)
        end
        C.alt_click!(ex)
        res = ex.circulation_result[]
        @test res !== nothing
        @test res.line ≈ 2Ω * 100 atol = 1e-9              # exact on the linear field
        @test res.area ≈ 2Ω * 100 atol = 1e-9               # exact for constant vorticity
        @test res.coverage_fraction == 1.0
        @test res.valid_area ≈ res.requested_area ≈ 100.0
        @test occursin("Γ (line)", tool_summary(ex))
        @test occursin("px²/frame", tool_summary(ex))
        C.click!(ex, 1.0, 1.0)                             # closed → new contour
        @test length(ex.tool_points[]) == 1 && ex.circulation_result[] === nothing
        C.alt_click!(ex)                                   # < 3 vertices: cancel
        @test isempty(ex.tool_points[])
        clear_tool!(ex)
        @test ex.circulation_result[] === nothing

        # scaled circulation carries length²/time
        set_tool!(exs, :circulation)
        for p in ((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0))
            C.click!(exs, p...)                            # converted coords (×2)
        end
        C.alt_click!(exs)
        @test occursin("mm²/s", tool_summary(exs))
        @test exs.circulation_result[].line ≈ 2Ω / 0.5 * 400 atol = 1e-6
        @test exs.circulation_result[].coverage_fraction == 1.0

        # Partly outside the grid: retain the partial integral and show coverage.
        set_tool!(ex, :circulation)
        for p in ((10.0, 10.0), (25.0, 10.0), (25.0, 15.0), (10.0, 15.0))
            C.click!(ex, p...)
        end
        C.alt_click!(ex)
        partial = ex.circulation_result[]
        @test partial.requested_area ≈ 75.0
        @test partial.valid_area ≈ 50.0
        @test partial.coverage_fraction ≈ 2 / 3
        @test partial.area ≈ 2Ω * partial.valid_area atol = 1e-9
        @test occursin("partial", tool_summary(ex))
        @test occursin("coverage", tool_summary(ex))

        # Masked data and an entirely outside contour both report no valid area.
        invalid = PIVResult(xs, copy(xs), u, v, ones(n, n), ones(n, n),
                            fill(NaN, n, n), fill(NaN, n, n), falses(n, n),
                            trues(n, n), PIVParameters(window_size = 16, overlap = (8, 8)))
        ex_invalid = ResultExplorer(invalid)
        set_tool!(ex_invalid, :circulation)
        for p in ((5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0))
            C.click!(ex_invalid, p...)
        end
        C.alt_click!(ex_invalid)
        @test isnan(ex_invalid.circulation_result[].area)
        @test ex_invalid.circulation_result[].coverage_fraction == 0
        @test occursin("no valid area", tool_summary(ex_invalid))
        set_tool!(ex, :circulation)
        for p in ((30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0))
            C.click!(ex, p...)
        end
        C.alt_click!(ex)
        @test isnan(ex.circulation_result[].area)
        @test ex.circulation_result[].requested_area ≈ 100.0
        @test ex.circulation_result[].valid_area == 0
        @test occursin("no valid area", tool_summary(ex))

        # The coverage metric uses the scaled coordinate area as well.
        set_tool!(exs, :circulation)
        for p in ((20.0, 20.0), (50.0, 20.0), (50.0, 30.0), (20.0, 30.0))
            C.click!(exs, p...)
        end
        C.alt_click!(exs)
        @test exs.circulation_result[].coverage_fraction ≈ 2 / 3
        @test exs.circulation_result[].valid_area ≈ 200.0
        @test occursin("mm²/s", tool_summary(exs))
        @test occursin("partial", tool_summary(exs))

        # non-planar results reject analysis tools; frame switches revert
        @test_throws ArgumentError set_tool!(ResultExplorer(r_stereo), :profile)
        @test_throws ArgumentError set_tool!(ex, :nope)
        exm = ResultExplorer([rot, r_ptv])
        set_tool!(exm, :circulation)
        C.click!(exm, 5.0, 5.0)
        set_frame!(exm, 2)
        @test exm.tool[] == :inspect
        @test isempty(exm.tool_points[])
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

    @testset "Lazy ResultExplorer controller (no GL)" begin
        C = HammerheadGUI.Controllers
        path = joinpath(mktempdir(), "lazy-mixed.jld2")
        scale = PhysicalScale(0.2, 0.5, "mm", "s")
        entries = Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}[
            with_scale(r_unc, scale), with_scale(r_stereo, scale),
            with_scale(r_ptv, scale), with_scale(r_track, scale),
            with_scale(r_plain, scale)]
        save_results(path, entries)
        eager = ResultExplorer(path)
        ex = ResultExplorer(path; lazy = true)
        @test eager.results isa Vector
        @test ex.results isa C._LazyDisplayResults
        @test nframes(ex) == 5 && ex.count[] == 5
        @test ex.results.source isa ResultFile
        @test ex.results.index == 1
        @test current_result(ex) === current_result(ex) # one read/conversion per frame
        @test current_result(ex).x == current_result(eager).x
        @test isequal(current_result(ex).u, current_result(eager).u)
        @test_throws ArgumentError push_result!(ex, r_plain)
        @test nframes(ex) == 5
        set_field!(ex, :vorticity)
        C.current_field_values(ex)
        @test length(ex.derived_cache) == 1
        set_tool!(ex, :profile)
        C.click!(ex, current_result(ex).x[1], current_result(ex).y[1])
        select_nearest!(ex, current_result(ex).x[1], current_result(ex).y[1])
        for i in (2, 3, 4, 5, 1)
            set_frame!(ex, i)
            set_frame!(eager, i)
            @test typeof(current_result(ex)) == typeof(current_result(eager))
            @test available_fields(current_result(ex)) == available_fields(current_result(eager))
            @test ex.field[] == eager.field[]
            @test isempty(ex.derived_cache)
            @test ex.tool[] == :inspect && isempty(ex.tool_points[])
            @test ex.results.index == i
        end
        @test current_result(ex).scale.length_unit == "mm"
        # Repeated derived browsing retains only the selected derivative tensor.
        for i in (1, 5, 1, 5)
            set_frame!(ex, i)
            set_field!(ex, :vorticity)
            C.current_field_values(ex)
            @test collect(keys(ex.derived_cache)) == [i]
        end
        @test occursin("one cached frame", sprint(show, MIME"text/plain"(), ex.results))
        set_frame!(ex, 99)
        @test ex.frame[] == 5
        set_frame!(ex, 0)
        @test ex.frame[] == 1
        ex.frame[] = 3
        @test current_result(ex) isa PTVResult
        @test ex.field[] == :magnitude
        @test_throws BoundsError (ex.frame[] = 99)
        @test ex.frame[] == 3 && current_result(ex) isa PTVResult

        invalid_path = joinpath(mktempdir(), "unreadable-entry.jld2")
        Hammerhead.jldopen(invalid_path, "w") do f
            f["format_version"] = Hammerhead.RESULTS_FORMAT_VERSION
            f["results/000001"] = r_unc
            f["results/000002"] = 123
            f["results/000003"] = r_ptv
        end
        broken = ResultExplorer(ResultFile(invalid_path))
        select_nearest!(broken, r_unc.x[1], r_unc.y[1])
        selected = broken.selection[]
        set_field!(broken, :vorticity)
        C.current_field_values(broken)
        set_tool!(broken, :profile)
        C.click!(broken, r_unc.x[1], r_unc.y[1])
        @test_throws ArgumentError set_frame!(broken, 2)
        @test broken.frame[] == 1 && broken.selection[] == selected
        @test broken.field[] == :vorticity && broken.tool[] == :profile
        @test length(broken.tool_points[]) == 1 && length(broken.derived_cache) == 1
        @test occursin("not a supported Hammerhead result", broken.status[])
        @test_throws ArgumentError (broken.frame[] = 2)
        @test broken.frame[] == 1 && broken.selection[] == selected
        @test current_result(broken) isa PIVResult
        set_frame!(broken, 3)
        @test current_result(broken) isa PTVResult
        @test isempty(broken.status[]) && broken.selection[] === nothing
        @test broken.field[] == :magnitude && isempty(broken.derived_cache)
        # A detectable external write also preserves the selected payload.
        Hammerhead.jldopen(invalid_path, "a+") do f
            f["results/000004"] = r_track
        end
        @test_throws ArgumentError set_frame!(broken, 1)
        @test broken.frame[] == 3 && current_result(broken) isa PTVResult
        @test occursin("changed since indexing", broken.status[])
        empty_path = joinpath(mktempdir(), "empty.jld2")
        save_results(empty_path, PIVResult[])
        @test_throws ArgumentError ResultExplorer(empty_path; lazy = true)
    end

    @testset "Lazy result_explorer view (offscreen)" begin
        path = joinpath(mktempdir(), "lazy-view.jld2")
        entries = Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}[
            with_scale(r_unc, PhysicalScale(0.2, 0.5, "mm", "s")),
            r_stereo, r_ptv, r_track]
        save_results(path, entries)
        ex = ResultExplorer(path; lazy = true)
        fig = result_explorer(ex; size = (900, 650))
        initial = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(initial) == (650, 900)
        for i in (2, 3, 4, 1)
            set_frame!(ex, i)
            @test size(colorbuffer(fig; px_per_unit = 1)) == size(initial)
        end
        @test !isempty(colorbuffer(result_explorer(path; lazy = true); px_per_unit = 1))
        # The view and controller release the prior payload after refresh.
        @noinline function weak_previous_display(ex)
            weak = WeakRef(current_result(ex).u)
            set_frame!(ex, 3)
            return weak
        end
        old_array = weak_previous_display(ex)
        colorbuffer(fig; px_per_unit = 1)
        GC.gc(true)
        @test old_array.value === nothing

        invalid_path = joinpath(mktempdir(), "lazy-view-error.jld2")
        Hammerhead.jldopen(invalid_path, "w") do f
            f["format_version"] = Hammerhead.RESULTS_FORMAT_VERSION
            f["results/000001"] = r_unc
            f["results/000002"] = 123
            f["results/000003"] = r_ptv
        end
        broken = ResultExplorer(invalid_path; lazy = true)
        broken_fig = result_explorer(broken; size = (1000, 700))
        before = copy(colorbuffer(broken_fig; px_per_unit = 1))
        slider = only(filter(b -> b isa Slider, broken_fig.content))
        set_close_to!(slider, 2)  # view catches, reports, and restores the slider
        @test broken.frame[] == 1 && slider.value[] == 1
        @test occursin("not a supported Hammerhead result", broken.status[])
        @test any(b -> b isa Label && b.text[] == broken.status[], broken_fig.content)
        @test colorbuffer(broken_fig; px_per_unit = 1) != before
        set_close_to!(slider, 3)
        @test broken.frame[] == 3 && slider.value[] == 3
        @test isempty(broken.status[]) && current_result(broken) isa PTVResult
        @test !isempty(colorbuffer(broken_fig; px_per_unit = 1))
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

        # color-range controls re-render too (mode switch + manual bound)
        img3 = copy(img2)
        set_color_mode!(ex, :full)
        set_color_limits!(ex; max = 0.5)
        img4 = colorbuffer(fig; px_per_unit = 1)
        @test img4 != img3

        # stereo view renders too
        figs = result_explorer(r_stereo)
        @test !isempty(colorbuffer(figs; px_per_unit = 1))

        # derived field renders, and the analysis tools draw their overlays:
        # a profile grows a side-panel axis, a circulation contour closes
        set_frame!(ex, 1)
        set_field!(ex, :vorticity)
        imgd = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(imgd) == size(img1)
        set_tool!(ex, :profile)
        HammerheadGUI.Controllers.click!(ex, r_unc.x[2], r_unc.y[2])
        HammerheadGUI.Controllers.click!(ex, r_unc.x[end - 1], r_unc.y[end - 1])
        @test ex.profile_data[] !== nothing
        imgp = copy(colorbuffer(fig; px_per_unit = 1))
        @test imgp != imgd
        set_tool!(ex, :circulation)
        for p in ((r_unc.x[2], r_unc.y[2]), (r_unc.x[end - 1], r_unc.y[2]),
                  (r_unc.x[end - 1], r_unc.y[end - 1]))
            HammerheadGUI.Controllers.click!(ex, p...)
        end
        HammerheadGUI.Controllers.alt_click!(ex)
        @test ex.circulation_result[] !== nothing
        imgc = copy(colorbuffer(fig; px_per_unit = 1))
        @test imgc != imgp
        set_tool!(ex, :inspect)

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

        # completed accumulates live during the run, in order
        bcl = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                          window_schedule = [32],
                          padding = false, apodization = :none)
        live_counts = Int[]
        on(v -> push!(live_counts, length(v)), bcl.completed)
        start!(bcl; async = false)
        @test live_counts == [0, 1, 2]   # reset, then one per finished pair
        @test length(bcl.completed[]) == 2
        @test bcl.completed[] == bcl.results[]

        # cancellation keeps the completed prefix
        bcc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                          window_schedule = [32],
                          padding = false, apodization = :none)
        on(p -> p[1] == 1 && cancel!(bcc), bcc.progress)
        start!(bcc; async = false)
        @test length(bcc.completed[]) == 1

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

    @testset "ROI editor and batch controller (no GL)" begin
        C = HammerheadGUI.Controllers
        ed = ROIEditor(imgA)
        @test ed.roi[] === nothing
        @test occursin("full image", C.roi_summary(ed))
        C.click!(ed, 111.6, 119.6)
        @test ed.anchor[] == (112, 120)
        C.click!(ed, 17.1, 24.8)   # opposite ordering, columns = x
        @test ed.roi[].rows == 25:120
        @test ed.roi[].cols == 17:112
        @test ed.anchor[] === nothing
        C.click!(ed, -100, 500)
        @test ed.anchor[] == (1, 128)
        @test ed.roi[].rows == 25:120   # selection retained while drawing
        @test_throws ArgumentError apply_roi!(BatchRunner(), ed)
        @test_throws ArgumentError C.click!(ed, NaN, 1)
        clear_roi!(ed)
        @test ed.roi[] === nothing && ed.anchor[] === nothing
        set_roi!(ed, "25", "120", "17", "112")
        previous = ed.roi[]
        @test_throws ArgumentError set_roi!(ed, "25.5", "120", "17", "112")
        @test_throws ArgumentError set_roi!(ed, 120, 25, 17, 112)
        @test_throws ArgumentError set_roi!(ed, 0, 100, 1, 100)
        @test_throws BoundsError set_roi!(ed, 1, 129, 1, 128)
        @test ed.roi[] === previous
        @test_throws ArgumentError ROIEditor(zeros(0, 2))
        # File-backed editing and preflight use the same bounds as matrices.
        frame_path = joinpath(mktempdir(), "roi-frame.png")
        C.FileIO.save(frame_path, C.Gray.(zeros(20, 30)))
        file_editor = ROIEditor(frame_path; roi = ROI(2:20, 3:30))
        @test size(file_editor.image) == (20, 30)
        file_batch = BatchRunner(files = Any[frame_path, frame_path], window_schedule = [8])
        apply_roi!(file_batch, file_editor)
        @test C.validate(file_batch) === nothing
        @test_throws BoundsError set_roi!(file_batch, ROI(1:21, 1:30))

        rr = ROI(25:120, 17:112)
        mask = falses(size(imgA)); mask[25:64, 17:56] .= true
        out = joinpath(mktempdir(), "roi-batch.jld2")
        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                         window_schedule = [32], mask = mask, output_path = out,
                         padding = false, apodization = :none,
                         pixel_size = 0.02, dt = 0.001,
                         length_unit = "mm", time_unit = "s")
        apply_roi!(bc, ed)
        @test C.validate(bc) === nothing
        @test_throws BoundsError set_roi!(bc, ROI(1:129, 1:128))
        @test bc.roi[] === previous
        before_a, before_b = copy(imgA), copy(imgB)
        seen_sizes = Tuple{Int,Int}[]
        set_preprocess!(bc, frame -> (push!(seen_sizes, size(frame)); 0.5 .* frame))
        # A mid-run form change affects only the next run, even with two pairs.
        on(bc.completed) do completed
            isempty(completed) || clear_roi!(bc)
        end
        start!(bc; async = false)
        @test occursin("done", bc.status[])
        @test length(bc.results[]) == 2
        @test seen_sizes == fill(size(imgA), 4) # preprocess full frames, then crop
        @test imgA == before_a && imgB == before_b
        expected = run_piv(0.5 .* imgA, 0.5 .* imgB, C.build_parameters(bc);
                           roi = rr, mask, scale = C.build_scale(bc))
        local_result = run_piv(0.5 .* imgA[rr.rows, rr.cols],
                               0.5 .* imgB[rr.rows, rr.cols], C.build_parameters(bc);
                               mask = mask[rr.rows, rr.cols])
        for result in bc.results[]
            @test result.x == expected.x == local_result.x .+ 16
            @test result.y == expected.y == local_result.y .+ 24
            @test isequal(result.u, expected.u)
            @test isequal(result.v, expected.v)
            @test result.mask == expected.mask && any(result.mask)
            @test result.scale.pixel_size == 0.02 && result.scale.dt == 0.001
            @test physical(result).x ≈ result.x .* 0.02
        end
        @test load_results(out)[1].x == expected.x
        @test load_results(out)[1].mask == expected.mask
        @test load_results(out)[1].scale.length_unit == "mm"

        # Core also accepts an already cropped mask; presets forward ROI too.
        bp = BatchRunner(files = Any[imgA, imgB], effort = :low,
                         roi = (rr.rows, rr.cols), mask = mask[rr.rows, rr.cols])
        @test C.validate(bp) === nothing
        start!(bp; async = false)
        preset = run_piv(imgA, imgB; effort = :low, roi = rr,
                         mask = mask[rr.rows, rr.cols])
        @test isequal(bp.results[][1].u, preset.u)
        @test bp.results[][1].x == preset.x
        bp.mask[] = falses(10, 10)
        @test occursin("mask must match", C.validate(bp))
        bp.mask[] = nothing
        bp.roi[] = ROI(1:129, 1:128) # direct observable edits still validate
        @test C.validate(bp) !== nothing
        start!(bp; async = false)
        @test !bp.running[] && !occursin("done", bp.status[])
        clear_roi!(bp)
        @test C.validate(bp) === nothing

        small = ROI(25:72, 17:64)
        rejected_path = joinpath(mktempdir(), "invalid-roi.jld2")
        small_batch = BatchRunner(files = Any[imgA, imgB], roi = small,
                                  window_schedule = [64, 32], output_path = rejected_path)
        @test occursin("smaller than search area", C.validate(small_batch))
        start!(small_batch; async = false)
        @test !small_batch.running[] && small_batch.results[] === nothing
        @test !isfile(rejected_path)  # reject before opening incremental output
        # Presets adapt their largest windows to the selected ROI dimensions.
        for effort in (:medium, :high)
            adapted = BatchRunner(files = Any[imgA, imgB], roi = small, effort = effort)
            @test C.validate(adapted) === nothing
            start!(adapted; async = false)
            @test occursin("done", adapted.status[])
            local_expected = run_piv(imgA[small.rows, small.cols],
                                      imgB[small.rows, small.cols]; effort)
            result = adapted.results[][1]
            @test result.x == local_expected.x .+ 16
            @test result.y == local_expected.y .+ 24
            @test isequal(result.u, local_expected.u)
            @test isequal(result.v, local_expected.v)
            @test result.parameters.window_size == local_expected.parameters.window_size
        end
        # ROI capture precedes even synchronous running-state observers.
        for async in (false, true)
            frozen = BatchRunner(files = Any[imgA, imgB], roi = rr,
                                 window_schedule = [32], padding = false,
                                 apodization = :none)
            on(running -> running && clear_roi!(frozen), frozen.running)
            start!(frozen; async)
            if async
                @test timedwait(() -> !frozen.running[], 30) == :ok
            end
            @test occursin("done", frozen.status[])
            @test frozen.roi[] === nothing
            @test first(frozen.results[][1].x) == first(expected.x)
            @test first(frozen.results[][1].y) == first(expected.y)
        end
    end

    @testset "roi_editor view (offscreen)" begin
        C = HammerheadGUI.Controllers
        ed = ROIEditor(imgA)
        bc = BatchRunner(files = Any[imgA, imgB])
        fig = roi_editor(ed; batch = bc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (560, 900)
        C.click!(ed, 20, 30)
        C.click!(ed, 110, 120)
        @test colorbuffer(fig; px_per_unit = 1) != img1
        boxes = filter(b -> b isa Textbox, fig.content)
        @test [box.displayed_string[] for box in boxes] == ["30", "120", "20", "110"]
        buttons = filter(b -> b isa Button, fig.content)
        button(label) = only(filter(b -> b.label[] == label, buttons))
        for (box, value) in zip(boxes, ("25", "90", "15", "100"))
            box.displayed_string[] = value
        end
        button("set bounds").clicks[] += 1
        @test ed.roi[].rows == 25:90 && ed.roi[].cols == 15:100
        button("apply to batch").clicks[] += 1
        @test bc.roi[].rows == 25:90 && bc.roi[].cols == 15:100
        boxes[1].displayed_string[] = "invalid"
        button("set bounds").clicks[] += 1
        @test ed.roi[].rows == 25:90
        @test any(b -> b isa Label && occursin("must be integers", b.text[]), fig.content)
        button("full image").clicks[] += 1
        button("apply to batch").clicks[] += 1
        @test ed.roi[] === nothing && bc.roi[] === nothing
        @test [box.displayed_string[] for box in boxes] == ["1", "128", "1", "128"]
        # Embedded view constructs and renders in a composite layout.
        embedded = Figure(size = (1100, 600))
        roi_editor!(embedded[1, 1], ROIEditor(imgA; roi = ROI(5:80, 7:90)))
        Label(embedded[1, 2], "comparison")
        @test !isempty(colorbuffer(embedded; px_per_unit = 1))
    end

    @testset "batch_runner view (offscreen)" begin
        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB],
                         window_schedule = [32],
                         padding = false, apodization = :none)
        fig = batch_runner(bc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (720, 960)
        set_roi!(bc, ROI(17:112, 25:120))
        @test colorbuffer(fig; px_per_unit = 1) != img1
        roi_reset = only(filter(b -> b isa Button && b.label[] == "full image", fig.content))
        roi_reset.clicks[] += 1
        @test bc.roi[] === nothing

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

    @testset "PreprocessPreview controller (no GL)" begin
        C = HammerheadGUI.Controllers

        pp = PreprocessPreview(imgA)
        @test all(s -> !s.enabled, pp.steps[])
        @test C.pipeline_summary(pp) == "no preprocessing"
        @test build_preprocess(pp) === nothing
        @test pp.processed[] == imgA              # identity pipeline

        # composition matches the direct core calls, in order
        enable_step!(pp, :intensity_cap)
        set_step_param!(pp, :intensity_cap, :n_sigma, "1.5")
        enable_step!(pp, :highpass_filter)
        direct = highpass_filter!(intensity_cap!(copy(imgA); n_sigma = 1.5); sigma = 3.0)
        @test pp.processed[] == direct
        @test apply_pipeline(pp, imgA) == direct
        f = build_preprocess(pp)
        @test f(imgA) == direct
        @test pp.processed[] == direct            # inputs never mutated
        keep = copy(imgA); f(keep); @test keep == imgA

        # snapshot semantics: editing after build does not change the closure
        set_step_param!(pp, :intensity_cap, :n_sigma, 3.0)
        @test f(imgA) == direct

        # toggling off restores the shorter pipeline
        enable_step!(pp, :intensity_cap, false)
        @test pp.processed[] == highpass_filter(imgA)

        # reordering matters and is clamped at the ends
        pp2 = PreprocessPreview(imgA; enabled = [:invert_image, :percentile_stretch])
        before = copy(pp2.processed[])
        move_step!(pp2, :invert_image, -5)        # invert now runs first
        @test pp2.processed[] != before
        @test first(filter(s -> s.enabled, pp2.steps[])).name == :invert_image
        move_step!(pp2, :invert_image, -1)        # already first: no-op
        @test first(pp2.steps[]).name == :invert_image

        # invalid parameters revert instead of wedging the pipeline
        @test_throws ArgumentError set_step_param!(pp2, :percentile_stretch, :low, 120.0)
        @test pp2.processed[] == apply_pipeline(pp2, imgA)   # still consistent
        @test_throws ArgumentError set_step_param!(pp2, :percentile_stretch, :low, "junk")
        @test_throws ArgumentError set_step_param!(pp2, :clahe, :nope, 1.0)
        @test_throws ArgumentError enable_step!(pp2, :nope)

        # background subtraction needs a computed background
        @test_throws ArgumentError enable_step!(pp2, :subtract_background)
        set_background!(pp2, [imgA, imgB])
        enable_step!(pp2, :subtract_background)
        @test pp2.background[] == min.(imgA, imgB)
        set_background!(pp2, nothing)             # clears and disables
        @test !C._step(pp2, :subtract_background).enabled

        # A built pipeline owns the background, even if the preview's matrix
        # is edited in place after it is handed to a batch.
        snapshot_preview = PreprocessPreview(fill(0.8, 4, 4))
        set_background!(snapshot_preview, [fill(0.2, 4, 4)])
        enable_step!(snapshot_preview, :subtract_background)
        frozen = build_preprocess(snapshot_preview)
        snapshot_preview.background[] .= 0.7
        @test frozen(fill(0.8, 4, 4)) ≈ fill(0.6, 4, 4)
        @test apply_pipeline(snapshot_preview, fill(0.8, 4, 4)) ≈ fill(0.1, 4, 4)

        # batch integration: the pipeline forwards into run_piv_sequence
        pp3 = PreprocessPreview(imgA; enabled = [:intensity_cap])
        bc = BatchRunner(files = Any[imgA, imgB], window_schedule = [32],
                         padding = false, apodization = :none)
        set_preprocess!(bc, pp3)
        @test bc.preprocess[] isa Function
        start!(bc; async = false)
        g = build_preprocess(pp3)
        direct_pp = run_piv(g(imgA), g(imgB),
                            multipass_parameters([32]; padding = false,
                                                 apodization = :none))
        @test bc.results[][1].u == direct_pp.u
        @test imgA == pp3.image[]                 # batch never mutated the frames
        set_preprocess!(bc, nothing)
        @test bc.preprocess[] === nothing
    end

    @testset "PreprocessPreview correlation probe (no GL)" begin
        C = HammerheadGUI.Controllers

        # no pair / no click: probe inactive with explanatory summaries
        pp0 = PreprocessPreview(imgA)
        C.click!(pp0, 64.0, 64.0)
        @test pp0.probe_result[] === nothing
        @test occursin("pair frame", probe_summary(pp0))

        pp = PreprocessPreview(imgA; pair = imgB)
        @test pp.probe_result[] === nothing
        @test occursin("click", probe_summary(pp))

        # probe on the known uniform shift (fixtures: u ≈ 3, v ≈ 2)
        C.click!(pp, 64.0, 64.0)
        res = pp.probe_result[]
        @test res !== nothing && !res.clamped
        @test res.du ≈ 3.0 atol = 0.3
        @test res.dv ≈ 2.0 atol = 0.3
        @test res.peak_ratio > 1.0
        s = probe_summary(pp)
        @test occursin("du = ", s) && occursin("peak ratio", s)

        # the numbers follow the pipeline live
        enable_step!(pp, :highpass_filter)
        res2 = pp.probe_result[]
        @test res2 !== nothing && res2 != res            # recomputed
        @test res2.du ≈ 3.0 atol = 0.3                   # still the true shift
        enable_step!(pp, :highpass_filter, false)
        @test pp.probe_result[].peak_ratio ≈ res.peak_ratio

        # border clicks clamp the window instead of throwing
        C.click!(pp, 2.0, 2.0)
        resb = pp.probe_result[]
        @test resb !== nothing && resb.clamped
        @test resb.x0 == 1 && resb.y0 == 1
        @test isfinite(resb.du)
        @test occursin("clamped", probe_summary(pp))

        # window-size handling: parse, validation, too-big windows
        set_probe_window!(pp, "32")
        @test pp.probe_window[] == 32
        @test pp.probe_result[].window == 32
        @test_throws ArgumentError set_probe_window!(pp, 7)
        @test_throws ArgumentError set_probe_window!(pp, "abc")
        set_probe_window!(pp, 256)                        # larger than the 128² frame
        @test pp.probe_result[] === nothing
        @test occursin("does not fit", probe_summary(pp))
        set_probe_window!(pp, 64)
        @test pp.probe_result[] !== nothing

        # clearing the probe and the pair
        clear_probe!(pp)
        @test pp.probe_result[] === nothing
        set_pair!(pp, nothing)
        @test occursin("pair frame", probe_summary(pp))
        @test_throws ArgumentError set_pair!(pp, rand(16, 16))
    end

    @testset "preprocess_preview view (offscreen)" begin
        pp = PreprocessPreview(imgA; enabled = [:percentile_stretch])
        fig = preprocess_preview(pp)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (640, 1200)

        # drive through the controller: preview and summary refresh
        enable_step!(pp, :invert_image)
        move_step!(pp, :invert_image, -10)
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test size(img2) == size(img1)
        @test img2 != img1

        # probe rectangle + numbers render and follow the probe
        ppp = PreprocessPreview(imgA; pair = imgB)
        figp = preprocess_preview(ppp)
        imgp1 = copy(colorbuffer(figp; px_per_unit = 1))
        HammerheadGUI.Controllers.click!(ppp, 64.0, 64.0)
        imgp2 = colorbuffer(figp; px_per_unit = 1)
        @test imgp2 != imgp1
    end

    @testset "live results during a batch (offscreen)" begin
        # push_result! grows the sequence (through physical) and notifies count
        scale = PhysicalScale(2.0, 1.0, "mm", "frame")
        exg = ResultExplorer(r_plain)
        counts = Int[]
        on(n -> push!(counts, n), exg.count)
        push_result!(exg, with_scale(r_plain, scale))
        @test nframes(exg) == 2 && counts == [2]
        @test current_result(exg) === r_plain            # frame unchanged
        set_frame!(exg, 2)
        @test current_result(exg).scale !== nothing      # converted on append
        @test current_result(exg).x[1] == 2.0 * r_plain.x[1]

        # open an explorer mid-run from the live `completed` accumulator and
        # follow the rest of the batch as it appends
        bc = BatchRunner(files = Any[imgA, imgB, imgA, imgB, imgA, imgB],
                         window_schedule = [32],
                         padding = false, apodization = :none)
        live = Ref{Any}(nothing)
        opened_at = Ref(0)
        on(bc.completed) do v
            if live[] === nothing && !isempty(v)
                live[] = ResultExplorer(copy(v))        # opened mid-run
                opened_at[] = length(v)
            elseif live[] !== nothing
                while nframes(live[]) < length(v)       # live append
                    push_result!(live[], v[nframes(live[]) + 1])
                end
            end
        end
        start!(bc; async = true)
        t0 = time()
        while bc.running[] && time() - t0 < 60
            sleep(0.02)
        end
        @test !bc.running[]
        @test opened_at[] == 1                           # opened after pair 1
        @test nframes(live[]) == 3                       # followed to the end

        # the view's frame slider grows with the appends
        fig = result_explorer(live[])
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        push_result!(live[], r_plain)
        set_frame!(live[], nframes(live[]))
        @test live[].frame[] == 4
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test size(img2) == size(img1)
    end

    @testset "ScaleTool controller (no GL)" begin
        C = HammerheadGUI.Controllers

        st = ScaleTool(imgA)
        @test C.pixel_distance(st) === nothing
        @test pixel_size(st) === nothing
        @test physical_scale(st) === nothing
        @test occursin("click two points", C.scale_summary(st))

        # two clicks define the line; a third starts a new one
        C.click!(st, 10.0, 20.0)
        @test pixel_size(st) === nothing
        C.click!(st, 10.0, 70.0)                  # 50 px vertical line
        @test C.pixel_distance(st) ≈ 50.0
        set_separation!(st, "25")                 # 25 mm over 50 px
        st.dt[] = 0.002
        st.time_unit[] = "s"
        @test pixel_size(st) ≈ 0.5
        sc = physical_scale(st)
        @test sc isa PhysicalScale
        @test sc.pixel_size ≈ 0.5 && sc.dt == 0.002
        @test sc.length_unit == "mm" && sc.time_unit == "s"
        @test occursin("mm/px", C.scale_summary(st))
        C.click!(st, 1.0, 1.0)                    # restart
        @test length(st.points[]) == 1
        C.clear_points!(st)
        @test isempty(st.points[])

        # validation
        @test_throws ArgumentError set_separation!(st, -1.0)
        @test_throws ArgumentError set_separation!(st, "nope")
        @test_throws ArgumentError C.set_dt!(st, 0.0)
        # coincident points define no scale
        C.click!(st, 5.0, 5.0); C.click!(st, 5.0, 5.0)
        @test pixel_size(st) === nothing

        # hand-off into the batch form
        st2 = ScaleTool(imgA)
        C.click!(st2, 0.0, 0.0); C.click!(st2, 30.0, 40.0)   # 50 px diagonal
        set_separation!(st2, 5.0)                            # 0.1 mm/px
        st2.dt[] = 0.01
        st2.time_unit[] = "s"
        bc = BatchRunner()
        @test_throws ArgumentError apply_scale!(bc, ScaleTool(imgA))
        apply_scale!(bc, st2)
        @test bc.pixel_size[] ≈ 0.1
        @test bc.dt[] == 0.01
        @test bc.length_unit[] == "mm" && bc.time_unit[] == "s"
        @test C.build_scale(bc) isa PhysicalScale
    end

    @testset "scale_tool view (offscreen)" begin
        st = ScaleTool(imgA)
        bc = BatchRunner()
        fig = scale_tool(st; batch = bc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (560, 900)
        HammerheadGUI.Controllers.click!(st, 20.0, 20.0)
        HammerheadGUI.Controllers.click!(st, 100.0, 20.0)
        set_separation!(st, 8.0)
        img2 = colorbuffer(fig; px_per_unit = 1)
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

        # SelfCalPass stores a median disparity magnitude, not a signed median.
        passes = [Hammerhead.SelfCalPass(2.85, 2.74, 1200, 0.10,
                                         (a = -0.674, b = 0.001, c = -0.004)),
                  Hammerhead.SelfCalPass(0.46, 0.03, 1200, 0.09, nothing)]
        report = SelfCalibrationReport(passes, false, 0.05,
                                       [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                       [0.0, 0.0, 0.674], [r_unc, r_plain])
        s = C.selfcal_summary(report)
        @test occursin("pass 1", s) && occursin("plane a = -0.674", s)
        @test occursin("no correction", s)
        @test occursin("not converged", s)
        @test occursin("median magnitude", s)
        @test occursin("RMS above", s)
        @test occursin("final disparity map", s)
        @test !occursin("signed median", s)
        @test occursin("shift 0.674", s)

        # Equal and opposite component residuals have zero signed median,
        # but nonzero magnitude and RMS. The review must not recommend that median.
        opposed_u = [1.0 -1.0; 1.0 -1.0]
        @test median(vec(opposed_u)) == 0.0
        opposed = PIVResult([1.0, 2.0], [1.0, 2.0], opposed_u, zeros(2, 2),
                            ones(2, 2), ones(2, 2), fill(NaN, 2, 2), fill(NaN, 2, 2),
                            falses(2, 2), falses(2, 2), PIVParameters(window_size = 16, overlap = 8))
        opposed_pass = Hammerhead.SelfCalPass(1.0, 1.0, 4, NaN, nothing)
        opposed_report = SelfCalibrationReport([opposed_pass], false, 0.05,
                                                [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                                [0.0, 0.0, 0.0], [opposed])
        opposed_summary = C.selfcal_summary(opposed_report)
        @test occursin("median magnitude 1", opposed_summary)
        @test occursin("RMS 1", opposed_summary)
        @test occursin("final disparity map", opposed_summary)
        @test !occursin("signed median", opposed_summary)

        without_maps = SelfCalibrationReport([opposed_pass], false, 0.05,
                                              [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                              [0.0, 0.0, 0.0], PIVResult[])
        @test occursin("keep_disparity_maps = true", C.selfcal_summary(without_maps))
        converged = SelfCalibrationReport([opposed_pass], true, 1.1,
                                           [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
                                           [0.0, 0.0, 0.0], PIVResult[])
        converged_summary = C.selfcal_summary(converged)
        @test occursin("converged (RMS tolerance", converged_summary)
        @test !occursin("not converged", converged_summary)
    end

    @testset "StereoBatchRunner controller (no GL)" begin
        C = HammerheadGUI.Controllers

        # two-camera synthetic rig at ±20° (make_test_camera recipe)
        function stereo_fixture(θdeg)
            θ = deg2rad(θdeg)
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
            return CalibrationReview(plates, zs;
                                     spacing = 15.0, origin_offset = (30.0, 7.5)),
                   plates
        end
        cr1, plates1 = stereo_fixture(20.0)
        cr2, plates2 = stereo_fixture(-20.0)

        # dewarper construction from the fitted reviews
        dw1, dw2 = build_dewarpers(cr1, cr2)
        @test dw1.grid == dw2.grid
        @test !all(dw1.mask .| dw2.mask)          # a shared visible region exists
        unfit = CalibrationReview(plates1[1:2], [-3.0, 0.0];
                                  spacing = 15.0, origin_offset = (30.0, 7.5))
        @test_throws ArgumentError build_dewarpers(unfit, cr2)

        # validation walks the failure modes in order
        sbc = StereoBatchRunner()
        @test occursin("dewarpers", C.validate(sbc))
        set_dewarpers!(sbc, dw1, dw2)
        @test occursin("both cameras", C.validate(sbc))
        add_files!(sbc, Any[plates1[2], plates1[2]]; camera = 1)
        @test occursin("both cameras", C.validate(sbc))
        add_files!(sbc, Any[plates2[2], plates2[2], plates2[2], plates2[2]]; camera = 2)
        @test occursin("pairs", C.validate(sbc))
        clear_files!(sbc)
        @test_throws ArgumentError add_files!(sbc, Any[plates1[1]]; camera = 3)

        # mismatched dewarp grids are rejected
        othergrid = DewarpGrid(x = -8.0:0.5:8.0, y = 8.0:-0.5:-8.0)
        dw_other = ImageDewarper(cr2.camera[], othergrid, (512, 512))
        @test_throws ArgumentError set_dewarpers!(sbc, dw1, dw_other)

        # synchronized run on identical frames (zero displacement), with a
        # dt-only scale and live accumulation
        out = joinpath(mktempdir(), "stereo_batch.jld2")
        sbc2 = StereoBatchRunner(dewarpers = (dw1, dw2),
                                 window_schedule = [32],
                                 output_path = out,
                                 dt = 0.01, length_unit = "mm", time_unit = "s")
        add_files!(sbc2, Any[plates1[2], plates1[2]]; camera = 1)
        add_files!(sbc2, Any[plates2[2], plates2[2]]; camera = 2)
        @test C.validate(sbc2) === nothing
        sc = C.build_scale(sbc2)
        @test sc isa PhysicalScale && sc.pixel_size == 1.0 && sc.dt == 0.01
        live = Int[]
        on(v -> push!(live, length(v)), sbc2.completed)
        start!(sbc2; async = false)
        @test sbc2.results[] !== nothing && length(sbc2.results[]) == 1
        r = sbc2.results[][1]
        @test r isa StereoPIVResult
        @test abs(median(filter(!isnan, r.u))) < 0.1   # identical frames
        @test r.scale !== nothing && r.scale.time_unit == "s"
        @test live == [0, 1]
        @test length(load_results(out)) == 1
        @test occursin("done", sbc2.status[])

        # native between-acquisition cancellation keeps the prefix, no throw
        sbc3 = StereoBatchRunner(dewarpers = (dw1, dw2), window_schedule = [32])
        add_files!(sbc3, Any[plates1[2], plates1[2], plates1[2], plates1[2]]; camera = 1)
        add_files!(sbc3, Any[plates2[2], plates2[2], plates2[2], plates2[2]]; camera = 2)
        on(p -> p[1] == 1 && cancel!(sbc3), sbc3.progress)
        start!(sbc3; async = false)
        @test length(sbc3.results[]) == 1
        @test occursin("cancelled after 1 of 2", sbc3.status[])

        # the explorer browses the stereo batch output directly
        exs = ResultExplorer(sbc2.results[])
        @test current_result(exs) isa StereoPIVResult

        @test_throws ArgumentError set_effort!(sbc2, :turbo)
        @test_throws ArgumentError C.set_dt!(sbc2, "-2")
        set_schedule!(sbc2, "48 24")
        @test sbc2.window_schedule[] == [48, 24]
    end

    @testset "stereo batch & calibration views (offscreen)" begin
        θ = deg2rad(20.0)
        R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
        camC = R' * [0.0, 0.0, -500.0]
        K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
        cam1 = PinholeCamera(K, R, -R * camC)
        R2 = [cos(-θ) 0.0 -sin(-θ); 0.0 1.0 0.0; sin(-θ) 0.0 cos(-θ)]
        cam2C = R2' * [0.0, 0.0, -500.0]
        cam2 = PinholeCamera(K, R2, -R2 * cam2C)
        zs = [-3.0, 0.0, 3.0]
        mk = (; spacing = 15.0, marker_square = (-30.0, -7.5),
              marker_triangle = (-15.0, -7.5))
        plates1 = [render_calibration_target(cam1, (512, 512); z, mk...) for z in zs]
        plates2 = [render_calibration_target(cam2, (512, 512); z, mk...) for z in zs]
        cr1 = CalibrationReview(plates1, zs; spacing = 15.0, origin_offset = (30.0, 7.5))
        cr2 = CalibrationReview(plates2, zs; spacing = 15.0, origin_offset = (30.0, 7.5))

        # rig-setup workflow view: embedded reviews render, and the build
        # path installs the dewarpers into a batch controller
        sbc = StereoBatchRunner()
        fig = stereo_calibration(cr1, cr2; batch = sbc)
        img1 = copy(colorbuffer(fig; px_per_unit = 1))
        @test size(img1) == (850, 1500)
        set_dewarpers!(sbc, cr1, cr2)          # the button's code path
        @test sbc.dewarpers[] !== nothing
        set_plane!(cr1, 2)                     # embedded review stays live
        img2 = colorbuffer(fig; px_per_unit = 1)
        @test img2 != img1

        # batch form renders and reflects controller changes
        figb = stereo_batch_runner(sbc)
        imgb1 = copy(colorbuffer(figb; px_per_unit = 1))
        @test size(imgb1) == (600, 960)
        add_files!(sbc, Any[plates1[2], plates1[2]]; camera = 1)
        add_files!(sbc, Any[plates2[2], plates2[2]]; camera = 2)
        set_effort!(sbc, :low)
        imgb2 = colorbuffer(figb; px_per_unit = 1)
        @test imgb2 != imgb1
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
                  Hammerhead.SelfCalPass(0.46, 0.03, 1200, 0.09, nothing)]
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
