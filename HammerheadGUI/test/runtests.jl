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
end
