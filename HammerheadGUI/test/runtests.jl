using Test
using HammerheadGUI
using HammerheadGUI.GLMakie
using HammerheadGUI.Hammerhead
using Statistics: median

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
        # Periodic uniform shift: a pixel at (row, col) in A appears at
        # (row + 2, col + 3) in B, so (du, dv) = (3, 2) by the package
        # sign convention.
        imgA = rand(128, 128)
        imgB = circshift(imgA, (2, 3))
        result = run_piv(imgA, imgB, PIVParameters(window_size = 32))
        @test median(result.u) ≈ 3.0 atol = 0.3
        @test median(result.v) ≈ 2.0 atol = 0.3

        # GLMakie loads Makie, so Hammerhead's Makie extension must be
        # active and renderable through the GL backend.
        fig = plot_vector_field(result)
        @test fig isa Figure
        img = colorbuffer(fig; px_per_unit = 1)
        @test !isempty(img)
    end
end
