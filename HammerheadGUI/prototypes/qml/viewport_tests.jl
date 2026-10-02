using Test, GLMakie, Hammerhead, TOML
include("adapter.jl")
include("viewport.jl")
state = Prototype.State()
fig, ax = viewport(state)
mkpath(joinpath(@__DIR__, "artifacts"))
const started = time_ns()
@testset "Standalone candidate viewport (not native Qt input)" begin
    initial = copy(colorbuffer(fig; px_per_unit = 1))
    @test size(initial) == (650, 900)
    @test length(current_result(state.explorer).u) == 16384
    @test ax.yreversed[]
    save(joinpath(@__DIR__, "artifacts", "dense_vectors.png"), fig; visible = false)
    # Programmatic image zoom/pan validates transformed plotting, not native gestures.
    limits!(ax, 12, 60, 8, 56)
    ax.yreversed[] = true
    panned = copy(colorbuffer(fig; px_per_unit = 1))
    @test panned != initial
    limits_before = ax.finallimits[]
    Prototype.pick(state, 20, 20)
    @test ax.finallimits[] == limits_before
    @test ax.yreversed[]
    @test state.explorer.selection[] !== nothing
    selected = copy(colorbuffer(fig; px_per_unit = 1))
    @test selected != initial
    foreach(p -> Prototype.pick(state, p...; drawing = true), [(10, 10), (30, 10), (30, 30)])
    Prototype.close_mask(state)
    @test any(state.batch.mask[])
    @test colorbuffer(fig; px_per_unit = 1) != selected
    # File coordinates outside the demo extent must determine limits.
    mktempdir() do dir
        r = Prototype.dense_result(8)
        moved = PIVResult(r.x .+ 1000, r.y .+ 2000, r.u, r.v, r.peak_ratio,
                          r.correlation_moment, r.uncertainty_u, r.uncertainty_v,
                          r.outliers, r.mask, r.parameters)
        path = joinpath(dir, "translated.jld2")
        save_results(path, [moved])
        @test Prototype.open_results(state, path)
        colorbuffer(fig; px_per_unit = 1)
        @test ax.finallimits[].origin[1] > 900
        @test ax.finallimits[].origin[2] > 1900
        @test !ax.scene.plots[1].visible[]
        @test ax.yreversed[]
    end
end
open(joinpath(@__DIR__, "artifacts", "standalone_viewport.toml"), "w") do io
    TOML.print(io, Dict("julia" => string(VERSION), "os" => string(Sys.KERNEL),
                       "elapsed_seconds" => (time_ns() - started) / 1e9,
                       "vectors" => 16384, "image_pixels" => 1024^2,
                       "retained_state_bytes" => Base.summarysize(state),
                       "native_input_measured" => false))
end
GLMakie.closeall()
