using Test, Hammerhead, HammerheadGUI, HammerheadGUI.Controllers, Observables
include("adapter.jl")
using .Prototype

@testset "QML candidate controller adapter" begin
    state = Prototype.State()
    @test length(current_result(state.explorer).u) == 16384
    @test !Prototype.schedule(state, "32, nope")
    @test state.batch.window_schedule[] == [32]
    @test !isempty(state.schedule_error[])
    @test Prototype.schedule(state, "32, 16")
    @test isempty(state.schedule_error[])
    old = state.explorer
    @test !Prototype.open_results(state, joinpath(@__DIR__, "missing.jld2"))
    @test state.explorer === old
    @test !isempty(state.open_error[])
    mktempdir() do dir
        path = joinpath(dir, "results.jld2")
        save_results(path, [Prototype.dense_result(8), Prototype.dense_result(12)])
        @test Prototype.open_results(state, path)
        @test state.explorer.results isa HammerheadGUI.Controllers._LazyDisplayResults
        @test state.count[] == 2
        @test Prototype.navigate(state, 2)
        @test state.frame[] == 2
        Prototype.pick(state, 4, 4)
        @test state.explorer.selection[] !== nothing
        @test !isempty(state.selection[])
        selection = state.explorer.selection[]
        frame = state.frame[]
        mixed = joinpath(dir, "mixed.jld2")
        save_results(mixed, [Prototype.dense_result(8), with_scale(Prototype.dense_result(8), PhysicalScale(pixel_size = 0.2))])
        @test Prototype.open_results(state, mixed)
        @test !Prototype.navigate(state, 2)
        @test state.frame[] == 1
        @test state.explorer.frame[] == 1
        @test occursin("unscaled", state.open_error[])
        scaled = joinpath(dir, "scaled.jld2")
        save_results(scaled, [with_scale(Prototype.dense_result(8), PhysicalScale(pixel_size = 0.2))])
        previous = state.explorer
        @test !Prototype.open_results(state, scaled)
        @test state.explorer === previous
    end
    state.explorer = old
    Prototype.navigate(state, 1)
    foreach(p -> Prototype.pick(state, p...; drawing = true), [(10, 10), (20, 10), (20, 20)])
    Prototype.close_mask(state)
    @test length(state.mask.polygons[]) == 1
    @test any(state.batch.mask[])
    # Cooperative cancellation retains the completed first pair.
    on(state.batch.completed) do results
        isempty(results) || Prototype.cancel_batch(state)
    end
    @test Prototype.run_batch(state)
    Prototype.wait_batch(state)
    @test occursin("cancelled", state.status[])
    @test length(state.batch.completed[]) == 1
    @test state.count[] == 1
    @test state.frame[] == 1
end
