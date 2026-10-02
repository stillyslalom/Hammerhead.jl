# Scientific application ownership over hidden GLFW screens, not Qt teardown.
using Test, GLMakie, Observables
include("adapter.jl")
include("viewport.jl")
include("viewport_lifecycle.jl")

function ownership_cycle(registry, state)
    lease = ViewportLifecycle.create_viewport!(registry, () -> begin
        fig, ax, refresh, subscriptions, detach = viewport(state; managed = true)
        (fig, ax), refresh, subscriptions, detach
    end)
    fig, ax = lease.payload
    reference = WeakRef(fig)
    listener_count = length(events(fig).mousebutton.listeners)
    before = copy(colorbuffer(fig; px_per_unit = 1, visible = false))
    @test size(before) == (650, 900)
    @test ax.yreversed[]
    Prototype.pick(state, 20, 20)
    @test state.explorer.selection[] !== nothing
    @test colorbuffer(fig; px_per_unit = 1, visible = false) != before
    ViewportLifecycle.request_release!(registry)
    @test length(events(fig).mousebutton.listeners) == listener_count - 1
    @test state.refresh() === nothing
    for screen in copy(fig.scene.current_screens)
        GLMakie.destroy!(screen)
    end
    @test isempty(fig.scene.current_screens)
    ViewportLifecycle.acknowledge_release!(registry, lease.generation)
    GLMakie.Makie.current_figure() === fig && GLMakie.Makie.current_figure!(nothing)
    ViewportLifecycle.detach_viewport!(registry, lease.generation)
    lease, reference
end

@testset "Fresh scientific viewport figures and disposed callbacks" begin
    registry = ViewportLifecycle.Registry()
    state = Prototype.State()
    baseline = length(GLMakie.ALL_SCREENS)
    leases = ViewportLifecycle.Lease[]
    references = WeakRef[]
    for _ in 1:3
        # Clear selection so each generation's picking changes its own image.
        state.explorer.selection[] = nothing
        lease, reference = ownership_cycle(registry, state)
        push!(leases, lease); push!(references, reference)
        @test registry.active === nothing
        @test length(GLMakie.ALL_SCREENS) == baseline
    end
    GC.gc()
    @test all(reference.value === nothing for reference in references)
    @test all(lease.payload === nothing && isempty(lease.subscriptions) for lease in leases)
    @test registry.generation == 3
end
GLMakie.closeall()
